import subprocess
import sys
import time
import collections
import threading
import pyaudio as _pyaudio
import numpy as _np

from model.whisper_stt import get_recorder, transcribe_and_check
from detection.aggression import AggressionDetector
from sender.http_client import (
    send_alert,
    check_backend_connection,
    start_heartbeat,
)
from model.yamnet_infer import load_yamnet, load_class_names
from audio.led_indicator import LEDIndicator


# ── Rolling audio ring buffer (FIX 4 / FIX 6) ───────────────────────────────
# RealtimeSTT owns the mic for transcription but never hands us the raw audio,
# so YAMNet/tone had nothing to analyze. This background thread keeps the last
# ~3 s (48000 samples @ 16 kHz) of mic audio available so the audio-primary
# path (Track A) can classify HOW a detected word was said.
_AUDIO_RING = collections.deque(maxlen=48000)
_AUDIO_LOCK = threading.Lock()


def _select_ring_device(pa):
    """Pick an input device for the rolling buffer. Prefer PulseAudio
    ('pulse'/'default') so the buffer can capture CONCURRENTLY with RealtimeSTT
    — PulseAudio duplicates a capture source to multiple clients, whereas a raw
    ALSA 'hw' device is exclusive and the two streams would fight over the mic.
    Falls back to the reported default input device."""
    for want in ("pulse", "default"):
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0 and want in info["name"].lower():
                return i
    try:
        return pa.get_default_input_device_info()["index"]
    except Exception:
        return None


def _audio_ring_thread():
    pa = _pyaudio.PyAudio()
    dev = _select_ring_device(pa)
    try:
        stream = pa.open(
            format=_pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            input_device_index=dev,
            frames_per_buffer=1024,
        )
    except Exception as e:
        # Degrade gracefully: no ring buffer ⇒ detector falls back to Track B.
        print(f"[AUDIO RING] Could not open input device {dev}: {e}")
        pa.terminate()
        return
    print(f"[AUDIO RING] Buffer started (device {dev})")
    while True:
        try:
            data = stream.read(1024, exception_on_overflow=False)
            chunk = _np.frombuffer(data, dtype=_np.int16)
            with _AUDIO_LOCK:
                _AUDIO_RING.extend(chunk)
        except Exception as e:
            print(f"[AUDIO RING] Error: {e}")
            break


def get_audio_snapshot():
    with _AUDIO_LOCK:
        return _np.array(list(_AUDIO_RING), dtype=_np.int16)


def get_ip():
    """Current global IPv4 on wlan0. On a phone hotspot the Pi gets its address
    by DHCP from the phone, so this is the address the MacBook must SSH to when
    raspberrypi.local fails (hotspots often block mDNS). Falls back to the first
    non-IPv6 token from `hostname -I`."""
    try:
        out = subprocess.run(
            ['ip', '-4', '-o', 'addr', 'show', 'wlan0'],
            capture_output=True, text=True
        ).stdout
        for part in out.split():
            if '/' in part and part.count('.') == 3:   # e.g. 10.151.131.42/24
                return part.split('/')[0]
    except Exception:
        pass
    try:
        ips = subprocess.run(
            ['hostname', '-I'], capture_output=True, text=True
        ).stdout.strip().split()
        for ip in ips:
            if ip.count('.') == 3:                      # skip IPv6 tokens
                return ip
    except Exception:
        pass
    return "unknown"


def get_mac():
    try:
        with open('/sys/class/net/wlan0/address') as f:
            return f.read().strip()
    except Exception:
        return "unknown"


def get_ssid():
    try:
        result = subprocess.run(
            ['iwgetid', '-r'],
            capture_output=True, text=True
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def print_network_banner(prefix="[NETWORK]"):
    """Print the live network identity clearly so it is easy to find in the log
    even after a WiFi switch or DHCP renew (re-printed periodically, not just at
    boot)."""
    ip = get_ip()
    ssid = get_ssid()
    print("=" * 50)
    print(f"{prefix} WiFi SSID : {ssid}")
    print(f"{prefix} Pi IP     : {ip}")
    print(f"{prefix} Pi MAC    : {get_mac()}   <- find this in the phone's hotspot device list")
    print(f"{prefix} SSH (IP)  : ssh echosense@{ip}")
    print(f"{prefix} SSH (mDNS): ssh echosense@raspberrypi.local")
    print("=" * 50)


def _network_status_thread(interval=60):
    """Re-print the network banner every `interval`s so the CURRENT hotspot IP is
    always visible in the log — the boot-time line goes stale if WiFi changes."""
    while True:
        time.sleep(interval)
        print_network_banner(prefix="[NET-HEARTBEAT]")


def main():
    print("=" * 50)
    print("  EchoSense Edge AI System")
    print("  Acoustic Bullying Detection (5-layer)")
    print("  Davao del Norte State College")
    print("=" * 50)

    led = LEDIndicator()
    led.startup()

    # Start the rolling audio buffer so YAMNet/tone (Track A) have audio to read.
    ring_t = threading.Thread(target=_audio_ring_thread, daemon=True)
    ring_t.start()

    print("\n[INIT] Checking backend connection...")
    check_backend_connection()
    start_heartbeat(
        interval=60,
        info_provider=lambda: {
            "ip": get_ip(),
            "ssid": get_ssid(),
            "mac": get_mac(),
            "hostname": "raspberrypi",
        },
    )

    print_network_banner()
    # Re-print the live IP/SSID every 60s so the current hotspot address is
    # always discoverable in the log (the boot line goes stale on a WiFi switch).
    threading.Thread(target=_network_status_thread, args=(60,), daemon=True).start()

    print("[INIT] Loading YAMNet model...")
    interpreter = load_yamnet()
    class_names = load_class_names()
    print("[INIT] YAMNet loaded!")

    print("[INIT] Loading RealtimeSTT...")
    get_recorder()

    detector = AggressionDetector(
        interpreter=interpreter,
        class_names=class_names,
    )

    led.listening_start()
    print("\n[HEARTBEAT] Started")
    print("\n[MAIN] Listening...")
    print("[INIT] Press Ctrl+C to stop\n")

    try:
        while True:
            result = transcribe_and_check()

            if not result or not result.get("has_profanity"):
                continue

            audio_snap = get_audio_snapshot()
            alert = detector.process_with_audio(result, audio_snap)

            if alert and alert.get("should_alert"):
                led.alert()
                send_alert(
                    severity=alert["severity"],
                    confidence=alert["confidence"],
                    duration=alert["duration"],
                    transcribed_text=alert.get("transcribed_text", ""),
                    detected_words=alert.get("detected_words", []),
                    categories=alert.get("categories", []),
                    yamnet_class=alert.get("yamnet_class", "Speech"),
                    yamnet_score=alert.get("yamnet_score", 0.0),
                    emotion=alert.get("emotion", "neutral"),
                    tone_data=alert.get("tone_data", {}),
                    waveform_snapshot=alert.get("waveform_snapshot", []),
                    language=alert.get("language", "tl"),
                    hard_hits=alert.get("hard_hits", []),
                    soft_hits=alert.get("soft_hits", []),
                    duration_gate=alert.get("duration_gate", ""),
                    required_duration=alert.get("required_duration", 0),
                )

    except KeyboardInterrupt:
        print("\n[STOP] EchoSense stopped.")
    finally:
        led.cleanup()
        sys.exit(0)


if __name__ == "__main__":
    main()
