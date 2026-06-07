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
    try:
        result = subprocess.run(
            ['hostname', '-I'],
            capture_output=True, text=True
        )
        ips = result.stdout.strip().split()
        return ips[0] if ips else "unknown"
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
    start_heartbeat(interval=60)

    ip = get_ip()
    ssid = get_ssid()
    print(f"[NETWORK] WiFi:   {ssid}")
    print(f"[NETWORK] IP:     {ip}")
    print(f"[NETWORK] SSH:    ssh echosense@{ip}")
    print(f"[NETWORK] OR:     ssh echosense@raspberrypi.local")

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
