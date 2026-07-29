import subprocess
import sys
import time
import threading

from model.whisper_stt import get_recorder, transcribe_and_check
from detection.aggression import AggressionDetector
from sender.http_client import (
    send_alert,
    check_backend_connection,
    start_heartbeat,
)
from model.yamnet_infer import load_yamnet, load_class_names
from model.monitored_terms import start_dictionary_sync
from audio.led_indicator import LEDIndicator
from config import SHOW_TRANSCRIPT_TEXT


def should_ship_runtime_log(line):
    """Keep diagnostic transcript text local to the authorized edge console."""

    return not str(line or "").startswith("[TRANSCRIPT]")


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


def process_transcription_result(result, detector, led, alert_sender=send_alert):
    """Process one consumed STT result without performing any capture itself."""

    if not result:
        return False
    if result.get("quality_accepted") is False:
        return False
    if result.get("context_suppressed_all"):
        return False
    if not result.get("has_profanity"):
        return False

    audio_event = result.get("audio_event")
    alert = detector.process_with_audio(result, audio_event)
    event_id = result.get("event_id") or "unavailable"
    should_alert = bool(alert and alert.get("should_alert"))
    print(f"[DECISION] event={event_id} alert={should_alert}")

    if not should_alert:
        return False

    led.alert()
    alert_sender(
        severity=alert["severity"],
        confidence=alert["confidence"],
        duration=alert["duration"],
        transcribed_text=alert.get("transcribed_text", ""),
        detected_words=alert.get("detected_words", []),
        categories=alert.get("categories", []),
        yamnet_class=alert.get("yamnet_class", "NotRun"),
        yamnet_score=alert.get("yamnet_score", 0.0),
        yamnet_ran=alert.get("yamnet_ran", False),
        event_id=alert.get("event_id"),
        emotion=alert.get("emotion", "neutral"),
        tone_data=alert.get("tone_data", {}),
        waveform_snapshot=alert.get("waveform_snapshot", []),
        language=alert.get("language", "unknown"),
        language_confidence=alert.get("language_confidence"),
        matched_terms=alert.get("matched_terms", []),
        hard_hits=alert.get("hard_hits", []),
        soft_hits=alert.get("soft_hits", []),
        duration_gate=alert.get("duration_gate", ""),
        required_duration=alert.get("required_duration", 0),
    )
    return True


def main():
    from sender.http_client import push_log_line
    from sender.http_client import start_log_flush_thread
    import builtins as _builtins

    _orig_print = _builtins.print

    def _log_print(*args, **kwargs):
        _orig_print(*args, **kwargs)
        line = " ".join(str(a) for a in args)
        if should_ship_runtime_log(line):
            push_log_line(line)

    _builtins.print = _log_print
    start_log_flush_thread()

    print(f"[CONFIG] show_transcript_text={str(SHOW_TRANSCRIPT_TEXT).lower()}")
    if SHOW_TRANSCRIPT_TEXT:
        print(
            "[NOTICE] Exact finalized transcript logging is enabled "
            "for authorized testing."
        )

    print("=" * 50)
    print("  EchoSense Edge AI System")
    print("  Classroom Acoustic Risk Detection (5-layer)")
    print("  Davao del Norte State College")
    print("=" * 50)

    led = LEDIndicator()
    led.startup()

    print("\n[INIT] Checking backend connection...")
    check_backend_connection()
    start_dictionary_sync()
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
            process_transcription_result(result, detector, led)

    except KeyboardInterrupt:
        print("\n[STOP] EchoSense stopped.")
    finally:
        led.cleanup()
        sys.exit(0)


if __name__ == "__main__":
    main()
