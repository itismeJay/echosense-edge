import getpass
import subprocess
import signal
import time
import threading

from sender.http_client import (
    send_alert,
)
from config import (
    DEVICE_HOSTNAME,
    NETWORK_INTERFACE,
    PLATFORM,
    SHOW_TRANSCRIPT_TEXT,
)
# Registers the en/tl Whisper engine. Kept at top level on purpose: the
# RealtimeSTT transcription child process re-imports this module.
import model.language_restricted_engine  # noqa: F401


def should_ship_runtime_log(line):
    """Keep diagnostic transcript text local to the authorized edge console."""

    return not str(line or "").startswith("[TRANSCRIPT]")


def _run(args):
    return subprocess.run(args, capture_output=True, text=True).stdout.strip()


def get_ip():
    """Current IPv4 on NETWORK_INTERFACE (wlan0 on the Pi, en0 on a Mac). On a
    phone hotspot the Pi gets its address by DHCP from the phone, so this is
    the address the MacBook must SSH to when mDNS fails (hotspots often block
    it). Falls back to the first IPv4 token from `hostname -I` on Linux."""
    if PLATFORM == "macos":
        try:
            return _run(['ipconfig', 'getifaddr', NETWORK_INTERFACE]) or "unknown"
        except Exception:
            return "unknown"
    try:
        out = _run(['ip', '-4', '-o', 'addr', 'show', NETWORK_INTERFACE])
        for part in out.split():
            if '/' in part and part.count('.') == 3:   # e.g. 10.151.131.42/24
                return part.split('/')[0]
    except Exception:
        pass
    try:
        for ip in _run(['hostname', '-I']).split():
            if ip.count('.') == 3:                      # skip IPv6 tokens
                return ip
    except Exception:
        pass
    return "unknown"


def get_mac():
    try:
        if PLATFORM == "macos":
            out = _run(['ifconfig', NETWORK_INTERFACE])
            return out.split('ether ', 1)[1].split()[0] if 'ether ' in out else "unknown"
        with open(f'/sys/class/net/{NETWORK_INTERFACE}/address') as f:
            return f.read().strip()
    except Exception:
        return "unknown"


def get_ssid():
    # macOS hides the SSID from command-line tools without location access.
    if PLATFORM == "macos":
        return "unknown"
    try:
        return _run(['iwgetid', '-r']) or "unknown"
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
    print(f"{prefix} Device IP : {ip}")
    print(f"{prefix} Device MAC: {get_mac()}   <- find this in the phone's hotspot device list")
    print(f"{prefix} SSH (IP)  : ssh {getpass.getuser()}@{ip}")
    print(f"{prefix} SSH (mDNS): ssh {getpass.getuser()}@{DEVICE_HOSTNAME}.local")
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

    queued = alert_sender(
        severity=alert["severity"],
        severity_evidence=alert.get("severity_evidence"),
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
        event_start_timestamp=(
            audio_event.started_at if audio_event is not None else None
        ),
        event_end_timestamp=(
            audio_event.ended_at if audio_event is not None else None
        ),
        trigger_type="KEYWORD",
    )
    print(f"[OUTBOX] enqueue result event={event_id} queued={bool(queued)}")
    if not queued:
        return False
    led.alert()
    return True


def main():
    from runtime import EdgeRuntime
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

    stop_event = threading.Event()

    def _stop_on_signal(signum, _frame):
        print(f"[STOP] Shutdown signal received signal={signum}")
        stop_event.set()

    signal.signal(signal.SIGTERM, _stop_on_signal)
    signal.signal(signal.SIGINT, _stop_on_signal)

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

    runtime = EdgeRuntime(
        stop_event=stop_event,
        network_info_provider=lambda: {
            "ip": get_ip(),
            "ssid": get_ssid(),
            "mac": get_mac(),
            "hostname": DEVICE_HOSTNAME,
        },
    )
    print_network_banner()
    # Re-print the live IP/SSID every 60s so the current hotspot address is
    # always discoverable in the log (the boot line goes stale on a WiFi switch).
    threading.Thread(target=_network_status_thread, args=(60,), daemon=True).start()

    print("\n[MAIN] Starting one-source AFTER detection runtime...")
    print("[INIT] Press Ctrl+C to stop\n")
    exit_code = 0
    try:
        runtime.run()
    except KeyboardInterrupt:
        stop_event.set()
        print("\n[STOP] EchoSense stopped.")
    except Exception as exc:
        exit_code = 1
        print(f"[FATAL] runtime_error={type(exc).__name__}")
    finally:
        runtime.stop()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
