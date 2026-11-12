import getpass
import os
import signal
import subprocess
import sys
import threading
import time

from sender.http_client import (
    send_alert,
)
from config import (
    DEVICE_HOSTNAME,
    NETWORK_INTERFACE,
    PLATFORM,
    SHOW_TRANSCRIPT_TEXT,
)


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
        output = _run([
            'nmcli', '-t', '--escape', 'no', '-f', 'IN-USE,SSID',
            'device', 'wifi', 'list', 'ifname', NETWORK_INTERFACE,
            '--rescan', 'no',
        ])
        for line in output.splitlines():
            in_use, separator, ssid = line.partition(':')
            if separator and in_use == '*' and ssid:
                return ssid
    except Exception:
        pass
    try:
        return _run(['iwgetid', '-r']) or "unknown"
    except Exception:
        return "unknown"


def get_network_info():
    """Heartbeat fields used by device status and dashboard Wi-Fi guidance."""

    ip_address = get_ip()
    ssid = get_ssid()
    connected = ip_address != "unknown" and ssid != "unknown"
    return {
        "ip": ip_address,
        "ssid": ssid,
        "mac": get_mac(),
        "hostname": DEVICE_HOSTNAME,
        "wifi_status": "connected" if connected else "disconnected",
        "wifi_change_method": "local_portal",
        "wifi_local_url": (
            f"http://{ip_address}/" if connected else "http://10.42.0.1/"
        ),
        "wifi_remote_scan_supported": _remote_scan_supported(),
    }


def _remote_scan_supported():
    try:
        from provisioning.wifi_setup.remote_commands import remote_scan_supported

        return remote_scan_supported()
    except Exception:
        return False


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
        network_info_provider=get_network_info,
    )
    print_network_banner()
    # Re-print the live IP/SSID every 60s so the current hotspot address is
    # always discoverable in the log (the boot line goes stale on a WiFi switch).
    threading.Thread(target=_network_status_thread, args=(60,), daemon=True).start()
    # Dashboard Wi-Fi changes (simulated and reported as unsupported off the Pi).
    from provisioning.wifi_setup.remote_commands import start_wifi_command_poll
    start_wifi_command_poll(stop_event)

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


def activate_if_needed(credentials=None, activate=None, restart=os.execv):
    """QR-set-up box without a key yet: wait for an admin to add it, then restart.

    Credentials are read once at import time by config and the sender modules, so after
    the key is saved the process restarts itself and comes up with it, exactly like a
    normal start. Returns False when nothing had to be done.
    """

    import config

    credentials = credentials or config.DEVICE_CREDENTIALS
    if credentials.source != "needs_activation":
        return False
    if activate is None:
        from provisioning.activation import Speaker, activate_until_ready

        def activate():
            return activate_until_ready(
                config.API_URL,
                config.IDENTITY_FILE,
                config.DEVICE_CREDENTIALS_FILE,
                speak=Speaker(config.PLATFORM, config.SPEAK_COMMAND),
                poll_seconds=config.ACTIVATION_POLL_SECONDS,
                announce_seconds=config.ACTIVATION_ANNOUNCE_SECONDS,
            )

    print(f"[SETUP] device={credentials.device_id} has no key yet; starting activation")
    activate()
    print("[SETUP] Restarting with the new device key")
    # exec replaces the process without flushing Python's buffers; flush so the setup
    # lines reach the log.
    sys.stdout.flush()
    sys.stderr.flush()
    script = os.path.abspath(sys.argv[0])
    restart(sys.executable, [sys.executable, script, *sys.argv[1:]])
    return True


if __name__ == "__main__":
    activate_if_needed()
    raise SystemExit(main())
