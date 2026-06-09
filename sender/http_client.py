import requests
import time
import threading
from config import API_URL, LOCATION

def send_alert(severity, confidence, duration,
               transcribed_text="", detected_words=None,
               categories=None, language="unknown",
               hard_hits=None, soft_hits=None,
               required_duration=None, duration_gate=None,
               yamnet_class="Unknown", yamnet_score=0.0,
               emotion="unknown", tone_data=None,
               waveform_snapshot=None, retries=3):
    payload = {
        "severity": str(severity),
        "confidence": float(round(float(confidence), 4)),
        "duration": float(round(float(duration), 2)),
        "required_duration": float(required_duration) if required_duration is not None else None,
        "duration_gate": duration_gate,
        "location": LOCATION,
        "transcribed_text": transcribed_text,
        "detected_words": detected_words or [],
        "categories": categories or [],
        "language": language,
        "hard_hits": hard_hits or [],
        "soft_hits": soft_hits or [],
        "yamnet_class": yamnet_class,
        "yamnet_score": float(round(float(yamnet_score), 4)),
        "emotion": emotion,
        "rms": round(float(tone_data.get("rms", 0)), 2) if tone_data else 0,
        "energy_variance": round(float(tone_data.get("energy_variance", 0)), 2) if tone_data else 0,
        "zero_crossing_rate": round(float(tone_data.get("zero_crossing_rate", 0)), 4) if tone_data else 0,
        "peak_to_average": round(float(tone_data.get("peak_to_average", 0)), 2) if tone_data else 0,
        "waveform_snapshot": waveform_snapshot or []
    }
    for attempt in range(retries):
        try:
            print(f"[SENDER] Sending alert... (attempt {attempt + 1})")
            response = requests.post(f"{API_URL}/alerts/", json=payload, timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"[SENDER] Alert sent! ID: {data.get('id')} Severity: {data.get('severity')}")
                return True
            else:
                print(f"[SENDER] Failed! Status: {response.status_code}")
        except Exception as e:
            print(f"[SENDER] Error: {e}")
            time.sleep(3)
    return False

def _heartbeat_loop(interval, info_provider=None):
    while True:
        time.sleep(interval)
        # Best-effort: include the Pi's live LAN IP/SSID so the device can be
        # located from the backend even when raspberrypi.local can't be reached
        # (e.g. a phone hotspot that blocks mDNS). Unknown fields are harmless;
        # if the backend ignores the body the heartbeat still works.
        info = {}
        try:
            info = info_provider() if info_provider else {}
        except Exception:
            info = {}
        try:
            response = requests.post(
                f"{API_URL}/system-settings/heartbeat",
                json=info or None,
                timeout=10
            )
            ip = info.get("ip", "?")
            print(f"[HEARTBEAT] {response.status_code} ip={ip}")
        except Exception as e:
            print(f"[HEARTBEAT] Error: {e}")

def start_heartbeat(interval=60, info_provider=None):
    t = threading.Thread(
        target=_heartbeat_loop, args=(interval, info_provider), daemon=True
    )
    t.start()
    print(f"[HEARTBEAT] Started — posting every {interval}s")

def check_backend_connection():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            print(f"[SENDER] Backend connected!")
            return True
    except:
        print(f"[SENDER] Backend not reachable!")
        return False


# ── Remote log shipping ─────────────────────────────────────────────────────
# Buffers print() output and ships it to the backend so admins can read the
# Pi's logs from the dashboard without SSH. Best-effort: failures are swallowed
# so log shipping can never take down the detection loop.
import queue as _queue
import threading as _threading

_LOG_BUFFER = []
_LOG_LOCK = _threading.Lock()

def push_log_line(line: str) -> None:
    global _LOG_BUFFER
    if not line or not line.strip():
        return
    with _LOG_LOCK:
        _LOG_BUFFER.append(line.strip())
        if len(_LOG_BUFFER) >= 20:
            lines = _LOG_BUFFER.copy()
            _LOG_BUFFER.clear()
            _threading.Thread(
                target=_send_logs_batch,
                args=(lines,),
                daemon=True,
            ).start()

def _send_logs_batch(lines: list) -> None:
    try:
        requests.post(
            f"{API_URL}/system/logs",
            json={"lines": lines},
            timeout=5,
        )
    except Exception:
        pass  # logs are not critical — fail silently

def flush_logs() -> None:
    global _LOG_BUFFER
    with _LOG_LOCK:
        if _LOG_BUFFER:
            lines = _LOG_BUFFER.copy()
            _LOG_BUFFER.clear()
            _threading.Thread(
                target=_send_logs_batch,
                args=(lines,),
                daemon=True,
            ).start()

def start_log_flush_thread() -> None:
    import time as _time
    def _flush_loop():
        while True:
            _time.sleep(10)
            flush_logs()
    t = _threading.Thread(target=_flush_loop, daemon=True)
    t.start()
    print("[LOGS] Log sender started")
