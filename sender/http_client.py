import requests
import time
import threading
from config import API_URL, LOCATION

_LANGUAGE_VALUES = {"fil", "ceb", "en", "mixed", "unknown"}
_MATCHED_TERM_LANGUAGE_VALUES = {"fil", "ceb", "en"}


def _optional_confidence(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not 0.0 < value <= 1.0:
        return None
    return round(value, 4)


def _clean_matched_terms(matched_terms):
    cleaned = []
    seen = set()
    for item in matched_terms or []:
        if not isinstance(item, dict):
            continue
        term = str(item.get("term") or "").strip()
        source_language = str(item.get("language") or "").strip().lower()
        language = (
            source_language
            if source_language in _MATCHED_TERM_LANGUAGE_VALUES
            else None
        )
        match_type = str(item.get("match_type") or "").strip().lower()
        if not term or len(term) > 100:
            continue
        if match_type not in {"word", "phrase"}:
            match_type = "phrase" if len(term.split()) > 1 else "word"

        term_id = item.get("term_id")
        if (
            isinstance(term_id, bool)
            or not isinstance(term_id, int)
            or term_id <= 0
        ):
            term_id = None
        key = (
            ("id", term_id)
            if term_id is not None
            else ("text", term.lower(), language, match_type)
        )
        if key in seen:
            continue

        evidence = {
            "term": term,
            "language": language,
            "match_type": match_type,
        }
        if term_id is not None:
            evidence["term_id"] = term_id
        cleaned.append(evidence)
        seen.add(key)
    return cleaned


def build_alert_payload(
    severity,
    confidence,
    duration,
    transcribed_text="",
    detected_words=None,
    categories=None,
    language="unknown",
    hard_hits=None,
    soft_hits=None,
    required_duration=None,
    duration_gate=None,
    yamnet_class="Unknown",
    yamnet_score=0.0,
    yamnet_ran=False,
    emotion="unknown",
    tone_data=None,
    waveform_snapshot=None,
    language_confidence=None,
    matched_terms=None,
    event_id=None,
):
    """Build the production payload while retaining every legacy field."""

    language = str(language or "unknown").lower()
    if language not in _LANGUAGE_VALUES:
        language = "unknown"
    transcript = str(transcribed_text or "")
    yamnet_ran = bool(yamnet_ran)
    if yamnet_ran:
        payload_yamnet_class = str(yamnet_class or "Unknown")
        payload_yamnet_score = float(round(float(yamnet_score), 4))
    else:
        # Backend null support is not established. This explicit sentinel means
        # "not measured"; it is never interpreted as a real YAMNet score.
        payload_yamnet_class = "NotRun"
        payload_yamnet_score = 0.0
    return {
        "event_id": str(event_id) if event_id else None,
        "severity": str(severity),
        "confidence": float(round(float(confidence), 4)),
        "duration": float(round(float(duration), 2)),
        "required_duration": float(required_duration) if required_duration is not None else None,
        "duration_gate": duration_gate,
        "location": LOCATION,
        # New contract name plus the legacy field still consumed by production.
        "transcript": transcript,
        "transcribed_text": transcript,
        "detected_words": detected_words or [],
        "categories": categories or [],
        "language": language,
        "language_confidence": _optional_confidence(language_confidence),
        "matched_terms": _clean_matched_terms(matched_terms),
        "hard_hits": hard_hits or [],
        "soft_hits": soft_hits or [],
        "yamnet_class": payload_yamnet_class,
        "yamnet_score": payload_yamnet_score,
        "yamnet_ran": yamnet_ran,
        "emotion": emotion,
        "rms": round(float(tone_data.get("rms", 0)), 2) if tone_data else 0,
        "energy_variance": round(float(tone_data.get("energy_variance", 0)), 2) if tone_data else 0,
        "zero_crossing_rate": round(float(tone_data.get("zero_crossing_rate", 0)), 4) if tone_data else 0,
        "peak_to_average": round(float(tone_data.get("peak_to_average", 0)), 2) if tone_data else 0,
        "waveform_snapshot": waveform_snapshot or [],
    }


def send_alert(severity, confidence, duration,
               transcribed_text="", detected_words=None,
               categories=None, language="unknown",
               hard_hits=None, soft_hits=None,
               required_duration=None, duration_gate=None,
               yamnet_class="Unknown", yamnet_score=0.0,
               yamnet_ran=False,
               emotion="unknown", tone_data=None,
               waveform_snapshot=None, language_confidence=None,
               matched_terms=None, event_id=None, retries=3):
    payload = build_alert_payload(
        severity=severity,
        confidence=confidence,
        duration=duration,
        transcribed_text=transcribed_text,
        detected_words=detected_words,
        categories=categories,
        language=language,
        language_confidence=language_confidence,
        matched_terms=matched_terms,
        hard_hits=hard_hits,
        soft_hits=soft_hits,
        required_duration=required_duration,
        duration_gate=duration_gate,
        yamnet_class=yamnet_class,
        yamnet_score=yamnet_score,
        yamnet_ran=yamnet_ran,
        emotion=emotion,
        tone_data=tone_data,
        waveform_snapshot=waveform_snapshot,
        event_id=event_id,
    )
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
