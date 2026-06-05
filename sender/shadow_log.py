"""Shadow logging for post-deployment tuning.

Appends one JSON line per event to logs/shadow_log.jsonl:
  - type "alert"     — an alert that fired (and was sent to the backend)
  - type "near_miss" — profanity heard and 3+ stages passed, but no alert

This lets us review what the system caught and missed (and tune thresholds)
WITHOUT changing live alert behavior. Decision evidence only — never raw audio.
"""
import json
import os
from datetime import datetime

_LOG_PATH = "/home/echosense/echosense-edge/logs/shadow_log.jsonl"


def _write(record: dict) -> None:
    try:
        record = {"timestamp": datetime.now().isoformat(timespec="seconds"), **record}
        os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
        with open(_LOG_PATH, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[SHADOWLOG] Error: {e}")


def log_alert(result: dict, sent=None) -> None:
    """Log a fired alert (called from main.py after send_alert)."""
    tone = result.get("tone_data") or {}
    _write({
        "type":            "alert",
        "transcript":      result.get("transcribed_text", ""),
        "detected_words":  result.get("detected_words", []),
        "layers_passed":   5,
        "track":           result.get("track"),
        "emotion":         result.get("emotion"),
        "rms":             round(float(tone.get("rms", 0)), 1),
        "duration":        result.get("duration"),
        "severity":        result.get("severity"),
        "confidence":      round(float(result.get("confidence", 0.0)), 3),
        "fired":           True,
        "sent_to_backend": sent,
    })
    print(f"[SHADOWLOG] alert track={result.get('track')} "
          f"sev={result.get('severity')} → {_LOG_PATH}")


def log_near_miss(info: dict) -> None:
    """Log a near-miss (called from the detector when an alert does not fire)."""
    record = {"type": "near_miss", "fired": False}
    record.update(info)
    _write(record)
    print(f"[SHADOWLOG] near_miss layers={info.get('layers_passed')} "
          f"track={info.get('track')}")
