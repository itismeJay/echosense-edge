"""Deterministic schema-v2 fixtures shared by sender/outbox unit tests."""

from datetime import datetime, timedelta, timezone
from uuid import NAMESPACE_URL, uuid5

from sender.payload_v2 import build_alert_payload_v2


DEVICE_ID = "classroom-test-pi"
DEVICE_KEY = "synthetic-device-key-for-tests-only"
EVENT_END = datetime(2026, 8, 4, 0, 0, 2, tzinfo=timezone.utc)


def event_id_for(name="default"):
    return str(uuid5(NAMESPACE_URL, f"echosense-edge-test:{name}"))


def schema_v2_payload(name="default", **overrides):
    values = {
        "event_id": event_id_for(name),
        "trigger_type": "KEYWORD",
        "severity": "MEDIUM",
        "event_start_timestamp": EVENT_END - timedelta(seconds=1.25),
        "event_end_timestamp": EVENT_END,
        "device_identifier": DEVICE_ID,
        "transcribed_text": "NON-PRODUCTION OUTBOX TEST — exact text.\n",
        "detected_words": ["synthetic monitored phrase"],
        "confidence": 0.5,
        "language": "en",
        "matched_terms": [],
        "severity_evidence": {
            "level": "MEDIUM",
            "reasons": ["term_category:synthetic"],
            "term_categories": {
                "synthetic": ["synthetic monitored phrase"],
            },
            "supporting_evidence": [],
        },
        "yamnet_class": "NotRun",
        "yamnet_score": 0.0,
        "yamnet_ran": False,
        "emotion": "neutral",
        "tone_data": {},
        "waveform_snapshot": [0, 1, 0],
    }
    values.update(overrides)
    return build_alert_payload_v2(**values)
