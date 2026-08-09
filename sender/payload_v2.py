"""Build the strict backend schema-v2 body for one finalized Edge alert."""

from __future__ import annotations

from datetime import datetime, timezone
import math
import re
from uuid import UUID


SCHEMA_VERSION = 2
REVIEW_MESSAGE = (
    "Unverified possible-aggression alert. Human review required."
)
TRIGGER_TYPES = frozenset({"KEYWORD", "ACOUSTIC", "TEST"})
SEVERITY_LEVELS = frozenset({"LOW", "MEDIUM", "HIGH"})
LANGUAGE_VALUES = frozenset({"fil", "ceb", "en", "mixed", "unknown"})
MATCHED_TERM_LANGUAGE_VALUES = frozenset({"fil", "ceb", "en"})

_FORBIDDEN_PRIVATE_KEYS = frozenset({
    "audio",
    "audio_base64",
    "audio_blob",
    "audio_bytes",
    "audio_data",
    "audio_debug",
    "audio_samples",
    "debug_transcript",
    "partial_transcript",
    "pcm",
    "pcm_samples",
    "raw_audio",
    "raw_pcm",
    "raw_samples",
    "raw_vosk_text",
    "recorded_audio",
    "vosk_partial",
    "wav",
    "wave_file",
    "waveform_bytes",
})
_TONE_MEASUREMENT_KEYS = (
    "rms",
    "energy_variance",
    "zero_crossing_rate",
    "peak_to_average",
)
_REQUIRED_PAYLOAD_FIELDS = frozenset({
    "event_id",
    "schema_version",
    "trigger_type",
    "severity",
    "severity_reasons",
    "review_message",
    "device_identifier",
    "event_start_timestamp",
    "event_end_timestamp",
    "severity_evidence",
    "monitored_terms",
    "monitored_word_detected",
    "detailed_acoustic_evidence",
    "transcription_status",
    "language",
    "matched_terms",
    "test_mode",
})
_OPTIONAL_PAYLOAD_FIELDS = frozenset({
    "confidence",
    "direct_address_evidence",
    "language_confidence",
    "laughter_context",
    "repetition_evidence",
    "tone_evidence",
    "transcribed_text",
    "waveform_snapshot",
})
_ALLOWED_PAYLOAD_FIELDS = _REQUIRED_PAYLOAD_FIELDS | _OPTIONAL_PAYLOAD_FIELDS


class AlertPayloadValidationError(ValueError):
    """A finalized alert cannot safely satisfy the backend contract."""


def _normalized_key(value: object) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().casefold(),
    ).strip("_")


def _reject_private_content(value, *, key="") -> None:
    if _normalized_key(key) in _FORBIDDEN_PRIVATE_KEYS:
        raise AlertPayloadValidationError(
            "raw audio or debug speech fields are not allowed"
        )
    if isinstance(value, dict):
        for nested_key, nested_value in value.items():
            _reject_private_content(nested_value, key=nested_key)
    elif isinstance(value, (list, tuple)):
        for nested_value in value:
            _reject_private_content(nested_value)


def normalize_event_id(value: object) -> str:
    """Return the backend's canonical UUID string representation."""

    try:
        parsed = UUID(str(value).strip())
    except (AttributeError, TypeError, ValueError) as exc:
        raise AlertPayloadValidationError("event_id must be a valid UUID") from exc
    return str(parsed)


def _normalize_enum(value: object, allowed, field_name: str) -> str:
    normalized = str(value or "").strip().upper()
    if normalized not in allowed:
        choices = ", ".join(sorted(allowed))
        raise AlertPayloadValidationError(
            f"{field_name} must be one of: {choices}"
        )
    return normalized


def _utc_timestamp(value: object, field_name: str) -> tuple[datetime, str]:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise AlertPayloadValidationError(
                f"{field_name} must be an ISO-8601 timestamp"
            ) from exc
    else:
        raise AlertPayloadValidationError(
            f"{field_name} must be a timezone-aware timestamp"
        )
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise AlertPayloadValidationError(
            f"{field_name} must include a timezone"
        )
    utc_value = parsed.astimezone(timezone.utc)
    serialized = utc_value.isoformat(timespec="microseconds").replace(
        "+00:00",
        "Z",
    )
    return utc_value, serialized


def _finite_number(
    value: object,
    field_name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise AlertPayloadValidationError(f"{field_name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AlertPayloadValidationError(f"{field_name} must be numeric") from exc
    if not math.isfinite(number):
        raise AlertPayloadValidationError(f"{field_name} must be finite")
    if minimum is not None and number < minimum:
        raise AlertPayloadValidationError(f"{field_name} is below its minimum")
    if maximum is not None and number > maximum:
        raise AlertPayloadValidationError(f"{field_name} exceeds its maximum")
    return number


def _bounded_strings(values, field_name: str, *, required=False) -> list[str]:
    cleaned = []
    for item in values or []:
        text = str(item or "").strip()
        if not text or len(text) > 200:
            raise AlertPayloadValidationError(
                f"{field_name} entries must be 1 to 200 characters"
            )
        if text not in cleaned:
            cleaned.append(text)
    if len(cleaned) > 50:
        raise AlertPayloadValidationError(
            f"{field_name} may contain at most 50 entries"
        )
    if required and not cleaned:
        raise AlertPayloadValidationError(f"{field_name} is required")
    return cleaned


def _severity_evidence(value, severity: str) -> dict:
    if value is None:
        raise AlertPayloadValidationError("severity_evidence is required")
    if not isinstance(value, dict):
        raise AlertPayloadValidationError(
            "severity_evidence must be an object"
        )
    level = _normalize_enum(
        value.get("level"),
        SEVERITY_LEVELS,
        "severity_evidence.level",
    )
    if level != severity:
        raise AlertPayloadValidationError(
            "severity_evidence.level must match severity"
        )
    reasons = _bounded_strings(
        value.get("reasons"),
        "severity_evidence.reasons",
        required=True,
    )
    supporting = _bounded_strings(
        value.get("supporting_evidence"),
        "severity_evidence.supporting_evidence",
    )
    source_categories = value.get("term_categories") or {}
    if not isinstance(source_categories, dict) or len(source_categories) > 50:
        raise AlertPayloadValidationError(
            "severity_evidence.term_categories must be a bounded object"
        )
    term_categories = {}
    for category, terms in source_categories.items():
        category_name = str(category or "").strip()
        if not category_name or len(category_name) > 200:
            raise AlertPayloadValidationError(
                "severity evidence category names must be 1 to 200 characters"
            )
        term_categories[category_name] = _bounded_strings(
            terms,
            f"severity_evidence.term_categories.{category_name}",
        )
    total_characters = sum(len(item) for item in reasons + supporting)
    total_characters += sum(
        len(category) + sum(len(term) for term in terms)
        for category, terms in term_categories.items()
    )
    if total_characters > 16_000:
        raise AlertPayloadValidationError(
            "severity_evidence exceeds 16000 total characters"
        )
    evidence = {
        "level": level,
        "reasons": reasons,
        "term_categories": term_categories,
        "supporting_evidence": supporting,
    }
    _reject_private_content(evidence)
    return evidence


def _clean_terms(detected_words) -> list[str]:
    terms = []
    for item in detected_words or []:
        term = str(item or "").strip()
        if not term or len(term) > 200:
            raise AlertPayloadValidationError(
                "monitored terms must be 1 to 200 characters"
            )
        if term not in terms:
            terms.append(term)
    if len(terms) > 50:
        raise AlertPayloadValidationError(
            "monitored_terms may contain at most 50 entries"
        )
    return terms


def _clean_matched_terms(matched_terms) -> list[dict]:
    cleaned = []
    seen = set()
    for item in matched_terms or []:
        if not isinstance(item, dict):
            raise AlertPayloadValidationError(
                "matched_terms entries must be objects"
            )
        term = str(item.get("term") or item.get("matched_text") or "").strip()
        term_id = item.get("term_id")
        if isinstance(term_id, bool) or not isinstance(term_id, int) or term_id <= 0:
            term_id = None
        if not term and term_id is None:
            raise AlertPayloadValidationError(
                "matched terms require term_id or term"
            )
        if len(term) > 100:
            raise AlertPayloadValidationError(
                "matched term text exceeds 100 characters"
            )
        language = str(item.get("language") or "").strip().lower()
        if language not in MATCHED_TERM_LANGUAGE_VALUES:
            language = None
        match_type = str(item.get("match_type") or "exact").strip().lower()
        if not match_type or len(match_type) > 30:
            raise AlertPayloadValidationError(
                "matched term match_type must be 1 to 30 characters"
            )
        identity = ("id", term_id) if term_id is not None else ("term", term.casefold())
        if identity in seen:
            continue
        evidence = {"match_type": match_type}
        if term_id is not None:
            evidence["term_id"] = term_id
        if term:
            evidence["term"] = term
        if language is not None:
            evidence["language"] = language
        cleaned.append(evidence)
        seen.add(identity)
    if len(cleaned) > 50:
        raise AlertPayloadValidationError(
            "matched_terms may contain at most 50 entries"
        )
    return cleaned


def _acoustic_evidence(yamnet_ran, yamnet_class, yamnet_score) -> dict:
    ran = bool(yamnet_ran)
    evidence = {
        "role": "supporting_only",
        "yamnet": {"ran": ran},
    }
    if ran:
        label = str(yamnet_class or "").strip()
        if not label or label.casefold() == "notrun" or len(label) > 200:
            raise AlertPayloadValidationError(
                "yamnet_class must be a measured label when yamnet_ran is true"
            )
        evidence["yamnet"].update({
            "label": label,
            "score": round(
                _finite_number(
                    yamnet_score,
                    "yamnet_score",
                    minimum=0.0,
                    maximum=1.0,
                ),
                4,
            ),
        })
    return evidence


def _tone_evidence(emotion, tone_data) -> dict | None:
    measurements = {}
    source = tone_data or {}
    if not isinstance(source, dict):
        raise AlertPayloadValidationError("tone_data must be an object")
    for key in _TONE_MEASUREMENT_KEYS:
        if key in source and source[key] is not None:
            measurements[key] = round(
                _finite_number(source[key], f"tone_data.{key}"),
                4,
            )
    label = str(emotion or "").strip()
    if not measurements and label.casefold() in {"", "unknown"}:
        return None
    evidence = {"role": "supporting_only"}
    if label and label.casefold() != "unknown":
        if len(label) > 100:
            raise AlertPayloadValidationError("emotion exceeds 100 characters")
        evidence["emotion"] = label
    if measurements:
        evidence["measurements"] = measurements
    return evidence


def _clean_waveform(values) -> list[int]:
    if values is None:
        return []
    if not isinstance(values, (list, tuple)):
        raise AlertPayloadValidationError(
            "waveform_snapshot must be a sparse numeric list"
        )
    if len(values) > 256:
        raise AlertPayloadValidationError(
            "waveform_snapshot may contain at most 256 points"
        )
    cleaned = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise AlertPayloadValidationError(
                "waveform_snapshot points must be integers"
            )
        if not 0 <= value <= 32768:
            raise AlertPayloadValidationError(
                "waveform_snapshot points must be between 0 and 32768"
            )
        cleaned.append(value)
    return cleaned


def build_alert_payload_v2(
    *,
    event_id,
    trigger_type,
    severity,
    event_start_timestamp,
    event_end_timestamp,
    device_identifier,
    transcribed_text="",
    detected_words=None,
    confidence=None,
    language="unknown",
    language_confidence=None,
    matched_terms=None,
    severity_evidence=None,
    yamnet_class="NotRun",
    yamnet_score=0.0,
    yamnet_ran=False,
    emotion="unknown",
    tone_data=None,
    waveform_snapshot=None,
) -> dict:
    """Return one deterministic, backend-compatible schema-v2 body."""

    canonical_event_id = normalize_event_id(event_id)
    normalized_trigger = _normalize_enum(
        trigger_type,
        TRIGGER_TYPES,
        "trigger_type",
    )
    if normalized_trigger == "TEST":
        raise AlertPayloadValidationError(
            "real Edge alerts must not use the TEST trigger"
        )
    normalized_severity = _normalize_enum(
        severity,
        SEVERITY_LEVELS,
        "severity",
    )
    device_id = str(device_identifier or "").strip()
    if not device_id or len(device_id) > 100:
        raise AlertPayloadValidationError(
            "device_identifier must be 1 to 100 characters"
        )
    started_at, serialized_start = _utc_timestamp(
        event_start_timestamp,
        "event_start_timestamp",
    )
    ended_at, serialized_end = _utc_timestamp(
        event_end_timestamp,
        "event_end_timestamp",
    )
    if ended_at < started_at:
        raise AlertPayloadValidationError(
            "event_end_timestamp must not precede event_start_timestamp"
        )

    monitored_terms = _clean_terms(detected_words)
    evidence = _severity_evidence(
        severity_evidence,
        normalized_severity,
    )
    payload = {
        "event_id": canonical_event_id,
        "schema_version": SCHEMA_VERSION,
        "trigger_type": normalized_trigger,
        "severity": normalized_severity,
        "severity_reasons": list(evidence["reasons"]),
        "review_message": REVIEW_MESSAGE,
        "device_identifier": device_id,
        "event_start_timestamp": serialized_start,
        "event_end_timestamp": serialized_end,
        "severity_evidence": evidence,
        "monitored_terms": monitored_terms,
        "monitored_word_detected": bool(monitored_terms),
        "detailed_acoustic_evidence": _acoustic_evidence(
            yamnet_ran,
            yamnet_class,
            yamnet_score,
        ),
        "transcription_status": (
            "complete" if str(transcribed_text or "").strip() else "unavailable"
        ),
        "language": str(language or "unknown").strip().lower(),
        "matched_terms": _clean_matched_terms(matched_terms),
        "test_mode": False,
    }
    if payload["language"] not in LANGUAGE_VALUES:
        raise AlertPayloadValidationError(
            "language must be fil, ceb, en, mixed, or unknown"
        )
    transcript = str(transcribed_text or "")
    if len(transcript) > 10_000:
        raise AlertPayloadValidationError(
            "transcribed_text exceeds 10000 characters"
        )
    if transcript:
        payload["transcribed_text"] = transcript
    if confidence is not None:
        payload["confidence"] = round(
            _finite_number(
                confidence,
                "confidence",
                minimum=0.0,
                maximum=1.0,
            ),
            4,
        )
    if language_confidence is not None:
        payload["language_confidence"] = round(
            _finite_number(
                language_confidence,
                "language_confidence",
                minimum=0.0,
                maximum=1.0,
            ),
            4,
        )
    tone_evidence = _tone_evidence(emotion, tone_data)
    if tone_evidence is not None:
        payload["tone_evidence"] = tone_evidence
    supporting = set(evidence["supporting_evidence"])
    if "repeated_monitored_evidence" in supporting:
        payload["repetition_evidence"] = {"detected": True}
    if "direct_target_pattern" in supporting:
        payload["direct_address_evidence"] = {"detected": True}
    if "laughter_or_excitement_marker_present" in supporting:
        payload["laughter_context"] = {"detected": True}
    waveform = _clean_waveform(waveform_snapshot)
    if waveform:
        payload["waveform_snapshot"] = waveform

    _reject_private_content(payload)
    validate_alert_payload_v2(payload)
    return payload


def validate_alert_payload_v2(payload: object) -> None:
    """Fail closed unless a body matches the Edge's locked schema-v2 subset."""

    if not isinstance(payload, dict):
        raise AlertPayloadValidationError("alert payload must be an object")
    missing = _REQUIRED_PAYLOAD_FIELDS - payload.keys()
    unexpected = payload.keys() - _ALLOWED_PAYLOAD_FIELDS
    if missing:
        raise AlertPayloadValidationError("schema-v2 payload is incomplete")
    if unexpected:
        raise AlertPayloadValidationError(
            "schema-v2 payload contains unsupported fields"
        )
    if (
        isinstance(payload.get("schema_version"), bool)
        or payload.get("schema_version") != 2
    ):
        raise AlertPayloadValidationError("schema_version must be integer 2")
    if normalize_event_id(payload.get("event_id")) != payload.get("event_id"):
        raise AlertPayloadValidationError("event_id must use canonical UUID form")
    if payload.get("trigger_type") not in {"KEYWORD", "ACOUSTIC"}:
        raise AlertPayloadValidationError("trigger_type is invalid for a real alert")
    if payload.get("severity") not in SEVERITY_LEVELS:
        raise AlertPayloadValidationError("severity must be uppercase")
    if payload.get("review_message") != REVIEW_MESSAGE:
        raise AlertPayloadValidationError("review_message is invalid")
    device_id = payload.get("device_identifier")
    if not isinstance(device_id, str) or not device_id.strip() or len(device_id) > 100:
        raise AlertPayloadValidationError("device_identifier is invalid")
    started_at, _serialized_start = _utc_timestamp(
        payload.get("event_start_timestamp"),
        "event_start_timestamp",
    )
    ended_at, _serialized_end = _utc_timestamp(
        payload.get("event_end_timestamp"),
        "event_end_timestamp",
    )
    if ended_at < started_at:
        raise AlertPayloadValidationError(
            "event_end_timestamp must not precede event_start_timestamp"
        )
    if payload.get("test_mode") is not False:
        raise AlertPayloadValidationError("test_mode must be false")
    severity_reasons = _bounded_strings(
        payload.get("severity_reasons"),
        "severity_reasons",
        required=True,
    )
    evidence = _severity_evidence(
        payload.get("severity_evidence"),
        payload["severity"],
    )
    if severity_reasons != evidence["reasons"]:
        raise AlertPayloadValidationError(
            "severity_reasons must match severity_evidence.reasons"
        )
    if payload.get("transcription_status") == "complete" and not str(
        payload.get("transcribed_text") or ""
    ).strip():
        raise AlertPayloadValidationError(
            "complete transcription requires transcribed_text"
        )
    if payload.get("language") not in LANGUAGE_VALUES:
        raise AlertPayloadValidationError("language is invalid")
    _reject_private_content(payload)
