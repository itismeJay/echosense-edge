"""Map one finalized incident onto the preserved schema-v2/outbox boundary."""

from __future__ import annotations

from config import DASHBOARD_TEST_MODE, ECHOSENSE_DEVICE_ID
from detection.evidence import FinalizedIncident
from sender.http_client import send_alert


def _language_for(text_evidence):
    normalized = []
    ordered = sorted(
        text_evidence,
        key=lambda item: (item.observed_monotonic, item.observed_at),
    )
    for item in ordered:
        value = item.language.lower()
        if value == "tl":
            value = "fil"
        if value in {"fil", "ceb", "en"} and value not in normalized:
            normalized.append(value)
        elif value == "mixed":
            return "mixed"
    if len(normalized) > 1:
        return "mixed"
    return normalized[0] if normalized else "unknown"


def _transcript_for(text_evidence):
    values = []
    identities = set()
    total = 0
    for item in text_evidence:
        if not item.qualifies_to_open:
            continue
        text = item.transcript
        identity = (
            item.normalized_transcript.strip().casefold()
            or text.strip().casefold()
        )
        if not text or not identity or identity in identities:
            continue
        separator = 1 if values else 0
        remaining = 10_000 - total - separator
        if remaining <= 0:
            break
        values.append(text[:remaining])
        identities.add(identity)
        total += len(values[-1]) + separator
    return "\n".join(values)


def _strongest_acoustic(evidence):
    ran = [item for item in evidence if item.yamnet_ran]
    return max(
        ran,
        key=lambda item: (
            item.qualifies_to_open,
            item.yamnet_aggressive,
            item.tone_aggressive,
            item.yamnet_score,
        ),
        default=None,
    )


def _emotion_label(tone):
    return str(getattr(tone, "emotion_label", "") or "unknown")


def _tone_context(incident):
    """Tone vote values for tone_evidence / laughter_context (no audio)."""

    tone = incident.tone
    context = {"reason": incident.summary} if incident.summary else {}
    if tone is None:
        return context or None
    context.update({
        "emotion_score": tone.emotion_score,
        "angry_score": tone.angry_score,
        "angry_tone": tone.angry_tone,
        "shout_score": min(1.0, tone.shout_score),
        "rms_ratio": tone.rms_ratio,
        "laughter": {
            "detected": tone.laughter,
            "score": min(1.0, tone.laughter_score),
            "sources": list(tone.laughter_sources),
            "mocking": "mocking_with_laughter" in incident.severity_reasons,
        },
    })
    return context


def finalized_incident_send_arguments(
    incident: FinalizedIncident,
    *,
    device_identifier=ECHOSENSE_DEVICE_ID,
    dashboard_test_mode=DASHBOARD_TEST_MODE,
):
    """Return only existing send_alert/schema-v2 fields; never include PCM."""

    if not isinstance(incident, FinalizedIncident):
        raise TypeError("incident must be FinalizedIncident")
    qualifying_text = tuple(
        item for item in incident.text_evidence if item.qualifies_to_open
    )
    terms = list(dict.fromkeys(
        term
        for evidence in qualifying_text
        for term in evidence.decision_terms
    ))[:50]
    matched = []
    seen = set()
    for evidence in qualifying_text:
        for item in evidence.matched_terms:
            if item.term not in evidence.decision_terms:
                continue
            value = item.backend_value()
            identity = value.get("term_id") or value.get("term")
            if identity in seen:
                continue
            seen.add(identity)
            matched.append(value)
            if len(matched) >= 50:
                break
        if len(matched) >= 50:
            break
    acoustic = _strongest_acoustic(incident.acoustic_evidence)
    tone = max(
        incident.acoustic_evidence,
        key=lambda item: (
            item.qualifies_to_open,
            item.tone_aggressive,
            item.energy_variance,
            item.zero_crossing_rate,
        ),
        default=None,
    )
    language_confidences = [
        item.language_confidence
        for item in qualifying_text
        if item.language_confidence is not None
    ]
    return {
        "severity": incident.severity,
        "severity_evidence": {
            "level": incident.severity,
            "reasons": list(incident.severity_reasons),
            "term_categories": {
                category: list(values)
                for category, values in incident.term_categories.items()
            },
            "supporting_evidence": list(incident.supporting_evidence),
        },
        "confidence": incident.confidence,
        "duration": incident.duration_seconds,
        "transcribed_text": _transcript_for(qualifying_text),
        "detected_words": terms,
        "yamnet_class": acoustic.yamnet_label if acoustic else "NotRun",
        "yamnet_score": acoustic.yamnet_score if acoustic else 0.0,
        "yamnet_ran": acoustic is not None,
        "event_id": incident.event_id,
        # Speech-emotion label (emotion2vec) of the deciding utterance; it
        # only ever changes severity together with a monitored word.
        "emotion": _emotion_label(incident.tone),
        "tone_context": _tone_context(incident),
        "tone_data": ({
            "rms": tone.rms,
            "energy_variance": tone.energy_variance,
            "zero_crossing_rate": tone.zero_crossing_rate,
            "peak_to_average": tone.peak_to_average,
        } if tone else {}),
        "waveform_snapshot": list(incident.waveform_snapshot),
        "language": _language_for(qualifying_text),
        "language_confidence": (
            max(language_confidences) if language_confidences else None
        ),
        "matched_terms": matched,
        "event_start_timestamp": incident.triggered_at,
        "event_end_timestamp": incident.finalized_at,
        # Dashboard Test Mode labels a genuine detector decision at the
        # delivery boundary. It never changes the incident's evidence,
        # severity, confidence, correction provenance, or fusion result.
        "trigger_type": "TEST" if dashboard_test_mode else incident.trigger_type,
        "test_mode": bool(dashboard_test_mode),
        "acoustic_evidence_role": (
            "primary_trigger"
            if incident.trigger_type == "ACOUSTIC"
            else "supporting_only"
        ),
        "device_identifier": device_identifier,
    }


def enqueue_finalized_incident(
    incident: FinalizedIncident,
    *,
    alert_sender=send_alert,
    device_identifier=ECHOSENSE_DEVICE_ID,
    dashboard_test_mode=DASHBOARD_TEST_MODE,
):
    """Validate and SQLite-enqueue one incident through the proven sender."""

    return bool(alert_sender(**finalized_incident_send_arguments(
        incident,
        device_identifier=device_identifier,
        dashboard_test_mode=dashboard_test_mode,
    )))
