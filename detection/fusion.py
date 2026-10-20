"""Explainable event-level fusion for text and acoustic observations."""

from __future__ import annotations

from dataclasses import dataclass

from detection.evidence import AcousticEvidence, TextEvidence
from detection.severity import (
    HIGH,
    LOW,
    MEDIUM,
    calculate_severity,
)
from detection.thresholds import YAMNET_THRESHOLD


@dataclass(frozen=True, slots=True)
class EventFusionDecision:
    level: str
    reasons: tuple[str, ...]
    term_categories: dict[str, tuple[str, ...]]
    supporting_evidence: tuple[str, ...]
    confidence: float


def _strong_acoustic(evidence):
    qualifying = [item for item in evidence if item.qualifies_to_open]
    strong_yamnet = any(
        item.yamnet_aggressive
        and item.yamnet_score >= max(0.65, YAMNET_THRESHOLD + 0.15)
        for item in qualifying
    )
    multi_feature_repeated = sum(item.tone_aggressive for item in qualifying) >= 2
    sustained_duration = sum(item.duration_seconds for item in qualifying) >= 1.5
    return bool(strong_yamnet or (multi_feature_repeated and sustained_duration))


def fuse_event_evidence(
    text_evidence: tuple[TextEvidence, ...],
    acoustic_evidence: tuple[AcousticEvidence, ...],
    *,
    duration_seconds: float,
):
    """Fuse observable evidence without treating loudness/duration as proof."""

    qualifying_text = [item for item in text_evidence if item.qualifies_to_open]
    qualifying_acoustic = [
        item for item in acoustic_evidence if item.qualifies_to_open
    ]
    if not qualifying_text and not qualifying_acoustic:
        raise ValueError("no qualifying detector evidence to fuse")

    terms = tuple(dict.fromkeys(
        term
        for item in qualifying_text
        for term in item.detected_terms
    ))
    transcript = " ".join(item.transcript for item in qualifying_text)
    repeated = any(item.repeated for item in qualifying_text)
    text_decision = calculate_severity(
        terms,
        transcript=transcript,
        repeated=repeated,
        acoustic_aggressive=False,
        laughter_present=any(item.laughter_present for item in qualifying_text),
    )
    reasons = list(text_decision.reasons if qualifying_text else ())
    supporting = list(
        text_decision.supporting_evidence if qualifying_text else ()
    )
    categories = dict(text_decision.term_categories if qualifying_text else {})
    level = text_decision.level if qualifying_text else LOW
    strong_acoustic = _strong_acoustic(qualifying_acoustic)

    if qualifying_acoustic:
        supporting.append("qualifying_acoustic_evidence")
        if any(item.yamnet_aggressive for item in qualifying_acoustic):
            supporting.append("yamnet_aggressive_class")
        if any(item.tone_aggressive for item in qualifying_acoustic):
            supporting.append("multi_feature_aggressive_tone")
        if strong_acoustic:
            supporting.append("strong_or_sustained_acoustic_evidence")

    if qualifying_text and qualifying_acoustic:
        supporting.append("text_acoustic_corroboration")
        if level == LOW:
            level = MEDIUM
            reasons.append("promoted_by:text_acoustic_corroboration")
        elif level == MEDIUM and strong_acoustic:
            level = HIGH
            reasons.append("promoted_by:strong_acoustic_corroboration")
    elif qualifying_acoustic:
        # Conservative initial policy: acoustic-only incidents never become HIGH.
        level = MEDIUM if strong_acoustic else LOW
        reasons.append(
            "acoustic_only:strong_or_sustained"
            if strong_acoustic
            else "acoustic_only:qualifying_observation"
        )

    if duration_seconds > 0:
        supporting.append("incident_duration_observed")
    if qualifying_text and qualifying_acoustic:
        confidence = 0.9 if level == HIGH else 0.8
    elif qualifying_text:
        confidence = {LOW: 0.6, MEDIUM: 0.75, HIGH: 0.9}[level]
    else:
        confidence = 0.75 if level == MEDIUM else 0.6
    return EventFusionDecision(
        level=level,
        reasons=tuple(dict.fromkeys(reasons)),
        term_categories=categories,
        supporting_evidence=tuple(dict.fromkeys(supporting)),
        confidence=confidence,
    )
