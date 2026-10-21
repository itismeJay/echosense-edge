"""Explainable event-level fusion for text and acoustic observations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from config import DetectionRuntimeConfig
from detection.evidence import AcousticEvidence, TextEvidence
from detection.severity import (
    HIGH,
    LOW,
    MEDIUM,
    SEVERITY_ORDER,
    calculate_severity,
    is_targeted,
)


_DEFAULT_CONFIG = DetectionRuntimeConfig()


@dataclass(frozen=True, slots=True)
class EventFusionDecision:
    level: str
    reasons: tuple[str, ...]
    term_categories: dict[str, tuple[str, ...]]
    supporting_evidence: tuple[str, ...]
    confidence: float
    # False when policy says this incident must not be sent at all.
    deliver: bool = True


def is_time_linked(text: TextEvidence, acoustic: AcousticEvidence, window_seconds):
    """True when the acoustic window overlaps the utterance +/- the window.

    Wall-clock times are used because text is observed only after Whisper
    finishes; its monotonic timestamp is transcription time, not speech time.
    """

    window = timedelta(seconds=float(window_seconds))
    spoken_start = text.spoken_started_at or text.observed_at
    return (
        acoustic.interval_started_at <= text.observed_at + window
        and acoustic.interval_ended_at >= spoken_start - window
    )


def linked_acoustic(text_items, acoustic_items, window_seconds):
    return [
        acoustic
        for acoustic in acoustic_items
        if any(
            is_time_linked(text, acoustic, window_seconds)
            for text in text_items
        )
    ]


def _covered_seconds(windows):
    """Length of the union of window intervals (windows overlap by design)."""

    total = 0.0
    current_start = current_end = None
    for item in sorted(windows, key=lambda w: w.monotonic_start):
        if current_end is None or item.monotonic_start > current_end:
            if current_end is not None:
                total += current_end - current_start
            current_start, current_end = item.monotonic_start, item.monotonic_end
        else:
            current_end = max(current_end, item.monotonic_end)
    if current_end is not None:
        total += current_end - current_start
    return total


def strong_acoustic(windows, config=_DEFAULT_CONFIG):
    """Strong = a confident YAMNet aggressive class, or a sustained share of
    tone-aggressive windows among all the windows considered."""

    windows = list(windows)
    if not windows:
        return False
    if any(
        item.yamnet_aggressive and item.yamnet_score >= config.strong_yamnet_score
        for item in windows
    ):
        return True
    tone = [item for item in windows if item.tone_aggressive]
    return bool(
        tone
        and len(tone) / len(windows) >= config.strong_tone_ratio
        and _covered_seconds(tone) >= config.strong_tone_min_seconds
    )


def _targeted_utterances(text_items):
    return sum(
        1
        for item in text_items
        if is_targeted(item.detected_terms, item.transcript)
    )


def _fuse_text(text_item, acoustic_evidence, targeted_count, config):
    linked = linked_acoustic(
        (text_item,),
        acoustic_evidence,
        config.corroboration_window_seconds,
    )
    linked_qualifying = [item for item in linked if item.qualifies_to_open]
    strong = strong_acoustic(linked, config) if linked_qualifying else False
    repeated_in_incident = targeted_count >= 2 and is_targeted(
        text_item.detected_terms, text_item.transcript,
    )
    decision = calculate_severity(
        text_item.detected_terms,
        transcript=text_item.transcript,
        repeated=repeated_in_incident,
        acoustic_aggressive=bool(linked_qualifying),
        laughter_present=text_item.laughter_present,
        strong_acoustic=strong,
    )
    supporting = list(decision.supporting_evidence)
    if repeated_in_incident:
        supporting.append("targeted_repetition_within_incident")
    if linked_qualifying:
        supporting.append("text_acoustic_corroboration")
    return (
        decision.level,
        list(decision.reasons),
        supporting,
        decision.term_categories,
        bool(linked_qualifying),
    )


def fuse_event_evidence(
    text_evidence: tuple[TextEvidence, ...],
    acoustic_evidence: tuple[AcousticEvidence, ...],
    *,
    duration_seconds: float,
    config: DetectionRuntimeConfig | None = None,
):
    """Fuse observable evidence without treating loudness/duration as proof.

    Each text observation is scored on its own transcript, and only acoustic
    windows time-linked to that observation may promote it. Without a
    monitored word nothing is delivered.
    """

    config = config or _DEFAULT_CONFIG
    qualifying_text = [item for item in text_evidence if item.qualifies_to_open]
    qualifying_acoustic = [
        item for item in acoustic_evidence if item.qualifies_to_open
    ]
    if not qualifying_text and not qualifying_acoustic:
        raise ValueError("no qualifying detector evidence to fuse")

    reasons = []
    supporting = []
    categories = {}
    deliver = True
    corroborated = False
    if qualifying_text:
        level = LOW
        targeted_count = _targeted_utterances(qualifying_text)
        for item in qualifying_text:
            item_level, item_reasons, item_supporting, item_categories, linked = (
                _fuse_text(item, acoustic_evidence, targeted_count, config)
            )
            if SEVERITY_ORDER[item_level] > SEVERITY_ORDER[level]:
                level = item_level
            reasons.extend(item_reasons)
            supporting.extend(item_supporting)
            for category, terms in item_categories.items():
                categories[category] = tuple(dict.fromkeys(
                    categories.get(category, ()) + terms
                ))
            corroborated = corroborated or linked
        if qualifying_acoustic and not corroborated:
            supporting.append("unlinked_acoustic_evidence_ignored")
    else:
        # Sound never creates an alert by itself: kept in the local log only.
        level = LOW
        deliver = False
        reasons.append("acoustic_only:never_sent")
        if strong_acoustic(acoustic_evidence, config):
            supporting.append("strong_or_sustained_acoustic_evidence")

    if qualifying_acoustic:
        supporting.append("qualifying_acoustic_evidence")
        if any(item.yamnet_aggressive for item in qualifying_acoustic):
            supporting.append("yamnet_aggressive_class")
        if any(item.tone_aggressive for item in qualifying_acoustic):
            supporting.append("multi_feature_aggressive_tone")

    if duration_seconds > 0:
        supporting.append("incident_duration_observed")
    if corroborated:
        confidence = 0.9 if level == HIGH else 0.8
    else:
        confidence = {LOW: 0.6, MEDIUM: 0.75, HIGH: 0.9}[level]
    return EventFusionDecision(
        level=level,
        reasons=tuple(dict.fromkeys(reasons)),
        term_categories=categories,
        supporting_evidence=tuple(dict.fromkeys(supporting)),
        confidence=confidence,
        deliver=deliver,
    )
