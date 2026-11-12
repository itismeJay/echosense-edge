"""Explainable event-level fusion for text and acoustic observations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from config import DetectionRuntimeConfig
from detection import school_policy
from detection.evidence import AcousticEvidence, TextEvidence
from detection.llm_semantic import (
    BANTER_LABELS,
    BANTER_MIN_CONFIDENCE,
    THREAT_REVIEW_LABELS,
    THREAT_REVIEW_MIN_CONFIDENCE,
    meaning_only_severity,
)
from detection.soundalike import FUZZY_PROVENANCE
from detection import tone_vote
from detection.text_detector import FUZZY_REVIEW_REASON
from detection.severity import (
    HIGH,
    LOW,
    MEDIUM,
    SEVERITY_ORDER,
    calculate_severity,
    is_descriptive_only,
    is_targeted,
)


_DEFAULT_CONFIG = DetectionRuntimeConfig()

# Short ASR outputs are especially vulnerable to noise/hallucination. These
# terms remain observable exact evidence, but need targeting, repetition, or
# time-linked aggressive acoustics before an alert leaves the device.
AMBIGUOUS_SHORT_TERMS = frozenset({"iyot"})

# YAMNet classes that mean someone is crying, and the score that counts.
CRYING_CLASSES = frozenset({"Crying, sobbing", "Baby cry, infant cry", "Whimper"})
CRYING_MIN_SCORE = 0.3

# Plain-language severity reasons added by the semantic rules.
BANTER_REASON = "possible joking/banter \u2014 check context"
ANGRY_BANTER_REASON = "angry insult (AI suggested joking \u2014 check)"
MEANING_REASON = "detected by meaning"
MEANING_REVIEW_REASON = "possible concern \u2014 check context"
DISTRESS_REASON = "possible victim distress"
REPEATED_REASON = "repeated"
THREAT_REVIEW_REASON = "possible game/story talk \u2014 check context"
_THREAT_CATEGORIES = ("threat_like_phrase", "self_harm_directive")


@dataclass(frozen=True, slots=True)
class EventFusionDecision:
    level: str
    reasons: tuple[str, ...]
    term_categories: dict[str, tuple[str, ...]]
    supporting_evidence: tuple[str, ...]
    confidence: float
    # False when policy says this incident must not be sent at all.
    deliver: bool = True
    # Plain-language reason line of the deciding utterance.
    summary: str = ""
    # detection.tone_vote.ToneVote of the deciding utterance (or None).
    tone: object = None


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


def fuzzy_only(text_item) -> bool:
    """True when every term that opened the utterance is a fuzzy match."""

    provenance = {item.term: item.provenance for item in text_item.term_provenance}
    terms = text_item.decision_terms
    return bool(terms) and all(
        provenance.get(term) == FUZZY_PROVENANCE for term in terms
    )


def _targeted_utterances(text_items):
    return sum(
        1
        for item in text_items
        if is_targeted(item.decision_terms, item.decision_text)
    )


def _tone_for(text_item, linked, thresholds=None):
    """Tone vote for one utterance: its own clip + laughter in linked windows."""

    linked_laughter = max(
        (
            max(item.laughter_class_score, item.laughter_score)
            for item in linked
        ),
        default=0.0,
    )
    return tone_vote.vote(
        text_item.tone_signals,
        text_item.audio_tags,
        linked_laughter_score=linked_laughter,
        thresholds=thresholds,
    )


def crying_after(text_item, acoustic_evidence) -> bool:
    """YAMNet heard crying in a window that starts at or after the words."""

    spoken = text_item.spoken_started_at or text_item.observed_at
    return any(
        item.yamnet_ran
        and item.yamnet_label in CRYING_CLASSES
        and item.yamnet_score >= CRYING_MIN_SCORE
        and item.interval_started_at >= spoken
        for item in acoustic_evidence
    )


def _bump(level):
    return {LOW: MEDIUM, MEDIUM: HIGH, HIGH: HIGH}[level]


def _semantic_evidence(judgement, *, banter, repeated, distress=False):
    items = []
    if judgement is not None:
        if judgement.ok:
            items.extend([
                f"semantic_label:{judgement.label}",
                f"semantic_confidence:{judgement.confidence:.2f}",
                f"semantic_targeted:{str(judgement.targeted).lower()}",
            ])
            if judgement.victim_distress:
                # Claude's flag counts only with a reaction in another utterance.
                items.append(
                    "semantic_victim_distress:"
                    + ("true" if distress else "unconfirmed")
                )
            if judgement.reason:
                items.append(f"semantic_reason:{judgement.reason[:160]}")
            items.append(f"semantic_model:{judgement.model_id[:60]}")
        elif judgement.status == "fallback":
            items.append(
                f"semantic_fallback:lexicon_only:{judgement.fallback_reason[:60]}"
            )
    items.append(f"semantic_banter:{str(banter).lower()}")
    items.append(f"semantic_repeated:{str(repeated).lower()}")
    return items


def apply_semantic_rules(
    level, text_item, acoustic_evidence, categories, summary,
    *, angry_tone=False, targeted=False, others=(),
):
    """Claude-judgement and repetition rules on top of the lexicon decision.

    Lexicon threat         HIGH; banter/quoting >= 0.9 + game/story word -> MEDIUM
    word + banter >= 0.8   LOW "possible joking/banter" (never dropped),
                           unless angry tone + targeted: stays HIGH
    meaning-only           conf >= 0.90 MEDIUM (threat >= 0.95 HIGH), 0.75-0.90
                           LOW "possible concern", below 0.75 no alert (caller)
    distress / crying      +1 level (reaction in a different utterance:
                           ``others`` in this incident, or a previous turn
                           Claude also flagged; or YAMNet crying after)
    3+ similar in 10 min   +1 level
    Returns (level, reasons, supporting, summary).
    """

    judgement = text_item.llm
    ok = judgement is not None and judgement.ok
    reasons, supporting = [], []
    # Similarity-window (fuzzy) guesses keep their LOW review cap.
    lexicon_threat = (
        any(categories.get(name) for name in _THREAT_CATEGORIES)
        and not fuzzy_only(text_item)
    )
    banter = False
    if lexicon_threat:
        level = HIGH
        if (
            ok
            and judgement.label in THREAT_REVIEW_LABELS
            and judgement.confidence >= THREAT_REVIEW_MIN_CONFIDENCE
            and text_item.game_story_context
        ):
            level = MEDIUM
            reasons.append(THREAT_REVIEW_REASON)
            summary = "Threat words in game/story talk \u2014 check context"
    elif (
        ok
        and not text_item.opened_by_meaning
        and judgement.label in BANTER_LABELS
        and judgement.confidence >= BANTER_MIN_CONFIDENCE
    ):
        banter = True
        if angry_tone and targeted:
            # Angry voice aimed at someone outweighs Claude's "joking".
            level = HIGH
            reasons.append(ANGRY_BANTER_REASON)
            summary = "Angry insult (AI suggested joking \u2014 check)"
        else:
            level = LOW
            reasons.append(BANTER_REASON)
            summary = "Possible joking/banter \u2014 check context"
    # Distress needs an actual reaction in a DIFFERENT utterance: one in
    # this incident ("tama na", "stop", crying tag), or a previous turn that
    # Claude also read as distress. Never inferred from the threat itself.
    reaction_now = any(
        other is not text_item and getattr(other, "victim_reaction", False)
        for other in others
    )
    distress = reaction_now or bool(
        ok and judgement.victim_distress and text_item.reaction_in_previous_turns
    )
    crying = crying_after(text_item, acoustic_evidence)
    if distress or crying:
        if reaction_now:
            supporting.append("victim_reaction_in_incident")
        elif distress:
            supporting.append("victim_reaction_in_previous_turns")
        level = _bump(level)
        reasons.append(DISTRESS_REASON)
        if crying:
            supporting.append("yamnet_crying_after_insult")
        summary = f"{summary}; possible victim distress" if summary else DISTRESS_REASON
    if text_item.pattern_repeated:
        level = _bump(level)
        reasons.append(REPEATED_REASON)
        summary = (
            f"{summary}; repeated 3+ times in 10 minutes"
            if summary else "Repeated 3+ times in 10 minutes"
        )
    if lexicon_threat and SEVERITY_ORDER[level] < SEVERITY_ORDER[MEDIUM]:
        level = MEDIUM
    supporting.extend(_semantic_evidence(
        judgement, banter=banter, repeated=text_item.pattern_repeated,
        distress=distress,
    ))
    return level, reasons, supporting, summary


def _fuse_meaning_only(text_item, acoustic_evidence, config, others=()):
    """No monitored word: Claude found a targeted insult/threat/exclusion/shaming."""

    linked = linked_acoustic(
        (text_item,), acoustic_evidence, config.corroboration_window_seconds,
    )
    vote = _tone_for(text_item, linked)
    judgement = text_item.llm
    level = meaning_only_severity(judgement.label, judgement.confidence) or LOW
    summary = f"Detected by meaning: {judgement.label} (no monitored word)"
    reasons = [MEANING_REASON]
    if level == LOW:
        reasons.append(MEANING_REVIEW_REASON)
        summary = f"Possible concern \u2014 check context ({judgement.label}, no monitored word)"
    level, extra_reasons, extra_supporting, summary = apply_semantic_rules(
        level, text_item, acoustic_evidence, {}, summary, others=others,
    )
    supporting = list(vote.evidence()) + extra_supporting
    return (
        level,
        reasons + extra_reasons,
        supporting,
        {},
        vote,
        summary,
        bool([item for item in linked if item.qualifies_to_open]),
    )


def _fuse_text(text_item, acoustic_evidence, targeted_count, config, others=()):
    if text_item.opened_by_meaning and text_item.llm is not None:
        return _fuse_meaning_only(text_item, acoustic_evidence, config, others)
    linked = linked_acoustic(
        (text_item,),
        acoustic_evidence,
        config.corroboration_window_seconds,
    )
    linked_qualifying = [item for item in linked if item.qualifies_to_open]
    repeated_in_incident = targeted_count >= 2 and is_targeted(
        text_item.decision_terms, text_item.decision_text,
    )
    vote = _tone_for(text_item, linked)
    laughter = vote.laughter or text_item.laughter_present
    decision = calculate_severity(
        text_item.decision_terms,
        transcript=text_item.decision_text,
        repeated=repeated_in_incident,
        laughter_present=laughter,
        angry_tone=vote.angry_tone,
        angry_cues=vote.loud_cues,
        # Loud/YAMNet-only windows are recorded, never used as aggression.
        acoustic_aggressive=bool(linked_qualifying),
    )
    level = decision.level
    reasons = list(decision.reasons)
    summary = decision.summary
    if fuzzy_only(text_item) and level != LOW:
        # Only similarity-window guesses, no exact word: needs review.
        level = LOW
        reasons.append(FUZZY_REVIEW_REASON)
        summary = "Possible monitored word (uncertain match), needs review"
    level, semantic_reasons, semantic_supporting, summary = apply_semantic_rules(
        level, text_item, acoustic_evidence, decision.term_categories, summary,
        angry_tone=vote.angry_tone,
        targeted=is_targeted(text_item.decision_terms, text_item.decision_text),
        others=others,
    )
    reasons.extend(semantic_reasons)
    supporting = list(decision.supporting_evidence)
    supporting.extend(semantic_supporting)
    supporting.extend(vote.evidence())
    if repeated_in_incident:
        supporting.append("targeted_repetition_within_incident")
    if linked_qualifying:
        # Observation only: loud/YAMNet windows never promote by themselves.
        supporting.append("text_acoustic_corroboration")
    if any(item.laughter_detected for item in linked):
        supporting.append("acoustic_laughter_observed")
    if vote.laughter:
        supporting.append("laughter_context")
    return (
        level,
        reasons,
        supporting,
        decision.term_categories,
        vote,
        summary,
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

    Each text observation is scored on its own transcript and its own tone
    vote (detection/tone_vote.py: speech emotion = angry AND a shout cue).
    Without a monitored word nothing is delivered.
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
    # True when an utterance was said with angry tone (tone_vote). Loudness
    # alone never sets it.
    corroborated = False
    summary = ""
    deciding_tone = None
    any_linked = False
    if qualifying_text:
        level = None
        targeted_count = _targeted_utterances(qualifying_text)
        for item in qualifying_text:
            (
                item_level, item_reasons, item_supporting, item_categories,
                vote, item_summary, linked,
            ) = _fuse_text(
                item, acoustic_evidence, targeted_count, config, text_evidence,
            )
            any_linked = any_linked or linked
            if level is None or SEVERITY_ORDER[item_level] > SEVERITY_ORDER[level]:
                level = item_level
                summary = item_summary
                deciding_tone = vote
            reasons.extend(item_reasons)
            supporting.extend(item_supporting)
            for category, terms in item_categories.items():
                categories[category] = tuple(dict.fromkeys(
                    categories.get(category, ()) + terms
                ))
            corroborated = corroborated or vote.angry_tone
        if qualifying_acoustic and not any_linked:
            supporting.append("unlinked_acoustic_evidence_ignored")
        # Soft descriptive words ("ugly", "too short") are everyday speech:
        # send them only when aimed at someone or said in an aggressive tone.
        # A school on Strict sensitivity wants them sent anyway.
        descriptive_only = is_descriptive_only(
            term for item in qualifying_text for term in item.decision_terms
        )
        if descriptive_only and not targeted_count and not corroborated:
            if school_policy.is_strict():
                supporting.append("strict_sensitivity:descriptive_untargeted_sent")
            else:
                deliver = False
                reasons.append("descriptive_untargeted:never_sent")
        decision_terms = [
            term for item in qualifying_text for term in item.decision_terms
        ]
        ambiguous_short_only = bool(decision_terms) and set(decision_terms) <= (
            AMBIGUOUS_SHORT_TERMS
        )
        repeated_support = len(qualifying_text) >= 2 or any(
            item.repeated for item in qualifying_text
        )
        if (
            ambiguous_short_only
            and not targeted_count
            and not corroborated
            and not repeated_support
        ):
            deliver = False
            reasons.append("ambiguous_short_term:needs_support")
    else:
        # Sound never creates an alert by itself: kept in the local log only.
        level = LOW
        deliver = False
        summary = "Sound only, no monitored word: not sent"
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
        summary=summary,
        tone=deciding_tone,
    )
