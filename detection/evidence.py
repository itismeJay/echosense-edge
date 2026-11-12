"""Typed, non-PCM detector and finalized-incident evidence contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math

from detection.severity import SEVERITY_LEVELS


SEMANTIC_LABELS = frozenset({
    "neutral",
    "academic_discussion",
    "playful_teasing",
    "targeted_belittling",
    "threat",
    "repeated_harassment",
    "uncertain",
})
SEMANTIC_STATUSES = frozenset({
    "disabled",
    "unavailable",
    "classified",
    "low_confidence",
    "skipped",
    "error",
})


def _aware(value: datetime, name: str):
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")


def _finite(value: float, name: str):
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class MatchedTermEvidence:
    term: str
    language: str | None = None
    match_type: str = "exact"
    term_id: int | None = None

    def backend_value(self):
        value = {"term": self.term, "match_type": self.match_type}
        if self.language:
            value["language"] = self.language
        if self.term_id is not None:
            value["term_id"] = self.term_id
        return value


@dataclass(frozen=True, slots=True)
class DetectedTermProvenance:
    term: str
    provenance: str


@dataclass(frozen=True, slots=True)
class TranscriptCorrectionEvidence:
    heard: str
    term: str
    kind: str
    provenance: str


@dataclass(frozen=True, slots=True)
class SemanticEvidence:
    """Bounded shadow-mode output from an optional sentence classifier.

    ``label`` is always ``uncertain`` for low-confidence or failed inference.
    ``candidate_label`` retains the model's untrusted output for evaluation.
    This evidence is deliberately separate from alert qualification.
    """

    status: str = "disabled"
    label: str = "uncertain"
    candidate_label: str | None = None
    confidence: float = 0.0
    model_id: str = "none"
    context_turns: int = 0
    inference_ms: float = 0.0
    scores: tuple[tuple[str, float], ...] = ()
    reasons: tuple[str, ...] = ()

    def __post_init__(self):
        if self.status not in SEMANTIC_STATUSES:
            raise ValueError("semantic status is invalid")
        if self.label not in SEMANTIC_LABELS:
            raise ValueError("semantic label is invalid")
        if (
            self.candidate_label is not None
            and self.candidate_label not in SEMANTIC_LABELS
        ):
            raise ValueError("semantic candidate label is invalid")
        _finite(self.confidence, "semantic confidence")
        _finite(self.inference_ms, "semantic inference_ms")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("semantic confidence must be between zero and one")
        if self.inference_ms < 0.0:
            raise ValueError("semantic inference_ms must be non-negative")
        if not 0 <= self.context_turns <= 64:
            raise ValueError("semantic context_turns is invalid")
        if not self.model_id or len(self.model_id) > 200:
            raise ValueError("semantic model_id is invalid")
        if len(self.scores) > len(SEMANTIC_LABELS):
            raise ValueError("semantic scores exceed their bound")
        seen = set()
        for name, score in self.scores:
            if name not in SEMANTIC_LABELS or name in seen:
                raise ValueError("semantic scores contain an invalid label")
            seen.add(name)
            _finite(score, f"semantic score {name}")
            if not 0.0 <= score <= 1.0:
                raise ValueError("semantic score must be between zero and one")
        if len(self.reasons) > 20:
            raise ValueError("semantic reasons exceed their bound")


@dataclass(frozen=True, slots=True)
class TextEvidence:
    observed_at: datetime
    observed_monotonic: float
    transcript: str
    normalized_transcript: str
    language: str
    language_confidence: float | None
    matched_terms: tuple[MatchedTermEvidence, ...]
    detected_terms: tuple[str, ...]
    hard_hits: tuple[str, ...]
    soft_hits: tuple[str, ...]
    categories: tuple[str, ...]
    direct_target: bool
    repeated: bool
    quality_accepted: bool
    quality_reasons: tuple[str, ...]
    quality_warnings: tuple[str, ...]
    baseline_severity: str
    reasons: tuple[str, ...]
    laughter_present: bool
    qualifies_to_open: bool
    # Wall-clock start of the spoken utterance, when known. observed_at is the
    # utterance end; together they bound when the words were actually said.
    spoken_started_at: datetime | None = None
    # The transcript after mishearing/sound-alike correction (see
    # detection/soundalike.py). Severity and targeting judge this text; the
    # payload always carries the original ``transcript``.
    match_text: str = ""
    # Internal evidence only. The schema-v2 matched_terms contract remains
    # unchanged; match_type still means dictionary word/phrase shape.
    term_provenance: tuple[DetectedTermProvenance, ...] = ()
    corrections: tuple[TranscriptCorrectionEvidence, ...] = ()
    # None preserves legacy constructed evidence. An explicit empty tuple means
    # terms were observed but none may independently qualify this incident.
    qualification_terms: tuple[str, ...] | None = None
    # Shadow evidence only. It cannot open, escalate, suppress, or deliver an
    # incident until a separately reviewed release changes that policy.
    semantic: SemanticEvidence | None = None
    # Per-utterance tone signals from model/emotion_recognizer.py (emotion2vec
    # label/scores, YAMNet shout/laughter maxima, speech-RMS ratio) and the
    # transcriber's audio-event tags ("laughs", "shouting"). Judged by
    # detection/tone_vote.py; tone alone never opens an incident.
    tone_signals: dict | None = None
    audio_tags: tuple[str, ...] = ()

    @property
    def decision_text(self) -> str:
        return self.match_text or self.transcript

    @property
    def decision_terms(self) -> tuple[str, ...]:
        if self.qualification_terms is None:
            return self.detected_terms
        return self.qualification_terms

    def __post_init__(self):
        _aware(self.observed_at, "observed_at")
        if self.spoken_started_at is not None:
            _aware(self.spoken_started_at, "spoken_started_at")
            if self.spoken_started_at > self.observed_at:
                raise ValueError("spoken_started_at must not follow observed_at")
        _finite(self.observed_monotonic, "observed_monotonic")
        if self.baseline_severity not in SEVERITY_LEVELS:
            raise ValueError("baseline_severity is invalid")
        if (
            len(self.transcript) > 10_000
            or len(self.normalized_transcript) > 10_000
            or len(self.match_text) > 10_000
        ):
            raise ValueError("transcript evidence exceeds its bound")
        for name in (
            "matched_terms",
            "detected_terms",
            "hard_hits",
            "soft_hits",
            "categories",
            "term_provenance",
            "corrections",
        ):
            if len(getattr(self, name)) > 50:
                raise ValueError(f"{name} exceeds its bound")
        if self.qualification_terms is not None and len(self.qualification_terms) > 50:
            raise ValueError("qualification_terms exceeds its bound")
        if len(self.reasons) > 100:
            raise ValueError("text reasons exceed their bound")
        if self.semantic is not None and not isinstance(
            self.semantic, SemanticEvidence
        ):
            raise TypeError("semantic must be SemanticEvidence")
        if self.tone_signals is not None and not isinstance(self.tone_signals, dict):
            raise TypeError("tone_signals must be a dict")
        if len(self.audio_tags) > 20:
            raise ValueError("audio_tags exceeds its bound")


@dataclass(frozen=True, slots=True)
class AcousticEvidence:
    interval_started_at: datetime
    interval_ended_at: datetime
    monotonic_start: float
    monotonic_end: float
    yamnet_ran: bool
    yamnet_label: str
    yamnet_score: float
    yamnet_aggressive: bool
    tone_aggressive: bool
    rms: float
    energy_variance: float
    zero_crossing_rate: float
    peak_to_average: float
    duration_seconds: float
    reasons: tuple[str, ...]
    qualifies_to_open: bool
    # Observed YAMNet event only. Laughter is not evidence of harmlessness,
    # playfulness, emotion, or intent and never qualifies an incident.
    laughter_detected: bool = False
    laughter_score: float = 0.0
    # Max YAMNet score of the laughter classes (Laughter, Giggle, Chuckle...)
    # in this window even when they are not the top class.
    laughter_class_score: float = 0.0

    def __post_init__(self):
        _aware(self.interval_started_at, "interval_started_at")
        _aware(self.interval_ended_at, "interval_ended_at")
        if self.interval_ended_at < self.interval_started_at:
            raise ValueError("acoustic wall-clock interval is invalid")
        for name in (
            "monotonic_start",
            "monotonic_end",
            "yamnet_score",
            "rms",
            "energy_variance",
            "zero_crossing_rate",
            "peak_to_average",
            "duration_seconds",
            "laughter_score",
            "laughter_class_score",
        ):
            _finite(getattr(self, name), name)
        if self.monotonic_end < self.monotonic_start:
            raise ValueError("acoustic monotonic interval is invalid")
        if not 0.0 <= self.yamnet_score <= 1.0:
            raise ValueError("yamnet_score must be between zero and one")
        if not 0.0 <= self.laughter_score <= 1.0:
            raise ValueError("laughter_score must be between zero and one")
        if not 0.0 <= self.laughter_class_score <= 1.0:
            raise ValueError("laughter_class_score must be between zero and one")
        if self.laughter_detected and not self.yamnet_ran:
            raise ValueError("laughter detection requires YAMNet inference")
        if self.duration_seconds < 0:
            raise ValueError("duration_seconds must be non-negative")
        if len(self.yamnet_label) > 200 or len(self.reasons) > 50:
            raise ValueError("acoustic evidence exceeds its bound")


@dataclass(frozen=True, slots=True)
class FinalizedIncident:
    event_id: str
    trigger_type: str
    triggered_at: datetime
    finalized_at: datetime
    duration_seconds: float
    text_evidence: tuple[TextEvidence, ...]
    acoustic_evidence: tuple[AcousticEvidence, ...]
    severity: str
    severity_reasons: tuple[str, ...]
    term_categories: dict[str, tuple[str, ...]]
    supporting_evidence: tuple[str, ...]
    confidence: float
    waveform_snapshot: tuple[int, ...] = ()
    # Plain-language reason line and the tone vote behind it (sent inside the
    # existing tone_evidence / laughter_context payload fields).
    summary: str = ""
    tone: object = None

    def __post_init__(self):
        if self.trigger_type not in {"KEYWORD", "ACOUSTIC"}:
            raise ValueError("trigger_type must identify the opening detector")
        _aware(self.triggered_at, "triggered_at")
        _aware(self.finalized_at, "finalized_at")
        if self.finalized_at < self.triggered_at:
            raise ValueError("finalized_at must not precede triggered_at")
        if self.severity not in SEVERITY_LEVELS:
            raise ValueError("severity is invalid")
        if not self.text_evidence and not self.acoustic_evidence:
            raise ValueError("a finalized incident requires detector evidence")
        if self.duration_seconds < 0:
            raise ValueError("duration_seconds must be non-negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")
        if len(self.text_evidence) > 256 or len(self.acoustic_evidence) > 256:
            raise ValueError("finalized evidence history exceeds its bound")
        if len(self.waveform_snapshot) > 256 or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= 32768
            for value in self.waveform_snapshot
        ):
            raise ValueError("waveform snapshot is invalid")
