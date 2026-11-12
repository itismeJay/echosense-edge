"""Optional sentence semantics with bounded, RAM-only conversation context.

This module owns the model seam and resource policy. It contains no phrase
rules and downloads nothing. A future adapter must implement ``classify`` and
return a calibrated prediction; until then the runtime reports the classifier
as unavailable. Outputs are shadow evidence and cannot affect alert decisions.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import threading
import time
from typing import Mapping, Protocol

from detection.evidence import (
    SEMANTIC_LABELS,
    SemanticEvidence,
)


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    text: str
    language: str
    observed_monotonic: float


@dataclass(frozen=True, slots=True)
class SemanticRequest:
    utterance: str
    language: str
    prior_turns: tuple[ConversationTurn, ...]


@dataclass(frozen=True, slots=True)
class SemanticPrediction:
    label: str
    confidence: float
    model_id: str
    scores: Mapping[str, float]


class SemanticClassifier(Protocol):
    """Adapter contract for an already-loaded, offline classifier."""

    def classify(self, request: SemanticRequest) -> SemanticPrediction:
        """Classify one utterance with bounded preceding context."""


class SemanticAnalyzer:
    """Run an injected classifier and maintain bounded in-memory context."""

    def __init__(
        self,
        *,
        classifier: SemanticClassifier | None = None,
        enabled: bool = False,
        min_confidence: float = 0.75,
        max_context_turns: int = 6,
        context_window_seconds: float = 90.0,
        max_context_chars: int = 4_000,
        timer=time.perf_counter,
    ):
        if not 0.0 <= float(min_confidence) <= 1.0:
            raise ValueError("min_confidence must be between zero and one")
        if not 1 <= int(max_context_turns) <= 64:
            raise ValueError("max_context_turns must be between 1 and 64")
        if not 0.0 <= float(context_window_seconds) <= 3_600.0:
            raise ValueError("context_window_seconds is outside safe bounds")
        if not 1 <= int(max_context_chars) <= 50_000:
            raise ValueError("max_context_chars is outside safe bounds")
        self.classifier = classifier
        self.enabled = bool(enabled)
        self.min_confidence = float(min_confidence)
        self.max_context_turns = int(max_context_turns)
        self.context_window_seconds = float(context_window_seconds)
        self.max_context_chars = int(max_context_chars)
        self._timer = timer
        self._turns = deque(maxlen=self.max_context_turns)
        self._lock = threading.Lock()

    @property
    def available(self):
        return self.enabled and self.classifier is not None

    def clear(self):
        """Forget all conversational text immediately."""

        with self._lock:
            self._turns.clear()

    def _context(self, observed_monotonic):
        cutoff = observed_monotonic - self.context_window_seconds
        while self._turns and self._turns[0].observed_monotonic < cutoff:
            self._turns.popleft()
        selected = []
        characters = 0
        for turn in reversed(self._turns):
            if characters + len(turn.text) > self.max_context_chars:
                break
            selected.append(turn)
            characters += len(turn.text)
        return tuple(reversed(selected))

    def _remember(self, text, language, observed_monotonic):
        # Each source transcript is already capped at 10,000 characters. The
        # tighter context cap prevents a single turn from exceeding the window.
        bounded = text[-self.max_context_chars:]
        self._turns.append(ConversationTurn(
            text=bounded,
            language=language,
            observed_monotonic=observed_monotonic,
        ))

    def assess(
        self,
        transcript: str,
        *,
        language: str,
        observed_monotonic: float,
        quality_accepted: bool,
    ) -> SemanticEvidence:
        """Return bounded shadow evidence; model failures never escape."""

        if not self.enabled:
            return SemanticEvidence(
                status="disabled",
                reasons=("experimental_semantics_disabled",),
            )
        if not quality_accepted or not str(transcript).strip():
            return SemanticEvidence(
                status="skipped",
                reasons=("transcript_quality_not_accepted",),
            )
        if self.classifier is None:
            return SemanticEvidence(
                status="unavailable",
                reasons=("semantic_classifier_not_installed",),
            )

        mono = float(observed_monotonic)
        language = str(language or "unknown").lower()[:32]
        with self._lock:
            context = self._context(mono)
        request = SemanticRequest(
            utterance=str(transcript)[:10_000],
            language=language,
            prior_turns=context,
        )
        started = self._timer()
        try:
            prediction = self.classifier.classify(request)
            elapsed_ms = max(0.0, (self._timer() - started) * 1_000.0)
            evidence = self._validated_evidence(
                prediction,
                context_turns=len(context),
                inference_ms=elapsed_ms,
            )
        except Exception as exc:
            elapsed_ms = max(0.0, (self._timer() - started) * 1_000.0)
            evidence = SemanticEvidence(
                status="error",
                inference_ms=elapsed_ms,
                reasons=(f"classifier_error:{type(exc).__name__}",),
            )
        with self._lock:
            self._remember(request.utterance, language, mono)
        return evidence

    def _validated_evidence(self, prediction, *, context_turns, inference_ms):
        if not isinstance(prediction, SemanticPrediction):
            raise TypeError("classifier must return SemanticPrediction")
        label = str(prediction.label)
        model_id = str(prediction.model_id or "")
        confidence = float(prediction.confidence)
        if label not in SEMANTIC_LABELS:
            raise ValueError("classifier returned an unknown label")
        if not model_id or len(model_id) > 200:
            raise ValueError("classifier returned an invalid model_id")
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("classifier returned an invalid confidence")
        raw_scores = dict(prediction.scores)
        if set(raw_scores) - SEMANTIC_LABELS:
            raise ValueError("classifier returned an unknown score label")
        scores = tuple(sorted(
            (name, float(score)) for name, score in raw_scores.items()
        ))
        for _name, score in scores:
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError("classifier returned an invalid score")
        if confidence < self.min_confidence or label == "uncertain":
            return SemanticEvidence(
                status="low_confidence",
                label="uncertain",
                candidate_label=label,
                confidence=confidence,
                model_id=model_id,
                context_turns=context_turns,
                inference_ms=inference_ms,
                scores=scores,
                reasons=("below_semantic_confidence_gate",),
            )
        return SemanticEvidence(
            status="classified",
            label=label,
            candidate_label=label,
            confidence=confidence,
            model_id=model_id,
            context_turns=context_turns,
            inference_ms=inference_ms,
            scores=scores,
            reasons=("shadow_evidence_only",),
        )
