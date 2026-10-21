"""Independent WHAT-was-said detector producing structured text evidence."""

from __future__ import annotations

from collections import OrderedDict, deque
from datetime import datetime, timezone
import threading
import time

from detection.evidence import MatchedTermEvidence, TextEvidence
from detection.severity import HIGH, calculate_severity, has_direct_target_evidence
from detection.transcript_quality import assess_transcript_quality
from model.blacklist import check_transcript


class TextDetector:
    """Analyze finalized transcripts without invoking audio inference or I/O."""

    def __init__(
        self,
        *,
        monotonic_clock=time.monotonic,
        wall_clock=lambda: datetime.now(timezone.utc),
        repetition_window_seconds=60.0,
        history_size=128,
    ):
        if repetition_window_seconds < 0 or history_size <= 0:
            raise ValueError("text history bounds are invalid")
        self._monotonic_clock = monotonic_clock
        self._wall_clock = wall_clock
        self._repetition_window = float(repetition_window_seconds)
        self._history_size = int(history_size)
        self._history = OrderedDict()
        self._lock = threading.Lock()
        self.evidence_count = 0

    def analyze_transcript(
        self,
        transcript: str,
        *,
        language="unknown",
        language_confidence=None,
        observed_at=None,
        observed_monotonic=None,
    ):
        quality = assess_transcript_quality(transcript)
        result = check_transcript(transcript) if quality.accepted else {
            "has_profanity": False,
            "detected_words": [],
            "hard_hits": [],
            "soft_hits": [],
            "categories": [],
            "matched_terms": [],
            "context_suppressed_all": False,
            "is_casual": False,
        }
        result.update({
            "transcribed_text": transcript,
            "normalized_text": quality.normalized_text,
            "language": language,
            "language_confidence": language_confidence,
            "quality_accepted": quality.accepted,
            "quality_reason_codes": quality.reason_codes,
            "quality_warnings": quality.warnings,
            "transcript_quality": quality,
        })
        return self.analyze_stt_result(
            result,
            observed_at=observed_at,
            observed_monotonic=observed_monotonic,
        )

    def analyze_stt_result(
        self,
        result: dict,
        *,
        observed_at=None,
        observed_monotonic=None,
    ):
        """Convert one STT result to evidence; no sender is reachable here."""

        mono = float(
            self._monotonic_clock()
            if observed_monotonic is None
            else observed_monotonic
        )
        audio_event = result.get("audio_event")
        wall = observed_at
        if wall is None and audio_event is not None:
            wall = getattr(audio_event, "ended_at", None)
        wall = (wall or self._wall_clock()).astimezone(timezone.utc)
        spoken_started_at = getattr(audio_event, "started_at", None)
        if spoken_started_at is not None:
            spoken_started_at = spoken_started_at.astimezone(timezone.utc)
            if spoken_started_at > wall:
                spoken_started_at = None
        transcript = str(result.get("transcribed_text", result.get("transcript", "")))
        normalized = str(result.get("normalized_text") or "")
        quality = result.get("transcript_quality")
        quality_accepted = bool(result.get("quality_accepted", True))
        quality_reasons = tuple(
            result.get("quality_reason_codes")
            or getattr(quality, "reason_codes", ())
        )
        quality_warnings = tuple(
            result.get("quality_warnings")
            or getattr(quality, "warnings", ())
        )
        if len(transcript) > 10_000 or len(normalized) > 10_000:
            transcript = transcript[:10_000]
            normalized = normalized[:10_000]
            quality_accepted = False
            quality_reasons = tuple(dict.fromkeys(
                quality_reasons + ("transcript_character_limit",)
            ))
        detected = tuple(dict.fromkeys(result.get("detected_words") or ()))[:50]
        hard_hits = tuple(dict.fromkeys(result.get("hard_hits") or ()))[:50]
        soft_hits = tuple(dict.fromkeys(result.get("soft_hits") or ()))[:50]
        laughter = bool(result.get("is_casual"))
        direct_target = has_direct_target_evidence(transcript)

        repeated = False
        if quality_accepted and detected:
            cutoff = mono - self._repetition_window
            with self._lock:
                for term in detected:
                    history = self._history.get(term)
                    if history is None:
                        if len(self._history) >= self._history_size:
                            self._history.popitem(last=False)
                        history = deque(maxlen=self._history_size)
                        self._history[term] = history
                    else:
                        self._history.move_to_end(term)
                    while history and history[0] < cutoff:
                        history.popleft()
                    if history:
                        repeated = True
                    history.append(mono)

        severity = calculate_severity(
            detected,
            transcript=transcript,
            repeated=repeated,
            laughter_present=laughter,
        )
        context_suppressed = bool(result.get("context_suppressed_all"))
        has_monitored = bool(result.get("has_profanity")) and bool(detected)
        limited_laughter = laughter and severity.level != HIGH
        qualifies = bool(
            quality_accepted
            and not context_suppressed
            and has_monitored
            and not limited_laughter
            and (
                severity.level == HIGH
                or bool(hard_hits)
                or repeated
                or direct_target
                or len(soft_hits) >= 2
            )
        )
        reasons = list(severity.reasons)
        reasons.extend(severity.supporting_evidence)
        if not quality_accepted:
            reasons.append("suppressed:transcript_quality")
        if context_suppressed:
            reasons.append("suppressed:harmless_or_ambiguous_context")
        if limited_laughter:
            reasons.append("suppressed:laughter_with_limited_text_evidence")
        if has_monitored and not qualifies and not limited_laughter:
            reasons.append("insufficient_text_context")

        matched = []
        for item in result.get("matched_terms") or ():
            if not isinstance(item, dict):
                continue
            term = str(item.get("term") or item.get("matched_text") or "").strip()
            if not term:
                continue
            term_id = item.get("term_id")
            matched.append(MatchedTermEvidence(
                term=term,
                language=(str(item.get("language")).lower() if item.get("language") else None),
                match_type=str(item.get("match_type") or "exact"),
                term_id=(term_id if isinstance(term_id, int) and not isinstance(term_id, bool) else None),
            ))
            if len(matched) >= 50:
                break

        evidence = TextEvidence(
            observed_at=wall,
            observed_monotonic=mono,
            transcript=transcript,
            normalized_transcript=normalized,
            language=str(result.get("language") or "unknown").lower(),
            language_confidence=result.get("language_confidence"),
            matched_terms=tuple(matched),
            detected_terms=detected,
            hard_hits=hard_hits,
            soft_hits=soft_hits,
            categories=tuple(dict.fromkeys(result.get("categories") or ()))[:50],
            direct_target=direct_target,
            repeated=repeated,
            quality_accepted=quality_accepted,
            quality_reasons=quality_reasons,
            quality_warnings=quality_warnings,
            baseline_severity=severity.level,
            reasons=tuple(dict.fromkeys(reasons)),
            laughter_present=laughter,
            qualifies_to_open=qualifies,
            spoken_started_at=spoken_started_at,
        )
        self.evidence_count += 1
        print(
            f"[TEXT_EVIDENCE] count={self.evidence_count} "
            f"qualifies={str(qualifies).lower()} severity={severity.level} "
            f"terms={len(detected)}"
        )
        return evidence


class TextDetectorWorker:
    """Consume the bounded STT FIFO and submit evidence off the main thread."""

    def __init__(self, detector, result_consumer, evidence_consumer):
        self.detector = detector
        self.result_consumer = result_consumer
        self.evidence_consumer = evidence_consumer
        self._stop_event = threading.Event()
        self._thread = None
        self.error = None

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self.error = None
        self._thread = threading.Thread(
            target=self._run,
            name="text-detector",
        )
        self._thread.start()

    def _run(self):
        try:
            while not self._stop_event.is_set():
                result = self.result_consumer(timeout=0.2)
                if result is None:
                    continue
                evidence = self.detector.analyze_stt_result(result)
                self.evidence_consumer(evidence)
        except Exception as exc:
            self.error = exc
            self._stop_event.set()
            print(f"[TEXT] worker_error={type(exc).__name__}")

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._thread = None
