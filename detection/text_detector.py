"""Independent WHAT-was-said detector producing structured text evidence."""

from __future__ import annotations

from collections import OrderedDict, deque
from datetime import datetime, timezone
import threading
import time

from dataclasses import dataclass

from detection.evidence import (
    LLM_HARMFUL_LABELS,
    DetectedTermProvenance,
    LLMJudgement,
    MatchedTermEvidence,
    TextEvidence,
    TranscriptCorrectionEvidence,
)
from detection.latency import LATENCY
from detection.llm_semantic import (
    meaning_only_severity,
    RecentTurns,
    RepetitionTracker,
    insult_families,
    meaning_family,
    SemanticGate,
    has_game_story_words,
    has_victim_reaction,
    semantic_input,
)
from detection.semantic import SemanticAnalyzer
from detection.severity import calculate_severity, has_direct_target_evidence
from detection.soundalike import FUZZY_PROVENANCE
from detection.transcript_quality import assess_transcript_quality, phrase_matches

# Reason recorded when only similarity-window (fuzzy) matches opened the
# utterance: severity is held at LOW for human review.
FUZZY_REVIEW_REASON = "fuzzy_match:capped_low_needs_review"
from model.blacklist import check_transcript

DETECTED_BY_MEANING_REASON = "semantic:detected_by_meaning"


@dataclass(slots=True)
class PendingText:
    """Text evidence fields computed before the semantic call returns."""

    fields: dict
    reasons: list
    handle: object = None


class TextDetector:
    """Analyze finalized transcripts without invoking audio inference or I/O."""

    def __init__(
        self,
        *,
        monotonic_clock=time.monotonic,
        wall_clock=lambda: datetime.now(timezone.utc),
        repetition_window_seconds=60.0,
        history_size=128,
        semantic_analyzer=None,
        semantic_gate=None,
    ):
        if repetition_window_seconds < 0 or history_size <= 0:
            raise ValueError("text history bounds are invalid")
        self._monotonic_clock = monotonic_clock
        self._wall_clock = wall_clock
        self._repetition_window = float(repetition_window_seconds)
        self._history_size = int(history_size)
        self._history = OrderedDict()
        self._lock = threading.Lock()
        self.semantic_analyzer = semantic_analyzer or SemanticAnalyzer()
        self.semantic_gate = semantic_gate or SemanticGate(mode="off")
        self._recent_turns = RecentTurns()
        self._repetition = RepetitionTracker()
        self.evidence_count = 0

    def clear_context(self):
        """Drop term and conversational histories retained in RAM."""

        with self._lock:
            self._history.clear()
        self.semantic_analyzer.clear()
        self._recent_turns.clear()
        self._repetition.clear()

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
        result = check_transcript(quality.checked_text) if quality.accepted else {
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

        return self.complete(self.prepare(
            result,
            observed_at=observed_at,
            observed_monotonic=observed_monotonic,
        ))

    def prepare(self, result: dict, *, observed_at=None, observed_monotonic=None):
        """Lexicon work plus starting the semantic call (does not wait for it)."""

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
        semantic = self.semantic_analyzer.assess(
            transcript,
            language=str(result.get("language") or "unknown").lower(),
            observed_monotonic=mono,
            quality_accepted=quality_accepted,
        )
        detected = tuple(dict.fromkeys(result.get("detected_words") or ()))[:50]
        hard_hits = tuple(dict.fromkeys(result.get("hard_hits") or ()))[:50]
        soft_hits = tuple(dict.fromkeys(result.get("soft_hits") or ()))[:50]
        laughter = bool(result.get("is_casual"))
        # Corrected text (mishearings, sound-alikes) decides severity and
        # targeting; the original transcript stays the evidence of record.
        match_text = str(result.get("match_text") or "")[:10_000]
        decision_text = match_text or transcript
        provenance_by_term = {}
        for item in result.get("term_provenance") or ():
            if not isinstance(item, dict):
                continue
            term = str(item.get("term") or "").strip()
            provenance = str(item.get("provenance") or "").strip().lower()
            if term and provenance:
                provenance_by_term[term] = provenance
        for term in detected:
            if term not in provenance_by_term:
                provenance_by_term[term] = (
                    "original_exact"
                    if phrase_matches(term, transcript)
                    else "unsupported_correction"
                )

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

        # ``repeated`` (same term within the history window) stays as
        # evidence metadata only. Severity repetition is judged within one
        # utterance here and across the incident in fusion.
        qualification_terms = []
        for term in detected:
            provenance = provenance_by_term[term]
            if provenance in {"original_exact", "trusted_alias"}:
                qualification_terms.append(term)
                continue
            if provenance == "explicit_mishearing" and (
                has_direct_target_evidence(decision_text, (term,))
                or calculate_severity((term,), transcript=decision_text).level == "HIGH"
            ):
                qualification_terms.append(term)
                continue
            if provenance == "fuzzy_soundalike" and repeated:
                qualification_terms.append(term)
                continue
            # Similarity-window matches may open, capped at LOW below.
            if provenance == FUZZY_PROVENANCE:
                qualification_terms.append(term)
        qualification_terms = tuple(dict.fromkeys(qualification_terms))
        direct_target = bool(
            qualification_terms
            and has_direct_target_evidence(decision_text, qualification_terms)
        )
        severity = calculate_severity(
            qualification_terms,
            transcript=decision_text,
            laughter_present=laughter,
        )
        fuzzy_only = bool(qualification_terms) and all(
            provenance_by_term[term] == FUZZY_PROVENANCE
            for term in qualification_terms
        )
        context_suppressed = bool(result.get("context_suppressed_all"))
        has_monitored = bool(result.get("has_profanity")) and bool(detected)
        has_qualifying_provenance = bool(qualification_terms)
        # Original words and trusted aliases may open. Curated explicit
        # mishearings require targeting/high-risk semantics; speculative fuzzy
        # matches require a separate repeated observation.
        qualifies = bool(
            quality_accepted
            and not context_suppressed
            and has_monitored
            and has_qualifying_provenance
        )
        reasons = list(severity.reasons)
        reasons.extend(severity.supporting_evidence)
        baseline_level = severity.level
        if fuzzy_only:
            baseline_level = "LOW"
            reasons.append(FUZZY_REVIEW_REASON)
        if not quality_accepted:
            reasons.append("suppressed:transcript_quality")
        if context_suppressed:
            reasons.append("suppressed:harmless_or_ambiguous_context")
        if has_monitored and not has_qualifying_provenance:
            reasons.append("suppressed:correction_requires_corroboration")

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

        term_provenance = tuple(
            DetectedTermProvenance(term=term, provenance=provenance_by_term[term])
            for term in detected
        )
        corrections = []
        for item in result.get("corrections") or ():
            if not isinstance(item, dict):
                continue
            heard = str(item.get("heard") or "").strip()
            term = str(item.get("term") or "").strip()
            if not heard or not term:
                continue
            corrections.append(TranscriptCorrectionEvidence(
                heard=heard,
                term=term,
                kind=str(item.get("kind") or "unknown"),
                provenance=str(
                    item.get("provenance") or "unsupported_correction"
                ),
            ))
            if len(corrections) >= 50:
                break

        tone_signals = (
            result.get("tone_signals")
            if isinstance(result.get("tone_signals"), dict) else None
        )
        audio_tags = tuple(
            str(tag)[:40] for tag in (result.get("audio_tags") or ())
        )[:20]
        language = str(result.get("language") or "unknown").lower()
        previous_turns = self._recent_turns.previous(mono)
        game_story_context = has_game_story_words(decision_text, *previous_turns)
        reaction_in_previous_turns = any(
            has_victim_reaction(turn) for turn in previous_turns
        )
        victim_reaction = has_victim_reaction(decision_text, audio_tags)
        handle = None
        if self.semantic_gate.wants(
            has_match=bool(detected), quality_accepted=quality_accepted,
        ) and transcript.strip():
            handle = self.semantic_gate.submit(semantic_input(
                decision_text,
                previous_turns=previous_turns,
                language=language,
                matched_terms=tuple(detected),
                tone_signals=tone_signals,
                audio_tags=audio_tags,
                laughter=laughter,
            ))
        if quality_accepted:
            self._recent_turns.remember(decision_text, mono)

        fields = dict(
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
            baseline_severity=baseline_level,
            laughter_present=laughter,
            qualifies_to_open=qualifies,
            spoken_started_at=spoken_started_at,
            match_text=match_text if match_text != transcript else "",
            term_provenance=term_provenance,
            corrections=tuple(corrections),
            qualification_terms=qualification_terms,
            semantic=semantic,
            tone_signals=tone_signals,
            audio_tags=audio_tags,
            game_story_context=game_story_context,
            reaction_in_previous_turns=reaction_in_previous_turns,
            victim_reaction=victim_reaction,
        )
        return PendingText(fields=fields, reasons=reasons, handle=handle)

    def complete(self, pending: PendingText) -> TextEvidence:
        """Wait for the semantic judgement (bounded) and build the evidence."""

        fields = pending.fields
        reasons = list(pending.reasons)
        judgement = None
        if pending.handle is not None:
            judgement = self.semantic_gate.resolve(pending.handle)
        elif self.semantic_gate.available:
            judgement = LLMJudgement(status="skipped")
        qualifies = fields["qualifies_to_open"]
        opened_by_meaning = bool(
            not qualifies
            and fields["quality_accepted"]
            and judgement is not None
            and judgement.ok
            and judgement.label in LLM_HARMFUL_LABELS
            and judgement.targeted
            and meaning_only_severity(judgement.label, judgement.confidence)
        )
        if opened_by_meaning:
            qualifies = True
            fields["baseline_severity"] = meaning_only_severity(
                judgement.label, judgement.confidence,
            )
            reasons.append(DETECTED_BY_MEANING_REASON)
        pattern_repeated = False
        if qualifies:
            # Same insult family only (lexicon root / canonical term); a
            # by-meaning alert repeats only with the same sentence.
            families = insult_families(fields["qualification_terms"])
            if not families and opened_by_meaning:
                families = meaning_family(
                    judgement.label,
                    fields.get("match_text") or fields.get("transcript"),
                )
            pattern_repeated = self._repetition.repeated(
                families, fields["observed_monotonic"],
            )
        if judgement is not None and judgement.status == "fallback":
            reasons.append(f"semantic_fallback:{judgement.fallback_reason[:60]}")
        evidence = TextEvidence(**{
            **fields,
            "qualifies_to_open": qualifies,
            "reasons": tuple(dict.fromkeys(reasons))[:100],
            "llm": judgement,
            "opened_by_meaning": opened_by_meaning,
            "pattern_repeated": pattern_repeated,
        })
        self.evidence_count += 1
        print(
            f"[TEXT_EVIDENCE] count={self.evidence_count} "
            f"qualifies={str(qualifies).lower()} "
            f"severity={evidence.baseline_severity} "
            f"terms={len(evidence.detected_terms)}"
            + (" by_meaning=true" if opened_by_meaning else "")
            + (" repeated_pattern=true" if pattern_repeated else "")
        )
        return evidence


def _record_latency(utterance_id, submission):
    if not utterance_id:
        return
    if submission is not None and (submission.opened or submission.joined):
        LATENCY.link_incident(submission.event_id, utterance_id)
    elif submission is not None and submission.duplicate_of:
        LATENCY.finish(utterance_id, f"duplicate_of:{submission.duplicate_of}")
    else:
        LATENCY.finish(utterance_id, "no_incident")


class TextDetectorWorker:
    """Consume the bounded STT FIFO and submit evidence off the main thread.

    With the semantic classifier on, one thread prepares utterances and starts
    their Claude calls while a second waits for each judgement in arrival
    order, so several calls overlap and evidence order is preserved.
    """

    def __init__(self, detector, result_consumer, evidence_consumer, *, max_in_flight=4):
        self.detector = detector
        self.result_consumer = result_consumer
        self.evidence_consumer = evidence_consumer
        self._stop_event = threading.Event()
        self._threads = []
        self._pending = None
        self._max_in_flight = max(1, int(max_in_flight))
        self.error = None

    @property
    def _thread(self):
        return self._threads[0] if self._threads else None

    def start(self):
        if any(thread.is_alive() for thread in self._threads):
            return
        self._stop_event.clear()
        self.error = None
        pipelined = bool(getattr(
            getattr(self.detector, "semantic_gate", None), "available", False,
        ))
        if pipelined:
            import queue

            self._pending = queue.Queue(maxsize=self._max_in_flight)
            self._threads = [
                threading.Thread(target=self._run_prepare, name="text-detector"),
                threading.Thread(target=self._run_complete, name="text-semantic"),
            ]
        else:
            self._pending = None
            self._threads = [threading.Thread(target=self._run, name="text-detector")]
        for thread in self._threads:
            thread.start()

    def _fail(self, exc):
        self.error = exc
        self._stop_event.set()
        print(f"[TEXT] worker_error={type(exc).__name__}")

    def _run(self):
        try:
            while not self._stop_event.is_set():
                result = self.result_consumer(timeout=0.2)
                if result is None:
                    continue
                evidence = self.detector.analyze_stt_result(result)
                submission = self.evidence_consumer(evidence)
                _record_latency(result.get("event_id"), submission)
        except Exception as exc:
            self._fail(exc)

    def _run_prepare(self):
        import queue

        try:
            while not self._stop_event.is_set():
                result = self.result_consumer(timeout=0.2)
                if result is None:
                    continue
                pending = self.detector.prepare(result)
                while not self._stop_event.is_set():
                    try:
                        self._pending.put((result, pending), timeout=0.2)
                        break
                    except queue.Full:
                        continue
        except Exception as exc:
            self._fail(exc)

    def _run_complete(self):
        import queue

        try:
            while not self._stop_event.is_set():
                try:
                    result, pending = self._pending.get(timeout=0.2)
                except queue.Empty:
                    continue
                evidence = self.detector.complete(pending)
                submission = self.evidence_consumer(evidence)
                _record_latency(result.get("event_id"), submission)
        except Exception as exc:
            self._fail(exc)

    def stop(self):
        self._stop_event.set()
        for thread in self._threads:
            thread.join(timeout=5.0)
        self._threads = []
        self.detector.clear_context()
