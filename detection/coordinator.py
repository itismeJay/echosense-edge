"""Thread-safe single-incident coordinator for independent detector evidence."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
import queue
import threading
import time
from uuid import uuid4

from audio.shared_source import AudioFrame, RollingAudioBuffer
from audio.waveform import get_waveform_snapshot_float32
from config import DetectionRuntimeConfig
from detection.evidence import AcousticEvidence, FinalizedIncident, TextEvidence
from detection.fusion import (
    fuse_event_evidence,
    is_time_linked,
    linked_acoustic,
    strong_acoustic,
)
from detection.severity import SEVERITY_ORDER, contains_urgent_directive


@dataclass(slots=True)
class ActiveIncident:
    event_id: str
    trigger_type: str
    triggered_at: datetime
    trigger_monotonic: float
    close_deadline: float
    hard_deadline: float
    text_evidence: deque
    acoustic_evidence: deque
    audio_context: RollingAudioBuffer
    opener: str
    evidence_overflow_count: int = 0
    non_qualifying_text_count: int = 0
    last_audio_sequence: int | None = None


@dataclass(frozen=True, slots=True)
class ClosedTextIncident:
    event_id: str
    closed_monotonic: float
    terms: frozenset
    severity: str
    duplicates: list = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class CoordinatorSubmission:
    event_id: str | None
    opened: bool = False
    joined: bool = False
    # Event id of the closed incident this text was logged against as a
    # duplicate (same terms, same-or-lower severity, inside the window).
    duplicate_of: str | None = None
    finalized: tuple[FinalizedIncident, ...] = ()


class EventCoordinator:
    """Own zero or one incident and finalize exactly once into one callback."""

    def __init__(
        self,
        *,
        config: DetectionRuntimeConfig | None = None,
        pre_trigger_buffer: RollingAudioBuffer | None = None,
        finalizer=None,
        monotonic_clock=time.monotonic,
        wall_clock=lambda: datetime.now(timezone.utc),
        event_id_factory=lambda: str(uuid4()),
    ):
        self.config = config or DetectionRuntimeConfig()
        self.pre_trigger_buffer = pre_trigger_buffer
        self.finalizer = finalizer
        self._monotonic_clock = monotonic_clock
        self._wall_clock = wall_clock
        self._event_id_factory = event_id_factory
        self._lock = threading.RLock()
        self._active = None
        # Most recently closed incident that carried text evidence. Only the
        # duplicate guard reads it; nothing is ever cooldown-suppressed.
        self._last_text_incident = None
        # Recent acoustic windows, so a transcript that arrives after Whisper
        # finishes can still be linked to the sound of the moment it was said.
        self._recent_acoustic = deque(maxlen=self.config.evidence_history_size)
        self._finalized_count = 0
        self._duplicate_count = 0
        self._closed = False

    @property
    def active_incident(self):
        with self._lock:
            return self._active

    @property
    def finalized_count(self):
        with self._lock:
            return self._finalized_count

    @property
    def duplicate_count(self):
        with self._lock:
            return self._duplicate_count

    @property
    def last_text_incident(self):
        with self._lock:
            return self._last_text_incident

    def _duplicate_of(self, evidence, evidence_time):
        """Return the closed text incident this text merely repeats, if any.

        Duplicate only when every term was already in that incident, the
        severity is not higher, and it is inside the duplicate window.
        Acoustic evidence and acoustic-only incidents never take part.
        """

        window = self.config.duplicate_window_seconds
        last = self._last_text_incident
        if window <= 0 or last is None or not isinstance(evidence, TextEvidence):
            return None
        terms = frozenset(evidence.detected_terms)
        if not terms or not terms <= last.terms:
            return None
        if evidence_time - last.closed_monotonic >= window:
            return None
        # Judge severity as this text would be fused on its own, so a repeat
        # that is now shouted (acoustically promoted) is a new incident. The
        # repetition promotion is ignored: repeating the incident's own terms
        # is exactly what the duplicate guard absorbs.
        candidate = replace(evidence, repeated=False)
        severity = fuse_event_evidence(
            (candidate,),
            tuple(linked_acoustic(
                (candidate,),
                tuple(self._recent_acoustic),
                self.config.corroboration_window_seconds,
            )),
            duration_seconds=0.0,
            config=self.config,
        ).level
        if SEVERITY_ORDER[severity] > SEVERITY_ORDER[last.severity]:
            return None
        return last

    def _new_active(self, trigger_type, wall_time, mono_time):
        event_id = self._event_id_factory()
        hard_deadline = mono_time + self.config.max_event_seconds
        close_deadline = min(
            hard_deadline,
            mono_time + self.config.post_trigger_seconds,
        )
        audio_context = RollingAudioBuffer(
            self.config.pre_trigger_seconds + self.config.max_event_seconds,
        )
        if self.pre_trigger_buffer is not None:
            audio_context.seed(
                self.pre_trigger_buffer.snapshot(
                    self.config.pre_trigger_seconds
                )
            )
        active = ActiveIncident(
            event_id=event_id,
            trigger_type=trigger_type,
            triggered_at=wall_time.astimezone(timezone.utc),
            trigger_monotonic=mono_time,
            close_deadline=close_deadline,
            hard_deadline=hard_deadline,
            text_evidence=deque(maxlen=self.config.evidence_history_size),
            acoustic_evidence=deque(maxlen=self.config.evidence_history_size),
            audio_context=audio_context,
            opener="TEXT" if trigger_type == "KEYWORD" else "ACOUSTIC",
        )
        self._active = active
        print(
            f"[INCIDENT] opened event={event_id} opener={active.opener} "
            f"close_in={close_deadline - mono_time:.3f}s"
        )
        return active

    def _extend(self, active, evidence_time):
        previous = active.close_deadline
        active.close_deadline = min(
            active.hard_deadline,
            max(
                active.close_deadline,
                evidence_time + self.config.event_extension_seconds,
            ),
        )
        if active.close_deadline > previous:
            print(
                f"[INCIDENT] extended event={active.event_id} "
                f"deadline={active.close_deadline:.6f}"
            )

    @staticmethod
    def _text_identity(evidence):
        return (
            evidence.observed_monotonic,
            evidence.transcript,
            evidence.detected_terms,
        )

    @staticmethod
    def _acoustic_identity(evidence):
        return (
            evidence.monotonic_start,
            evidence.monotonic_end,
            evidence.yamnet_label,
            evidence.yamnet_score,
            evidence.tone_aggressive,
        )

    def _append_bounded(self, active, collection, evidence, identity):
        if any(identity(existing) == identity(evidence) for existing in collection):
            return False
        full = len(collection) == collection.maxlen
        collection.append(evidence)
        if full:
            active.evidence_overflow_count += 1
            print(
                f"[INCIDENT] event={active.event_id} "
                f"evidence_overflow_count={active.evidence_overflow_count} "
                "policy=drop_oldest"
            )
        return True

    def _submit(self, evidence, trigger_type, evidence_time, wall_time):
        dispatch = []
        with self._lock:
            if self._closed:
                raise RuntimeError("coordinator is shut down")
            effective_now = max(self._monotonic_clock(), evidence_time)
            dispatch.extend(self._finalize_due_locked(effective_now))
            active = self._active
            qualifying = evidence.qualifies_to_open
            if active is None:
                duplicate_of = (
                    self._duplicate_of(evidence, evidence_time)
                    if qualifying
                    else None
                )
                if not qualifying:
                    result = CoordinatorSubmission(None, finalized=tuple(dispatch))
                elif duplicate_of is not None:
                    self._duplicate_count += 1
                    age = evidence_time - duplicate_of.closed_monotonic
                    if len(duplicate_of.duplicates) < self.config.evidence_history_size:
                        duplicate_of.duplicates.append(evidence)
                    print(
                        f"[INCIDENT] duplicate_of={duplicate_of.event_id} "
                        f"terms={','.join(sorted(evidence.detected_terms))} "
                        f"severity={evidence.baseline_severity} "
                        f"seconds_after_close={age:.1f} "
                        f"duplicates={len(duplicate_of.duplicates)} alert=false"
                    )
                    result = CoordinatorSubmission(
                        None,
                        duplicate_of=duplicate_of.event_id,
                        finalized=tuple(dispatch),
                    )
                else:
                    active = self._new_active(trigger_type, wall_time, evidence_time)
                    self._attach(active, evidence)
                    self._after_attach(active, evidence, evidence_time)
                    dispatch.extend(
                        self._finalize_due_locked(effective_now)
                    )
                    result = CoordinatorSubmission(
                        active.event_id,
                        opened=True,
                        finalized=tuple(dispatch),
                    )
            else:
                # Exact timestamp ties prefer text. This is the only case where
                # opener normalization occurs; later evidence never rewrites it.
                if (
                    evidence_time == active.trigger_monotonic
                    and trigger_type == "KEYWORD"
                    and active.trigger_type == "ACOUSTIC"
                ):
                    active.trigger_type = "KEYWORD"
                    active.opener = "TEXT"
                joined = self._attach(active, evidence)
                if joined and self._extends_incident(active, evidence):
                    self._extend(active, evidence_time)
                self._after_attach(active, evidence, evidence_time)
                dispatch.extend(
                    self._finalize_due_locked(effective_now)
                )
                result = CoordinatorSubmission(
                    active.event_id,
                    joined=joined,
                    finalized=tuple(dispatch),
                )
        self._dispatch(dispatch)
        return result

    def _extends_incident(self, active, evidence):
        """Only text, or strong acoustic time-linked to that text, extends.

        Ordinary acoustic windows are recorded but never keep an incident
        open; a loud classroom would otherwise hold every incident open until
        max_event_seconds.
        """

        if not evidence.qualifies_to_open:
            return False
        if isinstance(evidence, TextEvidence):
            return True
        window = self.config.corroboration_window_seconds
        texts = tuple(active.text_evidence)
        if not any(is_time_linked(text, evidence, window) for text in texts):
            return False
        return strong_acoustic(
            linked_acoustic(texts, tuple(active.acoustic_evidence), window),
            self.config,
        )

    def _after_attach(self, active, evidence, evidence_time):
        if isinstance(evidence, AcousticEvidence):
            return
        if not evidence.qualifies_to_open:
            return
        window = self.config.corroboration_window_seconds
        for recent in tuple(self._recent_acoustic):
            if is_time_linked(evidence, recent, window):
                self._append_bounded(
                    active,
                    active.acoustic_evidence,
                    recent,
                    self._acoustic_identity,
                )
        if contains_urgent_directive(evidence.detected_terms):
            # Threat-like and self-harm phrases are sent without waiting for
            # the incident window to close.
            active.close_deadline = min(active.close_deadline, evidence_time)
            print(
                f"[INCIDENT] urgent_term=true event={active.event_id} "
                "finalize=immediate"
            )

    def _attach(self, active, evidence):
        if isinstance(evidence, TextEvidence):
            if not evidence.qualifies_to_open:
                # Normal classroom words are not incident content. Retain only
                # a bounded activity count; never retain their transcript.
                active.non_qualifying_text_count = min(
                    active.non_qualifying_text_count + 1,
                    self.config.evidence_history_size,
                )
                return False
            return self._append_bounded(
                active,
                active.text_evidence,
                evidence,
                self._text_identity,
            )
        return self._append_bounded(
            active,
            active.acoustic_evidence,
            evidence,
            self._acoustic_identity,
        )

    def submit_text(self, evidence: TextEvidence):
        if not isinstance(evidence, TextEvidence):
            raise TypeError("submit_text requires TextEvidence")
        return self._submit(
            evidence,
            "KEYWORD",
            evidence.observed_monotonic,
            evidence.observed_at,
        )

    def submit_acoustic(self, evidence: AcousticEvidence):
        if not isinstance(evidence, AcousticEvidence):
            raise TypeError("submit_acoustic requires AcousticEvidence")
        with self._lock:
            self._recent_acoustic.append(evidence)
        return self._submit(
            evidence,
            "ACOUSTIC",
            evidence.monotonic_end,
            evidence.interval_ended_at,
        )

    def observe_audio(self, frame: AudioFrame):
        with self._lock:
            active = self._active
            if active is None or frame.monotonic_end < active.trigger_monotonic:
                return False
            if active.last_audio_sequence == frame.sequence:
                return False
            active.audio_context.append(frame)
            active.last_audio_sequence = frame.sequence
            return True

    def _finalize_due_locked(self, now):
        active = self._active
        if active is None or now < min(active.close_deadline, active.hard_deadline):
            return []
        deadline = min(active.close_deadline, active.hard_deadline)
        finalized = self._finalize_locked(deadline)
        return [finalized] if finalized is not None else []

    def _finalize_locked(self, final_monotonic):
        active = self._active
        if active is None:
            raise RuntimeError("there is no active incident")
        duration = max(0.0, final_monotonic - active.trigger_monotonic)
        fusion = fuse_event_evidence(
            tuple(active.text_evidence),
            tuple(active.acoustic_evidence),
            duration_seconds=duration,
            config=self.config,
        )
        if not fusion.deliver:
            # Policy-suppressed (e.g. ECHOSENSE_ACOUSTIC_ONLY_ALERTS=off). No
            # alert is sent and nothing is recorded for the duplicate guard.
            active.audio_context.clear()
            self._active = None
            print(
                f"[INCIDENT] suppressed event={active.event_id} "
                f"reasons={','.join(fusion.reasons)}"
            )
            return None
        pcm = active.audio_context.snapshot()
        waveform = (
            tuple(get_waveform_snapshot_float32(
                pcm,
                num_points=self.config.waveform_points,
            ))
            if pcm.size
            else ()
        )
        pcm = None
        active.audio_context.clear()
        finalized = FinalizedIncident(
            event_id=active.event_id,
            trigger_type=active.trigger_type,
            triggered_at=active.triggered_at,
            finalized_at=active.triggered_at + timedelta(seconds=duration),
            duration_seconds=duration,
            text_evidence=tuple(active.text_evidence),
            acoustic_evidence=tuple(active.acoustic_evidence),
            severity=fusion.level,
            severity_reasons=fusion.reasons,
            term_categories=fusion.term_categories,
            supporting_evidence=fusion.supporting_evidence,
            confidence=fusion.confidence,
            waveform_snapshot=waveform,
        )
        self._active = None
        if finalized.text_evidence:
            self._last_text_incident = ClosedTextIncident(
                event_id=finalized.event_id,
                closed_monotonic=final_monotonic,
                terms=frozenset(
                    term
                    for item in finalized.text_evidence
                    for term in item.detected_terms
                ),
                severity=finalized.severity,
            )
        self._finalized_count += 1
        print(
            f"[INCIDENT] finalized event={finalized.event_id} "
            f"severity={finalized.severity} text={len(finalized.text_evidence)} "
            f"acoustic={len(finalized.acoustic_evidence)}"
        )
        return finalized

    def _dispatch(self, finalized):
        if self.finalizer is None:
            return
        for incident in finalized:
            try:
                succeeded = self.finalizer(incident)
            except Exception as exc:
                print(
                    f"[OUTBOX] enqueue failed event={incident.event_id} "
                    f"error={type(exc).__name__}"
                )
                continue
            print(
                f"[OUTBOX] enqueue result event={incident.event_id} "
                f"queued={str(bool(succeeded)).lower()}"
            )

    def tick(self, now=None):
        with self._lock:
            finalized = self._finalize_due_locked(
                self._monotonic_clock() if now is None else float(now)
            )
        self._dispatch(finalized)
        return tuple(finalized)

    def force_finalize(self, now=None):
        with self._lock:
            if self._active is None:
                return ()
            current = self._monotonic_clock() if now is None else float(now)
            final_time = min(max(current, self._active.trigger_monotonic), self._active.hard_deadline)
            finalized = tuple(
                item
                for item in (self._finalize_locked(final_time),)
                if item is not None
            )
        self._dispatch(finalized)
        return finalized

    def shutdown(self, *, finalize_active=False):
        finalized = self.force_finalize() if finalize_active else ()
        with self._lock:
            if self._active is not None:
                self._active.audio_context.clear()
                self._active = None
            self._closed = True
        return finalized


class CoordinatorAudioWorker:
    """Move bounded source frames into active RAM context off capture thread."""

    def __init__(self, coordinator, frame_queue):
        self.coordinator = coordinator
        self.frame_queue = frame_queue
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
            name="coordinator-audio-context",
        )
        self._thread.start()

    def _run(self):
        try:
            while not self._stop_event.is_set():
                try:
                    frame = self.frame_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                self.coordinator.observe_audio(frame)
        except Exception as exc:
            self.error = exc
            self._stop_event.set()
            print(f"[INCIDENT] audio_worker_error={type(exc).__name__}")

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._thread = None
