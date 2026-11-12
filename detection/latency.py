"""Per-utterance pipeline latency, logged as one [LATENCY] line in ms.

Stages, in order:
  speech_end          RealtimeSTT VAD stopped the recording
  whisper_done        final Whisper decode (incl. any language retry) returned
  transcript_emitted  STT result queued for the text detector
  incident_enqueued   the incident holding this text was finalized + queued
  delivered           the outbox received a 2xx for that incident

Noise reduction runs upstream of the VAD, so it is reported separately as the
block hold (audio buffered before Whisper/VAD sees it) plus compute per block.
"""

from __future__ import annotations

from collections import OrderedDict
import threading
import time

STAGES = (
    "speech_end",
    "whisper_done",
    "transcript_emitted",
    "incident_enqueued",
    "delivered",
)
_DELTAS = (
    ("whisper_ms", "speech_end", "whisper_done"),
    ("emit_ms", "whisper_done", "transcript_emitted"),
    ("incident_window_ms", "transcript_emitted", "incident_enqueued"),
    ("delivery_ms", "incident_enqueued", "delivered"),
)


class LatencyTracker:
    def __init__(self, *, clock=time.monotonic, max_entries=64, logger=print):
        self._clock = clock
        self._max_entries = int(max_entries)
        self._logger = logger
        self._lock = threading.Lock()
        self._utterances = OrderedDict()
        self._incidents = OrderedDict()
        self._pending_speech_end = None
        self._nr_hold_ms = None
        self._nr_compute_ms = None

    @staticmethod
    def _bounded_put(mapping, key, value, limit):
        mapping[key] = value
        mapping.move_to_end(key)
        while len(mapping) > limit:
            mapping.popitem(last=False)

    def set_noise_reduction(self, *, hold_ms, compute_ms):
        with self._lock:
            self._nr_hold_ms = float(hold_ms)
            self._nr_compute_ms = float(compute_ms)

    def speech_end(self, at=None):
        with self._lock:
            self._pending_speech_end = self._clock() if at is None else at

    def begin(self, utterance_id, **extra):
        """Open an utterance at the last speech_end (or now if unknown)."""

        if not utterance_id:
            return
        with self._lock:
            start = self._pending_speech_end
            self._pending_speech_end = None
            entry = {
                "stages": {"speech_end": self._clock() if start is None else start},
                "extra": dict(extra),
                "incident": None,
            }
            self._bounded_put(
                self._utterances, utterance_id, entry, self._max_entries
            )

    def mark(self, utterance_id, stage, **extra):
        with self._lock:
            entry = self._utterances.get(utterance_id)
            if entry is None:
                return
            entry["stages"].setdefault(stage, self._clock())
            entry["extra"].update(extra)

    def link_incident(self, incident_id, utterance_id):
        """Attach an utterance to its incident; replay stages already seen."""

        lines = []
        with self._lock:
            entry = self._utterances.get(utterance_id)
            if entry is None or not incident_id:
                return
            entry["incident"] = incident_id
            incident = self._incidents.setdefault(
                incident_id, {"stages": {}, "utterances": []}
            )
            incident["utterances"].append(utterance_id)
            for stage, at in incident["stages"].items():
                entry["stages"].setdefault(stage, at)
            if "delivered" in entry["stages"]:
                lines.append(self._pop_line_locked(utterance_id, "delivered"))
        for line in lines:
            self._logger(line)

    def mark_incident(self, incident_id, stage):
        lines = []
        with self._lock:
            incident = self._incidents.get(incident_id)
            if incident is None:
                incident = {"stages": {}, "utterances": []}
                self._bounded_put(
                    self._incidents, incident_id, incident, self._max_entries
                )
            now = self._clock()
            incident["stages"].setdefault(stage, now)
            for utterance_id in tuple(incident["utterances"]):
                entry = self._utterances.get(utterance_id)
                if entry is None:
                    continue
                entry["stages"].setdefault(stage, now)
                if stage == "incident_enqueued":
                    lines.append(self._line_locked(utterance_id, "enqueued"))
                elif stage == "delivered":
                    lines.append(self._pop_line_locked(utterance_id, "delivered"))
            if stage == "delivered":
                self._incidents.pop(incident_id, None)
        for line in lines:
            self._logger(line)

    def finish(self, utterance_id, outcome):
        """Log and drop an utterance that will not reach an incident."""

        with self._lock:
            if utterance_id not in self._utterances:
                return
            line = self._pop_line_locked(utterance_id, outcome)
        self._logger(line)

    def _pop_line_locked(self, utterance_id, outcome):
        line = self._line_locked(utterance_id, outcome)
        self._utterances.pop(utterance_id, None)
        return line

    def _line_locked(self, utterance_id, outcome):
        entry = self._utterances[utterance_id]
        stages = entry["stages"]
        parts = [f"[LATENCY] utterance={utterance_id}"]
        if entry["incident"]:
            parts.append(f"incident={entry['incident']}")
        parts.append(f"outcome={outcome}")
        if self._nr_hold_ms is not None:
            parts.append(f"nr_hold_ms={self._nr_hold_ms:.0f}")
            parts.append(f"nr_compute_ms={self._nr_compute_ms:.1f}")
        for name, start, end in _DELTAS:
            if start in stages and end in stages:
                parts.append(f"{name}={(stages[end] - stages[start]) * 1000:.0f}")
        reached = [stage for stage in STAGES if stage in stages]
        if len(reached) > 1:
            total = stages[reached[-1]] - stages["speech_end"]
            parts.append(f"speech_end_to_{reached[-1]}_ms={total * 1000:.0f}")
        for key, value in entry["extra"].items():
            parts.append(f"{key}={value}")
        return " ".join(parts)


LATENCY = LatencyTracker()
