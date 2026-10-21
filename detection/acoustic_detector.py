"""Independent continuous HOW-it-sounded detector and bounded worker."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone
import math
import queue
import threading

import numpy as np

from audio.shared_source import AudioFrame, BoundedFrameQueue, CANONICAL_SAMPLE_RATE
from detection.evidence import AcousticEvidence
from detection.thresholds import TONE_MIN_ABSOLUTE_RMS, YAMNET_THRESHOLD
from model.tone_analyzer import analyze_tone_float32
from model.yamnet_infer import is_aggressive_sound, run_yamnet_float32


class NoiseFloorTracker:
    """Rolling, robust estimate of how loud the room normally is.

    The floor is a mid percentile of recent window RMS values, so brief shouts
    barely move it while a room that stays louder raises it. A window counts as
    "aggressive tone" only when its RMS reaches the adaptive threshold
    ``max(minimum_threshold, floor * multiplier)`` (and passes the existing
    variance/ZCR checks). ``minimum_threshold`` is an absolute RMS floor so
    normal speech in a quiet room never counts as aggressive. Until the startup
    calibration period has elapsed no window can be tone-aggressive.
    """

    def __init__(
        self,
        *,
        window_seconds=60.0,
        percentile=60.0,
        multiplier=2.5,
        minimum_threshold=TONE_MIN_ABSOLUTE_RMS,
        calibration_seconds=15.0,
        log_interval_seconds=30.0,
    ):
        self.window_seconds = float(window_seconds)
        self.percentile = float(percentile)
        self.multiplier = float(multiplier)
        self.minimum_threshold = float(minimum_threshold)
        self.calibration_seconds = float(calibration_seconds)
        self.log_interval_seconds = float(log_interval_seconds)
        self._observations = deque()
        self._started_at = None
        self._last_log_at = None

    @classmethod
    def from_config(cls, config):
        return cls(
            window_seconds=config.noise_floor_window_seconds,
            percentile=config.noise_floor_percentile,
            multiplier=config.tone_floor_multiplier,
            minimum_threshold=config.tone_min_absolute_rms,
            calibration_seconds=config.calibration_seconds,
            log_interval_seconds=config.noise_floor_log_interval_seconds,
        )

    def is_calibrated(self, now):
        if self._started_at is None:
            return self.calibration_seconds <= 0
        return now - self._started_at >= self.calibration_seconds

    @property
    def floor(self):
        if not self._observations:
            return None
        return float(np.percentile(
            [rms for _end, rms in self._observations],
            self.percentile,
        ))

    @property
    def threshold(self):
        floor = self.floor
        if floor is None:
            return self.minimum_threshold
        return max(self.minimum_threshold, floor * self.multiplier)

    def observe(self, rms, *, monotonic_start, monotonic_end):
        if self._started_at is None:
            self._started_at = float(monotonic_start)
        self._observations.append((float(monotonic_end), float(rms)))
        cutoff = monotonic_end - self.window_seconds
        while self._observations and self._observations[0][0] < cutoff:
            self._observations.popleft()
        if (
            self._last_log_at is None
            or monotonic_end - self._last_log_at >= self.log_interval_seconds
        ):
            self._last_log_at = float(monotonic_end)
            floor = self.floor
            print(
                f"[NOISE_FLOOR] floor_rms={floor:.1f} "
                f"floor_x{self.multiplier:g}={floor * self.multiplier:.1f} "
                f"absolute_min_rms={self.minimum_threshold:.1f} "
                f"tone_threshold_rms={self.threshold:.1f} "
                f"windows={len(self._observations)} "
                f"calibrated={str(self.is_calibrated(monotonic_end)).lower()}"
            )


class AcousticDetector:
    """Analyze canonical PCM windows regardless of transcript availability."""

    def __init__(
        self,
        *,
        interpreter=None,
        class_names=(),
        yamnet_runner=None,
        yamnet_threshold=YAMNET_THRESHOLD,
        unavailable_reason="model_not_loaded",
        noise_floor=None,
    ):
        self.noise_floor = noise_floor or NoiseFloorTracker()
        self.interpreter = interpreter
        self.class_names = tuple(class_names or ())
        self.yamnet_threshold = float(yamnet_threshold)
        if not math.isfinite(self.yamnet_threshold):
            raise ValueError("yamnet_threshold must be finite")
        if not 0.0 <= self.yamnet_threshold <= 1.0:
            raise ValueError("yamnet_threshold must be between zero and one")
        self._injected_runner = yamnet_runner
        self.yamnet_available = bool(
            yamnet_runner is not None
            or (interpreter is not None and self.class_names)
        )
        self.unavailable_reason = (
            "available" if self.yamnet_available else unavailable_reason
        )
        self.evidence_count = 0

    def _run_yamnet(self, samples):
        if self._injected_runner is not None:
            return self._injected_runner(samples)
        label, score, _scores = run_yamnet_float32(
            self.interpreter,
            samples,
            self.class_names,
        )
        return label, score

    def analyze_window(
        self,
        samples,
        *,
        interval_started_at,
        interval_ended_at,
        monotonic_start,
        monotonic_end,
    ):
        canonical = np.asarray(samples)
        if canonical.dtype != np.float32 or canonical.ndim != 1:
            raise TypeError("acoustic windows must be mono float32")
        if canonical.size == 0:
            raise ValueError("acoustic windows must not be empty")
        tone_threshold = self.noise_floor.threshold
        calibrating = not self.noise_floor.is_calibrated(float(monotonic_end))
        tone = analyze_tone_float32(
            canonical,
            CANONICAL_SAMPLE_RATE,
            rms_threshold=tone_threshold,
        )
        self.noise_floor.observe(
            tone["rms"],
            monotonic_start=float(monotonic_start),
            monotonic_end=float(monotonic_end),
        )
        yamnet_ran = False
        yamnet_label = "NotRun"
        yamnet_score = 0.0
        yamnet_aggressive = False
        reasons = []
        if self.yamnet_available:
            try:
                yamnet_label, yamnet_score = self._run_yamnet(canonical)
                yamnet_score = float(yamnet_score)
                if (
                    not math.isfinite(yamnet_score)
                    or not 0.0 <= yamnet_score <= 1.0
                ):
                    raise ValueError("YAMNet score must be finite and bounded")
                yamnet_ran = True
                yamnet_aggressive = is_aggressive_sound(
                    str(yamnet_label),
                    yamnet_score,
                    self.yamnet_threshold,
                )
                reasons.append(
                    f"yamnet:{yamnet_label}:{yamnet_score:.3f}"
                )
                if yamnet_aggressive:
                    reasons.append("qualifying:yamnet_aggressive_class")
            except Exception as exc:
                self.yamnet_available = False
                self.unavailable_reason = f"inference_failed:{type(exc).__name__}"
                yamnet_ran = False
                yamnet_label = "NotRun"
                yamnet_score = 0.0
                yamnet_aggressive = False
                reasons.append(f"yamnet_unavailable:{self.unavailable_reason}")
                print(
                    "[ACOUSTIC] degraded=true "
                    f"reason={self.unavailable_reason}"
                )
        else:
            reasons.append(f"yamnet_unavailable:{self.unavailable_reason}")

        reasons.append(f"tone_rms_threshold:{tone_threshold:.1f}")
        tone_aggressive = bool(tone["is_aggressive_tone"]) and not calibrating
        if calibrating:
            reasons.append("noise_floor_calibrating")
        if tone_aggressive:
            reasons.append("qualifying:multi_feature_aggressive_tone")
        qualifies = bool(yamnet_aggressive or tone_aggressive)
        if not qualifies:
            reasons.append("non_qualifying_acoustic_window")
        duration = canonical.size / CANONICAL_SAMPLE_RATE
        evidence = AcousticEvidence(
            interval_started_at=interval_started_at.astimezone(timezone.utc),
            interval_ended_at=interval_ended_at.astimezone(timezone.utc),
            monotonic_start=float(monotonic_start),
            monotonic_end=float(monotonic_end),
            yamnet_ran=yamnet_ran,
            yamnet_label=str(yamnet_label),
            yamnet_score=yamnet_score,
            yamnet_aggressive=yamnet_aggressive,
            tone_aggressive=tone_aggressive,
            rms=float(tone["rms"]),
            energy_variance=float(tone["energy_variance"]),
            zero_crossing_rate=float(tone["zero_crossing_rate"]),
            peak_to_average=float(tone["peak_to_average"]),
            duration_seconds=duration,
            reasons=tuple(reasons),
            qualifies_to_open=qualifies,
        )
        self.evidence_count += 1
        print(
            f"[ACOUSTIC_EVIDENCE] count={self.evidence_count} "
            f"qualifies={str(qualifies).lower()} yamnet_ran={str(yamnet_ran).lower()} "
            f"tone_aggressive={str(tone_aggressive).lower()} "
            f"rms={tone['rms']:.1f} floor_rms={self.noise_floor.floor:.1f} "
            f"tone_threshold_rms={tone_threshold:.1f}"
        )
        return evidence


class AcousticDetectorWorker:
    """Run overlapping acoustic windows away from the capture thread."""

    def __init__(
        self,
        detector: AcousticDetector,
        frame_queue: BoundedFrameQueue,
        evidence_consumer,
        *,
        window_samples=15_600,
        hop_samples=7_800,
    ):
        if window_samples <= 0 or hop_samples <= 0:
            raise ValueError("window and hop must be positive")
        if hop_samples > window_samples:
            raise ValueError("hop cannot exceed window")
        self.detector = detector
        self.frame_queue = frame_queue
        self.evidence_consumer = evidence_consumer
        self.window_samples = int(window_samples)
        self.hop_samples = int(hop_samples)
        self._stop_event = threading.Event()
        self._thread = None
        self._buffer = np.empty(0, dtype=np.float32)
        self._buffer_mono_start = None
        self._buffer_wall_start = None
        self._last_sequence = None
        self.discontinuity_count = 0
        self.discarded_incomplete_samples = 0
        self.error = None

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self.error = None
        self._thread = threading.Thread(
            target=self._run,
            name="acoustic-detector",
        )
        self._thread.start()

    def _reset_for_discontinuity(self):
        if self._buffer.size:
            self.discarded_incomplete_samples += self._buffer.size
        self._buffer = np.empty(0, dtype=np.float32)
        self._buffer_mono_start = None
        self._buffer_wall_start = None
        self.discontinuity_count += 1

    def _accept(self, frame: AudioFrame):
        if self._last_sequence is not None and frame.sequence != self._last_sequence + 1:
            self._reset_for_discontinuity()
            print(
                f"[ACOUSTIC_QUEUE] discontinuity_count={self.discontinuity_count}"
            )
        self._last_sequence = frame.sequence
        if self._buffer_mono_start is None:
            self._buffer_mono_start = frame.monotonic_start
            duration = frame.samples.size / CANONICAL_SAMPLE_RATE
            self._buffer_wall_start = frame.captured_at - timedelta(seconds=duration)
        self._buffer = np.concatenate((self._buffer, frame.samples))
        maximum_pending = self.window_samples + self.hop_samples
        if self._buffer.size > maximum_pending:
            overflow = self._buffer.size - maximum_pending
            self._buffer = self._buffer[overflow:]
            offset = overflow / CANONICAL_SAMPLE_RATE
            self._buffer_mono_start += offset
            self._buffer_wall_start += timedelta(seconds=offset)
            self.discontinuity_count += 1
            print(
                f"[ACOUSTIC_QUEUE] overflow_count={self.discontinuity_count} "
                "policy=drop_oldest_samples"
            )
        while self._buffer.size >= self.window_samples:
            window = np.array(
                self._buffer[:self.window_samples],
                dtype=np.float32,
                order="C",
                copy=True,
            )
            duration = self.window_samples / CANONICAL_SAMPLE_RATE
            mono_end = self._buffer_mono_start + duration
            wall_end = self._buffer_wall_start + timedelta(seconds=duration)
            evidence = self.detector.analyze_window(
                window,
                interval_started_at=self._buffer_wall_start,
                interval_ended_at=wall_end,
                monotonic_start=self._buffer_mono_start,
                monotonic_end=mono_end,
            )
            self.evidence_consumer(evidence)
            self._buffer = self._buffer[self.hop_samples:]
            hop_seconds = self.hop_samples / CANONICAL_SAMPLE_RATE
            self._buffer_mono_start += hop_seconds
            self._buffer_wall_start += timedelta(seconds=hop_seconds)

    def _run(self):
        try:
            while not self._stop_event.is_set():
                try:
                    frame = self.frame_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                self._accept(frame)
        except Exception as exc:
            self.error = exc
            self._stop_event.set()
            print(f"[ACOUSTIC] worker_error={type(exc).__name__}")

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._buffer.size:
            self.discarded_incomplete_samples += self._buffer.size
            self._buffer = np.empty(0, dtype=np.float32)
        self._thread = None
