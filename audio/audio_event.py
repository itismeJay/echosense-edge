"""Immutable, in-memory audio captured for one finalized STT utterance."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import numpy as np


def float32_samples_to_int16(samples: np.ndarray) -> np.ndarray:
    """Scale normalized float32 audio into the legacy signed-int16 domain."""
    if not isinstance(samples, np.ndarray) or samples.dtype != np.float32:
        raise TypeError("samples must be a float32 NumPy array")
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("samples must be non-empty mono audio")
    clipped = np.clip(samples, -1.0, 32767.0 / 32768.0)
    return np.rint(clipped * 32768.0).astype(np.int16)


@dataclass(frozen=True, slots=True)
class AudioEvent:
    """The exact mono float32 samples submitted for one transcription."""

    event_id: str
    started_at: datetime
    ended_at: datetime
    duration_ms: int
    sample_rate: int
    samples: np.ndarray

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id must not be empty")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.started_at.tzinfo is None or self.ended_at.tzinfo is None:
            raise ValueError("audio event timestamps must be timezone-aware")
        if self.started_at > self.ended_at:
            raise ValueError("started_at must not be after ended_at")
        if not isinstance(self.samples, np.ndarray):
            raise TypeError("samples must be a NumPy array")
        if self.samples.ndim != 1 or self.samples.size == 0:
            raise ValueError("samples must be a non-empty mono array")
        if self.samples.dtype != np.float32:
            raise TypeError("samples must have float32 dtype")
        if not self.samples.flags.c_contiguous:
            raise ValueError("samples must be contiguous")
        self.samples.setflags(write=False)
        expected_duration_ms = round(
            self.samples.size / self.sample_rate * 1000
        )
        if self.duration_ms != expected_duration_ms:
            raise ValueError("duration_ms must be derived from the sample count")

    @classmethod
    def from_float32_samples(
        cls,
        samples: np.ndarray,
        *,
        sample_rate: int = 16000,
        ended_at: datetime | None = None,
        max_samples: int | None = None,
    ) -> "AudioEvent":
        """Copy a bounded tail of finalized RealtimeSTT audio.

        The most recent samples are retained because transcript semantics do
        not require full raw PCM, while the tail best aligns with the final
        words and incident context. The returned array never shares memory
        with the potentially much larger third-party callback array.
        """
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not isinstance(samples, np.ndarray):
            raise TypeError("samples must be a NumPy array")
        if samples.ndim != 1:
            raise ValueError("samples must be one-dimensional mono audio")
        if samples.size == 0:
            raise ValueError("samples must not be empty")
        if not np.issubdtype(samples.dtype, np.number):
            raise TypeError("samples must have a numeric dtype")

        if max_samples is None:
            # Import lazily so this low-level value object has no configuration
            # initialization side effect until its factory is used.
            from config import DETECTION_RUNTIME

            max_samples = DETECTION_RUNTIME.max_stt_utterance_samples
        if isinstance(max_samples, bool) or not isinstance(max_samples, int):
            raise TypeError("max_samples must be an integer")
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")

        retained = samples[-max_samples:]

        copied_samples = np.array(
            retained,
            dtype=np.float32,
            order="C",
            copy=True,
        )
        if not np.all(np.isfinite(copied_samples)):
            raise ValueError("samples must contain only finite values")
        copied_samples.setflags(write=False)

        event_ended_at = ended_at or datetime.now(timezone.utc)
        if event_ended_at.tzinfo is None:
            raise ValueError("ended_at must be timezone-aware")
        event_ended_at = event_ended_at.astimezone(timezone.utc)
        duration_ms = round(copied_samples.size / sample_rate * 1000)

        return cls(
            event_id=str(uuid4()),
            started_at=event_ended_at - timedelta(
                seconds=copied_samples.size / sample_rate
            ),
            ended_at=event_ended_at,
            duration_ms=duration_ms,
            sample_rate=sample_rate,
            samples=copied_samples,
        )
