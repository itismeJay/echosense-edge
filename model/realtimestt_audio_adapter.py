"""Isolated bridge from RealtimeSTT's callback audio to an AudioEvent."""

import threading

import numpy as np

from audio.audio_event import AudioEvent


class RealtimeSTTAudioEventAdapter:
    """Own the pending finalized event for one synchronous ``text()`` cycle.

    RealtimeSTT 1.0.2 currently passes the finalized transcription audio to
    ``on_transcription_start``. The audio payload is not a documented stable
    contract, so this dependency is intentionally isolated here. Package
    upgrades must run the adapter contract tests before deployment.
    """

    def __init__(self, sample_rate: int = 16000):
        self._sample_rate = sample_rate
        self._lock = threading.Lock()
        self._pending_event = None

    def clear_pending(self) -> None:
        with self._lock:
            self._pending_event = None

    def consume_pending(self) -> AudioEvent | None:
        with self._lock:
            event = self._pending_event
            self._pending_event = None
            return event

    def on_transcription_start(self, audio) -> bool:
        """Capture callback audio and return falsy so transcription continues."""
        try:
            if not isinstance(audio, np.ndarray) or audio.dtype != np.float32:
                raise TypeError(
                    "RealtimeSTT callback audio must be a float32 NumPy array"
                )
            event = AudioEvent.from_float32_samples(
                audio,
                sample_rate=self._sample_rate,
            )
        except (TypeError, ValueError) as exc:
            self.clear_pending()
            print(f"[AUDIO_EVENT] unavailable: {exc}")
            return False

        with self._lock:
            self._pending_event = event
        print(
            f"[AUDIO_EVENT] id={event.event_id} samples={event.samples.size} "
            f"duration_ms={event.duration_ms}"
        )
        return False
