"""One bounded, application-owned source for canonical continuous audio."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import queue
import threading
import time
from typing import Iterable

import numpy as np


CANONICAL_SAMPLE_RATE = 16_000
CANONICAL_CHANNELS = 1
CANONICAL_DTYPE = np.dtype(np.float32)


@dataclass(frozen=True, slots=True)
class AudioFormat:
    sample_rate: int = CANONICAL_SAMPLE_RATE
    channels: int = CANONICAL_CHANNELS
    dtype: str = "float32"

    def __post_init__(self):
        if self.sample_rate != CANONICAL_SAMPLE_RATE:
            raise ValueError("canonical audio must be 16000 Hz")
        if self.channels != CANONICAL_CHANNELS:
            raise ValueError("canonical audio must be mono")
        if np.dtype(self.dtype) != CANONICAL_DTYPE:
            raise ValueError("canonical audio must use float32 samples")


@dataclass(frozen=True, slots=True)
class AudioFrame:
    """Immutable RAM-only PCM handed to every detector consumer."""

    sequence: int
    captured_at: datetime
    monotonic_start: float
    monotonic_end: float
    samples: np.ndarray
    format: AudioFormat = AudioFormat()

    def __post_init__(self):
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        if self.captured_at.tzinfo is None:
            raise ValueError("captured_at must be timezone-aware")
        if self.monotonic_start > self.monotonic_end:
            raise ValueError("monotonic interval is invalid")
        if not isinstance(self.samples, np.ndarray):
            raise TypeError("samples must be a NumPy array")
        if self.samples.dtype != CANONICAL_DTYPE:
            raise TypeError("samples must be float32")
        if self.samples.ndim != 1 or self.samples.size == 0:
            raise ValueError("samples must be non-empty mono audio")
        if not self.samples.flags.c_contiguous:
            raise ValueError("samples must be contiguous")
        if not np.all(np.isfinite(self.samples)):
            raise ValueError("samples must be finite")
        if np.max(np.abs(self.samples)) > 1.0:
            raise ValueError("float32 samples must be normalized")
        self.samples.setflags(write=False)

    @classmethod
    def canonical(
        cls,
        samples,
        *,
        sequence: int,
        monotonic_end: float,
        captured_at: datetime | None = None,
    ) -> "AudioFrame":
        source = np.asarray(samples)
        if source.ndim != 1:
            raise ValueError("source audio must be one-dimensional mono")
        if source.dtype == np.int16:
            canonical = np.array(
                source,
                dtype=np.float32,
                order="C",
                copy=True,
            ) / 32768.0
        elif np.issubdtype(source.dtype, np.floating):
            canonical = np.array(
                source,
                dtype=np.float32,
                order="C",
                copy=True,
            )
        else:
            raise TypeError("source audio must be int16 PCM or floating point")
        if canonical.size == 0:
            raise ValueError("source audio must not be empty")
        if not np.all(np.isfinite(canonical)):
            raise ValueError("source audio must be finite")
        if np.max(np.abs(canonical)) > 1.0:
            raise ValueError("floating point source audio must be normalized")
        duration = canonical.size / CANONICAL_SAMPLE_RATE
        return cls(
            sequence=sequence,
            captured_at=(captured_at or datetime.now(timezone.utc)).astimezone(
                timezone.utc
            ),
            monotonic_start=float(monotonic_end) - duration,
            monotonic_end=float(monotonic_end),
            samples=canonical,
        )


class BoundedFrameQueue:
    """Non-blocking subscriber queue with observable drop-oldest overflow."""

    def __init__(self, name: str, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.name = str(name)
        self.capacity = int(capacity)
        self._queue = queue.Queue(maxsize=self.capacity)
        self._lock = threading.Lock()
        self._dropped = 0

    @property
    def dropped_count(self):
        with self._lock:
            return self._dropped

    def put(self, frame: AudioFrame):
        try:
            self._queue.put_nowait(frame)
            return True
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            with self._lock:
                self._dropped += 1
                dropped = self._dropped
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                with self._lock:
                    self._dropped += 1
                return False
            if dropped == 1 or dropped % 100 == 0:
                print(
                    f"[AUDIO_QUEUE] consumer={self.name} "
                    f"overflow_count={dropped} policy=drop_oldest"
                )
            return False

    def get(self, timeout=None):
        return self._queue.get(timeout=timeout)

    def empty(self):
        return self._queue.empty()

    def qsize(self):
        return self._queue.qsize()


class RollingAudioBuffer:
    """Bounded in-memory context buffer measured by canonical sample count."""

    def __init__(self, seconds: float, sample_rate: int = CANONICAL_SAMPLE_RATE):
        if seconds < 0:
            raise ValueError("seconds must be non-negative")
        if sample_rate != CANONICAL_SAMPLE_RATE:
            raise ValueError("rolling buffer requires canonical sample rate")
        self.sample_rate = sample_rate
        self.max_samples = int(round(seconds * sample_rate))
        self._chunks = deque()
        self._sample_count = 0
        self._lock = threading.Lock()

    @property
    def sample_count(self):
        with self._lock:
            return self._sample_count

    def append(self, frame: AudioFrame):
        if frame.format.sample_rate != self.sample_rate:
            raise ValueError("frame sample rate does not match rolling buffer")
        if self.max_samples == 0:
            return
        samples = frame.samples
        if samples.size > self.max_samples:
            samples = samples[-self.max_samples:]
        with self._lock:
            self._chunks.append(samples)
            self._sample_count += samples.size
            while self._chunks and self._sample_count > self.max_samples:
                excess = self._sample_count - self.max_samples
                oldest = self._chunks[0]
                if oldest.size <= excess:
                    self._chunks.popleft()
                    self._sample_count -= oldest.size
                else:
                    self._chunks[0] = oldest[excess:]
                    self._sample_count -= excess

    def seed(self, samples):
        array = np.asarray(samples, dtype=np.float32)
        if array.size:
            self.append(
                AudioFrame.canonical(
                    array,
                    sequence=0,
                    monotonic_end=time.monotonic(),
                )
            )

    def snapshot(self, seconds: float | None = None):
        with self._lock:
            if not self._chunks:
                result = np.empty(0, dtype=np.float32)
            else:
                result = np.concatenate(tuple(self._chunks)).astype(
                    np.float32,
                    copy=False,
                )
        if seconds is not None:
            if seconds < 0:
                raise ValueError("seconds must be non-negative")
            requested = int(round(seconds * self.sample_rate))
            if requested == 0:
                result = result[:0]
            elif result.size > requested:
                result = result[-requested:]
        result = np.array(result, dtype=np.float32, order="C", copy=True)
        result.setflags(write=False)
        return result

    def clear(self):
        with self._lock:
            self._chunks.clear()
            self._sample_count = 0


class SharedAudioSource:
    """Fan out one canonical stream without waiting for any consumer."""

    def __init__(self, *, pre_trigger_seconds=2.0, max_frame_samples=15_600):
        if max_frame_samples <= 0:
            raise ValueError("max_frame_samples must be positive")
        self.format = AudioFormat()
        self.max_frame_samples = int(max_frame_samples)
        self.pre_trigger = RollingAudioBuffer(pre_trigger_seconds)
        self._subscribers = []
        self._subscriber_lock = threading.Lock()
        self._sequence = 0

    def subscribe(self, name: str, capacity: int):
        subscriber = BoundedFrameQueue(name, capacity)
        with self._subscriber_lock:
            self._subscribers.append(subscriber)
        return subscriber

    def publish(self, samples, *, monotonic_end=None, captured_at=None):
        if np.asarray(samples).size > self.max_frame_samples:
            raise ValueError("audio frame exceeds configured sample bound")
        frame = AudioFrame.canonical(
            samples,
            sequence=self._sequence,
            monotonic_end=(
                time.monotonic() if monotonic_end is None else monotonic_end
            ),
            captured_at=captured_at,
        )
        self._sequence += 1
        self.pre_trigger.append(frame)
        with self._subscriber_lock:
            subscribers = tuple(self._subscribers)
        for subscriber in subscribers:
            subscriber.put(frame)
        return frame

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError


class FakeAudioSource(SharedAudioSource):
    """Deterministic source for Mac tests; it never opens physical hardware."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.open_count = 0
        self.started = False

    def start(self):
        if not self.started:
            self.open_count += 1
            self.started = True
            print("[AUDIO_SOURCE] started type=fake")

    def emit(self, samples, **kwargs):
        if not self.started:
            raise RuntimeError("audio source is not started")
        return self.publish(samples, **kwargs)

    def stop(self):
        if self.started:
            self.started = False
            self.pre_trigger.clear()
            print("[AUDIO_SOURCE] stopped type=fake")


def select_input_device(
    inventory: Iterable[dict],
    *,
    preferred_name: str = "",
    strict: bool = True,
):
    """Select deterministically from a testable PyAudio-style inventory."""

    devices = sorted(
        (
            {
                "index": int(item["index"]),
                "name": str(item.get("name") or ""),
                "max_input_channels": int(
                    item.get("max_input_channels", item.get("maxInputChannels", 0))
                ),
            }
            for item in inventory
        ),
        key=lambda item: item["index"],
    )
    inputs = [item for item in devices if item["max_input_channels"] > 0]
    preference = str(preferred_name or "").strip().casefold()
    if preference:
        exact = [item for item in inputs if item["name"].casefold() == preference]
        partial = [item for item in inputs if preference in item["name"].casefold()]
        matches = exact or partial
        if matches:
            return matches[0]
        if strict:
            raise RuntimeError("preferred microphone was not found")
    if strict and not preference:
        raise RuntimeError("strict microphone selection requires a preferred name")
    if not inputs:
        raise RuntimeError("no input microphone is available")
    return inputs[0]


class PyAudioSharedSource(SharedAudioSource):
    """The sole production physical reader; frames are converted at its edge."""

    def __init__(
        self,
        *,
        preferred_name: str,
        strict: bool,
        frame_samples: int = 1_040,
        pre_trigger_seconds: float = 2.0,
        pyaudio_factory=None,
    ):
        super().__init__(
            pre_trigger_seconds=pre_trigger_seconds,
            max_frame_samples=frame_samples,
        )
        self.preferred_name = preferred_name
        self.strict = strict
        self.frame_samples = int(frame_samples)
        self._pyaudio_factory = pyaudio_factory
        self._pa = None
        self._stream = None
        self._thread = None
        self._stop_event = threading.Event()
        self.open_count = 0
        self.selected_device = None
        self.error = None

    def _factory(self):
        if self._pyaudio_factory is not None:
            return self._pyaudio_factory()
        import pyaudio

        return pyaudio.PyAudio()

    def _inventory(self):
        return [
            {
                "index": index,
                "name": info.get("name", ""),
                "max_input_channels": info.get("maxInputChannels", 0),
            }
            for index in range(self._pa.get_device_count())
            for info in (self._pa.get_device_info_by_index(index),)
        ]

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self.error = None
        self._pa = self._factory()
        selected = select_input_device(
            self._inventory(),
            preferred_name=self.preferred_name,
            strict=self.strict,
        )
        self.selected_device = selected
        channels = min(selected["max_input_channels"], 2)
        try:
            import pyaudio

            int16_format = pyaudio.paInt16
        except ImportError:
            int16_format = 8
        self._stream = self._pa.open(
            format=int16_format,
            channels=channels,
            rate=CANONICAL_SAMPLE_RATE,
            input=True,
            input_device_index=selected["index"],
            frames_per_buffer=self.frame_samples,
        )
        self.open_count += 1
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._read_loop,
            name="shared-audio-source",
        )
        self._thread.start()
        print(
            f"[AUDIO_SOURCE] started device_index={selected['index']} "
            f"device_name={selected['name']} format=mono-float32-16000"
        )

    def _read_loop(self):
        channels = min(self.selected_device["max_input_channels"], 2)
        while not self._stop_event.is_set():
            try:
                data = self._stream.read(
                    self.frame_samples,
                    exception_on_overflow=False,
                )
            except Exception as exc:
                if not self._stop_event.is_set():
                    self.error = exc
                    print(f"[AUDIO_SOURCE] fatal_read_error={type(exc).__name__}")
                break
            values = np.frombuffer(data, dtype=np.int16)
            if channels > 1:
                usable = values.size - (values.size % channels)
                values = values[:usable].reshape(-1, channels).mean(axis=1)
            canonical = np.asarray(values, dtype=np.float32) / 32768.0
            self.publish(canonical)

    def stop(self):
        self._stop_event.set()
        stream = self._stream
        if stream is not None:
            try:
                stream.stop_stream()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._pa is not None:
            self._pa.terminate()
        self._thread = None
        self._stream = None
        self._pa = None
        self.pre_trigger.clear()
        print("[AUDIO_SOURCE] stopped type=pyaudio")
