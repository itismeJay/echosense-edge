"""Bounded FIFO for finalized STT results with observable overflow."""

import queue
import threading


class BoundedResultQueue:
    def __init__(self, max_size: int, *, max_audio_samples=None):
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.max_size = int(max_size)
        if max_audio_samples is not None:
            if (
                isinstance(max_audio_samples, bool)
                or not isinstance(max_audio_samples, int)
                or max_audio_samples <= 0
            ):
                raise ValueError("max_audio_samples must be a positive integer")
        self.max_audio_samples = max_audio_samples
        self._queue = queue.Queue(maxsize=self.max_size)
        self._lock = threading.Lock()
        self._overflow_count = 0
        self._oversize_rejection_count = 0
        self._closed = False

    @property
    def overflow_count(self):
        with self._lock:
            return self._overflow_count

    @property
    def closed(self):
        with self._lock:
            return self._closed

    @property
    def oversize_rejection_count(self):
        with self._lock:
            return self._oversize_rejection_count

    @property
    def max_retained_pcm_bytes(self):
        if self.max_audio_samples is None:
            return None
        return self.max_size * self.max_audio_samples * 4

    def _audio_sample_count(self, result):
        if not isinstance(result, dict):
            return 0
        event = result.get("audio_event")
        samples = getattr(event, "samples", None)
        size = getattr(samples, "size", 0)
        return int(size) if size else 0

    def put(self, result):
        with self._lock:
            if self._closed:
                return False
        sample_count = self._audio_sample_count(result)
        if (
            self.max_audio_samples is not None
            and sample_count > self.max_audio_samples
        ):
            with self._lock:
                self._oversize_rejection_count += 1
                count = self._oversize_rejection_count
            if count == 1 or count % 100 == 0:
                print(
                    f"[STT_QUEUE] oversize_rejection_count={count} "
                    "policy=drop_newest"
                )
            return False
        try:
            self._queue.put_nowait(result)
            return True
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            with self._lock:
                self._overflow_count += 1
                count = self._overflow_count
            try:
                self._queue.put_nowait(result)
            except queue.Full:
                return False
            if count == 1 or count % 100 == 0:
                print(
                    f"[STT_QUEUE] overflow_count={count} "
                    "policy=drop_oldest"
                )
            return False

    def get(self, timeout=None):
        return self._queue.get(timeout=timeout)

    def get_nowait(self):
        return self._queue.get_nowait()

    def qsize(self):
        return self._queue.qsize()

    def clear(self):
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def close(self):
        with self._lock:
            self._closed = True
