import inspect
import queue
import threading
import unittest

import numpy as np

from audio.shared_source import (
    AudioFrame,
    AudioFormat,
    FakeAudioSource,
    RollingAudioBuffer,
    select_input_device,
)
from model.whisper_stt import NoiseReducer, RealtimeSTTFeedWorker, _recorder_loop


class SharedAudioAfterTests(unittest.TestCase):
    def test_one_source_open_fans_same_frame_to_both_consumers(self):
        source = FakeAudioSource(pre_trigger_seconds=2.0)
        text = source.subscribe("text", 2)
        acoustic = source.subscribe("acoustic", 2)
        source.start()
        source.start()
        frame = source.emit(np.ones(1_040, dtype=np.float32) * 0.25)

        self.assertEqual(source.open_count, 1)
        self.assertIs(text.get(timeout=0.1), frame)
        self.assertIs(acoustic.get(timeout=0.1), frame)
        source.stop()
        self.assertFalse(source.started)

    def test_canonical_boundary_converts_int16_and_rejects_stereo(self):
        source = FakeAudioSource()
        consumer = source.subscribe("test", 1)
        source.start()
        source.emit(np.array([32767, -32768], dtype=np.int16))
        frame = consumer.get(timeout=0.1)
        self.assertEqual(frame.samples.dtype, np.float32)
        self.assertEqual(frame.format, AudioFormat())
        self.assertAlmostEqual(float(frame.samples[0]), 32767 / 32768)
        with self.assertRaises(ValueError):
            source.emit(np.zeros((2, 2), dtype=np.float32))
        source.stop()

    def test_pretrigger_rolls_at_exact_configured_bound(self):
        source = FakeAudioSource(pre_trigger_seconds=2.0)
        source.start()
        for value in (0.1, 0.2, 0.3, 0.4, 0.5):
            source.emit(np.full(8_000, value, dtype=np.float32))
        snapshot = source.pre_trigger.snapshot()
        self.assertEqual(snapshot.size, 32_000)
        self.assertTrue(np.all(snapshot[:8_000] == np.float32(0.2)))
        self.assertTrue(np.all(snapshot[-8_000:] == np.float32(0.5)))
        source.stop()

    def test_slow_consumer_never_blocks_and_overflow_is_observable(self):
        source = FakeAudioSource()
        slow = source.subscribe("slow", 1)
        source.start()
        for sequence in range(5):
            source.emit(np.full(100, sequence / 10, dtype=np.float32))
        self.assertEqual(slow.qsize(), 1)
        self.assertEqual(slow.dropped_count, 4)
        self.assertAlmostEqual(float(slow.get().samples[0]), 0.4)
        source.stop()

    def test_rolling_buffer_is_ram_only_and_clears(self):
        buffer = RollingAudioBuffer(0.25)
        source = FakeAudioSource(pre_trigger_seconds=0.25)
        source.start()
        frame = source.emit(np.ones(4_000, dtype=np.float32) * 0.1)
        buffer.append(frame)
        self.assertEqual(buffer.sample_count, 4_000)
        self.assertFalse(hasattr(buffer, "path"))
        buffer.clear()
        self.assertEqual(buffer.sample_count, 0)
        source.stop()

    def test_realtimestt_production_mode_disables_its_microphone(self):
        source = inspect.getsource(_recorder_loop)
        self.assertIn("use_microphone=False", source)
        self.assertIn("on_transcription_start=_audio_event_adapter", source)
        self.assertIn("feed_audio", inspect.getsource(RealtimeSTTFeedWorker))

    def test_realtimestt_feed_worker_uses_bounded_canonical_frames(self):
        fed = threading.Event()

        class Recorder:
            def __init__(self):
                self.audio_queue = queue.Queue()
                self.calls = []

            def feed_audio(self, chunk, original_sample_rate):
                self.calls.append((chunk, original_sample_rate))
                fed.set()

        recorder = Recorder()
        frames = FakeAudioSource().subscribe("stt", 1)
        worker = RealtimeSTTFeedWorker(
            frames,
            recorder_provider=lambda: recorder,
            max_internal_chunks=1,
            noise_reducer=NoiseReducer(enabled=False),
        )
        worker.start()
        frames.put(AudioFrame.canonical(
            np.ones(100, dtype=np.float32) * 0.5,
            sequence=0,
            monotonic_end=1.0,
        ))
        self.assertTrue(fed.wait(1.0))
        worker.stop()
        self.assertEqual(recorder.calls[0][1], 16_000)
        self.assertEqual(len(recorder.calls[0][0]), 200)
        self.assertIsNone(worker.error)

    def test_microphone_selection_is_deterministic_and_strict(self):
        devices = [
            {"index": 4, "name": "Built-in", "maxInputChannels": 1},
            {"index": 2, "name": "EMEET M0 Plus USB", "maxInputChannels": 2},
            {"index": 1, "name": "Output", "maxInputChannels": 0},
        ]
        selected = select_input_device(
            devices,
            preferred_name="EMEET M0 Plus",
            strict=True,
        )
        self.assertEqual(selected["index"], 2)
        with self.assertRaises(RuntimeError):
            select_input_device(
                devices,
                preferred_name="Missing Mic",
                strict=True,
            )
        fallback = select_input_device(
            devices,
            preferred_name="Missing Mic",
            strict=False,
        )
        self.assertEqual(fallback["index"], 2)


if __name__ == "__main__":
    unittest.main()
