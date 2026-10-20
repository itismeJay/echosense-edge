"""Regression coverage for finalized STT PCM resource bounds."""

from datetime import datetime, timezone
import gc
from pathlib import Path
import tempfile
import unittest
import weakref

import numpy as np

from audio.audio_event import AudioEvent
from config import DetectionRuntimeConfig
from detection.text_detector import TextDetector
from model import whisper_stt
from model.realtimestt_audio_adapter import RealtimeSTTAudioEventAdapter
from model.result_queue import BoundedResultQueue


class FinalizedSttPcmBoundTests(unittest.TestCase):
    def tearDown(self):
        whisper_stt.clear_transcription_results()

    def test_short_finalized_utterance_is_preserved(self):
        samples = np.linspace(-0.25, 0.25, 8_000, dtype=np.float32)
        event = AudioEvent.from_float32_samples(samples, max_samples=16_000)
        np.testing.assert_array_equal(event.samples, samples)
        self.assertEqual(event.duration_ms, 500)

    def test_long_finalized_utterance_retains_only_most_recent_samples(self):
        samples = np.arange(40_000, dtype=np.float32)
        event = AudioEvent.from_float32_samples(samples, max_samples=16_000)
        self.assertEqual(event.samples.size, 16_000)
        np.testing.assert_array_equal(event.samples, samples[-16_000:])

    def test_default_queue_pcm_bound_is_calculable_and_enforced(self):
        config = DetectionRuntimeConfig()
        results = BoundedResultQueue(
            config.stt_result_queue_size,
            max_audio_samples=config.max_stt_utterance_samples,
        )
        for index in range(config.stt_result_queue_size):
            event = AudioEvent.from_float32_samples(
                np.full(
                    config.max_stt_utterance_samples,
                    index / 100,
                    dtype=np.float32,
                ),
                max_samples=config.max_stt_utterance_samples,
            )
            self.assertTrue(results.put({"audio_event": event}))
        queued = [results.get_nowait() for _ in range(results.qsize())]
        retained_bytes = sum(
            item["audio_event"].samples.nbytes for item in queued
        )
        self.assertEqual(
            retained_bytes,
            config.max_stt_result_queue_pcm_bytes,
        )
        self.assertEqual(
            results.max_retained_pcm_bytes,
            32 * 240_000 * 4,
        )

    def test_audio_event_keeps_no_reference_or_view_of_full_source(self):
        source = np.arange(32_000, dtype=np.float32)
        source_reference = weakref.ref(source)
        event = AudioEvent.from_float32_samples(source, max_samples=8_000)
        self.assertFalse(np.shares_memory(event.samples, source))
        del source
        gc.collect()
        self.assertIsNone(source_reference())
        self.assertEqual(event.samples.size, 8_000)

    def test_queue_rejects_oversize_audio_event_defensively(self):
        event = AudioEvent.from_float32_samples(
            np.ones(200, dtype=np.float32),
            max_samples=200,
        )
        results = BoundedResultQueue(32, max_audio_samples=100)
        self.assertFalse(results.put({"audio_event": event}))
        self.assertEqual(results.qsize(), 0)
        self.assertEqual(results.oversize_rejection_count, 1)

    def test_bounded_audio_still_supports_existing_text_detection(self):
        adapter = RealtimeSTTAudioEventAdapter(max_samples=16_000)
        adapter.on_transcription_start(
            np.ones(32_000, dtype=np.float32) * 0.1
        )
        event = adapter.consume_pending()
        whisper_stt._on_text("bobo", "tl", 0.9, event)
        result = whisper_stt.consume_transcription_result()
        evidence = TextDetector().analyze_stt_result(result)
        self.assertEqual(result["audio_event"].samples.size, 16_000)
        self.assertTrue(evidence.qualifies_to_open)

    def test_finalized_audio_remains_ram_only(self):
        with tempfile.TemporaryDirectory() as directory:
            before = tuple(Path(directory).iterdir())
            event = AudioEvent.from_float32_samples(
                np.ones(1_000, dtype=np.float32),
                max_samples=1_000,
            )
            after = tuple(Path(directory).iterdir())
        self.assertEqual(before, after)
        self.assertIsInstance(event.samples, np.ndarray)
        self.assertFalse(any("path" in name for name in event.__slots__))


if __name__ == "__main__":
    unittest.main()
