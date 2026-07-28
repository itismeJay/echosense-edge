import unittest
from datetime import datetime, timezone

import numpy as np

from audio.audio_event import AudioEvent, float32_samples_to_int16
from audio.waveform import get_waveform_snapshot_float32


class AudioEventTests(unittest.TestCase):
    def test_duration_uses_sample_count(self):
        event = AudioEvent.from_float32_samples(
            np.zeros(16000, dtype=np.float32),
            sample_rate=16000,
            ended_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )

        self.assertEqual(event.duration_ms, 1000)
        self.assertEqual(
            (event.ended_at - event.started_at).total_seconds(),
            1.0,
        )
        self.assertIsNotNone(event.started_at.tzinfo)
        self.assertIsNotNone(event.ended_at.tzinfo)

    def test_factory_copies_and_makes_samples_read_only(self):
        original = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        event = AudioEvent.from_float32_samples(original)

        original[0] = 0.9
        self.assertAlmostEqual(float(event.samples[0]), 0.1)
        self.assertFalse(event.samples.flags.writeable)
        with self.assertRaises(ValueError):
            event.samples[0] = 0.0

    def test_factory_outputs_mono_contiguous_float32(self):
        source = np.arange(12, dtype=np.float64)[::2]
        self.assertFalse(source.flags.c_contiguous)

        event = AudioEvent.from_float32_samples(source)

        self.assertEqual(event.samples.ndim, 1)
        self.assertEqual(event.samples.dtype, np.float32)
        self.assertTrue(event.samples.flags.c_contiguous)

    def test_malformed_samples_are_rejected(self):
        malformed = (
            None,
            [0.1, 0.2],
            np.array([], dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
            np.array([np.nan], dtype=np.float32),
        )

        for value in malformed:
            with self.subTest(value=repr(value)):
                with self.assertRaises((TypeError, ValueError)):
                    AudioEvent.from_float32_samples(value)

    def test_float32_to_int16_preserves_legacy_amplitude_domain(self):
        samples = np.array(
            [-1.0, -0.5, 0.0, 0.5, 1.0],
            dtype=np.float32,
        )

        converted = float32_samples_to_int16(samples)

        np.testing.assert_array_equal(
            converted,
            np.array([-32768, -16384, 0, 16384, 32767], dtype=np.int16),
        )

    def test_waveform_is_derived_from_float32_event_amplitudes(self):
        samples = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
        self.assertEqual(
            get_waveform_snapshot_float32(samples, num_points=5),
            [32768, 16384, 0, 16384, 32767],
        )


if __name__ == "__main__":
    unittest.main()
