from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import Mock, patch

import numpy as np

from audio.shared_source import AudioFrame, BoundedFrameQueue
from detection.acoustic_detector import AcousticDetector, AcousticDetectorWorker
from detection.evidence import AcousticEvidence
from model.yamnet_infer import is_aggressive_sound, load_yamnet_bundle


WALL = datetime(2026, 1, 1, tzinfo=timezone.utc)


def tone(*, aggressive=False, rms=50.0, peak_to_average=2.0):
    return {
        "rms": rms,
        "energy_variance": 4_000.0 if aggressive else 10.0,
        "zero_crossing_rate": 0.2 if aggressive else 0.01,
        "peak_to_average": peak_to_average,
        "is_aggressive_tone": aggressive,
    }


class AcousticDetectorAfterTests(unittest.TestCase):
    def analyze(self, detector, tone_result):
        samples = np.zeros(15_600, dtype=np.float32)
        with patch(
            "detection.acoustic_detector.analyze_tone_float32",
            return_value=tone_result,
        ):
            return detector.analyze_window(
                samples,
                interval_started_at=WALL,
                interval_ended_at=WALL + timedelta(seconds=0.975),
                monotonic_start=10.0,
                monotonic_end=10.975,
            )

    def test_yamnet_positive_opens_without_any_transcript(self):
        detector = AcousticDetector(
            yamnet_runner=lambda _samples: ("Screaming", 0.82),
        )
        evidence = self.analyze(detector, tone(aggressive=False))
        self.assertIsInstance(evidence, AcousticEvidence)
        self.assertTrue(evidence.yamnet_aggressive)
        self.assertTrue(evidence.qualifies_to_open)

    def test_non_aggressive_noise_does_not_open(self):
        detector = AcousticDetector(
            yamnet_runner=lambda _samples: ("Speech", 0.91),
        )
        evidence = self.analyze(detector, tone(aggressive=False, rms=80))
        self.assertFalse(evidence.yamnet_aggressive)
        self.assertFalse(evidence.qualifies_to_open)

    def test_loudness_alone_is_not_qualifying(self):
        detector = AcousticDetector(
            yamnet_runner=lambda _samples: ("Speech", 0.99),
        )
        evidence = self.analyze(detector, tone(aggressive=False, rms=2_000))
        self.assertFalse(evidence.tone_aggressive)
        self.assertFalse(evidence.qualifies_to_open)

    def test_loud_neutral_chair_drop_book_drop_and_clap_do_not_open(self):
        for label in ("chair drop", "book drop", "clap"):
            with self.subTest(label=label):
                detector = AcousticDetector(
                    yamnet_runner=lambda _samples: ("Bang", 0.99),
                )
                evidence = self.analyze(
                    detector,
                    tone(
                        aggressive=False,
                        rms=9_000,
                        peak_to_average=12.0,
                    ),
                )
                self.assertFalse(evidence.yamnet_aggressive)
                self.assertFalse(evidence.tone_aggressive)
                self.assertFalse(evidence.qualifies_to_open)

    def test_normal_classroom_noise_does_not_open(self):
        detector = AcousticDetector(
            yamnet_runner=lambda _samples: ("Speech", 0.88),
        )
        evidence = self.analyze(detector, tone(aggressive=False, rms=120))
        self.assertFalse(evidence.qualifies_to_open)

    def test_yell_and_shout_above_existing_threshold_qualify(self):
        for label in ("Yell", "Shout"):
            with self.subTest(label=label):
                detector = AcousticDetector(
                    yamnet_runner=lambda _samples, value=label: (value, 0.45),
                )
                evidence = self.analyze(detector, tone(aggressive=False))
                self.assertTrue(evidence.yamnet_aggressive)
                self.assertTrue(evidence.qualifies_to_open)

    def test_yamnet_score_below_existing_threshold_does_not_qualify(self):
        detector = AcousticDetector(
            yamnet_runner=lambda _samples: ("Screaming", 0.449),
        )
        evidence = self.analyze(detector, tone(aggressive=False))
        self.assertFalse(evidence.qualifies_to_open)

    def test_exact_label_policy_excludes_dog_whimper_and_children_shouting(self):
        self.assertTrue(is_aggressive_sound("Whimper", 0.8, 0.45))
        self.assertFalse(is_aggressive_sound("Whimper (dog)", 0.99, 0.45))
        self.assertFalse(is_aggressive_sound("dog whimper", 0.99, 0.45))
        self.assertFalse(is_aggressive_sound("Children shouting", 0.99, 0.45))

    def test_acoustic_threshold_rejects_non_finite_values(self):
        for threshold in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    AcousticDetector(yamnet_threshold=threshold)

    def test_existing_multi_feature_tone_can_open_in_degraded_mode(self):
        detector = AcousticDetector(unavailable_reason="model_file_missing")
        evidence = self.analyze(detector, tone(aggressive=True, rms=500))
        self.assertFalse(evidence.yamnet_ran)
        self.assertTrue(evidence.tone_aggressive)
        self.assertTrue(evidence.qualifies_to_open)
        self.assertIn("yamnet_unavailable:model_file_missing", evidence.reasons)

    def test_model_unavailable_state_is_explicit(self):
        status = load_yamnet_bundle(
            "/definitely/missing/model.tflite",
            "/definitely/missing/classes.csv",
            degraded_mode=True,
        )
        self.assertFalse(status.available)
        self.assertIn("model_file_missing", status.reason)
        with self.assertRaises(RuntimeError):
            load_yamnet_bundle(
                "/definitely/missing/model.tflite",
                "/definitely/missing/classes.csv",
                degraded_mode=False,
            )

    def test_worker_uses_overlapping_windows_and_discards_incomplete_tail(self):
        detector = Mock()
        detector.analyze_window.side_effect = ["one", "two"]
        results = []
        worker = AcousticDetectorWorker(
            detector,
            BoundedFrameQueue("acoustic", 2),
            results.append,
            window_samples=8,
            hop_samples=4,
        )
        frame = AudioFrame.canonical(
            np.linspace(-0.1, 0.1, 12, dtype=np.float32),
            sequence=0,
            monotonic_end=1.0,
            captured_at=WALL,
        )
        worker._accept(frame)
        self.assertEqual(results, ["one", "two"])
        self.assertEqual(detector.analyze_window.call_count, 2)
        worker.stop()
        self.assertEqual(worker.discarded_incomplete_samples, 4)


if __name__ == "__main__":
    unittest.main()
