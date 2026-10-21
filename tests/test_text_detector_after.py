import inspect
import unittest

from detection.evidence import TextEvidence
from detection.text_detector import TextDetector
from model import whisper_stt
from model.result_queue import BoundedResultQueue


class FakeClock:
    def __init__(self, value=10.0):
        self.value = value

    def __call__(self):
        return self.value


class TextDetectorAfterTests(unittest.TestCase):
    def setUp(self):
        whisper_stt.clear_transcription_results()

    def tearDown(self):
        whisper_stt.clear_transcription_results()

    def test_detector_is_independent_from_yamnet_and_sender(self):
        source = inspect.getsource(__import__(
            "detection.text_detector",
            fromlist=["TextDetector"],
        ))
        self.assertNotIn("yamnet", source.lower())
        self.assertNotIn("send_alert", source)
        self.assertNotIn("enqueue", source)

    def test_bobo_and_tanga_open_text_events(self):
        detector = TextDetector()
        for text in ("bobo", "tanga"):
            with self.subTest(text=text):
                evidence = detector.analyze_transcript(text, language="fil")
                self.assertIsInstance(evidence, TextEvidence)
                self.assertTrue(evidence.qualifies_to_open)
                # Opens an incident, but stays LOW without a direct target.
                self.assertEqual(evidence.baseline_severity, "LOW")

    def test_pangit_suppression_and_pangit_ka_direction_are_preserved(self):
        detector = TextDetector()
        bare = detector.analyze_transcript("pangit", language="fil")
        directed = detector.analyze_transcript("pangit ka", language="fil")
        self.assertFalse(bare.qualifies_to_open)
        self.assertTrue(directed.qualifies_to_open)
        self.assertTrue(directed.direct_target)

    def test_repetition_is_bounded_and_promotes_second_observation(self):
        clock = FakeClock()
        detector = TextDetector(
            monotonic_clock=clock,
            history_size=2,
        )
        first = detector.analyze_transcript("bobo ka")
        clock.value += 1
        second = detector.analyze_transcript("bobo ka")
        self.assertFalse(first.repeated)
        self.assertTrue(second.repeated)
        self.assertEqual(second.baseline_severity, "HIGH")
        self.assertEqual(detector._history["bobo"].maxlen, 2)

    def test_quality_and_laughter_protections_remain_in_evidence(self):
        detector = TextDetector()
        low_quality = detector.analyze_transcript(
            "bobo bobo bobo bobo bobo bobo"
        )
        laughter = detector.analyze_transcript("haha bobo")
        self.assertFalse(low_quality.quality_accepted)
        self.assertFalse(low_quality.qualifies_to_open)
        self.assertTrue(laughter.laughter_present)
        self.assertFalse(laughter.qualifies_to_open)

    def test_finalized_stt_results_are_fifo_not_latest_value(self):
        whisper_stt._on_text("first harmless", whisper_language="en")
        whisper_stt._on_text("second harmless", whisper_language="en")
        first = whisper_stt.consume_transcription_result()
        second = whisper_stt.consume_transcription_result()
        self.assertEqual(first["transcribed_text"], "first harmless")
        self.assertEqual(second["transcribed_text"], "second harmless")

    def test_result_queue_has_explicit_drop_oldest_overflow(self):
        results = BoundedResultQueue(2)
        self.assertTrue(results.put({"id": 1}))
        self.assertTrue(results.put({"id": 2}))
        self.assertFalse(results.put({"id": 3}))
        self.assertEqual(results.overflow_count, 1)
        self.assertEqual(results.get_nowait()["id"], 2)
        self.assertEqual(results.get_nowait()["id"], 3)
        results.close()
        self.assertFalse(results.put({"id": 4}))


if __name__ == "__main__":
    unittest.main()
