import ast
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from audio.audio_event import AudioEvent
from detection.aggression import AggressionDetector
from model.tone_analyzer import analyze_tone, analyze_tone_float32
from model.yamnet_infer import run_yamnet, run_yamnet_float32


class _FakeInterpreter:
    def __init__(self):
        self.input_value = None

    def get_input_details(self):
        return [{"index": 0}]

    def get_output_details(self):
        return [{"index": 1}]

    def set_tensor(self, _index, value):
        self.input_value = value.copy()

    def invoke(self):
        return None

    def get_tensor(self, _index):
        return np.array([[0.2, 0.8]], dtype=np.float32)


def _stt_result(word="yawa"):
    return {
        "has_profanity": True,
        "hard_hits": [word],
        "soft_hits": [],
        "detected_words": [word],
        "is_casual": False,
        "severity": "high",
        "categories": ["profanity"],
        "transcribed_text": f"{word} ka",
        "language": "ceb",
        "language_confidence": 0.9,
        "matched_terms": [],
    }


def _tone_result():
    return {
        "rms": 900.0,
        "energy_variance": 9000.0,
        "zero_crossing_rate": 0.2,
        "peak_to_average": 2.0,
        "is_aggressive_tone": True,
    }


class AudioAlignmentTests(unittest.TestCase):
    def test_yamnet_float32_is_not_divided_by_32768_again(self):
        interpreter = _FakeInterpreter()
        samples = np.full(15600, 0.5, dtype=np.float32)

        run_yamnet_float32(interpreter, samples, ["Speech", "Screaming"])

        np.testing.assert_array_equal(interpreter.input_value, samples)

    def test_legacy_yamnet_int16_is_explicitly_normalized(self):
        interpreter = _FakeInterpreter()
        samples = np.full(15600, 16384, dtype=np.int16)

        run_yamnet(interpreter, samples, ["Speech", "Screaming"])

        np.testing.assert_array_equal(
            interpreter.input_value,
            np.full(15600, 0.5, dtype=np.float32),
        )

    def test_float32_tone_adapter_preserves_int16_threshold_domain(self):
        int16_samples = np.array(
            [-32768, -12000, -500, 0, 500, 12000, 32767] * 1200,
            dtype=np.int16,
        )
        float32_samples = int16_samples.astype(np.float32) / 32768.0

        legacy = analyze_tone(int16_samples)
        synchronized = analyze_tone_float32(float32_samples)

        self.assertEqual(
            synchronized["is_aggressive_tone"],
            legacy["is_aggressive_tone"],
        )
        for key in (
            "rms",
            "energy_variance",
            "zero_crossing_rate",
            "peak_to_average",
        ):
            self.assertAlmostEqual(synchronized[key], legacy[key], places=5)

    def test_track_a_uses_one_event_for_yamnet_tone_waveform_and_duration(self):
        event = AudioEvent.from_float32_samples(
            np.linspace(-0.75, 0.75, 20000, dtype=np.float32)
        )
        observed = {}

        def tone(samples, sample_rate):
            observed["tone"] = samples
            observed["tone_rate"] = sample_rate
            return _tone_result()

        def yamnet(samples, sample_rate, interpreter, class_names):
            observed["yamnet"] = samples
            observed["yamnet_rate"] = sample_rate
            return "Screaming", 0.82

        def waveform(samples):
            observed["waveform"] = samples
            return [123] * 40

        detector = AggressionDetector(
            interpreter=object(),
            class_names=["Speech", "Screaming"],
        )
        with (
            patch("detection.aggression.analyze_tone_float32", side_effect=tone),
            patch("detection.aggression.classify_emotion", return_value="angry"),
            patch("detection.aggression.scan_audio_float32", side_effect=yamnet),
            patch(
                "detection.aggression.get_waveform_snapshot_float32",
                side_effect=waveform,
            ),
        ):
            alert = detector.process_with_audio(_stt_result(), event)

        self.assertIs(observed["tone"], event.samples)
        self.assertIs(observed["yamnet"], event.samples)
        self.assertIs(observed["waveform"], event.samples)
        np.testing.assert_array_equal(observed["tone"], observed["yamnet"])
        self.assertEqual(observed["tone_rate"], event.sample_rate)
        self.assertEqual(observed["yamnet_rate"], event.sample_rate)
        self.assertEqual(alert["event_id"], event.event_id)
        self.assertEqual(alert["duration"], 1.25)
        self.assertNotEqual(alert["duration"], 2.0)
        self.assertTrue(alert["yamnet_ran"])
        self.assertEqual(alert["yamnet_class"], "Screaming")
        self.assertEqual(alert["yamnet_score"], 0.82)
        self.assertEqual(alert["waveform_snapshot"], [123] * 40)

    def test_track_b_uses_real_event_duration_waveform_and_not_run_evidence(self):
        event = AudioEvent.from_float32_samples(
            np.full(28000, 0.05, dtype=np.float32)
        )
        observed = {}

        def waveform(samples):
            observed["waveform"] = samples
            return [45] * 40

        detector = AggressionDetector(interpreter=None, class_names=[])
        with (
            patch(
                "detection.aggression.analyze_tone_float32",
                return_value={
                    **_tone_result(),
                    "rms": 30.0,
                    "is_aggressive_tone": False,
                },
            ),
            patch("detection.aggression.classify_emotion", return_value="neutral"),
            patch(
                "detection.aggression.get_waveform_snapshot_float32",
                side_effect=waveform,
            ),
        ):
            alert = detector.process_with_audio(_stt_result("bobo"), event)

        self.assertIs(observed["waveform"], event.samples)
        self.assertEqual(alert["duration"], 1.75)
        self.assertNotEqual(alert["duration"], alert["required_duration"])
        self.assertEqual(alert["event_id"], event.event_id)
        self.assertFalse(alert["yamnet_ran"])
        self.assertEqual(alert["yamnet_class"], "NotRun")
        self.assertEqual(alert["yamnet_score"], 0.0)
        self.assertEqual(alert["waveform_snapshot"], [45] * 40)

    def test_track_b_preserves_successful_non_aggressive_yamnet_result(self):
        event = AudioEvent.from_float32_samples(
            np.full(28000, 0.05, dtype=np.float32)
        )
        detector = AggressionDetector(
            interpreter=object(),
            class_names=["Speech", "Screaming"],
        )
        with (
            patch(
                "detection.aggression.analyze_tone_float32",
                return_value={
                    **_tone_result(),
                    "rms": 30.0,
                    "is_aggressive_tone": False,
                },
            ),
            patch("detection.aggression.classify_emotion", return_value="neutral"),
            patch(
                "detection.aggression.scan_audio_float32",
                return_value=("Speech", 0.91),
            ),
        ):
            alert = detector.process_with_audio(_stt_result("bobo"), event)

        self.assertTrue(alert["yamnet_ran"])
        self.assertEqual(alert["yamnet_class"], "Speech")
        self.assertEqual(alert["yamnet_score"], 0.91)

    def test_appearance_path_marks_yamnet_as_not_run(self):
        event = AudioEvent.from_float32_samples(
            np.full(24000, 0.2, dtype=np.float32)
        )
        detector = AggressionDetector(
            interpreter=object(),
            class_names=["Speech", "Screaming"],
        )
        with (
            patch(
                "detection.aggression.analyze_tone_float32",
                return_value=_tone_result(),
            ),
            patch("detection.aggression.classify_emotion", return_value="angry"),
        ):
            alert = detector.process_with_audio(_stt_result("pangit"), event)

        self.assertIsNotNone(alert)
        self.assertFalse(alert["yamnet_ran"])
        self.assertEqual(alert["yamnet_class"], "NotRun")
        self.assertEqual(alert["yamnet_score"], 0.0)
        self.assertEqual(alert["duration"], 1.5)

    def test_missing_event_does_not_fabricate_duration(self):
        detector = AggressionDetector(interpreter=None, class_names=[])
        alert = detector.process_with_audio(_stt_result("bobo"), None)
        self.assertIsNone(alert)

    def test_main_has_no_second_microphone_reader_or_ring_start(self):
        source_path = Path(__file__).resolve().parents[1] / "main.py"
        source = source_path.read_text()
        tree = ast.parse(source)

        imported_names = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertNotIn("pyaudio", imported_names)
        self.assertNotIn("_audio_ring_thread", source)
        self.assertNotIn("_AUDIO_RING", source)
        self.assertNotIn("get_audio_snapshot", source)
        self.assertNotIn("pa.open", source)
        self.assertIn(
            'audio_event = result.get("audio_event")',
            source,
        )
        detector_source = (
            Path(__file__).resolve().parents[1]
            / "detection"
            / "aggression.py"
        ).read_text()
        self.assertNotIn("from audio.capture", detector_source)


if __name__ == "__main__":
    unittest.main()
