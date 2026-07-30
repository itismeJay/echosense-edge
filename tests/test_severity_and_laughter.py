import unittest
from unittest.mock import patch

import numpy as np

from audio.audio_event import AudioEvent
from detection.aggression import AggressionDetector
from detection.severity import HIGH, LOW, MEDIUM, calculate_severity
from model import whisper_stt
from model.blacklist import check_transcript
from sender.http_client import build_alert_payload


def _event(seconds=1.5):
    return AudioEvent.from_float32_samples(
        np.full(round(16000 * seconds), 0.05, dtype=np.float32)
    )


def _result(text, event, *, whisper_language="en"):
    whisper_stt._on_text(
        text,
        whisper_language=whisper_language,
        whisper_language_confidence=0.91,
        audio_event=event,
    )
    with whisper_stt._result_lock:
        result = whisper_stt._latest_result
        whisper_stt._latest_result = None
    whisper_stt._new_result_event.clear()
    return result


def _neutral_tone():
    return {
        "rms": 30.0,
        "energy_variance": 10.0,
        "zero_crossing_rate": 0.01,
        "peak_to_average": 2.0,
        "is_aggressive_tone": False,
    }


class SeverityAndLaughterRegressionTests(unittest.TestCase):
    def _text_alert(self, text, *, language="en"):
        event = _event()
        result = _result(text, event, whisper_language=language)
        detector = AggressionDetector(interpreter=None, class_names=[])
        with (
            patch(
                "detection.aggression.analyze_tone_float32",
                return_value=_neutral_tone(),
            ),
            patch(
                "detection.aggression.classify_emotion",
                return_value="neutral",
            ),
        ):
            alert = detector.process_with_audio(result, event)
        return result, alert

    def test_severe_phrase_without_laughter_alerts_high(self):
        result, alert = self._text_alert("Kill yourself.")

        self.assertEqual(result["severity"], HIGH)
        self.assertIsNotNone(alert)
        self.assertEqual(alert["severity"], HIGH)
        self.assertIn(
            "term_category:self_harm_directive",
            alert["severity_evidence"]["reasons"],
        )

    def test_severe_phrase_with_haha_alerts_high(self):
        text = "Kill yourself haha."
        result, alert = self._text_alert(text)

        self.assertTrue(result["is_casual"])
        self.assertIsNotNone(alert)
        self.assertEqual(alert["severity"], HIGH)
        self.assertEqual(alert["transcribed_text"], text)
        self.assertIn(
            "laughter_or_excitement_marker_present",
            alert["severity_evidence"]["supporting_evidence"],
        )

    def test_severe_phrase_survives_hehe_and_excitement_markers(self):
        for text in ("Kill yourself hehe.", "Kill yourself charot."):
            with self.subTest(text=text):
                result, alert = self._text_alert(text)

                self.assertTrue(result["is_casual"])
                self.assertIsNotNone(alert)
                self.assertEqual(alert["severity"], HIGH)
                self.assertEqual(alert["transcribed_text"], text)

    def test_severe_phrase_with_laughter_acoustic_evidence_alerts_high(self):
        event = _event()
        result = _result("Kill yourself.", event)
        detector = AggressionDetector(
            interpreter=object(),
            class_names=["Speech", "Laughter"],
        )
        with (
            patch(
                "detection.aggression.analyze_tone_float32",
                return_value=_neutral_tone(),
            ),
            patch(
                "detection.aggression.classify_emotion",
                return_value="neutral",
            ),
            patch(
                "detection.aggression.scan_audio_float32",
                return_value=("Laughter", 0.92),
            ),
        ):
            alert = detector.process_with_audio(result, event)

        self.assertIsNotNone(alert)
        self.assertEqual(alert["severity"], HIGH)
        self.assertTrue(alert["yamnet_ran"])
        self.assertEqual(alert["yamnet_class"], "Laughter")

    def test_harmless_laughter_only_creates_no_alert(self):
        event = _event()
        result = _result("Haha hehe.", event)
        alert = AggressionDetector().process_with_audio(result, event)

        self.assertFalse(result["has_profanity"])
        self.assertIsNone(alert)

    def test_harmless_sentence_without_monitored_term_creates_no_alert(self):
        event = _event()
        text = "Please open your science book."
        result = _result(text, event)
        alert = AggressionDetector().process_with_audio(result, event)

        self.assertEqual(result["transcribed_text"], text)
        self.assertFalse(result["has_profanity"])
        self.assertIsNone(alert)

    def test_low_risk_monitored_term_is_low(self):
        result = check_transcript("Pikon.")

        self.assertTrue(result["has_profanity"])
        self.assertEqual(result["severity"], LOW)

    def test_direct_insult_is_medium_and_repetition_promotes_high(self):
        direct = check_transcript("Bobo ka.")
        repeated = calculate_severity(
            direct["detected_words"],
            transcript="Bobo ka.",
            repeated=True,
        )

        self.assertEqual(direct["severity"], MEDIUM)
        self.assertEqual(repeated.level, HIGH)
        self.assertIn("promoted_by:repetition", repeated.reasons)

    def test_kill_yourself_is_high(self):
        result = check_transcript("kill yourself")

        self.assertEqual(result["severity"], HIGH)
        self.assertIn(
            "kill yourself",
            result["severity_evidence"]["term_categories"][
                "self_harm_directive"
            ],
        )

    def test_pangit_alert_severity_matches_direct_firing_rule(self):
        result, alert = self._text_alert("Pangit ka.", language="tl")

        self.assertEqual(result["severity"], MEDIUM)
        self.assertIsNotNone(alert)
        self.assertEqual(alert["severity"], MEDIUM)
        self.assertIn(
            "promoted_by:direct_target_pattern",
            alert["severity_evidence"]["reasons"],
        )

    def test_mixed_language_severe_phrase_is_high(self):
        text = "Patyon tika, you are worthless."
        result, alert = self._text_alert(text, language="tl")

        self.assertEqual(result["language"], "mixed")
        self.assertEqual(result["severity"], HIGH)
        self.assertIsNotNone(alert)
        self.assertEqual(alert["severity"], HIGH)
        self.assertEqual(alert["transcribed_text"], text)

    def test_uppercase_internal_severity_converts_only_at_api_boundary(self):
        payload = build_alert_payload(
            severity=HIGH,
            confidence=0.9,
            duration=1.5,
        )

        self.assertEqual(payload["severity"], "high")


if __name__ == "__main__":
    unittest.main()
