import unittest
from unittest.mock import Mock, patch

import numpy as np

from audio.audio_event import AudioEvent
from detection.aggression import AggressionDetector
from main import process_transcription_result
from model import whisper_stt


def _tone_result(*, aggressive=True):
    return {
        "rms": 900.0 if aggressive else 30.0,
        "energy_variance": 9000.0 if aggressive else 10.0,
        "zero_crossing_rate": 0.2 if aggressive else 0.01,
        "peak_to_average": 2.0,
        "is_aggressive_tone": aggressive,
    }


def _event(sample_count=24000):
    return AudioEvent.from_float32_samples(
        np.linspace(-0.25, 0.25, sample_count, dtype=np.float32)
    )


def _stt_result(text, event):
    whisper_stt._on_text(
        text,
        whisper_language="tl",
        whisper_language_confidence=0.91,
        audio_event=event,
    )
    with whisper_stt._result_lock:
        result = whisper_stt._latest_result
        whisper_stt._latest_result = None
    whisper_stt._new_result_event.clear()
    return result


def _alert_for(result):
    event = result["audio_event"]
    return {
        "should_alert": True,
        "severity": "medium",
        "confidence": 0.81,
        "duration": event.duration_ms / 1000.0,
        "transcribed_text": result["transcribed_text"],
        "event_id": event.event_id,
        "detected_words": result.get("detected_words", []),
        "matched_terms": result.get("matched_terms", []),
        "yamnet_class": "NotRun",
        "yamnet_score": 0.0,
        "yamnet_ran": False,
    }


class TextPipelineHardeningTests(unittest.TestCase):
    def test_28_rejected_transcript_does_not_invoke_yamnet(self):
        event = _event()
        result = _stt_result("bobo bobo bobo bobo bobo bobo", event)
        detector = AggressionDetector(
            interpreter=object(),
            class_names=["Speech", "Screaming"],
        )
        with (
            patch("detection.aggression.analyze_tone_float32") as tone,
            patch("detection.aggression.scan_audio_float32") as yamnet,
        ):
            alert = detector.process_with_audio(result, event)

        self.assertIsNone(alert)
        tone.assert_not_called()
        yamnet.assert_not_called()

    def test_29_context_suppressed_transcript_does_not_invoke_yamnet(self):
        event = _event()
        result = _stt_result("Pangit ang drawing.", event)
        detector = AggressionDetector(
            interpreter=object(),
            class_names=["Speech", "Screaming"],
        )
        with (
            patch("detection.aggression.analyze_tone_float32") as tone,
            patch("detection.aggression.scan_audio_float32") as yamnet,
        ):
            alert = detector.process_with_audio(result, event)

        self.assertIsNone(alert)
        tone.assert_not_called()
        yamnet.assert_not_called()

    def test_30_accepted_monitored_phrase_preserves_yamnet_execution(self):
        event = _event()
        result = _stt_result("Yawa ka.", event)
        detector = AggressionDetector(
            interpreter=object(),
            class_names=["Speech", "Screaming"],
        )
        with (
            patch(
                "detection.aggression.analyze_tone_float32",
                return_value=_tone_result(),
            ),
            patch(
                "detection.aggression.classify_emotion",
                return_value="angry",
            ),
            patch(
                "detection.aggression.scan_audio_float32",
                return_value=("Screaming", 0.82),
            ) as yamnet,
        ):
            alert = detector.process_with_audio(result, event)

        yamnet.assert_called_once()
        self.assertTrue(alert["yamnet_ran"])
        self.assertEqual(alert["yamnet_class"], "Screaming")

    def test_31_same_event_id_is_preserved(self):
        event = _event()
        result = _stt_result("Bobo ka.", event)
        detector = AggressionDetector(interpreter=None, class_names=[])
        with (
            patch(
                "detection.aggression.analyze_tone_float32",
                return_value=_tone_result(aggressive=False),
            ),
            patch(
                "detection.aggression.classify_emotion",
                return_value="neutral",
            ),
        ):
            alert = detector.process_with_audio(result, event)

        self.assertEqual(result["event_id"], event.event_id)
        self.assertEqual(alert["event_id"], event.event_id)

    def test_32_same_audio_event_samples_feed_tone_yamnet_and_waveform(self):
        event = _event()
        result = _stt_result("Yawa ka.", event)
        observed = {}

        def tone(samples, sample_rate):
            observed["tone"] = samples
            return _tone_result()

        def yamnet(samples, sample_rate, interpreter, class_names):
            observed["yamnet"] = samples
            return "Screaming", 0.82

        def waveform(samples):
            observed["waveform"] = samples
            return [1, 2, 3]

        detector = AggressionDetector(
            interpreter=object(),
            class_names=["Speech", "Screaming"],
        )
        with (
            patch("detection.aggression.analyze_tone_float32", side_effect=tone),
            patch(
                "detection.aggression.classify_emotion",
                return_value="angry",
            ),
            patch("detection.aggression.scan_audio_float32", side_effect=yamnet),
            patch(
                "detection.aggression.get_waveform_snapshot_float32",
                side_effect=waveform,
            ),
        ):
            detector.process_with_audio(result, event)

        self.assertIs(observed["tone"], event.samples)
        self.assertIs(observed["yamnet"], event.samples)
        self.assertIs(observed["waveform"], event.samples)

    def test_33_duration_remains_sample_derived(self):
        event = _event(sample_count=28000)
        result = _stt_result("Bobo ka.", event)
        detector = AggressionDetector(interpreter=None, class_names=[])
        with (
            patch(
                "detection.aggression.analyze_tone_float32",
                return_value=_tone_result(aggressive=False),
            ),
            patch(
                "detection.aggression.classify_emotion",
                return_value="neutral",
            ),
        ):
            alert = detector.process_with_audio(result, event)

        self.assertEqual(event.duration_ms, 1750)
        self.assertEqual(alert["duration"], 1.75)
        self.assertNotEqual(alert["duration"], alert["required_duration"])

    def test_34_yamnet_not_run_remains_explicit_and_zero(self):
        event = _event()
        result = _stt_result("Bobo ka.", event)
        detector = AggressionDetector(interpreter=None, class_names=[])
        with (
            patch(
                "detection.aggression.analyze_tone_float32",
                return_value=_tone_result(aggressive=False),
            ),
            patch(
                "detection.aggression.classify_emotion",
                return_value="neutral",
            ),
        ):
            alert = detector.process_with_audio(result, event)

        self.assertFalse(alert["yamnet_ran"])
        self.assertEqual(alert["yamnet_class"], "NotRun")
        self.assertEqual(alert["yamnet_score"], 0.0)

    def test_35_rejected_quality_sends_no_alert_payload(self):
        result = _stt_result(
            "bobo bobo bobo bobo bobo bobo",
            _event(),
        )
        detector = Mock()
        led = Mock()
        sender = Mock()

        sent = process_transcription_result(result, detector, led, sender)

        self.assertFalse(sent)
        detector.process_with_audio.assert_not_called()
        led.alert.assert_not_called()
        sender.assert_not_called()

    def test_36_fully_suppressed_context_sends_no_alert_payload(self):
        result = _stt_result("Pangit ang internet.", _event())
        detector = Mock()
        led = Mock()
        sender = Mock()

        sent = process_transcription_result(result, detector, led, sender)

        self.assertFalse(sent)
        detector.process_with_audio.assert_not_called()
        led.alert.assert_not_called()
        sender.assert_not_called()

    def test_37_valid_result_sends_one_payload(self):
        result = _stt_result("Bobo ka.", _event())
        detector = Mock()
        detector.process_with_audio.return_value = _alert_for(result)
        led = Mock()
        sender = Mock(return_value=True)

        sent = process_transcription_result(result, detector, led, sender)

        self.assertTrue(sent)
        detector.process_with_audio.assert_called_once_with(
            result,
            result["audio_event"],
        )
        led.alert.assert_called_once()
        sender.assert_called_once()

    def test_38_original_transcript_reaches_payload_unchanged(self):
        original = "Bobo KA!\n"
        result = _stt_result(original, _event())
        detector = Mock()
        detector.process_with_audio.return_value = _alert_for(result)
        sender = Mock(return_value=True)

        process_transcription_result(result, detector, Mock(), sender)

        self.assertEqual(
            sender.call_args.kwargs["transcribed_text"],
            original,
        )

    def test_39_one_audio_event_is_passed_to_one_decision_call(self):
        event = _event()
        result = _stt_result("Bobo ka.", event)
        detector = Mock()
        detector.process_with_audio.return_value = None

        process_transcription_result(result, detector, Mock(), Mock())

        detector.process_with_audio.assert_called_once_with(result, event)

    def test_40_pipeline_test_never_makes_a_real_backend_request(self):
        result = _stt_result("Bobo ka.", _event())
        detector = Mock()
        detector.process_with_audio.return_value = _alert_for(result)
        sender = Mock(return_value=True)

        with patch("sender.http_client.requests.post") as post:
            process_transcription_result(result, detector, Mock(), sender)

        post.assert_not_called()
        sender.assert_called_once()


if __name__ == "__main__":
    unittest.main()
