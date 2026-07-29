import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np

from audio.audio_event import AudioEvent
from config import (
    parse_boolean,
    show_transcript_text_from_environment,
)
from main import should_ship_runtime_log
from model import whisper_stt
from model.realtimestt_audio_adapter import RealtimeSTTAudioEventAdapter


def _event():
    return AudioEvent.from_float32_samples(
        np.linspace(-0.1, 0.1, 16000, dtype=np.float32)
    )


class TranscriptVisibilityTests(unittest.TestCase):
    def tearDown(self):
        with whisper_stt._result_lock:
            whisper_stt._latest_result = None
        whisper_stt._new_result_event.clear()

    def _render(
        self,
        text,
        *,
        enabled=True,
        language="en",
        confidence=0.91,
    ):
        event = _event()
        output = io.StringIO()
        with (
            patch.object(
                whisper_stt,
                "SHOW_TRANSCRIPT_TEXT",
                enabled,
            ),
            redirect_stdout(output),
        ):
            whisper_stt._on_text(
                text,
                whisper_language=language,
                whisper_language_confidence=confidence,
                audio_event=event,
            )
        with whisper_stt._result_lock:
            result = whisper_stt._latest_result
            whisper_stt._latest_result = None
        whisper_stt._new_result_event.clear()
        return event, result, output.getvalue()

    def _transcript_lines(self, output):
        return [
            line
            for line in output.splitlines()
            if line.startswith("[TRANSCRIPT]")
        ]

    def test_01_default_configuration_is_false(self):
        self.assertFalse(show_transcript_text_from_environment({}))

    def test_02_true_configuration_values_are_parsed(self):
        for value in ("1", "true", "TRUE", " yes ", "on"):
            with self.subTest(value=value):
                self.assertTrue(parse_boolean(value))

    def test_03_false_configuration_values_are_parsed(self):
        for value in (None, "", "0", "false", "FALSE", " no ", "off"):
            with self.subTest(value=value):
                self.assertFalse(parse_boolean(value))
        self.assertFalse(parse_boolean("unexpected"))

    def test_04_exact_text_is_absent_when_disabled(self):
        _event_value, _result, output = self._render(
            "Private harmless classroom sentence.",
            enabled=False,
        )
        self.assertNotIn("[TRANSCRIPT]", output)
        self.assertNotIn("Private harmless classroom sentence.", output)

    def test_05_exact_text_appears_when_enabled(self):
        _event_value, _result, output = self._render(
            "Good morning, class."
        )
        self.assertIn('text="Good morning, class."', output)

    def test_06_transcript_appears_exactly_once(self):
        _event_value, _result, output = self._render(
            "One finalized sentence."
        )
        self.assertEqual(len(self._transcript_lines(output)), 1)

    def test_07_event_id_matches_audio_event(self):
        event, _result, output = self._render("Event correlation test.")
        line = self._transcript_lines(output)[0]
        self.assertIn(f"event={event.event_id}", line)

    def test_08_language_is_included(self):
        _event_value, _result, output = self._render(
            "Maayong buntag.",
            language="ceb",
        )
        self.assertIn("language=ceb", self._transcript_lines(output)[0])

    def test_09_capitalization_is_preserved(self):
        text = "Good Morning, DNSC."
        _event_value, result, output = self._render(text)
        self.assertIn(json.dumps(text), output)
        self.assertEqual(result["transcribed_text"], text)

    def test_10_punctuation_is_preserved(self):
        text = "Ready, class—please listen!"
        _event_value, _result, output = self._render(text)
        self.assertIn(json.dumps(text, ensure_ascii=False), output)

    def test_11_filipino_text_is_preserved(self):
        text = "Magandang umaga. Pagsubok ito ng mikropono."
        _event_value, result, output = self._render(text, language="tl")
        self.assertIn(json.dumps(text, ensure_ascii=False), output)
        self.assertEqual(result["transcribed_text"], text)

    def test_12_cebuano_text_is_preserved(self):
        text = "Maayong buntag, pagsulay kini sa mikropono."
        _event_value, result, output = self._render(text, language="ceb")
        self.assertIn(json.dumps(text, ensure_ascii=False), output)
        self.assertEqual(result["transcribed_text"], text)

    def test_13_mixed_language_text_is_preserved(self):
        text = "Good morning, maghanda na para sa lesson."
        output = io.StringIO()
        with (
            patch.object(whisper_stt, "SHOW_TRANSCRIPT_TEXT", True),
            redirect_stdout(output),
        ):
            whisper_stt._log_finalized_transcript(
                "synthetic-event",
                "mixed",
                text,
            )
        self.assertIn(json.dumps(text, ensure_ascii=False), output.getvalue())
        self.assertIn("language=mixed", output.getvalue())

    def test_14_apostrophes_are_preserved(self):
        text = "Please don't close the teacher's notebook."
        _event_value, _result, output = self._render(text)
        self.assertIn(json.dumps(text), output)

    def test_15_quotes_are_safely_represented(self):
        text = 'The teacher said "good morning".'
        _event_value, _result, output = self._render(text)
        line = self._transcript_lines(output)[0]
        self.assertIn('\\"good morning\\"', line)
        self.assertEqual(json.loads(line.split(" text=", 1)[1]), text)

    def test_16_newlines_are_escaped(self):
        text = "First line.\nSecond line."
        _event_value, _result, output = self._render(text)
        line = self._transcript_lines(output)[0]
        self.assertIn("\\n", line)
        self.assertNotIn("\nSecond line.", line)

    def test_17_carriage_returns_are_escaped(self):
        text = "First part.\rSecond part."
        _event_value, _result, output = self._render(text)
        line = self._transcript_lines(output)[0]
        self.assertIn("\\r", line)
        self.assertNotIn("\rSecond part.", line)

    def test_18_tabs_are_escaped(self):
        text = "First\tsecond."
        _event_value, _result, output = self._render(text)
        line = self._transcript_lines(output)[0]
        self.assertIn("\\t", line)
        self.assertNotIn("\tsecond.", line)

    def test_19_empty_transcript_does_not_crash(self):
        _event_value, result, output = self._render("")
        self.assertIn('text=""', output)
        self.assertFalse(result["quality_accepted"])

    def test_20_raw_samples_are_not_printed(self):
        event, _result, output = self._render("Sample privacy test.")
        self.assertNotIn(str(event.samples), output)
        self.assertNotIn("array(", output)
        self.assertNotIn("waveform", output.lower())

    def test_21_recorder_text_remains_called_once(self):
        adapter = RealtimeSTTAudioEventAdapter()

        class Recorder:
            detected_language = "en"
            detected_language_probability = 0.9

            def __init__(self):
                self.text_calls = 0

            def text(self):
                self.text_calls += 1
                adapter.on_transcription_start(
                    np.ones(8000, dtype=np.float32)
                )
                return "One finalized transcript."

        recorder = Recorder()
        text, event, language, confidence = (
            whisper_stt._run_transcription_cycle(recorder, adapter)
        )
        self.assertEqual(recorder.text_calls, 1)
        self.assertEqual(text, "One finalized transcript.")
        self.assertIsNotNone(event)
        self.assertEqual(language, "en")
        self.assertEqual(confidence, 0.9)

    def test_22_display_setting_does_not_change_detection_result(self):
        _event_off, result_off, _output_off = self._render(
            "Pangit ang panahon.",
            enabled=False,
            language="tl",
        )
        _event_on, result_on, _output_on = self._render(
            "Pangit ang panahon.",
            enabled=True,
            language="tl",
        )
        comparable_fields = (
            "quality_accepted",
            "quality_reason_codes",
            "detected_words",
            "matched_terms",
            "match_candidate_count",
            "match_accepted_count",
            "match_suppressed_count",
            "context_suppressed_all",
        )
        for field in comparable_fields:
            with self.subTest(field=field):
                self.assertEqual(result_off[field], result_on[field])

    def test_23_transcript_diagnostic_is_not_shipped_as_remote_log(self):
        self.assertFalse(
            should_ship_runtime_log(
                '[TRANSCRIPT] event=synthetic language=en text="Harmless."'
            )
        )
        self.assertTrue(
            should_ship_runtime_log(
                "[TEXT_QUALITY] event=synthetic accepted=true tokens=1"
            )
        )


if __name__ == "__main__":
    unittest.main()
