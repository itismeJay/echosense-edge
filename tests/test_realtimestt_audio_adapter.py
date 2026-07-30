import inspect
import unittest
from unittest.mock import patch

import numpy as np
from RealtimeSTT.core.transcription_api import transcribe as realtimestt_transcribe

from model.realtimestt_audio_adapter import RealtimeSTTAudioEventAdapter
from model import whisper_stt
from model.whisper_stt import _run_transcription_cycle


class _FakeRecorder:
    def __init__(self, adapter, samples=None, error=None):
        self.adapter = adapter
        self.samples = samples
        self.error = error
        self.text_calls = 0
        self.detected_language = None
        self.detected_language_probability = None

    def text(self):
        self.text_calls += 1
        if self.error is not None:
            raise self.error
        if self.samples is not None:
            self.adapter.on_transcription_start(self.samples)
        self.detected_language = "en"
        self.detected_language_probability = 0.91
        return "Completed transcript."


class RealtimeSTTAudioAdapterTests(unittest.TestCase):
    def setUp(self):
        self.adapter = RealtimeSTTAudioEventAdapter()

    def test_callback_creates_copied_event_and_returns_falsy(self):
        original = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)

        callback_result = self.adapter.on_transcription_start(original)
        event = self.adapter.consume_pending()
        original[:] = 0.0

        self.assertFalse(callback_result)
        self.assertIsNotNone(event)
        self.assertFalse(np.all(event.samples == 0.0))
        self.assertEqual(event.duration_ms, 1000)

    def test_callback_handles_malformed_audio_safely(self):
        self.adapter.on_transcription_start(
            np.ones(16000, dtype=np.float32)
        )

        callback_result = self.adapter.on_transcription_start(None)

        self.assertFalse(callback_result)
        self.assertIsNone(self.adapter.consume_pending())

        callback_result = self.adapter.on_transcription_start(
            np.ones(100, dtype=np.int16)
        )
        self.assertFalse(callback_result)
        self.assertIsNone(self.adapter.consume_pending())

    def test_event_is_associated_with_one_text_result_and_consumed_once(self):
        samples = np.full(8000, 0.25, dtype=np.float32)
        recorder = _FakeRecorder(self.adapter, samples=samples)

        text, event, language, probability = _run_transcription_cycle(
            recorder,
            self.adapter,
        )

        self.assertEqual(text, "Completed transcript.")
        self.assertEqual(language, "en")
        self.assertEqual(probability, 0.91)
        np.testing.assert_array_equal(event.samples, samples)
        self.assertIsNone(self.adapter.consume_pending())
        self.assertEqual(recorder.text_calls, 1)

    def test_pending_event_is_cleared_before_each_cycle(self):
        self.adapter.on_transcription_start(
            np.ones(4000, dtype=np.float32)
        )
        recorder = _FakeRecorder(self.adapter, samples=None)

        _text, event, _language, _probability = _run_transcription_cycle(
            recorder,
            self.adapter,
        )

        self.assertIsNone(event)
        self.assertEqual(recorder.text_calls, 1)

    def test_failed_text_call_clears_pending_event(self):
        self.adapter.on_transcription_start(
            np.ones(4000, dtype=np.float32)
        )
        recorder = _FakeRecorder(
            self.adapter,
            error=RuntimeError("synthetic transcription failure"),
        )

        with self.assertRaises(RuntimeError):
            _run_transcription_cycle(recorder, self.adapter)

        self.assertIsNone(self.adapter.consume_pending())
        self.assertEqual(recorder.text_calls, 1)

    def test_adapter_is_isolated_from_recorder_internal_attributes(self):
        source = inspect.getsource(RealtimeSTTAudioEventAdapter)
        self.assertNotIn("last_transcription_bytes", source)
        self.assertNotIn("last_transcription_bytes_b64", source)
        self.assertNotIn("recorder.audio", source)

    def test_audio_event_is_attached_to_the_matching_project_result(self):
        samples = np.ones(16000, dtype=np.float32)
        recorder = _FakeRecorder(self.adapter, samples=samples)
        text, event, language, probability = _run_transcription_cycle(
            recorder,
            self.adapter,
        )
        checked = {
            "checked_text": text.lower(),
            "has_profanity": False,
            "hard_hits": [],
            "soft_hits": [],
            "detected_words": [],
            "categories": [],
            "severity": "low",
            "is_casual": False,
            "matched_terms": [],
        }

        with (
            patch(
                "model.whisper_stt.check_transcript",
                return_value=checked,
            ),
            patch(
                "model.whisper_stt.classify_transcript_language",
                return_value=("en", probability),
            ),
        ):
            whisper_stt._on_text(text, language, probability, event)

        with whisper_stt._result_lock:
            result = whisper_stt._latest_result
            whisper_stt._latest_result = None
        whisper_stt._new_result_event.clear()

        self.assertIs(result["audio_event"], event)
        self.assertEqual(result["event_id"], event.event_id)

    def test_whisper_wrapper_registers_only_the_adapter_callback(self):
        source = inspect.getsource(whisper_stt._recorder_loop)
        self.assertIn(
            "on_transcription_start=_audio_event_adapter.on_transcription_start",
            source,
        )
        self.assertEqual(source.count("_recorder.text()"), 0)
        cycle_source = inspect.getsource(whisper_stt._run_transcription_cycle)
        self.assertEqual(cycle_source.count("recorder.text()"), 1)

    def test_installed_realtimestt_callback_contract(self):
        source = inspect.getsource(realtimestt_transcribe)
        self.assertIn("audio_copy = copy.deepcopy(recorder.audio)", source)
        self.assertIn(
            "recorder.on_transcription_start(audio_copy)",
            source,
        )
        self.assertIn(
            "recorder.perform_final_transcription(audio_copy)",
            source,
        )

    def test_unsupported_auto_language_retries_same_event_as_tagalog(self):
        samples = np.full(16000, 0.2, dtype=np.float32)

        class Recorder:
            language = ""
            detected_language = None
            detected_language_probability = None

            def __init__(self, adapter):
                self.adapter = adapter
                self.text_calls = 0
                self.retry_calls = 0
                self.retry_audio = None

            def text(self):
                self.text_calls += 1
                self.adapter.on_transcription_start(samples)
                self.detected_language = "ru"
                self.detected_language_probability = 0.81
                return "Бабой."

            def perform_final_transcription(self, audio, use_prompt=True):
                self.retry_calls += 1
                self.retry_audio = audio
                self.detected_language = self.language
                self.detected_language_probability = 1.0
                return "Baboy."

        recorder = Recorder(self.adapter)
        text, event, language, probability = _run_transcription_cycle(
            recorder,
            self.adapter,
        )

        self.assertEqual(text, "Baboy.")
        self.assertEqual(language, "tl")
        self.assertEqual(probability, 1.0)
        self.assertEqual(recorder.text_calls, 1)
        self.assertEqual(recorder.retry_calls, 1)
        self.assertEqual(recorder.language, "")
        self.assertIs(recorder.retry_audio, event.samples)

    def test_retry_result_with_unsupported_script_is_rejected(self):
        samples = np.full(8000, 0.2, dtype=np.float32)

        class Recorder:
            language = ""
            detected_language = "ko"
            detected_language_probability = 0.8

            def text(self):
                self_adapter.on_transcription_start(samples)
                return "방일까?"

            def perform_final_transcription(self, audio, use_prompt=True):
                self.detected_language = "tl"
                self.detected_language_probability = 1.0
                return "방일까?"

        self_adapter = self.adapter
        text, event, language, probability = _run_transcription_cycle(
            Recorder(),
            self.adapter,
        )

        self.assertEqual(text, "")
        self.assertIsNotNone(event)
        self.assertIsNone(language)
        self.assertIsNone(probability)


if __name__ == "__main__":
    unittest.main()
