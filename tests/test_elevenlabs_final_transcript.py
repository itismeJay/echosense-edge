"""ElevenLabs transcripts are final: no Whisper language_retry, no guard drop."""

import contextlib
import io
import unittest
from unittest.mock import Mock, patch

import numpy as np

from audio.audio_event import AudioEvent
from model import whisper_stt
from model.stt_provider import _elevenlabs_language

ELEVENLABS_META = {"stt": {"provider": "elevenlabs", "fallback_reason": None,
                           "latency_ms": 300, "audio_tags": []}}


class _Recorder:
    def __init__(self, text, language, metadata):
        self._text, self._language, self._metadata = text, language, metadata
        self.perform_final_transcription = Mock(return_value="whisper text")

    def text(self):
        self.detected_language = self._language
        self.detected_language_probability = 1.0
        self.last_transcription_metadata = self._metadata
        return self._text


class _Adapter:
    def __init__(self):
        self.event = AudioEvent.from_float32_samples(np.zeros(16000, dtype=np.float32))

    def clear_pending(self):
        pass

    def consume_pending(self):
        return self.event


def _cycle(recorder):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        return whisper_stt._run_transcription_cycle(recorder, _Adapter()), out.getvalue()


class ElevenLabsFinalTranscriptTests(unittest.TestCase):
    def tearDown(self):
        whisper_stt.clear_transcription_results()

    def test_elevenlabs_text_skips_whisper_language_retry(self):
        for mode in ("tl", "tl+en"):
            for language in ("ceb", "fil", "tl", "en", "jw"):
                recorder = _Recorder("Pangit kaayo ka", language, ELEVENLABS_META)
                with patch.object(whisper_stt, "WHISPER_LANGUAGE", mode):
                    (text, _event, lang, _p, meta), log = _cycle(recorder)
                recorder.perform_final_transcription.assert_not_called()
                self.assertEqual(text, "Pangit kaayo ka")
                self.assertEqual(lang, language)
                self.assertEqual(meta["stt"]["provider"], "elevenlabs")
                self.assertNotIn("LANGUAGE_GUARD", log)

    def test_whisper_text_still_gets_language_retry(self):
        recorder = _Recorder("Pangit kaayo ka", "jw", {"avg_logprob": -0.2})
        recorder.detected_language = "tl"
        (text, *_), log = _cycle(recorder)
        recorder.perform_final_transcription.assert_called_once()
        self.assertIn("[LANGUAGE_GUARD] unsupported=jw", log)

    def test_final_transcript_and_detection_use_elevenlabs_text(self):
        out = io.StringIO()
        with patch.object(whisper_stt, "SHOW_TRANSCRIPT_TEXT", True), \
                contextlib.redirect_stdout(out):
            whisper_stt._on_text(
                "Pangit kaayo ka", whisper_language="ceb",
                whisper_language_confidence=1.0,
                audio_event=_Adapter().event, decode_metadata=ELEVENLABS_META,
            )
        result = whisper_stt.consume_transcription_result()
        self.assertNotIn("[LANGUAGE_GUARD] rejected", out.getvalue())
        self.assertIn('text="Pangit kaayo ka"', out.getvalue())
        self.assertEqual(result["transcribed_text"], "Pangit kaayo ka")

    def test_late_retry_transcript_keeps_original_audio_event(self):
        original = _Adapter().event
        with contextlib.redirect_stdout(io.StringIO()):
            whisper_stt._deliver_late("Pangit kaayo ka", "ceb", 1.0, original, ELEVENLABS_META)
        result = whisper_stt.consume_transcription_result()
        self.assertEqual(result["transcribed_text"], "Pangit kaayo ka")
        self.assertIs(result["audio_event"], original)

    def test_elevenlabs_language_codes_are_normalized(self):
        self.assertEqual(_elevenlabs_language("ceb", "ceb"), "ceb")
        self.assertEqual(_elevenlabs_language("eng", "ceb"), "en")
        self.assertEqual(_elevenlabs_language("tgl", "ceb"), "tl")
        self.assertEqual(_elevenlabs_language("fil", "ceb"), "fil")
        self.assertEqual(_elevenlabs_language("jav", "ceb"), "ceb")
        self.assertEqual(_elevenlabs_language(None, "ceb"), "ceb")


if __name__ == "__main__":
    unittest.main()
