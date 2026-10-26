"""False alarms from media audio and Whisper hallucinations.

Each filter is tested alone, then through the real path a finalized
transcript takes: whisper_stt._on_text -> TextDetector -> EventCoordinator.
"""

from collections import namedtuple
from datetime import datetime, timedelta, timezone
from pathlib import Path
import unittest

import numpy as np

from audio.audio_event import AudioEvent
from detection.false_alarm_filters import (
    DecodeMetrics,
    FilterConfig,
    WordScore,
    decode_gate,
    filter_terms_by_probability,
    hallucination_reason,
    media_classes_heard,
    media_gate,
    summarize_segments,
    term_probability,
)
from detection.text_detector import TextDetector
from model import whisper_stt
from tests.test_accuracy_context import CoordinatorHarness

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = FilterConfig()
WALL = datetime(2026, 10, 26, 8, 0, tzinfo=timezone.utc)


def metrics(words=(), *, no_speech=0.05, logprob=-0.3, compression=1.2):
    return DecodeMetrics(
        no_speech_prob=no_speech,
        avg_logprob=logprob,
        compression_ratio=compression,
        words=tuple(WordScore(w, p) for w, p in words),
        segment_count=1,
    )


def metadata(words=(), *, no_speech=0.05, logprob=-0.3, compression=1.2):
    return {
        "no_speech_prob": no_speech,
        "avg_logprob": logprob,
        "compression_ratio": compression,
        "words": [[w, p] for w, p in words],
        "segment_count": 1,
    }


SPEECH = [("Speech", 0.82), ("Conversation", 0.20), ("Narration, monologue", 0.08)]
MUSIC = [("Music", 0.71), ("Speech", 0.33), ("Pop music", 0.12)]
TELEVISION = [("Speech", 0.45), ("Television", 0.30), ("Music", 0.05)]


class DecodeGateTests(unittest.TestCase):
    def test_each_threshold_drops_the_utterance(self):
        self.assertEqual(
            decode_gate(metrics(no_speech=0.61), DEFAULTS), "no_speech_prob>0.6"
        )
        self.assertEqual(
            decode_gate(metrics(logprob=-1.01), DEFAULTS), "avg_logprob<-1"
        )
        self.assertEqual(
            decode_gate(metrics(compression=2.41), DEFAULTS),
            "compression_ratio>2.4",
        )

    def test_values_at_the_limits_are_kept(self):
        self.assertIsNone(
            decode_gate(
                metrics(no_speech=0.6, logprob=-1.0, compression=2.4), DEFAULTS
            )
        )

    def test_missing_metrics_never_drop(self):
        self.assertIsNone(decode_gate(None, DEFAULTS))

    def test_gate_can_be_switched_off(self):
        config = FilterConfig(decode_gate_enabled=False)
        self.assertIsNone(decode_gate(metrics(no_speech=0.99), config))


class HallucinationTests(unittest.TestCase):
    def test_known_filler_phrases_are_dropped(self):
        for text in (
            "Thank you for watching!",
            "Thanks for watching.",
            "Subscribe.",
            "Please subscribe. Thank you.",
            "Subtitles by the Amara.org community",
            "Salamat sa panonood!",
        ):
            self.assertEqual(
                hallucination_reason(text, DEFAULTS),
                "known_hallucination_phrase",
                text,
            )

    def test_one_word_repeated_is_dropped(self):
        self.assertEqual(
            hallucination_reason("you you you", DEFAULTS), "repeated_single_word"
        )
        self.assertEqual(
            hallucination_reason("Okay. Okay. Okay. Okay.", DEFAULTS),
            "repeated_single_word",
        )

    def test_real_speech_is_kept_even_next_to_a_filler(self):
        for text in (
            "bobo ka",
            "papatayin kita",
            "Thank you for watching, bobo ka talaga",
            "okay okay",
            "",
        ):
            self.assertIsNone(hallucination_reason(text, DEFAULTS), text)

    def test_filter_can_be_switched_off(self):
        config = FilterConfig(hallucination_enabled=False)
        self.assertIsNone(hallucination_reason("Thank you for watching", config))


class WordProbabilityTests(unittest.TestCase):
    def test_low_probability_word_does_not_count(self):
        kept, dropped, scores = filter_terms_by_probability(
            ["bobo"], metrics([(" bobo", 0.31), (" ka", 0.9)]), DEFAULTS
        )
        self.assertEqual((kept, dropped), ([], ["bobo"]))
        self.assertAlmostEqual(scores["bobo"], 0.31)

    def test_probability_at_the_limit_counts(self):
        kept, dropped, _ = filter_terms_by_probability(
            ["bobo"], metrics([(" Bobo!", 0.5)]), DEFAULTS
        )
        self.assertEqual((kept, dropped), (["bobo"], []))

    def test_threat_phrases_have_a_lower_floor(self):
        # Live clip: "Papatayin kita." decoded correctly at 0.49 / 0.68.
        live = metrics([(" Papatayin", 0.49), (" kita.", 0.68)])
        kept, dropped, _ = filter_terms_by_probability(["papatayin kita"], live, DEFAULTS)
        self.assertEqual((kept, dropped), (["papatayin kita"], []))
        unsure = metrics([(" Papatayin", 0.25), (" kita.", 0.68)])
        kept, dropped, _ = filter_terms_by_probability(["papatayin kita"], unsure, DEFAULTS)
        self.assertEqual((kept, dropped), ([], ["papatayin kita"]))

    def test_prompted_hallucination_is_caught(self):
        # Measured with the old hint prompt: Whisper wrote "bobo ka" at 0.01.
        kept, dropped, _ = filter_terms_by_probability(
            ["bobo"], metrics([(" bobo", 0.01), (" ka.", 0.76)]), DEFAULTS
        )
        self.assertEqual(dropped, ["bobo"])

    def test_a_phrase_takes_its_weakest_word(self):
        words = metrics([(" papatayin", 0.92), (" kita", 0.41)]).words
        self.assertAlmostEqual(term_probability("papatayin kita", words), 0.41)

    def test_slightly_different_spelling_still_aligns(self):
        words = metrics([(" bubo", 0.88)]).words
        self.assertAlmostEqual(term_probability("bobo", words), 0.88)

    def test_unscored_terms_are_kept(self):
        kept, dropped, scores = filter_terms_by_probability(
            ["tanga"], metrics([(" hello", 0.2)]), DEFAULTS
        )
        self.assertEqual((kept, dropped), (["tanga"], []))
        self.assertIsNone(scores["tanga"])
        kept, _, _ = filter_terms_by_probability(["bobo"], None, DEFAULTS)
        self.assertEqual(kept, ["bobo"])

    def test_filter_can_be_switched_off(self):
        config = FilterConfig(word_probability_enabled=False)
        kept, dropped, _ = filter_terms_by_probability(
            ["bobo"], metrics([(" bobo", 0.1)]), config
        )
        self.assertEqual((kept, dropped), (["bobo"], []))


class MediaGateTests(unittest.TestCase):
    def test_music_television_or_radio_suppresses_ordinary_terms(self):
        self.assertEqual(media_gate(MUSIC, ["bobo"], DEFAULTS), "media_playing:Music")
        self.assertEqual(
            media_gate(TELEVISION, ["tanga"], DEFAULTS), "media_playing:Television"
        )
        self.assertEqual(
            media_gate([("Radio", 0.4)], ["bobo"], DEFAULTS), "media_playing:Radio"
        )

    def test_threats_and_self_harm_are_never_suppressed(self):
        self.assertIsNone(media_gate(MUSIC, ["papatayin kita"], DEFAULTS))
        self.assertIsNone(media_gate(MUSIC, ["bobo", "papatayin kita"], DEFAULTS))
        self.assertIsNone(media_gate(MUSIC, ["patay ka na sana"], DEFAULTS))
        self.assertIsNone(media_gate(MUSIC, ["go kill yourself"], DEFAULTS))

    def test_speech_alone_is_not_media(self):
        self.assertIsNone(media_gate(SPEECH, ["bobo"], DEFAULTS))
        self.assertIsNone(media_gate(None, ["bobo"], DEFAULTS))

    def test_weak_or_low_ranked_media_scores_are_ignored(self):
        self.assertEqual(
            media_classes_heard([("Speech", 0.9), ("Music", 0.10)], DEFAULTS), []
        )
        ranked_fourth = [("Speech", 0.9), ("Conversation", 0.5), ("Child speech", 0.4), ("Music", 0.3)]
        self.assertEqual(media_classes_heard(ranked_fourth, DEFAULTS), [])

    def test_gate_can_be_switched_off(self):
        config = FilterConfig(media_gate_enabled=False)
        self.assertIsNone(media_gate(MUSIC, ["bobo"], config))


class ConfigurationTests(unittest.TestCase):
    def test_defaults_match_the_specification(self):
        self.assertEqual(DEFAULTS.no_speech_prob_max, 0.6)
        self.assertEqual(DEFAULTS.avg_logprob_min, -1.0)
        self.assertEqual(DEFAULTS.compression_ratio_max, 2.4)
        self.assertEqual(DEFAULTS.word_probability_min, 0.5)
        self.assertEqual(DEFAULTS.urgent_word_probability_min, 0.3)
        self.assertEqual(DEFAULTS.media_classes, ("Music", "Television", "Radio"))

    def test_every_filter_reads_the_environment(self):
        config = FilterConfig.from_environment({
            "ECHOSENSE_FILTER_NO_SPEECH_PROB_MAX": "0.8",
            "ECHOSENSE_FILTER_AVG_LOGPROB_MIN": "-1.5",
            "ECHOSENSE_FILTER_COMPRESSION_RATIO_MAX": "3",
            "ECHOSENSE_FILTER_WORD_PROBABILITY_MIN": "0.4",
            "ECHOSENSE_FILTER_URGENT_WORD_PROBABILITY_MIN": "0.2",
            "ECHOSENSE_FILTER_MEDIA_CLASSES": "Music, Singing",
            "ECHOSENSE_FILTER_MEDIA_GATE": "false",
            "ECHOSENSE_FILTER_HALLUCINATION": "0",
            "ECHOSENSE_FILTER_DECODE_GATE": "off",
            "ECHOSENSE_FILTER_WORD_PROBABILITY": "no",
        })
        self.assertEqual(config.no_speech_prob_max, 0.8)
        self.assertEqual(config.avg_logprob_min, -1.5)
        self.assertEqual(config.compression_ratio_max, 3.0)
        self.assertEqual(config.word_probability_min, 0.4)
        self.assertEqual(config.urgent_word_probability_min, 0.2)
        self.assertEqual(config.media_classes, ("Music", "Singing"))
        self.assertFalse(config.media_gate_enabled)
        self.assertFalse(config.hallucination_enabled)
        self.assertFalse(config.decode_gate_enabled)
        self.assertFalse(config.word_probability_enabled)
        self.assertEqual(FilterConfig.from_environment({}), DEFAULTS)


Segment = namedtuple(
    "Segment",
    "text start end no_speech_prob avg_logprob compression_ratio words",
)
Word = namedtuple("Word", "word probability")


class DecodeMetricsTests(unittest.TestCase):
    def test_segments_are_weighted_by_duration(self):
        summary = summarize_segments([
            Segment(" a", 0.0, 3.0, 0.1, -0.2, 1.1, [Word(" a", 0.9)]),
            Segment(" b", 3.0, 4.0, 0.9, -1.4, 2.6, [Word(" b", 0.2)]),
        ])
        self.assertAlmostEqual(summary["no_speech_prob"], 0.3)
        self.assertAlmostEqual(summary["avg_logprob"], -0.5)
        self.assertEqual(summary["compression_ratio"], 2.6)
        self.assertEqual(summary["words"], [[" a", 0.9], [" b", 0.2]])
        parsed = DecodeMetrics.from_metadata(summary)
        self.assertEqual(parsed.words[1], WordScore(" b", 0.2))

    def test_bad_metadata_reads_as_unavailable(self):
        self.assertIsNone(DecodeMetrics.from_metadata(None))
        self.assertIsNone(DecodeMetrics.from_metadata({"words": []}))


def audio_event(event_id, seconds=1.0):
    samples = np.zeros(int(16000 * seconds), dtype=np.float32)
    return AudioEvent(
        event_id=event_id,
        started_at=WALL,
        ended_at=WALL + timedelta(seconds=seconds),
        duration_ms=int(seconds * 1000),
        sample_rate=16000,
        samples=samples,
    )


class PipelineTests(unittest.TestCase):
    """The real _on_text -> TextDetector -> EventCoordinator path."""

    def setUp(self):
        self._classifier = whisper_stt._media_classifier
        self._filters = whisper_stt._FILTERS
        whisper_stt._FILTERS = DEFAULTS
        whisper_stt.clear_transcription_results()

    def tearDown(self):
        whisper_stt._media_classifier = self._classifier
        whisper_stt._FILTERS = self._filters
        whisper_stt.clear_transcription_results()

    def run_utterance(self, text, meta, top_classes=SPEECH, language="tl"):
        whisper_stt.set_media_classifier(lambda samples, rate: list(top_classes))
        whisper_stt._on_text(text, language, 0.9, audio_event("evt-" + text[:8]), meta)
        result = whisper_stt.consume_transcription_result(timeout=1.0)
        self.assertIsNotNone(result)
        evidence = TextDetector().analyze_stt_result(
            result, observed_at=WALL, observed_monotonic=1.0
        )
        harness = CoordinatorHarness()
        harness.text(evidence)
        return result, harness.finish(30.0)

    def test_bobo_ka_said_clearly_still_alerts(self):
        result, alerts = self.run_utterance(
            "Bobo ka!", metadata([(" Bobo", 0.93), (" ka!", 0.88)])
        )
        self.assertEqual(result["detected_words"], ["bobo"])
        self.assertEqual(len(alerts), 1)

    def test_threat_still_alerts_with_music_playing(self):
        result, alerts = self.run_utterance(
            "Papatayin kita",
            metadata([(" Papatayin", 0.9), (" kita", 0.86)]),
            top_classes=MUSIC,
        )
        self.assertIn("papatayin kita", result["detected_words"])
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].severity, "HIGH")

    def test_insult_heard_from_a_video_does_not_alert(self):
        result, alerts = self.run_utterance(
            "Bobo ka!",
            metadata([(" Bobo", 0.93), (" ka!", 0.88)]),
            top_classes=MUSIC,
        )
        self.assertEqual(result["detected_words"], [])
        self.assertIn("media_playing:Music", result["filter_reasons"])
        self.assertEqual(alerts, [])

    def test_unsure_word_does_not_alert(self):
        result, alerts = self.run_utterance(
            "Bobo ka", metadata([(" Bobo", 0.22), (" ka", 0.8)])
        )
        self.assertEqual(result["detected_words"], [])
        self.assertIn("low_word_probability", result["filter_reasons"])
        self.assertEqual(alerts, [])

    def test_hallucinated_utterance_from_a_quiet_room_does_not_alert(self):
        result, alerts = self.run_utterance(
            "Bobo ka", metadata([(" Bobo", 0.9), (" ka", 0.9)], no_speech=0.85)
        )
        self.assertEqual(result["transcribed_text"], "")
        self.assertEqual(result["filter_reasons"], ["no_speech_prob>0.6"])
        self.assertEqual(alerts, [])

    def test_threat_heard_clearly_survives_a_low_avg_logprob(self):
        # Measured live: "Papatayin kita!" decoded correctly at avg_logprob -1.02.
        result, alerts = self.run_utterance(
            "Papatayin kita!",
            metadata([(" Papatayin", 0.81), (" kita!", 0.74)], logprob=-1.02),
        )
        self.assertIn("papatayin kita", result["detected_words"])
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].severity, "HIGH")

    def test_ordinary_insult_is_still_dropped_by_the_decode_gate(self):
        result, alerts = self.run_utterance(
            "Bobo ka!", metadata([(" Bobo", 0.95), (" ka!", 0.9)], logprob=-1.3)
        )
        self.assertEqual(result["filter_reasons"], ["avg_logprob<-1"])
        self.assertEqual(alerts, [])

    def test_unsure_threat_words_do_not_override_the_decode_gate(self):
        result, alerts = self.run_utterance(
            "Papatayin kita",
            metadata([(" Papatayin", 0.2), (" kita", 0.1)], no_speech=0.9),
        )
        self.assertEqual(result["filter_reasons"], ["no_speech_prob>0.6"])
        self.assertEqual(alerts, [])

    def test_filler_phrase_is_dropped_before_matching(self):
        result, alerts = self.run_utterance(
            "Thank you for watching!", metadata([(" Thank", 0.9)])
        )
        self.assertEqual(result["filter_reasons"], ["known_hallucination_phrase"])
        self.assertEqual(alerts, [])

    def test_without_decode_metrics_the_word_gated_path_is_unchanged(self):
        result, alerts = self.run_utterance("Bobo ka!", None)
        self.assertEqual(result["detected_words"], ["bobo"])
        self.assertEqual(len(alerts), 1)


class FakeWhisper:
    """Stands in for faster_whisper.WhisperModel inside the real engine class."""

    def __init__(self):
        self.calls = []

    def detect_language(self, audio):
        return "jw", 0.6, [("jw", 0.6), ("tl", 0.3), ("en", 0.1)]

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        segment = Segment(
            " bobo ka", 0.0, 1.0, 0.04, -0.25, 1.1,
            [Word(" bobo", 0.91), Word(" ka", 0.87)],
        )
        return iter([segment]), None


class EngineTests(unittest.TestCase):
    def make_engine(self):
        from model import language_restricted_engine as module
        from RealtimeSTT.transcription_engines.base import TranscriptionEngineConfig

        engine_class = module.LanguageRestrictedFasterWhisperEngine
        engine = engine_class.__new__(engine_class)
        engine.config = TranscriptionEngineConfig(
            model="base", initial_prompt="bobo, tanga", beam_size=1,
        )
        engine.model = FakeWhisper()
        original = module._whisper_model
        module._whisper_model = lambda model: model
        self.addCleanup(setattr, module, "_whisper_model", original)
        return engine

    def test_decodes_without_the_hint_list_and_with_word_probabilities(self):
        engine = self.make_engine()
        result = engine.transcribe(np.zeros(16000, np.float32), use_prompt=True)
        call = engine.model.calls[0]
        self.assertIsNone(call["initial_prompt"])
        self.assertTrue(call["word_timestamps"])
        self.assertEqual(call["language"], "tl")
        self.assertEqual(result.text, "bobo ka")
        self.assertEqual(result.info.language, "tl")
        self.assertAlmostEqual(result.info.language_probability, 0.75)
        self.assertEqual(result.metadata["words"], [[" bobo", 0.91], [" ka", 0.87]])
        self.assertAlmostEqual(result.metadata["no_speech_prob"], 0.04)

    def test_realtimestt_warmup_call_succeeds(self):
        # RealtimeSTT 1.0.2 warms the engine with transcribe(audio,
        # language="en", use_prompt=False); this crashed before.
        engine = self.make_engine()
        engine.warmup(np.zeros(16000, np.float32))
        self.assertEqual(engine.model.calls[0]["language"], "en")

    def test_recorder_never_receives_the_hint_list(self):
        source = (ROOT / "model" / "whisper_stt.py").read_text()
        self.assertIn("initial_prompt=None,", source)
        self.assertNotIn("initial_prompt=initial_prompt", source)


if __name__ == "__main__":
    unittest.main()
