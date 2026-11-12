from pathlib import Path
import unittest
from unittest import mock

from model.language_guide import GuidedPhrase, LanguageGuide
from model.recording_metadata import RecordingDescriptor
from model.sentence_catalog import load_sentence_cases
from model.transcription_metrics import (
    aggregate_transcription_metrics,
    transcription_metrics,
)
from tools.evaluate_testset import bisaya_rank, refreshed_result, summarize
from tools.record_testset import classroom_phrases, recording_seconds


class SentenceCatalogTests(unittest.TestCase):
    def test_classroom_catalog_loads_all_700_sentences_with_stable_ids(self):
        cases = load_sentence_cases()

        self.assertEqual(len(cases), 700)
        self.assertEqual(len({case.sentence_id for case in cases}), 700)
        self.assertEqual({case.language for case in cases}, {"ceb", "en", "tl"})
        self.assertTrue(all(case.text.strip() for case in cases))
        self.assertTrue(all(case.source_path.is_file() for case in cases))

        first = cases[0]
        self.assertEqual(first.sentence_id, "ceb-0001")
        self.assertEqual(first.source_path.name, "cebuano_classroom_original_200.jsonl")

    def test_catalog_can_load_an_explicit_fixture(self):
        fixture = (
            Path(__file__).resolve().parent
            / "fixtures"
            / "english_classroom_250.jsonl"
        )

        cases = load_sentence_cases((fixture,))

        self.assertEqual(len(cases), 250)
        self.assertTrue(all(case.language == "en" for case in cases))

    def test_recording_catalog_exposes_all_700_sentences(self):
        phrases = classroom_phrases()

        self.assertEqual(len(phrases), 700)
        self.assertEqual(len({item[0] for item in phrases}), 700)
        self.assertEqual(
            {item[2] for item in phrases},
            {"bullying", "normal", "review"},
        )
        self.assertEqual({item[3] for item in phrases}, {"ceb", "en", "tl"})


class RecordingMetadataTests(unittest.TestCase):
    def test_metadata_round_trips_through_the_clip_filename(self):
        descriptor = RecordingDescriptor(
            sentence_id="ceb-0001",
            speaker_id="speaker-02",
            delivery="mocking",
            noise="classroom",
            distance="far",
        )

        filename = descriptor.filename()
        restored = RecordingDescriptor.from_filename(filename)

        self.assertEqual(filename, (
            "ceb-0001__speaker-02__mocking__classroom__far.wav"
        ))
        self.assertEqual(restored, descriptor)

    def test_legacy_filename_remains_readable(self):
        descriptor = RecordingDescriptor.from_filename("bogo_ka__near.wav")

        self.assertEqual(descriptor.sentence_id, "bogo_ka")
        self.assertEqual(descriptor.distance, "near")
        self.assertEqual(descriptor.speaker_id, "legacy")

    def test_unsafe_or_unknown_metadata_is_rejected(self):
        with self.assertRaises(ValueError):
            RecordingDescriptor(
                sentence_id="ceb-0001",
                speaker_id="../speaker",
                delivery="normal",
                noise="quiet",
                distance="near",
            )

    def test_long_sentences_receive_more_recording_time(self):
        self.assertEqual(recording_seconds("one two"), 4.0)
        self.assertGreater(
            recording_seconds("one two three four five six seven eight"),
            4.0,
        )
        self.assertLessEqual(
            recording_seconds(" ".join(["word"] * 100)),
            12.0,
        )


class LanguageGuideTests(unittest.TestCase):
    def test_similar_sentence_gets_a_spelling_suggestion(self):
        guide = LanguageGuide((
            GuidedPhrase("ceb-0001", "Bahoa nimog baba oy", "ceb"),
        ))

        suggestion = guide.suggest(
            "Bahua ni mug baba oy.",
            language="ceb",
        )

        self.assertIsNotNone(suggestion)
        self.assertEqual(suggestion.suggested_text, "Bahoa nimog baba oy")
        self.assertEqual(suggestion.source_id, "ceb-0001")
        self.assertTrue(suggestion.changed)
        self.assertGreaterEqual(suggestion.similarity, 0.8)

    def test_short_or_distant_speech_is_never_forced_to_a_phrase(self):
        guide = LanguageGuide((
            GuidedPhrase("ceb-0001", "Bahoa nimog baba oy", "ceb"),
        ))

        self.assertIsNone(guide.suggest("baba", language="ceb"))
        self.assertIsNone(
            guide.suggest("maayong buntag sa tanan", language="ceb")
        )

    def test_ambiguous_match_returns_no_suggestion(self):
        guide = LanguageGuide((
            GuidedPhrase("en-0001", "you are ugly today", "en"),
            GuidedPhrase("en-0002", "you are angry today", "en"),
        ))

        self.assertIsNone(
            guide.suggest("you are agly today", language="en")
        )

    def test_alternative_can_support_a_suggestion_without_mutating_primary(self):
        primary = "bahua mug ba ba oy"
        guide = LanguageGuide((
            GuidedPhrase("ceb-0001", "Bahoa nimog baba oy", "ceb"),
        ))

        suggestion = guide.suggest(
            primary,
            alternatives=("bahoa nimog baba oy",),
            language="ceb",
        )

        self.assertEqual(suggestion.heard_text, primary)
        self.assertEqual(suggestion.supporting_text, "bahoa nimog baba oy")
        self.assertEqual(primary, "bahua mug ba ba oy")

    def test_live_shadow_guidance_preserves_the_whisper_transcript(self):
        from model import whisper_stt

        original = "Bahua ni mug baba oy."
        guide = LanguageGuide((
            GuidedPhrase("ceb-0001", "Bahoa nimog baba oy", "ceb"),
        ))

        with mock.patch.object(whisper_stt, "_language_guide", guide):
            whisper_stt._on_text(original, whisper_language="ceb")
            result = whisper_stt.consume_transcription_result(timeout=1.0)

        self.assertEqual(result["transcribed_text"], original)
        self.assertEqual(
            result["language_guide"]["suggested_text"],
            "Bahoa nimog baba oy",
        )
        self.assertTrue(result["language_guide"]["shadow_only"])


class TranscriptionMetricTests(unittest.TestCase):
    def test_metrics_measure_words_characters_and_normalized_exactness(self):
        result = transcription_metrics("One two!", "one too")

        self.assertFalse(result.exact)
        self.assertEqual(result.reference_words, 2)
        self.assertEqual(result.word_errors, 1)
        self.assertEqual(result.word_error_rate, 0.5)
        self.assertEqual(result.character_errors, 1)
        self.assertAlmostEqual(result.character_error_rate, 1 / 7)

        punctuation_only = transcription_metrics(
            "Bahoa nimog baba oy!",
            "bahoa nimog baba oy.",
        )
        self.assertTrue(punctuation_only.exact)
        self.assertEqual(punctuation_only.word_error_rate, 0.0)

    def test_aggregate_weights_errors_by_reference_length(self):
        summary = aggregate_transcription_metrics((
            transcription_metrics("one two", "one two"),
            transcription_metrics("three four", "three"),
        ))

        self.assertEqual(summary["sentence_count"], 2)
        self.assertEqual(summary["exact_sentence_rate"], 0.5)
        self.assertEqual(summary["word_error_rate"], 0.25)

    def test_evaluator_summary_reports_sentence_accuracy_by_language(self):
        rows = [
            {
                "label": "bullying",
                "language": "ceb",
                "distance": "near",
                "text": "bahoa nimog baba oy",
                "alert": True,
                "utterances": [{"transcript": "bahua nimog baba oy"}],
            },
            {
                "label": "normal",
                "language": "en",
                "distance": "near",
                "text": "please open your books",
                "alert": False,
                "utterances": [{"transcript": "please open your books"}],
            },
        ]

        summary = summarize(rows, [])

        self.assertEqual(summary["transcript_sentence_count"], 2)
        self.assertEqual(summary["transcript_exact_sentence_rate"], 0.5)
        self.assertEqual(summary["transcript_exact_sentence_rate_ceb"], 0.0)
        self.assertEqual(summary["transcript_exact_sentence_rate_en"], 1.0)
        self.assertGreater(summary["transcript_word_error_rate_ceb"], 0.0)

    def test_evaluator_reports_hypothetical_guided_accuracy(self):
        rows = [{
            "label": "bullying",
            "language": "ceb",
            "distance": "near",
            "text": "bahoa nimog baba oy",
            "alert": True,
            "utterances": [{
                "transcript": "bahua ni mug baba oy",
                "language_guide": {
                    "matched": True,
                    "suggested_text": "bahoa nimog baba oy",
                },
            }],
        }]

        summary = summarize(rows, [])

        self.assertGreater(summary["transcript_word_error_rate"], 0.0)
        self.assertEqual(summary["guided_word_error_rate"], 0.0)
        self.assertEqual(summary["guided_exact_sentence_rate"], 1.0)

    def test_model_ranking_prioritizes_bisaya_transcription_accuracy(self):
        poor_transcript = {
            "summary": {
                "transcript_word_error_rate_ceb": 0.60,
                "transcript_exact_sentence_rate_ceb": 0.10,
                "recall_ceb": 1.0,
                "normal_fp_ceb": 0.0,
                "media_fp": 0.0,
                "normal_fp": 0.0,
                "whisper_mean_s": 1.0,
            }
        }
        faithful_transcript = {
            "summary": {
                "transcript_word_error_rate_ceb": 0.20,
                "transcript_exact_sentence_rate_ceb": 0.60,
                "recall_ceb": 0.90,
                "normal_fp_ceb": 0.0,
                "media_fp": 0.0,
                "normal_fp": 0.0,
                "whisper_mean_s": 1.2,
            }
        }

        ranked = sorted(
            (poor_transcript, faithful_transcript),
            key=bisaya_rank,
        )

        self.assertIs(ranked[0], faithful_transcript)

    def test_historical_result_summary_is_refreshed_from_saved_rows(self):
        stored = {
            "name": "old-result",
            "summary": {"recall": 1.0},
            "rows": [{
                "label": "bullying",
                "language": "ceb",
                "distance": "near",
                "text": "bahoa nimog baba oy",
                "alert": True,
                "utterances": [{
                    "transcript": "bahoa nimog baba oy",
                    "whisper_seconds": 1.25,
                }],
            }],
        }

        refreshed = refreshed_result(stored)

        self.assertEqual(refreshed["summary"]["transcript_word_error_rate"], 0.0)
        self.assertEqual(refreshed["summary"]["whisper_mean_s"], 1.25)
        self.assertEqual(stored["summary"], {"recall": 1.0})


if __name__ == "__main__":
    unittest.main()
