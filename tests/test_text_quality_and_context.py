import unittest

from detection.text_context import apply_harmless_context_rules
from detection.transcript_quality import (
    assess_transcript_quality,
    normalize_transcript,
)
from model import whisper_stt
from model.blacklist import (
    apply_phonetic_variants,
    check_transcript,
    word_matches,
)
from model.monitored_terms import (
    build_matched_terms,
    replace_backend_terms,
)


class TranscriptNormalizationTests(unittest.TestCase):
    def test_01_empty_text_normalizes_to_empty(self):
        self.assertEqual(normalize_transcript(""), "")

    def test_02_repeated_whitespace_collapses(self):
        self.assertEqual(
            normalize_transcript("  Please \n\t stop   now  "),
            "please stop now",
        )

    def test_03_ordinary_punctuation_forms_safe_boundaries(self):
        self.assertEqual(
            normalize_transcript("Ikaw,\nay… BOBO！"),
            "ikaw ay bobo",
        )
        self.assertTrue(word_matches("ikaw ay bobo", "Ikaw, ay… BOBO！"))

    def test_04_matching_is_case_insensitive(self):
        self.assertTrue(check_transcript("YOU ARE STUPID")["has_profanity"])

    def test_05_original_transcript_is_preserved(self):
        original = "  Pangit KA!\n"
        whisper_stt._on_text(original)
        with whisper_stt._result_lock:
            result = whisper_stt._latest_result
            whisper_stt._latest_result = None
        whisper_stt._new_result_event.clear()

        self.assertEqual(result["transcribed_text"], original)
        self.assertEqual(result["transcript"], original)
        self.assertEqual(result["normalized_text"], "pangit ka")

    def test_06_pang_is_never_rewritten_or_matched_as_pangit(self):
        self.assertEqual(normalize_transcript("pang"), "pang")
        self.assertEqual(apply_phonetic_variants("pang"), "pang")
        self.assertFalse(check_transcript("pang")["has_profanity"])


class TranscriptQualityAcceptanceTests(unittest.TestCase):
    def test_07_normal_short_sentence_is_accepted(self):
        result = assess_transcript_quality("Please stop that now.")
        self.assertTrue(result.accepted)
        self.assertEqual(result.reason_codes, ("accepted",))

    def test_08_mixed_filipino_english_sentence_is_accepted(self):
        result = assess_transcript_quality("Please, ayaw na. Stop muna.")
        self.assertTrue(result.accepted)
        self.assertEqual(result.token_count, 5)

    def test_09_natural_double_stop_is_accepted(self):
        result = assess_transcript_quality("Stop, stop!")
        self.assertTrue(result.accepted)
        self.assertEqual(result.longest_repeated_run, 2)

    def test_10_short_cebuano_double_repetition_is_accepted(self):
        result = assess_transcript_quality("Ayaw, ayaw!")
        self.assertTrue(result.accepted)
        self.assertEqual(result.normalized_text, "ayaw ayaw")


class TranscriptQualityRejectionTests(unittest.TestCase):
    def test_11_empty_transcript_is_rejected(self):
        result = assess_transcript_quality("")
        self.assertFalse(result.accepted)
        self.assertIn("empty_transcript", result.reason_codes)

    def test_12_punctuation_only_transcript_is_rejected(self):
        result = assess_transcript_quality("... !!!")
        self.assertFalse(result.accepted)
        self.assertIn("punctuation_only", result.reason_codes)
        self.assertIn("normalization_empty", result.reason_codes)

    def test_13_extreme_single_token_repetition_is_rejected(self):
        result = assess_transcript_quality("bobo bobo bobo bobo bobo bobo")
        self.assertFalse(result.accepted)
        self.assertIn("excessive_token_repetition", result.reason_codes)

    def test_14_repeated_phrase_pattern_is_rejected(self):
        result = assess_transcript_quality(
            "ikaw ay bobo ikaw ay bobo ikaw ay bobo"
        )
        self.assertFalse(result.accepted)
        self.assertIn("repeated_phrase_pattern", result.reason_codes)

    def test_15_low_unique_ratio_is_rejected_only_when_long_enough(self):
        result = assess_transcript_quality(
            "alpha beta alpha alpha beta alpha beta alpha beta alpha"
        )
        self.assertFalse(result.accepted)
        self.assertIn("low_unique_token_ratio", result.reason_codes)
        self.assertEqual(result.unique_token_count, 2)

    def test_16_very_long_repetitive_transcript_is_rejected(self):
        result = assess_transcript_quality(" ".join(["alpha", "beta"] * 61))
        self.assertFalse(result.accepted)
        self.assertIn("transcript_too_long", result.reason_codes)
        self.assertIn("low_unique_token_ratio", result.reason_codes)

    def test_17_rejection_reason_codes_are_stable(self):
        first = assess_transcript_quality("")
        second = assess_transcript_quality("")
        self.assertEqual(first.reason_codes, second.reason_codes)
        self.assertEqual(
            first.reason_codes,
            (
                "empty_transcript",
                "normalization_empty",
                "too_few_informative_tokens",
            ),
        )


class SafeMatchingTests(unittest.TestCase):
    def setUp(self):
        replace_backend_terms([])

    def tearDown(self):
        replace_backend_terms([])

    def test_18_term_matches_as_a_complete_token(self):
        result = check_transcript("Bobo ka.")
        self.assertIn("bobo", result["detected_words"])

    def test_19_phrase_matches_across_ordinary_punctuation(self):
        result = check_transcript("Ikaw, ay bobo.")
        terms = [item["term"] for item in result["matched_terms"]]
        self.assertIn("ikaw ay bobo", terms)

    def test_20_short_term_does_not_match_inside_longer_word(self):
        replace_backend_terms([{
            "term_id": 901,
            "slur_text": "bad",
            "language": "English",
        }])
        self.assertFalse(check_transcript("badminton practice")["has_profanity"])
        self.assertFalse(word_matches("bobo", "kabobohan"))

    def test_21_duplicate_matches_are_unique_and_deterministic(self):
        first = build_matched_terms("bobo bobo", ["bobo", "bobo"])
        second = build_matched_terms("bobo bobo", ["bobo", "bobo"])
        self.assertEqual(first, second)
        self.assertEqual(
            [item["term"] for item in first].count("bobo"),
            1,
        )

    def test_22_mixed_language_matching_continues(self):
        result = check_transcript("Ikaw ay bobo, buang kaayo ka.")
        self.assertEqual(
            {item["language"] for item in result["matched_terms"]},
            {"fil", "ceb"},
        )


class HarmlessContextSuppressionTests(unittest.TestCase):
    def setUp(self):
        replace_backend_terms([])

    def tearDown(self):
        replace_backend_terms([])

    def test_23_explicit_impersonal_contexts_are_suppressed(self):
        replace_backend_terms([{
            "term_id": 902,
            "slur_text": "bad",
            "language": "English",
        }])
        for phrase, term in (
            ("Pangit.", "pangit"),
            ("Pangit ang panahon.", "pangit"),
            ("Pangit ang drawing.", "pangit"),
            ("Pangit ang signal.", "pangit"),
            ("Pangit ang internet.", "pangit"),
            ("The weather is bad.", "bad"),
            ("This drawing is ugly.", "ugly"),
        ):
            with self.subTest(phrase=phrase):
                result = check_transcript(phrase)
                self.assertNotIn(term, result["detected_words"])
                self.assertTrue(result["context_suppressed_all"])

    def test_24_direct_second_person_insults_remain_candidates(self):
        for phrase in ("Pangit ka.", "Ang pangit mo.", "You are stupid."):
            with self.subTest(phrase=phrase):
                result = check_transcript(phrase)
                self.assertTrue(result["has_profanity"])
                self.assertFalse(result["context_suppressed_all"])

    def test_25_harmless_object_description_cannot_be_alert_evidence(self):
        result = check_transcript("Pangit ang drawing.")
        self.assertFalse(result["has_profanity"])
        self.assertEqual(result["matched_terms"], [])
        self.assertEqual(result["detected_words"], [])

    def test_26_context_suppression_has_an_auditable_reason(self):
        result = apply_harmless_context_rules(
            "pangit ang signal",
            ["pangit"],
        )
        self.assertEqual(len(result.suppressed_terms), 1)
        self.assertEqual(
            result.suppressed_terms[0].reason,
            "harmless_context",
        )

    def test_27_severe_phrase_is_not_suppressed(self):
        result = check_transcript("Pangit ang drawing, patyon tika.")
        self.assertIn("patyon tika", result["hard_hits"])
        self.assertTrue(result["has_profanity"])
        self.assertFalse(result["context_suppressed_all"])


if __name__ == "__main__":
    unittest.main()
