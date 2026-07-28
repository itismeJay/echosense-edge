import unittest

from model.blacklist import (
    check_transcript,
    contains_blacklisted_word,
    get_detected_words,
    word_matches,
)
from model.monitored_terms import (
    build_matched_terms,
    classify_transcript_language,
    normalize_text,
    replace_backend_terms,
)


class MultilingualPhraseDetectionTests(unittest.TestCase):
    def setUp(self):
        replace_backend_terms([])

    def tearDown(self):
        replace_backend_terms([])

    def _evidence_for(self, text):
        return check_transcript(text)["matched_terms"]

    def test_filipino_phrase_match(self):
        evidence = self._evidence_for("Ikaw ay bobo")
        self.assertIn(
            {
                "term": "ikaw ay bobo",
                "language": "fil",
                "match_type": "phrase",
            },
            evidence,
        )

    def test_bisaya_phrase_match(self):
        evidence = self._evidence_for("Buang kaayo ka")
        self.assertIn(
            {
                "term": "buang kaayo ka",
                "language": "ceb",
                "match_type": "phrase",
            },
            evidence,
        )

    def test_english_phrase_match(self):
        evidence = self._evidence_for("You are stupid")
        self.assertIn(
            {
                "term": "you are stupid",
                "language": "en",
                "match_type": "phrase",
            },
            evidence,
        )

    def test_mixed_language_phrase(self):
        evidence = self._evidence_for("Bobo kaayo ka")
        self.assertEqual(
            evidence,
            [{
                "term": "bobo kaayo ka",
                "language": "mixed",
                "match_type": "phrase",
            }],
        )
        language, confidence = classify_transcript_language("tl", 0.82, evidence)
        self.assertEqual(language, "mixed")
        self.assertIsNone(confidence)

    def test_punctuation_unicode_and_whitespace_normalization(self):
        text = "  IKAW,\n\tAY…  BOBO！ "
        self.assertEqual(normalize_text(text), "ikaw ay bobo")
        self.assertTrue(check_transcript(text)["has_profanity"])

    def test_exact_word_boundary_behavior(self):
        self.assertTrue(word_matches("bobo", "bobo ka"))
        self.assertFalse(word_matches("bobo", "kabobohan"))
        self.assertFalse(check_transcript("kabobohan")["has_profanity"])

    def test_multiple_matches(self):
        evidence = self._evidence_for(
            "Ikaw ay bobo, and you are stupid, buang kaayo ka."
        )
        terms = {item["term"] for item in evidence}
        self.assertEqual(
            terms,
            {"ikaw ay bobo", "you are stupid", "buang kaayo ka"},
        )
        language, confidence = classify_transcript_language("tl", 0.91, evidence)
        self.assertEqual(language, "mixed")
        self.assertIsNone(confidence)

    def test_duplicate_removal(self):
        replace_backend_terms([
            {
                "term_id": 12,
                "slur_text": "example monitored phrase",
                "language": "Bisaya",
            },
            {
                "term_id": 12,
                "slur_text": "example monitored phrase",
                "language": "Bisaya",
            },
        ])
        evidence = build_matched_terms(
            "example monitored phrase example monitored phrase",
            ["example monitored phrase", "example monitored phrase"],
        )
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["term_id"], 12)

    def test_unknown_language(self):
        self.assertEqual(
            classify_transcript_language(None, None, []),
            ("unknown", None),
        )
        self.assertEqual(
            classify_transcript_language("fr", 0.95, []),
            ("unknown", None),
        )

    def test_null_confidence_when_not_exposed(self):
        language, confidence = classify_transcript_language("tl", 0, [])
        self.assertEqual(language, "unknown")
        self.assertIsNone(confidence)

    def test_whisper_confidence_is_preserved_not_fabricated(self):
        language, confidence = classify_transcript_language("tl", 0.82, [])
        self.assertEqual(language, "fil")
        self.assertEqual(confidence, 0.82)

    def test_backend_dictionary_term_id_preservation(self):
        replace_backend_terms([{
            "term_id": 12,
            "slur_text": "example monitored phrase",
            "language": "Bisaya",
            "severity_weight": 0.8,
        }])
        result = check_transcript("Example monitored phrase!")
        self.assertTrue(result["has_profanity"])
        self.assertEqual(result["hard_hits"], [])
        self.assertIn("example monitored phrase", result["soft_hits"])
        self.assertEqual(
            result["matched_terms"],
            [{
                "term_id": 12,
                "term": "example monitored phrase",
                "language": "ceb",
                "match_type": "phrase",
            }],
        )

    def test_legacy_blacklist_api_remains_functional(self):
        self.assertTrue(contains_blacklisted_word("bobo ka"))
        self.assertIn("bobo", get_detected_words("bobo ka"))
        self.assertFalse(contains_blacklisted_word("ordinary classroom speech"))


if __name__ == "__main__":
    unittest.main()
