"""Characterization coverage for text behavior preserved by the AFTER design."""

import unittest

from detection.severity import HIGH, MEDIUM, calculate_severity
from detection.transcript_quality import assess_transcript_quality
from model.blacklist import check_transcript, word_matches


class AfterTextCharacterizationTests(unittest.TestCase):
    def test_bobo_is_qualifying_monitored_text(self):
        result = check_transcript("bobo")
        self.assertTrue(result["has_profanity"])
        self.assertIn("bobo", result["hard_hits"])
        self.assertEqual(result["severity"], MEDIUM)

    def test_tanga_is_qualifying_monitored_text(self):
        result = check_transcript("tanga")
        self.assertTrue(result["has_profanity"])
        self.assertIn("tanga", result["hard_hits"])
        self.assertEqual(result["severity"], MEDIUM)

    def test_bare_pangit_is_suppressed_without_target(self):
        result = check_transcript("pangit")
        self.assertFalse(result["has_profanity"])
        self.assertTrue(result["context_suppressed_all"])

    def test_pangit_ka_remains_directed_evidence(self):
        result = check_transcript("pangit ka")
        decision = calculate_severity(
            result["detected_words"],
            transcript="pangit ka",
        )
        self.assertTrue(result["has_profanity"])
        self.assertEqual(decision.level, MEDIUM)
        self.assertIn("direct_target_pattern", decision.supporting_evidence)

    def test_token_boundaries_remain_safe(self):
        self.assertTrue(word_matches("bobo", "bobo ka"))
        self.assertFalse(word_matches("bobo", "kabobohan"))

    def test_repetition_promotes_existing_text_evidence(self):
        decision = calculate_severity(["bobo"], repeated=True)
        self.assertEqual(decision.level, HIGH)
        self.assertIn("promoted_by:repetition", decision.reasons)

    def test_low_quality_repetition_is_rejected_before_matching(self):
        quality = assess_transcript_quality(
            "bobo bobo bobo bobo bobo bobo"
        )
        self.assertFalse(quality.accepted)
        self.assertIn("excessive_token_repetition", quality.reason_codes)

    def test_laughter_is_preserved_as_context(self):
        result = check_transcript("haha bobo")
        self.assertTrue(result["is_casual"])


if __name__ == "__main__":
    unittest.main()
