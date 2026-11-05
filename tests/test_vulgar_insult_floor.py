import unittest

from detection.severity import LOW, MEDIUM, calculate_severity


class VulgarInsultFloorTests(unittest.TestCase):
    def test_untargeted_bahog_baba_is_medium(self):
        for term, transcript in (
            ("baho og baba", "Báhog ba?"),
            ("bahog baba", "bahog baba"),
        ):
            with self.subTest(transcript=transcript):
                decision = calculate_severity([term], transcript=transcript)
                self.assertEqual(decision.level, MEDIUM)
                self.assertIn("medium_floor:vulgar_insult", decision.reasons)

    def test_laughter_does_not_lower_bahog_baba(self):
        decision = calculate_severity(
            ["bahog baba"], transcript="bahog baba haha", laughter_present=True
        )
        self.assertEqual(decision.level, MEDIUM)

    def test_negated_bahog_baba_stays_low(self):
        decision = calculate_severity(
            ["bahog baba"], transcript="ayaw ingna nga bahog baba"
        )
        self.assertEqual(decision.level, LOW)

    def test_plain_descriptive_word_still_low(self):
        self.assertEqual(calculate_severity(["baho"], transcript="baho").level, LOW)


if __name__ == "__main__":
    unittest.main()
