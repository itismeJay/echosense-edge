"""Targeting through intensifiers: Bisaya first, then Tagalog, then English."""

import unittest

from detection.severity import LOW, is_targeted
from model.blacklist import check_transcript


BISAYA_TARGETED = (
    ("bogo ka", "bogo"),
    ("bogo gyud kaayo ka", "bogo"),
    ("ikaw jud ang bogo", "bogo"),
    ("imong nawong pangit kaayo", "pangit"),
    ("imung nawong pangit", "pangit"),
    ("pangit kaayo nimo oy", "pangit"),
    ("patyon tika", "patyon"),
    ("sumbagon taka unya", "sumbagon"),
    ("dakoa nimog tiyan", "dakoa"),
)
TAGALOG_TARGETED = (
    ("tanga ka talaga", "tanga"),
    ("ikaw talaga ay napaka tanga", "tanga"),
    ("ang tanga mo naman", "tanga"),
    ("sobrang bobo mo", "bobo"),
)
ENGLISH_TARGETED = (
    ("You're such a fucking idiot", "idiot"),
    ("You’re such a fucking idiot", "idiot"),
    ("you are so stupid", "stupid"),
    ("you are really freaking dumb", "dumb"),
    ("ur such an idiot", "idiot"),
    ("u really dumb", "dumb"),
    ("you idiot", "idiot"),
)
NOT_TARGETED = (
    ("bogo ko", "bogo"),           # Bisaya self-directed
    ("bogo kaayo ko", "bogo"),
    ("akong drawing pangit", "pangit"),
    ("ka itom sa langit", "itom"),  # exclamative "ka", not "you"
    ("tanga ako", "tanga"),        # Tagalog self-directed
    ("I'm such an idiot", "idiot"),
)


class IntensifierTargetingTests(unittest.TestCase):
    def test_targeted_in_every_language(self):
        for text, term in BISAYA_TARGETED + TAGALOG_TARGETED + ENGLISH_TARGETED:
            with self.subTest(text=text):
                self.assertTrue(is_targeted([term], text))

    def test_self_directed_and_exclamative_stay_untargeted(self):
        for text, term in NOT_TARGETED:
            with self.subTest(text=text):
                self.assertFalse(is_targeted([term], text))

    def test_live_english_sentence_is_not_rated_unaimed(self):
        result = check_transcript("You're such a fucking idiot")
        self.assertTrue(result["detected_words"])
        self.assertNotEqual(result["severity"], LOW)
        self.assertIn(
            "promoted_by:direct_target_pattern",
            result["severity_evidence"]["reasons"],
        )


if __name__ == "__main__":
    unittest.main()
