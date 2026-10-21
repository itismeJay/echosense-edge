import unittest

from model.language_restricted_engine import ENGINE_NAME, choose_language, register


class LanguageRestrictionTests(unittest.TestCase):
    def test_only_english_and_tagalog_compete(self):
        # Bisaya often scores as jw/su/ms; it must never win.
        language, probability = choose_language(
            [("jw", 0.5), ("tl", 0.3), ("en", 0.1), ("ms", 0.1)],
        )
        self.assertEqual(language, "tl")
        self.assertAlmostEqual(probability, 0.75)
        self.assertEqual(choose_language([("en", 0.9), ("tl", 0.05)])[0], "en")

    def test_no_allowed_probability_falls_back_to_tagalog(self):
        self.assertEqual(choose_language([("jw", 1.0)]), ("tl", 0.0))
        self.assertEqual(choose_language(()), ("tl", 0.0))

    def test_engine_is_registered_with_realtimestt(self):
        from RealtimeSTT.transcription_engines import factory

        self.assertTrue(register())
        self.assertIn(ENGINE_NAME, factory.ENGINE_CLASS_PATHS)


if __name__ == "__main__":
    unittest.main()
