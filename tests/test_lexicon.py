"""The Bisaya Grade 6 lexicon (lexicon/bisaya_bullying.yaml) end to end.

Every phrase goes through the real matcher, severity and fusion policy.
"""

import contextlib
import io
import re
import unittest

from detection.lexicon import CONTEXT, INSULT, LEXICON, SELF_HARM, SWEAR, THREAT
from tests.test_soundalike import sent

# Exclusion phrases with no "you" word are everyday speech too
# ("layo diri ang balay"), so alone they are not sent.
UNTARGETED_CONTEXT_OK = {"layo diri", "palayo diri", "pahawa diha"}


def quiet_sent(text):
    with contextlib.redirect_stdout(io.StringIO()):
        return sent(text)


class LexiconShapeTests(unittest.TestCase):
    def test_every_root_has_at_least_ten_phrases(self):
        short = {r.name: len(r.phrases) for r in LEXICON.roots if len(r.phrases) < 10}
        self.assertEqual(short, {})

    def test_size(self):
        self.assertGreaterEqual(len(LEXICON.phrases), 700)


class LexiconDetectionTests(unittest.TestCase):
    def test_every_phrase_is_sent(self):
        missed = [
            p.text for p in LEXICON.phrases
            if p.text not in UNTARGETED_CONTEXT_OK and not quiet_sent(p.text)[0]
        ]
        self.assertEqual(missed, [])

    def test_threats_and_self_harm_are_high(self):
        wrong = [
            (p.text, quiet_sent(p.text)) for p in LEXICON.phrases
            if p.role in (THREAT, SELF_HARM) and quiet_sent(p.text) != (True, "HIGH")
        ]
        self.assertEqual(wrong, [])

    def test_aimed_insults_are_at_least_medium(self):
        aimed = re.compile(r"\b(ka|nimo|niyo|ninyo|kamo|mo|ikaw|kag)\b")
        wrong = [
            p.text for p in LEXICON.phrases
            if p.role == INSULT and aimed.search(p.text)
            and quiet_sent(p.text)[1] not in ("MEDIUM", "HIGH")
        ]
        self.assertEqual(wrong, [])

    def test_bare_swear_words_are_low(self):
        wrong = [
            p.text for p in LEXICON.phrases
            if p.role == SWEAR and quiet_sent(p.text) != (True, "LOW")
        ]
        self.assertEqual(wrong, [])

    def test_whisper_style_spellings_are_caught(self):
        variants = []
        for p in LEXICON.phrases:
            if p.text in UNTARGETED_CONTEXT_OK:
                continue
            variants.append(re.sub(r"\s+(ka|mo)$", r"\1", p.text))       # joined
            variants.append(p.text.translate(str.maketrans({"o": "u"})))  # o/u
            variants.append(p.text.capitalize().replace("a", "á", 1) + ".")
        caught = sum(quiet_sent(v)[0] for v in variants)
        self.assertGreaterEqual(caught / len(variants), 0.97)


class LexiconFalseAlarmTests(unittest.TestCase):
    def test_everyday_sentences_never_alert(self):
        alarms = [
            (root.name, text) for root in LEXICON.roots for text in root.everyday
            if quiet_sent(text)[0]
        ]
        self.assertEqual(alarms, [])

    def test_end_only_phrase_alerts_only_at_the_end_of_a_sentence(self):
        # "pangita nimo oy" = how ugly you are; with more words after it,
        # "pangita" is "look for" ("pangita nimo oy ang libro").
        self.assertIn("pangita nimo oy", LEXICON.end_only)
        for text in ("Pangita nimo oy.", "pangita nimo uy!"):
            with self.subTest(text=text):
                self.assertEqual(quiet_sent(text), (True, "MEDIUM"))
        # Still sent after a lead-in; "hala" is an excitement marker (LOW).
        self.assertTrue(quiet_sent("Hala, pangita nimo oy.")[0])
        for text in (
            "Pangita nimo oy ang libro.",
            "Pangita nimo oy, ang imong bag.",
            "Pangitaa nimo oy ang libro.",
        ):
            with self.subTest(text=text):
                self.assertFalse(quiet_sent(text)[0])

    def test_end_only_rule_also_holds_for_backend_terms(self):
        import model.monitored_terms as monitored_terms
        from model.blacklist import check_transcript

        previous = monitored_terms.get_backend_terms()
        monitored_terms.replace_backend_terms(
            [{"term_id": 9001, "slur_text": "pangita nimo oy", "language": "ceb"}]
        )
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                result = check_transcript("Pangita nimo oy ang libro.")
            self.assertEqual(result["detected_words"], [])
            self.assertEqual(result["matched_terms"], [])
        finally:
            monitored_terms.replace_backend_terms(
                [
                    {"term_id": t.term_id, "slur_text": t.term, "language": t.language}
                    for t in previous
                ]
            )

    def test_negated_insults_do_not_alert(self):
        for text in ("dili ka bogo", "dili man ka bogo uy", "dili ka buang"):
            with self.subTest(text=text):
                self.assertFalse(quiet_sent(text)[0])


if __name__ == "__main__":
    unittest.main()
