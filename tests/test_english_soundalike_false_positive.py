"""Alert 204: English "Let's cut to the chase" was matched as Bisaya "kuto".

ElevenLabs detected English, the edge relabelled it Bisaya, the sound-alike
matcher joined "cut to" into "kuto" and the direct-target rule made it MEDIUM.
"""

import contextlib
import io
import unittest

from detection.evidence import LLMJudgement
from detection.soundalike import correct_transcript
from detection.text_detector import SOUNDALIKE_UNCONFIRMED_REASON, TextDetector
from model import whisper_stt
from model.blacklist import check_transcript
from tests.test_elevenlabs_final_transcript import ELEVENLABS_META, _Adapter
from tests.test_llm_semantic import FakeClient, WALL, fuse, gate_for, judgement

INCIDENT_TEXT = (
    "This is my trading result... Hey, you're really pretty. "
    "Let's cut to the chase. What's your Snapchat?"
)


def analyze(text, language, answer=None):
    gate = gate_for(FakeClient(answer)) if answer is not None else None
    detector = TextDetector(semantic_gate=gate)
    with contextlib.redirect_stdout(io.StringIO()):
        return detector.analyze_transcript(
            text, language=language, observed_at=WALL, observed_monotonic=0.0,
        )


def alerted(evidence):
    if not evidence.qualifies_to_open:
        return False
    with contextlib.redirect_stdout(io.StringIO()):
        return fuse(evidence).deliver


def on_text(text, language, metadata):
    with contextlib.redirect_stdout(io.StringIO()):
        whisper_stt._on_text(
            text, whisper_language=language, whisper_language_confidence=1.0,
            audio_event=_Adapter().event, decode_metadata=metadata,
        )
    return whisper_stt.consume_transcription_result()


def detected(text, language):
    with contextlib.redirect_stdout(io.StringIO()):
        return check_transcript(text, language=language)["detected_words"]


class EnglishFalsePositiveTests(unittest.TestCase):
    def tearDown(self):
        whisper_stt.clear_transcription_results()

    def test_cut_to_the_chase_in_english_never_alerts(self):
        for text in ("Let's cut to the chase", INCIDENT_TEXT):
            with self.subTest(text=text):
                self.assertEqual(detected(text, "en"), [])
                self.assertFalse(alerted(analyze(text, "en")))
                # Even when Claude is unsure the way it was in alert 204.
                self.assertFalse(alerted(analyze(
                    text, "en", judgement("shaming", 0.45, targeted=True),
                )))

    def test_elevenlabs_english_keeps_its_language_and_matches_nothing(self):
        result = on_text(INCIDENT_TEXT, "en", ELEVENLABS_META)
        self.assertEqual(result["language"], "en")
        self.assertEqual(result["detected_words"], [])

    def test_elevenlabs_detected_language_is_not_relabelled_by_matched_words(self):
        # A Bisaya word inside English speech no longer turns the transcript
        # into "ceb"; the word itself still matches exactly.
        result = on_text("You are so pangit", "en", ELEVENLABS_META)
        self.assertEqual(result["language"], "en")
        self.assertIn("pangit", result["detected_words"])

    def test_whisper_forced_language_is_still_decided_by_matched_words(self):
        result = on_text("Yawa ka.", "tl", {"avg_logprob": -0.2})
        self.assertEqual(result["language"], "ceb")

    def test_english_keeps_word_for_word_spelling_variants_only(self):
        # Exact words and the lexicon's variants still match in English.
        self.assertIn("bobo", detected("you are so bobo", "en"))
        self.assertIn("pangit", detected("you are so pangit", "en"))
        # Joined/re-split Bisaya guesses do not.
        self.assertEqual(detected("tamboka", "en"), [])


class JoiningAcrossWordsTests(unittest.TestCase):
    def test_short_terms_are_never_built_by_joining_words(self):
        self.assertEqual(correct_transcript("cut to", ("kuto",)).corrections, ())
        for heard in ("bo bo", "bob o", "ku to"):
            with self.subTest(heard=heard):
                result = correct_transcript(heard, ("bobo", "kuto"))
                self.assertEqual(result.corrections, ())
        self.assertEqual(detected("Let's cut to the chase", "ceb"), [])
        self.assertEqual(detected("Let's cut to the chase", None), [])

    def test_never_joins_across_english_function_words(self):
        for heard in ("pang it ka", "pangi to ka"):
            with self.subTest(heard=heard):
                result = correct_transcript(heard, ("pangit ka",))
                self.assertEqual(result.corrections, ())
        # A function word kept as its own word is fine.
        result = correct_transcript("kil you", ("kill you",))
        self.assertEqual(result.text, "kill you")

    def test_longer_terms_may_still_be_rejoined(self):
        self.assertIn("bogo ka", detected("bo go ka", "ceb"))
        self.assertIn("sumbagon tika", detected("sumbagonti ka", "ceb"))


class BisayaStillDetectedTests(unittest.TestCase):
    def test_kuto_ka_alerts(self):
        for text in ("Kuto ka!", "Kuto ka, uy!"):
            with self.subTest(text=text):
                self.assertIn("kuto", detected(text, "ceb"))
                self.assertTrue(alerted(analyze(text, "ceb")))
                self.assertTrue(alerted(analyze(
                    text, "ceb", judgement("insult", 0.8, targeted=True),
                )))

    def test_kutoon_ka_is_matched_and_alerts_with_claude(self):
        self.assertIn("kutohon ka", detected("Kutoon ka", "ceb"))
        self.assertTrue(alerted(analyze(
            "Kutoon ka", "ceb", judgement("insult", 0.8, targeted=True),
        )))

    def test_code_switched_kuto_ka_uy_from_elevenlabs(self):
        result = on_text("Kuto ka, uy!", "ceb", ELEVENLABS_META)
        self.assertEqual(result["language"], "ceb")
        self.assertIn("kuto", result["detected_words"])


class SoundalikeNeedsClaudeTests(unittest.TestCase):
    # "tamboka" is "tambok ka" only after the sound-alike matcher splits it.
    TEXT = "Tamboka!"

    def test_confirmed_insult_still_alerts(self):
        for label, confidence in (("insult", 0.6), ("insult", 0.9), ("threat", 0.7),
                                  ("exclusion", 0.8)):
            with self.subTest(label=label, confidence=confidence):
                evidence = analyze(
                    self.TEXT, "ceb", judgement(label, confidence, targeted=True),
                )
                self.assertTrue(alerted(evidence))

    def test_unconfirmed_aimed_soundalike_goes_to_review_only(self):
        # Not dropped: sent with needs_review (dashboard, no push).
        for label, confidence in (("insult", 0.59), ("shaming", 0.45),
                                  ("banter", 0.9), ("none", 0.9)):
            with self.subTest(label=label, confidence=confidence):
                evidence = analyze(self.TEXT, "ceb", judgement(label, confidence))
                self.assertEqual(evidence.review_reasons, (SOUNDALIKE_UNCONFIRMED_REASON,))
                self.assertIn(SOUNDALIKE_UNCONFIRMED_REASON, evidence.reasons)

    def test_without_claude_the_lexicon_rules_stand(self):
        self.assertTrue(alerted(analyze(self.TEXT, "ceb")))
        fallback = LLMJudgement(status="fallback", fallback_reason="timeout")
        self.assertTrue(alerted(analyze(self.TEXT, "ceb", fallback)))

    def test_words_as_written_never_need_confirmation(self):
        evidence = analyze("Tambok ka!", "ceb", judgement("banter", 0.9))
        self.assertTrue(alerted(evidence))
        self.assertNotIn(SOUNDALIKE_UNCONFIRMED_REASON, evidence.reasons)


if __name__ == "__main__":
    unittest.main()
