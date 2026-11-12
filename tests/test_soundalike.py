"""Mishearing and sound-alike correction, and the descriptive-word policy."""

from dataclasses import replace
from datetime import datetime, timezone
import unittest

from tests.tone_helpers import with_tone

from detection.fusion import fuse_event_evidence
from detection.severity import is_descriptive_only
from detection.soundalike import correct_transcript, sound_key, spelling_key
from detection.text_detector import TextDetector
import model.blacklist as blacklist
from model.blacklist import check_transcript
from tests.test_event_coordinator_after import WALL
from tests.test_fusion_after import tone_window


def evidence_for(transcript, at=0.0):
    detector = TextDetector()
    return detector.analyze_transcript(
        transcript,
        observed_at=datetime(2026, 10, 27, tzinfo=timezone.utc),
        observed_monotonic=at,
    )


def sent(transcript, acoustic=()):
    evidence = evidence_for(transcript)
    if not evidence.qualifies_to_open:
        return False, None
    decision = fuse_event_evidence((evidence,), tuple(acoustic), duration_seconds=1.0)
    return decision.deliver, decision.level


# Everyday classroom speech. None of it may be corrected or matched.
EVERYDAY = [
    # Bisaya
    "maayong buntag class", "hulam ko sa imong lapis", "human na mo",
    "adto ta sa canteen", "pangitaa ang imong libro", "pangita ang imong bag",
    "taga asa ka", "unsa imong ngalan", "asa ka padulong", "kaon na ta",
    "ayaw kalimti ang assignment", "patay na ang suga", "namatay ang iro nila",
    "buwan na sa sunod", "bayad na ko sa field trip", "basaha ang pangutana",
    "gamay ra kaayo ang papel", "dako kaayo ang balay nila", "salamat kaayo ma'am",
    "sige lang kay ok ra", "tabangi ko ani", "unsay answer sa number tres",
    "hulat sa ko diri", "gutom na ko", "uli na ta", "pila imong score",
    "nindot kaayo imong drawing", "lingaw kaayo ang dula ganina", "bukas ang pultahan",
    "kinsa ang absent karon", "dali na diri", "isulat sa notebook",
    # Tagalog
    "magandang umaga po", "pahiram ng lapis", "tapos na ba kayo",
    "anong sagot sa tanong", "tara na sa canteen", "salamat po teacher",
    "ang ganda ng drawing mo", "pakibasa ang libro", "patayin mo ang ilaw",
    "bukas na lang natin tapusin", "sino ang may baon", "hintayin mo ako",
    # English
    "good morning class", "can I borrow your pencil", "please open your books",
    "the party is on friday", "my shirt is blue", "this pasta is good",
    "put it in your pocket", "the bucket is full", "who is absent today",
    "shut the door please", "you are my best friend", "you did a great job",
]


class MishearingTests(unittest.TestCase):
    def test_requested_bobo_ka_mishearings(self):
        for heard in ("baboka", "babo ka", "bubo ka"):
            with self.subTest(heard=heard):
                result = correct_transcript(heard)
                self.assertEqual(result.text, "bobo ka")
                self.assertEqual(result.corrections[0].kind, "mishearing")

    def test_pangit_ka_heard_as_bang_itka_alerts_as_targeted(self):
        result = check_transcript("Bang itka.")
        self.assertIn("pangit ka", result["detected_words"])
        self.assertEqual(result["severity"], "MEDIUM")
        self.assertEqual(sent("Bang itka.")[0], True)

    def test_original_transcript_is_kept_as_evidence(self):
        evidence = evidence_for("Bang itka.")
        self.assertEqual(evidence.transcript, "Bang itka.")
        self.assertEqual(evidence.match_text, "pangit ka")
        self.assertTrue(evidence.direct_target)


class SoundalikeTests(unittest.TestCase):
    def setUp(self):
        self.previous = blacklist.SOUNDALIKE_ENABLED
        blacklist.SOUNDALIKE_ENABLED = True

    def tearDown(self):
        blacklist.SOUNDALIKE_ENABLED = self.previous

    def test_spelling_and_sound_keys_fold_bisaya_variation(self):
        self.assertEqual(spelling_key("bogo"), spelling_key("bugu"))
        self.assertEqual(spelling_key("piste"), spelling_key("pisti"))
        self.assertEqual(spelling_key("bwang"), spelling_key("buang"))
        self.assertEqual(sound_key("bangit"), sound_key("pangit"))

    def test_misheard_insults_are_corrected(self):
        cases = {
            "Bwang ka": "buang ka",
            "Yawaka.": "yawa ka",
            "patyon tikan": "patyon tika",
            "Patyun tika": "patyon tika",
        }
        for heard, term in cases.items():
            with self.subTest(heard=heard):
                self.assertIn(term, check_transcript(heard)["detected_words"])

    def test_fuzzy_threat_needs_support_but_safe_spelling_alias_stays_high(self):
        self.assertEqual(sent("patyon tikan"), (False, None))
        self.assertEqual(sent("Patyun tika"), (True, "HIGH"))

    def test_exact_monitored_word_is_never_turned_into_another(self):
        self.assertEqual(check_transcript("bobo ka")["detected_words"], ["bobo ka", "bobo"])

    def test_pronoun_is_never_changed_or_invented(self):
        result = check_transcript("ang bobo ko")
        self.assertEqual(result["corrections"], [])
        self.assertEqual(result["severity"], "LOW")
        self.assertEqual(correct_transcript("ang bobo", ["ang bobo mo"]).corrections, ())

    def test_never_matches_inside_a_word(self):
        self.assertEqual(check_transcript("patay na ang manok")["detected_words"], [])
        self.assertEqual(check_transcript("taga asa ka")["detected_words"], [])

    def test_descriptive_single_words_never_fuzz(self):
        self.assertEqual(check_transcript("my shirt is blue")["detected_words"], [])

    def test_everyday_speech_is_not_corrected_or_matched(self):
        for sentence in EVERYDAY:
            with self.subTest(sentence=sentence):
                result = check_transcript(sentence)
                self.assertEqual(result["corrections"], [])
                self.assertFalse(sent(sentence)[0], result["detected_words"])

    def test_never_correct_word_is_not_changed_inside_an_exact_span(self):
        # "pangitaa nimo oy" (look for it) folds to the same letters as the
        # insult "pangita nimo oy", but "pangitaa" must stay as it is.
        result = blacklist.correct_for_matching("Pangitaa nimo oy.")
        self.assertEqual(result.corrections, ())
        self.assertIn("pangita nimo oy", blacklist.HARD_TRIGGERS)

    def test_switch_off_keeps_only_the_explicit_mishearings(self):
        blacklist.SOUNDALIKE_ENABLED = False
        self.assertEqual(check_transcript("Bwang ka")["detected_words"], [])
        self.assertIn("pangit ka", check_transcript("Bang itka.")["detected_words"])


class DescriptiveWordPolicyTests(unittest.TestCase):
    def test_descriptive_terms(self):
        self.assertTrue(is_descriptive_only(["too short"]))
        self.assertTrue(is_descriptive_only(["pangit", "pangit kaayo"]))
        self.assertFalse(is_descriptive_only(["pangit ka"]))
        self.assertFalse(is_descriptive_only(["bobo"]))
        self.assertFalse(is_descriptive_only([]))

    def test_untargeted_descriptive_words_are_not_sent(self):
        for sentence in (
            "This wall won't cut it. It's too short and too flimsy.",
            "pangit kaayo ang panahon karon",
            "the drawing is ugly",
        ):
            with self.subTest(sentence=sentence):
                self.assertFalse(sent(sentence)[0])

    def test_descriptive_words_aimed_at_someone_are_sent(self):
        for sentence in ("pangit ka", "you're so ugly", "tambok ka", "baho ka"):
            with self.subTest(sentence=sentence):
                self.assertEqual(sent(sentence), (True, "MEDIUM"))

    def test_descriptive_word_with_aggressive_tone_is_sent(self):
        evidence = replace(evidence_for("pangit kaayo"), observed_at=WALL)
        self.assertFalse(
            fuse_event_evidence((evidence,), (), duration_seconds=1.0).deliver
        )
        loud_only = fuse_event_evidence(
            (evidence,), (tone_window(0.0),), duration_seconds=1.0,
        )
        self.assertFalse(loud_only.deliver)
        decision = fuse_event_evidence(
            (with_tone(evidence),), (tone_window(0.0),), duration_seconds=1.0,
        )
        self.assertTrue(decision.deliver)

    def test_hard_insults_threats_and_self_harm_keep_their_rules(self):
        self.assertEqual(sent("bobo"), (True, "LOW"))
        self.assertEqual(sent("papatayin kita"), (True, "HIGH"))
        self.assertEqual(sent("kill yourself"), (True, "HIGH"))
        self.assertEqual(sent("bunalan tika"), (True, "HIGH"))
        self.assertEqual(sent("atay ka"), (True, "MEDIUM"))


if __name__ == "__main__":
    unittest.main()
