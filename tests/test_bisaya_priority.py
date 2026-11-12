"""Bisaya is the deployment language: test-set transcripts must alert or stay quiet."""

import unittest

from detection.transcript_quality import normalize_transcript
from model.blacklist import check_transcript
from tests.test_soundalike import sent


class BisayaPriorityTests(unittest.TestCase):
    def test_accents_from_bisaya_tuned_models_are_folded(self):
        self.assertEqual(normalize_transcript("Gágà ka. Pangít!"), "gaga ka pangit")
        # Non-Latin marks are kept.
        self.assertEqual(normalize_transcript("नमस्ते"), "नमस्ते")

    def test_test_set_bisaya_insults_alert(self):
        # Transcripts as Whisper produced them for the recorded test set.
        for heard in (
            "gágà ka.", "Gaga ka.", "atáy ka.", "Pangít ka.", "Buóang ka.",
            "Bogo ka.", "baho ka.", "Pística.", "Patyúntica", "Banalanti ka.",
            "Binalántica", "animalka", "bunalan tika", "Bugo ka.",
        ):
            with self.subTest(heard=heard):
                delivered, _level = sent(heard)
                self.assertTrue(delivered, check_transcript(heard))

    def test_bisaya_threats_are_high(self):
        for heard in (
            "Tadyakan tika!", "sagpaon taka", "Kulatahon tika", "sumbagon taka",
            "Bunalan taka.", "Patyon tika.",
        ):
            with self.subTest(heard=heard):
                self.assertEqual(sent(heard), (True, "HIGH"))

    def test_new_directed_insults_alert(self):
        for heard in (
            "Amaw ka!", "maot ka", "luod ka", "mananap ka", "bati kaayo ka",
            "ulaga ka", "pisti ka", "Boang ka!", "Buanga!", "Animala ka",
        ):
            with self.subTest(heard=heard):
                self.assertTrue(check_transcript(heard)["detected_words"], heard)

    def test_split_kaayo_is_not_a_directed_insult(self):
        # Whisper writes "pangit kaayo" (very bad) as "pangit ka ayo".
        for heard in (
            "Pangit ka ayo ang panahon karon.",
            "Pangít-ga áyo ang panahón karón.",
            "pangit kaayo ang panahon karon",
        ):
            with self.subTest(heard=heard):
                self.assertFalse(sent(heard)[0], check_transcript(heard))

    def test_bisaya_everyday_words_near_new_terms_stay_quiet(self):
        for heard in (
            "mga mananap sa lasang", "Pista na ugma", "patay na ang suga",
            "Adtuta sa Cantin.", "Human namo.", "Maáyong buntag, Klas.",
            "Ayáw kalímpi ang asáinment.", "Pangita-a ang ímong libro.",
            "Hulam ko sa imong lápis.", "maot ang panahon",
        ):
            with self.subTest(heard=heard):
                self.assertFalse(sent(heard)[0], check_transcript(heard))


if __name__ == "__main__":
    unittest.main()


# Grade 6 Bisaya list (school plan). (phrase, expected severity when sent)
GRADE6 = [
    ("tambok ka", "MEDIUM"), ("baboy ka", "MEDIUM"), ("pangit ka", "MEDIUM"),
    ("itom ka", "MEDIUM"), ("pandak ka", "MEDIUM"), ("niwang ka", "MEDIUM"),
    ("bungi ka", "MEDIUM"), ("dakog ilong ka", "MEDIUM"), ("dakog tiyan ka", "MEDIUM"),
    ("buktot ka", "MEDIUM"), ("baho ka", "MEDIUM"), ("baho imong baba", "MEDIUM"),
    ("hugaw ka", "MEDIUM"), ("kutohon ka", "MEDIUM"), ("bisaya kaayo ka", "MEDIUM"),
    ("bogo ka", "MEDIUM"), ("buang ka", "MEDIUM"), ("buwang", "LOW"),
    ("walay utok", "LOW"), ("hinay ug utok", "LOW"), ("abnoy ka", "MEDIUM"),
    ("bobo ka", "MEDIUM"), ("tanga ka", "MEDIUM"), ("ungo ka", "MEDIUM"),
    ("pobre ka", "MEDIUM"), ("walay kwarta ka", "MEDIUM"), ("ilo ka", "MEDIUM"),
    ("anak sa pobre", "LOW"), ("layo diri ka", "MEDIUM"),
    ("ayaw ug duol nako ikaw", "MEDIUM"), ("walay ganahan nimo", "MEDIUM"),
    ("dili ka apil", "MEDIUM"), ("walay pulos ka", "MEDIUM"),
    ("sipsip ka", "MEDIUM"), ("salingkapaw ka", "MEDIUM"),
    ("tikasan ka", "MEDIUM"), ("yawa ka", "MEDIUM"), ("animal ka", "MEDIUM"),
    ("piste ka", "MEDIUM"), ("atay ka", "MEDIUM"), ("gaga ka", "MEDIUM"),
    ("gago ka", "MEDIUM"), ("leche ka", "MEDIUM"), ("bayot ka", "MEDIUM"),
    ("patyon tika", "HIGH"), ("bunalan tika", "HIGH"), ("sumbagon tika", "HIGH"),
    ("sipaon tika", "HIGH"), ("hulata ko sa gawas", "HIGH"),
    ("ayaw pagsumbong", "HIGH"), ("magpakamatay ka", "HIGH"),
]

# Everyday Bisaya built from the same words. None may be sent.
GRADE6_EVERYDAY = [
    "itom nga lapis", "tambok nga manok", "pangitaa ang libro",
    "adobong baboy", "baboy nga sinugba", "bungi", "tambok",
    "pandak man ang lamesa", "hugaw ang salog", "pobre ang pamilya nila",
    "niwang ang iro", "ilo na siya", "layo diri ang balay",
    "walay kwarta ang canteen", "walay kwarta ko karon",
    "ayaw ug duol sa kalayo", "sipsip sa juice", "ungo sa sine", "leche flan",
    "taga asa ka", "ilo ang bata sa istorya", "dakog tiyan ang iro",
    "hulata ko sa canteen", "sipa ang bola", "bisaya kaayo ang kanta",
    "tikas kaayo siya",
]


class Grade6BisayaTests(unittest.TestCase):
    def test_grade6_list_is_sent_with_policy_severity(self):
        for phrase, level in GRADE6:
            with self.subTest(phrase=phrase):
                self.assertEqual(sent(phrase), (True, level), check_transcript(phrase))

    def test_everyday_sentences_with_the_same_words_are_not_sent(self):
        for sentence in GRADE6_EVERYDAY:
            with self.subTest(sentence=sentence):
                self.assertFalse(sent(sentence)[0], check_transcript(sentence))


class Grade6RecordingTests(unittest.TestCase):
    """Transcripts the Bisaya model produced for the Grade 6 recordings."""

    def test_recorded_mishearings_alert(self):
        for heard, level in (
            ("Súmbag gúnte ka.", "HIGH"), ("Sipánte ka.", "HIGH"),
            ("Banalante", "HIGH"), ("Lécheque", "MEDIUM"),
            ("Sa lingkapaw ka.", "MEDIUM"), ("Palayo diricaw", "MEDIUM"),
            ("Póbrika.", "MEDIUM"), ("Pandacas", "MEDIUM"),
            ("Ayáw ug duál nákò.", "LOW"),
        ):
            with self.subTest(heard=heard):
                self.assertEqual(sent(heard), (True, level), check_transcript(heard))

    def test_recorded_everyday_sentences_stay_quiet(self):
        for heard in ("Dakúg tiyán ang írò.", "palayo diri ang balay"):
            with self.subTest(heard=heard):
                self.assertFalse(sent(heard)[0], check_transcript(heard))

    def test_target_word_confidence_does_not_hide_a_clear_insult(self):
        # Measured: a clear "Ilo ka." scored ilo=0.83, ka=0.13.
        from detection.false_alarm_filters import term_probability
        from tests.test_false_alarm_filters import metrics

        words = metrics([(" Ilo", 0.83), (" ka.", 0.13)]).words
        self.assertAlmostEqual(term_probability("ilo ka", words), 0.83)
        # Threat pronouns stay scored: they are part of the threat.
        words = metrics([(" papatayin", 0.92), (" kita", 0.41)]).words
        self.assertAlmostEqual(term_probability("papatayin kita", words), 0.41)


class ResplitTests(unittest.TestCase):
    def test_same_letters_split_differently_match(self):
        # Live, 2026-11-04: "sumbagon tika" decoded as "Súmbagónti ka."
        for heard in ("Súmbagónti ka.", "patyonti ka", "bunalanti ka"):
            with self.subTest(heard=heard):
                self.assertEqual(sent(heard), (True, "HIGH"), check_transcript(heard))

    def test_resplit_never_changes_a_pronoun(self):
        self.assertNotIn("bobo ka", check_transcript("bobo ko")["detected_words"])
