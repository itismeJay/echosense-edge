"""Regression coverage for correction provenance and conservative alerting."""

from datetime import timedelta
import unittest

from tests.tone_helpers import with_tone

from detection.coordinator import EventCoordinator
from detection.fusion import fuse_event_evidence
from detection.text_detector import TextDetector
from model.blacklist import PHONETIC_VARIANTS, check_transcript
from model.monitored_terms import replace_backend_terms
from tests.test_event_coordinator_after import (
    EVENT_ID,
    WALL,
    FakeClock,
    acoustic_evidence,
)


def evidence_for(text, *, at=0.0, detector=None):
    return (detector or TextDetector()).analyze_transcript(
        text,
        observed_at=WALL + timedelta(seconds=at),
        observed_monotonic=at,
    )


def text_only_delivers(text):
    evidence = evidence_for(text)
    if not evidence.qualifies_to_open:
        return False
    return fuse_event_evidence(
        (evidence,), (), duration_seconds=1.0,
    ).deliver


class MatchProvenanceTests(unittest.TestCase):
    def setUp(self):
        replace_backend_terms([])

    def tearDown(self):
        replace_backend_terms([])

    def test_original_exact_match_is_recorded_separately_from_match_type(self):
        result = check_transcript("Bogo ka.")
        provenance = {
            item["term"]: item["provenance"]
            for item in result["term_provenance"]
        }
        self.assertEqual(provenance["bogo ka"], "original_exact")
        self.assertEqual(result["matched_terms"][0]["match_type"], "phrase")
        evidence = evidence_for("Bogo ka.")
        self.assertTrue(evidence.qualifies_to_open)

    def test_safe_spelling_alias_is_trusted_but_preserves_original(self):
        evidence = evidence_for("Patyun tika.")
        self.assertEqual(evidence.transcript, "Patyun tika.")
        self.assertEqual(evidence.match_text, "patyon tika")
        self.assertEqual(
            {item.term: item.provenance for item in evidence.term_provenance}[
                "patyon tika"
            ],
            "trusted_alias",
        )
        self.assertTrue(evidence.qualifies_to_open)
        self.assertEqual(evidence.baseline_severity, "HIGH")

    def test_curated_mishearing_is_distinct_from_fuzzy_evidence(self):
        evidence = evidence_for("Bang itka.")
        self.assertEqual(evidence.corrections[0].provenance, "explicit_mishearing")
        self.assertTrue(evidence.direct_target)
        self.assertTrue(evidence.qualifies_to_open)

    def test_fuzzy_only_threat_does_not_open_an_incident_by_itself(self):
        evidence = evidence_for("patyon tikan")
        self.assertEqual(evidence.match_text, "patyon tika")
        self.assertEqual(evidence.corrections[0].provenance, "fuzzy_soundalike")
        self.assertEqual(evidence.detected_terms, ("patyon tika",))
        self.assertEqual(evidence.qualification_terms, ())
        self.assertFalse(evidence.qualifies_to_open)

    def test_repeated_fuzzy_evidence_can_be_marked_for_review(self):
        detector = TextDetector()
        first = evidence_for("patyon tikan", at=0, detector=detector)
        second = evidence_for("patyon tikan", at=1, detector=detector)
        self.assertFalse(first.qualifies_to_open)
        self.assertTrue(second.repeated)
        self.assertTrue(second.qualifies_to_open)

    def test_similarity_window_match_is_fuzzy_and_capped_low(self):
        evidence = evidence_for("Tambong gyud ka.")
        self.assertEqual(evidence.transcript, "Tambong gyud ka.")
        provenance = {
            item.term: item.provenance for item in evidence.term_provenance
        }
        self.assertEqual(provenance["tambok gyud ka"], "fuzzy")
        self.assertEqual(set(provenance.values()), {"fuzzy"})
        self.assertTrue(evidence.qualifies_to_open)
        self.assertEqual(evidence.baseline_severity, "LOW")
        decision = fuse_event_evidence((evidence,), (), duration_seconds=1.0)
        self.assertEqual(decision.level, "LOW")
        self.assertIn("fuzzy_match:capped_low_needs_review", decision.reasons)

    def test_fuzzy_cap_lifts_when_exact_evidence_is_also_heard(self):
        evidence = evidence_for("Tambong gyud ka, bobo ka!")
        decision = fuse_event_evidence((evidence,), (), duration_seconds=1.0)
        self.assertNotEqual(decision.level, "LOW")
        self.assertNotIn("fuzzy_match:capped_low_needs_review", decision.reasons)

    def test_similarity_window_leaves_close_everyday_words_alone(self):
        for text in (
            "Ang kulit mo, hahaha!",       # 94% like "ang liit mo" overall
            "Bai, maayo kaayo ka mokanta!",
            "Grabe, ang init sa classroom.",
            "Kinsa may absent karon?",
            "Congrats, you're top of the class!",
            "kabus nga pamilya sa istorya",  # not "kabus ka"
            "Nobody wants to hear you talk!",  # "to" is not "you"
        ):
            corrections = check_transcript(text)["corrections"]
            self.assertFalse(
                [c for c in corrections if c["provenance"] == "fuzzy"], text,
            )

    def test_repeated_tail_does_not_hide_the_insult_before_it(self):
        text = "Ang pangit mo ang mga mga mga mga mga mga mga."
        evidence = evidence_for(text)
        self.assertEqual(evidence.transcript, text)
        self.assertTrue(evidence.qualifies_to_open)
        self.assertIn("ang pangit mo", evidence.detected_terms)

    def test_backend_term_keeps_id_and_original_provenance(self):
        replace_backend_terms([{
            "term_id": 77,
            "slur_text": "classroom test term",
            "language": "English",
        }])
        result = check_transcript("Classroom test term.")
        self.assertEqual(result["matched_terms"][0]["term_id"], 77)
        self.assertEqual(
            result["term_provenance"],
            [{"term": "classroom test term", "provenance": "original_exact"}],
        )


class IyotRegressionTests(unittest.TestCase):
    def test_unsafe_buwan_mapping_is_removed(self):
        self.assertNotIn("buwan", PHONETIC_VARIANTS)
        result = check_transcript("buwan na sa sunod")
        self.assertEqual(result["corrections"], [])
        self.assertEqual(result["detected_words"], [])

    def test_kadiyot_does_not_match_iyot_inside_a_word(self):
        result = check_transcript("kadiyot lang")
        self.assertNotIn("iyot", result["detected_words"])

    def test_exact_iyot_is_observed_but_not_delivered_without_support(self):
        evidence = evidence_for("iyot")
        self.assertIn("iyot", evidence.detected_terms)
        self.assertTrue(evidence.qualifies_to_open)
        decision = fuse_event_evidence((evidence,), (), duration_seconds=1.0)
        self.assertFalse(decision.deliver)
        self.assertIn("ambiguous_short_term:needs_support", decision.reasons)

    def test_exact_iyot_said_angrily_remains_reviewable(self):
        evidence = with_tone(evidence_for("iyot"))
        decision = fuse_event_evidence(
            (evidence,),
            (acoustic_evidence(at=0, score=0.82),),
            duration_seconds=1.0,
        )
        self.assertTrue(decision.deliver)
        self.assertIn("text_acoustic_corroboration", decision.supporting_evidence)

    def test_coordinator_suppresses_unsupported_exact_iyot(self):
        clock = FakeClock()
        finalized = []
        coordinator = EventCoordinator(
            monotonic_clock=clock,
            event_id_factory=lambda: EVENT_ID,
            finalizer=lambda incident: finalized.append(incident) or True,
        )
        submission = coordinator.submit_text(evidence_for("iyot"))
        self.assertTrue(submission.opened)
        clock.value = 5.0
        coordinator.tick()
        self.assertIsNone(coordinator.active_incident)
        self.assertEqual(finalized, [])


class HarmlessReferenceTests(unittest.TestCase):
    def test_academic_mentions_are_suppressed(self):
        for text in (
            'The word "idiot" is offensive.',
            "Ang salitang bobo ay isang insulto.",
            "Ang pulong yawa kay daotan.",
            "The Cebuano word iyot is profanity.",
        ):
            with self.subTest(text=text):
                evidence = evidence_for(text)
                self.assertFalse(evidence.qualifies_to_open)

    def test_direct_use_of_same_terms_is_not_suppressed(self):
        for text in ("You are an idiot", "bobo ka", "yawa ka"):
            with self.subTest(text=text):
                self.assertTrue(evidence_for(text).qualifies_to_open)


class LabeledTranscriptFixtureTests(unittest.TestCase):
    NEGATIVE = (
        "buwan na sa sunod",
        "pangita ang imong bag",
        "kadiyot lang",
        "maayong buntag class",
        "magandang umaga po",
        "please open your books",
        "Please ayaw kalimti your assignment",
        'The word "idiot" is offensive.',
        "Ang salitang bobo ay isang insulto.",
        "iyot",
        "patyon tikan",
    )
    POSITIVE = (
        "bogo ka",
        "bobo ka",
        "you are stupid",
        "patyon tika",
        "papatayin kita",
        "kill yourself",
        "Patyun tika",
        "Bang itka",
    )

    def test_labeled_text_fixture_has_no_false_results(self):
        false_positives = [
            text for text in self.NEGATIVE if text_only_delivers(text)
        ]
        false_negatives = [
            text for text in self.POSITIVE if not text_only_delivers(text)
        ]
        self.assertEqual(false_positives, [])
        self.assertEqual(false_negatives, [])


if __name__ == "__main__":
    unittest.main()
