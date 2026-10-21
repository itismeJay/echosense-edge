"""End-to-end severity policy: no monitored word, no alert.

Real TextDetector + EventCoordinator + fusion, one fresh room per case.
"""

import unittest

from detection.severity import HIGH, LOW, MEDIUM
from tests.test_accuracy_context import CoordinatorHarness, windows
from tests.test_duplicate_guard_and_tone import Speech


def run(transcript, *, shouted=False):
    """Return (finalized incidents, finalized-during-submit) for one utterance."""

    harness = CoordinatorHarness()
    say = Speech()
    if shouted:
        for window in windows(0.5, 2.0):
            harness.acoustic(window)
    submission = None
    if transcript is not None:
        submission = harness.text(say(transcript, 2.0, spoken_seconds=1.5))
    immediate = tuple(submission.finalized) if submission else ()
    return harness.finish(30), immediate


class SeverityPolicyTableTests(unittest.TestCase):
    TABLE = (
        ("bobo", False, LOW, "held_low:no_aggression"),
        ("ang bobo ko", False, LOW, "held_low:self_directed"),
        ("stupid game", False, LOW, "held_low:about_object"),
        ("bobo ka", False, MEDIUM, "promoted_by:direct_target_pattern"),
        ("TANGA!", True, MEDIUM, "promoted_by:time_linked_aggressive_tone"),
        ("BOBO KA!", True, HIGH, "promoted_by:direct_target_and_strong_shout"),
        ("bobo ka, bobo ka talaga", False, HIGH, "promoted_by:targeted_repetition"),
        ("kill you", False, HIGH, "high:threat_or_self_harm_term"),
    )

    def test_policy_table(self):
        for transcript, shouted, level, reason in self.TABLE:
            with self.subTest(transcript=transcript, shouted=shouted):
                finalized, _immediate = run(transcript, shouted=shouted)
                self.assertEqual(len(finalized), 1)
                self.assertEqual(finalized[0].severity, level)
                self.assertIn(reason, finalized[0].severity_reasons)
                self.assertTrue(finalized[0].severity_reasons)

    def test_kill_you_is_enqueued_immediately(self):
        _finalized, immediate = run("kill you")
        self.assertEqual(len(immediate), 1)
        self.assertEqual(immediate[0].severity, HIGH)

    def test_shouting_with_no_words_is_no_alert(self):
        finalized, _immediate = run(None, shouted=True)
        self.assertEqual(finalized, [])

    def test_shouting_normal_words_is_no_alert(self):
        finalized, _immediate = run("ANO BA!", shouted=True)
        self.assertEqual(finalized, [])

    def test_laughter_never_lowers_a_threat(self):
        finalized, _immediate = run("kill you haha")
        self.assertEqual(finalized[0].severity, HIGH)

    def test_laughing_insult_is_low(self):
        finalized, _immediate = run("bobo ka haha")
        self.assertEqual(finalized[0].severity, LOW)
        self.assertIn("held_low:playful_context", finalized[0].severity_reasons)

    def test_two_targeted_utterances_in_one_incident_are_high(self):
        harness = CoordinatorHarness()
        say = Speech()
        harness.text(say("bobo ka", 1.0))
        harness.text(say("tanga ka", 2.0))
        finalized = harness.finish(30)
        self.assertEqual(len(finalized), 1)
        self.assertEqual(finalized[0].severity, HIGH)
        self.assertIn(
            "targeted_repetition_within_incident",
            finalized[0].supporting_evidence,
        )

    def test_untargeted_repeat_in_incident_stays_low(self):
        harness = CoordinatorHarness()
        say = Speech()
        harness.text(say("bobo", 1.0))
        harness.text(say("bobo talaga", 2.0))
        self.assertEqual(harness.finish(30)[0].severity, LOW)


if __name__ == "__main__":
    unittest.main()
