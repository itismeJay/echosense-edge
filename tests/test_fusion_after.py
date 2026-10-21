from dataclasses import replace
import unittest

from config import DetectionRuntimeConfig
from detection.fusion import fuse_event_evidence
from tests.test_event_coordinator_after import acoustic_evidence, text_evidence


def directed(at=0.0):
    """MEDIUM text evidence: a medium term with direct-target evidence."""

    return replace(
        text_evidence(at),
        transcript="bobo ka",
        normalized_transcript="bobo ka",
    )


def tone_window(end, *, aggressive=True):
    return replace(
        acoustic_evidence(at=end, score=0.1, qualifies=aggressive),
        yamnet_ran=False,
        yamnet_label="NotRun",
        yamnet_aggressive=False,
        tone_aggressive=aggressive,
    )


class FusionAfterTests(unittest.TestCase):
    def test_strong_text_only_remains_high(self):
        evidence = replace(
            text_evidence(),
            transcript="kill yourself",
            normalized_transcript="kill yourself",
            detected_terms=("kill yourself",),
            hard_hits=("kill yourself",),
            baseline_severity="HIGH",
        )
        decision = fuse_event_evidence((evidence,), (), duration_seconds=5)
        self.assertEqual(decision.level, "HIGH")
        self.assertIn("term_category:self_harm_directive", decision.reasons)

    def test_moderate_text_only_remains_medium(self):
        decision = fuse_event_evidence(
            (directed(),),
            (),
            duration_seconds=5,
        )
        self.assertEqual(decision.level, "MEDIUM")

    def test_text_plus_strong_acoustic_corroboration_can_promote(self):
        decision = fuse_event_evidence(
            (directed(),),
            (acoustic_evidence(score=0.82),),
            duration_seconds=5,
        )
        self.assertEqual(decision.level, "HIGH")
        self.assertIn(
            "promoted_by:strong_acoustic_corroboration",
            decision.reasons,
        )
        self.assertIn("text_acoustic_corroboration", decision.supporting_evidence)

    def test_repeated_text_is_explainably_promoted(self):
        repeated = replace(directed(), repeated=True)
        decision = fuse_event_evidence((repeated,), (), duration_seconds=3)
        self.assertEqual(decision.level, "HIGH")
        self.assertIn("promoted_by:repetition", decision.reasons)

    def test_acoustic_only_defaults_to_low_only(self):
        weak = acoustic_evidence(score=0.46)
        strong = acoustic_evidence(score=0.82)
        for evidence in (weak, strong):
            decision = fuse_event_evidence((), (evidence,), duration_seconds=5)
            self.assertEqual(decision.level, "LOW")
            self.assertTrue(decision.deliver)
            self.assertIn("acoustic_only_policy:low_only", decision.reasons)

    def test_acoustic_only_on_needs_strong_and_sustained_for_medium(self):
        config = DetectionRuntimeConfig(acoustic_only_alerts="on")
        short = (acoustic_evidence(at=1, score=0.82),)
        sustained = tuple(
            acoustic_evidence(at=1 + 0.4875 * index, score=0.82)
            for index in range(7)
        )
        self.assertEqual(
            fuse_event_evidence((), short, duration_seconds=5, config=config).level,
            "LOW",
        )
        decision = fuse_event_evidence(
            (), sustained, duration_seconds=5, config=config,
        )
        self.assertEqual(decision.level, "MEDIUM")
        self.assertIn("acoustic_only:strong_and_sustained", decision.reasons)

    def test_acoustic_only_off_is_not_delivered(self):
        config = DetectionRuntimeConfig(acoustic_only_alerts="off")
        decision = fuse_event_evidence(
            (), (acoustic_evidence(score=0.9),), duration_seconds=5, config=config,
        )
        self.assertFalse(decision.deliver)

    def test_single_loud_non_aggressive_sound_is_no_alert_evidence(self):
        loud = replace(
            acoustic_evidence(qualifies=False),
            rms=5_000.0,
        )
        with self.assertRaises(ValueError):
            fuse_event_evidence((), (loud,), duration_seconds=15)

    def test_sustained_multi_feature_acoustics_are_strong_but_capped(self):
        first = replace(tone_window(1), duration_seconds=0.975)
        second = replace(first, monotonic_start=1.0, monotonic_end=2.0)
        decision = fuse_event_evidence(
            (),
            (first, second),
            duration_seconds=10,
        )
        self.assertEqual(decision.level, "LOW")
        self.assertIn(
            "strong_or_sustained_acoustic_evidence",
            decision.supporting_evidence,
        )

    def test_harmless_text_and_normal_audio_have_no_fusion_result(self):
        with self.assertRaises(ValueError):
            fuse_event_evidence(
                (text_evidence(qualifies=False, transcript="hello"),),
                (acoustic_evidence(qualifies=False),),
                duration_seconds=5,
            )


if __name__ == "__main__":
    unittest.main()
