from dataclasses import replace
import unittest

from detection.fusion import fuse_event_evidence
from tests.test_event_coordinator_after import acoustic_evidence, text_evidence


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
            (text_evidence(),),
            (),
            duration_seconds=5,
        )
        self.assertEqual(decision.level, "MEDIUM")

    def test_text_plus_strong_acoustic_corroboration_can_promote(self):
        decision = fuse_event_evidence(
            (text_evidence(),),
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
        repeated = replace(text_evidence(), repeated=True)
        decision = fuse_event_evidence((repeated,), (), duration_seconds=3)
        self.assertEqual(decision.level, "HIGH")
        self.assertIn("promoted_by:repetition", decision.reasons)

    def test_acoustic_only_is_conservatively_low_or_medium_never_high(self):
        weak = acoustic_evidence(score=0.46)
        strong = acoustic_evidence(score=0.82)
        self.assertEqual(
            fuse_event_evidence((), (weak,), duration_seconds=5).level,
            "LOW",
        )
        self.assertEqual(
            fuse_event_evidence((), (strong,), duration_seconds=5).level,
            "MEDIUM",
        )

    def test_single_loud_non_aggressive_sound_is_no_alert_evidence(self):
        loud = replace(
            acoustic_evidence(qualifies=False),
            rms=5_000.0,
        )
        with self.assertRaises(ValueError):
            fuse_event_evidence((), (loud,), duration_seconds=15)

    def test_sustained_multi_feature_acoustics_remain_medium(self):
        first = replace(
            acoustic_evidence(at=1, score=0.1),
            yamnet_ran=False,
            yamnet_label="NotRun",
            yamnet_aggressive=False,
            tone_aggressive=True,
            duration_seconds=0.975,
        )
        second = replace(first, monotonic_start=1.0, monotonic_end=2.0)
        decision = fuse_event_evidence(
            (),
            (first, second),
            duration_seconds=10,
        )
        self.assertEqual(decision.level, "MEDIUM")
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
