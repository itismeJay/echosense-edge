from dataclasses import replace
from datetime import timedelta
import unittest
from unittest.mock import patch

import numpy as np

from config import (
    SemanticRuntimeConfig,
    semantic_runtime_config_from_environment,
)
from detection.acoustic_detector import AcousticDetector
from detection.fusion import fuse_event_evidence
from detection.semantic import (
    SemanticAnalyzer,
    SemanticPrediction,
)
from detection.text_detector import TextDetector
from detection.upgrade_evaluation import (
    EvaluationObservation,
    grouped_comparison,
    shadow_candidate_alert,
    shadow_candidate_case_alert,
)
from model.yamnet_infer import is_laughter_sound
from tests.test_acoustic_detector_after import WALL, tone
from tests.test_event_coordinator_after import acoustic_evidence


class ScriptedClassifier:
    """Interface test double; it is not evidence of model accuracy."""

    def __init__(self, predictions):
        self.predictions = iter(predictions)
        self.requests = []

    def classify(self, request):
        self.requests.append(request)
        return next(self.predictions)


def prediction(label, confidence=0.9):
    return SemanticPrediction(
        label=label,
        confidence=confidence,
        model_id="scripted-interface-test-only",
        scores={label: confidence},
    )


class SemanticUpgradeTests(unittest.TestCase):
    def test_experimental_feature_is_disabled_by_default(self):
        self.assertFalse(SemanticRuntimeConfig().enabled)
        self.assertFalse(
            semantic_runtime_config_from_environment({}).enabled
        )
        enabled = semantic_runtime_config_from_environment({
            "ECHOSENSE_EXPERIMENTAL_SEMANTICS": "true",
        })
        self.assertTrue(enabled.enabled)

    def test_enabled_without_model_is_explicitly_unavailable(self):
        evidence = SemanticAnalyzer(enabled=True).assess(
            "A natural sentence",
            language="en",
            observed_monotonic=1.0,
            quality_accepted=True,
        )
        self.assertEqual(evidence.status, "unavailable")
        self.assertEqual(evidence.label, "uncertain")
        self.assertIn("semantic_classifier_not_installed", evidence.reasons)

    def test_context_is_bounded_ordered_and_ram_clearable(self):
        classifier = ScriptedClassifier([
            prediction("neutral"),
            prediction("targeted_belittling"),
            prediction("repeated_harassment"),
        ])
        analyzer = SemanticAnalyzer(
            classifier=classifier,
            enabled=True,
            max_context_turns=2,
            context_window_seconds=10.0,
            max_context_chars=100,
        )
        analyzer.assess(
            "first", language="en", observed_monotonic=1, quality_accepted=True
        )
        second = analyzer.assess(
            "ikaduha", language="ceb", observed_monotonic=2,
            quality_accepted=True,
        )
        third = analyzer.assess(
            "pangatlo", language="mixed", observed_monotonic=3,
            quality_accepted=True,
        )
        self.assertEqual(second.context_turns, 1)
        self.assertEqual(third.context_turns, 2)
        self.assertEqual(
            [turn.text for turn in classifier.requests[2].prior_turns],
            ["first", "ikaduha"],
        )
        analyzer.clear()
        self.assertEqual(len(analyzer._turns), 0)

    def test_low_confidence_is_not_treated_as_neutral(self):
        classifier = ScriptedClassifier([
            prediction("neutral", confidence=0.51),
        ])
        evidence = SemanticAnalyzer(
            classifier=classifier,
            enabled=True,
            min_confidence=0.75,
        ).assess(
            "ambiguous", language="mixed", observed_monotonic=1,
            quality_accepted=True,
        )
        self.assertEqual(evidence.status, "low_confidence")
        self.assertEqual(evidence.label, "uncertain")
        self.assertEqual(evidence.candidate_label, "neutral")

    def test_classifier_failure_is_bounded_and_explicit(self):
        class Broken:
            def classify(self, _request):
                raise RuntimeError("test failure")

        evidence = SemanticAnalyzer(
            classifier=Broken(), enabled=True
        ).assess(
            "text", language="en", observed_monotonic=1,
            quality_accepted=True,
        )
        self.assertEqual(evidence.status, "error")
        self.assertEqual(evidence.label, "uncertain")
        self.assertIn("classifier_error:RuntimeError", evidence.reasons)

    def test_shadow_classifier_cannot_change_baseline_alert_decisions(self):
        # These cases cover current monitored-term behavior in four language
        # groups. Scripted outputs exercise integration only, not AI accuracy.
        cases = (
            ("Please open your book", "en"),
            ("bogo ka", "ceb"),
            ("bobo ka", "fil"),
            ("you are bogo talaga", "mixed"),
        )
        classifier = ScriptedClassifier([
            prediction("targeted_belittling") for _case in cases
        ])
        baseline = TextDetector()
        experimental = TextDetector(semantic_analyzer=SemanticAnalyzer(
            classifier=classifier,
            enabled=True,
        ))
        for index, (text, language) in enumerate(cases):
            with self.subTest(language=language):
                before = baseline.analyze_transcript(
                    text, language=language, observed_monotonic=index + 1
                )
                after = experimental.analyze_transcript(
                    text, language=language, observed_monotonic=index + 1
                )
                self.assertEqual(
                    after.qualifies_to_open, before.qualifies_to_open
                )
                self.assertEqual(after.baseline_severity, before.baseline_severity)
                self.assertEqual(after.decision_terms, before.decision_terms)
                self.assertEqual(after.semantic.status, "classified")

    def test_semantic_prediction_alone_cannot_open_an_incident(self):
        classifier = ScriptedClassifier([prediction("threat", 0.99)])
        evidence = TextDetector(semantic_analyzer=SemanticAnalyzer(
            classifier=classifier,
            enabled=True,
        )).analyze_transcript(
            "An unfamiliar indirect sentence with no monitored term",
            language="en",
        )
        self.assertEqual(evidence.semantic.label, "threat")
        self.assertFalse(evidence.qualifies_to_open)

    def test_existing_high_priority_threat_is_not_regressed(self):
        classifier = ScriptedClassifier([prediction("neutral", 0.99)])
        evidence = TextDetector(semantic_analyzer=SemanticAnalyzer(
            classifier=classifier,
            enabled=True,
        )).analyze_transcript("kill yourself", language="en")
        self.assertTrue(evidence.qualifies_to_open)
        self.assertEqual(evidence.baseline_severity, "HIGH")
        self.assertTrue(shadow_candidate_alert(evidence))

    def test_offline_candidate_can_be_measured_without_runtime_effect(self):
        classifier = ScriptedClassifier([
            prediction("targeted_belittling", 0.99),
            prediction("neutral", 0.99),
        ])
        detector = TextDetector(semantic_analyzer=SemanticAnalyzer(
            classifier=classifier,
            enabled=True,
        ))
        unseen = detector.analyze_transcript(
            "Nobody asked you to sit with us.", language="en"
        )
        baseline_positive = detector.analyze_transcript(
            "bobo ka", language="fil"
        )
        self.assertFalse(unseen.qualifies_to_open)
        self.assertTrue(shadow_candidate_alert(unseen))
        self.assertTrue(baseline_positive.qualifies_to_open)
        self.assertFalse(shadow_candidate_alert(baseline_positive))
        # The proposal is an evaluator function; evidence remains unchanged.
        self.assertFalse(unseen.qualifies_to_open)
        self.assertTrue(baseline_positive.qualifies_to_open)
        self.assertTrue(shadow_candidate_case_alert(
            (unseen,), baseline_alert=False
        ))
        self.assertFalse(shadow_candidate_case_alert(
            (baseline_positive,), baseline_alert=True
        ))

    def test_acoustic_laughter_is_observed_but_never_qualifies(self):
        detector = AcousticDetector(
            yamnet_runner=lambda _samples: ("Laughter", 0.82),
        )
        samples = np.zeros(15_600, dtype=np.float32)

        with patch(
            "detection.acoustic_detector.analyze_tone_float32",
            return_value=tone(aggressive=False),
        ):
            evidence = detector.analyze_window(
                samples,
                interval_started_at=WALL,
                interval_ended_at=WALL + timedelta(seconds=0.975),
                monotonic_start=1.0,
                monotonic_end=1.975,
            )
        self.assertTrue(evidence.laughter_detected)
        self.assertEqual(evidence.laughter_score, 0.82)
        self.assertFalse(evidence.qualifies_to_open)
        self.assertIn("observed:acoustic_laughter", evidence.reasons)

    def test_laughter_exact_label_policy_and_fusion_do_not_change_severity(self):
        self.assertTrue(is_laughter_sound("Laughter", 0.8, 0.45))
        self.assertFalse(is_laughter_sound("Baby laughter", 0.99, 0.45))
        text = TextDetector().analyze_transcript(
            "kill yourself", language="en", observed_at=WALL,
            observed_monotonic=2.0,
        )
        laughter = replace(
            acoustic_evidence(at=2.0, qualifies=False),
            interval_started_at=WALL,
            interval_ended_at=WALL + timedelta(seconds=1),
            yamnet_label="Laughter",
            yamnet_score=0.9,
            yamnet_aggressive=False,
            laughter_detected=True,
            laughter_score=0.9,
        )
        decision = fuse_event_evidence((text,), (laughter,), duration_seconds=1)
        self.assertEqual(decision.level, "HIGH")
        self.assertIn(
            "acoustic_laughter_observed", decision.supporting_evidence
        )

    def test_evaluator_reports_regressions_and_withholds_short_hourly_rate(self):
        observations = (
            EvaluationObservation(
                case_id="one",
                language="ceb",
                condition="near",
                expected_alert=True,
                baseline_alert=True,
                experimental_alert=False,
                duration_seconds=10,
                baseline_latency_ms=5,
                experimental_latency_ms=8,
            ),
            EvaluationObservation(
                case_id="two",
                language="ceb",
                condition="near",
                expected_alert=False,
                baseline_alert=True,
                experimental_alert=False,
                duration_seconds=10,
                baseline_latency_ms=7,
                experimental_latency_ms=10,
            ),
        )
        report = grouped_comparison(observations)["overall"]
        self.assertEqual(report["baseline"]["false_positives"], 1)
        self.assertEqual(report["experimental"]["false_negatives"], 1)
        self.assertIsNone(report["baseline"]["false_alerts_per_hour"])
        self.assertEqual(report["experimental"]["median_latency_ms"], 9)
        self.assertEqual(report["baseline"]["accuracy"], 0.5)

    def test_evaluator_reports_category_and_semantic_label_accuracy(self):
        observations = (
            EvaluationObservation(
                case_id="natural-cebano",
                language="ceb",
                condition="clean",
                expected_alert=True,
                baseline_alert=False,
                experimental_alert=True,
                duration_seconds=2,
                baseline_latency_ms=1,
                experimental_latency_ms=3,
                category="indirect_belittling",
                expected_semantic_label="targeted_belittling",
                predicted_semantic_label="targeted_belittling",
            ),
            EvaluationObservation(
                case_id="academic-tagalog",
                language="fil",
                condition="clean",
                expected_alert=False,
                baseline_alert=False,
                experimental_alert=False,
                duration_seconds=2,
                baseline_latency_ms=1,
                experimental_latency_ms=3,
                category="academic_discussion",
                expected_semantic_label="academic_discussion",
                predicted_semantic_label=None,
            ),
        )
        report = grouped_comparison(observations)
        self.assertIn("category:indirect_belittling", report)
        semantic = report["overall"]["semantic_labels"]
        self.assertEqual(semantic["coverage"], 0.5)
        self.assertEqual(semantic["accuracy_including_unavailable"], 0.5)
        self.assertEqual(semantic["accuracy_when_predicted"], 1.0)


if __name__ == "__main__":
    unittest.main()
