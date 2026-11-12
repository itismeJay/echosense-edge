"""Deterministic metrics for baseline/experimental held-out comparisons."""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics

from detection.evidence import SEMANTIC_LABELS
from detection.severity import contains_urgent_directive


SEMANTIC_ALERT_LABELS = frozenset({
    "targeted_belittling",
    "threat",
    "repeated_harassment",
})
SEMANTIC_NON_ALERT_LABELS = frozenset({
    "neutral",
    "academic_discussion",
    "playful_teasing",
})


@dataclass(frozen=True, slots=True)
class EvaluationObservation:
    case_id: str
    language: str
    condition: str
    expected_alert: bool | None
    baseline_alert: bool
    experimental_alert: bool
    duration_seconds: float
    baseline_latency_ms: float
    experimental_latency_ms: float
    category: str = "unspecified"
    acoustic_context: str = "none"
    expected_semantic_label: str | None = None
    predicted_semantic_label: str | None = None


def _percentile(values, percentile):
    values = sorted(float(value) for value in values)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def binary_metrics(observations, decision_field):
    """Calculate transparent binary metrics without zero-division fiction."""

    all_observations = tuple(observations)
    observations = tuple(
        item for item in all_observations if item.expected_alert is not None
    )
    tp = fp = tn = fn = 0
    for item in observations:
        predicted = bool(getattr(item, decision_field))
        if predicted and item.expected_alert:
            tp += 1
        elif predicted:
            fp += 1
        elif item.expected_alert:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None
        and precision + recall
        else None
    )
    accuracy = (tp + tn) / len(observations) if observations else None
    negative_seconds = sum(
        item.duration_seconds for item in observations if not item.expected_alert
    )
    false_alerts_per_hour = None
    # Less than one hour of negatives is too little exposure for a stable rate.
    if negative_seconds >= 3_600.0:
        false_alerts_per_hour = fp / (negative_seconds / 3_600.0)
    prefix = decision_field.removesuffix("_alert")
    latencies = [
        getattr(item, f"{prefix}_latency_ms") for item in observations
    ]
    return {
        "cases": len(observations),
        "unadjudicated_cases": len(all_observations) - len(observations),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "false_alerts_per_hour": false_alerts_per_hour,
        "negative_exposure_seconds": negative_seconds,
        "median_latency_ms": (
            statistics.median(latencies) if latencies else None
        ),
        "p95_latency_ms": _percentile(latencies, 0.95),
    }


def semantic_label_metrics(observations):
    """Report semantic accuracy without hiding unavailable predictions."""

    labeled = tuple(
        item for item in observations if item.expected_semantic_label is not None
    )
    predicted = tuple(
        item for item in labeled
        if item.predicted_semantic_label in SEMANTIC_LABELS
    )
    correct = sum(
        item.predicted_semantic_label == item.expected_semantic_label
        for item in labeled
    )
    covered_correct = sum(
        item.predicted_semantic_label == item.expected_semantic_label
        for item in predicted
    )
    confusion = {}
    for item in labeled:
        expected = item.expected_semantic_label
        actual = item.predicted_semantic_label or "unavailable"
        confusion.setdefault(expected, {})
        confusion[expected][actual] = confusion[expected].get(actual, 0) + 1
    return {
        "labeled_cases": len(labeled),
        "predicted_cases": len(predicted),
        "coverage": len(predicted) / len(labeled) if labeled else None,
        "accuracy_including_unavailable": (
            correct / len(labeled) if labeled else None
        ),
        "accuracy_when_predicted": (
            covered_correct / len(predicted) if predicted else None
        ),
        "confusion": confusion,
    }


def shadow_candidate_alert(text_evidence):
    """Offline-only candidate policy; never used by runtime fusion.

    Existing urgent threat/self-harm terms always remain alerts.  A classified
    semantic label may otherwise add or suppress a candidate alert so held-out
    evaluation can measure both recall and false-positive effects.  Uncertain,
    failed, disabled, or unavailable semantics fall back to the baseline.
    """

    baseline = bool(text_evidence.qualifies_to_open)
    if contains_urgent_directive(text_evidence.decision_terms):
        return True
    semantic = text_evidence.semantic
    if semantic is None or semantic.status != "classified":
        return baseline
    if semantic.label in SEMANTIC_ALERT_LABELS:
        return True
    if semantic.label in SEMANTIC_NON_ALERT_LABELS:
        return False
    return baseline


def shadow_candidate_case_alert(text_evidence, *, baseline_alert):
    """Apply the offline semantic proposal across one conversation.

    Positive classified evidence may add a candidate alert. Suppression is
    considered only when every utterance was classified into a non-alert label;
    any uncertainty falls back to the final fused baseline decision.
    """

    items = tuple(text_evidence)
    if any(contains_urgent_directive(item.decision_terms) for item in items):
        return True
    semantics = [item.semantic for item in items]
    classified = [
        semantic for semantic in semantics
        if semantic is not None and semantic.status == "classified"
    ]
    if any(item.label in SEMANTIC_ALERT_LABELS for item in classified):
        return True
    if (
        semantics
        and len(classified) == len(semantics)
        and all(item.label in SEMANTIC_NON_ALERT_LABELS for item in classified)
    ):
        return False
    return bool(baseline_alert)


def grouped_comparison(observations):
    """Report identical metrics overall and across evaluation strata."""

    observations = tuple(observations)
    groups = {"overall": observations}
    for dimension in ("language", "condition", "category", "acoustic_context"):
        values = sorted({getattr(item, dimension) for item in observations})
        for value in values:
            groups[f"{dimension}:{value}"] = tuple(
                item for item in observations
                if getattr(item, dimension) == value
            )
    return {
        group: {
            "baseline": binary_metrics(items, "baseline_alert"),
            "experimental": binary_metrics(items, "experimental_alert"),
            "semantic_labels": semantic_label_metrics(items),
        }
        for group, items in groups.items()
    }
