#!/usr/bin/env python3
"""Compare baseline and semantic-shadow paths on consenting held-out JSONL.

No model is downloaded. Without ``--adapter``, the report explicitly marks the
semantic model unavailable. An adapter is a local ``module:factory`` whose
factory returns an object implementing detection.semantic.SemanticClassifier.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import importlib
import json
import resource
from pathlib import Path
import sysconfig
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detection.semantic import SemanticAnalyzer  # noqa: E402
from detection.fusion import fuse_event_evidence  # noqa: E402
from detection.text_detector import TextDetector  # noqa: E402
from detection.upgrade_evaluation import (  # noqa: E402
    EvaluationObservation,
    grouped_comparison,
    shadow_candidate_case_alert,
)


EVALUATION_CATEGORIES = frozenset({
    "bullying",
    "normal",
    "happy",
    "other",
    "mixed",
    "neutral_conversation",
    "academic_discussion",
    "playful_teasing",
    "targeted_insult",
    "threat",
    "indirect_belittling",
    "repeated_harassment",
    "uncertain",
})
ACOUSTIC_CONTEXTS = frozenset({
    "none",
    "aggression_observed",
    "laughter_observed",
    "aggression_and_laughter_observed",
})
SEMANTIC_LABELS = frozenset({
    "neutral",
    "academic_discussion",
    "playful_teasing",
    "targeted_belittling",
    "threat",
    "repeated_harassment",
    "uncertain",
})
PROVISIONAL_LABELS = frozenset({
    "potential_targeted_aggression",
    "clear_non_aggressive",
    "context_dependent",
    "reported_or_quoted_aggression",
    "threat_or_intimidation",
    "boundary_or_defensive",
})


def _adapter(value):
    if not value:
        return None
    module_name, separator, factory_name = value.partition(":")
    if not separator or not module_name or not factory_name:
        raise ValueError("adapter must use module:factory syntax")
    factory = getattr(importlib.import_module(module_name), factory_name)
    classifier = factory()
    if not callable(getattr(classifier, "classify", None)):
        raise TypeError("adapter factory did not return a semantic classifier")
    return classifier


def _read_cases(path):
    cases = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            utterances = value.get("utterances")
            if not isinstance(utterances, list) or not utterances or not all(
                isinstance(item, str) and item.strip() for item in utterances
            ):
                raise ValueError(
                    f"line {line_number}: utterances must be non-empty strings"
                )
            duration = float(value.get("duration_seconds", 0.0))
            if duration < 0:
                raise ValueError(f"line {line_number}: duration is negative")
            category = str(
                value.get("category") or "uncertain"
            ).strip().lower()
            if category not in EVALUATION_CATEGORIES:
                raise ValueError(
                    f"line {line_number}: unsupported evaluation category"
                )
            acoustic_context = str(
                value.get("acoustic_context") or "none"
            ).strip().lower()
            if acoustic_context not in ACOUSTIC_CONTEXTS:
                raise ValueError(
                    f"line {line_number}: unsupported acoustic_context"
                )
            expected_label = value.get("expected_semantic_label")
            if expected_label is not None:
                expected_label = str(expected_label).strip().lower()
                if expected_label not in SEMANTIC_LABELS:
                    raise ValueError(
                        f"line {line_number}: unsupported semantic label"
                    )
            data_origin = str(
                value.get("data_origin") or "unspecified"
            ).strip().lower()
            if "child" in data_origin or "minor" in data_origin:
                raise ValueError(
                    f"line {line_number}: children's speech is not permitted"
                )
            expected_alert = value.get("expected_alert")
            if expected_alert is not None and not isinstance(
                expected_alert, bool
            ):
                raise ValueError(
                    f"line {line_number}: expected_alert must be boolean or null"
                )
            provisional_label = value.get("provisional_label")
            if provisional_label is not None:
                provisional_label = str(provisional_label).strip().lower()
                if provisional_label not in PROVISIONAL_LABELS:
                    raise ValueError(
                        f"line {line_number}: unsupported provisional label"
                    )
            cases.append({
                "id": str(value.get("id") or f"line-{line_number}"),
                "language": str(value.get("language") or "unknown").lower(),
                "condition": str(value.get("condition") or "unspecified").lower(),
                "expected_alert": expected_alert,
                "duration_seconds": duration,
                "utterances": tuple(utterances),
                "category": category,
                "acoustic_context": acoustic_context,
                "expected_semantic_label": expected_label,
                "data_origin": data_origin,
                "source_category": value.get("source_category"),
                "provisional_label": provisional_label,
            })
    return cases


def _run_case(case, *, classifier):
    baseline = TextDetector()
    experimental = TextDetector(semantic_analyzer=SemanticAnalyzer(
        classifier=classifier,
        enabled=True,
    ))
    baseline_ms = experimental_ms = 0.0
    statuses = []
    predicted_semantic_label = None
    baseline_items = []
    experimental_items = []
    for index, utterance in enumerate(case["utterances"]):
        observed = float(index + 1)
        started = time.perf_counter()
        baseline_evidence = baseline.analyze_transcript(
            utterance,
            language=case["language"],
            observed_monotonic=observed,
        )
        baseline_ms += (time.perf_counter() - started) * 1_000.0
        started = time.perf_counter()
        experimental_evidence = experimental.analyze_transcript(
            utterance,
            language=case["language"],
            observed_monotonic=observed,
        )
        experimental_ms += (time.perf_counter() - started) * 1_000.0
        baseline_items.append(baseline_evidence)
        experimental_items.append(experimental_evidence)
        statuses.append(experimental_evidence.semantic.status)
        if experimental_evidence.semantic.status in {
            "classified", "low_confidence"
        }:
            predicted_semantic_label = experimental_evidence.semantic.label
    started = time.perf_counter()
    baseline_alert = _fused_alert(
        baseline_items, duration_seconds=case["duration_seconds"]
    )
    baseline_ms += (time.perf_counter() - started) * 1_000.0
    started = time.perf_counter()
    runtime_shadow_alert = _fused_alert(
        experimental_items, duration_seconds=case["duration_seconds"]
    )
    experimental_alert = shadow_candidate_case_alert(
        experimental_items, baseline_alert=baseline_alert
    )
    experimental_ms += (time.perf_counter() - started) * 1_000.0
    baseline.clear_context()
    experimental.clear_context()
    return EvaluationObservation(
        case_id=case["id"],
        language=case["language"],
        condition=case["condition"],
        expected_alert=case["expected_alert"],
        baseline_alert=baseline_alert,
        experimental_alert=experimental_alert,
        duration_seconds=case["duration_seconds"],
        baseline_latency_ms=baseline_ms,
        experimental_latency_ms=experimental_ms,
        category=case["category"],
        acoustic_context=case["acoustic_context"],
        expected_semantic_label=case["expected_semantic_label"],
        predicted_semantic_label=predicted_semantic_label,
    ), tuple(statuses), runtime_shadow_alert


def _fused_alert(text_items, *, duration_seconds):
    text_items = tuple(text_items)
    if not any(item.qualifies_to_open for item in text_items):
        return False
    return fuse_event_evidence(
        text_items,
        (),
        duration_seconds=duration_seconds,
    ).deliver


def _peak_rss_mb():
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux reports KiB.
    if sysconfig.get_platform().startswith("macosx"):
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="held-out JSONL file")
    parser.add_argument(
        "--adapter",
        help="local semantic adapter as module:factory; never downloaded",
    )
    parser.add_argument(
        "--dataset-role",
        choices=("smoke", "development", "held_out"),
        default="smoke",
        help="provenance label for the report; default cannot support claims",
    )
    args = parser.parse_args(argv)
    rss_before_load = _peak_rss_mb()
    load_started = time.perf_counter()
    classifier = _adapter(args.adapter)
    adapter_load_ms = (time.perf_counter() - load_started) * 1_000.0
    rss_after_load = _peak_rss_mb()
    cases = _read_cases(args.dataset)
    observations = []
    statuses = []
    runtime_decision_mismatches = 0
    evaluation_started = time.perf_counter()
    cpu_started = time.process_time()
    # Detection diagnostics are useful but must not corrupt the JSON report.
    with redirect_stdout(sys.stderr):
        for case in cases:
            observation, case_statuses, runtime_shadow_alert = _run_case(
                case, classifier=classifier
            )
            observations.append(observation)
            statuses.extend(case_statuses)
            if runtime_shadow_alert != observation.baseline_alert:
                runtime_decision_mismatches += 1
    evaluation_wall_ms = (time.perf_counter() - evaluation_started) * 1_000.0
    evaluation_cpu_seconds = time.process_time() - cpu_started
    rss_after_evaluation = _peak_rss_mb()
    report = {
        "semantic_model": (
            "adapter_supplied" if classifier is not None else "unavailable"
        ),
        "runtime_policy": "shadow_only_no_decision_effect",
        "offline_candidate_policy": (
            "semantic_labels_compared_only; urgent_baseline_directives_preserved"
        ),
        "dataset_role": args.dataset_role,
        "case_count": len(cases),
        "runtime_decision_mismatches": runtime_decision_mismatches,
        "semantic_status_counts": {
            status: statuses.count(status) for status in sorted(set(statuses))
        },
        "metrics": grouped_comparison(observations),
        "resource_usage": {
            "adapter_load_ms": adapter_load_ms,
            "evaluation_wall_ms": evaluation_wall_ms,
            "evaluation_cpu_seconds": evaluation_cpu_seconds,
            "peak_rss_mb_before_adapter": rss_before_load,
            "peak_rss_mb_after_adapter": rss_after_load,
            "peak_rss_mb_after_evaluation": rss_after_evaluation,
            "peak_rss_growth_mb": max(
                0.0, rss_after_evaluation - rss_before_load
            ),
        },
        "limitations": [
            "metrics describe this input file only",
            "dataset_role=smoke or development cannot support accuracy claims",
            "AI-on is an offline candidate policy, not a runtime decision path",
            "semantic output does not change runtime alert decisions",
            "acoustic context is reported only and never proves emotion or intent",
            "false_alerts_per_hour requires at least 3600 negative seconds",
            "peak RSS is a process high-water mark, not current model-only memory",
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
