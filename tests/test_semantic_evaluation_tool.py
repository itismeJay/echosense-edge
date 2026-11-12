from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
import unittest

from tools.evaluate_semantic_upgrade import (
    EVALUATION_CATEGORIES,
    _read_cases,
    main,
)


FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "semantic_evaluation_smoke.jsonl"
)
CEBUANO_200_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "cebuano_classroom_original_200.jsonl"
)
TAGALOG_250_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "tagalog_classroom_250.jsonl"
)
ENGLISH_250_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "english_classroom_250.jsonl"
)


class SemanticEvaluationToolTests(unittest.TestCase):
    def test_english_250_preserves_ids_source_categories_and_review_rows(self):
        cases = _read_cases(ENGLISH_250_FIXTURE)
        self.assertEqual([case["id"] for case in cases], [
            str(value) for value in range(1, 251)
        ])
        self.assertEqual(
            {category: sum(
                case["source_category"] == category for case in cases
            ) for category in (
                "bullying", "normal", "happy", "other", "mixed"
            )},
            {
                "bullying": 50,
                "normal": 50,
                "happy": 50,
                "other": 50,
                "mixed": 50,
            },
        )
        self.assertEqual(
            sum(case["expected_alert"] is None for case in cases), 3
        )
        self.assertEqual(
            len({case["utterances"][0] for case in cases}), 250
        )

    def test_tagalog_250_preserves_ids_source_categories_and_review_rows(self):
        cases = _read_cases(TAGALOG_250_FIXTURE)
        self.assertEqual([case["id"] for case in cases], [
            str(value) for value in range(1, 251)
        ])
        self.assertEqual(
            {category: sum(
                case["source_category"] == category for case in cases
            ) for category in (
                "bullying", "normal", "happy", "other", "mixed"
            )},
            {
                "bullying": 50,
                "normal": 50,
                "happy": 50,
                "other": 50,
                "mixed": 50,
            },
        )
        self.assertEqual(
            sum(case["expected_alert"] is None for case in cases), 6
        )
        self.assertEqual(
            len({case["utterances"][0] for case in cases}), 250
        )

    def test_cebuano_200_preserves_ids_source_categories_and_review_rows(self):
        cases = _read_cases(CEBUANO_200_FIXTURE)
        self.assertEqual([case["id"] for case in cases], [
            str(value) for value in range(1, 201)
        ])
        self.assertEqual(
            {category: sum(
                case["source_category"] == category for case in cases
            ) for category in ("bullying", "normal", "happy", "other")},
            {"bullying": 50, "normal": 50, "happy": 50, "other": 50},
        )
        self.assertEqual(
            sum(case["expected_alert"] is None for case in cases), 8
        )
        self.assertEqual(
            len({case["utterances"][0] for case in cases}), 200
        )

    def test_dataset_reader_preserves_source_labels_and_null_adjudication(self):
        case = {
            "id": "17",
            "language": "ceb",
            "category": "uncertain",
            "source_category": "other",
            "provisional_label": "context_dependent",
            "expected_alert": None,
            "data_origin": "user_supplied_text",
            "utterances": ["Context is needed."],
        }
        with NamedTemporaryFile("w", suffix=".jsonl") as handle:
            handle.write(json.dumps(case) + "\n")
            handle.flush()
            loaded = _read_cases(handle.name)[0]
        self.assertIsNone(loaded["expected_alert"])
        self.assertEqual(loaded["source_category"], "other")
        self.assertEqual(loaded["provisional_label"], "context_dependent")

    def test_smoke_matrix_covers_every_required_category_and_language(self):
        cases = _read_cases(FIXTURE)
        self.assertEqual({case["language"] for case in cases}, {
            "en", "ceb", "fil", "mixed",
        })
        self.assertEqual({case["category"] for case in cases}, {
            "neutral_conversation",
            "academic_discussion",
            "playful_teasing",
            "targeted_insult",
            "threat",
            "indirect_belittling",
            "repeated_harassment",
        })
        self.assertTrue(
            {case["category"] for case in cases} <= EVALUATION_CATEGORIES
        )
        self.assertTrue(all(
            case["data_origin"] == "adult_authored_synthetic"
            for case in cases
        ))
        self.assertEqual(
            sum(case["expected_alert"] for case in cases),
            sum(not case["expected_alert"] for case in cases),
        )
        self.assertTrue({
            "quoted_speech", "negated_speech", "non_directed_speech",
        } <= {case["condition"] for case in cases})

    def test_dataset_reader_rejects_children_speech_origin(self):
        case = {
            "id": "not-allowed",
            "language": "ceb",
            "category": "neutral_conversation",
            "expected_alert": False,
            "data_origin": "child_recording",
            "utterances": ["redacted"],
        }
        with NamedTemporaryFile("w", suffix=".jsonl") as handle:
            handle.write(json.dumps(case) + "\n")
            handle.flush()
            with self.assertRaisesRegex(ValueError, "children's speech"):
                _read_cases(handle.name)

    def test_no_adapter_report_is_valid_json_and_never_fakes_ai(self):
        output = io.StringIO()
        diagnostics = io.StringIO()
        with redirect_stdout(output), redirect_stderr(diagnostics):
            result = main([str(FIXTURE), "--dataset-role", "smoke"])
        report = json.loads(output.getvalue())
        self.assertEqual(result, 0)
        self.assertEqual(report["semantic_model"], "unavailable")
        self.assertEqual(report["runtime_decision_mismatches"], 0)
        self.assertEqual(
            report["semantic_status_counts"], {"unavailable": 25}
        )
        baseline = report["metrics"]["overall"]["baseline"]
        experimental = report["metrics"]["overall"]["experimental"]
        for name in (
            "true_positives", "false_positives", "true_negatives",
            "false_negatives", "accuracy", "precision", "recall",
        ):
            self.assertEqual(baseline[name], experimental[name])


if __name__ == "__main__":
    unittest.main()
