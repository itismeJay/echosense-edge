from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from detection.semantic import ConversationTurn, SemanticRequest
from model.semantic_nli import (
    TransformerNliClassifier,
    _entailment_index,
    _softmax,
)


@dataclass
class FakeConfig:
    label2id: dict
    id2label: dict


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeTokenizer:
    def __init__(self):
        self.pairs = None
        self.options = None

    def __call__(self, pairs, **options):
        self.pairs = pairs
        self.options = options
        return {"input_ids": "bounded-test-input"}


class FakeModel:
    def __init__(self):
        self.config = FakeConfig(
            label2id={"contradiction": 0, "neutral": 1, "entailment": 2},
            id2label={},
        )

    def __call__(self, **_encoded):
        # Candidate order is neutral, academic, playful, targeted, threat,
        # repeated. The threat entailment logit is intentionally strongest.
        logits = np.asarray([
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -0.5],
            [0.0, 0.0, -0.2],
            [0.0, 0.0, 0.4],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 0.1],
        ])
        return type("Output", (), {"logits": FakeTensor(logits)})()


class FakeTorch:
    @staticmethod
    def inference_mode():
        return nullcontext()


class SemanticNliAdapterTests(unittest.TestCase):
    def test_entailment_index_never_guesses_model_label_order(self):
        config = FakeConfig(label2id={}, id2label={"0": "ENTAILMENT"})
        self.assertEqual(_entailment_index(config), 0)
        with self.assertRaises(ValueError):
            _entailment_index(FakeConfig(label2id={}, id2label={}))

    def test_softmax_is_bounded_and_rejects_non_finite_values(self):
        scores = _softmax((1001.0, 1000.0))
        self.assertAlmostEqual(sum(scores), 1.0)
        self.assertGreater(scores[0], scores[1])
        with self.assertRaises(ValueError):
            _softmax((float("nan"),))

    def test_local_adapter_batches_fixed_hypotheses_with_bounded_context(self):
        tokenizer = FakeTokenizer()
        model = FakeModel()

        def loader(_path):
            return FakeTorch(), tokenizer, model

        with TemporaryDirectory() as directory:
            classifier = TransformerNliClassifier(
                directory,
                model_id="offline-test-model",
                max_length=128,
                component_loader=loader,
            )
            request = SemanticRequest(
                utterance="current statement",
                language="mixed",
                prior_turns=(ConversationTurn(
                    text="earlier statement",
                    language="ceb",
                    observed_monotonic=1.0,
                ),),
            )
            prediction = classifier.classify(request)

        self.assertEqual(prediction.label, "threat")
        self.assertEqual(prediction.model_id, "offline-test-model")
        self.assertEqual(set(prediction.scores), {
            "neutral",
            "academic_discussion",
            "playful_teasing",
            "targeted_belittling",
            "threat",
            "repeated_harassment",
        })
        self.assertEqual(tokenizer.options["max_length"], 128)
        self.assertEqual(tokenizer.options["return_tensors"], "pt")
        self.assertEqual(len(tokenizer.pairs), 6)
        self.assertIn("earlier statement", tokenizer.pairs[0][0])
        self.assertIn("current statement", tokenizer.pairs[0][0])

    def test_adapter_requires_an_existing_local_directory(self):
        missing = Path("definitely-missing-semantic-model")
        with self.assertRaises(FileNotFoundError):
            TransformerNliClassifier(missing)


if __name__ == "__main__":
    unittest.main()
