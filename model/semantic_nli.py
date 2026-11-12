"""Local-only zero-shot NLI adapter for experimental sentence semantics.

The adapter deliberately has no model-hub integration.  It accepts only an
existing local model directory and asks Transformers to stay offline.  Runtime
code does not construct it automatically; it is intended for the held-out
evaluation tool and explicit shadow-only injection.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

from detection.semantic import SemanticPrediction, SemanticRequest


_CANDIDATE_HYPOTHESES = (
    ("neutral", "This is a neutral conversation."),
    (
        "academic_discussion",
        "This discusses a subject for learning rather than attacking a person.",
    ),
    (
        "playful_teasing",
        "This is mutual playful teasing rather than belittling or threatening someone.",
    ),
    (
        "targeted_belittling",
        "A person is directly belittling, excluding, or insulting someone.",
    ),
    (
        "threat",
        "A person is threatening harm or directing someone to harm themselves.",
    ),
    (
        "repeated_harassment",
        "A person is repeatedly harassing, excluding, or humiliating someone.",
    ),
)


def _normalized_label(value):
    return str(value or "").strip().casefold().replace("-", "_")


def _entailment_index(config):
    """Resolve the NLI entailment logit without assuming a label order."""

    label2id = getattr(config, "label2id", None) or {}
    for label, index in label2id.items():
        if _normalized_label(label) == "entailment":
            return int(index)
    id2label = getattr(config, "id2label", None) or {}
    for index, label in id2label.items():
        if _normalized_label(label) == "entailment":
            return int(index)
    raise ValueError("model config does not identify the entailment logit")


def _softmax(values):
    values = tuple(float(value) for value in values)
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("entailment logits must be finite and non-empty")
    maximum = max(values)
    exponentials = tuple(math.exp(value - maximum) for value in values)
    denominator = sum(exponentials)
    return tuple(value / denominator for value in exponentials)


def _premise(request: SemanticRequest) -> str:
    parts = []
    if request.prior_turns:
        parts.append("Earlier conversation:")
        parts.extend(f"- {turn.text}" for turn in request.prior_turns)
        parts.append("Current utterance:")
    parts.append(request.utterance)
    return "\n".join(parts)


def _load_transformer_components(model_dir):
    """Import optional dependencies only when an adapter is constructed."""

    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
    except ImportError as exc:
        raise RuntimeError(
            "semantic adapter requires locally installed torch and transformers"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        use_fast=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
        use_safetensors=True,
    )
    model.to("cpu")
    model.eval()
    return torch, tokenizer, model


class TransformerNliClassifier:
    """Batch zero-shot hypotheses through a local sequence NLI model."""

    def __init__(
        self,
        model_dir,
        *,
        model_id=None,
        max_length=256,
        component_loader=_load_transformer_components,
    ):
        path = Path(model_dir).expanduser()
        if not path.is_dir():
            raise FileNotFoundError("semantic model directory does not exist")
        if not 32 <= int(max_length) <= 512:
            raise ValueError("semantic max_length must be between 32 and 512")
        safe_model_id = str(model_id or path.name).strip()
        if not safe_model_id or len(safe_model_id) > 200:
            raise ValueError("semantic model_id is invalid")

        self.model_dir = path.resolve()
        self.model_id = safe_model_id
        self.max_length = int(max_length)
        self._torch, self._tokenizer, self._model = component_loader(
            str(self.model_dir)
        )
        self._entailment_index = _entailment_index(self._model.config)

    def classify(self, request: SemanticRequest) -> SemanticPrediction:
        if not isinstance(request, SemanticRequest):
            raise TypeError("request must be SemanticRequest")
        premise = _premise(request)
        labels = tuple(label for label, _hypothesis in _CANDIDATE_HYPOTHESES)
        pairs = [
            (premise, hypothesis)
            for _label, hypothesis in _CANDIDATE_HYPOTHESES
        ]
        encoded = self._tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        with self._torch.inference_mode():
            outputs = self._model(**encoded)
        logits = outputs.logits.detach().cpu().numpy()
        if len(logits.shape) != 2 or logits.shape[0] != len(labels):
            raise ValueError("NLI model returned an unexpected logits shape")
        if not 0 <= self._entailment_index < logits.shape[1]:
            raise ValueError("entailment logit is outside the model output")
        scores = _softmax(logits[:, self._entailment_index])
        best_index = max(range(len(scores)), key=scores.__getitem__)
        return SemanticPrediction(
            label=labels[best_index],
            confidence=scores[best_index],
            model_id=self.model_id,
            scores=dict(zip(labels, scores)),
        )


def create_classifier():
    """Factory for ``tools/evaluate_semantic_upgrade.py --adapter``.

    No default path is provided: the caller must point at an approved local
    snapshot, which prevents an enabled flag from initiating a download.
    """

    model_dir = os.environ.get("ECHOSENSE_SEMANTIC_MODEL_DIR", "").strip()
    if not model_dir:
        raise RuntimeError("ECHOSENSE_SEMANTIC_MODEL_DIR is required")
    max_length = int(os.environ.get("ECHOSENSE_SEMANTIC_MAX_LENGTH", "256"))
    model_id = os.environ.get("ECHOSENSE_SEMANTIC_MODEL_ID")
    return TransformerNliClassifier(
        model_dir,
        model_id=model_id,
        max_length=max_length,
    )
