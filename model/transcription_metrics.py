"""Deterministic transcript accuracy metrics for recorded speech evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from model.language_guide import normalized_words


def _edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_item in enumerate(reference, 1):
        current = [reference_index]
        for hypothesis_index, hypothesis_item in enumerate(hypothesis, 1):
            current.append(min(
                current[-1] + 1,
                previous[hypothesis_index] + 1,
                previous[hypothesis_index - 1]
                + (reference_item != hypothesis_item),
            ))
        previous = current
    return previous[-1]


@dataclass(frozen=True, slots=True)
class TranscriptionMetrics:
    exact: bool
    reference_words: int
    word_errors: int
    word_error_rate: float
    reference_characters: int
    character_errors: int
    character_error_rate: float


def _error_rate(errors, reference_length, hypothesis_length):
    if reference_length:
        return errors / reference_length
    return 0.0 if hypothesis_length == 0 else 1.0


def transcription_metrics(reference, hypothesis) -> TranscriptionMetrics:
    reference_words = normalized_words(reference)
    hypothesis_words = normalized_words(hypothesis)
    reference_text = " ".join(reference_words)
    hypothesis_text = " ".join(hypothesis_words)
    word_errors = _edit_distance(reference_words, hypothesis_words)
    character_errors = _edit_distance(reference_text, hypothesis_text)
    return TranscriptionMetrics(
        exact=reference_words == hypothesis_words,
        reference_words=len(reference_words),
        word_errors=word_errors,
        word_error_rate=_error_rate(
            word_errors,
            len(reference_words),
            len(hypothesis_words),
        ),
        reference_characters=len(reference_text),
        character_errors=character_errors,
        character_error_rate=_error_rate(
            character_errors,
            len(reference_text),
            len(hypothesis_text),
        ),
    )


def aggregate_transcription_metrics(results):
    values = tuple(results)
    sentence_count = len(values)
    reference_words = sum(item.reference_words for item in values)
    word_errors = sum(item.word_errors for item in values)
    reference_characters = sum(item.reference_characters for item in values)
    character_errors = sum(item.character_errors for item in values)
    return {
        "sentence_count": sentence_count,
        "exact_sentence_rate": (
            sum(item.exact for item in values) / sentence_count
            if sentence_count
            else None
        ),
        "word_error_rate": (
            word_errors / reference_words if reference_words else None
        ),
        "character_error_rate": (
            character_errors / reference_characters
            if reference_characters
            else None
        ),
        "word_errors": word_errors,
        "reference_words": reference_words,
        "character_errors": character_errors,
        "reference_characters": reference_characters,
    }
