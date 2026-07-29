"""Narrow, auditable harmless-context rules for ambiguous monitored terms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from detection.transcript_quality import normalize_transcript


_DIRECT_TARGET_TOKENS = frozenset({
    "ka", "mo", "ikaw", "imo", "imong", "nimo", "you", "your", "you're",
    # Synthetic, identity-free token used by tests/integrations for a named target.
    "classmate",
})
_IMPERSONAL_OBJECTS = frozenset({
    "drawing", "internet", "panahon", "signal", "weather",
})
_AMBIGUOUS_TERMS = frozenset({"pangit", "bad", "ugly"})


@dataclass(frozen=True, slots=True)
class SuppressedTerm:
    term: str
    reason: str


@dataclass(frozen=True, slots=True)
class TermContextResult:
    accepted_terms: tuple[str, ...]
    suppressed_terms: tuple[SuppressedTerm, ...]


def _contains_sequence(tokens: tuple[str, ...], sequence: tuple[str, ...]) -> bool:
    length = len(sequence)
    return any(
        tokens[index:index + length] == sequence
        for index in range(len(tokens) - length + 1)
    )


def _suppression_reason(
    term: str,
    tokens: tuple[str, ...],
) -> str | None:
    if term not in _AMBIGUOUS_TERMS or set(tokens) & _DIRECT_TARGET_TOKENS:
        return None

    # A bare ambiguous adjective has no targeting evidence. Severe monitored
    # words are not in _AMBIGUOUS_TERMS and therefore never enter this rule.
    if tokens == tuple(term.split()):
        return "ambiguous_without_target"

    if term == "pangit":
        harmless = any(
            _contains_sequence(tokens, ("pangit", article, object_name))
            for article in ("ang", "yung")
            for object_name in _IMPERSONAL_OBJECTS
        )
        return "harmless_context" if harmless else None

    # English object rules intentionally require an explicit impersonal subject
    # plus copula; other uses are not broadly suppressed.
    harmless = any(
        _contains_sequence(tokens, (determiner, object_name, "is", term))
        for determiner in ("the", "this")
        for object_name in _IMPERSONAL_OBJECTS
    )
    return "harmless_context" if harmless else None


def apply_harmless_context_rules(
    normalized_text: object,
    candidate_terms: Iterable[str],
) -> TermContextResult:
    """Suppress only configured ambiguous terms in explicit object contexts."""

    tokens = tuple(normalize_transcript(normalized_text).split())
    accepted: list[str] = []
    suppressed: list[SuppressedTerm] = []
    seen = set()
    for source_term in candidate_terms:
        term = normalize_transcript(source_term)
        if not term or term in seen:
            continue
        seen.add(term)
        reason = _suppression_reason(term, tokens)
        if reason:
            suppressed.append(
                SuppressedTerm(term=term, reason=reason)
            )
        else:
            accepted.append(term)
    return TermContextResult(tuple(accepted), tuple(suppressed))
