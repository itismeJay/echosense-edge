"""Narrow, auditable harmless-context rules for ambiguous monitored terms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from detection import school_policy
from detection.transcript_quality import normalize_transcript


_DIRECT_TARGET_TOKENS = frozenset({
    "ka", "mo", "ikaw", "imo", "imong", "nimo",
    "nimong", "kamo", "ninyo", "niyo", "ninyong", "inyo", "inyong", "kag",
    "mog", "nimog", "ninyog", "you", "your", "you're",
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


_NEGATORS = frozenset({"dili", "di", "not", "hindi"})
# Words that may sit between the negator and the insult: "dili man ka bogo".
_NEGATION_FILLERS = frozenset({
    "ka", "man", "gyud", "jud", "ikaw", "mo", "kamo", "pud", "sad", "baya",
    "na", "ba", "you", "are", "re", "ra",
})


def _occurrences(tokens, term_tokens):
    length = len(term_tokens)
    return [
        index for index in range(len(tokens) - length + 1)
        if tokens[index:index + length] == term_tokens
    ]


def _negated(index: int, tokens: tuple[str, ...]) -> bool:
    cursor = index - 1
    while cursor >= 0 and tokens[cursor] in _NEGATION_FILLERS:
        cursor -= 1
    return cursor >= 0 and tokens[cursor] in _NEGATORS


def _negation_reason(term: str, tokens: tuple[str, ...]) -> str | None:
    """"dili ka bogo" (you're NOT stupid) is not an insult.

    Only when every occurrence is directly negated; terms that contain the
    negator themselves ("dili ka apil") are never touched.
    """

    term_tokens = tuple(term.split())
    if not term_tokens or set(term_tokens) & _NEGATORS:
        return None
    found = _occurrences(tokens, term_tokens)
    if found and all(_negated(index, tokens) for index in found):
        return "negated"
    return None


def _suppression_reason(
    term: str,
    tokens: tuple[str, ...],
) -> str | None:
    negation = _negation_reason(term, tokens)
    if negation:
        return negation
    if term not in _AMBIGUOUS_TERMS or set(tokens) & _DIRECT_TARGET_TOKENS:
        return None

    # A bare ambiguous adjective has no targeting evidence. Severe monitored
    # words are not in _AMBIGUOUS_TERMS and therefore never enter this rule.
    # Strict sensitivity alerts on it anyway (harmless objects stay suppressed).
    if tokens == tuple(term.split()):
        return None if school_policy.is_strict() else "ambiguous_without_target"

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
