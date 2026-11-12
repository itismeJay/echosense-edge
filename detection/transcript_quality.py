"""Deterministic transcript normalization and pre-decision quality checks."""

from __future__ import annotations

from dataclasses import dataclass
import unicodedata

from detection.thresholds import (
    TRANSCRIPT_LOW_UNIQUE_RATIO,
    TRANSCRIPT_MAX_REPEATED_TOKEN_RUN,
    TRANSCRIPT_MAX_TOKENS,
    TRANSCRIPT_MIN_TOKENS_FOR_PHRASE_CYCLE,
    TRANSCRIPT_MIN_TOKENS_FOR_UNIQUE_RATIO,
    TRANSCRIPT_REPETITION_RATIO_LIMIT,
)


_APOSTROPHES = frozenset({"'", "\N{RIGHT SINGLE QUOTATION MARK}", "\N{MODIFIER LETTER APOSTROPHE}"})
_STRONG_BOUNDARIES = frozenset({".", "!", "?", ";", ":", "/", "\\", "|"})


@dataclass(frozen=True, slots=True)
class TranscriptQualityResult:
    """Immutable, non-audio evidence produced before monitored-term matching."""

    accepted: bool
    normalized_text: str
    token_count: int
    unique_token_count: int
    repetition_ratio: float
    longest_repeated_run: int
    reason_codes: tuple[str, ...]
    warnings: tuple[str, ...]
    # Text the detector matches. Equal to the transcript unless a repeated
    # tail was trimmed; the transcript itself is never changed.
    checked_text: str = ""


def _normalized_characters(value: object) -> str:
    """Normalize Unicode form/case without compatibility-folding word content.

    Accents on Latin letters are dropped: Bisaya-tuned Whisper models write
    stress marks ("gágà ka", "Pangít") that the word lists never carry.
    Marks on non-Latin letters are kept.
    """

    decomposed = unicodedata.normalize("NFD", str(value or ""))
    kept = []
    for character in decomposed:
        if (
            unicodedata.combining(character)
            and kept
            and kept[-1].isascii()
            and kept[-1].isalpha()
        ):
            continue
        kept.append(character)
    return unicodedata.normalize("NFC", "".join(kept)).lower()


def _is_word_character(character: str) -> bool:
    category = unicodedata.category(character)
    return character.isalnum() or category.startswith("M")


def _is_internal_apostrophe(text: str, index: int) -> bool:
    if text[index] not in _APOSTROPHES or index == 0 or index + 1 >= len(text):
        return False
    return _is_word_character(text[index - 1]) and _is_word_character(text[index + 1])


def transcript_token_segments(value: object) -> tuple[tuple[str, ...], ...]:
    """Tokenize without allowing phrases to cross sentence/path boundaries.

    Commas, brackets, dashes, line breaks, and Unicode ellipses remain ordinary
    word boundaries, so normal dictated punctuation does not break a phrase.
    Full stops, sentence punctuation, slashes, and pipes terminate a segment,
    preventing a monitored phrase from being assembled across unrelated text.
    Linguistic apostrophes are retained inside their token.
    """

    text = _normalized_characters(value)
    segments: list[tuple[str, ...]] = []
    tokens: list[str] = []
    current: list[str] = []

    def finish_token() -> None:
        if current:
            tokens.append("".join(current))
            current.clear()

    def finish_segment() -> None:
        finish_token()
        if tokens:
            segments.append(tuple(tokens))
            tokens.clear()

    for index, character in enumerate(text):
        if _is_word_character(character):
            current.append(character)
        elif _is_internal_apostrophe(text, index):
            # Canonicalize typographic apostrophes only; word content is intact.
            current.append("'")
        elif character in _STRONG_BOUNDARIES:
            finish_segment()
        else:
            finish_token()
    finish_segment()
    return tuple(segments)


def normalize_transcript(value: object) -> str:
    """Return lowercase, single-spaced matching text without joining tokens."""

    return " ".join(
        token
        for segment in transcript_token_segments(value)
        for token in segment
    )


def phrase_matches(term: object, text: object) -> bool:
    """Match an exact token/phrase within one valid text segment."""

    term_tokens = tuple(normalize_transcript(term).split())
    if not term_tokens:
        return False
    length = len(term_tokens)
    for segment in transcript_token_segments(text):
        for index in range(len(segment) - length + 1):
            if segment[index:index + length] == term_tokens:
                return True
    return False


def phrase_ends_segment(term: object, text: object) -> bool:
    """Like ``phrase_matches``, but only when nothing follows in its sentence.

    "pangita nimo oy." (how ugly you are) ends the sentence; "pangita nimo oy
    ang libro" (look for the book) goes on, so it does not count.
    """

    term_tokens = tuple(normalize_transcript(term).split())
    if not term_tokens:
        return False
    length = len(term_tokens)
    return any(
        len(segment) >= length and segment[-length:] == term_tokens
        for segment in transcript_token_segments(text)
    )


def _longest_repeated_run(tokens: tuple[str, ...]) -> int:
    longest = 0
    current = 0
    previous = None
    for token in tokens:
        if token == previous:
            current += 1
        else:
            previous = token
            current = 1
        longest = max(longest, current)
    return longest


def _has_repeated_phrase_pattern(tokens: tuple[str, ...]) -> bool:
    """Detect an utterance made entirely from a phrase repeated at least 3x."""

    token_count = len(tokens)
    if token_count < TRANSCRIPT_MIN_TOKENS_FOR_PHRASE_CYCLE:
        return False
    for phrase_length in range(2, token_count // 3 + 1):
        if token_count % phrase_length:
            continue
        repeats = token_count // phrase_length
        phrase = tokens[:phrase_length]
        if repeats >= 3 and phrase * repeats == tokens:
            return True
    return False


# A Whisper loop at the end of real speech: "ang pangit mo ang mga mga mga".
_TAIL_PHRASE_MAX_TOKENS = 3
_TAIL_MIN_REPEATS = 3


def _repeated_tail_cut(tokens: tuple[str, ...]):
    """Index after the first copy of a looping tail, or None.

    A tail is one word repeated more than the allowed run, or a 2-3 word
    phrase repeated 3+ times. One copy is kept. ``None`` when there is no such
    tail or nothing but the loop was said (the whole transcript repeats).
    """

    count = len(tokens)
    cut = None
    for length in range(1, _TAIL_PHRASE_MAX_TOKENS + 1):
        unit = tokens[count - length:]
        if len(unit) < length:
            break
        start = count
        while start - length >= 0 and tokens[start - length:start] == unit:
            start -= length
        repeats = (count - start) // length
        needed = (
            TRANSCRIPT_MAX_REPEATED_TOKEN_RUN + 1 if length == 1 else _TAIL_MIN_REPEATS
        )
        # Something other than the loop's own words must precede it, else
        # the whole transcript is repetition ("alpha beta alpha beta ...").
        if repeats >= needed and set(tokens[:start]) - set(unit):
            keep = start + length
            cut = keep if cut is None else min(cut, keep)
    return cut


def _trimmed_text(segments, keep: int) -> str:
    kept = []
    for segment in segments:
        if keep <= 0:
            break
        kept.append(" ".join(segment[:keep]))
        keep -= len(segment)
    return ". ".join(kept)


_REPETITION_REASONS = frozenset({
    "excessive_token_repetition",
    "repeated_phrase_pattern",
    "low_unique_token_ratio",
})


def _assess_tokens(original, normalized, tokens):
    token_count = len(tokens)
    unique_token_count = len(set(tokens))
    unique_ratio = unique_token_count / token_count if token_count else 0.0
    repetition_ratio = 1.0 - unique_ratio if token_count else 0.0
    longest_run = _longest_repeated_run(tokens)
    reasons: list[str] = []

    if not original.strip():
        reasons.append("empty_transcript")
    elif not any(character.isalnum() for character in original):
        reasons.append("punctuation_only")
    if not normalized:
        reasons.append("normalization_empty")
        reasons.append("too_few_informative_tokens")
    if token_count > TRANSCRIPT_MAX_TOKENS:
        reasons.append("transcript_too_long")
    if longest_run > TRANSCRIPT_MAX_REPEATED_TOKEN_RUN:
        reasons.append("excessive_token_repetition")
    if _has_repeated_phrase_pattern(tokens):
        reasons.append("repeated_phrase_pattern")
    if (
        token_count >= TRANSCRIPT_MIN_TOKENS_FOR_UNIQUE_RATIO
        and unique_ratio < TRANSCRIPT_LOW_UNIQUE_RATIO
    ):
        reasons.append("low_unique_token_ratio")
    if (
        token_count >= TRANSCRIPT_MIN_TOKENS_FOR_PHRASE_CYCLE
        and repetition_ratio >= TRANSCRIPT_REPETITION_RATIO_LIMIT
        and "excessive_token_repetition" not in reasons
        and "repeated_phrase_pattern" not in reasons
    ):
        reasons.append("excessive_token_repetition")
    return (
        reasons, token_count, unique_token_count, repetition_ratio, longest_run,
    )


def assess_transcript_quality(value: object) -> TranscriptQualityResult:
    """Identify low-quality or possibly artifact-like decoding conservatively.

    These signals are filters, not evidence that Whisper hallucinated and not
    proof of any real-world behavior. When repetition would reject the
    utterance and it is only a looping tail after real words ("ang pangit mo
    ang mga mga mga"), the tail is trimmed to one copy (``checked_text``) and
    the rest is judged. A transcript that is repetition throughout is still
    rejected. ``normalized_text`` and the caller's transcript stay complete.
    """

    original = str(value or "")
    checked = original
    normalized = normalize_transcript(original)
    tokens = tuple(normalized.split())
    reasons, token_count, unique_token_count, repetition_ratio, longest_run = (
        _assess_tokens(original, normalized, tokens)
    )
    warnings: list[str] = []
    if reasons and set(reasons) <= _REPETITION_REASONS:
        segments = transcript_token_segments(original)
        flat = tuple(token for segment in segments for token in segment)
        cut = _repeated_tail_cut(flat) if flat == tokens else None
        if cut is not None:
            trimmed = _assess_tokens(original, normalized, tokens[:cut])
            if not trimmed[0]:
                checked = _trimmed_text(segments, cut)
                (
                    reasons, token_count, unique_token_count,
                    repetition_ratio, longest_run,
                ) = trimmed
                warnings.append("repeated_tail_trimmed")

    if not reasons and longest_run == TRANSCRIPT_MAX_REPEATED_TOKEN_RUN:
        warnings.append("suspicious_repetition_below_rejection_threshold")

    accepted = not reasons
    return TranscriptQualityResult(
        accepted=accepted,
        normalized_text=normalized,
        token_count=token_count,
        unique_token_count=unique_token_count,
        repetition_ratio=round(repetition_ratio, 4),
        longest_repeated_run=longest_run,
        reason_codes=("accepted",) if accepted else tuple(reasons),
        warnings=tuple(warnings),
        checked_text=checked,
    )
