"""Conservative sentence guidance that never overwrites an ASR transcript."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
import unicodedata


_WORD_PATTERN = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.UNICODE)


def normalized_words(text: object) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return tuple(_WORD_PATTERN.findall(normalized))


@dataclass(frozen=True, slots=True)
class GuidedPhrase:
    source_id: str
    text: str
    language: str

    def __post_init__(self):
        if not self.source_id or not self.text.strip() or not self.language:
            raise ValueError("guided phrase fields must be non-empty")


@dataclass(frozen=True, slots=True)
class LanguageSuggestion:
    heard_text: str
    supporting_text: str
    suggested_text: str
    source_id: str
    language: str
    similarity: float
    changed: bool


class LanguageGuide:
    """Find one unique, high-confidence phrase suggestion or return ``None``.

    The guide is intentionally read-only with respect to the transcript. Its
    result is evidence for evaluation or later matching, never a replacement
    for what the speech recognizer actually emitted.
    """

    def __init__(
        self,
        phrases,
        *,
        minimum_words=4,
        minimum_similarity=0.80,
        minimum_margin=0.06,
    ):
        self.minimum_words = max(1, int(minimum_words))
        self.minimum_similarity = float(minimum_similarity)
        self.minimum_margin = max(0.0, float(minimum_margin))
        prepared = []
        for phrase in phrases:
            if not isinstance(phrase, GuidedPhrase):
                raise TypeError("phrases must contain GuidedPhrase values")
            words = normalized_words(phrase.text)
            if len(words) >= self.minimum_words:
                prepared.append((phrase, words))
        self._phrases = tuple(prepared)

    @classmethod
    def from_sentence_cases(cls, cases, **kwargs):
        return cls(
            tuple(
                GuidedPhrase(case.sentence_id, case.text, case.language)
                for case in cases
            ),
            **kwargs,
        )

    @staticmethod
    def _similarity(left, right):
        left_text = " ".join(left)
        right_text = " ".join(right)
        character_score = SequenceMatcher(None, left_text, right_text).ratio()
        token_score = SequenceMatcher(None, left, right).ratio()
        # ASR commonly splits or joins Cebuano clitics (``nimog`` / ``ni mug``),
        # so character agreement carries more weight than token boundaries.
        return 0.90 * character_score + 0.10 * token_score

    def suggest(self, transcript, alternatives=(), *, language=None):
        heard_text = str(transcript or "")
        candidate_texts = (heard_text,) + tuple(
            str(value or "") for value in alternatives
        )
        prepared_inputs = [
            (value, normalized_words(value))
            for value in candidate_texts
            if len(normalized_words(value)) >= self.minimum_words
        ]
        if not prepared_inputs:
            return None
        normalized_language = str(language or "").strip().casefold()
        ranked = []
        for phrase, phrase_words in self._phrases:
            if normalized_language and phrase.language.casefold() != normalized_language:
                continue
            supporting_text, score = max(
                (
                    (value, self._similarity(words, phrase_words))
                    for value, words in prepared_inputs
                ),
                key=lambda item: item[1],
            )
            ranked.append((score, phrase.source_id, phrase, supporting_text))
        if not ranked:
            return None
        ranked.sort(key=lambda item: (-item[0], item[1]))
        best_score, _source_id, best, supporting_text = ranked[0]
        runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
        if best_score < self.minimum_similarity:
            return None
        if best_score - runner_up < self.minimum_margin:
            return None
        heard_words = normalized_words(heard_text)
        suggested_words = normalized_words(best.text)
        return LanguageSuggestion(
            heard_text=heard_text,
            supporting_text=supporting_text,
            suggested_text=best.text,
            source_id=best.source_id,
            language=best.language,
            similarity=best_score,
            changed=heard_words != suggested_words,
        )
