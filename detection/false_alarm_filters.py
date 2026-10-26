"""Filters for false alarms from media audio and Whisper hallucinations.

Pure functions, no model or I/O, so each rule is unit-tested on its own.
model/whisper_stt.py applies them to every finalized utterance, in this order:

1. decode quality   no_speech_prob, avg_logprob and compression_ratio from the
                    Whisper decode (a quiet room or music produces confident-
                    looking text with these warning signs).
2. hallucination    utterances made only of known Whisper filler phrases
                    ("Thank you for watching", "Subscribe", ...) or of one
                    word repeated.
3. word probability a matched monitored term counts only when Whisper was
                    reasonably sure of the words that form it (0.5; 0.3 for
                    threat and self-harm phrases). With the old hint prompt
                    Whisper wrote "bobo" at probability 0.01: this filter
                    catches exactly that.
4. media playing    YAMNet hears Music, Television or Radio: no alert, unless
                    the term is a threat or self-harm phrase.

Every rule can be tuned or switched off with ECHOSENSE_FILTER_* variables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import re
import unicodedata
from typing import Iterable, Mapping, Sequence

from detection.severity import contains_urgent_directive

DEFAULT_MEDIA_CLASSES = ("Music", "Television", "Radio")

# Whole-utterance fillers Whisper produces from silence, music or credits.
# Matching is on normalized text; an utterance is dropped only when nothing
# else remains once these phrases are removed.
HALLUCINATION_PHRASES = (
    "thank you for watching",
    "thanks for watching",
    "thank you so much for watching",
    "thank you for listening",
    "please subscribe",
    "like and subscribe",
    "subscribe to my channel",
    "subscribe",
    "subtitles by the amara org community",
    "subtitles by",
    "amara org",
    "salamat sa panonood",
    "salamat po sa panonood",
    "thank you",
    "you",
)


def _env_float(environ, name, default):
    raw = environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(environ, name, default):
    raw = environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(environ, name, default):
    raw = environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class FilterConfig:
    decode_gate_enabled: bool = True
    no_speech_prob_max: float = 0.6
    avg_logprob_min: float = -1.0
    compression_ratio_max: float = 2.4
    hallucination_enabled: bool = True
    repeated_word_min: int = 3
    word_probability_enabled: bool = True
    word_probability_min: float = 0.5
    # Threat and self-harm phrases get a lower floor: a correctly decoded live
    # "papatayin kita" scored 0.49 on its first word. Missing a threat costs
    # more than an extra alert.
    urgent_word_probability_min: float = 0.3
    media_gate_enabled: bool = True
    media_classes: tuple = DEFAULT_MEDIA_CLASSES
    media_top_k: int = 3
    media_min_score: float = 0.15

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None):
        environ = os.environ if environ is None else environ
        classes = tuple(
            item.strip()
            for item in str(
                environ.get("ECHOSENSE_FILTER_MEDIA_CLASSES")
                or ",".join(DEFAULT_MEDIA_CLASSES)
            ).split(",")
            if item.strip()
        )
        return cls(
            decode_gate_enabled=_env_bool(
                environ, "ECHOSENSE_FILTER_DECODE_GATE", True
            ),
            no_speech_prob_max=_env_float(
                environ, "ECHOSENSE_FILTER_NO_SPEECH_PROB_MAX", 0.6
            ),
            avg_logprob_min=_env_float(
                environ, "ECHOSENSE_FILTER_AVG_LOGPROB_MIN", -1.0
            ),
            compression_ratio_max=_env_float(
                environ, "ECHOSENSE_FILTER_COMPRESSION_RATIO_MAX", 2.4
            ),
            hallucination_enabled=_env_bool(
                environ, "ECHOSENSE_FILTER_HALLUCINATION", True
            ),
            repeated_word_min=max(
                2, _env_int(environ, "ECHOSENSE_FILTER_REPEATED_WORD_MIN", 3)
            ),
            word_probability_enabled=_env_bool(
                environ, "ECHOSENSE_FILTER_WORD_PROBABILITY", True
            ),
            word_probability_min=_env_float(
                environ, "ECHOSENSE_FILTER_WORD_PROBABILITY_MIN", 0.5
            ),
            urgent_word_probability_min=_env_float(
                environ, "ECHOSENSE_FILTER_URGENT_WORD_PROBABILITY_MIN", 0.3
            ),
            media_gate_enabled=_env_bool(
                environ, "ECHOSENSE_FILTER_MEDIA_GATE", True
            ),
            media_classes=classes or DEFAULT_MEDIA_CLASSES,
            media_top_k=max(1, _env_int(environ, "ECHOSENSE_FILTER_MEDIA_TOP_K", 3)),
            media_min_score=_env_float(
                environ, "ECHOSENSE_FILTER_MEDIA_MIN_SCORE", 0.15
            ),
        )


@dataclass(frozen=True)
class WordScore:
    word: str
    probability: float


@dataclass(frozen=True)
class DecodeMetrics:
    no_speech_prob: float
    avg_logprob: float
    compression_ratio: float
    words: tuple = field(default_factory=tuple)
    segment_count: int = 0

    @classmethod
    def from_metadata(cls, metadata):
        """Read the plain dict the transcription engine attaches; None if absent."""

        if not isinstance(metadata, Mapping):
            return None
        try:
            words = tuple(
                WordScore(str(word), float(probability))
                for word, probability in metadata.get("words") or ()
            )
            return cls(
                no_speech_prob=float(metadata["no_speech_prob"]),
                avg_logprob=float(metadata["avg_logprob"]),
                compression_ratio=float(metadata["compression_ratio"]),
                words=words,
                segment_count=int(metadata.get("segment_count", 0)),
            )
        except (KeyError, TypeError, ValueError):
            return None


def summarize_segments(segments: Sequence) -> dict:
    """Turn faster-whisper segments into a picklable metrics dict.

    no_speech_prob and avg_logprob are averaged over segments weighted by
    duration; compression_ratio is the worst segment's. Words keep their own
    probabilities.
    """

    segments = list(segments or ())
    if not segments:
        return {
            "no_speech_prob": 0.0,
            "avg_logprob": 0.0,
            "compression_ratio": 0.0,
            "words": [],
            "segment_count": 0,
        }
    weights = [
        max(float(getattr(s, "end", 0.0)) - float(getattr(s, "start", 0.0)), 0.01)
        for s in segments
    ]
    total = sum(weights)

    def weighted(attribute):
        return sum(
            float(getattr(s, attribute, 0.0)) * w for s, w in zip(segments, weights)
        ) / total

    words = []
    for segment in segments:
        for word in getattr(segment, "words", None) or ():
            words.append([str(word.word), float(word.probability)])
    return {
        "no_speech_prob": weighted("no_speech_prob"),
        "avg_logprob": weighted("avg_logprob"),
        "compression_ratio": max(
            float(getattr(s, "compression_ratio", 0.0)) for s in segments
        ),
        "words": words,
        "segment_count": len(segments),
    }


def decode_gate(metrics: DecodeMetrics | None, config: FilterConfig):
    """Return the reason to drop the whole utterance, or None to keep it."""

    if not config.decode_gate_enabled or metrics is None:
        return None
    if metrics.no_speech_prob > config.no_speech_prob_max:
        return f"no_speech_prob>{config.no_speech_prob_max:g}"
    if metrics.avg_logprob < config.avg_logprob_min:
        return f"avg_logprob<{config.avg_logprob_min:g}"
    if metrics.compression_ratio > config.compression_ratio_max:
        return f"compression_ratio>{config.compression_ratio_max:g}"
    return None


def normalize_words(text: str) -> list[str]:
    folded = unicodedata.normalize("NFKD", str(text or "").lower())
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return re.findall(r"[a-z0-9']+", folded)


_PHRASE_TOKENS = tuple(
    sorted(
        (tuple(normalize_words(phrase)) for phrase in HALLUCINATION_PHRASES),
        key=len,
        reverse=True,
    )
)


def hallucination_reason(text: str, config: FilterConfig):
    """Return why the utterance looks like a Whisper hallucination, or None."""

    if not config.hallucination_enabled:
        return None
    tokens = normalize_words(text)
    if not tokens:
        return None
    if len(tokens) >= config.repeated_word_min and len(set(tokens)) == 1:
        return "repeated_single_word"
    index = 0
    while index < len(tokens):
        for phrase in _PHRASE_TOKENS:
            if tuple(tokens[index:index + len(phrase)]) == phrase:
                index += len(phrase)
                break
        else:
            return None
    return "known_hallucination_phrase"


def _similar(left: str, right: str) -> bool:
    if left == right:
        return True
    try:
        from rapidfuzz import fuzz
    except Exception:  # pragma: no cover - rapidfuzz is a runtime dependency
        return False
    # 75 lets one changed letter in a four-letter word align ("bubo"/"bobo").
    return fuzz.ratio(left, right) >= 75


def term_probability(term: str, words: Sequence[WordScore]):
    """Lowest probability among the Whisper words that spell ``term``.

    Each token of the term is matched to its most similar word in the
    utterance. Returns None when a token cannot be found, so the caller can
    keep the term rather than guess.
    """

    scored = [
        (normalize_words(item.word), item.probability) for item in words or ()
    ]
    flat = [
        (token, probability)
        for tokens, probability in scored
        for token in tokens
    ]
    probabilities = []
    for token in normalize_words(term):
        candidates = [p for word, p in flat if _similar(token, word)]
        if not candidates:
            return None
        probabilities.append(max(candidates))
    return min(probabilities) if probabilities else None


def filter_terms_by_probability(
    terms: Iterable[str],
    metrics: DecodeMetrics | None,
    config: FilterConfig,
):
    """Split terms into (kept, dropped, scores). Unscored terms are kept."""

    terms = list(dict.fromkeys(terms or ()))
    scores = {
        term: (term_probability(term, metrics.words) if metrics else None)
        for term in terms
    }
    if not config.word_probability_enabled or metrics is None or not metrics.words:
        return terms, [], scores
    kept, dropped = [], []
    for term in terms:
        probability = scores[term]
        floor = (
            config.urgent_word_probability_min
            if contains_urgent_directive([term])
            else config.word_probability_min
        )
        if probability is not None and probability < floor:
            dropped.append(term)
        else:
            kept.append(term)
    return kept, dropped, scores


def _label(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def media_classes_heard(top_classes, config: FilterConfig):
    """Media labels among YAMNet's top-k classes with at least min_score."""

    wanted = {_label(item) for item in config.media_classes}
    ranked = sorted(top_classes or (), key=lambda item: float(item[1]), reverse=True)
    return [
        label
        for label, score in ranked[:config.media_top_k]
        if _label(label) in wanted and float(score) >= config.media_min_score
    ]


def decode_gate_overridden(
    terms: Iterable[str],
    metrics: DecodeMetrics | None,
    config: FilterConfig,
) -> bool:
    """A threat or self-harm phrase heard clearly is never dropped by the decode gate.

    avg_logprob is low for short phrases in a noisy room even when the words
    are right (a live "papatayin kita" measured -1.02), so the decode gate
    yields when the urgent words themselves are confident.
    """

    urgent = [term for term in terms or () if contains_urgent_directive([term])]
    if not urgent:
        return False
    kept, _dropped, _scores = filter_terms_by_probability(urgent, metrics, config)
    return bool(kept)


def media_gate(top_classes, terms: Iterable[str], config: FilterConfig):
    """Return the reason to suppress the alert because media is playing, or None.

    Threat and self-harm phrases are never suppressed: a real threat said over
    music must still alert.
    """

    if not config.media_gate_enabled:
        return None
    heard = media_classes_heard(top_classes, config)
    if not heard:
        return None
    if contains_urgent_directive(terms):
        return None
    return "media_playing:" + "+".join(heard)
