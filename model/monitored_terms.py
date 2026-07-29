"""Multilingual monitored-term matching and backend dictionary synchronization.

The live detector keeps its existing hard/soft acoustic and context gates.  This
module adds structured evidence around the terms that those gates already use,
and adds backend dictionary entries as soft evidence unless an entry is already
part of the local hard-trigger list.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Iterable, Mapping, Optional, Sequence

import requests

from config import API_URL
from detection.transcript_quality import normalize_transcript, phrase_matches


SUPPORTED_LANGUAGES = frozenset({"fil", "ceb", "en", "mixed", "unknown"})
WHISPER_LANGUAGE_ALIASES = {
    "tl": "fil",
    "fil": "fil",
    "tagalog": "fil",
    "en": "en",
    "english": "en",
    "ceb": "ceb",
    "cebuano": "ceb",
}
BACKEND_LANGUAGE_ALIASES = {
    **WHISPER_LANGUAGE_ALIASES,
    "filipino": "fil",
    "bisaya": "ceb",
    "mixed": "mixed",
}

# Representative local fallback phrases are explicitly grouped by language.
# The full legacy hard/soft lists remain in blacklist.py so their alert gating
# and severity behavior do not change.
LOCAL_TERMS_BY_LANGUAGE = {
    "fil": {
        "ikaw ay bobo",
        "pangit ka talaga",
        "ang tanga mo",
        "napaka bobo mo",
        "walang kwenta ka",
        "papatayin kita",
        "mukha kang baboy",
        "bobo",
        "tanga",
        "gago",
        "pangit",
    },
    "ceb": {
        "buang kaayo ka",
        "yawa ka gyud",
        "wala kay pulos",
        "patyon tika",
        "dakog ilong",
        "murag baboy ka",
        "dili ka gusto",
        "yawa",
        "buang",
        "bogo",
        "tambok",
    },
    "en": {
        "you are stupid",
        "you are so ugly",
        "nobody likes you",
        "you are worthless",
        "kill yourself",
        "go away loser",
        "stupid",
        "idiot",
        "loser",
    },
    "mixed": {
        "bobo kaayo ka",
        "stupid kaayo ka",
        "you are bobo",
        "fat ka",
        "flat nose ka",
        "slow kaayo",
        "pangit kaayo imong face",
    },
}

_LANGUAGE_MARKERS = {
    "fil": {
        "ako", "ang", "ay", "bobo", "gago", "hindi", "ikaw", "kita",
        "mukha", "mo", "ng", "napaka", "namin", "pangit", "papatayin",
        "parang", "sarili", "tanga", "walang",
    },
    "ceb": {
        "bogo", "buang", "dakog", "dili", "giatay", "gyud", "ilong",
        "imo", "imong", "kaayo", "kay", "murag", "nimo", "og", "patyon",
        "piste", "pulos", "tambok", "tika", "ug", "wala", "walay", "yawa",
    },
    "en": {
        "are", "dumb", "fat", "go", "idiot", "kill", "likes", "loser",
        "nobody", "stupid", "ugly", "worthless", "you", "your",
    },
}


def normalize_text(value: str) -> str:
    """Backward-compatible alias for the centralized safe normalizer."""

    return normalize_transcript(value)


def _matches(term: str, source_text: str) -> bool:
    return phrase_matches(term, source_text)


def normalize_language(value: object, *, backend: bool = False) -> str:
    aliases = BACKEND_LANGUAGE_ALIASES if backend else WHISPER_LANGUAGE_ALIASES
    normalized = str(value or "").strip().lower()
    mapped = aliases.get(normalized, normalized)
    return mapped if mapped in SUPPORTED_LANGUAGES else "unknown"


def infer_term_language(term: str) -> str:
    """Infer a fallback term's language from explicit groups and strong tokens."""

    normalized = normalize_text(term)
    for language, terms in LOCAL_TERMS_BY_LANGUAGE.items():
        if normalized in terms:
            return language

    tokens = set(normalized.split())
    scores = {
        language: len(tokens & markers)
        for language, markers in _LANGUAGE_MARKERS.items()
    }
    evidenced = {language for language, score in scores.items() if score > 0}
    if len(evidenced) > 1:
        return "mixed"
    if len(evidenced) == 1:
        return next(iter(evidenced))
    return "unknown"


@dataclass(frozen=True)
class MonitoredTerm:
    term: str
    language: str
    term_id: Optional[int] = None

    @property
    def match_type(self) -> str:
        return "phrase" if len(self.term.split()) > 1 else "word"

    def evidence(self) -> dict:
        evidence = {
            "term": self.term,
            "language": self.language,
            "match_type": self.match_type,
        }
        if self.term_id is not None:
            evidence["term_id"] = self.term_id
        return evidence


_BACKEND_TERMS: tuple[MonitoredTerm, ...] = ()
_BACKEND_TERMS_LOCK = threading.Lock()
_SYNC_THREAD: Optional[threading.Thread] = None
_SYNC_THREAD_LOCK = threading.Lock()


def parse_backend_terms(entries: Iterable[Mapping]) -> tuple[MonitoredTerm, ...]:
    """Validate backend dictionary rows without inventing missing IDs."""

    parsed = []
    seen_ids = set()
    for entry in entries or ():
        term_id = entry.get("term_id")
        if isinstance(term_id, bool) or not isinstance(term_id, int):
            continue
        term = normalize_text(entry.get("slur_text") or entry.get("term"))
        if not term or term_id in seen_ids:
            continue
        parsed.append(
            MonitoredTerm(
                term=term,
                language=normalize_language(entry.get("language"), backend=True),
                term_id=term_id,
            )
        )
        seen_ids.add(term_id)
    return tuple(sorted(parsed, key=lambda item: (-len(item.term.split()), item.term_id)))


def replace_backend_terms(entries: Iterable[Mapping]) -> tuple[MonitoredTerm, ...]:
    """Atomically replace synchronized terms.  Public for deterministic tests."""

    parsed = parse_backend_terms(entries)
    global _BACKEND_TERMS
    with _BACKEND_TERMS_LOCK:
        _BACKEND_TERMS = parsed
    return parsed


def get_backend_terms() -> tuple[MonitoredTerm, ...]:
    with _BACKEND_TERMS_LOCK:
        return _BACKEND_TERMS


def match_backend_terms(normalized_text: str) -> list[MonitoredTerm]:
    return [
        term
        for term in get_backend_terms()
        if _matches(term.term, normalized_text)
    ]


def _local_candidates(normalized_text: str, legacy_hits: Sequence[str]) -> list[MonitoredTerm]:
    candidates = []
    seen = set()

    for language in sorted(LOCAL_TERMS_BY_LANGUAGE):
        for term in sorted(LOCAL_TERMS_BY_LANGUAGE[language]):
            if _matches(term, normalized_text):
                key = (term, language)
                if key not in seen:
                    candidates.append(MonitoredTerm(term=term, language=language))
                    seen.add(key)

    for hit in legacy_hits or ():
        term = normalize_text(hit)
        # Ignore stale legacy evidence that does not exactly occur in this text.
        if candidates and not _matches(term, normalized_text):
            continue
        language = infer_term_language(term)
        key = (term, language)
        if term and key not in seen:
            candidates.append(MonitoredTerm(term=term, language=language))
            seen.add(key)

    # Prefer the most specific phrase over a local word contained by it.  Terms
    # carrying backend IDs are handled separately and are never discarded.
    candidates.sort(
        key=lambda item: (
            -len(item.term.split()),
            -len(item.term),
            item.term,
            item.language,
        )
    )
    preferred = []
    for candidate in candidates:
        if any(
            candidate.term != selected.term
            and _matches(candidate.term, selected.term)
            for selected in preferred
        ):
            continue
        preferred.append(candidate)
    return preferred


def build_matched_terms(
    normalized_text: str,
    legacy_hits: Sequence[str] = (),
    excluded_terms: Sequence[str] = (),
) -> list[dict]:
    """Build de-duplicated evidence, preserving every real backend term ID."""

    excluded = {normalize_text(term) for term in excluded_terms}
    backend_matches = [
        item
        for item in match_backend_terms(normalized_text)
        if item.term not in excluded
    ]
    local_matches = [
        item
        for item in _local_candidates(normalized_text, legacy_hits)
        if item.term not in excluded
    ]

    evidence = []
    seen = set()
    backend_texts = {item.term for item in backend_matches}
    for item in [*backend_matches, *local_matches]:
        # Prefer the synchronized representation when it is the same term and
        # language, because it carries the authoritative term ID.
        if item.term_id is None and item.term in backend_texts:
            continue
        key = (
            ("id", item.term_id)
            if item.term_id is not None
            else ("text", item.term, item.language, item.match_type)
        )
        if key in seen:
            continue
        evidence.append(item.evidence())
        seen.add(key)
    return evidence


def _real_confidence(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(confidence) or confidence <= 0.0 or confidence > 1.0:
        return None
    return confidence


def classify_transcript_language(
    whisper_language: object,
    whisper_confidence: object,
    matched_terms: Sequence[Mapping],
) -> tuple[str, Optional[float]]:
    """Combine native Whisper metadata with strong monitored-term evidence."""

    detected = normalize_language(whisper_language)
    confidence = _real_confidence(whisper_confidence)
    evidence_languages = {
        normalize_language(item.get("language"), backend=True)
        for item in matched_terms or ()
    }
    evidence_languages.discard("unknown")

    if "mixed" in evidence_languages or len(evidence_languages - {"mixed"}) > 1:
        return "mixed", None

    supported_evidence = evidence_languages & {"fil", "ceb", "en"}
    if len(supported_evidence) == 1:
        language = next(iter(supported_evidence))
        return language, confidence if detected == language else None

    if detected in {"fil", "ceb", "en"} and confidence is not None:
        return detected, confidence
    return "unknown", None


def sync_backend_dictionary(session=requests, timeout: float = 10.0) -> bool:
    """Fetch current monitored terms.  Failures leave the last good copy active."""

    try:
        response = session.get(f"{API_URL}/dictionary/", timeout=timeout)
        response.raise_for_status()
        entries = response.json()
        if not isinstance(entries, list):
            raise ValueError("dictionary response is not a list")
        parsed = replace_backend_terms(entries)
        print(f"[DICTIONARY] Synchronized {len(parsed)} monitored terms")
        return True
    except Exception as exc:
        print(f"[DICTIONARY] Sync unavailable ({type(exc).__name__}); using current terms")
        return False


def start_dictionary_sync(interval: float = 300.0) -> threading.Thread:
    """Synchronize now, then refresh periodically in a single daemon thread."""

    global _SYNC_THREAD
    with _SYNC_THREAD_LOCK:
        if _SYNC_THREAD and _SYNC_THREAD.is_alive():
            return _SYNC_THREAD

        sync_backend_dictionary()

        def _sync_loop() -> None:
            while True:
                time.sleep(interval)
                sync_backend_dictionary()

        _SYNC_THREAD = threading.Thread(
            target=_sync_loop,
            name="dictionary-sync",
            daemon=True,
        )
        _SYNC_THREAD.start()
        return _SYNC_THREAD
