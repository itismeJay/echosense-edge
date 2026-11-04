"""Correct Whisper mishearings of monitored words before matching.

Whisper has no Cebuano mode, so Bisaya insults come back mis-spelled or
mis-split: "pangit ka" as "Bang itka.", "bobo ka" as "baboka". This module
rewrites such spans to the monitored term so the normal exact matcher,
context rules and severity policy see the intended words. The original
transcript is never changed; callers keep it as the evidence of record.

Two layers:

1. ``MISHEARINGS``: explicit heard-phrase -> term rewrites, always on.
2. Sound-alike matching (``ECHOSENSE_SOUNDALIKE_MATCH``, default on):
   spaces between tokens are ignored, Bisaya spelling variation is folded
   (o/u, e/i, c/k/q, w/u, y/i, f/p, v/b, doubled letters), and
   - terms shorter than 5 letters must match exactly after folding;
   - phrases and hard insults of 6+ letters also fold voicing (b/p, d/t,
     g/k) and allow 1 letter substituted, inserted or dropped (2 for 9+
     letters), keeping the same first sound. Descriptive single words
     ("short", "ugly") never fuzz.
   Tokens in ``NEVER_CORRECT`` (common words that sit close to a monitored
   word, e.g. "pangita" = to look for) are never rewritten.

Matching never crosses a sentence boundary and never starts or ends inside
a token, so "patay" (dead) cannot become "atay".
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Iterable

from detection import school_policy
from detection.transcript_quality import transcript_token_segments


# Heard (as Whisper writes it) -> monitored term. Whole tokens only.
MISHEARINGS = {
    "baboka": "bobo ka",
    "babo ka": "bobo ka",
    "bubo ka": "bobo ka",
    "bubu ka": "bobo ka",
    "bobu ka": "bobo ka",
    "boboka": "bobo ka",
    "bubuka": "bobo ka",
    "tangaka": "tanga ka",
    "pangitka": "pangit ka",
    "bang itka": "pangit ka",
    "bangit ka": "pangit ka",
    "panget ka": "pangit ka",
    "pangid ka": "pangit ka",
    "yawaka": "yawa ka",
    "buangka": "buang ka",
    "bogoka": "bogo ka",
    "pisteka": "piste ka",
    "gagaka": "gaga ka",
    "bahoka": "baho ka",
    # Bisaya, heard in the test set by stock and Bisaya-tuned small models.
    "banalanti ka": "bunalan tika",
    "banalantika": "bunalan tika",
    "binalantika": "bunalan tika",
    "binalanti ka": "bunalan tika",
    "bunalanti ka": "bunalan tika",
    "patiuntika": "patyon tika",
    "patyuntika": "patyon tika",
    "patriuntica": "patyon tika",
    "atayka": "atay ka",
    "animalka": "animal ka",
    "amawka": "amaw ka",
    "boangka": "buang ka",
    "maotka": "maot ka",
    "luodka": "luod ka",
    # Grade 6 recordings (Bisaya-tuned small, 2026-11-04)
    "sumbag gunte ka": "sumbagon tika",
    "sumbagunte ka": "sumbagon tika",
    "sumbagon te ka": "sumbagon tika",
    "sipante ka": "sipaon tika",
    "sipanteka": "sipaon tika",
    "sipaonte ka": "sipaon tika",
    "lecheque": "leche ka",
    "lecheca": "leche ka",
    "lecheka": "leche ka",
    "sa lingkapaw ka": "salingkapaw ka",
    "salingkapawka": "salingkapaw ka",
    "ug dual nako": "ug duol nako",
    "diricaw": "diri ikaw",
    "dirikaw": "diri ikaw",
    "pobrika": "pobre ka",
    "pandacas": "pandak ka",
    "pandakas": "pandak ka",
    "banalante": "bunalan tika",
    "nihuangka": "niwang ka",
    "nihuang ka": "niwang ka",
    "itub ka": "itom ka",
    "itubka": "itom ka",
    "dikasang ka": "tikasan ka",
    "dikasan ka": "tikasan ka",
    "pandas ka": "pandak ka",
    "dakugtian ka": "dakog tiyan ka",
}

# Whisper splits the Bisaya intensifier "kaayo" (very) into "ka ayo", which
# turns "pangit kaayo ang panahon" (the weather is very bad) into the
# directed insult "pangit ka". These token pairs are re-joined before any
# matching. "ka ayo" is not a Bisaya phrase on its own.
JOINED_WORDS = {
    # Whisper joins short Bisaya words: "baboy ramo" = "baboy ra mo".
    "ramo": ("ra", "mo"),
    "rako": ("ra", "ko"),
    "rakag": ("ra", "kag"),
}

SPLIT_WORDS = {
    ("ka", "ayo"): "kaayo",
    ("ka", "ayu"): "kaayo",
    ("ka", "ayoo"): "kaayo",
    ("kaa", "yo"): "kaayo",
}

# Ordinary words that sit one sound away from a monitored word. Never
# rewritten, whatever the sound-alike rules say.
NEVER_CORRECT = frozenset({
    # Bisaya
    "pangita", "pangitaa", "pangitag", "mangita", "nangita", "gipangita",
    "patay", "namatay", "buwan", "bulan", "bayad", "bukas", "buka",
    "bunga", "bango", "banga", "pista", "pisti", "gaba", "gabii", "basa",
    "bahin", "dako", "gamay", "kana", "kini", "kaayo", "kaon", "ayaw",
    "yawi", "yaya", "papa", "mama", "baba", "bobot", "bugas", "buhat",
    "bubong", "gabaan", "bunal", "bunali",
    # "sa" (at/of) is everywhere: "bata sa istorya" is not "bata pa".
    "sa", "taga", "asa", "tiga",
    # "man" (particle) is everywhere: "dili man ka" is not "dili maayo ka".
    "man", "bukid", "mura", "kag", "animal", "ilaha", "ila", "ilang", "uban", "ungod",
    # Tagalog
    "tanghali", "tangan", "tanong", "bobong", "patayin",
    # English
    "shirt", "sport", "stupidly", "shot", "shop", "pocket", "bucket",
    "packet", "party", "pasta", "full",
})

FUZZY_MIN_LETTERS = 6
# Exact-spelling matches may cover longer phrases than close matches:
# "pangit kaayu imung nawung" is "pangit kaayo imong nawong".
EXACT_SPAN_TOKENS = 5
# Who an insult is aimed at. Correction may never change one of these into
# another ("bobo ko" = I'm stupid must not become "bobo ka" = you're stupid).
PRONOUNS = frozenset({
    "ka", "ko", "mo", "ako", "ikaw", "kita", "tika", "taka", "nako", "nimo",
    "imo", "imong", "akong", "ta", "kami", "kamo", "sila", "siya", "you",
    "me", "i", "he", "she", "we", "they",
})
_FOLD = str.maketrans({
    "c": "k", "q": "k", "f": "p", "v": "b", "z": "s",
    "o": "u", "e": "i", "w": "u", "y": "i",
})
_DEVOICE = str.maketrans({"b": "p", "d": "t", "g": "k"})
_NON_LETTERS = re.compile(r"[^a-z]+")
_DOUBLES = re.compile(r"(.)\1+")


def spelling_key(text: str) -> str:
    """Letters only, Bisaya spelling variants folded, doubled letters merged."""

    letters = _NON_LETTERS.sub("", str(text or "").lower()).replace("ph", "f")
    return _DOUBLES.sub(r"\1", letters.translate(_FOLD))


def sound_key(text: str) -> str:
    """Spelling key with voiced/voiceless pairs (b/p, d/t, g/k) merged."""

    return _DOUBLES.sub(r"\1", spelling_key(text).translate(_DEVOICE))


@dataclass(frozen=True, slots=True)
class Correction:
    heard: str
    term: str
    kind: str  # "mishearing" or "soundalike"

    def evidence(self) -> dict:
        return {"heard": self.heard, "term": self.term, "kind": self.kind}


@dataclass(frozen=True, slots=True)
class CorrectionResult:
    text: str
    corrections: tuple[Correction, ...]


@dataclass(frozen=True, slots=True)
class _Term:
    term: str
    tokens: tuple[str, ...]
    spelling: str
    sound: str
    fuzzy: bool


def _allowed_mismatches(length: int, *, single_word: bool = False) -> int:
    # Strict sensitivity allows a second differing sound on a single word from
    # 7 letters (normal: from 9). Multi-word spans keep the normal rule: joined
    # everyday phrases are long, and "dili ka" must never become "duling ka".
    long_word = 7 if single_word and school_policy.is_strict() else 9
    return 2 if length >= long_word else 1


def _edit_distance_within(a: str, b: str, limit: int) -> bool:
    """Levenshtein distance <= limit (banded, early exit)."""

    if abs(len(a) - len(b)) > limit:
        return False
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (x != y),
            )
        if min(current) > limit:
            return False
        previous = current
    return previous[-1] <= limit


def _fuzzy_equal(a: str, b: str, *, single_word: bool = False) -> bool:
    if not a or not b or a[0] != b[0]:
        return False
    return _edit_distance_within(
        a, b, _allowed_mismatches(len(b), single_word=single_word)
    )


@lru_cache(maxsize=8)
def _index(
    terms: tuple[str, ...],
    fuzzy_terms: frozenset[str],
    exact_terms: frozenset[str] = frozenset(),
):
    """Group terms by token count for matching; cached per term set."""

    prepared = []
    for term in terms:
        tokens = tuple(term.split())
        spelling = spelling_key(term)
        if not tokens or len(spelling) < 2:
            continue
        prepared.append(_Term(
            term=term,
            tokens=tokens,
            spelling=spelling,
            sound=sound_key(term),
            fuzzy=(
                term not in exact_terms
                and len(spelling) >= FUZZY_MIN_LETTERS
                and (len(tokens) > 1 or term in fuzzy_terms)
            ),
        ))
    # Longest terms first so "pangit ka" wins over "pangit".
    prepared.sort(key=lambda item: (-len(item.spelling), item.term))
    return tuple(prepared), NEVER_CORRECT


def _keeps_fixed_words(span_tokens, candidate, single_terms):
    """A pronoun or an already-monitored word must survive unchanged."""

    for token in span_tokens:
        if (token in PRONOUNS or token in single_terms) and token not in candidate.tokens:
            return False
    return True


def _changes_blocked_word(span_tokens, candidate, blocked):
    """A never-correct word may sit inside a repaired span, never be changed.

    "mura kag ungguy" -> "mura kag unggoy" keeps "mura" and "kag" as they are;
    "pangitaa nimo oy" (look for it) must not become "pangita nimo oy".
    """

    if len(span_tokens) != len(candidate.tokens):
        return any(token in blocked for token in span_tokens)
    return any(
        token in blocked and token != wanted
        for token, wanted in zip(span_tokens, candidate.tokens)
    )


def _soundalike(span_tokens, candidates, blocked, single_terms, exact_only=False):
    # A whole multi-word span with exactly a term's letters is not a
    # "correction" of a common word ("mura kag ungguy" = "mura kag unggoy"),
    # so only close matching respects the never-correct list.
    if any(token in blocked for token in span_tokens) and not (
        exact_only and len(span_tokens) > 1
    ):
        return None
    joined = "".join(span_tokens)
    spelling = spelling_key(joined)
    if len(spelling) < 2:
        return None
    # An exact spelling match anywhere beats a close match on a longer
    # term: "tamboka" is "tambok ka", never "tambok kaayo".
    for candidate in candidates:
        if spelling == candidate.spelling and not _changes_blocked_word(
            span_tokens, candidate, blocked
        ) and (
            _keeps_fixed_words(span_tokens, candidate, single_terms)
            or (len(span_tokens) == len(candidate.tokens) and len(span_tokens) > 1)
        ):
            return candidate
    if exact_only:
        return None
    sound = None
    for candidate in candidates:
        # Same letters, different word split ("sumbagonti ka" = "sumbagon
        # tika"): nothing is changed or invented, so a pronoun inside the
        # span is safe. "bobo ko" still never becomes "bobo ka".
        if (
            spelling == candidate.spelling
            and len(span_tokens) == len(candidate.tokens)
            and len(span_tokens) > 1
        ):
            return candidate
        if not _keeps_fixed_words(span_tokens, candidate, single_terms):
            continue
        if spelling == candidate.spelling:
            return candidate
        if not candidate.fuzzy:
            continue
        if abs(len(candidate.sound) - len(spelling)) > 2:
            continue
        # The target must really be heard: a fuzzy match may not invent the
        # pronoun ("ang bobo" is not "ang bobo mo").
        if any(
            spelling_key(token) not in spelling
            for token in candidate.tokens
            if token in PRONOUNS
        ):
            continue
        if sound is None:
            sound = sound_key(joined)
        single_word = len(span_tokens) == 1 and len(candidate.tokens) == 1
        if sound == candidate.sound or _fuzzy_equal(
            sound, candidate.sound, single_word=single_word
        ):
            return candidate
    return None


def _mishearing_tables():
    by_length = {}
    for heard, term in MISHEARINGS.items():
        tokens = tuple(heard.split())
        by_length.setdefault(len(tokens), {})[tokens] = term
    return sorted(by_length.items(), reverse=True)


_MISHEARING_TABLES = _mishearing_tables()


# "gyudka" -> "gyud ka", "kaayomo" -> "kaayo mo": a particle joined to "you".
_JOINED_PARTICLE = re.compile(
    r"^(gyud|jud|man|baya|kaayo|kaayu|kayo|ra|pud|sad)(ka|mo|ko)$"
)


def _split_joined(token):
    parts = JOINED_WORDS.get(token)
    if parts:
        return parts
    match = _JOINED_PARTICLE.match(token)
    return (match.group(1), match.group(2)) if match else None


def _rejoin_split_words(tokens, corrections):
    expanded = []
    for token in tokens:
        parts = _split_joined(token)
        if parts:
            expanded.extend(parts)
            corrections.append(Correction(token, " ".join(parts), "mishearing"))
        else:
            expanded.append(token)
    tokens = expanded
    out = []
    index = 0
    while index < len(tokens):
        pair = tuple(tokens[index:index + 2])
        joined = SPLIT_WORDS.get(pair) if len(pair) == 2 else None
        if joined:
            out.append(joined)
            corrections.append(Correction(" ".join(pair), joined, "mishearing"))
            index += 2
            continue
        out.append(tokens[index])
        index += 1
    return out


def correct_transcript(
    text: object,
    terms: Iterable[str] = (),
    *,
    fuzzy_terms: Iterable[str] = (),
    exact_terms: Iterable[str] = (),
    soundalike: bool = True,
    mishearings: bool = True,
    max_span_tokens: int = 3,
) -> CorrectionResult:
    """Return ``text`` with mishearings rewritten to monitored terms.

    ``terms`` are the monitored terms (normalized). ``fuzzy_terms`` are the
    single-word terms allowed to match with one differing sound (hard
    insults). Segments are re-joined with ". " so later phrase matching
    still cannot cross a sentence boundary.
    """

    segments = transcript_token_segments(text)
    candidates, blocked = (
        _index(
            tuple(sorted(set(terms))), frozenset(fuzzy_terms), frozenset(exact_terms),
        )
        if soundalike
        else ((), frozenset())
    )
    exact_terms = {candidate.tokens for candidate in candidates}
    single_terms = {
        candidate.tokens[0] for candidate in candidates if len(candidate.tokens) == 1
    }
    corrections: list[Correction] = []
    output_segments = []
    for segment in segments:
        tokens = list(segment)
        if mishearings:
            tokens = _rejoin_split_words(tokens, corrections)
        out: list[str] = []
        index = 0
        while index < len(tokens):
            replaced = False
            for length, table in (_MISHEARING_TABLES if mishearings else ()):
                span = tuple(tokens[index:index + length])
                if len(span) == length and span in table:
                    term = table[span]
                    out.extend(term.split())
                    corrections.append(Correction(" ".join(span), term, "mishearing"))
                    index += length
                    replaced = True
                    break
            if replaced:
                continue
            if candidates:
                # An exact monitored term starting here always wins over a
                # sound-alike guess: keep it untouched.
                for length in range(max_span_tokens, 0, -1):
                    span = tuple(tokens[index:index + length])
                    if len(span) == length and span in exact_terms:
                        out.extend(span)
                        index += length
                        replaced = True
                        break
            if replaced:
                continue
            if candidates:
                # Exact spellings at every span length before any close
                # match: "tamboka niyo" is "tambok ka" + "niyo", not a
                # near-miss of "tambok kaayo".
                attempts = [
                    (length, True) for length in range(EXACT_SPAN_TOKENS, 0, -1)
                ] + [
                    (length, False) for length in range(max_span_tokens, 0, -1)
                ]
                for length, exact in attempts:
                    span = tuple(tokens[index:index + length])
                    if len(span) != length:
                        continue
                    match = _soundalike(
                        span, candidates, blocked, single_terms, exact_only=exact,
                    )
                    if match is not None:
                        out.extend(match.tokens)
                        if match.tokens != span:
                            corrections.append(
                                Correction(" ".join(span), match.term, "soundalike")
                            )
                        index += length
                        replaced = True
                        break
            if not replaced:
                out.append(tokens[index])
                index += 1
        output_segments.append(" ".join(out))
    return CorrectionResult(". ".join(output_segments), tuple(corrections))
