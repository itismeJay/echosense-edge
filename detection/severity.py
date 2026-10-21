"""Authoritative, explainable severity rules for edge alert decisions.

Severity describes observable evidence in one finalized event. It does not
confirm bullying, intent, guilt, danger, or speaker identity.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


LOW = "LOW"
MEDIUM = "MEDIUM"
HIGH = "HIGH"
SEVERITY_LEVELS = (LOW, MEDIUM, HIGH)
SEVERITY_ORDER = {LOW: 0, MEDIUM: 1, HIGH: 2}

# These values are unchanged from detection/thresholds.py. They live here
# because duration is one input to the authoritative severity calculator.
SEVERITY_HIGH_DURATION = 7.0
SEVERITY_MEDIUM_DURATION = 4.0


# Observable phrase categories. These are exact normalized monitored terms,
# never statements about the speaker's intent.
SELF_HARM_DIRECTIVES = frozenset({
    "kill yourself",
    "go kill yourself",
    "go die",
    "patay ka na sana",
})

THREAT_LIKE_PHRASES = frozenset({
    "kill you",
    "patyon tika",
    "patyon ka nako",
    "papatayin kita",
    "mamamatay ka",
    "gusto kag sumbagay",
    "suwayi rag duol",
    "sumbagay ta",
    "bungoan tika",
    "suntukan ta",
    "sampalan tika",
})

SEVERE_DIRECT_HARASSMENT_TERMS = frozenset({
    "yawa",
    "giatay",
    "bilat",
    "kayat",
    "iyot",
    "putangina",
    "putang ina",
    "tang ina",
    "tangina",
    "pakyu",
    "puta",
    "anak ng puta",
    "anak og puta",
    "pesteng yawa",
    "monggi",
    "retard",
    "uling",
    "murag uling",
    "agta",
    "negra",
    "negro",
    "bungi",
    "kulisap ang buhok",
    "mukha kang unggoy",
})

MEDIUM_DIRECT_HARASSMENT_TERMS = frozenset({
    "bogo",
    "bugok",
    "bulok",
    "buang",
    "buang ka",
    "buang kaayo",
    "gago",
    "gaga",
    "bobo",
    "tanga",
    "ulol",
    "way utok",
    "walang kwenta",
    "wala kang kwenta",
    "walay gustong makig uban sa imo",
    "isuka ka sa imuhang mama",
    "worthless",
    "inutil",
    "bugits",
    "bogo kaayo",
    "bulok man ka",
    "tambokikoy",
    "murag litson",
    "murag ungo",
    "nawong mo murag ungo",
    "itom kaayo murag uling",
    "dakog ulo walay laman",
    "ang pangit ng mukha mo",
    "nobody likes you",
    "you are worthless",
    "dakog ilong",
    "away ta",
})

LOW_RISK_MONITORED_TERMS = frozenset({
    "pangit",
    "tambok",
    "itom",
    "putot",
    "baho",
    "pikon",
    "sumbong",
    "hilak nasad",
    "iiyak na yan",
    "ugly",
    "fat",
    "crybaby",
    "loser",
    "freak",
    "ampon",
    "luod kaayo ka",
    "dakog dunggan",
    "pango",
    "niwang",
    "pandak",
    "malaking ilong",
    "malaking tenga",
})

HIGH_SEVERITY_TERMS = frozenset(
    SELF_HARM_DIRECTIVES
    | THREAT_LIKE_PHRASES
    | SEVERE_DIRECT_HARASSMENT_TERMS
)
MEDIUM_SEVERITY_TERMS = MEDIUM_DIRECT_HARASSMENT_TERMS
LOW_SEVERITY_TERMS = LOW_RISK_MONITORED_TERMS

_DIRECT_TARGET_TOKENS = frozenset({
    "ka",
    "kaw",
    "mo",
    "ikaw",
    "imo",
    "imong",
    "nimo",
    "you",
    "your",
    "you're",
    "youre",
    "ur",
    "u",
    "classmate",
})

# Context rules that keep mild/medium terms at LOW. They never apply to
# threat-like, self-harm, or severe terms.
_SELF_DIRECTED_TOKENS = frozenset({
    "ko", "ako", "akoa", "nako", "i'm", "im", "me", "myself",
})
_SELF_DIRECTED_SEQUENCES = (("i", "am"),)
_NEGATION_TOKENS = frozenset({
    "huwag", "wag", "ayaw", "don't", "dont", "bawal",
})
_NEGATION_SEQUENCES = (
    ("do", "not"),
    ("hindi", "dapat"),
    ("stop", "saying"),
)
_OBJECT_ARTICLES = frozenset({"yung", "ang", "ning", "kining", "kanang", "the"})
# Things, not people. Deliberately an allowlist: an unknown noun (possibly a
# classmate's name) is never treated as an object.
_OBJECT_NOUNS = frozenset({
    "game", "games", "question", "questions", "test", "tests", "thing",
    "things", "quiz", "exam", "homework", "assignment", "project", "lesson",
    "movie", "show", "song", "video", "joke", "rule", "rules", "idea",
    "phone", "computer", "app", "wifi", "internet", "signal", "book",
    "laro", "tanong", "pagsusulit", "sagot", "aralin", "takdang",
    "dula", "pangutana", "libro", "kanta",
})
_WORD_PATTERN = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.UNICODE)


def normalize_severity(value: object, *, default: str = LOW) -> str:
    """Normalize legacy lowercase inputs to the internal uppercase contract."""

    normalized = str(value or "").strip().upper()
    return normalized if normalized in SEVERITY_ORDER else default


def max_severity(*values: object) -> str:
    """Return the strongest valid severity, accepting legacy lowercase values."""

    normalized = [normalize_severity(value) for value in values]
    return max(normalized, key=SEVERITY_ORDER.__getitem__, default=LOW)


def severity_from_confidence(confidence: float) -> str:
    """Compatibility rule for legacy callers that only have a confidence value."""

    if confidence >= 0.85:
        return HIGH
    if confidence >= 0.70:
        return MEDIUM
    return LOW


def severity_from_duration(duration: float) -> str:
    """Return the unchanged duration-only severity floor."""

    if duration >= SEVERITY_HIGH_DURATION:
        return HIGH
    if duration >= SEVERITY_MEDIUM_DURATION:
        return MEDIUM
    return LOW


def _normalize_term(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def has_direct_target_evidence(transcript: object) -> bool:
    tokens = {
        token.lower()
        for token in _WORD_PATTERN.findall(str(transcript or "").lower())
    }
    return bool(tokens & _DIRECT_TARGET_TOKENS)


def _tokens(text: object) -> tuple[str, ...]:
    normalized = str(text or "").lower().replace("\u2019", "'")
    return tuple(_WORD_PATTERN.findall(normalized))


def _sequence_positions(tokens, sequence):
    length = len(sequence)
    return [
        index
        for index in range(len(tokens) - length + 1)
        if tokens[index:index + length] == sequence
    ]


def _term_positions(tokens, terms):
    positions = []
    for term in terms:
        positions.extend(_sequence_positions(tokens, tuple(_tokens(term))))
    return sorted(positions)


def _max_term_occurrences(tokens, terms) -> int:
    return max(
        (len(_sequence_positions(tokens, tuple(_tokens(term)))) for term in terms),
        default=0,
    )


def _context_downgrade(tokens, terms) -> str | None:
    """Explain why mild/medium terms are not directed at a person, if so."""

    positions = _term_positions(tokens, terms)
    if not positions:
        return None
    first = positions[0]
    if any(token in _NEGATION_TOKENS for token in tokens[:first]) or any(
        index < first
        for sequence in _NEGATION_SEQUENCES
        for index in _sequence_positions(tokens, sequence)
    ):
        return "negated_or_instructional"
    if set(tokens) & _DIRECT_TARGET_TOKENS:
        return None
    if set(tokens) & _SELF_DIRECTED_TOKENS or any(
        _sequence_positions(tokens, sequence)
        for sequence in _SELF_DIRECTED_SEQUENCES
    ):
        return "self_directed"
    for term in terms:
        term_tokens = tuple(_tokens(term))
        for index in _sequence_positions(tokens, term_tokens):
            after = tokens[index + len(term_tokens):index + len(term_tokens) + 2]
            if after[:1] and after[0] in _OBJECT_NOUNS:
                return "about_object"
            if (
                len(after) == 2
                and after[0] in _OBJECT_ARTICLES
                and after[1] in _OBJECT_NOUNS
            ):
                return "about_object"
    return None


@dataclass(frozen=True, slots=True)
class SeverityDecision:
    level: str
    reasons: tuple[str, ...]
    term_categories: dict[str, tuple[str, ...]]
    supporting_evidence: tuple[str, ...]
    context_downgrade: str | None = None

    def evidence(self) -> dict:
        return {
            "level": self.level,
            "reasons": list(self.reasons),
            "term_categories": {
                category: list(terms)
                for category, terms in self.term_categories.items()
            },
            "supporting_evidence": list(self.supporting_evidence),
        }


def is_targeted(monitored_terms: Iterable[object], transcript: object) -> bool:
    """Direct-target wording (ka/mo/ikaw/you...) not negated/self/object."""

    terms = [
        term
        for term in (_normalize_term(item) for item in monitored_terms or ())
        if term
    ]
    if not terms or not has_direct_target_evidence(transcript):
        return False
    return _context_downgrade(_tokens(transcript), terms) is None


def calculate_severity(
    monitored_terms: Iterable[object],
    *,
    transcript: object = "",
    duration: float = 0.0,
    repeated: bool = False,
    acoustic_aggressive: bool = False,
    laughter_present: bool = False,
    strong_acoustic: bool = False,
) -> SeverityDecision:
    """Severity policy: no monitored word, no alert. Sound alone never alerts.

    LOW    monitored word with no aggression (self-directed, about a thing,
           negated, playful, or untargeted).
    MEDIUM monitored word + aggression: direct target, or a time-linked
           aggressive tone/shout (``acoustic_aggressive``).
    HIGH   threat/self-harm term (always); or direct target AND a time-linked
           strong shout (``strong_acoustic``); or targeted words repeated in
           the utterance or across the incident (``repeated``).

    ``acoustic_aggressive``/``strong_acoustic`` must already be time-linked to
    this utterance. ``duration`` is accepted for legacy callers and ignored.
    Laughter never lowers a threat. Every level records its reasons.
    """

    terms = tuple(dict.fromkeys(
        term
        for term in (_normalize_term(item) for item in monitored_terms or ())
        if term
    ))
    categories = {
        "self_harm_directive": tuple(
            term for term in terms if term in SELF_HARM_DIRECTIVES
        ),
        "threat_like_phrase": tuple(
            term for term in terms if term in THREAT_LIKE_PHRASES
        ),
        "severe_direct_harassment": tuple(
            term for term in terms if term in SEVERE_DIRECT_HARASSMENT_TERMS
        ),
        "direct_harassment": tuple(
            term for term in terms if term in MEDIUM_DIRECT_HARASSMENT_TERMS
        ),
        "low_risk_monitored_term": tuple(
            term for term in terms if term in LOW_RISK_MONITORED_TERMS
        ),
    }
    categories = {
        category: matched
        for category, matched in categories.items()
        if matched
    }

    def decision(level, reasons, supporting, context=None):
        return SeverityDecision(
            level=level,
            reasons=tuple(dict.fromkeys(reasons)),
            term_categories=categories,
            supporting_evidence=tuple(dict.fromkeys(supporting)),
            context_downgrade=context,
        )

    if not terms:
        return decision(LOW, ["no_monitored_term"], [])

    reasons = [f"term_category:{category}" for category in categories]
    if not categories:
        reasons.append("term_category:uncategorized_monitored_term")
    supporting: list[str] = []
    tokens = _tokens(transcript)
    if _max_term_occurrences(tokens, terms) >= 2:
        repeated = True
        supporting.append("repeated_within_utterance")
    direct_target = has_direct_target_evidence(transcript)
    if direct_target:
        supporting.append("direct_target_pattern")
    if repeated:
        supporting.append("repeated_monitored_evidence")
    if acoustic_aggressive:
        supporting.append("aggressive_acoustic_evidence")
    if strong_acoustic:
        supporting.append("strong_or_sustained_acoustic_evidence")
    if laughter_present:
        supporting.append("laughter_or_excitement_marker_present")

    if "self_harm_directive" in categories or "threat_like_phrase" in categories:
        reasons.append("high:threat_or_self_harm_term")
        if laughter_present:
            reasons.append("laughter_ignored:threat_or_self_harm_term")
        return decision(HIGH, reasons, supporting)

    context = _context_downgrade(tokens, terms)
    if context:
        reasons.append(f"held_low:{context}")
        if acoustic_aggressive:
            reasons.append(f"corroboration_ignored:{context}")
        return decision(LOW, reasons, supporting, context)

    targeted = direct_target
    if laughter_present and not strong_acoustic and not repeated:
        reasons.append("held_low:playful_context")
        return decision(LOW, reasons, supporting)

    if targeted and strong_acoustic:
        reasons.append("promoted_by:direct_target_and_strong_shout")
        return decision(HIGH, reasons, supporting)
    if targeted and repeated:
        reasons.append("promoted_by:targeted_repetition")
        return decision(HIGH, reasons, supporting)
    if targeted or acoustic_aggressive:
        if targeted:
            reasons.append("promoted_by:direct_target_pattern")
        if acoustic_aggressive:
            reasons.append("promoted_by:time_linked_aggressive_tone")
        return decision(MEDIUM, reasons, supporting)
    reasons.append("held_low:no_aggression")
    return decision(LOW, reasons, supporting)


def contains_high_severity_term(monitored_terms: Iterable[object]) -> bool:
    return calculate_severity(monitored_terms).level == HIGH


def contains_urgent_directive(monitored_terms: Iterable[object]) -> bool:
    normalized = {_normalize_term(term) for term in monitored_terms or ()}
    return bool(normalized & (SELF_HARM_DIRECTIVES | THREAT_LIKE_PHRASES))
