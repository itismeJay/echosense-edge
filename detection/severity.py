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
    "mo",
    "ikaw",
    "imo",
    "imong",
    "nimo",
    "you",
    "your",
    "you're",
    "classmate",
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


def _promote(level: str) -> str:
    if level == LOW:
        return MEDIUM
    if level == MEDIUM:
        return HIGH
    return HIGH


@dataclass(frozen=True, slots=True)
class SeverityDecision:
    level: str
    reasons: tuple[str, ...]
    term_categories: dict[str, tuple[str, ...]]
    supporting_evidence: tuple[str, ...]

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


def calculate_severity(
    monitored_terms: Iterable[object],
    *,
    transcript: object = "",
    duration: float = 0.0,
    repeated: bool = False,
    acoustic_aggressive: bool = False,
    laughter_present: bool = False,
) -> SeverityDecision:
    """Calculate severity from observable text, duration, and acoustic evidence."""

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

    reasons: list[str] = []
    supporting: list[str] = []
    if any(
        category in categories
        for category in (
            "self_harm_directive",
            "threat_like_phrase",
            "severe_direct_harassment",
        )
    ):
        level = HIGH
        reasons.extend(
            f"term_category:{category}"
            for category in (
                "self_harm_directive",
                "threat_like_phrase",
                "severe_direct_harassment",
            )
            if category in categories
        )
    elif "direct_harassment" in categories:
        level = MEDIUM
        reasons.append("term_category:direct_harassment")
    else:
        level = LOW
        reasons.append(
            "term_category:low_risk_or_uncategorized"
            if terms
            else "no_monitored_term"
        )

    direct_target = has_direct_target_evidence(transcript)
    if direct_target:
        supporting.append("direct_target_pattern")
        if terms and level == LOW:
            level = MEDIUM
            reasons.append("promoted_by:direct_target_pattern")

    if repeated:
        supporting.append("repeated_monitored_evidence")
        promoted = _promote(level)
        if promoted != level:
            level = promoted
            reasons.append("promoted_by:repetition")

    if acoustic_aggressive:
        supporting.append("aggressive_acoustic_evidence")
        if terms and level == LOW:
            level = MEDIUM
            reasons.append("promoted_by:acoustic_evidence")

    duration_level = severity_from_duration(float(duration or 0.0))
    if SEVERITY_ORDER[duration_level] > SEVERITY_ORDER[level]:
        level = duration_level
        reasons.append(f"duration_floor:{duration_level}")

    if laughter_present:
        # Laughter is context metadata, never evidence that cancels a HIGH term.
        supporting.append("laughter_or_excitement_marker_present")

    return SeverityDecision(
        level=level,
        reasons=tuple(dict.fromkeys(reasons)),
        term_categories=categories,
        supporting_evidence=tuple(dict.fromkeys(supporting)),
    )


def contains_high_severity_term(monitored_terms: Iterable[object]) -> bool:
    return calculate_severity(monitored_terms).level == HIGH


def contains_urgent_directive(monitored_terms: Iterable[object]) -> bool:
    normalized = {_normalize_term(term) for term in monitored_terms or ()}
    return bool(normalized & (SELF_HARM_DIRECTIVES | THREAT_LIKE_PHRASES))
