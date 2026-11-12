"""Bisaya bullying lexicon: lexicon/bisaya_bullying.yaml expanded into phrases.

Each root expands through the patterns of its ``kind`` (adjective, noun,
swear, threat) plus hand-written ``extra`` phrases and ``end_only`` phrases
(which count only when nothing follows them in the sentence; enforced in
model/blacklist.py). ``everyday`` sentences use
the same words innocently and must never alert (tests/test_lexicon.py).

Pure data: imports nothing from detection.severity or model.blacklist, so both
can import it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re

LEXICON_PATH = Path(__file__).resolve().parents[1] / "lexicon" / "bisaya_bullying.yaml"

# What a phrase does, which detection/severity.py turns into a level.
INSULT = "insult"        # aimed insult: MEDIUM
SWEAR = "swear"          # bare swear word: LOW
CONTEXT = "context"      # exclusion/pressure: MEDIUM when aimed
THREAT = "threat"        # HIGH
SELF_HARM = "self_harm"  # HIGH


# Core forms that keep sound-alike ("close enough") matching.
CORE_PATTERNS = frozenset({
    "pattern:{w} ka", "pattern:{t} tika", "pattern:{t} taka",
})


# Hand-written phrases listed under a root's ``end_only:`` key.
END_ONLY_SOURCE = "end_only"


@dataclass(frozen=True, slots=True)
class Phrase:
    text: str
    category: str
    root: str
    role: str
    source: str  # "pattern:<pattern>" or "extra"
    priority: str


@dataclass(frozen=True, slots=True)
class Root:
    name: str
    category: str
    meaning: str
    kind: str
    priority: str
    phrases: tuple[Phrase, ...]
    everyday: tuple[str, ...]


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _role_for(kind: str, severity: str | None) -> str:
    if severity in (THREAT, SELF_HARM, CONTEXT):
        return severity
    if kind == "threat":
        return THREAT
    return INSULT


def _expand(name, spec, category, patterns):
    kind = spec.get("kind", "phrase_only")
    priority = spec.get("priority", "medium")
    role = _role_for(kind, spec.get("severity"))
    out: dict[str, Phrase] = {}

    def add(text, phrase_role, source):
        text = _clean(text)
        if text and text not in out:
            out[text] = Phrase(text, category, name, phrase_role, source, priority)

    if kind == "adjective":
        for pattern in patterns["adjective"]:
            add(pattern.format(w=name), role, f"pattern:{pattern}")
        if spec.get("exclamative"):
            for pattern in patterns["exclamative"]:
                add(pattern.format(x=spec["exclamative"]), role, f"pattern:{pattern}")
    elif kind == "noun":
        for pattern in patterns["noun"]:
            add(pattern.format(w=name), role, f"pattern:{pattern}")
    elif kind == "swear":
        for pattern in patterns["swear"]:
            add(pattern.format(w=name), INSULT, f"pattern:{pattern}")
        if spec.get("bare"):
            for pattern in patterns["swear_bare"]:
                add(pattern.format(w=name), SWEAR, f"pattern:{pattern}")
        if spec.get("vocative"):
            for pattern in patterns["vocative"]:
                text = pattern.format(v=spec["vocative"])
                add(text, SWEAR if text == spec["vocative"] else INSULT, f"pattern:{pattern}")
    elif kind == "threat":
        for pattern in patterns["threat"]:
            add(pattern.format(t=name), THREAT, f"pattern:{pattern}")
    for text in spec.get("extra") or ():
        add(text, role, "extra")
    for text in spec.get("end_only") or ():
        add(text, role, END_ONLY_SOURCE)
    return Root(
        name=name,
        category=category,
        meaning=str(spec.get("meaning", "")),
        kind=kind,
        priority=priority,
        phrases=tuple(out.values()),
        everyday=tuple(_clean(item) for item in spec.get("everyday") or ()),
    )


@dataclass(frozen=True, slots=True)
class Lexicon:
    roots: tuple[Root, ...]
    labels: dict

    @property
    def phrases(self) -> tuple[Phrase, ...]:
        seen: dict[str, Phrase] = {}
        for root in self.roots:
            for phrase in root.phrases:
                seen.setdefault(phrase.text, phrase)
        return tuple(seen.values())

    def texts(self, *roles: str) -> frozenset[str]:
        return frozenset(p.text for p in self.phrases if not roles or p.role in roles)

    @property
    def end_only(self) -> frozenset[str]:
        """Phrases that count only at the end of a sentence (``end_only:``)."""

        return frozenset(p.text for p in self.phrases if p.source == END_ONLY_SOURCE)

    @property
    def generated(self) -> frozenset[str]:
        """Pattern-made variants: exact matching only, never fuzzy.

        The core forms ("tambok ka", "patyon tika", "yawaa") keep sound-alike
        matching, so a misheard threat still alerts.
        """
        return frozenset(
            p.text for p in self.phrases if p.source not in CORE_PATTERNS
        )


@lru_cache(maxsize=1)
def load_lexicon(path: str | None = None) -> Lexicon:
    import yaml

    data = yaml.safe_load(Path(path or LEXICON_PATH).read_text(encoding="utf-8"))
    patterns = data["patterns"]
    roots = []
    labels = {}
    for category, block in data["categories"].items():
        labels[category] = block.get("label", category)
        for name, spec in (block.get("roots") or {}).items():
            roots.append(_expand(str(name), spec or {}, category, patterns))
    return Lexicon(tuple(roots), labels)


LEXICON = load_lexicon()
