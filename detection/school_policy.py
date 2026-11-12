"""The school's detection policy, set from the backend's /devices/me/config.

Kept dependency-free so the detection modules (fusion, context rules, sound-alike
matching, the word lists) can read it without import cycles. model/school_config.py
owns fetching, caching and applying it.

- Sensitivity "normal" keeps the default rules.
- Sensitivity "strict": soft descriptive words alert even when not aimed at anyone,
  and sound-alike matching allows one more differing sound on longer words.
- Turned-off words: global words this school switched off. They never alert, even
  when the edge's built-in word lists contain them.
"""

from __future__ import annotations

import threading

NORMAL = "normal"
STRICT = "strict"
SENSITIVITY_LEVELS = frozenset({NORMAL, STRICT})

_lock = threading.Lock()
_sensitivity = NORMAL
_disabled_terms: frozenset[str] = frozenset()


def set_policy(sensitivity: str = NORMAL, disabled_terms=()) -> None:
    """Replace the whole policy at once. Unknown sensitivity values fall back to normal."""

    global _sensitivity, _disabled_terms
    value = str(sensitivity or "").strip().lower()
    with _lock:
        _sensitivity = value if value in SENSITIVITY_LEVELS else NORMAL
        _disabled_terms = frozenset(term for term in disabled_terms if term)


def reset_policy() -> None:
    set_policy(NORMAL, ())


def sensitivity() -> str:
    with _lock:
        return _sensitivity


def is_strict() -> bool:
    return sensitivity() == STRICT


def disabled_terms() -> frozenset[str]:
    with _lock:
        return _disabled_terms
