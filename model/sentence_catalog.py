"""Validated multilingual sentence catalog used by recording and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURES = (
    PROJECT_ROOT / "tests" / "fixtures" / "cebuano_classroom_original_200.jsonl",
    PROJECT_ROOT / "tests" / "fixtures" / "tagalog_classroom_250.jsonl",
    PROJECT_ROOT / "tests" / "fixtures" / "english_classroom_250.jsonl",
)

_LANGUAGE_CODES = {
    "ceb": "ceb",
    "cebuano": "ceb",
    "bisaya": "ceb",
    "en": "en",
    "english": "en",
    "fil": "tl",
    "filipino": "tl",
    "tagalog": "tl",
    "tl": "tl",
}


@dataclass(frozen=True, slots=True)
class SentenceCase:
    sentence_id: str
    text: str
    language: str
    expected_alert: bool | None
    category: str
    source_category: str
    source_path: Path


def _language_code(value: object) -> str:
    code = _LANGUAGE_CODES.get(str(value or "").strip().casefold())
    if code is None:
        raise ValueError(f"unsupported sentence language: {value!r}")
    return code


def load_sentence_cases(paths=DEFAULT_FIXTURES) -> tuple[SentenceCase, ...]:
    """Load one validated case per fixture utterance in deterministic order."""

    cases = []
    seen_ids = set()
    for raw_path in paths:
        path = Path(raw_path).resolve()
        with path.open(encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, 1):
                if not raw_line.strip():
                    continue
                try:
                    item = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid sentence fixture {path.name}:{line_number}"
                    ) from exc
                language = _language_code(item.get("language"))
                utterances = item.get("utterances")
                if not isinstance(utterances, list) or not utterances:
                    raise ValueError(
                        f"sentence fixture has no utterance {path.name}:{line_number}"
                    )
                raw_id = str(item.get("id") or line_number).strip()
                for utterance_index, utterance in enumerate(utterances, 1):
                    text = str(utterance or "").strip()
                    if not text:
                        raise ValueError(
                            f"sentence fixture has empty text {path.name}:{line_number}"
                        )
                    numeric_id = raw_id.zfill(4) if raw_id.isdigit() else raw_id
                    suffix = "" if len(utterances) == 1 else f"-{utterance_index:02d}"
                    sentence_id = f"{language}-{numeric_id}{suffix}"
                    if sentence_id in seen_ids:
                        raise ValueError(f"duplicate sentence id: {sentence_id}")
                    seen_ids.add(sentence_id)
                    expected = item.get("expected_alert")
                    if expected not in {True, False, None}:
                        raise ValueError(
                            f"invalid expected_alert for {sentence_id}"
                        )
                    cases.append(SentenceCase(
                        sentence_id=sentence_id,
                        text=text,
                        language=language,
                        expected_alert=expected,
                        category=str(item.get("category") or "unknown"),
                        source_category=str(
                            item.get("source_category") or "unknown"
                        ),
                        source_path=path,
                    ))
    return tuple(cases)
