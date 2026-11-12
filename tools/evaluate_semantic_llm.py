#!/usr/bin/env python3
"""Before/after report for the Claude semantic classifier (text only, no audio).

Sets: held-out split (tests/fixtures/classroom_split_seed42.json), the 30
benchmark clips' ElevenLabs transcripts (--asr-results, optional) and
tests/fixtures/banter_40.jsonl. "Before" is lexicon-only (mode off); "after"
uses ECHOSENSE_SEMANTIC_MODE (default all) and needs ANTHROPIC_API_KEY.

  venv/bin/python tools/evaluate_semantic_llm.py --asr-results <asr-bench>/results.json
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import csv
from datetime import datetime, timedelta, timezone
import io
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402,F401  (loads edge.env)
from detection.fusion import fuse_event_evidence  # noqa: E402
from detection.llm_semantic import SYSTEM_PROMPT, SemanticGate  # noqa: E402
from detection.text_detector import TextDetector  # noqa: E402

FIXTURES = PROJECT_ROOT / "tests" / "fixtures"
SPLIT_FILES = {
    "ceb": "cebuano_classroom_original_200.jsonl",
    "eng": "english_classroom_250.jsonl",
    "tgl": "tagalog_classroom_250.jsonl",
}
ASR_KEY = "ElevenLabs scribe_v3 [ceb +wordkeyterms]"
WALL = datetime(2026, 11, 13, tzinfo=timezone.utc)
# Haiku 5.5 list price, USD per million tokens (prompts under 100K tokens).
PRICE_IN, PRICE_OUT = 0.10, 0.50


def _jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def held_out_cases():
    split = json.loads((FIXTURES / "classroom_split_seed42.json").read_text())
    rows = {
        f"{language}:{row['id']}": (language, row)
        for language, name in SPLIT_FILES.items()
        for row in _jsonl(FIXTURES / name)
    }
    cases = []
    for key in split["held_out"]:
        language, row = rows[key]
        cases.append({
            "id": key, "language": language,
            "expected": bool(row["expected_alert"]),
            "label": row.get("provisional_label"),
            "utterances": row["utterances"],
        })
    return cases


def asr_cases(results_path):
    results_path = Path(results_path)
    categories = {}
    with open(results_path.parent / "transcripts.csv", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            categories[row["file"]] = row["category"]
    rows = json.loads(results_path.read_text())[ASR_KEY]
    return [{
        "id": name, "language": language,
        "expected": categories[name] == "bullying",
        "utterances": [hypothesis],
    } for name, language, _reference, hypothesis, _seconds in rows]


def banter_cases():
    return [{
        "id": f"banter:{row['id']}", "language": row["language"],
        "expected": bool(row["expected_alert"]), "kind": row["kind"],
        "utterances": row["utterances"],
    } for row in _jsonl(FIXTURES / "banter_40.jsonl")]


def run_case(case, gate):
    detector = TextDetector(semantic_gate=gate)
    evidence = []
    for index, text in enumerate(case["utterances"]):
        at = index * 3.0
        evidence.append(detector.analyze_transcript(
            text, language=case["language"],
            observed_at=WALL + timedelta(seconds=at), observed_monotonic=at,
        ))
    qualifying = tuple(item for item in evidence if item.qualifies_to_open)
    judgements = [item.llm for item in evidence if item.llm is not None]
    alert, level, reasons = False, None, ()
    if qualifying:
        decision = fuse_event_evidence(qualifying, (), duration_seconds=1.0)
        alert, level, reasons = decision.deliver, decision.level, decision.reasons
    return {
        **case, "alert": alert, "level": level, "reasons": list(reasons),
        "labels": [j.label if j.ok else j.status for j in judgements],
        "fallbacks": sum(1 for j in judgements if j.status == "fallback"),
        "calls": sum(1 for j in judgements if j.status in ("classified", "fallback")),
        "input_tokens": sum(j.input_tokens for j in judgements),
        "output_tokens": sum(j.output_tokens for j in judgements),
        "latencies_ms": [j.latency_ms for j in judgements if j.ok],
    }


def run(cases, gate, workers):
    with redirect_stdout(io.StringIO()):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(lambda case: run_case(case, gate), cases))


def metrics(rows):
    tp = sum(r["alert"] and r["expected"] for r in rows)
    fp = sum(r["alert"] and not r["expected"] for r in rows)
    fn = sum(not r["alert"] and r["expected"] for r in rows)
    fp_med = sum(
        r["alert"] and not r["expected"] and r["level"] in ("MEDIUM", "HIGH")
        for r in rows
    )
    return {
        "n": len(rows), "tp": tp, "fp": fp, "fn": fn, "fp_medium_plus": fp_med,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
    }


def _pct(value):
    return "-" if value is None else f"{value * 100:.0f}%"


def table(name, before, after):
    print(f"\n{name}")
    print("| lang | n | precision before→after | recall before→after "
          "| false alerts before→after | false ≥MEDIUM before→after |")
    print("|---|---|---|---|---|---|")
    languages = sorted({r["language"] for r in before})
    for language in languages + ["all"]:
        pick = (lambda r: True) if language == "all" else (lambda r: r["language"] == language)
        b = metrics([r for r in before if pick(r)])
        a = metrics([r for r in after if pick(r)]) if after else None
        def pair(key, fmt=str):
            return f"{fmt(b[key])}→{fmt(a[key])}" if a else fmt(b[key])
        print(f"| {language} | {b['n']} | {pair('precision', _pct)} | "
              f"{pair('recall', _pct)} | {pair('fp')} | {pair('fp_medium_plus')} |")


def cost(rows_sets):
    rows = [row for rows in rows_sets for row in rows]
    calls = sum(r["calls"] - r["fallbacks"] for r in rows)
    if calls:
        tin = sum(r["input_tokens"] for r in rows) / calls
        tout = sum(r["output_tokens"] for r in rows) / calls
        source = f"measured over {calls} calls"
    else:
        tin = len(SYSTEM_PROMPT) / 3.6 + 130
        tout = 70
        source = "estimated (no calls made)"
    latencies = sorted(ms for r in rows for ms in r.get("latencies_ms", ()))
    if latencies:
        mean = sum(latencies) / len(latencies)
        p90 = latencies[int(0.9 * (len(latencies) - 1))]
        print(f"Latency per call: mean {mean:.0f} ms, median "
              f"{latencies[len(latencies) // 2]:.0f} ms, p90 {p90:.0f} ms, "
              f"max {latencies[-1]:.0f} ms")
    per_1000 = 1000 * (tin * PRICE_IN + tout * PRICE_OUT) / 1_000_000
    print(f"\nCost: ~{tin:.0f} input + {tout:.0f} output tokens per utterance "
          f"({source}) = ${per_1000:.3f} per 1000 utterances at Haiku 5.5 list price")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--asr-results", type=Path,
                        help="asr-bench results.json (transcripts.csv beside it)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, help="write per-case rows as JSON")
    args = parser.parse_args(argv)

    sets = {"held-out split": held_out_cases(), "banter_40": banter_cases()}
    if args.asr_results:
        sets["30 clips (ElevenLabs transcripts)"] = asr_cases(args.asr_results)

    off = SemanticGate(mode="off")
    gate = SemanticGate.from_environment(log=lambda line: print(line, file=sys.stderr))
    gate._max_workers = max(gate._max_workers, args.workers)
    if not gate.available:
        print(f"[EVAL] semantic classifier unavailable "
              f"({gate.unavailable_reason or 'mode=off'}): lexicon-only (before) only",
              file=sys.stderr)
    output = {}
    for name, cases in sets.items():
        before = run(cases, off, args.workers)
        after = run(cases, gate, args.workers) if gate.available else None
        output[name] = {"before": before, "after": after}
        table(name, before, after)
        threats = [i for i, r in enumerate(before) if r.get("label") == "threat_or_intimidation"]
        if threats:
            def counts(rows):
                alerted = sum(rows[i]["alert"] for i in threats)
                high = sum(rows[i]["alert"] and rows[i]["level"] == "HIGH" for i in threats)
                return f"{alerted}/{len(threats)} alerted, {high} HIGH"
            print(f"threat recall: before {counts(before)}"
                  + (f"; after {counts(after)}" if after else ""))
        if after:
            from detection.fusion import THREAT_REVIEW_REASON
            lowered = [r["id"] for r in after if THREAT_REVIEW_REASON in r["reasons"]]
            print(f"threats lowered to MEDIUM (game/story): {len(lowered)} {lowered}")
        if after:
            fallbacks = sum(r["fallbacks"] for r in after)
            print(f"fallbacks (timeout/error): {fallbacks}")
    cost([v["after"] for v in output.values() if v["after"]])
    gate.shutdown()
    if args.out:
        args.out.write_text(json.dumps(output, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
