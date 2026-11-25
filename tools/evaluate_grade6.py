"""Run the Grade 6 Bisaya sentence set through the real TextDetector.

    venv/bin/python tools/evaluate_grade6.py [--csv PATH] [--json OUT] [--show N]

Text only: no Scribe, no audio, Claude semantic layer off (lexicon rules
only). Each sentence goes through TextDetector.analyze_transcript and the
fusion policy, the same path tests/test_lexicon.py uses. Reports the share
that alerts, misses grouped by pattern, and severity mismatches against the
CSV's own severity column (which is a guide, not ground truth).
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import contextlib
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_CSV = ROOT / "lexicon" / "grade6" / "grade6_bisaya_bullying.csv"
LEVEL = {"low": "LOW", "med": "MEDIUM", "high": "HIGH"}

# Pattern groups asked about by name; everything else is grouped by category.
GROUPS = (
    ("Ka-{w} nimo", lambda p: p.startswith("Ka-{w} nimo")),
    ("Ka{w} nimo", lambda p: p.startswith("Ka{w} nimo")),
    ("Pagka-{w} / Pagka{w}", lambda p: p.startswith("Pagka")),
    ("{a} nimo uy/oy", lambda p: p.startswith("{a} nimo")),
    ("Kining {a}", lambda p: p.startswith("Kining {a}")),
    ("ta ka", lambda p: "ta ka" in p),
    ("fixed phrases", lambda p: p == "fixed"),
)


def group_of(row):
    for name, test in GROUPS:
        if test(row["pattern"]):
            return name
    return f"{row['category']} (other patterns)"


def judge(texts):
    from detection.fusion import fuse_event_evidence
    from detection.text_detector import TextDetector

    detector = TextDetector()
    when = datetime(2026, 11, 25, tzinfo=timezone.utc)
    out = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for i, text in enumerate(texts):
            detector.clear_context()
            evidence = detector.analyze_transcript(
                text, observed_at=when, observed_monotonic=float(i * 100),
            )
            if not evidence.qualifies_to_open:
                out[text] = None
                continue
            decision = fuse_event_evidence((evidence,), (), duration_seconds=1.0)
            out[text] = decision.level if decision.deliver else None
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--json", type=Path, help="write per-sentence results here")
    parser.add_argument("--show", type=int, default=8, help="examples per group")
    args = parser.parse_args(argv)

    rows = list(csv.DictReader(args.csv.open(encoding="utf-8")))
    levels = judge([r["bisaya"] for r in rows])
    alerted = [r for r in rows if levels[r["bisaya"]]]
    print(f"{len(alerted)}/{len(rows)} alert ({100 * len(alerted) / len(rows):.1f}%)\n")

    by_group = defaultdict(list)
    for r in rows:
        by_group[group_of(r)].append(r)
    print("Coverage by pattern group:")
    for name, members in sorted(by_group.items()):
        hit = sum(1 for r in members if levels[r["bisaya"]])
        print(f"  {name:38s} {hit:4d}/{len(members):<4d} ({100 * hit / len(members):5.1f}%)")

    print("\nMisses by pattern group:")
    for name, members in sorted(by_group.items()):
        missed = [r for r in members if not levels[r["bisaya"]]]
        if not missed:
            continue
        print(f"  {name}: {len(missed)} missed")
        for r in missed[: args.show]:
            print(f"      {r['bisaya']}")

    mismatch = [
        r for r in alerted if levels[r["bisaya"]] != LEVEL[r["severity"]]
    ]
    print(f"\nSeverity differs from the CSV: {len(mismatch)} of {len(alerted)} alerts")
    pairs = Counter((LEVEL[r["severity"]], levels[r["bisaya"]]) for r in mismatch)
    for (want, got), n in pairs.most_common():
        examples = [r["bisaya"] for r in mismatch
                    if (LEVEL[r["severity"]], levels[r["bisaya"]]) == (want, got)]
        print(f"  CSV {want:6s} -> got {got:6s}: {n:4d}  e.g. {' | '.join(examples[:3])}")

    if args.json:
        args.json.write_text(json.dumps(
            [dict(r, got=levels[r["bisaya"]], group=group_of(r)) for r in rows],
            ensure_ascii=False, indent=1,
        ), encoding="utf-8")


if __name__ == "__main__":
    main()
