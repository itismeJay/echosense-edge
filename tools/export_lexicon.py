"""Export the Bisaya lexicon as a review sheet for a teacher.

    venv/bin/python tools/export_lexicon.py

Writes lexicon/bisaya_bullying_review.csv (opens in Excel or Google Sheets):
one row per phrase with its category, meaning, severity and how it was made,
plus empty "Teacher: OK?" and "Teacher: notes" columns. Everyday sentences
that must NOT alert are listed at the end so the teacher can add more.
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detection.lexicon import LEXICON, LEXICON_PATH  # noqa: E402

SEVERITY = {
    "insult": "MEDIUM when aimed at someone",
    "context": "MEDIUM when aimed at someone",
    "swear": "LOW (bare swear word)",
    "threat": "HIGH (always)",
    "self_harm": "HIGH (always)",
}
OUT = ROOT / "lexicon" / "bisaya_bullying_review.csv"


def main():
    import yaml

    data = yaml.safe_load(LEXICON_PATH.read_text(encoding="utf-8"))
    record_next = set((data.get("record_next") or {}).get("bullying") or ())
    with OUT.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "Category", "Root word", "Meaning", "Phrase", "Severity",
            "How it was made", "Record next", "Teacher: OK? (Y/N)", "Teacher: notes",
        ])
        for root in LEXICON.roots:
            for phrase in root.phrases:
                made = (
                    "hand-written" if phrase.source == "extra"
                    else "pattern " + phrase.source.split(":", 1)[1]
                )
                writer.writerow([
                    LEXICON.labels[root.category], root.name, root.meaning,
                    phrase.text, SEVERITY[phrase.role], made,
                    "yes" if phrase.text in record_next else "", "", "",
                ])
        writer.writerow([])
        writer.writerow(["MUST NOT ALERT (everyday sentences using the same words)"])
        for root in LEXICON.roots:
            for text in root.everyday:
                writer.writerow([LEXICON.labels[root.category], root.name, "", text,
                                 "no alert", "everyday check", "", "", ""])
    total = len(LEXICON.phrases)
    print(f"Wrote {OUT.relative_to(ROOT)}: {total} phrases, "
          f"{sum(len(r.everyday) for r in LEXICON.roots)} everyday sentences")


if __name__ == "__main__":
    main()
