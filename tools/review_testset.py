"""Interactively review local EchoSense test clips without changing audio.

    venv/bin/python tools/review_testset.py --dir ~/echosense-testset
    venv/bin/python tools/review_testset.py --dir ~/echosense-testset --summary

The newest evaluator JSON under ``results/`` supplies Whisper transcripts.
Decisions are stored separately in ``review_decisions.json`` beside the clips.
"""

from __future__ import annotations

import argparse
from array import array
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import subprocess
import sys
import wave


DEFAULT_DIR = Path.home() / "echosense-testset"
DECISIONS = {"v": "VALID", "r": "RERECORD", "u": "UNCERTAIN"}


def normalize(text):
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def comparison(expected, transcript):
    expected = normalize(expected)
    transcript = normalize(transcript)
    if not transcript:
        return 0.0, "NO_TRANSCRIPT"
    score = SequenceMatcher(None, expected, transcript).ratio()
    return score, "CHECK_BY_EAR" if score < 0.55 else "similar"


def audit_wav(path):
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.getnframes()
        raw = handle.readframes(frames)
    samples = array("h")
    if width == 2:
        samples.frombytes(raw)
        if sys.byteorder != "little":
            samples.byteswap()
    duration = frames / rate if rate else 0.0
    rms = (
        (sum(sample * sample for sample in samples) / len(samples)) ** 0.5
        if samples else 0.0
    )
    peak = max((abs(sample) for sample in samples), default=0)
    clipped = sum(abs(sample) >= 32767 for sample in samples)
    flags = []
    if (channels, width, rate) != (1, 2, 16000):
        flags.append(f"FORMAT_{channels}ch_{width * 8}bit_{rate}Hz")
    if duration < 0.5:
        flags.append("TOO_SHORT")
    if rms < 20:
        flags.append("SILENCE")
    elif rms < 150:
        flags.append("VERY_QUIET")
    if clipped:
        flags.append(f"CLIPPED_{clipped}")
    return {
        "duration_seconds": round(duration, 3),
        "rms": round(rms, 1),
        "peak": peak,
        "clipped_samples": clipped,
        "flags": flags,
    }


def load_evaluation(directory, result_path=None):
    if result_path is None:
        candidates = sorted(
            (directory / "results").glob("*.json"),
            key=lambda path: path.stat().st_mtime,
        )
        if not candidates:
            raise SystemExit("No evaluator JSON found; run tools/evaluate_testset.py first")
        result_path = candidates[-1]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    return {
        f"{row['id']}__{row['distance']}.wav": row for row in result["rows"]
    }, result_path


def transcript(row):
    return " | ".join(
        utterance.get("transcript", "") for utterance in row.get("utterances", [])
        if utterance.get("transcript", "")
    )


def load_decisions(path):
    if not path.exists():
        return {"decisions": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("decisions", {})
    return data


def save_decisions(path, data):
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def print_summary(rows, decisions):
    rerecord = sorted(
        name for name, item in decisions.items() if item["decision"] == "RERECORD"
    )
    uncertain = sorted(
        name for name, item in decisions.items() if item["decision"] == "UNCERTAIN"
    )
    distance = sorted(
        name for name, item in decisions.items() if item.get("distance_label_suspect")
    )
    diagnostic = sorted(
        name for name, item in decisions.items()
        if item["decision"] == "VALID"
        and rows.get(name, {}).get("label") == "bullying"
        and not rows.get(name, {}).get("alert")
    )
    print(f"Reviewed: {len(decisions)}/{len(rows)}")
    for label, names in (
        ("RERECORD", rerecord),
        ("UNCERTAIN", uncertain),
        ("DISTANCE_LABEL_SUSPECT", distance),
        ("VALID_DETECTION_FAILURE_DIAGNOSTIC", diagnostic),
    ):
        print(f"\n{label} ({len(names)}):")
        print("\n".join(names) if names else "(none)")


def review(args):
    manifest = json.loads((args.dir / "phrases.json").read_text(encoding="utf-8"))
    phrases = {item["id"]: item for item in manifest["phrases"]}
    rows, result_path = load_evaluation(args.dir, args.results)
    state = load_decisions(args.decisions)
    state["dataset"] = str(args.dir)
    state["evaluation"] = str(result_path)
    clips = sorted(args.dir.glob("*__*.wav"))
    if args.summary:
        print_summary(rows, state["decisions"])
        return
    print(f"Clips: {len(clips)}  Evaluation: {result_path}")
    print("v/r/u = VALID/RERECORD/UNCERTAIN; append d if distance label is suspect")
    print("p = replay, s = skip, q = save and quit. Existing audio is never changed.\n")
    for number, path in enumerate(clips, 1):
        if path.name in state["decisions"] and not args.all:
            continue
        pid, _, distance = path.stem.rpartition("__")
        phrase = phrases.get(pid)
        row = rows.get(path.name, {})
        heard = transcript(row)
        score, comparison_flag = comparison(phrase["text"], heard)
        audio = audit_wav(path)
        print(f"[{number}/{len(clips)}] {path.name}")
        print(f"Expected:   {phrase['text']}  [{phrase['label']}/{phrase['language']}; {distance}]")
        print(f"Transcript: {heard or '(no speech)'}")
        print(
            f"Compare:    {score:.0%} {comparison_flag}; "
            f"audio={audio['duration_seconds']:.3f}s rms={audio['rms']:.1f} "
            f"peak={audio['peak']} flags={','.join(audio['flags']) or 'none'}"
        )
        while True:
            subprocess.run(["afplay", str(path)], check=False)
            answer = input("Decision [v/r/u, optional d; p/s/q]: ").strip().lower()
            if answer == "p":
                continue
            if answer == "q":
                save_decisions(args.decisions, state)
                print_summary(rows, state["decisions"])
                return
            if answer == "s":
                break
            decision = DECISIONS.get(answer[:1])
            if decision and set(answer[1:]) <= {"d"}:
                note = input("Note (optional): ").strip()
                state["decisions"][path.name] = {
                    "decision": decision,
                    "distance_label_suspect": "d" in answer[1:],
                    "note": note,
                    "expected": phrase["text"],
                    "transcript": heard,
                    "comparison_score": round(score, 3),
                    "technical": audio,
                }
                save_decisions(args.decisions, state)
                print()
                break
            print("Enter v, r, u, vd, rd, ud, p, s, or q.")
    print_summary(rows, state["decisions"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--all", action="store_true", help="revisit already reviewed clips")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args(argv)
    if args.decisions is None:
        args.decisions = args.dir / "review_decisions.json"
    review(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
