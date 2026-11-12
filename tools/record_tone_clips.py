"""Record the 18 tone test clips: the same 6 Bisaya insults said 3 ways.

    venv/bin/python tools/record_tone_clips.py                 # all 18
    venv/bin/python tools/record_tone_clips.py --mic "EMEET"   # EMEET mic
    venv/bin/python tools/record_tone_clips.py --only angry    # one set
    venv/bin/python tools/record_tone_clips.py --redo laugh_03_animal_ka

Sets (6 clips each, same words so only the delivery differs):
    calm   normal speaking voice, as if chatting          -> MEDIUM, no angry tone
    angry  shout it angrily at someone                    -> HIGH (angry tone)
    laugh  say it while laughing at someone (mocking)     -> MEDIUM "mocking with laughter"

Clips go to ~/echosense-tone-clips/<set>_<nn>_<words>.wav (16 kHz mono int16)
with manifest.csv (file,set,text). Then run:

    venv/bin/python tools/tone_report.py ~/echosense-tone-clips          # tone only
    venv/bin/python tools/tone_report.py ~/echosense-tone-clips --stt    # + ElevenLabs
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.record_testset import DEFAULT_MIC, Microphone, _beep, _level_note, _save  # noqa: E402

DEFAULT_DIR = Path.home() / "echosense-tone-clips"
SECONDS = 4.0
PHRASES = ("Bogo ka", "Yawa ka", "Animal ka", "Buang ka", "Pangit ka", "Baho ka")
SETS = {
    "calm": "Say it in a NORMAL, calm voice (like chatting).",
    "angry": "SHOUT it ANGRILY, as if at someone you are mad at.",
    "laugh": "Say it while LAUGHING at someone (mocking), e.g. 'Bogo ka, hahaha'.",
}


def clips():
    for set_name in SETS:
        for number, text in enumerate(PHRASES, 1):
            slug = text.lower().replace(" ", "_")
            spoken = f"{text}, hahaha" if set_name == "laugh" else text
            yield f"{set_name}_{number:02d}_{slug}", set_name, spoken


def write_manifest(directory):
    with open(directory / "manifest.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "set", "text"])
        for name, set_name, text in clips():
            writer.writerow([f"{name}.wav", set_name, text])


def record_one(mic, directory, name, set_name, text):
    while True:
        input(f"\n[{name}]  {SETS[set_name]}\n  \"{text}\"   Press Enter, speak after the beep ({SECONDS:g} s)... ")
        _beep("Tink")
        samples = mic.record(SECONDS)
        _beep("Pop")
        print("  " + _level_note(samples))
        answer = input("  Enter = keep, r = redo, s = skip: ").strip().lower()
        if answer == "r":
            continue
        if answer != "s":
            _save(directory / f"{name}.wav", samples)
        return


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--mic", default=DEFAULT_MIC)
    parser.add_argument("--only", choices=tuple(SETS))
    parser.add_argument("--redo", help="one clip name, e.g. angry_02_yawa_ka")
    parser.add_argument("--all", action="store_true", help="re-record clips that exist")
    args = parser.parse_args(argv)
    args.dir.mkdir(parents=True, exist_ok=True)
    write_manifest(args.dir)
    mic = Microphone(args.mic)
    print(f"Microphone: {mic.name}. Clips: {args.dir}")
    try:
        for name, set_name, text in clips():
            if args.redo and name != Path(args.redo).stem:
                continue
            if args.only and set_name != args.only:
                continue
            if (args.dir / f"{name}.wav").exists() and not (args.all or args.redo):
                continue
            record_one(mic, args.dir, name, set_name, text)
            time.sleep(0.2)
    finally:
        mic.close()
    print(f"\nDone. Report: venv/bin/python tools/tone_report.py {args.dir} --stt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
