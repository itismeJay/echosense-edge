"""Record the EchoSense accuracy test set, phrase by phrase.

    venv/bin/python tools/record_testset.py            # 700 sentences, all passes
    venv/bin/python tools/record_testset.py --catalog legacy
    venv/bin/python tools/record_testset.py --pass far # one pass only
    venv/bin/python tools/record_testset.py --redo pangit_ka__far

Clips go to ~/echosense-testset/<phrase id>__<near|far|media>.wav (16 kHz mono
int16) with the expected labels in phrases.json. Existing clips are skipped, so
an interrupted session resumes where it stopped.

near   you press Enter, then say the phrase at a normal distance.
far    hands-free: the Mac reads the next phrase aloud, beeps, records, and
       moves on, so you can stand 2-3 m away the whole time.
media  play a YouTube/TikTok clip near the Mac; nothing should alert.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import wave

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.sentence_catalog import load_sentence_cases
from model.language_guide import normalized_words
from model.recording_metadata import (
    DELIVERIES,
    NOISE_CONDITIONS,
    RecordingDescriptor,
)

SAMPLE_RATE = 16000
PHRASE_SECONDS = 4.0
MEDIA_SECONDS = 15.0
DEFAULT_DIR = Path.home() / "echosense-testset"
DEFAULT_MIC = "MacBook Pro Microphone"
ACTIVE_CATALOG = "legacy"
RECORDING_SPEAKER_ID = "legacy"
RECORDING_DELIVERY = "normal"
RECORDING_NOISE = "quiet"


def classroom_phrases():
    """Return the 700 validated sentence cases in recorder tuple form."""

    labels = {True: "bullying", False: "normal", None: "review"}
    return tuple(
        (
            case.sentence_id,
            case.text,
            labels[case.expected_alert],
            case.language,
        )
        for case in load_sentence_cases()
    )


def recording_seconds(text):
    """Allow complete sentences more time while retaining a strict cap."""

    words = len(normalized_words(text))
    return max(PHRASE_SECONDS, min(12.0, 2.0 + 0.75 * words))

# label: "bullying" must alert, "normal" and "media" must not.
PHRASES = [
    # Bullying, Bisaya
    ("pangit_ka", "pangit ka", "bullying", "ceb"),
    ("yawa_ka", "yawa ka", "bullying", "ceb"),
    ("buang_ka", "buang ka", "bullying", "ceb"),
    ("bogo_ka", "bogo ka", "bullying", "ceb"),
    ("atay_ka", "atay ka", "bullying", "ceb"),
    ("piste_ka", "piste ka", "bullying", "ceb"),
    ("animal_ka", "animal ka", "bullying", "ceb"),
    ("baho_ka", "baho ka", "bullying", "ceb"),
    ("gaga_ka", "gaga ka", "bullying", "ceb"),
    ("patyon_tika", "patyon tika", "bullying", "ceb"),
    ("bunalan_tika", "bunalan tika", "bullying", "ceb"),
    # Bullying, Bisaya Grade 6 list (body, intelligence, poverty, exclusion,
    # swear words, threats). Said the way a student aims it.
    ("tambok_ka", "tambok ka", "bullying", "ceb"),
    ("baboy_ka", "baboy ka", "bullying", "ceb"),
    ("itom_ka", "itom ka", "bullying", "ceb"),
    ("pandak_ka", "pandak ka", "bullying", "ceb"),
    ("niwang_ka", "niwang ka", "bullying", "ceb"),
    ("bungi_ka", "bungi ka", "bullying", "ceb"),
    ("dakog_ilong", "dakog ilong", "bullying", "ceb"),
    ("dakog_tiyan_ka", "dakog tiyan ka", "bullying", "ceb"),
    ("buktot_ka", "buktot ka", "bullying", "ceb"),
    ("baho_imong_baba", "baho imong baba", "bullying", "ceb"),
    ("hugaw_ka", "hugaw ka", "bullying", "ceb"),
    ("kutohon_ka", "kutohon ka", "bullying", "ceb"),
    ("bisaya_kaayo_ka", "bisaya kaayo ka", "bullying", "ceb"),
    ("buwang", "buwang", "bullying", "ceb"),
    ("walay_utok", "walay utok", "bullying", "ceb"),
    ("hinay_ug_utok", "hinay ug utok", "bullying", "ceb"),
    ("abnoy_ka", "abnoy ka", "bullying", "ceb"),
    ("ungo_ka", "ungo ka", "bullying", "ceb"),
    ("pobre_ka", "pobre ka", "bullying", "ceb"),
    ("walay_kwarta_ka", "walay kwarta ka", "bullying", "ceb"),
    ("ilo_ka", "ilo ka", "bullying", "ceb"),
    ("anak_sa_pobre", "anak ka sa pobre", "bullying", "ceb"),
    ("layo_diri_ikaw", "palayo diri ikaw", "bullying", "ceb"),
    ("ayaw_ug_duol_nako", "ayaw ug duol nako", "bullying", "ceb"),
    ("walay_ganahan_nimo", "walay ganahan nimo", "bullying", "ceb"),
    ("dili_ka_apil", "dili ka apil", "bullying", "ceb"),
    ("walay_pulos_ka", "wala kay pulos", "bullying", "ceb"),
    ("sipsip_ka", "sipsip ka", "bullying", "ceb"),
    ("salingkapaw_ka", "salingkapaw ka", "bullying", "ceb"),
    ("tikasan_ka", "tikasan ka", "bullying", "ceb"),
    ("gago_ka", "gago ka", "bullying", "ceb"),
    ("leche_ka", "leche ka", "bullying", "ceb"),
    ("bayot_ka", "bayot ka", "bullying", "ceb"),
    ("sumbagon_tika", "sumbagon tika", "bullying", "ceb"),
    ("sipaon_tika", "sipaon tika", "bullying", "ceb"),
    ("hulata_ko_sa_gawas", "hulata ko sa gawas", "bullying", "ceb"),
    ("ayaw_pagsumbong", "ayaw pagsumbong", "bullying", "ceb"),
    ("magpakamatay_ka", "magpakamatay ka", "bullying", "ceb"),
    # Bullying, Tagalog
    ("bobo_ka", "bobo ka", "bullying", "tl"),
    ("tanga_ka", "tanga ka", "bullying", "tl"),
    ("papatayin_kita", "papatayin kita", "bullying", "tl"),
    # Bullying, English
    ("youre_so_stupid", "you're so stupid", "bullying", "en"),
    ("shut_up", "shut up", "bullying", "en"),
    # Normal, Bisaya (some sound close to monitored words on purpose)
    ("maayong_buntag", "maayong buntag class", "normal", "ceb"),
    ("hulam_lapis", "hulam ko sa imong lapis", "normal", "ceb"),
    ("human_na_mo", "human na mo?", "normal", "ceb"),
    ("adto_canteen", "adto ta sa canteen", "normal", "ceb"),
    ("pangitaa_libro", "pangitaa ang imong libro", "normal", "ceb"),
    ("pangit_panahon", "pangit kaayo ang panahon karon", "normal", "ceb"),
    ("ayaw_kalimti", "ayaw kalimti ang assignment", "normal", "ceb"),
    # Normal, Bisaya: everyday sentences with the Grade 6 words
    ("itom_nga_lapis", "itom nga lapis", "normal", "ceb"),
    ("tambok_nga_manok", "tambok nga manok", "normal", "ceb"),
    ("adobong_baboy", "adobong baboy", "normal", "ceb"),
    ("pandak_man_ang_lamesa", "pandak man ang lamesa", "normal", "ceb"),
    ("hugaw_ang_salog", "hugaw ang salog", "normal", "ceb"),
    ("niwang_ang_iro", "niwang ang iro", "normal", "ceb"),
    ("walay_kwarta_ko_karon", "walay kwarta ko karon", "normal", "ceb"),
    ("ayaw_ug_duol_sa_kalayo", "ayaw ug duol sa kalayo", "normal", "ceb"),
    ("layo_diri_ang_balay", "layo diri ang balay", "normal", "ceb"),
    ("taga_asa_ka", "taga asa ka", "normal", "ceb"),
    ("ilo_ang_bata_sa_istorya", "ilo ang bata sa istorya", "normal", "ceb"),
    ("hulata_ko_sa_canteen", "hulata ko sa canteen", "normal", "ceb"),
    ("sipa_ang_bola", "sipa-a ang bola", "normal", "ceb"),
    ("dakog_tiyan_ang_iro", "dakog tiyan ang iro", "normal", "ceb"),
    ("bisaya_kaayo_ang_kanta", "bisaya kaayo ang kanta", "normal", "ceb"),
    # Normal, Tagalog
    ("magandang_umaga", "magandang umaga po", "normal", "tl"),
    ("pahiram_lapis", "pahiram ng lapis", "normal", "tl"),
    ("tapos_na_ba", "tapos na ba kayo?", "normal", "tl"),
    # Normal, English
    ("good_morning", "good morning class", "normal", "en"),
    ("borrow_pencil", "can I borrow your pencil?", "normal", "en"),
    ("open_books", "please open your books", "normal", "en"),
]

def _lexicon_record_next():
    """Phrases marked `record_next` in lexicon/bisaya_bullying.yaml."""

    import re
    import yaml

    path = Path(__file__).resolve().parents[1] / "lexicon" / "bisaya_bullying.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")).get("record_next") or {}
    known = {item[0] for item in PHRASES}
    out = []
    for label, key in (("bullying", "bullying"), ("normal", "everyday")):
        for text in data.get(key) or ():
            pid = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
            if pid not in known:
                out.append((pid, text, label, "ceb"))
                known.add(pid)
    return out


# Grade 6 lexicon phrases chosen for recording (see the YAML).
PHRASES += _lexicon_record_next()
LEGACY_PHRASES = tuple(PHRASES)

MEDIA = [
    ("media_bisaya", "a Bisaya vlog or TikTok with people talking", "media", "ceb"),
    ("media_tagalog", "a Tagalog teleserye, movie or vlog clip", "media", "tl"),
    ("media_music", "a music video with lyrics (any language)", "media", "mixed"),
    ("media_english", "an English YouTube video or TikTok with talking", "media", "en"),
]


def _beep(sound="Tink"):
    subprocess.run(
        ["afplay", f"/System/Library/Sounds/{sound}.aiff"],
        check=False, capture_output=True,
    )


def _say(text):
    subprocess.run(["say", "-r", "170", text], check=False, capture_output=True)


class Microphone:
    def __init__(self, name):
        import pyaudio

        self._pa = pyaudio.PyAudio()
        self.index = None
        for index in range(self._pa.get_device_count()):
            info = self._pa.get_device_info_by_index(index)
            if info.get("maxInputChannels", 0) > 0 and name.lower() in info["name"].lower():
                self.index = index
                self.name = info["name"]
                break
        if self.index is None:
            raise SystemExit(f"Microphone not found: {name}")
        self._format = pyaudio.paInt16

    def record(self, seconds):
        stream = self._pa.open(
            format=self._format, channels=1, rate=SAMPLE_RATE, input=True,
            input_device_index=self.index, frames_per_buffer=1024,
        )
        frames = []
        total = int(SAMPLE_RATE * seconds)
        read = 0
        try:
            while read < total:
                count = min(1024, total - read)
                frames.append(stream.read(count, exception_on_overflow=False))
                read += count
        finally:
            stream.stop_stream()
            stream.close()
        return np.frombuffer(b"".join(frames), dtype=np.int16)

    def close(self):
        self._pa.terminate()


def _save(path, samples):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(samples.astype(np.int16).tobytes())


def _level(samples):
    data = samples.astype(np.float32)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    # Loudest 0.5 s, so silence around the phrase does not hide it.
    window = SAMPLE_RATE // 2
    loudest = max(
        (float(np.sqrt(np.mean(data[i:i + window] ** 2)))
         for i in range(0, max(1, data.size - window), SAMPLE_RATE // 10)),
        default=0.0,
    )
    return peak, loudest


def _level_note(samples):
    peak, loudest = _level(samples)
    note = f"peak={peak:.0f} speech_rms={loudest:.0f}"
    if peak >= 32000:
        return note + "  (clipped: speak a bit softer)"
    if loudest < 150:
        return note + "  (very quiet: check you spoke during the beep)"
    return note


def _write_manifest(directory):
    entries = [
        {"id": pid, "text": text, "label": label, "language": language}
        for pid, text, label, language in PHRASES + MEDIA
    ]
    (directory / "phrases.json").write_text(
        json.dumps({
            "sample_rate": SAMPLE_RATE,
            "catalog": ACTIVE_CATALOG,
            "recording_session": {
                "speaker_id": RECORDING_SPEAKER_ID,
                "delivery": RECORDING_DELIVERY,
                "noise": RECORDING_NOISE,
            },
            "phrases": entries,
        }, indent=2,
                   ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _clip_path(directory, pid, distance):
    if ACTIVE_CATALOG == "legacy":
        return directory / f"{pid}__{distance}.wav"
    descriptor = RecordingDescriptor(
        sentence_id=pid,
        speaker_id=RECORDING_SPEAKER_ID,
        delivery=RECORDING_DELIVERY,
        noise=RECORDING_NOISE,
        distance=distance,
    )
    return directory / descriptor.filename()


def near_pass(mic, directory, redo):
    print("\n=== NEAR: normal distance from the MacBook ===")
    print("Say each phrase in a normal classroom voice, the way a student would.")
    todo = [p for p in PHRASES if redo or not _clip_path(directory, p[0], "near").exists()]
    for number, (pid, text, _label, language) in enumerate(todo, 1):
        path = _clip_path(directory, pid, "near")
        seconds = recording_seconds(text)
        while True:
            input(f"\n[{number}/{len(todo)}] ({language})  \"{text}\"\n"
                  f"  Press Enter, then say it after the beep ({seconds:g} s)... ")
            _beep("Tink")
            samples = mic.record(seconds)
            _beep("Pop")
            print("  " + _level_note(samples))
            answer = input("  Enter = keep, r = redo, s = skip: ").strip().lower()
            if answer == "r":
                continue
            if answer != "s":
                _save(path, samples)
            break


def far_pass(mic, directory, redo):
    print("\n=== FAR: stand about 2-3 m from the MacBook ===")
    print("Hands-free. The Mac reads each phrase aloud, then beeps: say the phrase")
    print("after the beep in a normal voice. Ctrl-C stops; rerun to resume.")
    todo = [p for p in PHRASES if redo or not _clip_path(directory, p[0], "far").exists()]
    if not todo:
        return
    input("Press Enter, then walk 2-3 m away. The first phrase starts in 10 s... ")
    for remaining in range(10, 0, -1):
        print(f"  {remaining}", end="\r", flush=True)
        time.sleep(1)
    quiet = []
    for number, (pid, text, _label, language) in enumerate(todo, 1):
        print(f"[{number}/{len(todo)}] ({language})  \"{text}\"")
        _say(f"Next: {text}")
        time.sleep(0.4)
        _beep("Tink")
        samples = mic.record(recording_seconds(text))
        _beep("Pop")
        path = _clip_path(directory, pid, "far")
        _save(path, samples)
        note = _level_note(samples)
        print("  " + note)
        if "very quiet" in note:
            quiet.append(path.stem)
        time.sleep(1.5)
    _say("Done")
    if quiet:
        print("\nVery quiet far clips (re-record with --redo <id>__far):")
        for clip_id in quiet:
            print(f"  {clip_id}")


def media_pass(mic, directory, redo):
    print("\n=== MEDIA: YouTube/TikTok near the Mac (nothing should alert) ===")
    todo = [p for p in MEDIA if redo or not _clip_path(directory, p[0], "media").exists()]
    for number, (pid, description, _label, _language) in enumerate(todo, 1):
        path = _clip_path(directory, pid, "media")
        while True:
            input(f"\n[{number}/{len(todo)}] Play {description} on a phone or another\n"
                  "  device about 1 m from the Mac, at normal volume.\n"
                  "  Press Enter once it is playing (records 15 s)... ")
            _beep("Tink")
            samples = mic.record(MEDIA_SECONDS)
            _beep("Pop")
            print("  " + _level_note(samples))
            answer = input("  Enter = keep, r = redo, s = skip: ").strip().lower()
            if answer == "r":
                continue
            if answer != "s":
                _save(path, samples)
            break


def redo_one(mic, directory, name):
    try:
        descriptor = RecordingDescriptor.from_filename(name)
    except ValueError as exc:
        raise SystemExit(f"Unknown clip: {name}") from exc
    pid = descriptor.sentence_id
    distance = descriptor.distance
    lookup = {p[0]: p for p in PHRASES + MEDIA}
    if pid not in lookup or distance not in {"near", "far", "media"}:
        raise SystemExit(f"Unknown clip: {name}")
    _pid, text, _label, _language = lookup[pid]
    seconds = MEDIA_SECONDS if distance == "media" else recording_seconds(text)
    input(f"\"{text}\" ({distance}). Press Enter, speak after the beep... ")
    if distance == "far":
        print("  Walk 2-3 m away: recording starts in 8 s.")
        time.sleep(8)
        _say(f"Next: {text}")
        time.sleep(0.4)
    _beep("Tink")
    samples = mic.record(seconds)
    _beep("Pop")
    target = directory / f"{Path(name).stem}.wav"
    _save(target, samples)
    print("  " + _level_note(samples))


def main(argv=None):
    global ACTIVE_CATALOG
    global PHRASES
    global RECORDING_DELIVERY
    global RECORDING_NOISE
    global RECORDING_SPEAKER_ID
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path)
    parser.add_argument("--mic", default=DEFAULT_MIC)
    parser.add_argument(
        "--catalog",
        choices=("classroom700", "legacy"),
        default="classroom700",
        help="sentence catalog to record (default: classroom700)",
    )
    parser.add_argument(
        "--speaker-id",
        default="speaker-01",
        help="anonymous filename-safe speaker identifier",
    )
    parser.add_argument("--delivery", choices=DELIVERIES, default="normal")
    parser.add_argument("--noise", choices=NOISE_CONDITIONS, default="quiet")
    parser.add_argument("--pass", dest="which", choices=["all", "near", "far", "media"],
                        default="all")
    parser.add_argument("--redo", metavar="ID__DISTANCE",
                        help="re-record one clip, e.g. pangit_ka__far")
    parser.add_argument("--rerecord", action="store_true",
                        help="re-record every clip of the chosen pass")
    args = parser.parse_args(argv)

    ACTIVE_CATALOG = args.catalog
    RECORDING_SPEAKER_ID = (
        "legacy" if args.catalog == "legacy" else args.speaker_id
    )
    RECORDING_DELIVERY = args.delivery
    RECORDING_NOISE = args.noise
    if args.dir is None:
        args.dir = (
            DEFAULT_DIR
            if args.catalog == "legacy"
            else DEFAULT_DIR / "classroom700"
        )
    PHRASES = list(
        classroom_phrases()
        if args.catalog == "classroom700"
        else LEGACY_PHRASES
    )
    if args.catalog == "classroom700":
        RecordingDescriptor(
            sentence_id=PHRASES[0][0],
            speaker_id=RECORDING_SPEAKER_ID,
            delivery=RECORDING_DELIVERY,
            noise=RECORDING_NOISE,
            distance="near",
        )

    args.dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(args.dir)
    mic = Microphone(args.mic)
    print(f"Microphone: {mic.name}\nSaving to:  {args.dir}")
    try:
        if args.redo:
            redo_one(mic, args.dir, args.redo)
            return 0
        if args.which in ("all", "near"):
            near_pass(mic, args.dir, args.rerecord)
        if args.which in ("all", "media"):
            media_pass(mic, args.dir, args.rerecord)
        if args.which in ("all", "far"):
            far_pass(mic, args.dir, args.rerecord)
    except KeyboardInterrupt:
        print("\nStopped. Rerun the same command to resume.")
        return 1
    finally:
        mic.close()
    clips = sorted(args.dir.glob("*.wav"))
    print(f"\n{len(clips)} clips in {args.dir}")
    (args.dir / "RECORDING_DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S\n"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
