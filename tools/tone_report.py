"""Per-clip tone report: emotion, shout score, laughter, angry_tone, severity.

    venv/bin/python tools/tone_report.py ~/echosense-tone-clips
    venv/bin/python tools/tone_report.py ~/echosense-tone-clips --stt     # ElevenLabs text+tags
    venv/bin/python tools/tone_report.py ~/asr-bench/clips --stt --logprobs

Text: with --stt each clip is transcribed by ElevenLabs exactly as on the box
(scribe, ceb, keyterms, audio tags, word logprobs). Without it the text comes
from manifest.csv / transcripts.csv (columns file + text/reference) with any
"haha" removed, so laughter is judged from the audio only.

The speech-loudness baseline is the median speech RMS of the folder's "calm"
clips (all clips when none are named calm_*), like a classroom's usual voice.
Thresholds come from ECHOSENSE_TONE_* (detection/tone_vote.py).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import re
import sys
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402,F401  (loads edge.env)
import numpy as np  # noqa: E402

from detection.false_alarm_filters import (  # noqa: E402
    FilterConfig,
    filter_terms_by_elevenlabs_logprob,
)
from detection.fusion import fuse_event_evidence  # noqa: E402
from detection.text_detector import TextDetector  # noqa: E402
from detection.tone_vote import ToneThresholds, vote  # noqa: E402
from model.emotion_recognizer import (  # noqa: E402
    EmotionRecognizer,
    speech_rms,
    yamnet_class_maxima,
)

_HAHA = re.compile(r"\b(?:ha){2,}h?\b[!.,]*", re.IGNORECASE)


def read_wav(path):
    with wave.open(str(path), "rb") as handle:
        if handle.getframerate() != 16000 or handle.getnchannels() != 1:
            raise SystemExit(f"{path.name}: needs 16 kHz mono")
        pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def manifest_texts(directory):
    texts = {}
    for name in ("manifest.csv", "transcripts.csv"):
        path = directory / name
        if not path.is_file():
            continue
        with open(path, encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                text = row.get("text") or row.get("reference") or ""
                texts[row.get("file", "")] = text
    return texts


def elevenlabs():
    from model.stt_provider import ElevenLabsClient, STTSettings, load_keyterms

    settings = STTSettings.from_environment()
    if not settings.api_key:
        raise SystemExit("--stt needs ELEVENLABS_API_KEY in edge.env")
    return ElevenLabsClient(settings, load_keyterms())


def analyze(paths, *, stt=False, show_logprobs=False, recognizer=None, yamnet=None):
    recognizer = recognizer or EmotionRecognizer()
    if yamnet is None:
        from model.yamnet_infer import load_yamnet_bundle

        bundle = load_yamnet_bundle()
        yamnet = (bundle.interpreter, bundle.class_names) if bundle.available else None
    client = elevenlabs() if stt else None
    from model.stt_provider import split_audio_tags

    audio = {path: read_wav(path) for path in paths}
    rms = {path: speech_rms(samples) for path, samples in audio.items()}
    calm = [rms[p] for p in paths if p.name.startswith("calm")] or list(rms.values())
    baseline = float(np.median(calm))
    texts = {}
    for directory in {p.parent for p in paths}:
        texts.update({directory / k: v for k, v in manifest_texts(directory).items()})
    thresholds = ToneThresholds.from_environment()
    filters = FilterConfig.from_environment()
    rows = []
    for path in paths:
        samples = audio[path]
        emotion = recognizer.analyze(samples)
        maxima = yamnet_class_maxima(samples, *yamnet) if yamnet else None
        signals = {
            "emotion": emotion,
            "yamnet": maxima,
            "rms": {"utterance_rms": rms[path], "baseline_rms": baseline,
                    "ratio": rms[path] / baseline if baseline > 0 else None,
                    "baseline_count": len(calm)},
        }
        tags, words, text = (), [], ""
        if client is not None:
            raw, _language, words = client.transcribe_detailed(samples)
            text, tags = split_audio_tags(raw)
        else:
            text = " ".join(_HAHA.sub(" ", texts.get(path, "")).split())
        tone = vote(signals, tags, thresholds=thresholds)
        evidence = TextDetector().analyze_transcript(text, language="ceb")
        dropped, scores = [], {}
        if words:
            _kept, dropped, scores = filter_terms_by_elevenlabs_logprob(
                evidence.decision_terms, words, filters,
            )
        severity, reason = "none", "No monitored word"
        if evidence.qualifies_to_open:
            evidence = replace(evidence, tone_signals=signals, audio_tags=tuple(tags))
            fused = fuse_event_evidence((evidence,), (), duration_seconds=1.0)
            severity = fused.level if fused.deliver else f"not sent ({fused.level})"
            reason = fused.summary
            if dropped and set(dropped) >= set(evidence.decision_terms):
                severity, reason = "dropped", "low ElevenLabs word confidence"
        rows.append({
            "clip": path.name,
            "text": text,
            "tags": list(tags),
            "emotion": emotion["label"],
            "emotion_score": emotion["score"],
            "angry": emotion.get("scores", {}).get("angry", 0.0),
            "emotion_ms": emotion["ms"],
            "shout": (maxima or {}).get("shout_score", 0.0),
            "shout_label": (maxima or {}).get("shout_label"),
            "laugh": (maxima or {}).get("laughter_score", 0.0),
            "rms_ratio": signals["rms"]["ratio"],
            "angry_tone": tone.angry_tone,
            "laughter": tone.laughter,
            "terms": list(evidence.decision_terms),
            "term_probability": {k: v for k, v in scores.items() if v is not None},
            "dropped": dropped,
            "severity": severity,
            "reason": reason,
        })
        if show_logprobs and words:
            print(f"  {path.name}: " + " ".join(
                f"{w['text']}:{w['logprob']:.2f}" if w.get("logprob") is not None else w["text"]
                for w in words
            ))
    return rows, baseline, thresholds


def print_table(rows, baseline, thresholds):
    print(f"\nbaseline speech RMS={baseline:.4f}  thresholds: angry>={thresholds.angry} "
          f"shout>={thresholds.shout} rms_ratio>={thresholds.rms_ratio} laugh>={thresholds.laugh}\n")
    header = ("clip", "emotion", "angry", "shout", "laugh", "rms×", "angry_tone",
              "laughter", "severity", "reason")
    print(" | ".join(header))
    print(" | ".join("---" for _ in header))
    for r in rows:
        ratio = "n/a" if r["rms_ratio"] is None else f"{r['rms_ratio']:.2f}"
        print(" | ".join((
            r["clip"], f"{r['emotion']} {r['emotion_score']:.2f}", f"{r['angry']:.2f}",
            f"{r['shout']:.2f}", f"{r['laugh']:.2f}", ratio,
            "YES" if r["angry_tone"] else "no", "YES" if r["laughter"] else "no",
            r["severity"], r["reason"],
        )))
    angry = sum(r["angry_tone"] for r in rows)
    print(f"\nangry_tone: {angry}/{len(rows)}   laughter: {sum(r['laughter'] for r in rows)}/{len(rows)}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path, help="folder(s) or .wav files")
    parser.add_argument("--stt", action="store_true", help="transcribe with ElevenLabs")
    parser.add_argument("--logprobs", action="store_true", help="print ElevenLabs word logprobs")
    parser.add_argument("--json", type=Path, help="also write rows as JSON")
    args = parser.parse_args(argv)
    paths = []
    for item in args.paths:
        item = item.expanduser()
        paths.extend(sorted(item.glob("*.wav")) if item.is_dir() else [item])
    if not paths:
        raise SystemExit("no .wav clips found")
    rows, baseline, thresholds = analyze(paths, stt=args.stt, show_logprobs=args.logprobs)
    print_table(rows, baseline, thresholds)
    if args.json:
        args.json.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
