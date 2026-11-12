"""Run the recorded test set offline through the real EchoSense text pipeline.

Configs are ranked by Bisaya word error rate, exact-sentence accuracy, alert
recall, then false alarms; Tagalog/English columns are informational.

    venv/bin/python tools/evaluate_testset.py --name a --model base --language auto \\
        --soundalike off --mishearings off
    venv/bin/python tools/evaluate_testset.py --report

Per clip: the same noise reducer as the live feed, Silero VAD to cut
utterances (as RealtimeSTT does), the live Whisper decode
(model/language_restricted_engine.decode_restricted), the live
post-processing in model/whisper_stt._on_text (decode gate, hallucination,
word-probability and YAMNet media filters, term matching with sound-alike
correction), TextDetector and event fusion. A clip "alerts" when any of its
utterances would be sent to the backend.

Not simulated: the acoustic tone/shout corroboration (needs a live,
calibrated room noise floor). It can only raise severity, or let an
untargeted descriptive word through when shouted.

Results go to <testset>/results/<name>.json; --report prints every result.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import statistics
import sys
import time
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.recording_metadata import RecordingDescriptor
from model.transcription_metrics import (
    aggregate_transcription_metrics,
    transcription_metrics,
)

DEFAULT_DIR = Path.home() / "echosense-testset" / "classroom700"
SAMPLE_RATE = 16000
# Raspberry Pi 5 is about 3-4x slower than this Mac for faster-whisper int8.
PI_FACTOR_LOW, PI_FACTOR_HIGH = 3.0, 4.0
LANGUAGE_GROUPS = (("ceb", "Bisaya"), ("tl", "Tagalog"), ("en", "English"))


def load_clip(path):
    with wave.open(str(path), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
        rate = handle.getframerate()
    if rate != SAMPLE_RATE:
        raise ValueError(f"{path} is {rate} Hz, expected {SAMPLE_RATE}")
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


class Pipeline:
    def __init__(self, args):
        import model.blacklist as blacklist
        import model.whisper_stt as whisper_stt
        from detection.text_detector import TextDetector
        from model.yamnet_infer import load_yamnet_bundle, top_classes_float32

        self.args = args
        self.blacklist = blacklist
        self.whisper_stt = whisper_stt
        blacklist.SOUNDALIKE_ENABLED = args.soundalike == "on"
        blacklist.MISHEARINGS_ENABLED = args.mishearings == "on"
        self.detector = TextDetector()
        yamnet = load_yamnet_bundle(degraded_mode=True)
        if yamnet.available:
            whisper_stt.set_media_classifier(
                lambda samples, rate: top_classes_float32(
                    samples, rate, yamnet.interpreter, yamnet.class_names,
                )
            )
        self.reducer = whisper_stt.NoiseReducer()
        self._load_model()
        self._load_vad()
        self._sync_dictionary()

    def _load_model(self):
        from faster_whisper import WhisperModel

        model = self.args.model
        if (ROOT / "models" / model).is_dir():
            model = str(ROOT / "models" / model)
        started = time.perf_counter()
        # 4 threads like the Pi 5's 4 cores.
        self.model = WhisperModel(
            model, device="cpu", compute_type="int8", cpu_threads=4,
        )
        self.load_seconds = time.perf_counter() - started

    def _load_vad(self):
        from silero_vad import get_speech_timestamps, load_silero_vad

        self._vad_model = load_silero_vad()
        self._speech_timestamps = get_speech_timestamps

    def _sync_dictionary(self):
        from model.monitored_terms import sync_backend_dictionary, get_backend_terms

        with contextlib.redirect_stdout(io.StringIO()):
            synced = sync_backend_dictionary()
        self.dictionary_terms = len(get_backend_terms()) if synced else 0

    def denoise(self, samples):
        blocks = self.reducer.process(samples)
        # Flush the tail the reducer is still holding.
        pending = self.reducer._pending
        self.reducer._pending = np.empty(0, dtype=np.float32)
        out = np.concatenate(blocks + [pending]) if blocks else pending
        return out.astype(np.float32)

    def utterances(self, samples):
        """Speech regions as RealtimeSTT would cut them (0.45 s end silence)."""

        import torch

        stamps = self._speech_timestamps(
            torch.from_numpy(samples), self._vad_model,
            sampling_rate=SAMPLE_RATE,
            threshold=0.4,
            min_silence_duration_ms=450,
            speech_pad_ms=200,
            min_speech_duration_ms=150,
        )
        regions = [(item["start"], item["end"]) for item in stamps]
        if not regions:
            return []
        # RealtimeSTT keeps recordings >= 0.5 s.
        return [
            samples[start:end] for start, end in regions
            if end - start >= SAMPLE_RATE // 4
        ]

    def decode(self, audio):
        from model.language_restricted_engine import decode_restricted

        started = time.perf_counter()
        text, language, probability, segments = decode_restricted(
            self.model, audio,
            language_mode=self.args.language,
            beam_size=self.args.beam_size,
            initial_prompt=self.args.prompt or None,
            gain=self.args.gain,
        )
        fallback = False
        if (
            self.args.language == "tl+en"
            and not self.blacklist.check_transcript(text).get("detected_words")
        ):
            en_text, en_language, en_probability, en_segments = decode_restricted(
                self.model, audio, language="en", beam_size=self.args.beam_size,
                initial_prompt=self.args.prompt or None, gain=self.args.gain,
            )
            fallback = True
            if self.blacklist.check_transcript(en_text).get("detected_words"):
                text, language, probability, segments = (
                    en_text, en_language, en_probability, en_segments,
                )
        return text, language, probability, segments, time.perf_counter() - started, fallback

    def judge(self, audio, text, language, probability, segments):
        from audio.audio_event import AudioEvent
        from detection.false_alarm_filters import summarize_segments
        from detection.fusion import fuse_event_evidence

        event = AudioEvent.from_float32_samples(audio, sample_rate=SAMPLE_RATE)
        with contextlib.redirect_stdout(io.StringIO()):
            self.whisper_stt._on_text(
                text, language, probability, event, summarize_segments(segments),
            )
            result = self.whisper_stt.consume_transcription_result(timeout=1.0)
            evidence = self.detector.analyze_stt_result(result)
        alert = False
        severity = None
        reasons = ()
        if evidence.qualifies_to_open:
            fusion = fuse_event_evidence((evidence,), (), duration_seconds=1.0)
            alert = fusion.deliver
            severity = fusion.level
            reasons = fusion.reasons
        return {
            "transcript": text,
            "language": language,
            "terms": list(evidence.detected_terms),
            "corrections": result.get("corrections", []),
            "filter_reasons": result.get("filter_reasons", []),
            "alert": alert,
            "severity": severity if alert else None,
            "not_sent_reasons": [r for r in reasons if "never_sent" in r],
            "language_guide": result.get("language_guide", {
                "shadow_only": True,
                "matched": False,
            }),
        }


def evaluate(args):
    directory = args.dir
    manifest = json.loads((directory / "phrases.json").read_text(encoding="utf-8"))
    phrases = {item["id"]: item for item in manifest["phrases"]}
    clips = sorted(directory.glob("*__*.wav"))
    if not clips:
        raise SystemExit(f"No clips in {directory}")
    pipeline = Pipeline(args)
    # Warm-up decode so the first clip does not pay one-time costs.
    pipeline.decode(np.zeros(SAMPLE_RATE, dtype=np.float32))

    rows = []
    for path in clips:
        try:
            descriptor = RecordingDescriptor.from_filename(path.name)
        except ValueError:
            print(f"SKIP invalid recording filename: {path.name}")
            continue
        phrase = phrases.get(descriptor.sentence_id)
        if phrase is None:
            continue
        samples = pipeline.denoise(load_clip(path))
        utterances = pipeline.utterances(samples)
        decoded = []
        for audio in utterances:
            text, language, probability, segments, seconds, fallback = pipeline.decode(audio)
            outcome = pipeline.judge(audio, text, language, probability, segments)
            outcome["whisper_seconds"] = round(seconds, 3)
            outcome["english_fallback"] = fallback
            outcome["audio_seconds"] = round(audio.size / SAMPLE_RATE, 2)
            decoded.append(outcome)
        alerted = any(item["alert"] for item in decoded)
        rows.append({
            "id": descriptor.sentence_id,
            "speaker_id": descriptor.speaker_id,
            "delivery": descriptor.delivery,
            "noise": descriptor.noise,
            "distance": descriptor.distance,
            "label": phrase["label"],
            "language": phrase["language"],
            "text": phrase["text"],
            "alert": alerted,
            "utterances": decoded,
        })
        mark = "ALERT" if alerted else "  -  "
        heard = " | ".join(item["transcript"] for item in decoded) or "(no speech)"
        print(
            f"{mark} {phrase['label']:8} {descriptor.distance:5} "
            f"{phrase['text'][:28]:28} heard: {heard[:70]}"
        )

    times = [u["whisper_seconds"] for row in rows for u in row["utterances"]]
    summary = summarize(rows, times)
    out = {
        "name": args.name,
        "settings": {
            "model": args.model, "language": args.language, "prompt": args.prompt,
            "gain": args.gain, "soundalike": args.soundalike,
            "mishearings": args.mishearings, "beam_size": args.beam_size,
        },
        "dictionary_terms": pipeline.dictionary_terms,
        "model_load_seconds": round(pipeline.load_seconds, 1),
        "summary": summary,
        "rows": rows,
    }
    results = directory / "results"
    results.mkdir(exist_ok=True)
    (results / f"{args.name}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    print_table([out])


def _rate(rows):
    if not rows:
        return None
    return sum(row["alert"] for row in rows) / len(rows)


def _row_transcript(row, *, guided=False):
    values = []
    for item in row.get("utterances", ()):
        text = item.get("transcript", "")
        guidance = item.get("language_guide") or {}
        if guided and guidance.get("matched") and guidance.get("suggested_text"):
            text = guidance["suggested_text"]
        values.append(text)
    return " ".join(values)


def summarize(rows, times):
    bullying = [row for row in rows if row["label"] == "bullying"]
    summary = {
        "recall": _rate(bullying),
        "recall_n": len(bullying),
        "recall_near": _rate([r for r in bullying if r["distance"] == "near"]),
        "recall_far": _rate([r for r in bullying if r["distance"] == "far"]),
        "normal_fp": _rate([row for row in rows if row["label"] == "normal"]),
        "normal_n": sum(row["label"] == "normal" for row in rows),
        "media_fp": _rate([row for row in rows if row["label"] == "media"]),
        "media_n": sum(row["label"] == "media" for row in rows),
        "whisper_mean_s": statistics.mean(times) if times else None,
        "whisper_p90_s": (
            sorted(times)[max(0, int(len(times) * 0.9) - 1)] if times else None
        ),
    }
    for code, _name in LANGUAGE_GROUPS:
        group = [row for row in bullying if row["language"] == code]
        summary[f"recall_{code}"] = _rate(group)
        summary[f"recall_{code}_n"] = len(group)
    # The deployment school speaks Bisaya: its false alarms rank configs.
    bisaya_normal = [
        row for row in rows if row["label"] == "normal" and row["language"] == "ceb"
    ]
    summary["normal_fp_ceb"] = _rate(bisaya_normal)
    summary["normal_fp_ceb_n"] = len(bisaya_normal)
    transcript_metrics = [
        transcription_metrics(row["text"], _row_transcript(row))
        for row in rows
    ]
    transcript_summary = aggregate_transcription_metrics(transcript_metrics)
    for key, value in transcript_summary.items():
        summary[f"transcript_{key}"] = value
    for code, _name in LANGUAGE_GROUPS:
        language_metrics = [
            metric
            for row, metric in zip(rows, transcript_metrics)
            if row["language"] == code
        ]
        language_summary = aggregate_transcription_metrics(language_metrics)
        summary[f"transcript_exact_sentence_rate_{code}"] = (
            language_summary["exact_sentence_rate"]
        )
        summary[f"transcript_word_error_rate_{code}"] = (
            language_summary["word_error_rate"]
        )
        summary[f"transcript_character_error_rate_{code}"] = (
            language_summary["character_error_rate"]
        )
    guided_metrics = [
        transcription_metrics(row["text"], _row_transcript(row, guided=True))
        for row in rows
    ]
    guided_summary = aggregate_transcription_metrics(guided_metrics)
    for key, value in guided_summary.items():
        summary[f"guided_{key}"] = value
    return summary


def bisaya_rank(item):
    """Sort by faithful Bisaya transcription before alert-policy outcomes.

    English/Tagalog metrics are reported but never used to pick a config.
    """

    s = item["summary"]

    def high(value):
        return -(value if value is not None else -1.0)

    def low(value):
        return value if value is not None else 2.0

    return (
        low(s.get("transcript_word_error_rate_ceb")),
        high(s.get("transcript_exact_sentence_rate_ceb")),
        high(s.get("recall_ceb")),
        low(s.get("normal_fp_ceb")),
        low(s.get("media_fp")),
        low(s.get("normal_fp")),
        s.get("whisper_mean_s") or 0.0,
    )


def _pct(value):
    return "  n/a" if value is None else f"{value * 100:4.0f}%"


def print_table(results):
    header = (
        f"{'config':14} {'Bisaya':>6} {'BisFP':>5} {'recall':>6} {'exact':>6} "
        f"{'WER':>6} {'gWER':>6} "
        f"{'Tagalog':>7} {'English':>7} "
        f"{'near':>5} {'far':>5} {'normFP':>6} {'mediaFP':>7} "
        f"{'mac s':>6} {'Pi5 s (est)':>12}"
    )
    print("\n" + header)
    print("-" * len(header))
    for item in results:
        s = item["summary"]
        mean = s["whisper_mean_s"] or 0.0
        print(
            f"{item['name']:14} {_pct(s['recall_ceb']):>6} "
            f"{_pct(s.get('normal_fp_ceb')):>5} {_pct(s['recall']):>6} "
            f"{_pct(s.get('transcript_exact_sentence_rate')):>6} "
            f"{_pct(s.get('transcript_word_error_rate')):>6} "
            f"{_pct(s.get('guided_word_error_rate')):>6} "
            f"{_pct(s['recall_tl']):>7} {_pct(s['recall_en']):>7} "
            f"{_pct(s['recall_near']):>5} {_pct(s['recall_far']):>5} "
            f"{_pct(s['normal_fp']):>6} {_pct(s['media_fp']):>7} "
            f"{mean:6.2f} {mean * PI_FACTOR_LOW:5.1f}-{mean * PI_FACTOR_HIGH:4.1f}"
        )


def refreshed_result(result):
    """Return a copy of a saved result with its summary recomputed from rows."""

    rows = result.get("rows") or []
    times = [
        u["whisper_seconds"]
        for row in rows for u in row.get("utterances", ())
        if u.get("whisper_seconds") is not None
    ]
    return {**result, "summary": summarize(rows, times)}


def report(args):
    results = [
        refreshed_result(json.loads(path.read_text(encoding="utf-8")))
        for path in sorted((args.dir / "results").glob("*.json"))
    ]
    print_table(sorted(results, key=bisaya_rank))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--name")
    parser.add_argument("--model", default="base",
                        help="size name, or a folder under models/")
    parser.add_argument("--language", default="auto", choices=["auto", "tl", "en", "tl+en"])
    parser.add_argument("--prompt", default="")
    parser.add_argument("--beam-size", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--gain", default="off", choices=["off", "peak", "rms"])
    parser.add_argument("--soundalike", default="off", choices=["on", "off"])
    parser.add_argument("--mishearings", default="on", choices=["on", "off"])
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args(argv)
    if args.report:
        report(args)
        return 0
    if not args.name:
        parser.error("--name is required")
    evaluate(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
