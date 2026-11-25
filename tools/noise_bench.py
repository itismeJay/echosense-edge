"""Noise-robustness benchmark: labeled speech mixed with classroom noise.

    venv/bin/python tools/noise_bench.py mix                 # build ~/echosense-noisebench
    venv/bin/python tools/noise_bench.py cost [--subset N]   # Scribe minutes estimate
    venv/bin/python tools/noise_bench.py run [--subset N]    # run, write results.csv

Each clip goes through the live per-utterance path with no backend contact:
HybridTranscriptionExecutor (ElevenLabs scribe_v3 ceb + keyterms, tone analyzer
in parallel) -> whisper_stt._on_text (filters, matching, media gate) ->
TextDetector with the Haiku SemanticGate -> EventCoordinator (with
AcousticDetector frames, warmed on 5 s of the same noise) -> finalized
incident. The school word list comes from the cached school config (read
only). ElevenLabs and Haiku answers are cached in the bench folder.

Not simulated: RealtimeSTT VAD cutting (each mix is one utterance; noise-only
clips are cut into 15 s utterances, the recording cap).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import random
import sys
import threading
import time
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SR = 16000
SPEECH_DIR = Path.home() / "echosense-testset"
NOISE_DIR = Path.home() / "echosense-noise"
BENCH = Path.home() / "echosense-noisebench"
SNRS = (20, 10, 5, 0)
TYPES = {
    "crowd": "classroom_", "laughter": "laughter_", "lesson": "lesson_teacher_",
    "rain": "rain_roof_", "fan": "fan_", "outside": "outside_", "singing": "singing_",
}
SEED = 20261117
NOISE_CHUNK_S = 15.0


def read_wav(path):
    with wave.open(str(path), "rb") as h:
        assert h.getframerate() == SR and h.getnchannels() == 1, path
        return np.frombuffer(h.readframes(h.getnframes()), np.int16).astype(np.float32) / 32768.0


def write_wav(path, x):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as h:
        h.setnchannels(1)
        h.setsampwidth(2)
        h.setframerate(SR)
        h.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def speech_power(x):
    """Mean power of the active speech frames (frame RMS >= 10% of the loudest)."""
    n = 320
    frames = x[: len(x) // n * n].reshape(-1, n)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    active = frames[rms >= 0.1 * rms.max()]
    return float((active ** 2).mean())


def labels():
    phrases = json.loads((SPEECH_DIR / "phrases.json").read_text())["phrases"]
    return {p["id"]: p for p in phrases}


def speech_clips():
    lab = labels()
    out = []
    for p in sorted(SPEECH_DIR.glob("*.wav")):
        pid = p.stem.split("__")[0]
        if pid in lab and lab[pid]["label"] in ("bullying", "normal"):
            out.append((p, lab[pid]))
    return out


def noise_files(kind):
    return sorted(NOISE_DIR.glob(TYPES[kind] + "*.wav"))


def noise_segment(rng, noise, length):
    if len(noise) < length:
        noise = np.tile(noise, length // len(noise) + 1)
    start = rng.randrange(0, len(noise) - length + 1)
    return noise[start:start + length]


def cmd_mix(_args):
    rng = random.Random(SEED)
    noises = {k: [(f, read_wav(f)) for f in noise_files(k)] for k in TYPES}
    rows = []
    for path, meta in speech_clips():
        speech = read_wav(path)
        p_s = speech_power(speech)
        for kind in TYPES:
            for snr in SNRS:
                nfile, noise = rng.choice(noises[kind])
                seg = noise_segment(rng, noise, len(speech) + 5 * SR)
                warm, seg = seg[: 5 * SR], seg[5 * SR:]
                gain = np.sqrt(p_s / (10 ** (snr / 10)) / max(float((seg ** 2).mean()), 1e-12))
                mix = speech + gain * seg
                peak = float(np.abs(mix).max())
                scale = 0.99 / peak if peak > 0.99 else 1.0
                out = BENCH / "mixes" / kind / f"snr{snr}" / path.name
                write_wav(out, mix * scale)
                write_wav(out.with_suffix(".warm.wav"), gain * warm * scale)
                rows.append({"clip": path.name, "phrase": meta["id"], "label": meta["label"],
                             "noise_type": kind, "snr_db": snr, "noise_file": nfile.name,
                             "mix": str(out.relative_to(BENCH)), "peak_scale": round(scale, 3)})
    with open(BENCH / "mixes.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} mixes to {BENCH / 'mixes'}")


def plan(subset):
    with open(BENCH / "mixes.csv") as h:
        mixes = list(csv.DictReader(h))
    clips = sorted({(m["clip"], m["label"]) for m in mixes})
    if subset:
        rng = random.Random(SEED)
        bully = [c for c, l in clips if l == "bullying"]
        normal = [c for c, l in clips if l == "normal"]
        nb = round(subset * len(bully) / len(clips))
        chosen = set(rng.sample(bully, nb) + rng.sample(normal, subset - nb))
        mixes = [m for m in mixes if m["clip"] in chosen]
    else:
        chosen = {c for c, _ in clips}
    return sorted(chosen), mixes


def seconds(path):
    with wave.open(str(path)) as h:
        return h.getnframes() / SR


def cmd_cost(args):
    chosen, mixes = plan(args.subset)
    clean = sum(seconds(SPEECH_DIR / c) for c in chosen)
    mixed = sum(seconds(BENCH / m["mix"]) for m in mixes)
    noise = sum(seconds(f) for f in NOISE_DIR.glob("*.wav"))
    cache = load_cache("stt")
    print(f"clips={len(chosen)} mixes={len(mixes)}  Scribe minutes: clean={clean/60:.1f} "
          f"mixed={mixed/60:.1f} noise_only={noise/60:.1f} total={(clean+mixed+noise)/60:.1f} "
          f"(cached entries: {len(cache)})")


# ---------------------------------------------------------------- caches
_locks = {"stt": threading.Lock(), "llm": threading.Lock()}
_caches = {}


def load_cache(name):
    if name not in _caches:
        path = BENCH / f"cache_{name}.pkl"
        _caches[name] = pickle.loads(path.read_bytes()) if path.exists() else {}
    return _caches[name]


def save_cache(name):
    with _locks[name]:
        path = BENCH / f"cache_{name}.pkl"
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(_caches[name]))
        tmp.replace(path)


class CacheMiss(RuntimeError):
    pass


class CachedSTT:
    cache_only = False

    def __init__(self, client):
        self.client = client
        self.calls = 0
        self.seconds = 0.0

    def transcribe_detailed(self, audio, sample_rate=SR):
        key = hashlib.sha1(np.asarray(audio, np.float32).tobytes()).hexdigest()
        cache = load_cache("stt")
        if key not in cache and self.cache_only:
            raise CacheMiss("Scribe result not cached (--cache-only)")
        if key not in cache:
            for attempt in range(3):
                try:
                    cache[key] = self.client.transcribe_detailed(audio, sample_rate)
                    break
                except Exception as exc:
                    reason = getattr(exc, "reason", type(exc).__name__)
                    print(f"[BENCH] scribe_error={reason} attempt={attempt + 1}")
                    if attempt == 2 or str(reason).startswith("quota"):
                        raise
                    time.sleep(5)
            self.calls += 1
            self.seconds += len(audio) / SR
            if self.calls % 20 == 0:
                save_cache("stt")
        return cache[key]


class CachedLLM:
    def __init__(self, client):
        self.client = client
        self.model = getattr(client, "model", "")

    def classify(self, item):
        key = repr(item)
        cache = load_cache("llm")
        if key not in cache:
            cache[key] = self.client.classify(item)
        return cache[key]


# ---------------------------------------------------------------- pipeline
class Bench:
    def __init__(self):
        import contextlib, io
        import config  # noqa: F401  loads edge.env
        from config import DETECTION_RUNTIME, SEMANTIC_RUNTIME
        import model.whisper_stt as ws
        from model import school_config
        from model.stt_provider import HybridTranscriptionExecutor, STTSettings, ElevenLabsClient, load_keyterms
        from model.yamnet_infer import load_yamnet_bundle, top_classes_float32
        from detection.llm_semantic import SemanticGate

        self.cfg = DETECTION_RUNTIME
        self.sem_cfg = SEMANTIC_RUNTIME
        self.ws = ws
        # The newest cached school config on this machine (read only, no fetch).
        newest = max((ROOT / "data").glob("school_config.*.json"),
                     key=lambda f: f.stat().st_mtime, default=None)
        self.school = "none"
        if newest is not None:
            body = json.loads(newest.read_text())
            school_config.apply_config(body["config"], origin="cache")
            self.school = f"{newest.name} ({body.get('device_id')}, {body['config'].get('sensitivity')})"
        media = load_yamnet_bundle(degraded_mode=True)
        ws.set_media_classifier(lambda s, r: top_classes_float32(s, r, media.interpreter, media.class_names))
        self.yamnet = load_yamnet_bundle(degraded_mode=True)
        settings = STTSettings.from_environment()
        if not settings.api_key or settings.provider != "elevenlabs":
            raise SystemExit("needs ELEVENLABS_API_KEY and provider elevenlabs")
        self.stt = CachedSTT(ElevenLabsClient(settings, load_keyterms()))
        self.settings = settings
        self.Hybrid = HybridTranscriptionExecutor
        # Local Whisper: the second try when Scribe returns (almost) nothing.
        from config import WHISPER_MODEL
        from model.language_restricted_engine import RestrictedWhisperExecutor
        self.whisper = RestrictedWhisperExecutor(
            model=WHISPER_MODEL, device="cpu", compute_type="int8", beam_size=1,
            initial_prompt=None, suppress_tokens=(-1,), batch_size=0, vad_filter=False)
        gate = SemanticGate.from_environment()
        if not gate.available:
            raise SystemExit("Haiku semantic gate unavailable (no OPENROUTER/ANTHROPIC key)")
        gate.client = CachedLLM(gate.client)
        self.gate = gate
        from detection.llm_semantic import SemanticInput
        try:  # warm the HTTP connection so the first clip is not a cold timeout
            gate.client.client.classify(SemanticInput(transcript="maayong buntag"))
        except Exception as exc:
            print(f"[BENCH] haiku_warmup_failed={type(exc).__name__}")
        self.quiet = lambda: contextlib.redirect_stdout(io.StringIO())

    def tone_analyzer(self):
        from model.emotion_recognizer import ToneAnalyzer
        return ToneAnalyzer()

    def process(self, audio, tone, warm=None):
        """One utterance through the live path; returns a result dict."""
        from datetime import datetime, timedelta, timezone
        from audio.audio_event import AudioEvent
        from audio.shared_source import AudioFrame, RollingAudioBuffer
        from detection.acoustic_detector import AcousticDetector, AcousticDetectorWorker, NoiseFloorTracker
        from detection.coordinator import EventCoordinator
        from detection.semantic import SemanticAnalyzer
        from detection.text_detector import TextDetector

        cfg = self.cfg
        incidents = []
        pre = RollingAudioBuffer(cfg.pre_trigger_seconds)
        coord = EventCoordinator(config=cfg, pre_trigger_buffer=pre,
                                 finalizer=lambda inc: incidents.append(inc) or True)
        acoustic_ev = []

        def on_acoustic(ev):
            acoustic_ev.append(ev)
            return coord.submit_acoustic(ev)

        worker = AcousticDetectorWorker(
            AcousticDetector(interpreter=self.yamnet.interpreter, class_names=self.yamnet.class_names,
                             unavailable_reason=self.yamnet.reason,
                             noise_floor=NoiseFloorTracker.from_config(cfg)),
            None, on_acoustic, window_samples=cfg.yamnet_window_samples,
            hop_samples=cfg.yamnet_hop_samples)
        stream = audio if warm is None else np.concatenate((warm, audio))
        n = cfg.frame_samples
        t_end = time.monotonic()
        t0 = t_end - len(stream) / SR
        wall_end = datetime.now(timezone.utc)
        with self.quiet():
            for i in range(0, len(stream) - n + 1, n):
                frame = AudioFrame(
                    sequence=i // n,
                    captured_at=wall_end - timedelta(seconds=(len(stream) - i - n) / SR),
                    monotonic_start=t0 + i / SR, monotonic_end=t0 + (i + n) / SR,
                    samples=np.ascontiguousarray(stream[i:i + n], dtype=np.float32))
                pre.append(frame)
                worker._accept(frame)
                coord.observe_audio(frame)
            executor = self.Hybrid(self.whisper, self.settings, client=self.stt, tone_analyzer=tone)
            result = executor.transcribe(audio)
            meta = result.metadata
            event = AudioEvent.from_float32_samples(audio, sample_rate=SR)
            self.ws._on_text(result.text, result.info.language, result.info.language_probability,
                             event, meta)
            stt_result = self.ws.consume_transcription_result(timeout=5.0)
            detector = TextDetector(
                history_size=cfg.evidence_history_size, semantic_gate=self.gate,
                semantic_analyzer=SemanticAnalyzer(
                    classifier=None, enabled=self.sem_cfg.enabled,
                    min_confidence=self.sem_cfg.min_confidence,
                    max_context_turns=self.sem_cfg.max_context_turns,
                    context_window_seconds=self.sem_cfg.context_window_seconds,
                    max_context_chars=self.sem_cfg.max_context_chars))
            evidence = detector.analyze_stt_result(stt_result)
            coord.submit_text(evidence)
            coord.force_finalize()
            coord.shutdown(finalize_active=True)
        tone_sig = meta.get("tone") or {}
        tv = getattr(evidence, "tone_vote", None)
        sup = list(incidents[0].supporting_evidence) if incidents else []
        inc = incidents[0] if incidents else None
        return {
            "transcript": stt_result.get("transcript", ""),
            "stt_language": result.info.language,
            "audio_tags": "|".join(meta.get("stt", {}).get("audio_tags") or ()),
            "matched": "|".join(stt_result.get("detected_words") or ()),
            "filter_reasons": "|".join(map(str, stt_result.get("filter_reasons") or ())),
            "decision_terms": "|".join(getattr(evidence, "decision_terms", ()) or ()),
            "qualifies": bool(getattr(evidence, "qualifies_to_open", False)),
            "emotion": (tone_sig.get("emotion") or {}).get("label"),
            "angry": round(float(((tone_sig.get("emotion") or {}).get("scores") or {}).get("angry", 0.0)), 3),
            "shout": round(float((tone_sig.get("yamnet") or {}).get("shout_score", 0.0) or 0.0), 3),
            "laugh": round(float((tone_sig.get("yamnet") or {}).get("laughter_score", 0.0) or 0.0), 3),
            "rms_ratio": (tone_sig.get("rms") or {}).get("ratio"),
            "acoustic_events": len(acoustic_ev),
            "semantic": ":".join(e.split(":", 1)[1] for e in sup
                                 if e.startswith(("semantic_label:", "semantic_confidence:", "semantic_fallback:"))),
            "alert": inc is not None,
            "needs_review": bool(inc and "needs_review:true" in inc.supporting_evidence),
            "stt_provider": stt_result.get("stt_provider", ""),
            "second_try_s": round((meta.get("stt", {}).get("second_try_ms") or 0) / 1000, 2),
            "severity": inc.severity if inc else "",
            "summary": inc.summary if inc else "",
            "supporting": "|".join(inc.supporting_evidence) if inc else "",
            "angry_tone": bool(getattr(getattr(inc, "tone", None), "angry_tone", False)) if inc else "",
            "laughter_vote": bool(getattr(getattr(inc, "tone", None), "laughter", False)) if inc else "",
        }


def cmd_run(args):
    chosen, mixes = plan(args.subset)
    lab = labels()
    CachedSTT.cache_only = bool(args.cache_only)
    bench = Bench()
    out = BENCH / (args.out or ("results_subset.csv" if args.subset else "results.csv"))
    rows = []
    started = time.time()

    def emit(row):
        rows.append(row)
        if len(rows) % 25 == 0:
            el = time.time() - started
            print(f"[BENCH] {len(rows)} done  {el/60:.1f} min  scribe_new={bench.stt.seconds/60:.1f} min",
                  flush=True)
            save_cache("llm")

    # Clean baseline: one tone session per condition, clips in fixed order.
    groups = [("clean", None)] + [(k, s) for k in TYPES for s in SNRS]
    for kind, snr in groups:
        tone = bench.tone_analyzer()
        for clip in chosen:
            pid = clip.split("__")[0]
            if kind == "clean":
                audio, warm, nfile = read_wav(SPEECH_DIR / clip), None, ""
            else:
                m = next(m for m in mixes if m["clip"] == clip and m["noise_type"] == kind
                         and int(m["snr_db"]) == snr)
                audio = read_wav(BENCH / m["mix"])
                warm = read_wav((BENCH / m["mix"]).with_suffix(".warm.wav"))
                nfile = m["noise_file"]
            r = bench.process(audio, tone, warm)
            emit({"set": "speech", "clip": clip, "phrase": pid, "label": lab[pid]["label"],
                  "reference": lab[pid]["text"], "noise_type": kind, "snr_db": "" if snr is None else snr,
                  "noise_file": nfile, **r})
    # Noise alone, cut into 15 s utterances (the recording cap).
    tone = bench.tone_analyzer()
    for f in sorted(NOISE_DIR.glob("*.wav")):
        kind = next(k for k, p in TYPES.items() if f.name.startswith(p))
        x = read_wav(f)
        step = int(NOISE_CHUNK_S * SR)
        for j in range(0, len(x), step):
            chunk = x[j:j + step]
            if len(chunk) < SR // 2:
                continue
            r = bench.process(chunk, tone, None)
            emit({"set": "noise_only", "clip": f"{f.name}@{j // SR}s", "phrase": "", "label": "noise",
                  "reference": "", "noise_type": kind, "snr_db": "", "noise_file": f.name, **r})
    save_cache("stt")
    save_cache("llm")
    with open(out, "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[BENCH] wrote {out} rows={len(rows)} new_scribe_minutes={bench.stt.seconds/60:.1f} "
          f"school_config={bench.school}")
    bench.gate.shutdown()
    bench.whisper.close()


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("mix")
    for name in ("cost", "run"):
        p = sub.add_parser(name)
        p.add_argument("--subset", type=int, default=0, help="random speech clips (0 = all)")
        p.add_argument("--out")
        p.add_argument("--cache-only", action="store_true",
                       help="never call Scribe; fail on an uncached clip")
    rp = sub.add_parser("report")
    rp.add_argument("--csv", default="results.csv")
    cp = sub.add_parser("compare")
    cp.add_argument("before")
    cp.add_argument("after")
    args = ap.parse_args(argv)
    {"mix": cmd_mix, "cost": cmd_cost, "run": cmd_run, "report": cmd_report,
     "compare": cmd_compare}[args.cmd](args)



# ---------------------------------------------------------------- report
RANK = {"": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}


def _norm(text):
    import re
    return re.sub(r"[^a-z' ]", " ", str(text).lower()).split()


def failure_reason(r, clean):
    if r["label"] == "normal":
        return f"false alarm: '{r['transcript']}' -> {r['matched'] or 'by meaning'} ({r['summary']})"
    if r["alert"] == "True":
        return f"severity {r['severity']} vs clean {clean.get('severity') or 'none'}"
    ref = [w for w in _norm(r["reference"]) if w not in ("ka", "tika", "ko", "sa", "ug", "nako", "nimo", "ang")]
    heard = set(_norm(r["transcript"]))
    if not r["transcript"].strip():
        return "empty transcript" + (f" ({r['filter_reasons']})" if r["filter_reasons"] else "")
    if r["matched"] and r["filter_reasons"]:
        return f"matched then filtered ({r['filter_reasons']})"
    if r["matched"]:
        return f"matched '{r['matched']}' but no incident ({r['semantic'] or 'gate'})"
    if ref and not any(w in heard for w in ref):
        return f"bad transcript: '{r['transcript']}'"
    return f"no match on '{r['transcript']}'"


def cmd_report(args):
    from collections import Counter, defaultdict
    path = BENCH / args.csv
    rows = list(csv.DictReader(open(path)))
    speech = [r for r in rows if r["set"] == "speech"]
    clean = {r["clip"]: r for r in speech if r["noise_type"] == "clean"}
    conds = [("clean", "")] + [(k, str(s)) for k in TYPES for s in SNRS]
    lines = ["| noise | SNR dB | catch rate (bully) | correct severity | false alarms (normal) |",
             "|---|---|---|---|---|"]
    by = defaultdict(list)
    for r in speech:
        by[(r["noise_type"], r["snr_db"])].append(r)
    for kind, snr in conds:
        rs = by[(kind, snr)]
        b = [r for r in rs if r["label"] == "bullying"]
        n = [r for r in rs if r["label"] == "normal"]
        caught = [r for r in b if r["alert"] == "True"]
        ref = [r for r in caught if clean[r["clip"]]["alert"] == "True"]
        same = [r for r in ref if r["severity"] == clean[r["clip"]]["severity"]]
        fa = [r for r in n if r["alert"] == "True"]
        pct = lambda a, t: f"{len(a)}/{t} ({100 * len(a) / t:.0f}%)" if t else "n/a"
        lines.append(f"| {kind} | {snr or '-'} | {pct(caught, len(b))} | {pct(same, len(ref))} | {pct(fa, len(n))} |")
    table = "\n".join(lines)
    # SNR summary across noise types
    snr_lines = ["| SNR dB | catch rate | correct severity | false alarms |", "|---|---|---|---|"]
    for snr in [""] + [str(s) for s in SNRS]:
        rs = [r for r in speech if r["snr_db"] == snr]
        b = [r for r in rs if r["label"] == "bullying"]
        n = [r for r in rs if r["label"] == "normal"]
        c = [r for r in b if r["alert"] == "True"]
        ref = [r for r in c if clean[r["clip"]]["alert"] == "True"]
        same = [r for r in ref if r["severity"] == clean[r["clip"]]["severity"]]
        f = [r for r in n if r["alert"] == "True"]
        snr_lines.append(f"| {snr or 'clean'} | {100*len(c)/max(len(b),1):.0f}% | "
                         f"{100*len(same)/max(len(ref),1):.0f}% | {100*len(f)/max(len(n),1):.0f}% |")
    # noise only
    noise = [r for r in rows if r["set"] == "noise_only"]
    na = [r for r in noise if r["alert"] == "True"]
    noise_lines = [f"{len(na)} of {len(noise)} noise-only utterances raised an alert."]
    for r in na:
        noise_lines.append(f"- `{r['clip']}` -> **{r['severity']}** \"{r['transcript'][:120]}\" "
                           f"matched `{r['matched'] or '-'}`; {r['summary']}; Haiku `{r['semantic'] or '-'}`")
    heard = [r for r in noise if r["transcript"].strip() and r["alert"] != "True"]
    noise_lines.append(f"\n{len(heard)} other noise utterances produced text but no alert"
                       f" ({sum(1 for r in heard if r['matched'])} had a matched word that was then dropped).")
    # failing phrases (noisy conditions only)
    fails = defaultdict(list)
    tot = Counter()
    for r in speech:
        if r["noise_type"] == "clean":
            continue
        tot[r["clip"]] += 1
        c = clean[r["clip"]]
        bad = ((r["label"] == "bullying" and (r["alert"] != "True" or
                (c["alert"] == "True" and r["severity"] != c["severity"])))
               or (r["label"] == "normal" and r["alert"] == "True"))
        if bad:
            fails[r["clip"]].append(r)
    worst = sorted(fails, key=lambda k: -len(fails[k]))[:10]
    fail_lines = ["| clip | label | clean result | failures | main reasons (example) |", "|---|---|---|---|---|"]
    for k in worst:
        c = clean[k]
        reasons = Counter(failure_reason(r, c).split(":")[0].split(" (")[0] for r in fails[k])
        ex = failure_reason(max(fails[k], key=lambda r: -int(r["snr_db"])), c)
        fail_lines.append(f"| {k} | {c['label']} | {c['severity'] or 'no alert'} | {len(fails[k])}/{tot[k]} | "
                          f"{', '.join(f'{n}× {w}' for w, n in reasons.most_common(3))}. e.g. {ex[:110]} |")
    # tone check
    tone_lines = ["| noise | alerts | angry_tone set | severity above clean | laughter vote set |", "|---|---|---|---|---|"]
    for kind in ("clean", "laughter", "singing", "crowd"):
        rs = [r for r in speech if r["noise_type"] == kind and r["alert"] == "True" and r["label"] == "bullying"]
        up = [r for r in rs if RANK[r["severity"]] > RANK[clean[r["clip"]]["severity"]]]
        tone_lines.append(f"| {kind} | {len(rs)} | {sum(r['angry_tone'] == 'True' for r in rs)} | "
                          f"{len(up)} | {sum(r['laughter_vote'] == 'True' for r in rs)} |")
    ups = [r for r in speech if r["noise_type"] in ("laughter", "singing", "crowd") and r["alert"] == "True" and r["label"] == "bullying"
           and RANK[r["severity"]] > RANK[clean[r["clip"]]["severity"]]]
    for r in ups[:10]:
        tone_lines.append(f"\n- `{r['noise_type']}/{r['snr_db']}dB {r['clip']}`: {clean[r['clip']]['severity']} -> "
                          f"{r['severity']} ({r['summary']}; angry={r['angry']} shout={r['shout']} laugh={r['laugh']})")
    out = {"table": table, "snr": "\n".join(snr_lines), "noise": "\n".join(noise_lines),
           "fails": "\n".join(fail_lines), "tone": "\n".join(tone_lines)}
    for k, v in out.items():
        print(f"\n## {k}\n{v}")
    (BENCH / (path.stem + "_sections.json")).write_text(json.dumps(out, ensure_ascii=False, indent=1))


# ---------------------------------------------------------------- compare
def pushed(r):
    """An alert that reaches the teacher (a needs_review alert is dashboard only)."""
    return r["alert"] == "True" and r.get("needs_review", "False") != "True"


def snr_metrics(rows):
    speech = [r for r in rows if r["set"] == "speech"]
    clean = {r["clip"]: r for r in speech if r["noise_type"] == "clean"}
    out = {}
    for snr in ["", "20", "10", "5", "0"]:
        rs = [r for r in speech if r["snr_db"] == snr]
        b = [r for r in rs if r["label"] == "bullying"]
        n = [r for r in rs if r["label"] == "normal"]
        c = [r for r in b if pushed(r)]
        ref = [r for r in c if pushed(clean[r["clip"]])]
        same = [r for r in ref if r["severity"] == clean[r["clip"]]["severity"]]
        out[snr or "clean"] = {
            "catch": (len(c), len(b)), "sev": (len(same), len(ref)),
            "fa": (sum(pushed(r) for r in n), len(n)),
            "review_b": sum(r["alert"] == "True" and not pushed(r) for r in b),
            "review_n": sum(r["alert"] == "True" and not pushed(r) for r in n),
        }
    noise = [r for r in rows if r["set"] == "noise_only"]
    out["noise_only"] = {"fa": (sum(pushed(r) for r in noise), len(noise)),
                         "review_n": sum(r["alert"] == "True" and not pushed(r) for r in noise)}
    return out


def cmd_compare(args):
    a = snr_metrics(list(csv.DictReader(open(BENCH / args.before))))
    b = snr_metrics(list(csv.DictReader(open(BENCH / args.after))))
    f = lambda t: f"{t[0]}/{t[1]} ({100 * t[0] / t[1]:.0f}%)" if t[1] else "n/a"
    print("| SNR dB | catch (before → after) | correct severity | false alarms (pushed) "
          "| needs_review bully / normal (after) |")
    print("|---|---|---|---|---|")
    for k in ["clean", "20", "10", "5", "0"]:
        print(f"| {k} | {f(a[k]['catch'])} → {f(b[k]['catch'])} | {f(a[k]['sev'])} → {f(b[k]['sev'])} "
              f"| {f(a[k]['fa'])} → {f(b[k]['fa'])} | {b[k]['review_b']} / {b[k]['review_n']} |")
    print(f"| noise only | | | {f(a['noise_only']['fa'])} → {f(b['noise_only']['fa'])} "
          f"| - / {b['noise_only']['review_n']} |")
    ra = {(r["set"], r["clip"], r["noise_type"], r["snr_db"]): r
          for r in csv.DictReader(open(BENCH / args.before))}
    print("\nchanged rows:")
    for r in csv.DictReader(open(BENCH / args.after)):
        o = ra.get((r["set"], r["clip"], r["noise_type"], r["snr_db"]))
        if o is None:
            continue
        st = lambda x: ("review " if x.get("needs_review") == "True" else "") + (x["severity"] or "none")
        if st(o) != st(r):
            print(f"  {r['noise_type']}/{r['snr_db'] or '-'} {r['clip']} [{r['label']}] "
                  f"{st(o)} -> {st(r)}  '{r['transcript'][:60]}' {r['summary'][:70]}")


if __name__ == "__main__":
    main()
