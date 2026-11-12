"""Speech emotion (emotion2vec_plus_base, ONNX) and per-utterance tone signals.

Runs on each utterance clip in a background thread, in parallel with the
ElevenLabs request (see model/stt_provider.py), so it adds no latency when it
finishes first. Runtime needs only onnxruntime + numpy (Mac and Pi 5 aarch64).
The ONNX file comes from tools/export_emotion2vec.py.

``ToneAnalyzer.start(samples)`` returns a future of a dict:
    emotion: {label, score, scores{label: p}, ms, status}
    yamnet:  {shout_score, shout_label, laughter_score, laughter_label}
    rms:     {utterance_rms, baseline_rms, ratio, baseline_count}

Log lines: ``[EMOTION] label=angry score=0.82 ms=140``. Audio is never logged.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import math
import os
from pathlib import Path
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# fp32 matches the reference model; int8 shifts the angry score (up to 0.2),
# so it is used only when it is the sole file present or configured.
DEFAULT_MODEL_CANDIDATES = (
    ROOT / "models" / "emotion2vec_plus_base.onnx",
    ROOT / "models" / "emotion2vec_plus_base.int8.onnx",
)
# Token order of emotion2vec_plus_base (tokens.txt).
LABELS = (
    "angry", "disgusted", "fearful", "happy", "neutral",
    "other", "sad", "surprised", "unknown",
)
SAMPLE_RATE = 16_000
MIN_SECONDS = 0.4
# Bounds Pi CPU time; the loudest part of a long clip is kept.
MAX_SECONDS = 8.0

SHOUT_CLASSES = ("Shout", "Yell", "Screaming", "Children shouting", "Battle cry")
LAUGHTER_CLASSES = ("Laughter", "Giggle", "Chuckle, chortle", "Snicker", "Belly laugh")


def _flag(env, name, default):
    value = str(env.get(name, "")).strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def resolve_model_path(environ=None):
    env = os.environ if environ is None else environ
    configured = str(env.get("ECHOSENSE_EMOTION_MODEL") or "").strip()
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else ROOT / path
    for candidate in DEFAULT_MODEL_CANDIDATES:
        if candidate.is_file():
            return candidate
    return DEFAULT_MODEL_CANDIDATES[0]


def frame_rms(samples, frame=480):
    """RMS (float scale, 0..1) of each 30 ms frame."""

    samples = np.asarray(samples, dtype=np.float32)
    usable = samples.size - samples.size % frame
    if usable <= 0:
        return np.array([float(np.sqrt(np.mean(samples ** 2)))] if samples.size else [0.0])
    frames = samples[:usable].reshape(-1, frame)
    return np.sqrt(np.mean(frames ** 2, axis=1))


def speech_rms(samples):
    """Loudness of the voiced part: 90th percentile of 30 ms frame RMS.

    Clips carry silence/pre-roll; the plain RMS would mostly measure that.
    """

    return float(np.percentile(frame_rms(samples), 90))


def _loudest_span(samples, max_samples):
    if samples.size <= max_samples:
        return samples
    energy = frame_rms(samples)
    per = 480
    width = max(1, max_samples // per)
    window = np.convolve(energy, np.ones(width), mode="valid")
    start = int(np.argmax(window)) * per
    return samples[start:start + max_samples]


class SpeechBaseline:
    """Rolling median of recent utterance speech RMS (not the silence floor).

    ``ratio`` compares one utterance with the median of the ones before it, so
    a loud class raises the baseline and a quiet class lowers it.
    """

    def __init__(self, size=30, min_count=5):
        self._values = deque(maxlen=int(size))
        self.min_count = int(min_count)
        self._lock = threading.Lock()

    def observe(self, rms):
        """Return (baseline or None, ratio or None, count), then add ``rms``."""

        with self._lock:
            count = len(self._values)
            baseline = float(np.median(self._values)) if count else None
            if rms > 0 and math.isfinite(rms):
                self._values.append(float(rms))
        if baseline is None or count < self.min_count or baseline <= 0:
            return baseline, None, count
        return baseline, float(rms) / baseline, count


class EmotionRecognizer:
    """emotion2vec_plus_base through onnxruntime; lazy-loaded, never raises."""

    def __init__(self, model_path=None, *, threads=None, session=None):
        self.model_path = Path(model_path) if model_path else resolve_model_path()
        self.threads = int(threads or os.environ.get("ECHOSENSE_EMOTION_THREADS") or 2)
        self._session = session
        self._load_error = None
        self._lock = threading.Lock()

    @property
    def available(self):
        return self._ensure() is not None

    def _ensure(self):
        if self._session is not None or self._load_error is not None:
            return self._session
        with self._lock:
            if self._session is not None or self._load_error is not None:
                return self._session
            if not self.model_path.is_file():
                self._load_error = "model_missing"
                print(f"[EMOTION] available=false reason=model_missing path={self.model_path.name}")
                return None
            try:
                import onnxruntime as ort

                options = ort.SessionOptions()
                options.intra_op_num_threads = self.threads
                options.inter_op_num_threads = 1
                self._session = ort.InferenceSession(
                    str(self.model_path), options, providers=["CPUExecutionProvider"],
                )
                print(f"[EMOTION] available=true model={self.model_path.name} threads={self.threads}")
            except Exception as exc:
                self._load_error = f"load_failed:{type(exc).__name__}"
                print(f"[EMOTION] available=false reason={self._load_error}")
        return self._session

    def analyze(self, samples, sample_rate=SAMPLE_RATE):
        started = time.perf_counter()
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if sample_rate != SAMPLE_RATE:
            return {"status": "bad_sample_rate", "label": "unknown", "score": 0.0, "scores": {}, "ms": 0}
        if samples.size < MIN_SECONDS * SAMPLE_RATE:
            return {"status": "too_short", "label": "unknown", "score": 0.0, "scores": {}, "ms": 0}
        session = self._ensure()
        if session is None:
            return {"status": "unavailable", "label": "unknown", "score": 0.0, "scores": {}, "ms": 0}
        clip = _loudest_span(samples, int(MAX_SECONDS * SAMPLE_RATE))
        try:
            output = session.run(None, {session.get_inputs()[0].name: clip.reshape(1, -1)})[0]
        except Exception as exc:
            return {"status": f"error:{type(exc).__name__}", "label": "unknown",
                    "score": 0.0, "scores": {}, "ms": int((time.perf_counter() - started) * 1000)}
        probs = np.asarray(output, dtype=np.float64).reshape(-1)
        scores = {label: round(float(p), 4) for label, p in zip(LABELS, probs)}
        label = max(scores, key=scores.get)
        ms = int((time.perf_counter() - started) * 1000)
        return {"status": "ok", "label": label, "score": scores[label], "scores": scores, "ms": ms}


def yamnet_class_maxima(samples, interpreter, class_names, runner=None):
    """Per-clip max score of the shout and laughter classes over 0.975 s windows."""

    if interpreter is None or not class_names:
        return None
    from model.yamnet_infer import YAMNET_INPUT_SIZE, run_yamnet_float32

    runner = runner or run_yamnet_float32
    index = {name: i for i, name in enumerate(class_names)}
    shout_ids = [(name, index[name]) for name in SHOUT_CLASSES if name in index]
    laugh_ids = [(name, index[name]) for name in LAUGHTER_CLASSES if name in index]
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size < YAMNET_INPUT_SIZE:
        samples = np.pad(samples, (0, YAMNET_INPUT_SIZE - samples.size))
    hop = YAMNET_INPUT_SIZE // 2
    best = {"shout_score": 0.0, "shout_label": None, "laughter_score": 0.0, "laughter_label": None}
    for start in range(0, samples.size - YAMNET_INPUT_SIZE + 1, hop):
        window = np.ascontiguousarray(samples[start:start + YAMNET_INPUT_SIZE])
        _top, _score, scores = runner(interpreter, window, class_names)
        for name, i in shout_ids:
            if float(scores[i]) > best["shout_score"]:
                best["shout_score"], best["shout_label"] = float(scores[i]), name
        for name, i in laugh_ids:
            if float(scores[i]) > best["laughter_score"]:
                best["laughter_score"], best["laughter_label"] = float(scores[i]), name
    best["shout_score"] = round(best["shout_score"], 4)
    best["laughter_score"] = round(best["laughter_score"], 4)
    return best


class ToneAnalyzer:
    """Emotion + YAMNet shout/laughter + speech-RMS ratio for one utterance."""

    def __init__(self, recognizer=None, *, yamnet=None, baseline=None, enabled=None):
        env = os.environ
        self.enabled = _flag(env, "ECHOSENSE_EMOTION_ENABLED", True) if enabled is None else enabled
        self.recognizer = recognizer or EmotionRecognizer()
        # (interpreter, class_names); YAMNet's interpreter is not thread-safe,
        # so the analyzer loads its own copy unless one is injected.
        self._yamnet = yamnet
        self._yamnet_loaded = yamnet is not None
        self.baseline = baseline or SpeechBaseline(
            size=int(env.get("ECHOSENSE_TONE_BASELINE_UTTERANCES") or 30),
        )
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tone")

    def _yamnet_bundle(self):
        if not self._yamnet_loaded:
            self._yamnet_loaded = True
            try:
                from model.yamnet_infer import load_yamnet_bundle

                bundle = load_yamnet_bundle()
                if bundle.available:
                    self._yamnet = (bundle.interpreter, bundle.class_names)
            except Exception as exc:
                print(f"[EMOTION] yamnet_unavailable={type(exc).__name__}")
        return self._yamnet

    def analyze(self, samples, sample_rate=SAMPLE_RATE, event_id="unavailable"):
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        emotion = self.recognizer.analyze(samples, sample_rate)
        bundle = self._yamnet_bundle()
        yamnet = None
        if bundle is not None:
            try:
                yamnet = yamnet_class_maxima(samples, *bundle)
            except Exception as exc:
                print(f"[EMOTION] event={event_id} yamnet_error={type(exc).__name__}")
        rms = speech_rms(samples) if samples.size else 0.0
        baseline, ratio, count = self.baseline.observe(rms)
        signals = {
            "emotion": emotion,
            "yamnet": yamnet,
            "rms": {
                "utterance_rms": round(rms, 5),
                "baseline_rms": None if baseline is None else round(baseline, 5),
                "ratio": None if ratio is None else round(ratio, 3),
                "baseline_count": count,
            },
        }
        angry = emotion.get("scores", {}).get("angry", 0.0)
        print(
            f"[EMOTION] event={event_id} label={emotion['label']} score={emotion['score']:.2f} "
            f"angry={angry:.2f} ms={emotion['ms']} status={emotion['status']} "
            f"shout={(yamnet or {}).get('shout_score', 0.0):.2f} "
            f"laughter={(yamnet or {}).get('laughter_score', 0.0):.2f} "
            f"rms_ratio={'n/a' if ratio is None else f'{ratio:.2f}'}"
        )
        return signals

    def start(self, samples, sample_rate=SAMPLE_RATE, event_id="unavailable"):
        """Begin analysis in the background; returns a Future or None."""

        if not self.enabled:
            return None
        try:
            return self._pool.submit(self.analyze, samples, sample_rate, event_id)
        except RuntimeError:
            return None

    @staticmethod
    def collect(future, timeout=None):
        """Wait for a started analysis; None on timeout/failure (never raises)."""

        if future is None:
            return None
        if timeout is None:
            timeout = float(os.environ.get("ECHOSENSE_EMOTION_TIMEOUT_SECONDS") or 5.0)
        try:
            return future.result(timeout=timeout)
        except Exception as exc:
            print(f"[EMOTION] collect_failed={type(exc).__name__}")
            return None

    def close(self):
        self._pool.shutdown(wait=False, cancel_futures=True)
