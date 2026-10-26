import csv
from dataclasses import dataclass
import math
from pathlib import Path
import re
import unicodedata

import numpy as np

from config import YAMNET_MODEL_PATH, YAMNET_CLASSES_PATH


@dataclass(frozen=True, slots=True)
class YamnetLoadResult:
    interpreter: object | None
    class_names: tuple[str, ...]
    available: bool
    reason: str


def load_class_names(path=YAMNET_CLASSES_PATH):
    class_names = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            class_names.append(row[2])
    return class_names


def load_yamnet(path=YAMNET_MODEL_PATH):
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter


def load_yamnet_bundle(
    model_path=YAMNET_MODEL_PATH,
    classes_path=YAMNET_CLASSES_PATH,
    *,
    degraded_mode=True,
):
    """Validate assets and explicitly return active or degraded model state."""

    model = Path(model_path)
    classes = Path(classes_path)
    missing = []
    if not model.is_file():
        missing.append("model_file_missing")
    if not classes.is_file():
        missing.append("class_map_missing")
    if missing:
        reason = ",".join(missing)
        if not degraded_mode:
            raise RuntimeError(f"YAMNet unavailable: {reason}")
        return YamnetLoadResult(None, (), False, reason)
    try:
        interpreter = load_yamnet(str(model))
        class_names = tuple(load_class_names(str(classes)))
        if not class_names:
            raise ValueError("class map is empty")
    except Exception as exc:
        reason = f"load_failed:{type(exc).__name__}"
        if not degraded_mode:
            raise RuntimeError(f"YAMNet unavailable: {reason}") from exc
        return YamnetLoadResult(None, (), False, reason)
    return YamnetLoadResult(interpreter, class_names, True, "available")


def _run_yamnet_float32(interpreter, audio_float, class_names):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], audio_float)
    interpreter.invoke()
    scores = interpreter.get_tensor(output_details[0]['index'])
    mean_scores = np.mean(scores, axis=0)
    top_index = np.argmax(mean_scores)
    top_class = class_names[top_index]
    top_score = mean_scores[top_index]
    return top_class, top_score, mean_scores


def run_yamnet(interpreter, audio_data, class_names):
    """Run one legacy int16 window after explicit amplitude normalization."""
    if not isinstance(audio_data, np.ndarray) or audio_data.ndim != 1:
        raise ValueError("YAMNet int16 audio must be a one-dimensional array")
    if audio_data.dtype != np.int16:
        raise TypeError("run_yamnet expects int16 audio")
    audio_float = audio_data.astype(np.float32) / 32768.0
    return _run_yamnet_float32(interpreter, audio_float, class_names)


def run_yamnet_float32(interpreter, audio_data, class_names):
    """Run one already-normalized float32 window without scaling it again."""
    if not isinstance(audio_data, np.ndarray) or audio_data.ndim != 1:
        raise ValueError("YAMNet float32 audio must be a one-dimensional array")
    if audio_data.dtype != np.float32:
        raise TypeError("run_yamnet_float32 expects float32 audio")
    return _run_yamnet_float32(interpreter, audio_data, class_names)


# Distress / aggression sounds ONLY. "Crowd" and "Noise" were removed because a
# Filipino classroom of 40-50 students matches them constantly — loud != bullying.
AGGRESSIVE_CLASSES = frozenset({
    # Exact labels in the validated 521-class map.
    "Screaming",
    "Yell",
    "Shout",
    "Crying, sobbing",
    "Whimper",
    "Wail, moan",
    # Controlled aliases used by compatible maps/tests. Matching remains exact.
    "Scream",
    "Crying",
    "Wail",
})


def normalize_yamnet_label(value: str) -> str:
    """Normalize spelling/spacing without broad semantic substring matching."""

    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


_NORMALIZED_AGGRESSIVE_CLASSES = frozenset(
    normalize_yamnet_label(label) for label in AGGRESSIVE_CLASSES
)


# YAMNet's tflite graph expects exactly 15600 samples (0.975s @ 16kHz) per call.
YAMNET_INPUT_SIZE = 15600


def is_aggressive_sound(class_name: str, score: float, threshold: float) -> bool:
    try:
        numeric_score = float(score)
        numeric_threshold = float(threshold)
    except (TypeError, ValueError):
        return False
    if (
        not math.isfinite(numeric_score)
        or not math.isfinite(numeric_threshold)
        or numeric_score < numeric_threshold
    ):
        return False
    return normalize_yamnet_label(class_name) in _NORMALIZED_AGGRESSIVE_CLASSES


def _scan_windows(interpreter, audio_np, class_names, window_runner):
    """Run YAMNet across a multi-second buffer by splitting it into 15600-sample
    windows. Returns (class, score) for the strongest AGGRESSIVE window if any
    aggressive class appears; otherwise the single highest-scoring window."""
    n = len(audio_np)
    if n < YAMNET_INPUT_SIZE:
        audio_np = np.pad(audio_np, (0, YAMNET_INPUT_SIZE - n))
        n = YAMNET_INPUT_SIZE

    num_windows = n // YAMNET_INPUT_SIZE
    best_overall = (class_names[0] if class_names else "Unknown", 0.0)
    best_aggressive = None

    for i in range(num_windows):
        window = audio_np[i * YAMNET_INPUT_SIZE:(i + 1) * YAMNET_INPUT_SIZE]
        cls, score, _ = window_runner(interpreter, window, class_names)
        score = float(score)
        if score > best_overall[1]:
            best_overall = (cls, score)
        # Aggressive-class match regardless of threshold (threshold check is done by caller)
        if is_aggressive_sound(cls, score, 0.0):
            if best_aggressive is None or score > best_aggressive[1]:
                best_aggressive = (cls, score)

    return best_aggressive if best_aggressive is not None else best_overall


def run_yamnet_scan(interpreter, audio_np, class_names):
    """Legacy scan for explicitly int16 audio."""
    if not isinstance(audio_np, np.ndarray) or audio_np.dtype != np.int16:
        raise TypeError("run_yamnet_scan expects an int16 NumPy array")
    return _scan_windows(interpreter, audio_np, class_names, run_yamnet)


def scan_audio_float32(samples, sample_rate, interpreter, class_names):
    """Scan synchronized mono float32 event samples at YAMNet's 16 kHz rate."""
    if sample_rate != 16000:
        raise ValueError("YAMNet requires 16000 Hz audio")
    if not isinstance(samples, np.ndarray) or samples.dtype != np.float32:
        raise TypeError("scan_audio_float32 expects a float32 NumPy array")
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("YAMNet audio must be non-empty mono samples")
    return _scan_windows(
        interpreter,
        samples,
        class_names,
        run_yamnet_float32,
    )


def top_classes_float32(samples, sample_rate, interpreter, class_names, k=5):
    """YAMNet's k most likely classes over a whole utterance.

    Scores are averaged across 0.975 s windows, so a song under speech shows up
    even when no single window is dominated by it. Returns [(label, score)].
    """
    if sample_rate != 16000:
        raise ValueError("YAMNet requires 16000 Hz audio")
    if not isinstance(samples, np.ndarray) or samples.dtype != np.float32:
        raise TypeError("top_classes_float32 expects a float32 NumPy array")
    if samples.ndim != 1 or samples.size == 0 or not class_names:
        return []
    if samples.size < YAMNET_INPUT_SIZE:
        samples = np.pad(samples, (0, YAMNET_INPUT_SIZE - samples.size))
    windows = samples.size // YAMNET_INPUT_SIZE
    total = None
    for i in range(windows):
        window = samples[i * YAMNET_INPUT_SIZE:(i + 1) * YAMNET_INPUT_SIZE]
        _cls, _score, mean_scores = run_yamnet_float32(interpreter, window, class_names)
        total = mean_scores if total is None else total + mean_scores
    averaged = total / windows
    order = np.argsort(averaged)[::-1][:k]
    return [(class_names[i], float(averaged[i])) for i in order]
