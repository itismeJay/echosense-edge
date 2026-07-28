import numpy as np
import csv
from ai_edge_litert.interpreter import Interpreter
from config import YAMNET_MODEL_PATH, YAMNET_CLASSES_PATH

def load_class_names():
    class_names = []
    with open(YAMNET_CLASSES_PATH) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            class_names.append(row[2])
    return class_names

def load_yamnet():
    interpreter = Interpreter(model_path=YAMNET_MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter


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
AGGRESSIVE_CLASSES = [
    "Screaming", "Scream",
    "Yell",      "Shout",
    "Crying",    "Whimper", "Wail"
]

# YAMNet's tflite graph expects exactly 15600 samples (0.975s @ 16kHz) per call.
YAMNET_INPUT_SIZE = 15600

def is_aggressive_sound(class_name: str, score: float, threshold: float) -> bool:
    if score < threshold:
        return False
    for aggressive in AGGRESSIVE_CLASSES:
        if aggressive.lower() in class_name.lower():
            return True
    return False

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
