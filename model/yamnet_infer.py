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

def run_yamnet(interpreter, audio_data, class_names):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    audio_float = audio_data.astype(np.float32) / 32768.0
    interpreter.set_tensor(input_details[0]['index'], audio_float)
    interpreter.invoke()
    scores = interpreter.get_tensor(output_details[0]['index'])
    mean_scores = np.mean(scores, axis=0)
    top_index = np.argmax(mean_scores)
    top_class = class_names[top_index]
    top_score = mean_scores[top_index]
    return top_class, top_score, mean_scores

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

def run_yamnet_scan(interpreter, audio_np, class_names):
    """Run YAMNet across a multi-second buffer by splitting it into 15600-sample
    windows. Returns (class, score) for the strongest AGGRESSIVE window if any
    aggressive class appears; otherwise the single highest-scoring window.
    Used by the 5-second sliding-window pipeline."""
    n = len(audio_np)
    if n < YAMNET_INPUT_SIZE:
        audio_np = np.pad(audio_np, (0, YAMNET_INPUT_SIZE - n))
        n = YAMNET_INPUT_SIZE

    num_windows = n // YAMNET_INPUT_SIZE
    best_overall = (class_names[0] if class_names else "Unknown", 0.0)
    best_aggressive = None

    for i in range(num_windows):
        window = audio_np[i * YAMNET_INPUT_SIZE:(i + 1) * YAMNET_INPUT_SIZE]
        cls, score, _ = run_yamnet(interpreter, window, class_names)
        score = float(score)
        if score > best_overall[1]:
            best_overall = (cls, score)
        # Aggressive-class match regardless of threshold (threshold check is done by caller)
        if is_aggressive_sound(cls, score, 0.0):
            if best_aggressive is None or score > best_aggressive[1]:
                best_aggressive = (cls, score)

    return best_aggressive if best_aggressive is not None else best_overall
