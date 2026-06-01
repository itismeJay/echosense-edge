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

AGGRESSIVE_CLASSES = [
    "Screaming", "Scream", "Yell",
    "Crying", "Whimper", "Wail",
    "Crowd", "Noise", "Shout"
]

def is_aggressive_sound(class_name: str, score: float, threshold: float) -> bool:
    if score < threshold:
        return False
    for aggressive in AGGRESSIVE_CLASSES:
        if aggressive.lower() in class_name.lower():
            return True    return False
