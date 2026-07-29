# EchoSense Edge Device Configuration

import os


_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off", ""})


def parse_boolean(value, *, default=False):
    """Parse a conservative boolean configuration value.

    Unknown values fall back to ``default`` so a typo cannot accidentally
    enable privacy-sensitive transcript logging.
    """

    if value is None:
        return bool(default)
    normalized = str(value).strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return bool(default)


def show_transcript_text_from_environment(environ=None):
    """Return whether exact finalized transcript logging is enabled."""

    source = os.environ if environ is None else environ
    return parse_boolean(
        source.get("ECHOSENSE_SHOW_TRANSCRIPT_TEXT"),
        default=False,
    )


# Privacy-safe by default. Authorized diagnostics may enable this through the
# service environment without changing alert decisions or backend transmission.
SHOW_TRANSCRIPT_TEXT = show_transcript_text_from_environment()

# Backend API
API_URL = "https://echosense-backend-75h3.onrender.com"

# Audio settings
SAMPLE_RATE = 16000
CHUNK_SIZE = 1040
CHANNELS = 1

# YAMNet model
YAMNET_MODEL_PATH = "/home/echosense/echosense-edge/yamnet.tflite"
YAMNET_CLASSES_PATH = "/home/echosense/echosense-edge/yamnet_class_map.csv"

# Vosk models
VOSK_FILIPINO_MODEL = "/home/echosense/echosense-edge/vosk-model-tl-ph-generic-0.6"
VOSK_ENGLISH_MODEL = "/home/echosense/echosense-edge/vosk-model-small-en-us-0.15"

# Sensor location (included in every alert payload)
LOCATION = "Grade 6 Classroom"

# NOTE: Detection thresholds live in detection/thresholds.py (the authoritative
# file the runtime actually reads). The former YAMNET_CONFIDENCE_THRESHOLD and
# DURATION_THRESHOLD duplicates here were dead code and have been removed.
