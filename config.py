# EchoSense Edge Device Configuration

import os
from pathlib import Path


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


def _environment_float(name, default, *, minimum=None):
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = float(default)
    if minimum is not None:
        value = max(float(minimum), value)
    return value


def _environment_int(name, default, *, minimum=None):
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = int(default)
    if minimum is not None:
        value = max(int(minimum), value)
    return value


_PROJECT_ROOT = Path(__file__).resolve().parent
_configured_outbox_path = Path(
    os.environ.get(
        "ECHOSENSE_OUTBOX_DATABASE_PATH",
        "data/alert_outbox.sqlite3",
    )
).expanduser()
if not _configured_outbox_path.is_absolute():
    _configured_outbox_path = _PROJECT_ROOT / _configured_outbox_path

# Persistent offline alert outbox. Remote synchronization still requires
# connectivity; these settings control resilient local retry behavior.
OUTBOX_ENABLED = parse_boolean(
    os.environ.get("ECHOSENSE_OUTBOX_ENABLED"),
    default=True,
)
OUTBOX_DATABASE_PATH = str(_configured_outbox_path.resolve())
OUTBOX_POLL_INTERVAL_SECONDS = _environment_float(
    "ECHOSENSE_OUTBOX_POLL_INTERVAL_SECONDS",
    1.0,
    minimum=0.1,
)
OUTBOX_REQUEST_TIMEOUT_SECONDS = _environment_float(
    "ECHOSENSE_OUTBOX_REQUEST_TIMEOUT_SECONDS",
    10.0,
    minimum=0.5,
)
OUTBOX_INITIAL_RETRY_DELAY_SECONDS = _environment_float(
    "ECHOSENSE_OUTBOX_INITIAL_RETRY_DELAY_SECONDS",
    5.0,
    minimum=0.1,
)
OUTBOX_BACKOFF_MULTIPLIER = _environment_float(
    "ECHOSENSE_OUTBOX_BACKOFF_MULTIPLIER",
    3.0,
    minimum=1.0,
)
OUTBOX_MAX_RETRY_DELAY_SECONDS = _environment_float(
    "ECHOSENSE_OUTBOX_MAX_RETRY_DELAY_SECONDS",
    300.0,
    minimum=1.0,
)
OUTBOX_JITTER_PERCENT = min(
    1.0,
    _environment_float(
        "ECHOSENSE_OUTBOX_JITTER_PERCENT",
        0.20,
        minimum=0.0,
    ),
)
OUTBOX_STALE_SENDING_SECONDS = _environment_float(
    "ECHOSENSE_OUTBOX_STALE_SENDING_SECONDS",
    60.0,
    minimum=1.0,
)
OUTBOX_DELIVERED_RETENTION_DAYS = _environment_float(
    "ECHOSENSE_OUTBOX_DELIVERED_RETENTION_DAYS",
    30.0,
    minimum=0.0,
)
OUTBOX_MAX_RECORDS_PER_CYCLE = _environment_int(
    "ECHOSENSE_OUTBOX_MAX_RECORDS_PER_CYCLE",
    10,
    minimum=1,
)
OUTBOX_BUSY_TIMEOUT_MS = _environment_int(
    "ECHOSENSE_OUTBOX_BUSY_TIMEOUT_MS",
    5000,
    minimum=100,
)
OUTBOX_HEALTH_LOG_INTERVAL_SECONDS = _environment_float(
    "ECHOSENSE_OUTBOX_HEALTH_LOG_INTERVAL_SECONDS",
    60.0,
    minimum=5.0,
)

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
