# EchoSense Edge Device Configuration

import os
from dataclasses import dataclass
import math
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
def backend_base_url_from_environment(environ=None):
    """Load the explicit backend target without a production fallback."""

    source = os.environ if environ is None else environ
    return str(
        source.get("ECHOSENSE_BACKEND_BASE_URL") or ""
    ).strip().rstrip("/")


API_URL = backend_base_url_from_environment()


def device_credentials_from_environment(environ=None):
    """Load edge credentials without logging or persisting either value."""

    source = os.environ if environ is None else environ
    device_id = str(source.get("ECHOSENSE_DEVICE_ID") or "").strip()
    device_key = str(source.get("ECHOSENSE_DEVICE_KEY") or "").strip()
    return device_id, device_key


ECHOSENSE_DEVICE_ID, ECHOSENSE_DEVICE_KEY = (
    device_credentials_from_environment()
)


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


def _source_float(source, name, default):
    try:
        return float(source.get(name, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc


def _source_int(source, name, default):
    try:
        return int(source.get(name, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


ACOUSTIC_ONLY_ALERT_MODES = ("on", "low_only", "off")


@dataclass(frozen=True, slots=True)
class DetectionRuntimeConfig:
    """Validated bounds and timing for the shared AFTER detection runtime."""

    sample_rate: int = 16_000
    channels: int = 1
    frame_samples: int = 1_040
    pre_trigger_seconds: float = 2.0
    # Incident close window: an incident finalizes once no qualifying evidence
    # has arrived for this long (ECHOSENSE_INCIDENT_WINDOW_SECONDS).
    post_trigger_seconds: float = 3.0
    # Only text (or time-linked strong acoustic) evidence extends an incident.
    event_extension_seconds: float = 2.0
    max_event_seconds: float = 8.0
    cooldown_seconds: float = 15.0
    audio_queue_frames: int = 64
    stt_result_queue_size: int = 32
    stt_internal_audio_chunks: int = 128
    # Safety/resource bound for RAM-only finalized STT evidence. This is not
    # an accuracy or aggression-detection threshold.
    max_stt_utterance_seconds: float = 15.0
    evidence_history_size: int = 128
    yamnet_window_samples: int = 15_600
    yamnet_hop_samples: int = 7_800
    waveform_points: int = 40
    # Adaptive acoustic noise floor. Elementary classrooms are loud all the
    # time, so "aggressive tone" is judged relative to the room, not absolute.
    noise_floor_window_seconds: float = 60.0
    noise_floor_percentile: float = 60.0
    tone_floor_multiplier: float = 2.5
    calibration_seconds: float = 15.0
    noise_floor_log_interval_seconds: float = 30.0
    # Text/acoustic fusion. Only acoustic windows within this many seconds of
    # a text observation may change that text's severity.
    corroboration_window_seconds: float = 1.5
    strong_yamnet_score: float = 0.65
    strong_tone_ratio: float = 0.40
    strong_tone_min_seconds: float = 1.0
    # Acoustic-only incidents: "on", "low_only", or "off".
    acoustic_only_alerts: str = "low_only"
    acoustic_only_medium_min_seconds: float = 3.0

    def __post_init__(self):
        if self.sample_rate != 16_000:
            raise ValueError("canonical sample_rate must be 16000 Hz")
        if self.channels != 1:
            raise ValueError("canonical audio must be mono")
        if not 0 < self.frame_samples <= 15_600:
            raise ValueError("frame_samples must be between 1 and 15600")
        for name in (
            "pre_trigger_seconds",
            "post_trigger_seconds",
            "event_extension_seconds",
            "cooldown_seconds",
        ):
            value = getattr(self, name)
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            if value > 300:
                raise ValueError(f"{name} exceeds its safe maximum")
        if not math.isfinite(float(self.max_event_seconds)):
            raise ValueError("max_event_seconds must be finite")
        if not 0 < self.max_event_seconds <= 60:
            raise ValueError("max_event_seconds must be between 0 and 60")
        if not math.isfinite(float(self.max_stt_utterance_seconds)):
            raise ValueError("max_stt_utterance_seconds must be finite")
        if not 0 < self.max_stt_utterance_seconds <= 30:
            raise ValueError(
                "max_stt_utterance_seconds must be between 0 and 30"
            )
        if self.pre_trigger_seconds > 30:
            raise ValueError("pre_trigger_seconds exceeds its safe maximum")
        for name in (
            "audio_queue_frames",
            "stt_result_queue_size",
            "stt_internal_audio_chunks",
            "evidence_history_size",
            "yamnet_window_samples",
            "yamnet_hop_samples",
            "waveform_points",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        capacity_maxima = {
            "audio_queue_frames": 512,
            "stt_result_queue_size": 64,
            "stt_internal_audio_chunks": 2_048,
            "evidence_history_size": 1_024,
            "waveform_points": 256,
        }
        for name, maximum in capacity_maxima.items():
            if getattr(self, name) > maximum:
                raise ValueError(f"{name} exceeds its safe maximum")
        if self.yamnet_hop_samples > self.yamnet_window_samples:
            raise ValueError("yamnet_hop_samples cannot exceed its window")
        if self.yamnet_window_samples != 15_600:
            raise ValueError("YAMNet window must be exactly 15600 samples")
        bounded = {
            "noise_floor_window_seconds": (1.0, 3_600.0),
            "noise_floor_percentile": (50.0, 70.0),
            "tone_floor_multiplier": (1.0, 20.0),
            "calibration_seconds": (0.0, 300.0),
            "noise_floor_log_interval_seconds": (1.0, 3_600.0),
            "corroboration_window_seconds": (0.0, 30.0),
            "strong_yamnet_score": (0.0, 1.0),
            "strong_tone_ratio": (0.0, 1.0),
            "strong_tone_min_seconds": (0.0, 60.0),
            "acoustic_only_medium_min_seconds": (0.0, 60.0),
        }
        for name, (minimum, maximum) in bounded.items():
            value = float(getattr(self, name))
            if not math.isfinite(value) or not minimum <= value <= maximum:
                raise ValueError(
                    f"{name} must be between {minimum:g} and {maximum:g}"
                )
        if self.acoustic_only_alerts not in ACOUSTIC_ONLY_ALERT_MODES:
            raise ValueError(
                "acoustic_only_alerts must be one of "
                + ", ".join(ACOUSTIC_ONLY_ALERT_MODES)
            )

    @property
    def max_stt_utterance_samples(self):
        return int(round(
            self.sample_rate * self.max_stt_utterance_seconds
        ))

    @property
    def max_stt_result_pcm_bytes(self):
        return self.max_stt_utterance_samples * 4

    @property
    def max_stt_result_queue_pcm_bytes(self):
        return self.stt_result_queue_size * self.max_stt_result_pcm_bytes


def detection_runtime_config_from_environment(environ=None):
    """Build one validated runtime configuration from ECHOSENSE_* values."""

    source = os.environ if environ is None else environ
    # ECHOSENSE_INCIDENT_WINDOW_SECONDS is the close window after an incident
    # opens; the older ECHOSENSE_POST_TRIGGER_SECONDS still wins when set.
    incident_window = _source_float(
        source, "ECHOSENSE_INCIDENT_WINDOW_SECONDS", 3.0
    )
    return DetectionRuntimeConfig(
        pre_trigger_seconds=_source_float(
            source, "ECHOSENSE_PRE_TRIGGER_SECONDS", 2.0
        ),
        post_trigger_seconds=_source_float(
            source, "ECHOSENSE_POST_TRIGGER_SECONDS", incident_window
        ),
        event_extension_seconds=_source_float(
            source, "ECHOSENSE_EVENT_EXTENSION_SECONDS", 2.0
        ),
        max_event_seconds=_source_float(
            source, "ECHOSENSE_MAX_EVENT_SECONDS", 8.0
        ),
        cooldown_seconds=_source_float(
            source, "ECHOSENSE_EVENT_COOLDOWN_SECONDS", 15.0
        ),
        audio_queue_frames=_source_int(
            source, "ECHOSENSE_AUDIO_QUEUE_FRAMES", 64
        ),
        stt_result_queue_size=_source_int(
            source, "ECHOSENSE_STT_RESULT_QUEUE_SIZE", 32
        ),
        stt_internal_audio_chunks=_source_int(
            source, "ECHOSENSE_STT_INTERNAL_AUDIO_CHUNKS", 128
        ),
        max_stt_utterance_seconds=_source_float(
            source, "ECHOSENSE_MAX_STT_UTTERANCE_SECONDS", 15.0
        ),
        evidence_history_size=_source_int(
            source, "ECHOSENSE_EVIDENCE_HISTORY_SIZE", 128
        ),
        yamnet_window_samples=_source_int(
            source, "ECHOSENSE_YAMNET_WINDOW_SAMPLES", 15_600
        ),
        yamnet_hop_samples=_source_int(
            source, "ECHOSENSE_YAMNET_HOP_SAMPLES", 7_800
        ),
        waveform_points=_source_int(
            source, "ECHOSENSE_WAVEFORM_POINTS", 40
        ),
        noise_floor_window_seconds=_source_float(
            source, "ECHOSENSE_NOISE_FLOOR_WINDOW_SECONDS", 60.0
        ),
        tone_floor_multiplier=_source_float(
            source, "ECHOSENSE_TONE_FLOOR_MULTIPLIER", 2.5
        ),
        calibration_seconds=_source_float(
            source, "ECHOSENSE_CALIBRATION_SECONDS", 15.0
        ),
        corroboration_window_seconds=_source_float(
            source, "ECHOSENSE_CORROBORATION_WINDOW_SECONDS", 1.5
        ),
        strong_tone_ratio=_source_float(
            source, "ECHOSENSE_STRONG_TONE_RATIO", 0.40
        ),
        acoustic_only_alerts=str(
            source.get("ECHOSENSE_ACOUSTIC_ONLY_ALERTS", "low_only")
        ).strip().lower(),
    )


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

# Shared audio and incident settings. These defaults are development starting
# values and remain environment-configurable for later classroom tuning.
DETECTION_RUNTIME = detection_runtime_config_from_environment()
SAMPLE_RATE = DETECTION_RUNTIME.sample_rate
CHUNK_SIZE = DETECTION_RUNTIME.frame_samples
CHANNELS = DETECTION_RUNTIME.channels
PRE_TRIGGER_SECONDS = DETECTION_RUNTIME.pre_trigger_seconds
POST_TRIGGER_SECONDS = DETECTION_RUNTIME.post_trigger_seconds
EVENT_EXTENSION_SECONDS = DETECTION_RUNTIME.event_extension_seconds
MAX_EVENT_SECONDS = DETECTION_RUNTIME.max_event_seconds
EVENT_COOLDOWN_SECONDS = DETECTION_RUNTIME.cooldown_seconds

# One application-owned microphone. The preference can be replaced by name or
# disabled with an empty value. Strict mode refuses unrelated fallback devices.
PREFERRED_MICROPHONE_NAME = str(
    os.environ.get("ECHOSENSE_PREFERRED_MICROPHONE", "EMEET M0 Plus")
).strip()
MICROPHONE_STRICT = parse_boolean(
    os.environ.get("ECHOSENSE_MICROPHONE_STRICT"),
    default=True,
)

# Repository-relative by default, with explicit environment overrides for Pi
# deployment. Model binaries remain ignored and are never downloaded at runtime.
_configured_yamnet_path = Path(
    os.environ.get("ECHOSENSE_YAMNET_MODEL_PATH", "yamnet.tflite")
).expanduser()
_configured_yamnet_classes_path = Path(
    os.environ.get(
        "ECHOSENSE_YAMNET_CLASSES_PATH",
        "yamnet_class_map.csv",
    )
).expanduser()
if not _configured_yamnet_path.is_absolute():
    _configured_yamnet_path = _PROJECT_ROOT / _configured_yamnet_path
if not _configured_yamnet_classes_path.is_absolute():
    _configured_yamnet_classes_path = (
        _PROJECT_ROOT / _configured_yamnet_classes_path
    )
YAMNET_MODEL_PATH = str(_configured_yamnet_path.resolve())
YAMNET_CLASSES_PATH = str(_configured_yamnet_classes_path.resolve())
ACOUSTIC_DEGRADED_MODE = parse_boolean(
    os.environ.get("ECHOSENSE_ACOUSTIC_DEGRADED_MODE"),
    default=True,
)

# Vosk models
VOSK_FILIPINO_MODEL = "/home/echosense/echosense-edge/vosk-model-tl-ph-generic-0.6"
VOSK_ENGLISH_MODEL = "/home/echosense/echosense-edge/vosk-model-small-en-us-0.15"

# Deprecated compatibility label. Schema-v2 delivery never sends this value;
# the backend derives classroom and school from the authenticated device.
LOCATION = "Grade 6 Classroom"

# NOTE: Detection thresholds live in detection/thresholds.py (the authoritative
# file the runtime actually reads). The former YAMNET_CONFIDENCE_THRESHOLD and
# DURATION_THRESHOLD duplicates here were dead code and have been removed.
