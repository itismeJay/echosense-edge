# EchoSense Edge Device Configuration

import os
from dataclasses import dataclass
import math
from pathlib import Path
import sys


_PROJECT_ROOT = Path(__file__).resolve().parent


def parse_env_file(text):
    """Parse KEY=value lines (optionally prefixed with ``export``)."""

    values = {}
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not key.isidentifier():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def load_edge_env(path=None, environ=None):
    """Load edge.env into the environment without overriding set variables.

    The same commit runs on the Mac and the Pi; only edge.env differs.
    ECHOSENSE_ENV_FILE picks another file, and an empty value disables it.
    Variables already exported (systemd, the demo scripts) always win.
    """

    target = os.environ if environ is None else environ
    if path is None:
        path = target.get("ECHOSENSE_ENV_FILE", "edge.env")
    if not str(path).strip():
        return {}
    env_path = Path(path).expanduser()
    if not env_path.is_absolute():
        env_path = _PROJECT_ROOT / env_path
    try:
        values = parse_env_file(env_path.read_text(encoding="utf-8"))
    except OSError:
        return {}
    for key, value in values.items():
        target.setdefault(key, value)
    return values


load_edge_env()


def detect_platform(environ=None):
    """Return "macos", "raspberrypi" or "linux" (ECHOSENSE_PLATFORM overrides)."""

    source = os.environ if environ is None else environ
    override = str(source.get("ECHOSENSE_PLATFORM") or "").strip().lower()
    if override:
        return override
    if sys.platform == "darwin":
        return "macos"
    try:
        model = Path("/proc/device-tree/model").read_bytes()
    except OSError:
        model = b""
    if b"Raspberry Pi" in model:
        return "raspberrypi"
    return "linux"


PLATFORM = detect_platform()


def project_path(value):
    """Resolve a configured path; relative paths are under the repo root."""

    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return path.resolve()


def _default_hostname():
    import socket

    return socket.gethostname().split(".")[0] or "unknown"


# Network identity reported in the log banner and heartbeat.
NETWORK_INTERFACE = str(
    os.environ.get("ECHOSENSE_NETWORK_INTERFACE")
    or ("en0" if PLATFORM == "macos" else "wlan0")
).strip()
DEVICE_HOSTNAME = str(
    os.environ.get("ECHOSENSE_HOSTNAME") or _default_hostname()
).strip()

DATA_DIR = project_path(os.environ.get("ECHOSENSE_DATA_DIR") or "data")
LOG_DIR = project_path(os.environ.get("ECHOSENSE_LOG_DIR") or "logs")
SHADOW_LOG_PATH = str(LOG_DIR / "shadow_log.jsonl")


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


# QR setup (see provisioning/): identity.json comes from tools/provision_box.py; the
# box saves its device key to device_credentials.json when an admin adds it to a room.
# Without an identity file the manual ECHOSENSE_DEVICE_ID/KEY flow is unchanged.
IDENTITY_FILE = project_path(
    os.environ.get("ECHOSENSE_IDENTITY_FILE") or (DATA_DIR / "identity.json")
)
DEVICE_CREDENTIALS_FILE = project_path(
    os.environ.get("ECHOSENSE_DEVICE_CREDENTIALS_FILE")
    or (IDENTITY_FILE.parent / "device_credentials.json")
)
ACTIVATION_POLL_SECONDS = max(
    1.0, float(os.environ.get("ECHOSENSE_ACTIVATION_POLL_SECONDS") or 10)
)
ACTIVATION_ANNOUNCE_SECONDS = max(
    30.0, float(os.environ.get("ECHOSENSE_ACTIVATION_ANNOUNCE_SECONDS") or 180)
)
SPEAK_COMMAND = os.environ.get("ECHOSENSE_SPEAK_COMMAND")


def resolve_device_credentials(environ=None):
    from provisioning.identity import resolve_credentials

    source = os.environ if environ is None else environ
    return resolve_credentials(source, IDENTITY_FILE, DEVICE_CREDENTIALS_FILE)


DEVICE_CREDENTIALS = resolve_device_credentials()
ECHOSENSE_DEVICE_ID = DEVICE_CREDENTIALS.device_id
ECHOSENSE_DEVICE_KEY = DEVICE_CREDENTIALS.device_key


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


@dataclass(frozen=True, slots=True)
class DetectionRuntimeConfig:
    """Validated bounds and timing for the shared AFTER detection runtime."""

    sample_rate: int = 16_000
    channels: int = 1
    frame_samples: int = 1_040
    pre_trigger_seconds: float = 2.0
    # Incident close window: an incident finalizes once no qualifying evidence
    # has arrived for this long (ECHOSENSE_INCIDENT_WINDOW_SECONDS).
    post_trigger_seconds: float = 1.5
    # Only text (or time-linked strong acoustic) evidence extends an incident.
    event_extension_seconds: float = 1.5
    max_event_seconds: float = 6.0
    # There is no global cooldown. After a text incident closes, the SAME
    # terms at the same-or-lower severity within this window are logged as a
    # duplicate of it instead of alerting again. 0 disables the guard.
    duplicate_window_seconds: float = 10.0
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
    # Absolute loudness floor for "aggressive tone" (int16 RMS). In a quiet
    # room floor*multiplier is low enough that normal speech would cross it.
    tone_min_absolute_rms: float = 400.0
    calibration_seconds: float = 15.0
    noise_floor_log_interval_seconds: float = 30.0
    # Text/acoustic fusion. Only acoustic windows within this many seconds of
    # a text observation may change that text's severity.
    corroboration_window_seconds: float = 1.5
    strong_yamnet_score: float = 0.65
    strong_tone_ratio: float = 0.40
    strong_tone_min_seconds: float = 1.0

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
            "duplicate_window_seconds",
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
            "tone_min_absolute_rms": (0.0, 32_768.0),
            "calibration_seconds": (0.0, 300.0),
            "noise_floor_log_interval_seconds": (1.0, 3_600.0),
            "corroboration_window_seconds": (0.0, 30.0),
            "strong_yamnet_score": (0.0, 1.0),
            "strong_tone_ratio": (0.0, 1.0),
            "strong_tone_min_seconds": (0.0, 60.0),
        }
        for name, (minimum, maximum) in bounded.items():
            value = float(getattr(self, name))
            if not math.isfinite(value) or not minimum <= value <= maximum:
                raise ValueError(
                    f"{name} must be between {minimum:g} and {maximum:g}"
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


@dataclass(frozen=True, slots=True)
class SemanticRuntimeConfig:
    """Resource bounds for the optional, shadow-only semantic classifier."""

    enabled: bool = False
    min_confidence: float = 0.75
    max_context_turns: int = 6
    context_window_seconds: float = 90.0
    max_context_chars: int = 4_000

    def __post_init__(self):
        if not 0.0 <= float(self.min_confidence) <= 1.0:
            raise ValueError("semantic min_confidence must be between 0 and 1")
        if not 1 <= int(self.max_context_turns) <= 64:
            raise ValueError("semantic max_context_turns must be between 1 and 64")
        if not 0.0 <= float(self.context_window_seconds) <= 3_600.0:
            raise ValueError("semantic context_window_seconds is unsafe")
        if not 1 <= int(self.max_context_chars) <= 50_000:
            raise ValueError("semantic max_context_chars is unsafe")


def semantic_runtime_config_from_environment(environ=None):
    """Load an explicitly opt-in semantic shadow configuration."""

    source = os.environ if environ is None else environ
    return SemanticRuntimeConfig(
        enabled=parse_boolean(
            source.get("ECHOSENSE_EXPERIMENTAL_SEMANTICS"),
            default=False,
        ),
        min_confidence=_source_float(
            source, "ECHOSENSE_SEMANTIC_MIN_CONFIDENCE", 0.75
        ),
        max_context_turns=_source_int(
            source, "ECHOSENSE_SEMANTIC_CONTEXT_TURNS", 6
        ),
        context_window_seconds=_source_float(
            source, "ECHOSENSE_SEMANTIC_CONTEXT_SECONDS", 90.0
        ),
        max_context_chars=_source_int(
            source, "ECHOSENSE_SEMANTIC_CONTEXT_CHARS", 4_000
        ),
    )


def detection_runtime_config_from_environment(environ=None):
    """Build one validated runtime configuration from ECHOSENSE_* values."""

    source = os.environ if environ is None else environ
    # ECHOSENSE_INCIDENT_WINDOW_SECONDS is the close window after an incident
    # opens; the older ECHOSENSE_POST_TRIGGER_SECONDS still wins when set.
    incident_window = _source_float(
        source, "ECHOSENSE_INCIDENT_WINDOW_SECONDS", 1.5
    )
    if source.get("ECHOSENSE_ACOUSTIC_ONLY_ALERTS") not in (None, ""):
        print(
            "[CONFIG] ECHOSENSE_ACOUSTIC_ONLY_ALERTS is ignored: "
            "sound alone never sends an alert"
        )
    return DetectionRuntimeConfig(
        pre_trigger_seconds=_source_float(
            source, "ECHOSENSE_PRE_TRIGGER_SECONDS", 2.0
        ),
        post_trigger_seconds=_source_float(
            source, "ECHOSENSE_POST_TRIGGER_SECONDS", incident_window
        ),
        event_extension_seconds=_source_float(
            source, "ECHOSENSE_EVENT_EXTENSION_SECONDS", 1.5
        ),
        max_event_seconds=_source_float(
            source, "ECHOSENSE_MAX_EVENT_SECONDS", 6.0
        ),
        duplicate_window_seconds=_source_float(
            source, "ECHOSENSE_DUPLICATE_WINDOW_SECONDS", 10.0
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
        tone_min_absolute_rms=_source_float(
            source, "ECHOSENSE_TONE_MIN_ABSOLUTE_RMS", 400.0
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
    )


# Persistent offline alert outbox. Remote synchronization still requires
# connectivity; these settings control resilient local retry behavior.
OUTBOX_ENABLED = parse_boolean(
    os.environ.get("ECHOSENSE_OUTBOX_ENABLED"),
    default=True,
)
OUTBOX_DATABASE_PATH = str(project_path(
    os.environ.get("ECHOSENSE_OUTBOX_DATABASE_PATH")
    or DATA_DIR / "alert_outbox.sqlite3"
))
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

# Explicit, fail-closed delivery mode for detector-generated dashboard tests.
# The operator runner sets these values for one foreground process; normal edge
# startup remains a real-alert runtime and live_validation.py overrides the mode
# back to false.  The reviewer token is memory-only and is never logged or stored
# in the outbox.
DASHBOARD_TEST_MODE = parse_boolean(
    os.environ.get("ECHOSENSE_DASHBOARD_TEST_MODE"),
    default=False,
)
DASHBOARD_TEST_EXPECTED_DEVICE_ID = str(
    os.environ.get("ECHOSENSE_DASHBOARD_TEST_EXPECTED_DEVICE_ID") or ""
).strip()
DASHBOARD_TEST_EXPECTED_BACKEND_ORIGIN = str(
    os.environ.get("ECHOSENSE_DASHBOARD_TEST_EXPECTED_BACKEND_ORIGIN") or ""
).strip().rstrip("/")
DASHBOARD_TEST_REVIEWER_TOKEN = str(
    os.environ.get("ECHOSENSE_DASHBOARD_TEST_REVIEWER_TOKEN") or ""
).strip()
DASHBOARD_TEST_VERIFY_TIMEOUT_SECONDS = _environment_float(
    "ECHOSENSE_DASHBOARD_TEST_VERIFY_TIMEOUT_SECONDS",
    12.0,
    minimum=1.0,
)
DASHBOARD_TEST_VERIFY_POLL_SECONDS = _environment_float(
    "ECHOSENSE_DASHBOARD_TEST_VERIFY_POLL_SECONDS",
    0.5,
    minimum=0.1,
)

# Shared audio and incident settings. These defaults are development starting
# values and remain environment-configurable for later classroom tuning.
DETECTION_RUNTIME = detection_runtime_config_from_environment()
SEMANTIC_RUNTIME = semantic_runtime_config_from_environment()
SAMPLE_RATE = DETECTION_RUNTIME.sample_rate
CHUNK_SIZE = DETECTION_RUNTIME.frame_samples
CHANNELS = DETECTION_RUNTIME.channels
PRE_TRIGGER_SECONDS = DETECTION_RUNTIME.pre_trigger_seconds
POST_TRIGGER_SECONDS = DETECTION_RUNTIME.post_trigger_seconds
EVENT_EXTENSION_SECONDS = DETECTION_RUNTIME.event_extension_seconds
MAX_EVENT_SECONDS = DETECTION_RUNTIME.max_event_seconds
DUPLICATE_WINDOW_SECONDS = DETECTION_RUNTIME.duplicate_window_seconds

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
YAMNET_MODEL_PATH = str(project_path(
    os.environ.get("ECHOSENSE_YAMNET_MODEL_PATH") or "yamnet.tflite"
))
YAMNET_CLASSES_PATH = str(project_path(
    os.environ.get("ECHOSENSE_YAMNET_CLASSES_PATH") or "yamnet_class_map.csv"
))
ACOUSTIC_DEGRADED_MODE = parse_boolean(
    os.environ.get("ECHOSENSE_ACOUSTIC_DEGRADED_MODE"),
    default=True,
)

# Whisper model for faster-whisper: a size name ("base") or a local model dir.
# A path (anything with a slash) is resolved against the repo root, so
# "models/sapinsapin_whisper-small-pld-ceb-ct2-int8" works on Mac and Pi.
# Pick by Bisaya recall: tools/evaluate_testset.py ranks configs that way.
_whisper_model = str(os.environ.get("ECHOSENSE_WHISPER_MODEL") or "base").strip()
WHISPER_MODEL = (
    str(project_path(_whisper_model))
    if "/" in _whisper_model or "\\" in _whisper_model
    else _whisper_model
)
# Whisper decode policy (model/language_restricted_engine.py):
#   tl     always Tagalog (closest Whisper language to Bisaya; default,
#          best Bisaya recall on every model tested)
#   auto   detect, restricted to English/Tagalog
#   en     always English
#   tl+en  Tagalog, then English only when Tagalog matched no monitored term
WHISPER_LANGUAGE = str(
    os.environ.get("ECHOSENSE_WHISPER_LANGUAGE") or "tl"
).strip().lower()
# Neutral context prompt. Never put monitored words here: Whisper then
# "hears" them in music and silence.
WHISPER_PROMPT = str(os.environ.get("ECHOSENSE_WHISPER_PROMPT") or "").strip()
# Gain before Whisper: off, peak (normalize peak) or rms (speech loudness).
WHISPER_GAIN = str(os.environ.get("ECHOSENSE_WHISPER_GAIN") or "off").strip().lower()
# Sound-alike correction of Whisper mishearings before term matching.
SOUNDALIKE_MATCH = parse_boolean(
    os.environ.get("ECHOSENSE_SOUNDALIKE_MATCH"), default=True,
)

# Legacy Vosk models (model/vosk_detect.py, not used by the live runtime)
VOSK_FILIPINO_MODEL = str(project_path(
    os.environ.get("ECHOSENSE_VOSK_FILIPINO_MODEL_PATH")
    or "vosk-model-tl-ph-generic-0.6"
))
VOSK_ENGLISH_MODEL = str(project_path(
    os.environ.get("ECHOSENSE_VOSK_ENGLISH_MODEL_PATH")
    or "vosk-model-small-en-us-0.15"
))

# Deprecated compatibility label. Schema-v2 delivery never sends this value;
# the backend derives classroom and school from the authenticated device.
LOCATION = "Grade 6 Classroom"

# NOTE: Detection thresholds live in detection/thresholds.py (the authoritative
# file the runtime actually reads). The former YAMNET_CONFIDENCE_THRESHOLD and
# DURATION_THRESHOLD duplicates here were dead code and have been removed.
