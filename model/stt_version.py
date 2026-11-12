"""The speech-recognition model this edge really runs, as reported to the backend."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path

from config import WHISPER_GAIN, WHISPER_LANGUAGE, WHISPER_MODEL, WHISPER_PROMPT

# backend system_settings.stt_model_version is VARCHAR(50).
MAX_LENGTH = 50


def _package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "?"


def model_label(model=WHISPER_MODEL):
    """ "whisper small", or a local converted model's name without its owner."""

    text = str(model or "").strip()
    if "/" in text or "\\" in text:
        name = Path(text).name.removesuffix("-ct2-int8")
        return name.split("_", 1)[-1] or "unknown"
    return f"whisper {text}" if text else "unknown"


def stt_model_version(
    model=WHISPER_MODEL,
    language=WHISPER_LANGUAGE,
    prompt=WHISPER_PROMPT,
    gain=WHISPER_GAIN,
):
    """e.g. "whisper small int8 tl+en (faster-whisper 1.2.1)"."""

    parts = [model_label(model), "int8", language]
    if prompt:
        parts.append("prompt")
    if gain and gain != "off":
        parts.append(f"gain:{gain}")
    core = " ".join(parts)
    version = _package_version("faster-whisper")
    for suffix in (f" (faster-whisper {version})", f" (fw {version})", ""):
        if len(core) + len(suffix) <= MAX_LENGTH:
            return core + suffix
    return core[:MAX_LENGTH]


def yamnet_version():
    return f"YAMNet TFLite (ai-edge-litert {_package_version('ai-edge-litert')})"[:MAX_LENGTH]
