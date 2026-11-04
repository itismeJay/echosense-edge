"""faster-whisper engine whose language detection is restricted to en/tl.

Stock auto-detect picks from all 99 Whisper languages, and Filipino/Bisaya
speech is often labelled jw/su/ms/nn. The edge then threw that decode away and
decoded again as Tagalog. Here the detection probabilities are read once and
only English and Tagalog compete, so every utterance is decoded exactly once.
Whisper has no Cebuano token; Bisaya is decoded as Tagalog, as the retry did.

RealtimeSTT runs the engine in a spawned child process. The child re-imports
``main.py``, which imports this module, so ``register()`` runs in both
processes. If registration is missing the recorder falls back to the stock
engine (see model/whisper_stt.py).
"""

from __future__ import annotations

import numpy as np

ENGINE_NAME = "echosense_faster_whisper_en_tl"
ALLOWED_LANGUAGES = ("en", "tl")
FALLBACK_LANGUAGE = "tl"
# ECHOSENSE_WHISPER_LANGUAGE values. "tl+en" decodes as Tagalog; the main
# process re-decodes as English only when Tagalog matched no monitored term.
LANGUAGE_MODES = ("auto", "tl", "en", "tl+en")
GAIN_MODES = ("off", "peak", "rms")
# rms gain: bring the loudest 0.5 s of speech to this level, at most x20.
RMS_TARGET = 0.1
MAX_GAIN = 20.0


def forced_language(mode):
    """The language a mode always decodes in first, or None for detection."""

    mode = str(mode or "auto").strip().lower()
    if mode in ("tl", "tl+en"):
        return "tl"
    if mode == "en":
        return "en"
    return None


def apply_gain(audio, mode="off", sample_rate=16000):
    """Loudness normalization before Whisper (far or quiet speech)."""

    mode = str(mode or "off").strip().lower()
    audio = np.asarray(audio, dtype=np.float32)
    if mode == "off" or audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak <= 1e-6:
        return audio
    if mode == "peak":
        return audio * (0.95 / peak)
    window = max(1, sample_rate // 2)
    hop = max(1, sample_rate // 10)
    loudest = max(
        float(np.sqrt(np.mean(audio[i:i + window] ** 2)))
        for i in range(0, max(1, audio.size - window + 1), hop)
    )
    if loudest <= 1e-6:
        return audio
    gain = min(MAX_GAIN, RMS_TARGET / loudest, 0.99 / peak * 4)
    if gain <= 1.0:
        return audio
    # Soft limiter so boosted transients do not hard-clip.
    return np.tanh(audio * gain).astype(np.float32)


def decode_restricted(
    whisper_model,
    audio,
    *,
    language=None,
    language_mode="auto",
    beam_size=1,
    initial_prompt=None,
    gain="off",
    suppress_tokens=(-1,),
    vad_filter=False,
    batch_model=None,
    batch_size=0,
):
    """One Whisper decode under the EchoSense language policy.

    Shared by the RealtimeSTT engine and tools/evaluate_testset.py so the
    offline experiment runs exactly the live decode. Returns (text,
    language, language_probability, segments).
    """

    audio = apply_gain(audio, gain)
    if language:
        chosen, probability = language, 1.0
    elif forced_language(language_mode):
        chosen, probability = forced_language(language_mode), 1.0
    else:
        try:
            _top, _probability, all_probs = whisper_model.detect_language(audio=audio)
            chosen, probability = choose_language(all_probs)
        except Exception:
            chosen, probability = FALLBACK_LANGUAGE, 0.0
    kwargs = {
        "language": chosen,
        "beam_size": beam_size,
        # Never a list of monitored words: that made Whisper "hear" them in
        # music, video audio and silence. A neutral context prompt is allowed.
        "initial_prompt": initial_prompt or None,
        "condition_on_previous_text": False,
        "suppress_tokens": list(suppress_tokens) if suppress_tokens is not None else None,
        "vad_filter": vad_filter,
        # Per-word probabilities feed the word-probability filter.
        "word_timestamps": True,
    }
    model = whisper_model
    if batch_model is not None and batch_size > 0:
        model = batch_model
        kwargs["batch_size"] = batch_size
        kwargs.pop("condition_on_previous_text")
    segments, _info = model.transcribe(audio, **kwargs)
    segments = list(segments)
    text = " ".join(segment.text for segment in segments).strip()
    return text, chosen, probability, segments


def choose_language(all_language_probs, allowed=ALLOWED_LANGUAGES):
    """Return (language, probability renormalized over ``allowed``)."""

    probabilities = {
        str(language): float(probability)
        for language, probability in all_language_probs or ()
        if language in allowed
    }
    total = sum(probabilities.values())
    if total <= 0:
        return FALLBACK_LANGUAGE, 0.0
    language = max(allowed, key=lambda item: probabilities.get(item, 0.0))
    return language, probabilities.get(language, 0.0) / total


def register():
    try:
        from RealtimeSTT.transcription_engines import factory
    except Exception:
        return False
    factory.ENGINE_CLASS_PATHS.setdefault(
        ENGINE_NAME,
        (__name__, "LanguageRestrictedFasterWhisperEngine"),
    )
    return True


def _whisper_model(model):
    """The WhisperModel itself, also when wrapped in a batched pipeline."""

    from faster_whisper import WhisperModel

    return model if isinstance(model, WhisperModel) else model.model


def _engine_class():
    from RealtimeSTT.transcription_engines.base import (
        TranscriptionInfo,
        TranscriptionResult,
    )
    from RealtimeSTT.transcription_engines.faster_whisper_engine import (
        FasterWhisperEngine,
    )

    from detection.false_alarm_filters import summarize_segments

    class LanguageRestrictedFasterWhisperEngine(FasterWhisperEngine):
        engine_name = ENGINE_NAME

        # RealtimeSTT 1.0.2 calls transcribe(audio, language=, use_prompt=)
        # and its base class takes no word_timestamps; this override decodes
        # directly so it no longer depends on the base signature.
        def transcribe(self, audio, language=None, use_prompt=True):
            from config import WHISPER_GAIN, WHISPER_LANGUAGE, WHISPER_PROMPT

            audio = self._normalize_audio(audio)
            text, chosen, probability, segments = decode_restricted(
                _whisper_model(self.model),
                audio,
                language=language,
                language_mode=WHISPER_LANGUAGE,
                beam_size=self.config.beam_size,
                initial_prompt=WHISPER_PROMPT if use_prompt else None,
                gain=WHISPER_GAIN,
                suppress_tokens=self.config.suppress_tokens,
                vad_filter=self.config.vad_filter,
                batch_model=self.model if self.config.batch_size > 0 else None,
                batch_size=self.config.batch_size,
            )
            result = TranscriptionResult(
                text=text,
                info=TranscriptionInfo(
                    language=chosen,
                    language_probability=probability,
                ),
            )
            # RealtimeSTT copies result.metadata to
            # recorder.last_transcription_metadata in the main process.
            result.metadata = summarize_segments(segments)
            return result

    return LanguageRestrictedFasterWhisperEngine


def __getattr__(name):
    if name == "LanguageRestrictedFasterWhisperEngine":
        return _engine_class()
    raise AttributeError(name)


register()
