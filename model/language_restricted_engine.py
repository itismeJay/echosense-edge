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

ENGINE_NAME = "echosense_faster_whisper_en_tl"
ALLOWED_LANGUAGES = ("en", "tl")
FALLBACK_LANGUAGE = "tl"


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


def _engine_class():
    from faster_whisper import WhisperModel
    from RealtimeSTT.transcription_engines.base import (
        TranscriptionInfo,
        TranscriptionResult,
    )
    from RealtimeSTT.transcription_engines.faster_whisper_engine import (
        FasterWhisperEngine,
    )

    class LanguageRestrictedFasterWhisperEngine(FasterWhisperEngine):
        engine_name = ENGINE_NAME

        def transcribe(
            self, audio, language=None, use_prompt=True, word_timestamps=False,
        ):
            if language:
                return super().transcribe(
                    audio, language, use_prompt, word_timestamps,
                )
            model = self.model
            if not isinstance(model, WhisperModel):
                model = model.model  # BatchedInferencePipeline wrapper
            try:
                _top, _probability, all_probs = model.detect_language(
                    audio=self._normalize_audio(audio),
                )
                chosen, probability = choose_language(all_probs)
            except Exception:
                chosen, probability = FALLBACK_LANGUAGE, 0.0
            result = super().transcribe(
                audio, chosen, use_prompt, word_timestamps,
            )
            return TranscriptionResult(
                text=result.text,
                info=TranscriptionInfo(
                    language=chosen,
                    language_probability=probability,
                ),
                metadata=result.metadata,
            )

    return LanguageRestrictedFasterWhisperEngine


def __getattr__(name):
    if name == "LanguageRestrictedFasterWhisperEngine":
        return _engine_class()
    raise AttributeError(name)


register()
