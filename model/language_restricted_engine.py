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
            audio = self._normalize_audio(audio)
            if language:
                chosen, probability = language, 1.0
            else:
                try:
                    _top, _probability, all_probs = _whisper_model(
                        self.model
                    ).detect_language(audio=audio)
                    chosen, probability = choose_language(all_probs)
                except Exception:
                    chosen, probability = FALLBACK_LANGUAGE, 0.0
            kwargs = {
                "language": chosen,
                "beam_size": self.config.beam_size,
                # No hint list: prompting Whisper with the monitored terms made
                # it "hear" them in music, video audio and silence. The
                # dictionary is used for matching only.
                "initial_prompt": None,
                "condition_on_previous_text": False,
                "suppress_tokens": self.config.suppress_tokens,
                "vad_filter": self.config.vad_filter,
                # Per-word probabilities feed the word-probability filter.
                "word_timestamps": True,
            }
            if self.config.batch_size > 0:
                kwargs["batch_size"] = self.config.batch_size
                kwargs.pop("condition_on_previous_text")
            segments, _info = self.model.transcribe(audio, **kwargs)
            segments = list(segments)
            result = TranscriptionResult(
                text=" ".join(segment.text for segment in segments).strip(),
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
