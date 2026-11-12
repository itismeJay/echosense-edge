"""faster-whisper engine whose language detection is restricted to en/tl.

Stock auto-detect picks from all 99 Whisper languages, and Filipino/Bisaya
speech is often labelled jw/su/ms/nn. The edge then threw that decode away and
decoded again as Tagalog. Here the detection probabilities are read once and
only English and Tagalog compete, so every utterance is decoded exactly once.
Whisper has no Cebuano token; Bisaya is decoded as Tagalog, as the retry did.

RealtimeSTT 1.0.2 does not expose a process-safe custom-engine registry. Its
supported ``transcription_executor`` hook is therefore used instead. The
executor below owns one spawned model worker and proxies RealtimeSTT requests
to it, preserving process isolation without mutating RealtimeSTT internals.
"""

from __future__ import annotations

import multiprocessing
import threading

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


class RestrictedWhisperStartupError(RuntimeError):
    """The isolated restricted-language model worker could not start."""


def _restricted_whisper_worker(connection, engine_config):
    """Own the faster-whisper model in one importable ``spawn`` target."""

    try:
        from RealtimeSTT.transcription_engines.base import TranscriptionEngineConfig

        engine = _engine_class()(TranscriptionEngineConfig(**engine_config))
    except Exception as exc:
        try:
            connection.send(("startup_error", type(exc).__name__, str(exc)))
        except (BrokenPipeError, EOFError, OSError):
            pass
        connection.close()
        return

    try:
        connection.send(("ready",))
        while True:
            try:
                message = connection.recv()
            except EOFError:
                break
            if not message or message[0] == "shutdown":
                break
            operation, request_id, audio, language, use_prompt = message
            if operation != "transcribe":
                connection.send((
                    "error",
                    request_id,
                    "ValueError",
                    f"unknown worker operation: {operation}",
                ))
                continue
            try:
                result = engine.transcribe(
                    audio,
                    language=language,
                    use_prompt=use_prompt,
                )
                connection.send((
                    "result",
                    request_id,
                    result.text,
                    result.info.language,
                    result.info.language_probability,
                    getattr(result, "metadata", None),
                ))
            except Exception as exc:
                connection.send((
                    "error",
                    request_id,
                    type(exc).__name__,
                    str(exc),
                ))
    finally:
        connection.close()


class RestrictedWhisperExecutor:
    """RealtimeSTT executor backed by a restricted-language spawned worker."""

    def __init__(
        self,
        *,
        model,
        device="cpu",
        compute_type="int8",
        gpu_device_index=0,
        beam_size=1,
        initial_prompt=None,
        suppress_tokens=(-1,),
        batch_size=0,
        vad_filter=False,
        normalize_audio=False,
        startup_timeout=60.0,
        request_timeout=120.0,
        multiprocessing_context=None,
    ):
        self._context = (
            multiprocessing_context or multiprocessing.get_context("spawn")
        )
        self._lock = threading.Lock()
        self._request_id = 0
        self._request_timeout = float(request_timeout)
        self._closed = False
        parent, child = self._context.Pipe()
        self._connection = parent
        engine_config = {
            "model": str(model),
            "compute_type": str(compute_type),
            "gpu_device_index": gpu_device_index,
            "device": str(device),
            "beam_size": int(beam_size),
            "initial_prompt": initial_prompt,
            "suppress_tokens": (
                None if suppress_tokens is None else list(suppress_tokens)
            ),
            "batch_size": int(batch_size),
            "vad_filter": bool(vad_filter),
            "normalize_audio": bool(normalize_audio),
        }
        self._process = self._context.Process(
            target=_restricted_whisper_worker,
            args=(child, engine_config),
            name="echosense-restricted-whisper",
            daemon=True,
        )
        try:
            self._process.start()
        except Exception:
            parent.close()
            child.close()
            raise
        child.close()
        try:
            self._wait_until_ready(float(startup_timeout))
        except Exception:
            self.close()
            raise

    @property
    def worker_pid(self):
        return self._process.pid

    @property
    def worker_alive(self):
        return self._process.is_alive()

    def _wait_until_ready(self, timeout):
        if timeout <= 0:
            raise ValueError("startup_timeout must be positive")
        if not self._connection.poll(timeout):
            if not self._process.is_alive():
                detail = f"worker exited with code {self._process.exitcode}"
            else:
                detail = f"worker did not become ready within {timeout:g}s"
            raise RestrictedWhisperStartupError(detail)
        try:
            message = self._connection.recv()
        except EOFError as exc:
            raise RestrictedWhisperStartupError(
                "worker closed its startup channel"
            ) from exc
        if message[0] == "ready":
            return
        if message[0] == "startup_error":
            _status, error_type, detail = message
            raise RestrictedWhisperStartupError(
                f"{error_type}: {detail}"
            )
        raise RestrictedWhisperStartupError(
            f"unexpected worker startup response: {message[0]}"
        )

    def transcribe(self, audio, language=None, use_prompt=True):
        """Execute one request and return RealtimeSTT's native result type."""

        from RealtimeSTT.transcription_engines.base import (
            TranscriptionInfo,
            TranscriptionResult,
        )

        with self._lock:
            if self._closed:
                raise RuntimeError("restricted Whisper executor is closed")
            if not self._process.is_alive():
                raise RuntimeError(
                    "restricted Whisper worker exited unexpectedly "
                    f"with code {self._process.exitcode}"
                )
            self._request_id += 1
            request_id = self._request_id
            self._connection.send((
                "transcribe",
                request_id,
                np.asarray(audio, dtype=np.float32),
                language,
                bool(use_prompt),
            ))
            if not self._connection.poll(self._request_timeout):
                raise TimeoutError(
                    "restricted Whisper transcription timed out after "
                    f"{self._request_timeout:g}s"
                )
            try:
                message = self._connection.recv()
            except EOFError as exc:
                raise RuntimeError(
                    "restricted Whisper worker closed during transcription"
                ) from exc
            status, response_id, *values = message
            if response_id != request_id:
                raise RuntimeError("restricted Whisper response ID mismatch")
            if status == "error":
                error_type, detail = values
                raise RuntimeError(f"{error_type}: {detail}")
            if status != "result":
                raise RuntimeError(
                    f"unexpected restricted Whisper response: {status}"
                )
            text, detected_language, probability, metadata = values
            result = TranscriptionResult(
                text=text,
                info=TranscriptionInfo(
                    language=detected_language,
                    language_probability=float(probability),
                ),
            )
            result.metadata = metadata
            return result

    def close(self, timeout=10.0):
        """Stop the model worker; terminate only if graceful shutdown fails."""

        wait = max(0.0, float(timeout))
        acquired = self._lock.acquire(timeout=min(wait, 1.0))
        if not acquired and self._process.is_alive():
            # A decode thread owns the IPC lock. Terminating the worker wakes
            # its blocking recv promptly so shutdown cannot wait 120 seconds.
            self._process.terminate()
            self._process.join(timeout=5.0)
            acquired = self._lock.acquire(timeout=1.0)
        if acquired:
            try:
                if self._closed:
                    return
                self._closed = True
                if self._process.is_alive():
                    try:
                        self._connection.send(("shutdown",))
                    except (BrokenPipeError, EOFError, OSError):
                        pass
                self._connection.close()
            finally:
                self._lock.release()
        else:
            if self._closed:
                return
            self._closed = True
            self._connection.close()
        self._process.join(timeout=wait)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5.0)
