from collections import deque
import json
import math
import queue
import threading
import time
import unicodedata

import numpy as np

from audio.audio_event import float32_samples_to_int16
from config import DETECTION_RUNTIME, SHOW_TRANSCRIPT_TEXT
from detection.latency import LATENCY
from detection.false_alarm_filters import (
    DecodeMetrics,
    FilterConfig,
    decode_gate,
    decode_gate_overridden,
    filter_terms_by_probability,
    hallucination_reason,
    media_gate,
)
from detection.severity import LOW
from detection.transcript_quality import assess_transcript_quality
from model.blacklist import check_transcript, restrict_to_terms
from model.language_restricted_engine import ENGINE_NAME as RESTRICTED_ENGINE
from model.language_restricted_engine import register as register_restricted_engine
from model.monitored_terms import classify_transcript_language, get_backend_terms
from model.realtimestt_audio_adapter import RealtimeSTTAudioEventAdapter
from model.result_queue import BoundedResultQueue

# Whisper decodes with NO initial_prompt. A prompt listing the monitored terms
# made Whisper "hear" those words in music, video audio and quiet rooms. The
# synced dictionary is used for matching only. build_initial_prompt() remains
# for diagnostics and tests; its result is not passed to Whisper.
WHISPER_PROMPT_MAX_TERMS = 24
WHISPER_PROMPT_MAX_CHARACTERS = 200

_SUPPORTED_WHISPER_LANGUAGES = frozenset({"en", "tl", "fil", "ceb"})
_REGIONAL_RETRY_LANGUAGE = "tl"

_recorder = None
_recorder_thread = None
_recorder_error = None
_recorder_stop_event = threading.Event()
_recorder_ready = threading.Event()
_audio_event_adapter = RealtimeSTTAudioEventAdapter(
    sample_rate=16000,
    max_samples=DETECTION_RUNTIME.max_stt_utterance_samples,
)
_result_queue = BoundedResultQueue(
    DETECTION_RUNTIME.stt_result_queue_size,
    max_audio_samples=DETECTION_RUNTIME.max_stt_utterance_samples,
)
_recording_limiter = None

# False-alarm filters (see detection/false_alarm_filters.py), tunable through
# ECHOSENSE_FILTER_* environment variables.
_FILTERS = FilterConfig.from_environment()
# Set by the runtime: (samples, sample_rate) -> [(yamnet_label, score)].
_media_classifier = None


def set_media_classifier(classifier):
    """Install the YAMNet top-classes function used by the media filter."""

    global _media_classifier
    _media_classifier = classifier


class RealtimeSTTRecordingLimiter:
    """Best-effort public-API stop watchdog for RealtimeSTT recordings.

    RealtimeSTT 1.0.2 has no maximum-recording constructor setting. Its
    documented recording callbacks and ``stop()`` method are used here to
    prevent an active recording from continuing indefinitely. The strict
    EchoSense-owned bound remains the sample cap applied to every AudioEvent.
    """

    def __init__(self, max_seconds, *, timer_factory=threading.Timer):
        self.max_seconds = float(max_seconds)
        if not math.isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise ValueError("max_seconds must be finite and positive")
        self._timer_factory = timer_factory
        self._lock = threading.Lock()
        self._recorder = None
        self._timer = None

    def bind(self, recorder):
        with self._lock:
            self._recorder = recorder

    def on_recording_start(self):
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            timer = self._timer_factory(
                self.max_seconds,
                self._stop_due_to_limit,
            )
            timer.daemon = True
            self._timer = timer
            timer.start()

    def on_recording_stop(self):
        LATENCY.speech_end()
        with self._lock:
            timer = self._timer
            self._timer = None
        if timer is not None:
            timer.cancel()

    def _stop_due_to_limit(self):
        with self._lock:
            recorder = self._recorder
            self._timer = None
        if recorder is None:
            return
        print(
            "[STT] recording_duration_limit_reached=true "
            f"max_seconds={self.max_seconds:g}"
        )
        try:
            recorder.stop()
        except Exception as exc:
            print(f"[STT] recording_stop_error={type(exc).__name__}")

    def shutdown(self):
        self.on_recording_stop()
        with self._lock:
            self._recorder = None


def build_initial_prompt(terms=None):
    """Return a short comma-separated term list, or None if none are synced."""

    if terms is None:
        terms = [item.term for item in get_backend_terms()]
    prompt_terms = []
    length = 0
    for term in dict.fromkeys(str(t).strip() for t in terms if str(t).strip()):
        added = len(term) + (2 if prompt_terms else 0)
        if (
            len(prompt_terms) >= WHISPER_PROMPT_MAX_TERMS
            or length + added > WHISPER_PROMPT_MAX_CHARACTERS
        ):
            break
        prompt_terms.append(term)
        length += added
    return ", ".join(prompt_terms) or None


class NoiseReducer:
    """Spectral-gating noise reduction applied before audio reaches Whisper.

    Audio is processed in short blocks. The noise profile is the quietest few
    recent blocks, which tracks the classroom's stationary background without
    treating speech as noise. Without noisereduce installed, audio passes
    through unchanged and that is logged once.
    """

    def __init__(
        self,
        *,
        sample_rate=16000,
        # 0.25 s blocks: every block is held this long before RealtimeSTT's
        # VAD sees it, so it adds directly to speech-end latency. Profile and
        # history keep the same 2 s / 20 s of audio as the old 0.5 s blocks.
        block_samples=4000,
        history_blocks=40,
        profile_blocks=8,
        prop_decrease=0.75,
        reduce_function=None,
        enabled=True,
    ):
        self.sample_rate = int(sample_rate)
        self.block_samples = int(block_samples)
        self.profile_blocks = int(profile_blocks)
        self.prop_decrease = float(prop_decrease)
        self._pending = np.empty(0, dtype=np.float32)
        self._history = deque(maxlen=int(history_blocks))
        self._reduce = reduce_function
        if self._reduce is None and enabled:
            try:
                import noisereduce

                self._reduce = noisereduce.reduce_noise
            except Exception as exc:
                print(
                    "[STT_NOISE] noisereduce_unavailable="
                    f"{type(exc).__name__} passthrough=true"
                )
        self.enabled = bool(enabled and self._reduce is not None)
        self.hold_ms = (
            self.block_samples / self.sample_rate * 1000 if self.enabled else 0.0
        )
        self.compute_ms = 0.0

    def process(self, samples):
        """Accept float32 samples; return zero or more processed blocks.

        When reduction is disabled, frames pass through immediately unbuffered.
        """

        if not self.enabled:
            return [np.asarray(samples, dtype=np.float32)]
        self._pending = np.concatenate(
            (self._pending, np.asarray(samples, dtype=np.float32))
        )
        blocks = []
        while self._pending.size >= self.block_samples:
            block = self._pending[:self.block_samples]
            self._pending = self._pending[self.block_samples:]
            blocks.append(self._process_block(block))
        return blocks

    def _process_block(self, block):
        rms = float(np.sqrt(np.mean(block ** 2)))
        self._history.append((rms, block))
        if len(self._history) < self.profile_blocks:
            return block
        quietest = sorted(self._history, key=lambda item: item[0])
        noise = np.concatenate(
            [item[1] for item in quietest[:self.profile_blocks]]
        )
        started = time.perf_counter()
        try:
            reduced = self._reduce(
                y=block,
                sr=self.sample_rate,
                y_noise=noise,
                stationary=True,
                prop_decrease=self.prop_decrease,
            )
        except Exception as exc:
            print(f"[STT_NOISE] reduce_failed={type(exc).__name__}")
            return block
        elapsed_ms = (time.perf_counter() - started) * 1000
        # Smoothed per-block compute time, reported on [LATENCY] lines.
        self.compute_ms = (
            elapsed_ms if self.compute_ms == 0.0
            else 0.8 * self.compute_ms + 0.2 * elapsed_ms
        )
        LATENCY.set_noise_reduction(
            hold_ms=self.hold_ms, compute_ms=self.compute_ms,
        )
        return np.clip(np.asarray(reduced, dtype=np.float32), -1.0, 1.0)


def _on_realtime_update(text: str):
    if text and text.strip():
        print(f"[LIVE] {text.strip()}")


def _log_finalized_transcript(event_id, language, original_text):
    """Optionally log one exact finalized transcript with safe line escaping."""

    if not SHOW_TRANSCRIPT_TEXT:
        return
    encoded_text = json.dumps(str(original_text or ""), ensure_ascii=False)
    print(
        f"[TRANSCRIPT] event={event_id} "
        f"language={language} text={encoded_text}"
    )


def _uses_supported_latin_script(text: str) -> bool:
    """Allow the Latin-script text used by English, Filipino, and Cebuano."""

    for character in str(text or ""):
        if not character.isalpha():
            continue
        if "LATIN" not in unicodedata.name(character, ""):
            return False
    return True


def _is_supported_transcription(text: str, whisper_language) -> bool:
    """Reject automatic decodes outside the deployment's language scope."""

    language = str(whisper_language or "").strip().lower()
    return (
        language in _SUPPORTED_WHISPER_LANGUAGES
        and _uses_supported_latin_script(text)
    )


def _retry_in_regional_language(recorder, audio_event):
    """Re-decode one finalized event as Tagalog when auto-detect wanders.

    Whisper has a Tagalog language token but no native Cebuano token. Tagalog
    is therefore the safest regional decoding constraint for Filipino,
    Cebuano/Bisaya, and ordinary classroom code-switching. Downstream monitored
    terms still assign evidence to ``fil``, ``ceb``, or ``en``.
    """

    if audio_event is None or not hasattr(recorder, "perform_final_transcription"):
        return "", None, None

    missing = object()
    previous_language = getattr(recorder, "language", missing)
    try:
        recorder.language = _REGIONAL_RETRY_LANGUAGE
        text = recorder.perform_final_transcription(
            audio_event.samples,
            use_prompt=False,
        )
        language = getattr(recorder, "detected_language", None)
        confidence = getattr(
            recorder,
            "detected_language_probability",
            None,
        )
    except Exception as exc:
        print(f"[LANGUAGE_GUARD] retry_failed={type(exc).__name__}")
        return "", None, None
    finally:
        if previous_language is missing:
            try:
                del recorder.language
            except AttributeError:
                pass
        else:
            recorder.language = previous_language

    if not _is_supported_transcription(text, language):
        return "", None, None
    return text, language, confidence


def _log_decode_metrics(event_id, metrics):
    if metrics is None:
        print(f"[DECODE] event={event_id} metrics=unavailable")
        return
    print(
        f"[DECODE] event={event_id} no_speech_prob={metrics.no_speech_prob:.2f} "
        f"avg_logprob={metrics.avg_logprob:.2f} "
        f"compression_ratio={metrics.compression_ratio:.2f} "
        f"words={len(metrics.words)} segments={metrics.segment_count}"
    )


def _media_top_classes(event_id, audio_event):
    if _media_classifier is None or audio_event is None:
        return None
    try:
        top = _media_classifier(audio_event.samples, audio_event.sample_rate)
    except Exception as exc:
        print(f"[YAMNET_TOP] event={event_id} ran=false error={type(exc).__name__}")
        return None
    summary = ",".join(f"{label}:{score:.2f}" for label, score in top)
    print(f"[YAMNET_TOP] event={event_id} top={summary or 'none'}")
    return top


def _apply_term_filters(result, original_text, metrics, audio_event, event_id):
    """Word-probability and media filters on an accepted, matched transcript."""

    terms = list(result.get("detected_words") or ())
    if not terms:
        return result
    kept, dropped, scores = filter_terms_by_probability(terms, metrics, _FILTERS)
    for term in terms:
        probability = scores.get(term)
        print(
            f"[WORD_PROB] event={event_id} term={term} "
            f"probability={'unscored' if probability is None else f'{probability:.2f}'} "
            f"kept={str(term in kept).lower()}"
        )
    if dropped:
        print(
            f"[FILTER] event={event_id} dropped_terms={dropped} "
            f"reason=word_probability<{_FILTERS.word_probability_min:g}"
            f"(urgent<{_FILTERS.urgent_word_probability_min:g})"
        )
        result = restrict_to_terms(result, kept, original_text)
        result.setdefault("filter_reasons", []).append("low_word_probability")
    if not kept:
        return result
    top_classes = _media_top_classes(event_id, audio_event)
    reason = media_gate(top_classes, kept, _FILTERS)
    if reason:
        print(
            f"[FILTER] event={event_id} dropped_terms={kept} reason={reason}"
        )
        result = restrict_to_terms(result, [], original_text)
        result.setdefault("filter_reasons", []).append(reason)
    return result


def _on_text(
    text: str,
    whisper_language=None,
    whisper_language_confidence=None,
    audio_event=None,
    decode_metadata=None,
):
    original_text = str(text or "")
    normalized_whisper_language = str(
        whisper_language or ""
    ).strip().lower()
    unsupported_final = (
        not _uses_supported_latin_script(original_text)
        or (
            normalized_whisper_language
            and normalized_whisper_language
            not in _SUPPORTED_WHISPER_LANGUAGES
        )
    )
    if original_text and unsupported_final:
        print(
            "[LANGUAGE_GUARD] rejected unsupported final transcription "
            f"language={whisper_language or 'unknown'}"
        )
        original_text = ""
        whisper_language = None
        whisper_language_confidence = None
    event_id = audio_event.event_id if audio_event is not None else "unavailable"
    metrics = DecodeMetrics.from_metadata(decode_metadata)
    filter_reasons = []
    if original_text:
        _log_decode_metrics(event_id, metrics)
        drop_reason = hallucination_reason(original_text, _FILTERS)
        if drop_reason is None:
            drop_reason = decode_gate(metrics, _FILTERS)
            if drop_reason and decode_gate_overridden(
                check_transcript(original_text).get("detected_words", ()),
                metrics,
                _FILTERS,
            ):
                print(
                    f"[FILTER] event={event_id} kept=utterance "
                    f"decode_gate={drop_reason} override=threat_or_self_harm"
                )
                drop_reason = None
        if drop_reason:
            print(
                f"[FILTER] event={event_id} dropped=utterance reason={drop_reason}"
            )
            filter_reasons.append(drop_reason)
            original_text = ""
    quality = assess_transcript_quality(original_text)

    if quality.accepted:
        print(
            f"[TEXT_QUALITY] event={event_id} accepted=true "
            f"tokens={quality.token_count} unique={quality.unique_token_count} "
            f"repetition_ratio={quality.repetition_ratio:.2f}"
        )
        result = check_transcript(original_text)
        result = _apply_term_filters(
            result, original_text, metrics, audio_event, event_id,
        )
    else:
        primary_reason = quality.reason_codes[0]
        print(
            f"[TEXT_QUALITY] event={event_id} accepted=false "
            f"reason={primary_reason}"
        )
        result = _empty_result()

    # The Whisper output remains unchanged for evidence/display. Matching uses
    # normalized_text/checked_text separately and never overwrites this field.
    result["transcribed_text"] = original_text
    result["transcript"] = original_text
    result["normalized_text"] = quality.normalized_text
    result["transcript_quality"] = quality
    result["filter_reasons"] = filter_reasons + list(result.get("filter_reasons", []))
    result["quality_accepted"] = quality.accepted
    result["quality_reason_codes"] = quality.reason_codes
    result["quality_warnings"] = quality.warnings
    language, language_confidence = classify_transcript_language(
        whisper_language,
        whisper_language_confidence,
        result.get("matched_terms", []),
    )
    result["language"] = language
    result["language_confidence"] = language_confidence
    result["audio_event"] = audio_event
    result["event_id"] = audio_event.event_id if audio_event is not None else None
    result["all_words"] = quality.normalized_text.split()
    _log_finalized_transcript(event_id, language, original_text)

    print(
        f"[TERM_MATCH] event={event_id} "
        f"candidates={result.get('match_candidate_count', 0)} "
        f"accepted={result.get('match_accepted_count', 0)} "
        f"suppressed={result.get('match_suppressed_count', 0)}"
    )
    for suppressed in result.get("suppressed_terms", []):
        print(
            f"[TERM_MATCH] event={event_id} suppressed "
            f"reason={suppressed['reason']} term={suppressed['term']}"
        )
    confidence_summary = (
        f"{language_confidence:.2f}"
        if language_confidence is not None
        else "unavailable"
    )
    print(f"[LANGUAGE] {language} confidence={confidence_summary}")

    if result["has_profanity"]:
        print(
            f"[STT] event={event_id} monitored_terms={result['detected_words']} "
            f"| cat: {result['categories']} "
            f"| sev: {result['severity']}"
        )

    _result_queue.put(result)
    if audio_event is not None:
        LATENCY.mark(audio_event.event_id, "transcript_emitted")


def _run_transcription_cycle(recorder, audio_event_adapter):
    """Run exactly one synchronous transcription and consume its event once."""
    audio_event_adapter.clear_pending()
    # Never let one utterance's decode metrics describe the next.
    recorder.last_transcription_metadata = None
    try:
        text = recorder.text()
    except Exception:
        audio_event_adapter.clear_pending()
        raise

    audio_event = audio_event_adapter.consume_pending()
    utterance_id = audio_event.event_id if audio_event is not None else None
    LATENCY.begin(utterance_id)
    detected_language = getattr(recorder, "detected_language", None)
    detected_language_probability = getattr(
        recorder,
        "detected_language_probability",
        None,
    )
    if text and not _is_supported_transcription(text, detected_language):
        print(
            "[LANGUAGE_GUARD] "
            f"unsupported={detected_language or 'unknown'} "
            f"retry={_REGIONAL_RETRY_LANGUAGE}"
        )
        text, detected_language, detected_language_probability = (
            _retry_in_regional_language(recorder, audio_event)
        )
        if not text:
            print("[LANGUAGE_GUARD] rejected=true")
        LATENCY.mark(utterance_id, "whisper_done", language_retry="true")
    else:
        LATENCY.mark(utterance_id, "whisper_done")

    return (
        text,
        audio_event,
        detected_language,
        detected_language_probability,
        getattr(recorder, "last_transcription_metadata", None),
    )


def _recorder_loop():
    global _recorder, _recorder_error, _recording_limiter
    try:
        from RealtimeSTT import AudioToTextRecorder

        print("[STT] Loading RealtimeSTT with Whisper base...")
        print("[STT] initial_prompt=none (dictionary used for matching only)")
        _recording_limiter = RealtimeSTTRecordingLimiter(
            DETECTION_RUNTIME.max_stt_utterance_seconds
        )
        register_restricted_engine()
        recorder_options = dict(
            # --- Final (accurate) transcription model -----------------------
            model="base",
            device="cpu",
            compute_type="int8",
            beam_size=1,
            batch_size=0,
            # Language detection is restricted to en/tl by the engine (see
            # model/language_restricted_engine.py), so there is one decode per
            # utterance. Whisper has no Cebuano token; downstream term evidence
            # handles ceb/mixed labels.
            spinner=False,
            # EchoSense owns the only physical microphone and feeds this
            # recorder. This verified RealtimeSTT 1.0.2 mode avoids a second
            # PortAudio reader.
            use_microphone=False,
            # VAD remains permissive for short/quiet classroom utterances.
            silero_sensitivity=0.6,
            webrtc_sensitivity=1,
            post_speech_silence_duration=0.45,
            min_length_of_recording=0.5,
            min_gap_between_recordings=0.1,
            on_recording_start=_recording_limiter.on_recording_start,
            on_recording_stop=_recording_limiter.on_recording_stop,
            on_transcription_start=_audio_event_adapter.on_transcription_start,
            # No hint list (see the note at the top of this module). Realtime
            # transcription remains disabled so only one Whisper model competes
            # for Pi CPU.
            initial_prompt=None,
        )
        try:
            _recorder = AudioToTextRecorder(
                transcription_engine=RESTRICTED_ENGINE, **recorder_options,
            )
            print("[STT] language_detection=en,tl engine=restricted")
        except Exception as exc:
            # Stock auto-detect; the regional retry below still guards it.
            print(
                f"[STT] restricted_engine_failed={type(exc).__name__} "
                "language_detection=auto fallback=true"
            )
            _recorder = AudioToTextRecorder(**recorder_options)
        _recording_limiter.bind(_recorder)
    except Exception as exc:
        _recorder_error = exc
        _recorder_ready.set()
        return
    print("[STT] Ready external_feed=true physical_reader=false")
    _recorder_ready.set()
    while not _recorder_stop_event.is_set():
        try:
            # Use the synchronous result so language metadata and text come from
            # the same transcription. RealtimeSTT's callback form starts a new
            # thread, which can race with the next utterance and overwrite the
            # recorder's detected_language fields.
            (
                text,
                audio_event,
                detected_language,
                detected_language_probability,
                decode_metadata,
            ) = _run_transcription_cycle(_recorder, _audio_event_adapter)
            event_id = (
                audio_event.event_id
                if audio_event is not None
                else "unavailable"
            )
            if audio_event is None:
                print("[AUDIO_EVENT] unavailable for completed transcription")
            print(
                f"[STT] event={event_id} transcript_length={len(text or '')} "
                f"language={detected_language or 'unknown'}"
            )
            _on_text(
                text,
                detected_language,
                detected_language_probability,
                audio_event,
                decode_metadata,
            )
        except Exception as exc:
            if _recorder_stop_event.is_set():
                break
            print(f"[STT] Error: {type(exc).__name__}")
            _recorder_stop_event.wait(1.0)

def get_recorder():
    global _recorder_thread, _recorder_error
    if _recorder is None:
        _recorder_error = None
        _recorder_stop_event.clear()
        _recorder_ready.clear()
        _recorder_thread = threading.Thread(
            target=_recorder_loop,
            name="realtimestt-transcription",
        )
        _recorder_thread.start()
        if not _recorder_ready.wait(timeout=60):
            raise RuntimeError("RealtimeSTT initialization timed out")
        if _recorder_error is not None:
            raise RuntimeError(
                f"RealtimeSTT initialization failed: {type(_recorder_error).__name__}"
            ) from _recorder_error
    return _recorder


def stop_recorder():
    """Stop RealtimeSTT and its owned workers without touching a microphone."""

    global _recorder, _recorder_thread, _recorder_error, _recording_limiter
    _recorder_stop_event.set()
    if _recording_limiter is not None:
        _recording_limiter.shutdown()
    recorder = _recorder
    if recorder is not None:
        try:
            recorder.abort()
        except Exception:
            pass
        try:
            recorder.shutdown()
        except Exception as exc:
            print(f"[STT] shutdown_error={type(exc).__name__}")
    if _recorder_thread is not None:
        _recorder_thread.join(timeout=10.0)
    _recorder = None
    _recorder_thread = None
    _recorder_error = None
    _recording_limiter = None
    _recorder_ready.clear()
    clear_transcription_results()
    print("[STT] stopped")

def _empty_result() -> dict:
    return {
        "has_profanity": False,
        "detected_words": [],
        "transcript": "",
        "transcribed_text": "",
        "hard_hits": [],
        "soft_hits": [],
        "is_casual": False,
        "severity": LOW,
        "severity_evidence": {
            "level": LOW,
            "reasons": ["no_monitored_term"],
            "term_categories": {},
            "supporting_evidence": [],
        },
        "categories": [],
        "language": "unknown",
        "language_confidence": None,
        "audio_event": None,
        "event_id": None,
        "matched_terms": [],
        "all_words": [],
        "word_count": 0,
        "checked_text": "",
        "normalized_text": "",
        "transcript_quality": None,
        "quality_accepted": False,
        "quality_reason_codes": (),
        "quality_warnings": (),
        "match_candidate_count": 0,
        "match_accepted_count": 0,
        "match_suppressed_count": 0,
        "suppressed_terms": [],
        "context_suppressed_all": False,
    }


def transcribe_and_check(audio_np=None) -> dict:
    get_recorder()
    try:
        return _result_queue.get(timeout=1.0)
    except queue.Empty:
        return _empty_result()


def consume_transcription_result(timeout=0.0):
    """Consume one completed result in FIFO order without starting hardware."""

    try:
        return _result_queue.get(timeout=timeout)
    except queue.Empty:
        return None


def clear_transcription_results():
    _result_queue.clear()


class RealtimeSTTFeedWorker:
    """Bounded bridge from canonical frames into RealtimeSTT manual input."""

    def __init__(
        self,
        frame_queue,
        *,
        recorder_provider=get_recorder,
        max_internal_chunks=None,
        noise_reducer=None,
    ):
        self.frame_queue = frame_queue
        self.recorder_provider = recorder_provider
        self.max_internal_chunks = int(
            max_internal_chunks
            or DETECTION_RUNTIME.stt_internal_audio_chunks
        )
        if self.max_internal_chunks <= 0:
            raise ValueError("max_internal_chunks must be positive")
        self.noise_reducer = noise_reducer
        self._stop_event = threading.Event()
        self._thread = None
        self.overflow_count = 0
        self.error = None

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self.error = None
        self._thread = threading.Thread(
            target=self._run,
            name="realtimestt-audio-feed",
        )
        self._thread.start()

    def _run(self):
        try:
            recorder = self.recorder_provider()
            while not self._stop_event.is_set():
                try:
                    frame = self.frame_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                internal_queue = getattr(recorder, "audio_queue", None)
                if (
                    internal_queue is not None
                    and internal_queue.qsize() >= self.max_internal_chunks
                ):
                    self.overflow_count += 1
                    if self.overflow_count == 1 or self.overflow_count % 100 == 0:
                        print(
                            f"[STT_AUDIO_QUEUE] overflow_count={self.overflow_count} "
                            "policy=drop_newest"
                        )
                    continue
                if self.noise_reducer is None:
                    self.noise_reducer = NoiseReducer()
                for block in self.noise_reducer.process(frame.samples):
                    pcm_bytes = float32_samples_to_int16(block).tobytes()
                    recorder.feed_audio(pcm_bytes, original_sample_rate=16000)
        except Exception as exc:
            self.error = exc
            self._stop_event.set()
            print(f"[STT] feed_worker_error={type(exc).__name__}")

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._thread = None

def get_model():
    return get_recorder()
