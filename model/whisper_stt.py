from RealtimeSTT import AudioToTextRecorder
from model.blacklist import check_transcript
from model.monitored_terms import classify_transcript_language
from model.realtimestt_audio_adapter import RealtimeSTTAudioEventAdapter
from detection.transcript_quality import assess_transcript_quality
from config import SHOW_TRANSCRIPT_TEXT
import json
import threading
import time

# FIX 2 — We deliberately do NOT pass an initial_prompt to Whisper. Priming it
# with a profanity word-list made it hallucinate those exact words out of
# background noise (rooster/dog/keyboard/music) — the single biggest false-alarm
# source. Detection is handled downstream by the blacklist, not by biasing STT.

_recorder = None
_latest_result = None
_result_lock = threading.Lock()
_new_result_event = threading.Event()
_recorder_ready = threading.Event()
_audio_event_adapter = RealtimeSTTAudioEventAdapter(sample_rate=16000)

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


def _on_text(
    text: str,
    whisper_language=None,
    whisper_language_confidence=None,
    audio_event=None,
):
    global _latest_result
    original_text = str(text or "")
    quality = assess_transcript_quality(original_text)
    event_id = audio_event.event_id if audio_event is not None else "unavailable"

    if quality.accepted:
        print(
            f"[TEXT_QUALITY] event={event_id} accepted=true "
            f"tokens={quality.token_count} unique={quality.unique_token_count} "
            f"repetition_ratio={quality.repetition_ratio:.2f}"
        )
        result = check_transcript(original_text)
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

    with _result_lock:
        _latest_result = result
    _new_result_event.set()


def _run_transcription_cycle(recorder, audio_event_adapter):
    """Run exactly one synchronous transcription and consume its event once."""
    audio_event_adapter.clear_pending()
    try:
        text = recorder.text()
    except Exception:
        audio_event_adapter.clear_pending()
        raise

    audio_event = audio_event_adapter.consume_pending()
    return (
        text,
        audio_event,
        getattr(recorder, "detected_language", None),
        getattr(recorder, "detected_language_probability", None),
    )


def _recorder_loop():
    global _recorder
    print("[STT] Loading RealtimeSTT with Whisper base...")
    _recorder = AudioToTextRecorder(
        # --- Final (accurate) transcription model ---------------------------
        model="base",            # int8 base = best speed/accuracy balance on the Pi
        # Want better Bisaya / code-switch accuracy? Comment the line above and
        # uncomment the one below — ~2x slower on the Pi, latency may rise to 3-5s:
        # model="small",
        device="cpu",            # the Pi has no GPU; default was 'cuda'
        compute_type="int8",     # was defaulting to float32 (slow). int8 = 2-4x faster
        beam_size=1,             # was 5; greedy decode is much faster, minor accuracy cost
        batch_size=0,            # plain WhisperModel (no batching) — matches the old pipeline
        # No forced language: faster-whisper auto-detects supported languages.
        # Whisper has Tagalog ("tl") and English tokens but no native Cebuano
        # token, so downstream monitored-term evidence handles ceb/mixed labels.
        spinner=False,
        # VAD tuned for QUIET, high-pitched Grade 6 voices at ~1 m. These are
        # deliberately permissive so short/soft words (bobo, pangit, dakog
        # ilong) are not missed. min_length_of_recording=0.3 is the key lever
        # for catching short words. False alarms are held back DOWNSTREAM
        # instead (consume-once + tightened Track B: a lone word never alerts;
        # it must repeat or be paired).
        # NOTE: in RealtimeSTT, silero_sensitivity is "more sensitive at higher
        # values" — if quiet speech is still missed, RAISE this, don't lower it.
        silero_sensitivity=0.6,
        webrtc_sensitivity=1,
        post_speech_silence_duration=0.6,
        min_length_of_recording=0.5,
        min_gap_between_recordings=0.1,
        on_transcription_start=_audio_event_adapter.on_transcription_start,
        # FIX 2 — initial_prompt removed (was priming profanity hallucinations).
        # --- Live word-by-word preview ([LIVE] lines) — DISABLED on the Pi ---
        # Running a 2nd (realtime) Whisper model alongside the base model
        # saturates the Pi CPU -> "audio queue exceeds latency limit, discarding
        # chunks" -> the model is fed broken audio and loops ("pangit ka, pangit
        # ka..."). Single model only. Re-enable these on a more powerful host:
        # enable_realtime_transcription=True,
        # realtime_model_type="tiny",
        # beam_size_realtime=1,
        # realtime_processing_pause=0.2,
        # initial_prompt_realtime=WHISPER_INITIAL_PROMPT,
        # on_realtime_transcription_update=_on_realtime_update,
    )
    print("[STT] Ready.")
    _recorder_ready.set()
    while True:
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
            )
        except Exception as e:
            print(f"[STT] Error: {e}")
            time.sleep(1)

def get_recorder():
    global _recorder
    if _recorder is None:
        t = threading.Thread(target=_recorder_loop, daemon=True)
        t.start()
        _recorder_ready.wait(timeout=60)
    return _recorder

def _empty_result() -> dict:
    return {
        "has_profanity": False,
        "detected_words": [],
        "transcript": "",
        "transcribed_text": "",
        "hard_hits": [],
        "soft_hits": [],
        "is_casual": False,
        "severity": "low",
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
    _new_result_event.wait(timeout=1.0)
    _new_result_event.clear()

    # FIX 1 — consume-once. Read AND clear the latest result so the same
    # utterance can never be processed twice. This is what stops the 60s re-fire
    # loop and the fake "repetition" that fired on a single word said once.
    global _latest_result
    with _result_lock:
        result = _latest_result
        _latest_result = None

    if result is None:
        return _empty_result()

    return result

def get_model():
    return get_recorder()
