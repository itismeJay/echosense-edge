from RealtimeSTT import AudioToTextRecorder
from model.blacklist import check_transcript, apply_phonetic_variants
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

def _on_realtime_update(text: str):
    if text and text.strip():
        print(f"[LIVE] {text.strip()}")

def _on_text(text: str):
    global _latest_result
    if not text or not text.strip():
        return

    text_clean = text.strip().lower()
    print(f"[STT] Heard: {text_clean}")

    corrected = apply_phonetic_variants(text_clean)
    if corrected != text_clean:
        print(f"[VARIANTS] '{text_clean}' -> '{corrected}'")

    result = check_transcript(corrected)
    result["transcribed_text"] = corrected
    result["language"] = "tl"
    result["all_words"] = corrected.split()

    print(f"[CHECK] Hard: {result['hard_hits']} Soft: {result['soft_hits']}")

    if result["has_profanity"]:
        print(
            f"[STT] HIT: {result['detected_words']} "
            f"| cat: {result['categories']} "
            f"| sev: {result['severity']}"
        )

    with _result_lock:
        _latest_result = result
    _new_result_event.set()

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
        language="tl",
        spinner=False,
        # FIX 3 — stricter VAD so noise (rooster/dog/keyboard/music/brief
        # clicks) never reaches Whisper; only clear, sustained human speech
        # passes. min_length_of_recording=1.0 alone drops most short noises.
        silero_sensitivity=0.6,
        webrtc_sensitivity=3,
        post_speech_silence_duration=0.8,
        min_length_of_recording=1.0,
        min_gap_between_recordings=0.3,
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
            _recorder.text(_on_text)
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
        "transcribed_text": "",
        "hard_hits": [],
        "soft_hits": [],
        "is_casual": False,
        "severity": "low",
        "categories": [],
        "language": "tl",
        "all_words": [],
        "word_count": 0,
        "checked_text": "",
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

    # FIX 7 — reject likely noise / Whisper hallucinations before they can alert.
    words = result.get("all_words") or result.get("transcribed_text", "").split()
    # Single-word transcriptions are almost always noise or a non-directed slip.
    if len(words) < 2:
        return _empty_result()
    # Hallucination signature: the same token(s) echoed back
    # ("thank you thank you thank you", "hello hello hello").
    if len(set(words)) < len(words) / 2:
        return _empty_result()

    return result

def get_model():
    return get_recorder()
