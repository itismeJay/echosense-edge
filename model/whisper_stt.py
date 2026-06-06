from RealtimeSTT import AudioToTextRecorder
from model.blacklist import check_transcript, apply_phonetic_variants
import threading
import time

WHISPER_INITIAL_PROMPT = (
    "bogo, bugok, bulok, bobo, tanga, gago, yawa, "
    "putangina, pangit, tambok, baho, hilak nasad, "
    "dakog ilong, pango, uling, retard, patyon tika, "
    "walang kwenta, wala kang kwenta, bata og yawa, "
    "buang, yawa kaayo, gago ka, bobo ka, pangit kaayo, "
    "bogo kaayo, bulok man ka, tambokikoy, bungi, "
    "hilak hilak, pikon, ampon, sumbong, luod kaayo"
)

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
        model="base",
        language="tl",
        spinner=False,
        silero_sensitivity=0.3,
        webrtc_sensitivity=2,
        post_speech_silence_duration=0.5,
        min_length_of_recording=0.5,
        min_gap_between_recordings=0.1,
        initial_prompt=WHISPER_INITIAL_PROMPT,
        on_realtime_transcription_update=_on_realtime_update,
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

def transcribe_and_check(audio_np=None) -> dict:
    get_recorder()
    _new_result_event.wait(timeout=1.0)
    _new_result_event.clear()

    with _result_lock:
        result = _latest_result

    if result is None:
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
    return result

def get_model():
    return get_recorder()
