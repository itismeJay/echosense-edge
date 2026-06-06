from faster_whisper import WhisperModel
import numpy as np
from model.blacklist import check_transcript

_model = None

# Vocabulary bias: priming Whisper with the words we care about makes the model
# far more likely to transcribe them correctly in a noisy room. Mixed
# Bisaya/Tagalog/English so all three demo paths benefit. Keep it short — long
# prompts still hurt the small/base models.
WHISPER_INITIAL_PROMPT = (
    "bogo, bugok, bulok, bobo, tanga, gago, yawa, "
    "putangina, pangit, tambok, baho, hilak nasad, "
    "dakog ilong, pango, uling, retard, patyon tika, "
    "walang kwenta, wala kang kwenta, bata og yawa, "
    "buang, yawa kaayo, gago ka, bobo ka, pangit kaayo, "
    "bogo kaayo, bulok man ka, tambokikoy, bungi, "
    "hilak hilak, pikon, ampon, sumbong, luod kaayo"
)


def get_model():
    global _model
    if _model is None:
        print("[WHISPER] Loading multilingual base model...")
        _model = WhisperModel("base", device="cpu", compute_type="int8")
        print("[WHISPER] Ready.")
    return _model


def preprocess_audio(audio_np: np.ndarray) -> np.ndarray:
    try:
        import noisereduce as nr
        reduced = nr.reduce_noise(
            y=audio_np.astype(float),
            sr=16000,
            prop_decrease=0.75
        )
        return reduced.astype(np.int16)
    except Exception:
        return audio_np


def transcribe_and_check(audio_np: np.ndarray) -> dict:
    model = get_model()
    clean  = preprocess_audio(audio_np)
    audio_float = clean.astype(np.float32) / 32768.0

    empty_result = {
        "has_profanity": False, "detected_words": [],
        "transcribed_text": "", "hard_hits": [],
        "soft_hits": [], "is_casual": False,
        "severity": "low", "categories": [],
        "language": "unknown", "all_words": [],
        "word_count": 0,
    }

    try:
        segments, info = model.transcribe(
            audio_float,
            language="tl",                     # force Filipino — auto-detect misfires on young,
                                               # high-pitched voices (guesses ZH/KO/AR); Davao
                                               # Grade 6 speech is always tl/Bisaya/en anyway
            task="transcribe",
            initial_prompt=WHISPER_INITIAL_PROMPT,  # bias toward our bullying vocabulary
            vad_filter=True,
            word_timestamps=True,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,           # drop segments Whisper deems silent
            compression_ratio_threshold=2.4,   # reject repetitive hallucinations
            log_prob_threshold=-1.0,           # reject low-confidence transcriptions
        )
        full_text = ""
        all_words = []
        for seg in segments:
            full_text += seg.text + " "
            if seg.words:
                all_words.extend([w.word.lower().strip()
                                   for w in seg.words])
        full_text = full_text.strip().lower()
        lang = info.language if info else "unknown"
    except Exception as e:
        print(f"[WHISPER] Error: {e}")
        return empty_result

    result = check_transcript(full_text)
    result["transcribed_text"] = full_text
    result["language"]         = lang
    result["all_words"]        = all_words

    if full_text:
        print(f"[WHISPER] [{lang.upper()}] {full_text}")
        print(f'[CHECK] Checking: "{full_text}"')
        print(f'[CHECK] After variants: "{result.get("checked_text", full_text)}"')
        print(f"[CHECK] Hard hits: {result['hard_hits']} Soft hits: {result['soft_hits']}")
    if result["has_profanity"]:
        print(f"[WHISPER] HIT: {result['detected_words']} "
              f"| cat: {result['categories']} "
              f"| sev: {result['severity']}")
    return result
