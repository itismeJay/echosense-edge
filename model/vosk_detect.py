import json
import vosk
from config import VOSK_FILIPINO_MODEL, VOSK_ENGLISH_MODEL
from model.blacklist import get_detected_words

filipino_model = vosk.Model(VOSK_FILIPINO_MODEL)
english_model = vosk.Model(VOSK_ENGLISH_MODEL)
filipino_rec = vosk.KaldiRecognizer(filipino_model, 16000)
english_rec = vosk.KaldiRecognizer(english_model, 16000)

def process_audio_chunk(audio_chunk: bytes):
    detected_words = []
    transcribed_text = ""
    if filipino_rec.AcceptWaveform(audio_chunk):
        result = json.loads(filipino_rec.Result())
        text = result.get("text", "")
        if text:
            transcribed_text += text + " "
            words = get_detected_words(text)
            detected_words.extend(words)
    if english_rec.AcceptWaveform(audio_chunk):
        result = json.loads(english_rec.Result())
        text = result.get("text", "")
        if text:
            transcribed_text += text + " "
            words = get_detected_words(text)
            detected_words.extend(words)
    has_profanity = len(detected_words) > 0
    return {"has_profanity": has_profanity, "detected_words": detected_words, "transcribed_text": transcribed_text.strip()}
