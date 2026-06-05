import numpy as np

VAD_RMS_THRESHOLD = 150   # was 80 — skip very quiet ambient noise (cuts Whisper hallucinations)


def is_voice_present(audio_np: np.ndarray) -> bool:
    if len(audio_np) == 0:
        return False
    rms = np.sqrt(np.mean(audio_np.astype(float) ** 2))
    return rms > VAD_RMS_THRESHOLD
