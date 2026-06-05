import numpy as np

VAD_RMS_THRESHOLD = 120  # sweet spot: blocks ambient noise, passes normal speaking voice,
                         # cuts Whisper hallucinations (80 was too low, 150 blocked normal speech)


def is_voice_present(audio_np: np.ndarray) -> bool:
    if len(audio_np) == 0:
        return False
    rms = np.sqrt(np.mean(audio_np.astype(float) ** 2))
    return rms > VAD_RMS_THRESHOLD
