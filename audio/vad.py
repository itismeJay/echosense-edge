import numpy as np

VAD_RMS_THRESHOLD = 50   # EMEET M0 Plus AGC + noise cancellation attenuates all audio; Grade 6
                         # kids speaking normally at 1 m land at RMS 50-150 after processing, so
                         # 120 blocked most young voices. 50 passes them (watch for more Whisper
                         # hallucinations on near-silence — Whisper's own VAD/no_speech is backstop).


def is_voice_present(audio_np: np.ndarray) -> bool:
    if len(audio_np) == 0:
        return False
    rms = np.sqrt(np.mean(audio_np.astype(float) ** 2))
    return rms > VAD_RMS_THRESHOLD
