import numpy as np
from detection.thresholds import (
    TONE_RMS_THRESHOLD,
    TONE_VARIANCE_THRESHOLD,
    TONE_ZCR_THRESHOLD,
)


def analyze_tone(audio_np, sample_rate=16000):
    """
    Analyzes the acoustic tone of audio to determine if it sounds aggressive.
    Uses prosodic features — no words needed, pure sound analysis.

    Returns a dict with:
    - rms: overall loudness (0-32767)
    - energy_variance: how uneven/tense the energy is (higher = more aggressive)
    - zero_crossing_rate: voice tension indicator (higher = more tense)
    - peak_to_average: how much the voice spikes (higher = more aggressive)
    - is_aggressive_tone: True if the combination indicates aggressive speech
    """
    audio_float = audio_np.astype(np.float32)

    # 1. RMS Energy — overall loudness
    rms = float(np.sqrt(np.mean(audio_float**2)))

    # 2. Energy variance — angry speech has uneven energy bursts
    # Joking speech is steady; aggressive speech spikes and drops
    frame_size = 512
    frames = [audio_float[i:i+frame_size]
              for i in range(0, len(audio_float)-frame_size, frame_size)]
    if frames:
        frame_energies = [float(np.sqrt(np.mean(f**2))) for f in frames]
        energy_variance = float(np.var(frame_energies))
    else:
        energy_variance = 0.0

    # 3. Zero Crossing Rate — how often the signal crosses zero
    # Aggressive/tense speech has more rapid direction changes
    zero_crossings = int(np.sum(np.diff(np.sign(audio_np)) != 0))
    zcr = zero_crossings / len(audio_np) if len(audio_np) > 0 else 0.0

    # 4. Peak-to-average ratio — aggressive voices spike sharply
    peak = float(np.max(np.abs(audio_float)))
    par = peak / (rms + 1e-10)

    # AGGRESSIVE TONE = loud AND uneven/tense AND high ZCR.
    # Thresholds from detection/thresholds.py, calibrated for EMEET M0 Plus.
    is_aggressive = bool(
        rms > TONE_RMS_THRESHOLD and
        energy_variance > TONE_VARIANCE_THRESHOLD and
        zcr > TONE_ZCR_THRESHOLD
    )

    return {
        "rms": rms,
        "energy_variance": energy_variance,
        "zero_crossing_rate": zcr,
        "peak_to_average": par,
        "is_aggressive_tone": is_aggressive
    }


def get_tone_confidence_boost(tone_result):
    """
    Returns an additional confidence boost based on how aggressive the tone is.
    This is added ON TOP of the profanity boost.

    Scale:
    - Very aggressive tone (all indicators high) = +0.10 boost
    - Moderately aggressive tone = +0.05 boost
    - Borderline aggressive tone = +0.0 boost
    """
    if not tone_result["is_aggressive_tone"]:
        return 0.0

    score = 0
    if tone_result["rms"] > 800:
        score += 1
    if tone_result["energy_variance"] > 8000:
        score += 1
    if tone_result["zero_crossing_rate"] > 0.15:
        score += 1

    if score >= 3:
        return 0.10
    elif score >= 2:
        return 0.05
    else:
        return 0.0


def classify_emotion(tone_result):
    """
    Maps prosodic tone features to a coarse emotion label for richer alert
    evidence. Checks most-severe first so the strongest match wins.

    Thresholds calibrated for EMEET OfficeCore M0 Plus (clean low-noise signal).

    angry      — loud, very uneven, very tense
    aggressive — raised voice, uneven, tense
    distressed — moderate volume, uneven, tense (e.g. crying/pleading)
    upset      — quiet-to-moderate but uneven
    neutral    — audible but calm
    silent     — effectively no signal
    """
    rms = tone_result["rms"]
    variance = tone_result["energy_variance"]
    zcr = tone_result["zero_crossing_rate"]

    if rms > 800 and variance > 5000 and zcr > 0.12:
        return "angry"
    elif rms > 400 and variance > 2000 and zcr > 0.08:
        return "aggressive"
    elif rms > 150 and variance > 3000 and zcr > 0.10:
        return "distressed"
    elif rms > 80 and variance > 1500:
        return "upset"
    elif rms > 20:
        return "neutral"
    else:
        return "silent"
