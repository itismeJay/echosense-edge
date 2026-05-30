import numpy as np


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

    # AGGRESSIVE TONE = loud AND uneven/tense AND spiky voice
    # These thresholds distinguish bullying tone from casual/joking tone:
    # - rms > 800: must be audibly loud (not a whisper or quiet chat)
    # - energy_variance > 5000: energy must be uneven (not steady casual speech)
    # - zcr > 0.1: voice must be tense (not relaxed joking tone)
    is_aggressive = bool(
        rms > 800 and
        energy_variance > 5000 and
        zcr > 0.1
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
    if tone_result["rms"] > 2000:
        score += 1
    if tone_result["energy_variance"] > 15000:
        score += 1
    if tone_result["zero_crossing_rate"] > 0.2:
        score += 1

    if score >= 3:
        return 0.10
    elif score >= 2:
        return 0.05
    else:
        return 0.0
