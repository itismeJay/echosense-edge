"""
Standalone test for prosodic tone analysis (model/tone_analyzer.py).

Synthesizes two audio clips — one that mimics loud, tense, uneven *aggressive*
speech and one that mimics quiet, steady *casual* speech — then confirms that
analyze_tone() classifies them correctly. No hardware or backend needed.

Run:  echosense-env/bin/python3 test_tone.py
"""
import numpy as np
from model.tone_analyzer import analyze_tone, get_tone_confidence_boost

SAMPLE_RATE = 16000


def make_aggressive_audio():
    """Loud, tense (high-frequency), uneven (bursty) — like an angry shout."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, SAMPLE_RATE, endpoint=False)
    # High-frequency carrier => many zero crossings (tense voice quality).
    carrier = np.sin(2 * np.pi * 1800 * t)
    # Bursty amplitude envelope => uneven energy (spikes and drops).
    env = np.ones(SAMPLE_RATE)
    for start in range(0, SAMPLE_RATE, 4000):
        env[start:start + 1500] *= 4.0
    signal = carrier * env * 4000 + rng.standard_normal(SAMPLE_RATE) * 1500
    return np.clip(signal, -32768, 32767).astype(np.int16)


def make_casual_audio():
    """Quiet, low-pitched, steady — like relaxed chit-chat."""
    rng = np.random.default_rng(1)
    t = np.linspace(0, 1, SAMPLE_RATE, endpoint=False)
    signal = np.sin(2 * np.pi * 200 * t) * 300 + rng.standard_normal(SAMPLE_RATE) * 10
    return np.clip(signal, -32768, 32767).astype(np.int16)


def show(label, tone):
    print(f"\n--- {label} ---")
    print(f"  RMS               : {tone['rms']:.0f}")
    print(f"  energy_variance   : {tone['energy_variance']:.0f}")
    print(f"  zero_crossing_rate: {tone['zero_crossing_rate']:.3f}")
    print(f"  peak_to_average   : {tone['peak_to_average']:.2f}")
    print(f"  is_aggressive_tone: {tone['is_aggressive_tone']}")
    print(f"  confidence_boost  : +{get_tone_confidence_boost(tone):.2f}")


def main():
    print("=" * 50)
    print("  Prosodic Tone Analysis — Test")
    print("=" * 50)

    aggressive = analyze_tone(make_aggressive_audio())
    casual = analyze_tone(make_casual_audio())

    show("AGGRESSIVE clip (loud, tense, uneven)", aggressive)
    show("CASUAL clip (quiet, steady)", casual)

    assert aggressive["is_aggressive_tone"] is True, \
        "Aggressive clip should be flagged as aggressive tone"
    assert casual["is_aggressive_tone"] is False, \
        "Casual clip should NOT be flagged as aggressive tone"

    print("\n" + "=" * 50)
    print("  PASS: aggressive=True, casual=False")
    print("  The system can now tell HOW a word was said.")
    print("=" * 50)


if __name__ == "__main__":
    main()
