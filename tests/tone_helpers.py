"""Shared tone-signal fixtures (see detection/tone_vote.py)."""

from dataclasses import replace

# Clearly angry and shouted: above any plausible configured threshold.
ANGRY_SHOUT = {
    "emotion": {"status": "ok", "label": "angry", "score": 0.97,
                "scores": {"angry": 0.97, "neutral": 0.02}, "ms": 100},
    "yamnet": {"shout_score": 0.95, "shout_label": "Shout",
               "laughter_score": 0.0, "laughter_label": None},
    "rms": {"utterance_rms": 0.3, "baseline_rms": 0.05, "ratio": 6.0, "baseline_count": 20},
}
# Loud (shout class + 6x usual speech) but the emotion model hears no anger.
LOUD_NOT_ANGRY = {
    "emotion": {"status": "ok", "label": "happy", "score": 0.9,
                "scores": {"angry": 0.01, "happy": 0.9}, "ms": 100},
    "yamnet": {"shout_score": 0.95, "shout_label": "Shout",
               "laughter_score": 0.0, "laughter_label": None},
    "rms": {"utterance_rms": 0.3, "baseline_rms": 0.05, "ratio": 6.0, "baseline_count": 20},
}


def with_tone(evidence, signals=ANGRY_SHOUT, tags=()):
    return replace(evidence, tone_signals=dict(signals), audio_tags=tuple(tags))
