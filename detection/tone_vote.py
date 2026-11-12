"""Tone vote: is a monitored utterance said angrily, or with laughter?

Volume alone never means aggressive. ``angry_tone`` needs the speech-emotion
model to hear anger AND one loudness/shout cue:

    angry_tone = emotion2vec angry >= T_angry
                 AND (YAMNet max(Shout, Yell, Screaming, Children shouting,
                      Battle cry) >= T_shout
                      OR speech RMS >= T_rms_ratio x rolling speech baseline
                      OR an ElevenLabs tag says shout/yell/scream)
    laughter   = YAMNet max(Laughter, Giggle, Chuckle...) >= T_laugh
                 OR an ElevenLabs tag says laugh/giggle/chuckle

Tone never creates an alert: it only changes the severity of an utterance
that already holds a monitored word (detection/severity.py).

Calibration (2026-11-13, tools/tone_report.py, 129 calmly read clips:
~/asr-bench/clips + ~/echosense-testset): emotion2vec alone said "angry"
(up to 1.00) on 8 calm clips, so anger needs a loud cue. Calm maxima were
shout 0.016, laughter 0.06, speech-loudness ratio 1.98 (near and far clips
mixed); angry_tone was 0/129 with the defaults below. Re-check them with the
recorded angry/laughing clips (tools/record_tone_clips.py).
Thresholds come from ``ECHOSENSE_TONE_*`` (see edge.env.example).
"""

from __future__ import annotations

from dataclasses import dataclass
import os

SHOUT_TAG_WORDS = ("shout", "yell", "scream")
LAUGH_TAG_WORDS = ("laugh", "giggl", "chuckl", "snicker")


def _env_float(env, name, default):
    try:
        value = str(env.get(name) or "").strip()
        return float(value) if value else float(default)
    except ValueError:
        return float(default)


@dataclass(frozen=True, slots=True)
class ToneThresholds:
    angry: float = 0.60
    shout: float = 0.30
    rms_ratio: float = 2.5
    laugh: float = 0.30

    @classmethod
    def from_environment(cls, environ=None):
        env = os.environ if environ is None else environ
        base = cls()
        return cls(
            angry=_env_float(env, "ECHOSENSE_TONE_ANGRY_MIN", base.angry),
            shout=_env_float(env, "ECHOSENSE_TONE_SHOUT_MIN", base.shout),
            rms_ratio=_env_float(env, "ECHOSENSE_TONE_RMS_RATIO", base.rms_ratio),
            laugh=_env_float(env, "ECHOSENSE_TONE_LAUGH_MIN", base.laugh),
        )


@dataclass(frozen=True, slots=True)
class ToneVote:
    angry_tone: bool = False
    laughter: bool = False
    emotion_label: str = "unknown"
    emotion_score: float = 0.0
    angry_score: float = 0.0
    shout_score: float = 0.0
    laughter_score: float = 0.0
    rms_ratio: float | None = None
    # Why angry_tone holds: "shouting detected", "much louder than usual
    # speech", "shout tag". Empty when angry_tone is False.
    loud_cues: tuple[str, ...] = ()
    laughter_sources: tuple[str, ...] = ()
    audio_tags: tuple[str, ...] = ()

    def evidence(self):
        """Short machine-readable strings for severity supporting_evidence."""

        items = [f"emotion:{self.emotion_label}:{self.emotion_score:.2f}",
                 f"emotion_angry_score:{self.angry_score:.2f}"]
        if self.angry_tone:
            items.append("angry_tone")
            items.extend(f"angry_tone_cue:{cue}" for cue in self.loud_cues)
        if self.laughter:
            items.extend(f"laughter_source:{src}" for src in self.laughter_sources)
        return tuple(items)


def _tag_hit(tags, words):
    return any(word in str(tag).lower() for tag in tags for word in words)


def vote(signals=None, audio_tags=(), *, linked_laughter_score=0.0,
         linked_shout_score=0.0, thresholds=None):
    """Combine one utterance's tone signals (model/emotion_recognizer.py).

    ``linked_*`` are YAMNet scores from acoustic windows time-linked to the
    utterance (laughter often comes right after the words).
    """

    thresholds = thresholds or ToneThresholds.from_environment()
    signals = signals or {}
    tags = tuple(str(tag).lower() for tag in (audio_tags or ()))
    emotion = signals.get("emotion") or {}
    scores = emotion.get("scores") or {}
    angry_score = float(scores.get("angry", 0.0) or 0.0)
    yamnet = signals.get("yamnet") or {}
    shout = max(float(yamnet.get("shout_score") or 0.0), float(linked_shout_score or 0.0))
    laugh = max(float(yamnet.get("laughter_score") or 0.0), float(linked_laughter_score or 0.0))
    ratio = (signals.get("rms") or {}).get("ratio")

    cues = []
    if shout >= thresholds.shout:
        cues.append("shouting detected")
    if ratio is not None and ratio >= thresholds.rms_ratio:
        cues.append("much louder than usual speech")
    if _tag_hit(tags, SHOUT_TAG_WORDS):
        cues.append("shouting heard by transcriber")
    angry_tone = angry_score >= thresholds.angry and bool(cues)

    sources = []
    if laugh >= thresholds.laugh:
        sources.append("yamnet")
    if _tag_hit(tags, LAUGH_TAG_WORDS):
        sources.append("transcriber_tag")
    return ToneVote(
        angry_tone=angry_tone,
        laughter=bool(sources),
        emotion_label=str(emotion.get("label") or "unknown"),
        emotion_score=float(emotion.get("score") or 0.0),
        angry_score=angry_score,
        shout_score=shout,
        laughter_score=laugh,
        rms_ratio=ratio,
        loud_cues=tuple(cues) if angry_tone else (),
        laughter_sources=tuple(sources),
        audio_tags=tags,
    )
