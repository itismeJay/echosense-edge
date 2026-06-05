import time
import numpy as np

from config import SAMPLE_RATE
from detection.thresholds import (
    YAMNET_THRESHOLD,
    DURATION_THREAT,
    DURATION_HARD_TRIGGER,
    DURATION_REPEATED_WORD,
    DURATION_MEDIUM_TRIGGER,
    DURATION_SOFT_TRIGGER,
    ALERT_COOLDOWN,
    PROFANITY_BOOST,
    TONE_RMS_THRESHOLD,
    TONE_VARIANCE_THRESHOLD,
    get_time_severity,
)
from model.yamnet_infer import is_aggressive_sound, run_yamnet_scan
from model.whisper_stt import transcribe_and_check
from model.tone_analyzer import analyze_tone, get_tone_confidence_boost, classify_emotion
from detection.context_gate import ContextGate
from audio.capture import get_waveform_snapshot

SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2}

# Threat words demand the fastest reaction (DURATION_THREAT).
THREAT_WORDS = {
    "patyon tika", "patyon ka nako", "kill you",
    "papatayin kita", "mamamatay ka", "away ta",
    "gusto kag sumbagay", "suwayi rag duol",
    "sumbagay ta",
}


def get_required_duration(
    hard_hits:   list,
    soft_hits:   list,
    is_repeated: bool
) -> float:
    """
    Returns the minimum seconds of sustained aggression needed before firing.

    - Threat words      = 1.5s (immediate danger)
    - Hard trigger      = 2.0s (severe, unambiguous)
    - Repeated word     = 2.0s (targeting already confirmed by repetition)
    - 2+ soft words     = 3.0s (clear pattern)
    - Single soft only  = 5.0s (need sustained evidence)
    """
    # Threats = near-immediate response
    if any(w in THREAT_WORDS for w in hard_hits):
        return DURATION_THREAT          # 1.5 seconds

    # Hard trigger = severe words, short duration needed
    if len(hard_hits) > 0:
        return DURATION_HARD_TRIGGER    # 2.0 seconds

    # Repeated word = targeting confirmed by repetition
    if is_repeated:
        return DURATION_REPEATED_WORD   # 2.0 seconds

    # 2+ soft words together = clear pattern
    if len(soft_hits) >= 2:
        return DURATION_MEDIUM_TRIGGER  # 3.0 seconds

    # Single soft word only = need full evidence
    return DURATION_SOFT_TRIGGER        # 5.0 seconds


class AggressionDetector:
    """5-layer acoustic bullying detector. Operates on a rolling window of the
    last few seconds of *voiced* audio (supplied by main.py).

    Layer order (cheapest discriminator first, so YAMNet/tone only run when a
    blacklist word was actually spoken):
      1. Faster-Whisper STT + blacklist  → no profanity ⇒ stop
      2. YAMNet aggressive class ≥ 0.72   → fail ⇒ stop
      3. Tone RMS/variance + emotion ∈ {angry, aggressive, distressed} → fail ⇒ stop
      4. ContextGate (laughter / repetition / hard vs soft) → not bullying ⇒ stop
      5. Tiered duration (per-tier required seconds) + cooldown (≥60s) → fail ⇒ stop
    All pass ⇒ build + return alert payload (else return None).

    `duration_seconds` is the amount of *continuous voiced audio* accumulated for
    the current utterance (tracked by main.py). It is sample-based, so it is
    immune to per-iteration STT/inference time on the Pi.
    """

    def __init__(self, interpreter=None, class_names=None):
        self.interpreter = interpreter
        self.class_names = class_names
        self.context_gate = ContextGate()

        self.last_alert_time = 0.0
        self.alert_cooldown = ALERT_COOLDOWN

    def process(self, audio_np, duration_seconds=None):
        if audio_np is None or len(audio_np) == 0:
            return None

        # ---------- LAYER 1: STT + blacklist ----------
        stt = transcribe_and_check(audio_np)
        if not stt["has_profanity"]:
            return None

        detected_words = stt["detected_words"]
        hard_hits = stt["hard_hits"]
        soft_hits = stt["soft_hits"]
        transcribed_text = stt["transcribed_text"]
        word_severity = stt["severity"]
        categories = stt["categories"]
        language = stt.get("language", "unknown")

        # ---------- LAYER 2: YAMNet aggressive class ----------
        if self.interpreter is None or not self.class_names:
            print("[LAYER 2] YAMNet not available — cannot confirm sound profile")
            return None
        yamnet_class, yamnet_score = run_yamnet_scan(
            self.interpreter, audio_np, self.class_names
        )
        yamnet_score = float(yamnet_score)
        print(f"[LAYER 2] YAMNet: {yamnet_class} ({yamnet_score:.2f})")
        if not is_aggressive_sound(yamnet_class, yamnet_score, YAMNET_THRESHOLD):
            print(f"[LAYER 2] Sound profile not aggressive enough "
                  f"(need {YAMNET_THRESHOLD}) — no alert")
            return None

        # ---------- LAYER 3: Tone / emotion ----------
        tone = analyze_tone(audio_np)
        emotion = classify_emotion(tone)
        rms = float(tone["rms"])
        variance = float(tone["energy_variance"])
        print(f"[LAYER 3] Tone RMS={rms:.0f} Var={variance:.0f} "
              f"ZCR={tone['zero_crossing_rate']:.3f} Emotion={emotion}")
        if rms < TONE_RMS_THRESHOLD:
            print(f"[LAYER 3] RMS {rms:.0f} < {TONE_RMS_THRESHOLD} (too quiet) — no alert")
            return None
        if variance < TONE_VARIANCE_THRESHOLD:
            print(f"[LAYER 3] Variance {variance:.0f} < {TONE_VARIANCE_THRESHOLD} (not bursty) — no alert")
            return None
        if emotion not in ("angry", "aggressive", "distressed"):
            print(f"[LAYER 3] Emotion '{emotion}' not angry/aggressive/distressed — no alert")
            return None

        # ---------- LAYER 4: Context gate ----------
        ctx = self.context_gate.check(
            detected_words=detected_words,
            emotion=emotion,
            transcribed_text=transcribed_text,
            is_casual=stt["is_casual"],
            hard_hits=hard_hits,
            soft_hits=soft_hits,
        )
        print(f"[LAYER 4] Context: {ctx['reason']} "
              f"(repetitions={ctx['max_repetitions']}, laughter={ctx['has_laughter']})")
        if not ctx["is_bullying_context"]:
            return None

        # ---------- LAYER 5: Tiered duration + cooldown ----------
        # `duration` is the continuous voiced-audio length accumulated by main.py
        # for this utterance (sample-based — immune to STT/inference latency).
        if duration_seconds is None:
            duration_seconds = len(audio_np) / SAMPLE_RATE
        duration = float(duration_seconds)

        # How long this severity tier must be sustained before firing.
        is_repeated = ctx.get("is_repeated", False)
        required_duration = get_required_duration(
            hard_hits=hard_hits,
            soft_hits=soft_hits,
            is_repeated=is_repeated,
        )
        print(f"[LAYER 5] Voiced {duration:.1f}s (need {required_duration:.1f}s "
              f"for this tier)")

        if duration < required_duration:
            print(f"[LAYER 5] Not yet sustained {required_duration:.1f}s — building")
            return None  # not long enough yet, keep accumulating

        current_time = time.time()
        if current_time - self.last_alert_time < self.alert_cooldown:
            remaining = self.alert_cooldown - (current_time - self.last_alert_time)
            print(f"[LAYER 5] Cooldown active ({remaining:.0f}s left) — suppressing")
            return None

        # ---------- ALL LAYERS PASSED — fire ----------
        time_severity = get_time_severity(duration)
        severity = max(
            [word_severity, time_severity],
            key=lambda s: SEVERITY_ORDER[s],
        )
        # Which tier gated this alert (for teacher-facing explanation).
        if required_duration == DURATION_THREAT:
            duration_gate = "threat"
        elif len(hard_hits) > 0 and required_duration == DURATION_HARD_TRIGGER:
            duration_gate = "hard"
        elif is_repeated:
            duration_gate = "repeated"
        elif required_duration == DURATION_MEDIUM_TRIGGER:
            duration_gate = "medium"
        else:
            duration_gate = "soft"

        tone_boost = get_tone_confidence_boost(tone)
        confidence = min(1.0, yamnet_score + PROFANITY_BOOST + tone_boost)

        self.last_alert_time = current_time

        print(f"[ALERT] BULLYING CONFIRMED | Severity={severity} "
              f"Confidence={confidence:.2f} Duration={duration:.1f}s "
              f"Gate={duration_gate} (need {required_duration:.1f}s) "
              f"Emotion={emotion} Words={detected_words} Categories={categories}")

        return {
            "should_alert": True,
            "severity": severity,
            "confidence": confidence,
            "duration": round(duration, 2),
            "required_duration": required_duration,
            "actual_duration": round(duration, 2),
            "duration_gate": duration_gate,
            "transcribed_text": transcribed_text,
            "detected_words": detected_words,
            "categories": categories,
            "hard_hits": hard_hits,
            "soft_hits": soft_hits,
            "language": language,
            "yamnet_class": yamnet_class,
            "yamnet_score": yamnet_score,
            "emotion": emotion,
            "tone_data": tone,
            "waveform_snapshot": get_waveform_snapshot(audio_np),
            "has_profanity": True,
            "tone_aggressive": tone["is_aggressive_tone"],
        }
