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
    QUIET_BASE_CONFIDENCE,
    get_time_severity,
)
from model.yamnet_infer import is_aggressive_sound, run_yamnet_scan
from model.whisper_stt import transcribe_and_check
from model.tone_analyzer import analyze_tone, get_tone_confidence_boost, classify_emotion
from detection.context_gate import ContextGate
from audio.capture import get_waveform_snapshot
from sender.shadow_log import log_near_miss

SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2}
ANGRY_EMOTIONS = {"angry", "aggressive", "distressed"}

# Threat words demand the fastest reaction (DURATION_THREAT).
THREAT_WORDS = {
    "patyon tika", "patyon ka nako", "kill you",
    "papatayin kita", "mamamatay ka",
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

        detected_words   = stt["detected_words"]
        hard_hits        = stt["hard_hits"]
        soft_hits        = stt["soft_hits"]
        transcribed_text = stt["transcribed_text"]
        word_severity    = stt["severity"]
        categories       = stt["categories"]
        language         = stt.get("language", "unknown")
        is_casual        = stt["is_casual"]
        has_hard         = len(hard_hits) > 0

        # ---------- Repetition context (shared by both tracks) ----------
        ctx = self.context_gate.check(
            detected_words=detected_words,
            emotion="neutral",          # tone not needed to track repetition
            transcribed_text=transcribed_text,
            is_casual=is_casual,
            hard_hits=hard_hits,
            soft_hits=soft_hits,
        )
        is_repeated = ctx.get("is_repeated", False)

        # Tone is computed once — used by Rule 4 / Track A and logged as evidence.
        tone       = analyze_tone(audio_np)
        emotion    = classify_emotion(tone)
        rms        = float(tone["rms"])
        variance   = float(tone["energy_variance"])
        tone_boost = get_tone_confidence_boost(tone)

        if duration_seconds is None:
            duration_seconds = len(audio_np) / SAMPLE_RATE
        duration = float(duration_seconds)

        print(f"[STT] '{transcribed_text}' hard={hard_hits} soft={soft_hits} "
              f"repeated={is_repeated} casual={is_casual}")
        print(f"[TONE] RMS={rms:.0f} Var={variance:.0f} Emotion={emotion}")

        pending_track = "?"

        def _near_miss(layers_passed, reason):
            """Record a near-miss (profanity heard, 3+ stages passed, no alert)
            then return None. Lets us tune the system from logs/shadow_log.jsonl
            without changing live behavior."""
            print(f"[NO ALERT] {reason}")
            if layers_passed >= 3:
                log_near_miss({
                    "transcript":     transcribed_text,
                    "detected_words": detected_words,
                    "layers_passed":  layers_passed,
                    "track":          pending_track,
                    "emotion":        emotion,
                    "rms":            round(rms, 1),
                    "duration":       round(duration, 2),
                    "severity":       word_severity,
                    "reason":         reason,
                })
            return None

        # ---------- Rule 5: laughter ALWAYS suppresses (any track) ----------
        if is_casual:
            return _near_miss(1, "Laughter present — kantiyawan, suppressed (Rule 5)")

        # ================= TRACK B — quiet / relational bullying =================
        # No scream required. Fires on repetition, 2+ hard words, hard+angry, or
        # 2+ soft words (RA 10627 'repeated / targeted' behavior).
        pending_track = "B"
        if self._should_fire_track_b(hard_hits, soft_hits, is_repeated, emotion, is_casual):
            print("[TRACK B] Quiet bullying criteria met")
            if is_repeated or len(hard_hits) >= 2 or (has_hard and emotion in ANGRY_EMOTIONS):
                required_duration = DURATION_HARD_TRIGGER      # 2.0s
                duration_gate = "repeated" if is_repeated else "hard"
            else:                                              # 2+ soft words
                required_duration = DURATION_MEDIUM_TRIGGER    # 3.0s
                duration_gate = "medium"

            if duration < required_duration:
                return _near_miss(3, f"Track B building "
                                     f"({duration:.1f}s < {required_duration:.1f}s)")

            now = time.time()
            if now - self.last_alert_time < self.alert_cooldown:
                remaining = self.alert_cooldown - (now - self.last_alert_time)
                return _near_miss(4, f"Track B cooldown ({remaining:.0f}s left)")

            confidence = min(
                1.0,
                QUIET_BASE_CONFIDENCE + PROFANITY_BOOST
                + (0.10 if is_repeated else 0.0) + tone_boost,
            )
            return self._fire(
                track="B", word_severity=word_severity, duration=duration,
                required_duration=required_duration, duration_gate=duration_gate,
                confidence=confidence, transcribed_text=transcribed_text,
                detected_words=detected_words, categories=categories,
                hard_hits=hard_hits, soft_hits=soft_hits, language=language,
                yamnet_class="(quiet track)", yamnet_score=0.0,
                emotion=emotion, tone=tone, audio_np=audio_np,
            )

        # ================= TRACK A — loud / shouted aggression =================
        pending_track = "A"
        if self.interpreter is None or not self.class_names:
            return _near_miss(1, "Track B not met; YAMNet unavailable for Track A")

        yamnet_class, yamnet_score = run_yamnet_scan(
            self.interpreter, audio_np, self.class_names
        )
        yamnet_score = float(yamnet_score)
        acoustic_aggressive = is_aggressive_sound(yamnet_class, yamnet_score, YAMNET_THRESHOLD)
        loud_tone = (
            rms >= TONE_RMS_THRESHOLD
            and variance >= TONE_VARIANCE_THRESHOLD
            and emotion in ANGRY_EMOTIONS
        )
        print(f"[TRACK A] YAMNet={yamnet_class} ({yamnet_score:.2f}) "
              f"aggressive={acoustic_aggressive} loud_tone={loud_tone}")

        if not (acoustic_aggressive and loud_tone):
            return _near_miss(2, "Track A: not loud + aggressive enough")

        required_duration = get_required_duration(hard_hits, soft_hits, is_repeated)
        if duration < required_duration:
            return _near_miss(3, f"Track A building "
                                 f"({duration:.1f}s < {required_duration:.1f}s)")

        now = time.time()
        if now - self.last_alert_time < self.alert_cooldown:
            remaining = self.alert_cooldown - (now - self.last_alert_time)
            return _near_miss(4, f"Track A cooldown ({remaining:.0f}s left)")

        if required_duration == DURATION_THREAT:
            duration_gate = "threat"
        elif has_hard and required_duration == DURATION_HARD_TRIGGER:
            duration_gate = "hard"
        elif is_repeated:
            duration_gate = "repeated"
        elif required_duration == DURATION_MEDIUM_TRIGGER:
            duration_gate = "medium"
        else:
            duration_gate = "soft"

        confidence = min(1.0, yamnet_score + PROFANITY_BOOST + tone_boost)
        return self._fire(
            track="A", word_severity=word_severity, duration=duration,
            required_duration=required_duration, duration_gate=duration_gate,
            confidence=confidence, transcribed_text=transcribed_text,
            detected_words=detected_words, categories=categories,
            hard_hits=hard_hits, soft_hits=soft_hits, language=language,
            yamnet_class=yamnet_class, yamnet_score=yamnet_score,
            emotion=emotion, tone=tone, audio_np=audio_np,
        )

    def _should_fire_track_b(self, hard_hits, soft_hits, is_repeated, emotion, is_casual):
        """Track B: quiet bullying detection. Fires at normal speaking volume —
        no scream required. Encodes the RA 10627 'repeated / targeted' rules."""
        # Rule 5: laughter always suppresses.
        if is_casual:
            return False
        # Rule 2 / 7: same word repeated 2+ times in 30s = targeting.
        if is_repeated:
            return True
        # Rule 3: two or more hard triggers together = directed insult.
        if len(hard_hits) >= 2:
            return True
        # Rule 4: one hard trigger + angry/aggressive tone.
        if len(hard_hits) >= 1 and emotion in ANGRY_EMOTIONS:
            return True
        # Rule 8: two or more soft triggers together = shaming pattern.
        if len(soft_hits) >= 2:
            return True
        # Rule 1 / 6: single word, calm, not repeated = NOT bullying.
        return False

    def _fire(self, *, track, word_severity, duration, required_duration,
              duration_gate, confidence, transcribed_text, detected_words,
              categories, hard_hits, soft_hits, language, yamnet_class,
              yamnet_score, emotion, tone, audio_np):
        """Stamp the cooldown, log, and build the alert payload."""
        time_severity = get_time_severity(duration)
        severity = max([word_severity, time_severity], key=lambda s: SEVERITY_ORDER[s])
        self.last_alert_time = time.time()

        print(f"[ALERT] BULLYING DETECTED | Track={track} Severity={severity} "
              f"Confidence={confidence:.2f} Duration={duration:.1f}s "
              f"Gate={duration_gate} Emotion={emotion} Words={detected_words}")

        return {
            "should_alert":      True,
            "track":             track,
            "severity":          severity,
            "confidence":        confidence,
            "duration":          round(duration, 2),
            "required_duration": required_duration,
            "actual_duration":   round(duration, 2),
            "duration_gate":     duration_gate,
            "transcribed_text":  transcribed_text,
            "detected_words":    detected_words,
            "categories":        categories,
            "hard_hits":         hard_hits,
            "soft_hits":         soft_hits,
            "language":          language,
            "yamnet_class":      yamnet_class,
            "yamnet_score":      yamnet_score,
            "emotion":           emotion,
            "tone_data":         tone,
            "waveform_snapshot": get_waveform_snapshot(audio_np),
            "has_profanity":     True,
            "tone_aggressive":   tone["is_aggressive_tone"],
        }
