import time
import numpy as np
from detection.thresholds import YAMNET_THRESHOLD, DURATION_THRESHOLD, get_final_severity, get_final_confidence, get_severity
from model.yamnet_infer import is_aggressive_sound
from model.tone_analyzer import analyze_tone, get_tone_confidence_boost, classify_emotion
from audio.capture import get_waveform_snapshot

# Volume threshold — RMS above this = loud/aggressive
LOUD_RMS_THRESHOLD = 800

class AggressionDetector:
    def __init__(self):
        self.aggressive_start_time = None
        self.last_alert_time = 0
        self.alert_cooldown = 10.0

    def process(self, yamnet_class, yamnet_score, has_profanity, detected_words,
                audio_np=None, transcribed_text=""):
        current_time = time.time()

        # Keep the genuine YAMNet score for evidence — `yamnet_score` may get
        # floored below when profanity is present.
        reported_yamnet_score = float(yamnet_score)
        detected_words = detected_words or []

        # Prosodic tone analysis — checks HOW something was said (acoustic),
        # not just WHAT was said. Pure sound analysis, no words needed.
        if audio_np is not None:
            tone = analyze_tone(audio_np)
            waveform_snapshot = get_waveform_snapshot(audio_np)
        else:
            tone = {"rms": 0.0, "energy_variance": 0.0, "zero_crossing_rate": 0.0,
                    "peak_to_average": 0.0, "is_aggressive_tone": False}
            waveform_snapshot = []

        emotion = classify_emotion(tone)
        rms = int(tone["rms"])
        aggressive_tone = tone["is_aggressive_tone"]
        is_loud = rms >= LOUD_RMS_THRESHOLD

        # CASE 4 — surface the prosodic features every iteration.
        print(f"[TONE ANALYSIS] RMS={tone['rms']:.0f} Variance={tone['energy_variance']:.0f} "
              f"ZCR={tone['zero_crossing_rate']:.3f} Aggressive={aggressive_tone}")

        # YAMNet aggressive-class path — unchanged.
        is_aggressive = is_aggressive_sound(yamnet_class, yamnet_score, YAMNET_THRESHOLD)

        # CASE 2 — High-confidence Speech now also requires an aggressive
        # acoustic tone (loud AND tense/uneven), not just loudness.
        if yamnet_class == "Speech" and yamnet_score >= 0.90:
            if is_loud and aggressive_tone:
                is_aggressive = True
            else:
                is_aggressive = False  # confident speech but calm/quiet = ignore

        # CASE 1 — Profanity gating via acoustic tone (the core fix).
        # "gago" while joking/quietly = casual tone = NO alert.
        # "GAGO!" shouted angrily = aggressive tone = alert.
        if has_profanity:
            if aggressive_tone:
                is_aggressive = True
                yamnet_score = max(yamnet_score, 0.75)
            else:
                is_aggressive = False

        if is_aggressive:
            if self.aggressive_start_time is None:
                self.aggressive_start_time = current_time
                print(f"[DETECTION] Aggressive sound: {yamnet_class} ({yamnet_score:.2f}) RMS={rms}")
            duration = current_time - self.aggressive_start_time
            if duration >= DURATION_THRESHOLD:
                if current_time - self.last_alert_time >= self.alert_cooldown:
                    # CASE 3 — stack the acoustic tone boost on top of the
                    # profanity boost, then re-derive severity from it.
                    tone_boost = get_tone_confidence_boost(tone)
                    confidence = min(1.0, get_final_confidence(yamnet_score, has_profanity) + tone_boost)
                    severity = get_severity(confidence)
                    self.last_alert_time = current_time
                    self.aggressive_start_time = None
                    print(f"[ALERT] Aggression detected! Severity: {severity} "
                          f"Confidence: {confidence:.2f} RMS={rms} Emotion={emotion} "
                          f"ToneBoost=+{tone_boost:.2f}")
                    return {"should_alert": True, "severity": severity, "confidence": confidence,
                            "duration": round(duration, 2),
                            "transcribed_text": transcribed_text,
                            "detected_words": detected_words,
                            "yamnet_class": yamnet_class,
                            "yamnet_score": reported_yamnet_score,
                            "emotion": emotion,
                            "tone_data": tone,
                            "waveform_snapshot": waveform_snapshot,
                            "has_profanity": has_profanity,
                            "tone_aggressive": aggressive_tone}
        else:
            if self.aggressive_start_time is not None:
                print(f"[DETECTION] Sound stopped (RMS={rms})")
            self.aggressive_start_time = None
        return {"should_alert": False,
                "transcribed_text": transcribed_text,
                "detected_words": detected_words,
                "yamnet_class": yamnet_class,
                "yamnet_score": reported_yamnet_score,
                "emotion": emotion,
                "tone_data": tone,
                "waveform_snapshot": waveform_snapshot,
                "has_profanity": has_profanity,
                "tone_aggressive": aggressive_tone}
