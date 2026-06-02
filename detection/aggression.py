import time
import numpy as np
from detection.thresholds import YAMNET_THRESHOLD, DURATION_THRESHOLD, TONE_RMS_THRESHOLD, PROFANITY_MIN_RMS, get_final_severity, get_final_confidence, get_severity, get_severity_by_duration
from model.yamnet_infer import is_aggressive_sound
from model.tone_analyzer import analyze_tone, get_tone_confidence_boost, classify_emotion
from audio.capture import get_waveform_snapshot

LOUD_RMS_THRESHOLD = TONE_RMS_THRESHOLD

class AggressionDetector:
    def __init__(self):
        self.aggressive_start_time = None
        self.last_alert_time = 0
        self.alert_cooldown = 10.0

    def process(self, yamnet_class, yamnet_score, has_profanity, detected_words,
                audio_np=None, transcribed_text=""):
        current_time = time.time()
        reported_yamnet_score = float(yamnet_score)
        detected_words = detected_words or []

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

        print(f"[TONE ANALYSIS] RMS={tone['rms']:.0f} Variance={tone['energy_variance']:.0f} ZCR={tone['zero_crossing_rate']:.3f} Aggressive={aggressive_tone}")

        is_aggressive = is_aggressive_sound(yamnet_class, yamnet_score, YAMNET_THRESHOLD)

        if yamnet_class == "Speech" and yamnet_score >= 0.90:
            if is_loud and aggressive_tone:
                is_aggressive = True
            else:
                is_aggressive = False

        if has_profanity:
            if aggressive_tone or rms > PROFANITY_MIN_RMS:
                is_aggressive = True
                yamnet_score = max(yamnet_score, 0.75)
            else:
                is_aggressive = False

        if is_aggressive:
            if self.aggressive_start_time is None:
                self.aggressive_start_time = current_time
                print(f"[DETECTION] Aggressive sound: {yamnet_class} ({yamnet_score:.2f}) RMS={rms}")
            duration = current_time - self.aggressive_start_time
            print(f"[DURATION] Aggressive for {duration:.1f}s")
        else:
            if self.aggressive_start_time is not None:
                duration = current_time - self.aggressive_start_time
                if duration >= DURATION_THRESHOLD:
                    if current_time - self.last_alert_time >= self.alert_cooldown:
                        tone_boost = get_tone_confidence_boost(tone)
                        confidence = min(1.0, get_final_confidence(yamnet_score, has_profanity) + tone_boost)
                        severity = get_severity_by_duration(duration)
                        self.last_alert_time = current_time
                        self.aggressive_start_time = None
                        print(f"[ALERT] Aggression detected! Severity: {severity} Confidence: {confidence:.2f} RMS={rms} Emotion={emotion} ToneBoost=+{tone_boost:.2f}")
                        return {
                            "should_alert": True,
                            "severity": severity,
                            "confidence": confidence,
                            "duration": round(duration, 2),
                            "transcribed_text": transcribed_text,
                            "detected_words": detected_words,
                            "yamnet_class": yamnet_class,
                            "yamnet_score": reported_yamnet_score,
                            "emotion": emotion,
                            "tone_data": tone,
                            "waveform_snapshot": waveform_snapshot,
                            "has_profanity": has_profanity,
                            "tone_aggressive": aggressive_tone
                        }
                else:
                    print(f"[DETECTION] Sound stopped too short ({duration:.1f}s) - ignored")
            if self.aggressive_start_time is not None:
                print(f"[DETECTION] Sound stopped (RMS={rms})")
            self.aggressive_start_time = None

        return {
            "should_alert": False,
            "transcribed_text": transcribed_text,
            "detected_words": detected_words,
            "yamnet_class": yamnet_class,
            "yamnet_score": reported_yamnet_score,
            "emotion": emotion,
            "tone_data": tone,
            "waveform_snapshot": waveform_snapshot,
            "has_profanity": has_profanity,
            "tone_aggressive": aggressive_tone
        }
