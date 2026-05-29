import time
import numpy as np
from detection.thresholds import YAMNET_THRESHOLD, DURATION_THRESHOLD, get_final_severity, get_final_confidence, is_aggressive_tone
from model.yamnet_infer import is_aggressive_sound

# Volume threshold — RMS above this = loud/aggressive
LOUD_RMS_THRESHOLD = 800

class AggressionDetector:
    def __init__(self):
        self.aggressive_start_time = None
        self.last_alert_time = 0
        self.alert_cooldown = 10.0

    def process(self, yamnet_class, yamnet_score, has_profanity, detected_words, audio_np=None):
        current_time = time.time()

        # Calculate volume (RMS)
        rms = 0
        if audio_np is not None:
            rms = int(np.sqrt(np.mean(audio_np.astype(np.float32)**2)))

        is_loud = rms >= LOUD_RMS_THRESHOLD
        is_aggressive = is_aggressive_sound(yamnet_class, yamnet_score, YAMNET_THRESHOLD)

        # High confidence Speech + LOUD = aggressive
        if yamnet_class == "Speech" and yamnet_score >= 0.90 and is_loud:
            is_aggressive = True
        elif yamnet_class == "Speech" and yamnet_score >= 0.90 and not is_loud:
            is_aggressive = False  # deep voice but not loud = ignore

        aggressive_tone = is_aggressive_tone(yamnet_score, has_profanity)

        if has_profanity and aggressive_tone and is_loud:
            is_aggressive = True
            yamnet_score = max(yamnet_score, 0.75)
            print(f"[TONE] Aggressive tone with profanity! RMS={rms}")
        elif has_profanity and (not aggressive_tone or not is_loud):
            print(f"[TONE] Casual speech — ignored (RMS={rms})")
            is_aggressive = False

        if is_aggressive:
            if self.aggressive_start_time is None:
                self.aggressive_start_time = current_time
                print(f"[DETECTION] Aggressive sound: {yamnet_class} ({yamnet_score:.2f}) RMS={rms}")
            duration = current_time - self.aggressive_start_time
            if duration >= DURATION_THRESHOLD:
                if current_time - self.last_alert_time >= self.alert_cooldown:
                    severity = get_final_severity(yamnet_score, has_profanity)
                    confidence = get_final_confidence(yamnet_score, has_profanity)
                    self.last_alert_time = current_time
                    self.aggressive_start_time = None
                    print(f"[ALERT] Aggression detected! Severity: {severity} Confidence: {confidence:.2f} RMS={rms}")
                    return {"should_alert": True, "severity": severity, "confidence": confidence, "duration": round(duration, 2), "yamnet_class": yamnet_class, "detected_words": detected_words, "has_profanity": has_profanity}
        else:
            if self.aggressive_start_time is not None:
                print(f"[DETECTION] Sound stopped (RMS={rms})")
            self.aggressive_start_time = None
        return {"should_alert": False}
