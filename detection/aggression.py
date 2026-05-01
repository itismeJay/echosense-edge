import time
from detection.thresholds import YAMNET_THRESHOLD, DURATION_THRESHOLD, get_final_severity, get_final_confidence
from model.yamnet_infer import is_aggressive_sound

class AggressionDetector:
    def __init__(self):
        self.aggressive_start_time = None
        self.last_alert_time = 0
        self.alert_cooldown = 10.0

    def process(self, yamnet_class, yamnet_score, has_profanity, detected_words):
        current_time = time.time()
        is_aggressive = is_aggressive_sound(yamnet_class, yamnet_score, YAMNET_THRESHOLD)
        if has_profanity:
            is_aggressive = True
            yamnet_score = max(yamnet_score, 0.65)
        if is_aggressive:
            if self.aggressive_start_time is None:
                self.aggressive_start_time = current_time
                print(f"[DETECTION] Aggressive sound started: {yamnet_class} ({yamnet_score:.2f})")
            duration = current_time - self.aggressive_start_time
            if duration >= DURATION_THRESHOLD:
                if current_time - self.last_alert_time >= self.alert_cooldown:
                    severity = get_final_severity(yamnet_score, has_profanity)
                    confidence = get_final_confidence(yamnet_score, has_profanity)
                    self.last_alert_time = current_time
                    self.aggressive_start_time = None
                    print(f"[ALERT] Aggression detected! Severity: {severity} Confidence: {confidence:.2f}")
                    return {"should_alert": True, "severity": severity, "confidence": confidence, "duration": round(duration, 2), "yamnet_class": yamnet_class, "detected_words": detected_words, "has_profanity": has_profanity}
        else:
            if self.aggressive_start_time is not None:
                print(f"[DETECTION] Aggressive sound stopped")
            self.aggressive_start_time = None
        return {"should_alert": False}
