YAMNET_THRESHOLD = 0.5
DURATION_THRESHOLD = 1.0
PROFANITY_BOOST = 0.15
CASUAL_SPEECH_MAX = 0.60
AGGRESSIVE_SPEECH_MIN = 0.75

def get_severity(confidence: float) -> str:
    if confidence >= 0.85:
        return "high"
    elif confidence >= 0.70:
        return "medium"
    else:
        return "low"

def get_final_severity(yamnet_confidence: float, has_profanity: bool) -> str:
    confidence = yamnet_confidence
    if has_profanity:
        confidence = min(1.0, confidence + PROFANITY_BOOST)
    return get_severity(confidence)

def get_final_confidence(yamnet_confidence: float, has_profanity: bool) -> float:
    if has_profanity:
        return min(1.0, yamnet_confidence + PROFANITY_BOOST)
    return yamnet_confidence

def is_aggressive_tone(confidence: float, has_profanity: bool) -> bool:
    if has_profanity and confidence >= CASUAL_SPEECH_MAX:
        return True
    if confidence >= AGGRESSIVE_SPEECH_MIN:
        return True
    return False
