YAMNET_THRESHOLD = 0.5
DURATION_THRESHOLD = 1.0
PROFANITY_BOOST = 0.15
CASUAL_SPEECH_MAX = 0.60
AGGRESSIVE_SPEECH_MIN = 0.75

# Prosodic tone-analysis thresholds (acoustic, used by model/tone_analyzer.py).
# NOTE: tone_analyzer.analyze_tone() currently hard-codes these same numbers
# inline; these constants are the canonical reference for tuning. If you change
# a value here, mirror it in analyze_tone() (and vice-versa).
TONE_RMS_THRESHOLD = 800          # minimum loudness for aggressive tone
TONE_VARIANCE_THRESHOLD = 5000    # minimum energy variance
TONE_ZCR_THRESHOLD = 0.1          # minimum zero crossing rate
TONE_CONFIDENCE_BOOST_HIGH = 0.10 # boost when all tone indicators are high
TONE_CONFIDENCE_BOOST_MED = 0.05  # boost when some tone indicators are high

def get_severity(confidence: float) -> str:
    if confidence >= 0.85:
        return "high"
    elif confidence >= 0.70:
        return "medium"
    else:
        return "low"

def get_severity_by_duration(duration: float) -> str:
    if duration >= 5.0:
        return "high"
    elif duration >= 3.0:
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
