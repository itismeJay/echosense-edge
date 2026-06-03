YAMNET_THRESHOLD = 0.55
DURATION_THRESHOLD = 3.0
PROFANITY_BOOST = 0.15
CASUAL_SPEECH_MAX = 0.60
AGGRESSIVE_SPEECH_MIN = 0.75

# Prosodic tone-analysis thresholds — calibrated for EMEET OfficeCore M0 Plus.
# EMEET noise floor: RMS 1-15 (silence), 50-300 (normal speech), 800+ (screaming).
# tone_analyzer.py imports these directly; change here, takes effect everywhere.
TONE_RMS_THRESHOLD = 300          # raised voice without requiring screaming
TONE_VARIANCE_THRESHOLD = 1000    # proportional to EMEET's lower dynamic range
TONE_ZCR_THRESHOLD = 0.08         # EMEET speech ZCR starts at 0.08
TONE_CONFIDENCE_BOOST_HIGH = 0.10 # boost when all tone indicators are high
TONE_CONFIDENCE_BOOST_MED = 0.05  # boost when some tone indicators are high
PROFANITY_MIN_RMS = 300            # floor for profanity path; above whisper/noise

def get_severity(confidence: float) -> str:
    if confidence >= 0.85:
        return "high"
    elif confidence >= 0.70:
        return "medium"
    else:
        return "low"

def get_severity_by_duration(duration: float) -> str:
    if duration >= 6.0:
        return "high"
    elif duration >= 4.0:
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
