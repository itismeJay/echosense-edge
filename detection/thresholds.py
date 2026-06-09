# ============================================================================
# EchoSense detection thresholds — AUTHORITATIVE copy (the runtime reads these).
# Production values for the Grade 6 Davao classroom deployment.
# ============================================================================

# --- Core detection gates ---------------------------------------------------
YAMNET_THRESHOLD        = 0.55   # Grade 6: lowered from 0.72 — 0.72 missed moderate aggression in normal speech

# --- Tiered duration gates — replaces the single DURATION_THRESHOLD ----------
# Match how fast the system reacts to how serious the spoken word is.
DURATION_THREAT         = 1.5
# Reason: threat words (patyon tika, kill you) = immediate danger.
# Teacher must be notified within ~2 seconds.

DURATION_HARD_TRIGGER   = 1.5
# Reason: severe words (yawa, bogo, bungi, uling, putangina) are unambiguous
# bullying even if brief; 2s confirms it was intentional, not accidental.

DURATION_REPEATED_WORD  = 1.5
# Reason: same word said 2x in 30s = targeting confirmed (RA 10627 defines
# bullying as REPEATED behavior). Repetition itself is the evidence — 2s confirms.

DURATION_MEDIUM_TRIGGER = 2.0
# Reason: 2+ soft words together (pangit + tambok, pango + baho) need slightly
# more context but still short. Grade 6: lowered 3.0 → 2.0 for quicker detection.

DURATION_SOFT_TRIGGER   = 3.0
# Reason: a single mild word (pikon, sumbong, ampon) alone needs a sustained
# pattern to rule out kantiyawan. Grade 6: lowered 5.0 → 3.0 for quicker detection.

ALERT_COOLDOWN          = 30.0   # Grade 6: lowered 60.0 → 30.0 — catch a 2nd incident in the same minute

# --- Quiet / relational bullying track --------------------------------------
# A second detection path that does NOT require a shout. It catches calm,
# mocking, or repeated taunts that the loud (YAMNet scream + RMS>=500) path
# misses entirely. To avoid false alarms it leans on the blacklist + repetition
# rather than loudness.
QUIET_RMS_FLOOR          = 150   # just confirm it is real speech, not a flatline
QUIET_TRACK_MIN_DURATION = 3.0   # quiet evidence must be sustained longer than a shout
QUIET_BASE_CONFIDENCE    = 0.60  # confidence floor for a quiet-track alert (no YAMNet scream score)

# --- Severity by duration (used AFTER an alert fires) -----------------------
SEVERITY_HIGH_DURATION   = 7.0   # 7+ seconds  = HIGH time severity
SEVERITY_MEDIUM_DURATION = 4.0   # 4-7 seconds = MEDIUM time severity
# below 4 seconds = LOW time severity

# --- Prosodic tone thresholds (EMEET OfficeCore M0 Plus) --------------------
TONE_RMS_THRESHOLD      = 150    # Grade 6: lowered 500 → 150 — kids speak at RMS 50-150 after EMEET AGC; 500 never fired
TONE_VARIANCE_THRESHOLD = 3000   # was 1000 — require bursty aggression
TONE_ZCR_THRESHOLD      = 0.10   # was 0.08
PROFANITY_MIN_RMS       = 400    # was 300 — floor for the profanity path
GRACE_PERIOD            = 1.5    # seconds a bullying streak may dip before it resets

# --- Confidence-shaping constants (used by tone_analyzer / aggression) ------
PROFANITY_BOOST            = 0.15
CASUAL_SPEECH_MAX          = 0.60
AGGRESSIVE_SPEECH_MIN      = 0.75
TONE_CONFIDENCE_BOOST_HIGH = 0.10
TONE_CONFIDENCE_BOOST_MED  = 0.05


def get_severity(confidence: float) -> str:
    if confidence >= 0.85:
        return "high"
    elif confidence >= 0.70:
        return "medium"
    else:
        return "low"


def get_time_severity(duration: float) -> str:
    """Time-based severity from how long the bullying was sustained.
    HIGH >= 7.0s, MEDIUM >= 4.0s, else LOW."""
    if duration >= SEVERITY_HIGH_DURATION:      # 7.0s
        return "high"
    if duration >= SEVERITY_MEDIUM_DURATION:    # 4.0s
        return "medium"
    return "low"


_SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2}


def max_severity(a: str, b: str) -> str:
    """Return the more severe of two severity labels."""
    if _SEVERITY_RANK.get(a, 0) >= _SEVERITY_RANK.get(b, 0):
        return a
    return b


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
