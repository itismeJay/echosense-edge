# ============================================================================
# EchoSense detection thresholds — AUTHORITATIVE copy (the runtime reads these).
# Production values for the Grade 6 Davao classroom deployment.
# ============================================================================

from detection.severity import (
    SEVERITY_HIGH_DURATION,
    SEVERITY_MEDIUM_DURATION,
    max_severity as _max_severity,
    severity_from_confidence,
    severity_from_duration,
)

# --- Core detection gates ---------------------------------------------------
YAMNET_THRESHOLD        = 0.45   # Grade 6: lowered to 0.45 — catch moderate aggression signals in normal speech

# --- Transcript quality gates -----------------------------------------------
# Finalized classroom utterances are normally far below 120 tokens. The limit
# bounds pathological decoder output without penalizing ordinary long sentences.
TRANSCRIPT_MAX_TOKENS = 120
# Two-word emphasis ("stop stop") and even a cautious third repetition remain
# accepted; four consecutive identical tokens are treated as a decoding-risk
# signal before any monitored term can reach acoustic analysis.
TRANSCRIPT_MAX_REPEATED_TOKEN_RUN = 3
# Ratio checks are unreliable on short speech, so they begin at eight tokens.
TRANSCRIPT_MIN_TOKENS_FOR_UNIQUE_RATIO = 8
# Fewer than 30% unique tokens in a sufficiently long utterance is suspicious.
TRANSCRIPT_LOW_UNIQUE_RATIO = 0.30
# Phrase-cycle and broad repetition checks begin only at six tokens, allowing
# natural short emphasis while catching three exact repeats of a two-word phrase.
TRANSCRIPT_MIN_TOKENS_FOR_PHRASE_CYCLE = 6
TRANSCRIPT_REPETITION_RATIO_LIMIT = 0.70

# --- Tiered duration gates — replaces the single DURATION_THRESHOLD ----------
# Match how fast the system reacts to how serious the spoken word is.
DURATION_THREAT         = 1.5
# Reason: threat words (patyon tika, kill you) = immediate danger.
# Teacher must be notified within ~2 seconds.

DURATION_HARD_TRIGGER   = 1.5
# Severe monitored terms use a shorter evidence-duration gate. This remains an
# unverified indicator and does not establish intent.

DURATION_REPEATED_WORD  = 1.5
# Repetition is supporting observable evidence; it does not identify a target or
# confirm bullying.

DURATION_MEDIUM_TRIGGER = 2.0
# Reason: 2+ soft words together (pangit + tambok, pango + baho) need slightly
# more context but still short. Grade 6: lowered 3.0 → 2.0 for quicker detection.

DURATION_SOFT_TRIGGER   = 3.0
# Reason: a single mild word (pikon, sumbong, ampon) alone needs a sustained
# pattern to rule out kantiyawan. Grade 6: lowered 5.0 → 3.0 for quicker detection.

ALERT_COOLDOWN          = 15.0   # Grade 6: lowered to 15s — catch back-to-back incidents quickly

# --- Quiet / relational bullying track --------------------------------------
# A second detection path that does NOT require a shout. It catches calm,
# mocking, or repeated taunts that the loud (YAMNet scream + RMS>=500) path
# misses entirely. To avoid false alarms it leans on the blacklist + repetition
# rather than loudness.
QUIET_RMS_FLOOR          = 150   # just confirm it is real speech, not a flatline
QUIET_TRACK_MIN_DURATION = 3.0   # quiet evidence must be sustained longer than a shout
QUIET_BASE_CONFIDENCE    = 0.60  # confidence floor for a quiet-track alert (no YAMNet scream score)

# --- Prosodic tone thresholds (EMEET OfficeCore M0 Plus) --------------------
TONE_RMS_THRESHOLD      = 100    # Grade 6: lowered to 100 — catch the quietest classroom voices after EMEET AGC

# --- Appearance / body "direct" single-utterance gate -----------------------
# Appearance insults (baboy, tambok, taba, pango, itom, uling, pandak, bungi…)
# alert on ONE utterance, but only when the voice carries it:
#   - "too quiet" floor: below APPEARANCE_MIN_RMS = treated as not directed.
#   - directed = (RMS >= APPEARANCE_MIN_RMS AND emotion is angry/upset/etc.)
#                OR clearly loud (RMS >= APPEARANCE_LOUD_RMS, a raised voice).
# Calm normal talk (emotion=neutral, moderate RMS) and near-silence never fire;
# a calm/quiet appearance word still needs repetition (Track B).
APPEARANCE_MIN_RMS      = 100    # below this = too quiet / not clearly directed
APPEARANCE_LOUD_RMS     = 400    # a raised voice fires even if monotone
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
    """Compatibility wrapper for the centralized uppercase severity rules."""

    return severity_from_confidence(confidence)


def get_time_severity(duration: float) -> str:
    """Compatibility wrapper for the centralized duration severity rule."""

    return severity_from_duration(duration)


def max_severity(a: str, b: str) -> str:
    return _max_severity(a, b)


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
