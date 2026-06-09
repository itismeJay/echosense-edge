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
    APPEARANCE_MIN_RMS,
    APPEARANCE_LOUD_RMS,
    get_time_severity,
)
from model.yamnet_infer import is_aggressive_sound, run_yamnet_scan
from model.whisper_stt import transcribe_and_check
from model.tone_analyzer import analyze_tone, get_tone_confidence_boost, classify_emotion
from model.blacklist import APPEARANCE_DIRECT_ROOTS
from detection.context_gate import ContextGate
from audio.capture import get_waveform_snapshot
from sender.shadow_log import log_near_miss

SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2}
# Grade 6: bullying tone is often calm/upset/tense, not full anger.
# NOTE: classify_emotion() (model/tone_analyzer.py) currently only emits
# angry/aggressive/distressed/upset/neutral/silent — "tense" and "fearful" are
# listed here for forward-compat but are inert until tone_analyzer emits them.
ANGRY_EMOTIONS = {"angry", "aggressive", "distressed", "upset", "tense", "fearful"}


def _distinct_hits(terms: list) -> list:
    """Collapse overlapping blacklist hits that are the SAME insult counted
    twice. Listing variants like {"buang", "buang ka", "buang kaayo"} means a
    single phrase ("buang ka") matches several entries; without this, that lone
    insult would trip the "2+ hard = directed tirade" rule and fire immediately
    on first (often casual) use. We keep only terms that are not a substring of
    a longer matched term, so "buang ka" counts as ONE distinct insult while two
    genuinely different insults ("bobo gago") still count as two."""
    return [t for t in terms if not any(t != o and t in o for o in terms)]

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
        # Tracks the start of the current quiet-bullying run for process_text()
        # (the RealtimeSTT path). The legacy process() path uses sample-based
        # duration from main.py and does not read this.
        self.aggressive_start_time = None

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
            if is_repeated:
                required_duration = DURATION_REPEATED_WORD     # 2.0s
                duration_gate = "repeated"
            elif has_hard:                                     # 2+ hard, hard+angry, hard+soft
                required_duration = DURATION_HARD_TRIGGER      # 2.0s
                duration_gate = "hard"
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
        """Track B firing rules — TIGHTENED (FIX 6).

        Fires when ONE of:
          - real repetition of a detected word across separate utterances, OR
          - 2+ hard triggers in the same utterance (directed insult), OR
          - a hard + a soft trigger in the same utterance ("bobo ka pangit mo"), OR
          - 1 hard + angry/aggressive tone (audio path only — tone is neutral on
            the text-only path, so this rule is inert there).

        NEVER fires on:
          - a single hard word said ONCE (the biggest false-positive fix),
          - 2+ soft words WITHOUT repetition,
          - anything with laughter.

        'is_repeated' must come from the context gate tracking DIFFERENT
        utterances over time — not the same stale result read twice (fixed in
        whisper_stt.transcribe_and_check via consume-once)."""
        # Grade 6 fix — single high-severity word fires immediately without
        # needing repetition. These words are unambiguous bullying even said
        # quietly once. Good students do not say these words. Alert teacher
        # immediately (RA 10627 — even one incident = bullying). Soft words
        # still need repetition/pairing.
        HIGH_SEVERITY_WORDS = {
            # Bisaya hard
            "yawa", "bogo", "bugok", "buang",
            "giatay", "piste", "putang ina",
            "patyon tika", "patyon ka nako",
            "bungoan tika", "suntukan ta",
            "away ta", "sampalan tika",
            # Tagalog hard
            "putangina", "pakyu", "tang ina",
            "tangina", "papatayin kita",
            "mamamatay ka",
            # English hard
            "kill yourself", "go kill yourself",
            "kill you",
            # Academic hard (very common Grade 6)
            "bobo", "tanga", "gago", "ulol",
            "inutil", "retard", "buang",
            "wala kang kwenta",
            "nobody likes you",
            "you are worthless",
        }

        single_high = any(
            w.lower() in HIGH_SEVERITY_WORDS
            for w in (hard_hits or [])
        )

        if single_high and not is_casual:
            print(
                f"[TRACK B] Single high-severity "
                f"word → immediate alert"
            )
            return True

        # Laughter always suppresses.
        if is_casual:
            return False
        # Real, cross-utterance repetition = targeting (RA 10627).
        if is_repeated:
            return True
        # Two or more DISTINCT hard triggers together = directed insult.
        # (Distinct so "buang ka" — which matches both "buang" and "buang ka" —
        # counts as one insult and does not fire on a single casual use.)
        if len(_distinct_hits(hard_hits)) >= 2:
            return True
        # Hard + soft together = combined directed insult.
        if len(hard_hits) >= 1 and len(soft_hits) >= 1:
            return True
        # One hard trigger + angry tone (audio path only; neutral on text path).
        if len(hard_hits) >= 1 and emotion in ANGRY_EMOTIONS:
            return True
        # Single hard word once, OR 2+ soft without repetition = NOT bullying.
        return False

    def process_text(self, stt_result: dict):
        """Event-driven Track B (the live RealtimeSTT path). One complete
        utterance in, one decision out. Runs the same logical layers as
        process() minus the audio-only ones (no waveform is available here):

          L1 STT + blacklist (already done upstream)
          L2 tone proxy   — no audio, so reject soft-only/non-repeated as casual
          L3 context gate — REAL cross-utterance repetition only (consume-once)
          L4 laughter     — kantiyawan always suppresses
          L5 duration     — tiered required seconds + 60s cooldown

        Note: because each utterance is now consumed exactly once
        (whisper_stt FIX 1), the old per-loop duration accumulation no longer
        exists. A single qualifying utterance (2 hard / hard+soft) is complete
        evidence on its own, and a repeated pattern uses the REAL elapsed span
        between utterances from the context gate."""
        current_time = time.time()

        if not stt_result.get("has_profanity"):
            return None

        hard_hits        = stt_result.get("hard_hits", [])
        soft_hits        = stt_result.get("soft_hits", [])
        transcribed_text = stt_result.get("transcribed_text", "")
        word_severity    = stt_result.get("severity", "low")
        categories       = stt_result.get("categories", [])
        detected_words   = stt_result.get("detected_words", [])
        is_casual        = stt_result.get("is_casual", False)

        # ---------- LAYER 4: laughter / casual ALWAYS suppresses ----------
        # Checked before the context gate so a joke ("bobo haha") is never
        # recorded as a bullying instance and can never build fake repetition.
        if is_casual:
            print("[NO ALERT] Laughter/casual present — kantiyawan (Layer 4)")
            return None

        # ---------- LAYER 3: context gate — REAL repetition only ----------
        ctx = self.context_gate.check(
            detected_words=detected_words,
            emotion="neutral",
            transcribed_text=transcribed_text,
            is_casual=is_casual,
            hard_hits=hard_hits,
            soft_hits=soft_hits,
        )
        is_repeated     = ctx.get("is_repeated", False)
        repetition_span = ctx.get("repetition_span", 0.0)

        # ---------- LAYER 2: tone proxy (no waveform on this path) ----------
        # Without audio we cannot measure anger. A soft-only utterance that is
        # not repeated is most likely casual chatter → skip.
        if not hard_hits and not is_repeated and len(soft_hits) < 2:
            print("[NO ALERT] Soft-only, not repeated — likely casual (Layer 2)")
            return None

        # ---------- Track B decision (single hard word once ≠ alert) ----------
        if not self._should_fire_track_b(
            hard_hits=hard_hits, soft_hits=soft_hits,
            is_repeated=is_repeated, emotion="neutral", is_casual=is_casual,
        ):
            print(f"[TRACK B] Criteria not met — hard={hard_hits} "
                  f"soft={soft_hits} repeated={is_repeated}")
            return None

        print("[TRACK B] Quiet bullying detected")

        # ---------- LAYER 5: tiered duration + cooldown ----------
        required = get_required_duration(hard_hits, soft_hits, is_repeated)
        # Repeated pattern → real elapsed span between utterances.
        # Single qualifying utterance → complete evidence, meets its tier now.
        duration = repetition_span if is_repeated else required

        print(f"[DURATION] {duration:.1f}s needed={required}s repeated={is_repeated}")
        if duration < required:
            print(f"[NO ALERT] Building ({duration:.1f}s < {required}s)")
            return None

        if current_time - self.last_alert_time < self.alert_cooldown:
            remaining = self.alert_cooldown - (current_time - self.last_alert_time)
            print(f"[COOLDOWN] {remaining:.0f}s remaining")
            return None

        self.last_alert_time = current_time

        time_severity = get_time_severity(duration)
        final_severity = max(
            [word_severity, time_severity],
            key=lambda s: SEVERITY_ORDER.get(s, 0)
        )

        gate = "repeated" if is_repeated else ("hard" if hard_hits else "medium")

        print(
            f"[ALERT] BULLYING CONFIRMED | Track=B "
            f"Severity={final_severity} Duration={duration:.1f}s "
            f"Gate={gate} Words={detected_words}"
        )

        return {
            "should_alert":      True,
            "track":             "B",
            "severity":          final_severity,
            "confidence":        0.85,
            "duration":          round(duration, 2),
            "transcribed_text":  transcribed_text,
            "detected_words":    detected_words,
            "categories":        categories,
            "yamnet_class":      "Speech",
            "yamnet_score":      0.60,
            "emotion":           "neutral",
            "tone_data":         {},
            "waveform_snapshot": [],
            "has_profanity":     True,
            "language":          stt_result.get("language", "tl"),
            "hard_hits":         hard_hits,
            "soft_hits":         soft_hits,
            "duration_gate":     gate,
            "required_duration": required,
        }

    def process_with_audio(self, stt_result: dict, audio_np=None) -> dict:
        """Audio-primary detection (FIX 4).

        Audio classifies HOW it was said (emotion / aggression via YAMNet+tone);
        text confirms WHAT word was said (the blacklist hit). Track A fires when
        the audio is aggressive AND a word was detected; otherwise we fall back to
        the quiet text path (Track B = process_text), which keeps its own strict
        gates (2+ soft / repetition) so a lone soft word never alerts quietly.
        """
        if not stt_result.get("has_profanity"):
            return None

        hard_hits      = stt_result.get("hard_hits", [])
        soft_hits      = stt_result.get("soft_hits", [])
        detected_words = stt_result.get("detected_words", [])
        is_casual      = stt_result.get("is_casual", False)

        # Laughter always suppresses.
        if is_casual:
            print("[NO ALERT] Kantiyawan")
            return None

        # Tone is computed ONCE from the rolling buffer (needs no YAMNet) and
        # reused by both the appearance branch and the YAMNet Track A below.
        tone = emotion = None
        if audio_np is not None and len(audio_np) >= 8000:
            tone = analyze_tone(audio_np)
            emotion = classify_emotion(tone)

            # ── APPEARANCE / BODY bullying — single utterance, audio-gated ──
            # For Grade 6, a directed appearance insult (baboy, tambok, taba,
            # pango, itom, uling, pandak, bungi, baho, baduy …) is bullying even
            # said ONCE — but only when the VOICE carries it. "Directed" = above
            # the too-quiet floor AND an angry/upset tone, OR clearly loud. Calm
            # normal talk (emotion=neutral) and near-silence never fire here; a
            # calm/quiet appearance word still needs repetition (Track B). Tone
            # only, so this keeps working even if YAMNet fails to load.
            appearance_hit = any(
                root in w
                for w in detected_words
                for root in APPEARANCE_DIRECT_ROOTS
            )
            directed = (
                (tone["rms"] >= APPEARANCE_MIN_RMS and emotion in ANGRY_EMOTIONS)
                or tone["rms"] >= APPEARANCE_LOUD_RMS
            )
            if appearance_hit and directed:
                print(f"[APPEARANCE] Directed appearance insult "
                      f"(RMS={tone['rms']:.0f} {emotion}) → single-utterance alert")
                return self._build_track_a_alert(
                    stt_result=stt_result,
                    audio_np=audio_np,
                    yamnet_class="(appearance)",
                    yamnet_score=0.60,
                    tone=tone,
                    emotion=emotion,
                )

        # ── TRACK A — Audio Primary (YAMNet + tone) ──────────────────────────
        if tone is not None and self.interpreter is not None and self.class_names:
            yamnet_class, yamnet_score = run_yamnet_scan(
                self.interpreter, audio_np, self.class_names
            )
            yamnet_score = float(yamnet_score)

            print(f"[AUDIO] YAMNet={yamnet_class}({yamnet_score:.2f})"
                  f" RMS={tone['rms']:.0f} Emotion={emotion}")

            audio_aggressive = (
                is_aggressive_sound(yamnet_class, yamnet_score, YAMNET_THRESHOLD)
                or (
                    tone["rms"] >= TONE_RMS_THRESHOLD
                    and emotion in ANGRY_EMOTIONS
                )
            )

            if audio_aggressive and (hard_hits or soft_hits):
                print("[TRACK A] Loud bullying — audio+text")
                return self._build_track_a_alert(
                    stt_result=stt_result,
                    audio_np=audio_np,
                    yamnet_class=yamnet_class,
                    yamnet_score=yamnet_score,
                    tone=tone,
                    emotion=emotion,
                )
            else:
                print("[TRACK A] Audio not aggressive — trying Track B")
        else:
            print("[TRACK A] No audio/YAMNet — trying Track B")

        # ── TRACK B — Text Primary (quiet / relational) ──────────────────────
        return self.process_text(stt_result)

    def _build_track_a_alert(self, stt_result, audio_np, yamnet_class,
                             yamnet_score, tone, emotion) -> dict:
        current_time = time.time()

        if current_time - self.last_alert_time < self.alert_cooldown:
            remaining = self.alert_cooldown - (current_time - self.last_alert_time)
            print(f"[COOLDOWN] {remaining:.0f}s remaining")
            return None

        self.last_alert_time = current_time
        self.aggressive_start_time = None

        word_sev = stt_result.get("severity", "low")
        final_sev = max([word_sev, "medium"], key=lambda s: SEVERITY_ORDER.get(s, 0))

        confidence = min(
            yamnet_score + 0.15 + get_tone_confidence_boost(tone),
            1.0,
        )

        hard_hits = stt_result.get("hard_hits", [])
        soft_hits = stt_result.get("soft_hits", [])

        print(
            f"[ALERT] BULLYING CONFIRMED | Track=A"
            f" Severity={final_sev}"
            f" YAMNet={yamnet_class}"
            f" Emotion={emotion}"
            f" Words={stt_result.get('detected_words')}"
        )

        return {
            "should_alert":      True,
            "track":             "A",
            "severity":          final_sev,
            "confidence":        round(confidence, 3),
            "duration":          2.0,
            "transcribed_text":  stt_result.get("transcribed_text", ""),
            "detected_words":    stt_result.get("detected_words", []),
            "categories":        stt_result.get("categories", []),
            "yamnet_class":      yamnet_class,
            "yamnet_score":      round(yamnet_score, 3),
            "emotion":           emotion,
            "tone_data":         tone,
            "waveform_snapshot": get_waveform_snapshot(audio_np),
            "has_profanity":     True,
            "language":          stt_result.get("language", "tl"),
            "hard_hits":         hard_hits,
            "soft_hits":         soft_hits,
            "duration_gate":     "hard" if hard_hits else "medium",
            "required_duration": 1.5,
        }

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
