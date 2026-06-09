import time
from collections import defaultdict


class ContextGate:
    def __init__(self):
        self.word_history      = defaultdict(list)
        self.REPETITION_WINDOW = 45.0   # seconds — Grade 6: widened 30 → 45 for spaced-out repeated bullying
        self.REPETITION_MIN    = 2      # same word 2x in 30s = bullying signal

    def check(
        self,
        detected_words:   list,
        emotion:          str,
        transcribed_text: str,
        is_casual:        bool,
        hard_hits:        list,
        soft_hits:        list,
    ) -> dict:

        now = time.time()

        # Clean old history
        for word in list(self.word_history.keys()):
            self.word_history[word] = [
                t for t in self.word_history[word]
                if now - t < self.REPETITION_WINDOW
            ]

        # Record current detections
        for word in detected_words:
            self.word_history[word].append(now)

        # Count repetitions
        max_rep = max(
            (len(self.word_history[w]) for w in detected_words),
            default=0
        )
        is_repeated = max_rep >= self.REPETITION_MIN

        # Real elapsed time between the first and last occurrence of the
        # most-repeated detected word. Used by the event-driven Track B path as
        # the "duration" (the old per-loop accumulation no longer exists now that
        # each utterance is consumed exactly once). 0.0 until something repeats.
        repetition_span = 0.0
        for w in detected_words:
            ts = self.word_history[w]
            if len(ts) >= self.REPETITION_MIN:
                repetition_span = max(repetition_span, ts[-1] - ts[0])

        # Emotion gate
        ALERT_EMOTIONS = {"angry", "aggressive", "distressed"}
        emotion_ok = emotion in ALERT_EMOTIONS

        # Hard trigger alone is enough if emotion is ok
        has_hard = len(hard_hits) > 0

        # Soft triggers need 2+ OR repetition
        soft_ok = len(soft_hits) >= 2 or is_repeated

        # Laughter suppressor
        # If student is laughing after/during the word = kantiyawan, not bullying
        # Exception: hard profanity + anger overrides even laughter
        laughing_suppressed = is_casual and not (has_hard and emotion_ok)

        # Final verdict
        is_bullying = (
            not laughing_suppressed and
            emotion_ok and
            (has_hard or soft_ok)
        )

        reason = "confirmed bullying context"
        if laughing_suppressed:
            reason = "laughter detected — likely kantiyawan, not bullying"
        elif not emotion_ok:
            reason = f"emotion is {emotion} — not angry/aggressive/distressed"
        elif not (has_hard or soft_ok):
            reason = "only one soft trigger — need repetition or 2+ words"

        return {
            "is_bullying_context": is_bullying,
            "max_repetitions":     max_rep,
            "is_repeated":         is_repeated,
            "repetition_span":     repetition_span,
            "emotion_ok":          emotion_ok,
            "has_laughter":        is_casual,
            "reason":              reason,
        }
