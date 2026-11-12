"""Tone vote (emotion2vec + shout cue), laughter, and the severity table."""

from dataclasses import replace
import threading
import time
import unittest

import numpy as np

from detection.false_alarm_filters import (
    FilterConfig,
    filter_terms_by_elevenlabs_logprob,
)
from detection.fusion import fuse_event_evidence
from detection.severity import HIGH, LOW, MEDIUM, calculate_severity
from detection.tone_vote import ToneThresholds, vote
from model.emotion_recognizer import (
    EmotionRecognizer,
    SpeechBaseline,
    ToneAnalyzer,
    speech_rms,
    yamnet_class_maxima,
)
from tests.test_accuracy_context import text
from tests.tone_helpers import ANGRY_SHOUT, LOUD_NOT_ANGRY, with_tone

T = ToneThresholds(angry=0.6, shout=0.3, rms_ratio=2.0, laugh=0.3)


def signals(angry=0.0, shout=0.0, laugh=0.0, ratio=None, label=None):
    label = label or ("angry" if angry >= 0.5 else "neutral")
    return {
        "emotion": {"label": label, "score": max(angry, 1 - angry),
                    "scores": {"angry": angry, "neutral": 1 - angry}},
        "yamnet": {"shout_score": shout, "laughter_score": laugh},
        "rms": {"ratio": ratio},
    }


class ToneVoteTests(unittest.TestCase):
    def test_volume_alone_is_never_angry(self):
        for loud in (signals(shout=0.99), signals(ratio=9.0),
                     signals(shout=0.99, ratio=9.0)):
            self.assertFalse(vote(loud, ("shouting",), thresholds=T).angry_tone)

    def test_angry_voice_without_a_loud_cue_is_not_angry_tone(self):
        self.assertFalse(vote(signals(angry=0.95), thresholds=T).angry_tone)

    def test_each_loud_cue_completes_the_vote(self):
        cases = (
            (signals(angry=0.9, shout=0.5), (), "shouting detected"),
            (signals(angry=0.9, ratio=2.5), (), "much louder than usual speech"),
            (signals(angry=0.9), ("yelling",), "shouting heard by transcriber"),
            (signals(angry=0.9), ("screams",), "shouting heard by transcriber"),
        )
        for sig, tags, cue in cases:
            with self.subTest(cue=cue, tags=tags):
                result = vote(sig, tags, thresholds=T)
                self.assertTrue(result.angry_tone)
                self.assertIn(cue, result.loud_cues)

    def test_thresholds_are_inclusive_and_strict_below(self):
        self.assertTrue(vote(signals(angry=0.6, shout=0.3), thresholds=T).angry_tone)
        self.assertFalse(vote(signals(angry=0.59, shout=0.3), thresholds=T).angry_tone)
        self.assertFalse(vote(signals(angry=0.6, shout=0.29, ratio=1.99), thresholds=T).angry_tone)

    def test_missing_signals_mean_no_tone(self):
        result = vote(None, (), thresholds=T)
        self.assertFalse(result.angry_tone)
        self.assertFalse(result.laughter)
        self.assertEqual(result.emotion_label, "unknown")

    def test_environment_thresholds(self):
        values = ToneThresholds.from_environment({
            "ECHOSENSE_TONE_ANGRY_MIN": "0.7", "ECHOSENSE_TONE_SHOUT_MIN": "0.4",
            "ECHOSENSE_TONE_RMS_RATIO": "3", "ECHOSENSE_TONE_LAUGH_MIN": "bad",
        })
        self.assertEqual((values.angry, values.shout, values.rms_ratio, values.laugh),
                         (0.7, 0.4, 3.0, ToneThresholds().laugh))


class LaughterTests(unittest.TestCase):
    def test_yamnet_laughter_classes(self):
        self.assertTrue(vote(signals(laugh=0.3), thresholds=T).laughter)
        self.assertFalse(vote(signals(laugh=0.29), thresholds=T).laughter)

    def test_linked_window_laughter_counts(self):
        result = vote(signals(), linked_laughter_score=0.8, thresholds=T)
        self.assertTrue(result.laughter)
        self.assertEqual(result.laughter_sources, ("yamnet",))

    def test_elevenlabs_tags(self):
        for tag in ("laughs", "laughter", "laughing", "giggles", "chuckles"):
            with self.subTest(tag=tag):
                result = vote(signals(), (tag,), thresholds=T)
                self.assertTrue(result.laughter)
                self.assertEqual(result.laughter_sources, ("transcriber_tag",))
        self.assertFalse(vote(signals(), ("applause",), thresholds=T).laughter)

    def test_laughter_is_not_angry_tone(self):
        self.assertFalse(vote(signals(laugh=0.9), ("laughs",), thresholds=T).angry_tone)


class SeverityTableTests(unittest.TestCase):
    """Rows of the requested table, through calculate_severity."""

    def level(self, terms, transcript, **kwargs):
        return calculate_severity(terms, transcript=transcript, **kwargs)

    def test_table(self):
        rows = (
            # terms, transcript, kwargs, level, summary
            ([], "ANO BA!", {"angry_tone": True}, LOW, "No monitored word"),
            (["bobo"], "ang bobo ko", {"angry_tone": True}, LOW,
             "Word said about oneself, not aimed at anyone"),
            (["bobo"], "bobo", {}, LOW, "Monitored word, not aimed at anyone"),
            (["bobo"], "bobo", {"laughter_present": True}, LOW,
             "Monitored word, not aimed at anyone"),
            (["bobo ka"], "bobo ka", {}, MEDIUM, "Insult aimed at someone"),
            (["bobo ka"], "bobo ka", {"laughter_present": True}, MEDIUM,
             "Insult aimed at someone, mocking with laughter"),
            (["bobo ka"], "bobo ka",
             {"angry_tone": True, "angry_cues": ("shouting detected",)}, HIGH,
             "Insult aimed at someone, said angrily (shouting detected)"),
            (["bobo ka"], "bobo ka",
             {"angry_tone": True, "laughter_present": True}, HIGH,
             "Insult aimed at someone, said angrily"),
            (["bobo ka"], "bobo ka", {"acoustic_aggressive": True,
                                      "strong_acoustic": True}, MEDIUM,
             "Insult aimed at someone"),
            (["kill you"], "kill you", {}, HIGH, "Threat or self-harm words"),
            (["kill you"], "kill you haha", {"laughter_present": True}, HIGH,
             "Threat or self-harm words"),
        )
        for terms, transcript, kwargs, level, summary in rows:
            with self.subTest(transcript=transcript, kwargs=kwargs):
                decision = self.level(terms, transcript, **kwargs)
                self.assertEqual(decision.level, level)
                self.assertEqual(decision.summary, summary)

    def test_mocking_reason_code(self):
        decision = self.level(["bobo ka"], "bobo ka", laughter_present=True)
        self.assertIn("mocking_with_laughter", decision.reasons)

    def test_end_to_end_tone_inputs(self):
        calm = text("bobo ka", 1.0)
        cases = (
            (calm, MEDIUM),
            (with_tone(calm, LOUD_NOT_ANGRY), MEDIUM),
            (with_tone(calm, ANGRY_SHOUT), HIGH),
            (with_tone(calm, {}, tags=("laughs",)), MEDIUM),
        )
        for evidence, level in cases:
            with self.subTest(tone=evidence.tone_signals, tags=evidence.audio_tags):
                decision = fuse_event_evidence((evidence,), (), duration_seconds=1.0)
                self.assertEqual(decision.level, level)
        mocking = fuse_event_evidence(
            (with_tone(calm, {}, tags=("laughs",)),), (), duration_seconds=1.0,
        )
        self.assertEqual(mocking.summary, "Insult aimed at someone, mocking with laughter")
        self.assertIn("laughter_context", mocking.supporting_evidence)

    def test_tone_alone_never_alerts(self):
        evidence = with_tone(text("ano ba", 1.0), ANGRY_SHOUT, tags=("shouting",))
        self.assertFalse(evidence.qualifies_to_open)


class EmotionRecognizerTests(unittest.TestCase):
    class Session:
        def __init__(self, probs):
            self.probs = np.asarray([probs], dtype=np.float32)
            self.calls = []

        def get_inputs(self):
            return [type("Input", (), {"name": "audio"})()]

        def run(self, _outputs, feed):
            self.calls.append(feed["audio"].shape)
            return [self.probs]

    def test_label_scores_and_crop(self):
        probs = [0.8, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0]
        session = self.Session(probs)
        recognizer = EmotionRecognizer(session=session)
        result = recognizer.analyze(np.zeros(16000 * 20, dtype=np.float32))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["label"], "angry")
        self.assertAlmostEqual(result["scores"]["angry"], 0.8, places=4)
        self.assertEqual(session.calls[0], (1, 16000 * 8))

    def test_short_clip_and_missing_model(self):
        self.assertEqual(
            EmotionRecognizer(session=self.Session([1] + [0] * 8))
            .analyze(np.zeros(100, dtype=np.float32))["status"],
            "too_short",
        )
        missing = EmotionRecognizer(model_path="/nonexistent/e2v.onnx")
        self.assertEqual(
            missing.analyze(np.zeros(16000, dtype=np.float32))["status"], "unavailable",
        )

    def test_speech_baseline_is_median_of_previous_utterances(self):
        baseline = SpeechBaseline(size=10, min_count=3)
        for value in (0.1, 0.1, 0.2):
            self.assertIsNone(baseline.observe(value)[1])
        base, ratio, count = baseline.observe(0.4)
        self.assertEqual((base, count), (0.1, 3))
        self.assertAlmostEqual(ratio, 4.0)

    def test_speech_rms_ignores_silence(self):
        clip = np.zeros(16000, dtype=np.float32)
        clip[:3200] = 0.2
        self.assertAlmostEqual(speech_rms(clip), 0.2, places=3)

    def test_yamnet_class_maxima_reads_named_classes(self):
        names = ("Speech", "Shout", "Laughter", "Giggle")

        def runner(_interpreter, _window, _names):
            return "Speech", 0.9, np.array([0.9, 0.4, 0.1, 0.6])

        result = yamnet_class_maxima(np.zeros(16000, dtype=np.float32), object(), names, runner)
        self.assertEqual((result["shout_label"], result["shout_score"]), ("Shout", 0.4))
        self.assertEqual((result["laughter_label"], result["laughter_score"]), ("Giggle", 0.6))


class ParallelToneTests(unittest.TestCase):
    def test_tone_runs_while_elevenlabs_waits(self):
        from model.stt_provider import HybridTranscriptionExecutor, STTSettings

        started = threading.Event()

        class SlowRecognizer:
            def analyze(self, samples, sample_rate=16000):
                started.set()
                time.sleep(0.3)
                return {"status": "ok", "label": "angry", "score": 0.9,
                        "scores": {"angry": 0.9}, "ms": 300}

        class Client:
            def transcribe_detailed(self, audio, sample_rate=16000):
                self.saw_tone_running = started.wait(1.0)
                time.sleep(0.3)
                return "Bogo ka (shouting)", "ceb", [
                    {"text": "Bogo", "type": "word", "logprob": -0.1},
                    {"text": "(shouting)", "type": "audio_event", "logprob": None},
                ]

        analyzer = ToneAnalyzer(SlowRecognizer(), yamnet=None, enabled=True)
        analyzer._yamnet_loaded = True
        client = Client()
        executor = HybridTranscriptionExecutor(
            object(), STTSettings(provider="elevenlabs", api_key="x"),
            client=client, tone_analyzer=analyzer,
        )
        begin = time.perf_counter()
        try:
            result = executor.transcribe(np.zeros(16000, dtype=np.float32))
        except ModuleNotFoundError:
            self.skipTest("RealtimeSTT not installed")
        elapsed = time.perf_counter() - begin
        analyzer.close()
        self.assertTrue(client.saw_tone_running)
        self.assertLess(elapsed, 0.55)
        self.assertEqual(result.metadata["tone"]["emotion"]["label"], "angry")
        self.assertEqual(result.metadata["stt"]["audio_tags"], ["shouting"])
        self.assertEqual(result.metadata["stt"]["words"][1]["type"], "audio_event")


class ElevenLabsLogprobFilterTests(unittest.TestCase):
    def words(self, **logprobs):
        return [{"text": w, "type": "word", "logprob": lp} for w, lp in logprobs.items()]

    def test_very_low_confidence_term_is_dropped(self):
        config = FilterConfig(elevenlabs_word_probability_min=0.05)
        kept, dropped, _ = filter_terms_by_elevenlabs_logprob(
            ["bogo ka"], self.words(Bogo=-6.0, ka=-0.1), config,
        )
        self.assertEqual((kept, dropped), ([], ["bogo ka"]))

    def test_confident_term_and_low_target_word_are_kept(self):
        config = FilterConfig(elevenlabs_word_probability_min=0.05)
        kept, dropped, _ = filter_terms_by_elevenlabs_logprob(
            ["bogo ka"], self.words(Bogo=-0.2, ka=-9.0), config,
        )
        self.assertEqual((kept, dropped), (["bogo ka"], []))

    def test_unscored_and_disabled_keep_everything(self):
        config = FilterConfig(elevenlabs_word_probability_min=0.05)
        self.assertEqual(filter_terms_by_elevenlabs_logprob(["bogo"], [], config)[0], ["bogo"])
        off = replace(config, elevenlabs_word_probability_enabled=False)
        self.assertEqual(
            filter_terms_by_elevenlabs_logprob(["bogo"], self.words(bogo=-9.0), off)[0],
            ["bogo"],
        )

    def test_threat_gets_the_lower_floor(self):
        config = FilterConfig(
            elevenlabs_word_probability_min=0.05,
            elevenlabs_urgent_word_probability_min=0.01,
        )
        kept, _dropped, _ = filter_terms_by_elevenlabs_logprob(
            ["kill you"], self.words(kill=-3.5, you=-0.1), config,
        )
        self.assertEqual(kept, ["kill you"])


class PayloadToneFieldsTests(unittest.TestCase):
    def test_tone_goes_into_existing_fields_only(self):
        from datetime import datetime, timezone
        from sender.payload_v2 import build_alert_payload_v2

        now = datetime(2026, 11, 12, tzinfo=timezone.utc)
        reason = "Insult aimed at someone, said angrily (shouting detected)"
        payload = build_alert_payload_v2(
            event_id="0b9d1c2e-1d7c-4c55-9a51-3f0f2f6f9e10",
            trigger_type="KEYWORD", severity="HIGH",
            event_start_timestamp=now, event_end_timestamp=now,
            device_identifier="es-test", transcribed_text="bogo ka",
            detected_words=["bogo ka"],
            severity_evidence={"level": "HIGH", "reasons": ["promoted_by:targeted_insult_angry_tone"],
                               "term_categories": {}, "supporting_evidence": ["angry_tone"]},
            emotion="angry",
            tone_context={
                "emotion_score": 0.91, "angry_score": 0.91, "angry_tone": True,
                "shout_score": 0.62, "rms_ratio": 2.4, "reason": reason,
                "laughter": {"detected": True, "score": 0.4, "sources": ["yamnet"],
                             "mocking": False},
            },
        )
        tone = payload["tone_evidence"]
        self.assertEqual(tone["emotion"], "angry")
        self.assertEqual(tone["emotion_score"], 0.91)
        self.assertTrue(tone["angry_tone"])
        self.assertEqual(tone["reason"], reason)
        self.assertEqual(payload["laughter_context"]["sources"], ["yamnet"])
        self.assertNotIn("emotion", payload)
        self.assertNotIn("reason", payload)


if __name__ == "__main__":
    unittest.main()
