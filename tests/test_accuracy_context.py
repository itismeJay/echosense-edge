"""Accuracy rules: adaptive noise floor, time-linked corroboration, casual-use
context, and the text-driven incident window."""

from dataclasses import replace
from datetime import timedelta
import unittest

import numpy as np

from config import DetectionRuntimeConfig, detection_runtime_config_from_environment
from detection.acoustic_detector import AcousticDetector, NoiseFloorTracker
from detection.coordinator import EventCoordinator
from detection.fusion import fuse_event_evidence
from detection.severity import HIGH, LOW, MEDIUM, calculate_severity
from detection.text_detector import TextDetector
from model.whisper_stt import NoiseReducer, build_initial_prompt
from tests.test_event_coordinator_after import (
    WALL,
    FakeClock,
    acoustic_evidence,
)


def text(transcript, at=0.0, *, spoken_seconds=0.8):
    """Real TextDetector evidence observed at ``at`` (utterance end)."""

    evidence = TextDetector().analyze_transcript(
        transcript,
        language="fil",
        observed_at=WALL + timedelta(seconds=at),
        observed_monotonic=at,
    )
    return replace(
        evidence,
        spoken_started_at=WALL + timedelta(seconds=at - spoken_seconds),
    )


def tone_window(end, *, aggressive=True):
    """A 0.975 s window classified only by the multi-feature tone check."""

    return replace(
        acoustic_evidence(at=end, score=0.1, qualifies=aggressive),
        yamnet_label="Speech",
        yamnet_aggressive=False,
        tone_aggressive=aggressive,
        qualifies_to_open=aggressive,
    )


def windows(start, end, **kwargs):
    """Overlapping windows (0.4875 s hop) whose ends span [start, end]."""

    count = int(round((end - start) / 0.4875)) + 1
    return [tone_window(start + 0.4875 * index, **kwargs) for index in range(count)]


class CoordinatorHarness:
    def __init__(self, config=None):
        self.clock = FakeClock()
        self.finalized = []
        self.coordinator = EventCoordinator(
            config=config or DetectionRuntimeConfig(),
            finalizer=lambda item: self.finalized.append(item) or True,
            monotonic_clock=self.clock,
        )

    def acoustic(self, evidence):
        self.clock.value = max(self.clock.value, evidence.monotonic_end)
        return self.coordinator.submit_acoustic(evidence)

    def text(self, evidence):
        self.clock.value = max(self.clock.value, evidence.observed_monotonic)
        return self.coordinator.submit_text(evidence)

    def finish(self, at):
        self.clock.value = at
        self.coordinator.tick()
        return self.finalized


class TimeLinkedCorroborationTests(unittest.TestCase):
    def test_calm_word_with_unrelated_noise_four_seconds_later_stays_medium(self):
        calm = text("you are stupid", 1.0)
        self.assertEqual(calm.baseline_severity, MEDIUM)
        noise = windows(5.0, 7.0)
        decision = fuse_event_evidence(
            (calm,), tuple(noise), duration_seconds=6,
        )
        self.assertEqual(decision.level, MEDIUM)
        self.assertIn(
            "unlinked_acoustic_evidence_ignored",
            decision.supporting_evidence,
        )
        # Same through the coordinator: the noise neither promotes the word
        # nor joins its incident.
        harness = CoordinatorHarness()
        harness.text(calm)
        for window in noise:
            harness.acoustic(window)
        finalized = harness.finish(20)
        self.assertEqual(finalized[0].severity, MEDIUM)

    def test_bobo_ka_shouted_at_the_same_moment_becomes_high(self):
        harness = CoordinatorHarness()
        for window in windows(0.5, 2.0):
            harness.acoustic(window)
        # Whisper delivers the transcript after the utterance ends.
        harness.text(text("bobo ka", 2.0, spoken_seconds=1.5))
        finalized = harness.finish(20)
        self.assertEqual(len(finalized), 1)
        self.assertEqual(finalized[0].severity, HIGH)
        self.assertIn(
            "promoted_by:strong_acoustic_corroboration",
            finalized[0].severity_reasons,
        )

    def test_late_transcript_still_links_to_shout_after_acoustic_incident_closed(self):
        harness = CoordinatorHarness()
        for window in windows(0.5, 2.0):
            harness.acoustic(window)
        harness.finish(5.5)
        self.assertEqual(harness.finalized[0].severity, LOW)
        # STT latency: the transcript of the shout arrives after that
        # acoustic-only incident closed. It is never suppressed and
        # still links to the shout that happened while it was spoken.
        result = harness.text(text("bobo ka", 6.0, spoken_seconds=5.0))
        self.assertTrue(result.opened)
        finalized = harness.finish(20)
        self.assertEqual(finalized[1].severity, HIGH)

    def test_strong_acoustic_needs_share_of_linked_windows(self):
        word = text("bobo ka", 2.0, spoken_seconds=1.5)
        mostly_calm = (
            windows(0.5, 2.5, aggressive=False)[:-1]
            + [tone_window(2.5)]
        )
        decision = fuse_event_evidence(
            (word,), tuple(mostly_calm), duration_seconds=3,
        )
        # One aggressive window among many calm ones corroborates (LOW->MEDIUM
        # only) but is not strong enough to make MEDIUM text HIGH.
        self.assertEqual(decision.level, MEDIUM)


class AcousticOnlyPolicyTests(unittest.TestCase):
    def test_acoustic_only_room_noise_stays_low(self):
        harness = CoordinatorHarness()
        for window in windows(0.5, 6.0):
            harness.acoustic(window)
        finalized = harness.finish(30)
        # No cooldown: sustained noise closes one incident and opens the
        # next, but every one stays LOW.
        self.assertGreaterEqual(len(finalized), 1)
        self.assertEqual({item.severity for item in finalized}, {LOW})
        for item in finalized:
            self.assertIn(
                "acoustic_only_policy:low_only",
                item.severity_reasons,
            )

    def test_acoustic_only_off_sends_nothing_and_blocks_nothing(self):
        harness = CoordinatorHarness(
            DetectionRuntimeConfig(acoustic_only_alerts="off"),
        )
        for window in windows(0.5, 2.0):
            harness.acoustic(window)
        harness.finish(6)
        self.assertEqual(harness.finalized, [])
        self.assertTrue(harness.text(text("bobo ka", 7.0)).opened)

    def test_environment_controls_new_settings(self):
        values = detection_runtime_config_from_environment({
            "ECHOSENSE_ACOUSTIC_ONLY_ALERTS": "ON",
            "ECHOSENSE_INCIDENT_WINDOW_SECONDS": "4",
            "ECHOSENSE_CORROBORATION_WINDOW_SECONDS": "1",
            "ECHOSENSE_STRONG_TONE_RATIO": "0.5",
            "ECHOSENSE_NOISE_FLOOR_WINDOW_SECONDS": "30",
            "ECHOSENSE_TONE_FLOOR_MULTIPLIER": "3",
            "ECHOSENSE_CALIBRATION_SECONDS": "10",
        })
        self.assertEqual(values.acoustic_only_alerts, "on")
        self.assertEqual(values.post_trigger_seconds, 4.0)
        self.assertEqual(values.corroboration_window_seconds, 1.0)
        self.assertEqual(values.strong_tone_ratio, 0.5)
        self.assertEqual(values.noise_floor_window_seconds, 30.0)
        self.assertEqual(values.tone_floor_multiplier, 3.0)
        self.assertEqual(values.calibration_seconds, 10.0)
        with self.assertRaises(ValueError):
            detection_runtime_config_from_environment({
                "ECHOSENSE_ACOUSTIC_ONLY_ALERTS": "sometimes",
            })


class NoiseFloorTests(unittest.TestCase):
    @staticmethod
    def steady(rms, seed):
        rng = np.random.default_rng(seed)
        return (rng.standard_normal(15_600) * rms / 32768).astype(np.float32)

    @staticmethod
    def bursty(amplitude):
        """Loud, uneven, high-ZCR window: passes the variance/ZCR checks."""

        rng = np.random.default_rng(7)
        samples = rng.standard_normal(15_600) * amplitude / 32768
        gate = (np.arange(15_600) // 512) % 2
        return (samples * gate).astype(np.float32)

    def feed(self, detector, samples, at):
        return detector.analyze_window(
            samples,
            interval_started_at=WALL + timedelta(seconds=at - 0.975),
            interval_ended_at=WALL + timedelta(seconds=at),
            monotonic_start=at - 0.975,
            monotonic_end=at,
        )

    def test_rolling_floor_rises_when_room_gets_louder(self):
        tracker = NoiseFloorTracker(calibration_seconds=15.0)
        detector = AcousticDetector(noise_floor=tracker)
        at = 1.0
        for index in range(120):  # one quiet minute
            self.feed(detector, self.steady(60, index), at)
            at += 0.5
        quiet_floor = tracker.floor
        quiet_threshold = tracker.threshold
        shout = self.bursty(1_200)  # RMS ~850, above the 400 absolute floor
        self.assertTrue(self.feed(detector, shout, at).tone_aggressive)
        for index in range(120):  # the room gets loud for a minute
            at += 0.5
            self.feed(detector, self.steady(400, 1_000 + index), at)
        self.assertGreater(tracker.floor, quiet_floor * 4)
        self.assertGreater(tracker.threshold, quiet_threshold)
        # The same burst is now ordinary for this room.
        at += 0.5
        self.assertFalse(self.feed(detector, shout, at).tone_aggressive)

    def test_threshold_never_drops_below_static_minimum(self):
        tracker = NoiseFloorTracker(calibration_seconds=0.0, minimum_threshold=100)
        for index in range(10):
            tracker.observe(5.0, monotonic_start=index, monotonic_end=index + 1)
        self.assertEqual(tracker.threshold, 100)

    def test_calibration_blocks_tone_aggression_at_startup(self):
        detector = AcousticDetector(
            noise_floor=NoiseFloorTracker(calibration_seconds=15.0),
        )
        evidence = self.feed(detector, self.bursty(2_000), 1.0)
        self.assertFalse(evidence.tone_aggressive)
        self.assertIn("noise_floor_calibrating", evidence.reasons)


class CasualUseContextTests(unittest.TestCase):
    def assertLevel(self, transcript, level, reason=None):
        evidence = text(transcript)
        self.assertEqual(evidence.baseline_severity, level, transcript)
        if reason:
            self.assertIn(reason, evidence.reasons, transcript)

    def test_self_directed_is_low(self):
        self.assertLevel("ang bobo ko", LOW, "downgraded_to_low:self_directed")
        self.assertLevel("buang ko", LOW, "downgraded_to_low:self_directed")
        self.assertLevel("I am worthless", LOW, "downgraded_to_low:self_directed")

    def test_about_a_thing_is_low(self):
        self.assertLevel("stupid game", LOW, "held_low:about_object")
        self.assertLevel("bobo yung tanong", LOW, "downgraded_to_low:about_object")

    def test_negated_or_instructional_is_low(self):
        self.assertLevel(
            "wag mong sabihing bobo",
            LOW,
            "downgraded_to_low:negated_or_instructional",
        )
        self.assertLevel(
            "ayaw pag ingon nga bogo",
            LOW,
            "downgraded_to_low:negated_or_instructional",
        )
        self.assertLevel(
            "don't call him tanga",
            LOW,
            "downgraded_to_low:negated_or_instructional",
        )

    def test_directed_terms_are_medium(self):
        self.assertLevel("bobo ka", MEDIUM)
        self.assertLevel("bogo ka", MEDIUM)
        self.assertLevel("you are stupid", MEDIUM)

    def test_untargeted_medium_term_is_low(self):
        self.assertLevel("bobo", LOW, "downgraded_to_low:no_direct_target")

    def test_repetition_within_utterance_escalates(self):
        self.assertLevel("bobo ka bobo ka", HIGH, "promoted_by:repetition")

    def test_threats_are_never_downgraded(self):
        self.assertLevel("kill you haha", HIGH)
        decision = calculate_severity(
            ["kill you"],
            transcript="wag kill you haha ko",
            laughter_present=True,
        )
        self.assertEqual(decision.level, HIGH)
        self.assertIsNone(decision.context_downgrade)

    def test_playful_mild_term_stays_low_unless_escalated(self):
        playful = calculate_severity(
            ["pangit"], transcript="pangit ka haha", laughter_present=True,
        )
        self.assertEqual(playful.level, LOW)
        self.assertIn("held_low:playful_context", playful.reasons)
        shouted = calculate_severity(
            ["pangit"],
            transcript="pangit ka haha",
            laughter_present=True,
            strong_acoustic=True,
        )
        self.assertEqual(shouted.level, MEDIUM)

    def test_negation_after_the_term_does_not_downgrade(self):
        self.assertLevel("bobo ka, wag ka ganyan", MEDIUM)


class IncidentWindowTests(unittest.TestCase):
    def test_noise_after_a_word_does_not_extend(self):
        harness = CoordinatorHarness()
        harness.text(text("bobo ka", 0.0))
        deadline = harness.coordinator.active_incident.close_deadline
        self.assertEqual(deadline, 3.0)
        for window in windows(2.0, 2.9):
            harness.acoustic(window)
        self.assertEqual(
            harness.coordinator.active_incident.close_deadline,
            deadline,
        )
        harness.finish(3.0)
        self.assertEqual(harness.finalized[0].duration_seconds, 3.0)

    def test_repeated_word_extends(self):
        harness = CoordinatorHarness()
        harness.text(text("bobo ka", 0.0))
        harness.text(replace(text("bobo ka", 2.5), observed_monotonic=2.5))
        self.assertEqual(harness.coordinator.active_incident.close_deadline, 4.5)
        finalized = harness.finish(4.5)
        self.assertEqual(len(finalized[0].text_evidence), 2)
        self.assertEqual(finalized[0].severity, HIGH)
        self.assertIn("repeated_within_incident", finalized[0].supporting_evidence)

    def test_linked_strong_acoustic_extends(self):
        harness = CoordinatorHarness()
        harness.text(text("bobo ka", 0.0))
        harness.acoustic(acoustic_evidence(at=1.0, score=0.9))
        self.assertEqual(harness.coordinator.active_incident.close_deadline, 3.0)
        harness.acoustic(acoustic_evidence(at=1.4, score=0.9))
        self.assertEqual(harness.coordinator.active_incident.close_deadline, 3.4)

    def test_threat_is_enqueued_without_waiting(self):
        harness = CoordinatorHarness()
        result = harness.text(text("kill you", 0.0))
        self.assertEqual(len(harness.finalized), 1)
        self.assertEqual(harness.finalized[0].severity, HIGH)
        self.assertEqual(result.finalized, tuple(harness.finalized))
        self.assertIsNone(harness.coordinator.active_incident)


class WhisperFrontEndTests(unittest.TestCase):
    def test_initial_prompt_is_short_and_comma_separated(self):
        self.assertIsNone(build_initial_prompt([]))
        self.assertEqual(
            build_initial_prompt(["bobo", "yawa", "bobo", " "]),
            "bobo, yawa",
        )
        prompt = build_initial_prompt([f"term{index}" for index in range(500)])
        self.assertLessEqual(len(prompt), 200)
        self.assertLessEqual(len(prompt.split(", ")), 24)

    def test_noise_reduction_runs_before_audio_is_fed(self):
        calls = []

        def reduce_noise(*, y, sr, y_noise, stationary, prop_decrease):
            calls.append((y.size, y_noise.size, sr, stationary))
            return y * 0.5

        reducer = NoiseReducer(
            reduce_function=reduce_noise, block_samples=1_000, profile_blocks=4,
        )
        out = []
        for _ in range(6):
            out.extend(reducer.process(np.full(1_000, 0.2, dtype=np.float32)))
        self.assertEqual(len(out), 6)
        # The first blocks build the noise profile; then reduction applies.
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0][1], 4_000)
        np.testing.assert_allclose(out[-1], 0.1)


if __name__ == "__main__":
    unittest.main()
