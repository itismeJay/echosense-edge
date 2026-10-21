"""No global cooldown: duplicate guard only; absolute tone floor; latency log."""

from datetime import timedelta
from dataclasses import replace
import unittest

import numpy as np

from config import DetectionRuntimeConfig, detection_runtime_config_from_environment
from detection.acoustic_detector import AcousticDetector, NoiseFloorTracker
from detection.latency import LatencyTracker
from detection.severity import HIGH, LOW, MEDIUM
from detection.text_detector import TextDetector
from tests.test_accuracy_context import CoordinatorHarness, NoiseFloorTests, windows
from tests.test_event_coordinator_after import WALL, FakeClock


class Speech:
    """One shared TextDetector, as in production, so repeats are flagged."""

    def __init__(self):
        self.detector = TextDetector()

    def __call__(self, transcript, at, *, spoken_seconds=0.8):
        evidence = self.detector.analyze_transcript(
            transcript,
            language="fil",
            observed_at=WALL + timedelta(seconds=at),
            observed_monotonic=at,
        )
        return replace(
            evidence,
            spoken_started_at=WALL + timedelta(seconds=at - spoken_seconds),
        )


class DuplicateGuardTests(unittest.TestCase):
    def test_live_test_sequence(self):
        harness = CoordinatorHarness()
        say = Speech()
        # Room noise: logged locally, no incident, nothing sent.
        for window in windows(0.5, 2.0):
            self.assertFalse(harness.acoustic(window).opened)
        harness.finish(3.5)
        self.assertEqual(harness.finalized, [])

        # "bobo ka" 2 s later is NOT suppressed.
        first = harness.text(say("bobo ka", 5.5))
        self.assertTrue(first.opened)
        self.assertIsNone(first.duplicate_of)
        harness.finish(7.0)
        self.assertEqual(len(harness.finalized), 1)
        self.assertEqual(harness.finalized[0].severity, MEDIUM)
        bobo_id = harness.finalized[0].event_id

        # Same "bobo ka" 5 s after close: duplicate of that incident, no alert
        # (even though the detector flags it as repeated).
        again = say("bobo ka", 12.0)
        self.assertTrue(again.repeated)
        duplicate = harness.text(again)
        self.assertEqual(duplicate.duplicate_of, bobo_id)
        self.assertFalse(duplicate.opened)
        self.assertIsNone(duplicate.event_id)
        self.assertIsNone(harness.coordinator.active_incident)
        self.assertEqual(harness.coordinator.duplicate_count, 1)
        self.assertEqual(
            harness.coordinator.last_text_incident.duplicates, [again],
        )
        harness.finish(12.5)
        self.assertEqual(len(harness.finalized), 1)

        # "kill you" right after: different terms -> new HIGH immediately.
        threat = harness.text(say("kill you", 13.0))
        self.assertTrue(threat.opened)
        self.assertEqual(len(harness.finalized), 2)
        self.assertEqual(harness.finalized[1].severity, HIGH)
        self.assertNotEqual(harness.finalized[1].event_id, bobo_id)

    def test_same_terms_after_window_open_new_incident(self):
        harness = CoordinatorHarness()
        say = Speech()
        harness.text(say("bobo ka", 1.0))
        harness.finish(2.5)
        result = harness.text(say("bobo ka", 12.5))  # 10 s after close
        self.assertTrue(result.opened)

    def test_extra_term_is_new_incident(self):
        harness = CoordinatorHarness()
        say = Speech()
        harness.text(say("bobo ka", 1.0))
        harness.finish(4.0)
        result = harness.text(say("bobo ka gago", 6.0))
        self.assertTrue(result.opened)
        self.assertIsNone(result.duplicate_of)

    def test_shouted_repeat_is_higher_severity_and_new(self):
        harness = CoordinatorHarness()
        say = Speech()
        harness.text(say("bobo ka", 1.0))
        harness.finish(4.0)
        self.assertEqual(harness.finalized[0].severity, MEDIUM)
        for window in windows(5.0, 6.5):
            harness.acoustic(window)
        result = harness.text(say("bobo ka", 9.5, spoken_seconds=4.5))
        self.assertTrue(result.opened)
        harness.finish(30)
        self.assertEqual(harness.finalized[-1].severity, HIGH)

    def test_window_zero_disables_the_guard(self):
        harness = CoordinatorHarness(
            DetectionRuntimeConfig(duplicate_window_seconds=0.0),
        )
        say = Speech()
        harness.text(say("bobo ka", 1.0))
        harness.finish(4.0)
        self.assertTrue(harness.text(say("bobo ka", 5.0)).opened)

    def test_sound_never_blocks_or_opens_anything(self):
        harness = CoordinatorHarness()
        for window in windows(0.5, 2.0):
            self.assertFalse(harness.acoustic(window).opened)
        self.assertIsNone(harness.coordinator.last_text_incident)
        self.assertTrue(harness.text(Speech()("bobo ka", 2.5)).opened)

    def test_environment_controls_duplicate_window(self):
        values = detection_runtime_config_from_environment({
            "ECHOSENSE_DUPLICATE_WINDOW_SECONDS": "0",
            "ECHOSENSE_TONE_MIN_ABSOLUTE_RMS": "350",
        })
        self.assertEqual(values.duplicate_window_seconds, 0.0)
        self.assertEqual(values.tone_min_absolute_rms, 350.0)
        with self.assertRaises(ValueError):
            DetectionRuntimeConfig(duplicate_window_seconds=-1)


class AbsoluteToneFloorTests(unittest.TestCase):
    feed = NoiseFloorTests.feed
    steady = staticmethod(NoiseFloorTests.steady)
    bursty = staticmethod(NoiseFloorTests.bursty)

    def quiet_room(self):
        tracker = NoiseFloorTracker.from_config(
            DetectionRuntimeConfig(calibration_seconds=0.0),
        )
        detector = AcousticDetector(noise_floor=tracker)
        at = 1.0
        for index in range(60):
            self.feed(detector, self.steady(64, index), at)
            at += 0.5
        return tracker, detector, at

    def test_normal_hello_in_quiet_room_is_not_aggressive(self):
        tracker, detector, at = self.quiet_room()
        self.assertLess(tracker.floor * tracker.multiplier, 400)
        self.assertEqual(tracker.threshold, 400)
        hello = self.bursty(350)  # RMS ~250: uneven/tense, but normal volume
        evidence = self.feed(detector, hello, at)
        self.assertGreater(evidence.rms, tracker.floor * 2.5)
        self.assertLess(evidence.rms, 400)
        self.assertFalse(evidence.tone_aggressive)
        self.assertIn("tone_rms_threshold:400.0", evidence.reasons)

    def test_raised_voice_in_quiet_room_is_aggressive(self):
        _tracker, detector, at = self.quiet_room()
        self.assertTrue(self.feed(detector, self.bursty(1_200), at).tone_aggressive)

    def test_loud_room_still_uses_rolling_floor(self):
        tracker = NoiseFloorTracker(calibration_seconds=0.0)
        for index in range(20):
            tracker.observe(300.0, monotonic_start=index, monotonic_end=index + 1)
        self.assertEqual(tracker.threshold, 750.0)


class LatencyTrackerTests(unittest.TestCase):
    def test_stage_deltas_and_late_link(self):
        clock = FakeClock()
        lines = []
        tracker = LatencyTracker(clock=clock, logger=lines.append)
        tracker.set_noise_reduction(hold_ms=250, compute_ms=5.0)
        clock.value = 10.0
        tracker.speech_end()
        clock.value = 10.8
        tracker.begin("u1")
        tracker.mark("u1", "whisper_done")
        clock.value = 10.81
        tracker.mark("u1", "transcript_emitted")
        clock.value = 13.81
        # Urgent incidents finalize inside submit, before the worker links.
        tracker.mark_incident("i1", "incident_enqueued")
        tracker.link_incident("i1", "u1")
        clock.value = 13.86
        tracker.mark_incident("i1", "delivered")
        self.assertEqual(len(lines), 1)
        line = lines[0]
        for expected in (
            "outcome=delivered", "incident=i1", "nr_hold_ms=250",
            "whisper_ms=800", "emit_ms=10", "incident_window_ms=3000",
            "delivery_ms=50", "speech_end_to_delivered_ms=3860",
        ):
            self.assertIn(expected, line)

    def test_non_incident_utterance_is_logged_once(self):
        lines = []
        tracker = LatencyTracker(clock=FakeClock(), logger=lines.append)
        tracker.begin("u2")
        tracker.mark("u2", "whisper_done")
        tracker.finish("u2", "no_incident")
        tracker.finish("u2", "no_incident")
        self.assertEqual(len(lines), 1)
        self.assertIn("outcome=no_incident", lines[0])


if __name__ == "__main__":
    unittest.main()
