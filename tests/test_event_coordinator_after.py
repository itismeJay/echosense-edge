from datetime import datetime, timedelta, timezone
import threading
import unittest

import numpy as np

from audio.shared_source import AudioFrame, RollingAudioBuffer
from config import DetectionRuntimeConfig
from detection.coordinator import EventCoordinator
from detection.evidence import AcousticEvidence, TextEvidence


WALL = datetime(2026, 1, 1, tzinfo=timezone.utc)
EVENT_ID = "11111111-1111-4111-8111-111111111111"


class FakeClock:
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self):
        return self.value


def text_evidence(at=0.0, *, qualifies=True, transcript="bobo"):
    return TextEvidence(
        observed_at=WALL + timedelta(seconds=at),
        observed_monotonic=at,
        transcript=transcript,
        normalized_transcript=transcript,
        language="fil",
        language_confidence=0.9,
        matched_terms=(),
        detected_terms=(("bobo",) if qualifies else ()),
        hard_hits=(("bobo",) if qualifies else ()),
        soft_hits=(),
        categories=(("academic_shaming",) if qualifies else ()),
        direct_target=False,
        repeated=False,
        quality_accepted=True,
        quality_reasons=("accepted",),
        quality_warnings=(),
        baseline_severity="MEDIUM" if qualifies else "LOW",
        reasons=(("term_category:direct_harassment",) if qualifies else ()),
        laughter_present=False,
        qualifies_to_open=qualifies,
    )


def acoustic_evidence(at=0.0, *, qualifies=True, score=0.8):
    return AcousticEvidence(
        interval_started_at=WALL + timedelta(seconds=at - 0.975),
        interval_ended_at=WALL + timedelta(seconds=at),
        monotonic_start=at - 0.975,
        monotonic_end=at,
        yamnet_ran=True,
        yamnet_label="Screaming" if qualifies else "Speech",
        yamnet_score=score,
        yamnet_aggressive=qualifies,
        tone_aggressive=False,
        rms=100.0,
        energy_variance=100.0,
        zero_crossing_rate=0.02,
        peak_to_average=2.0,
        duration_seconds=0.975,
        reasons=(("qualifying:yamnet_aggressive_class",) if qualifies else ("normal",)),
        qualifies_to_open=qualifies,
    )


class EventCoordinatorAfterTests(unittest.TestCase):
    def make(self, *, clock=None, config=None, pre=None):
        clock = clock or FakeClock()
        finalized = []
        coordinator = EventCoordinator(
            config=config or DetectionRuntimeConfig(),
            pre_trigger_buffer=pre,
            finalizer=lambda item: finalized.append(item) or True,
            monotonic_clock=clock,
            event_id_factory=lambda: EVENT_ID,
        )
        return coordinator, clock, finalized

    def finish(self, coordinator, clock, at=5.0):
        clock.value = at
        return coordinator.tick()

    def test_text_first_acoustic_joins_one_keyword_event(self):
        coordinator, clock, finalized = self.make()
        first = coordinator.submit_text(text_evidence(0))
        second = coordinator.submit_acoustic(acoustic_evidence(1))
        self.assertEqual(first.event_id, second.event_id)
        self.finish(coordinator, clock)
        self.assertEqual(len(finalized), 1)
        self.assertEqual(finalized[0].trigger_type, "KEYWORD")
        self.assertEqual(len(finalized[0].text_evidence), 1)
        self.assertEqual(len(finalized[0].acoustic_evidence), 1)

    def test_acoustic_first_opens_nothing_and_text_links_to_it(self):
        coordinator, clock, finalized = self.make()
        first = coordinator.submit_acoustic(acoustic_evidence(0))
        self.assertFalse(first.opened)
        self.assertIsNone(first.event_id)
        second = coordinator.submit_text(text_evidence(1))
        self.assertTrue(second.opened)
        self.finish(coordinator, clock)
        self.assertEqual(finalized[0].trigger_type, "KEYWORD")
        self.assertEqual(len(finalized[0].acoustic_evidence), 1)

    def test_active_incident_does_not_retain_non_qualifying_text_content(self):
        coordinator, clock, finalized = self.make()
        coordinator.submit_text(text_evidence(0))
        normal = text_evidence(
            1,
            qualifies=False,
            transcript="What page are we on?",
        )
        submission = coordinator.submit_text(normal)
        self.assertFalse(submission.joined)
        self.assertEqual(len(coordinator.active_incident.text_evidence), 1)
        self.assertEqual(
            coordinator.active_incident.non_qualifying_text_count,
            1,
        )
        self.finish(coordinator, clock)
        self.assertNotIn(
            "What page are we on?",
            [item.transcript for item in finalized[0].text_evidence],
        )

    def test_text_finalizes_once_and_acoustic_only_never_does(self):
        coordinator, clock, finalized = self.make()
        coordinator.submit_text(text_evidence())
        self.finish(coordinator, clock)
        coordinator.tick()
        self.assertEqual(len(finalized), 1)
        self.assertEqual(finalized[0].trigger_type, "KEYWORD")
        coordinator, clock, finalized = self.make()
        coordinator.submit_acoustic(acoustic_evidence())
        self.finish(coordinator, clock)
        self.assertEqual(finalized, [])

    def test_normal_text_and_audio_create_no_event(self):
        coordinator, clock, finalized = self.make()
        coordinator.submit_text(text_evidence(0, qualifies=False, transcript="hello"))
        coordinator.submit_acoustic(acoustic_evidence(0, qualifies=False))
        self.finish(coordinator, clock, 30)
        self.assertIsNone(coordinator.active_incident)
        self.assertEqual(finalized, [])

    def test_exact_timestamp_tie_is_deterministically_keyword(self):
        coordinator, clock, finalized = self.make()
        coordinator.submit_acoustic(acoustic_evidence(0))
        coordinator.submit_text(text_evidence(0))
        self.finish(coordinator, clock)
        self.assertEqual(len(finalized), 1)
        self.assertEqual(finalized[0].trigger_type, "KEYWORD")

    def test_multiple_observations_keep_same_event_id(self):
        coordinator, _clock, _finalized = self.make()
        ids = [coordinator.submit_text(text_evidence(0)).event_id]
        ids.append(coordinator.submit_text(text_evidence(1, transcript="tanga")).event_id)
        ids.append(coordinator.submit_acoustic(acoustic_evidence(2)).event_id)
        self.assertEqual(set(ids), {EVENT_ID})

    def test_evidence_history_and_exact_duplicate_are_bounded(self):
        config = DetectionRuntimeConfig(evidence_history_size=2)
        coordinator, _clock, _finalized = self.make(config=config)
        first = text_evidence(0)
        coordinator.submit_text(first)
        duplicate = coordinator.submit_text(first)
        self.assertFalse(duplicate.joined)
        coordinator.submit_text(text_evidence(1, transcript="tanga one"))
        coordinator.submit_text(text_evidence(2, transcript="tanga two"))
        self.assertEqual(len(coordinator.active_incident.text_evidence), 2)
        self.assertEqual(coordinator.active_incident.evidence_overflow_count, 1)

    def test_no_cooldown_same_terms_duplicate_then_new_event_after_window(self):
        ids = iter((EVENT_ID, "22222222-2222-4222-8222-222222222222"))
        clock = FakeClock()
        finalized = []
        coordinator = EventCoordinator(
            monotonic_clock=clock,
            event_id_factory=lambda: next(ids),
            finalizer=lambda item: finalized.append(item) or True,
        )
        coordinator.submit_text(text_evidence(0))
        self.finish(coordinator, clock, 5)
        # Closed at 3.0: same term 7 s later is a duplicate, 17 s later is new.
        duplicate = coordinator.submit_text(text_evidence(10))
        self.assertEqual(duplicate.duplicate_of, EVENT_ID)
        self.assertFalse(duplicate.opened)
        clock.value = 20
        opened = coordinator.submit_text(text_evidence(20))
        self.assertTrue(opened.opened)
        self.assertNotEqual(opened.event_id, EVENT_ID)

    def test_extension_never_exceeds_hard_maximum(self):
        coordinator, clock, finalized = self.make()
        coordinator.submit_text(text_evidence(0))
        coordinator.submit_text(text_evidence(1, transcript="tanga 1"))
        self.assertEqual(coordinator.active_incident.close_deadline, 2.5)
        for at in (2.0, 3.0, 4.0, 5.0, 5.8):
            coordinator.submit_text(
                text_evidence(at, transcript=f"tanga {at}")
            )
        self.assertEqual(coordinator.active_incident.close_deadline, 6.0)
        clock.value = 6
        coordinator.tick()
        self.assertEqual(len(finalized), 1)
        self.assertEqual(finalized[0].duration_seconds, 6.0)

    def test_pretrigger_context_is_bounded_to_two_seconds(self):
        pre = RollingAudioBuffer(2.0)
        for sequence in range(3):
            pre.append(AudioFrame.canonical(
                np.full(16_000, sequence / 10, dtype=np.float32),
                sequence=sequence,
                monotonic_end=float(sequence + 1),
                captured_at=WALL + timedelta(seconds=sequence + 1),
            ))
        coordinator, _clock, _finalized = self.make(pre=pre)
        coordinator.submit_text(text_evidence(3))
        self.assertLessEqual(
            coordinator.active_incident.audio_context.sample_count,
            32_000,
        )

    def test_shutdown_releases_active_pcm_context(self):
        pre = RollingAudioBuffer(2.0)
        pre.append(AudioFrame.canonical(
            np.ones(16_000, dtype=np.float32) * 0.1,
            sequence=0,
            monotonic_end=1.0,
            captured_at=WALL,
        ))
        coordinator, _clock, _finalized = self.make(pre=pre)
        coordinator.submit_text(text_evidence(1))
        active = coordinator.active_incident
        self.assertGreater(active.audio_context.sample_count, 0)
        coordinator.shutdown(finalize_active=False)
        self.assertEqual(active.audio_context.sample_count, 0)
        self.assertIsNone(coordinator.active_incident)

    def test_backend_timestamps_derive_from_monotonic_duration(self):
        coordinator, clock, finalized = self.make()
        coordinator.submit_text(text_evidence(0))
        clock.value = 5
        coordinator.tick()
        self.assertEqual(
            finalized[0].finalized_at - finalized[0].triggered_at,
            timedelta(seconds=1.5),
        )

    def test_two_detector_threads_race_into_one_incident(self):
        coordinator, _clock, _finalized = self.make()
        barrier = threading.Barrier(3)
        results = []

        def submit(call, evidence):
            barrier.wait()
            results.append(call(evidence).event_id)

        threads = [
            threading.Thread(target=submit, args=(coordinator.submit_text, text_evidence(0))),
            threading.Thread(target=submit, args=(coordinator.submit_acoustic, acoustic_evidence(0))),
        ]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join()
        self.assertEqual(results, [EVENT_ID, EVENT_ID])
        self.assertEqual(coordinator.active_incident.trigger_type, "KEYWORD")

    def test_timing_defaults_and_overrides_are_applied(self):
        default, _clock, _finalized = self.make()
        default.submit_text(text_evidence(0))
        self.assertEqual(default.config.pre_trigger_seconds, 2.0)
        self.assertEqual(default.active_incident.close_deadline, 1.5)
        self.assertEqual(default.active_incident.hard_deadline, 6.0)
        custom_config = DetectionRuntimeConfig(
            pre_trigger_seconds=1.0,
            post_trigger_seconds=2.0,
            event_extension_seconds=1.0,
            max_event_seconds=4.0,
            duplicate_window_seconds=7.0,
        )
        custom, _clock, _finalized = self.make(config=custom_config)
        custom.submit_text(text_evidence(0))
        self.assertEqual(custom.active_incident.close_deadline, 2.0)
        self.assertEqual(custom.active_incident.hard_deadline, 4.0)
        self.assertEqual(custom.config.duplicate_window_seconds, 7.0)


if __name__ == "__main__":
    unittest.main()
