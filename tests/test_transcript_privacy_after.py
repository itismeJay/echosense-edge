"""Privacy regressions at coordinator, schema-v2, and SQLite boundaries."""

from contextlib import redirect_stdout
from datetime import timedelta
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from detection.acoustic_detector import AcousticDetector
from detection.coordinator import EventCoordinator
from detection.text_detector import TextDetector
from sender.http_client import send_alert
from sender.incident_delivery import enqueue_finalized_incident
from sender.outbox import AlertOutbox
from tests.test_event_coordinator_after import FakeClock, WALL, acoustic_evidence


DEVICE_ID = "classroom-pi-privacy-test"


class _OutboxService:
    def __init__(self, outbox):
        self.outbox = outbox

    def enqueue_payload(self, payload):
        return bool(self.outbox.enqueue(payload).record_id)


class TranscriptPrivacyAfterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.outbox = AlertOutbox(
            Path(self.temp.name) / "outbox.sqlite3",
            logger=lambda _line: None,
        )
        self.outbox.initialize()
        self.clock = FakeClock()
        self.detector = TextDetector(monotonic_clock=self.clock)
        self.finalized = []
        self.coordinator = EventCoordinator(
            monotonic_clock=self.clock,
            finalizer=self._enqueue,
        )
        self.service = _OutboxService(self.outbox)

    def _enqueue(self, incident):
        self.finalized.append(incident)
        return enqueue_finalized_incident(
            incident,
            alert_sender=send_alert,
            device_identifier=DEVICE_ID,
        )

    def _text(self, text, at):
        self.clock.value = at
        return self.detector.analyze_transcript(
            text,
            language="fil",
            observed_at=WALL + timedelta(seconds=at),
            observed_monotonic=at,
        )

    def _finish(self, at):
        self.clock.value = at
        with patch(
            "sender.delivery.get_alert_delivery_service",
            return_value=self.service,
        ):
            self.coordinator.tick()
        self.assertEqual(len(self.finalized), 1)
        records = self.outbox.list_records()
        self.assertEqual(len(records), 1)
        return json.loads(records[0].payload_json), records[0].payload_json

    def test_acoustic_opener_excludes_normal_classroom_transcript_everywhere(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.coordinator.submit_acoustic(acoustic_evidence(0))
            normal = self._text("What page are we on?", 1)
            self.assertFalse(normal.qualifies_to_open)
            self.coordinator.submit_text(normal)
            payload, serialized = self._finish(5)
        self.assertEqual(payload["trigger_type"], "ACOUSTIC")
        self.assertEqual(payload["transcription_status"], "unavailable")
        self.assertNotIn("transcribed_text", payload)
        self.assertNotIn("what page", serialized.casefold())
        self.assertNotIn("what page", output.getvalue().casefold())
        self.assertEqual(self.finalized[0].text_evidence, ())

    def test_acoustic_opener_includes_only_later_qualifying_text(self):
        self.coordinator.submit_acoustic(acoustic_evidence(0))
        normal = self._text("Please open your books.", 1)
        qualifying = self._text("pangit ka", 2)
        self.coordinator.submit_text(normal)
        self.coordinator.submit_text(qualifying)
        payload, serialized = self._finish(5)
        self.assertEqual(payload["trigger_type"], "ACOUSTIC")
        self.assertEqual(payload["transcribed_text"], "pangit ka")
        self.assertNotIn("open your books", serialized.casefold())

    def test_text_opener_preserves_qualifying_transcript(self):
        evidence = self._text("bobo ka", 0)
        self.assertTrue(evidence.qualifies_to_open)
        self.coordinator.submit_text(evidence)
        payload, _serialized = self._finish(5)
        self.assertEqual(payload["trigger_type"], "KEYWORD")
        self.assertEqual(payload["transcribed_text"], "bobo ka")

    def test_pangit_ka_text_only_creates_one_keyword_row(self):
        evidence = self._text("pangit ka", 0)
        self.assertTrue(evidence.qualifies_to_open)
        self.coordinator.submit_text(evidence)
        payload, _serialized = self._finish(5)
        self.assertEqual(payload["trigger_type"], "KEYWORD")
        self.assertEqual(payload["transcribed_text"], "pangit ka")
        self.assertEqual(len(self.outbox.list_records()), 1)

    def test_multiple_observations_are_filtered_deduplicated_and_ordered(self):
        self.coordinator.submit_acoustic(acoustic_evidence(0))
        for text, at in (
            ("What page are we on?", 1),
            ("pangit ka", 2),
            ("Please open your books.", 2.5),
            ("bobo ka", 3),
            ("pangit ka", 3.5),
        ):
            self.coordinator.submit_text(self._text(text, at))
        payload, serialized = self._finish(6)
        self.assertEqual(payload["transcribed_text"], "pangit ka\nbobo ka")
        self.assertNotIn("what page", serialized.casefold())
        self.assertNotIn("open your books", serialized.casefold())

    def test_acoustic_only_is_valid_with_neutral_transcription_state(self):
        self.coordinator.submit_acoustic(acoustic_evidence(0, score=0.46))
        payload, _serialized = self._finish(5)
        self.assertEqual(payload["trigger_type"], "ACOUSTIC")
        self.assertIn(payload["severity"], {"LOW", "MEDIUM"})
        self.assertEqual(payload["transcription_status"], "unavailable")
        self.assertNotIn("transcribed_text", payload)
        self.assertFalse(payload["monitored_word_detected"])

    def test_loud_neutral_window_creates_no_incident_or_outbox_row(self):
        detector = AcousticDetector(
            yamnet_runner=lambda _samples: ("Bang", 0.99),
        )
        neutral_tone = {
            "rms": 9_000.0,
            "energy_variance": 10.0,
            "zero_crossing_rate": 0.01,
            "peak_to_average": 12.0,
            "is_aggressive_tone": False,
        }
        with patch(
            "detection.acoustic_detector.analyze_tone_float32",
            return_value=neutral_tone,
        ):
            evidence = detector.analyze_window(
                np.zeros(15_600, dtype=np.float32),
                interval_started_at=WALL,
                interval_ended_at=WALL + timedelta(seconds=0.975),
                monotonic_start=0.0,
                monotonic_end=0.975,
            )
        self.assertFalse(evidence.qualifies_to_open)
        self.coordinator.submit_acoustic(evidence)
        self.clock.value = 30
        self.coordinator.tick()
        self.assertIsNone(self.coordinator.active_incident)
        self.assertEqual(self.outbox.list_records(), [])


if __name__ == "__main__":
    unittest.main()
