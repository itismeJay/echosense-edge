import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from detection.coordinator import EventCoordinator
from sender.http_client import deliver_alert_payload, send_alert
from sender.incident_delivery import (
    enqueue_finalized_incident,
    finalized_incident_send_arguments,
)
from sender.outbox import AlertOutbox, OutboxSerializationError
from tests.test_event_coordinator_after import (
    FakeClock,
    acoustic_evidence,
    text_evidence,
)


DEVICE_ID = "classroom-pi-test"


class _OutboxService:
    def __init__(self, outbox):
        self.outbox = outbox

    def enqueue_payload(self, payload):
        return bool(self.outbox.enqueue(payload).record_id)


class _Response:
    status_code = 201
    headers = {}


class IncidentDeliveryAfterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.database_path = Path(self.temp.name) / "outbox.sqlite3"
        self.outbox = AlertOutbox(self.database_path, logger=lambda _line: None)
        self.outbox.initialize()
        self.clock = FakeClock()

    def run_incident(self, *, acoustic_first=False):
        coordinator = EventCoordinator(
            monotonic_clock=self.clock,
            finalizer=lambda incident: enqueue_finalized_incident(
                incident,
                alert_sender=send_alert,
                device_identifier=DEVICE_ID,
            ),
        )
        service = _OutboxService(self.outbox)
        with patch(
            "sender.delivery.get_alert_delivery_service",
            return_value=service,
        ):
            if acoustic_first:
                coordinator.submit_acoustic(acoustic_evidence(0))
                coordinator.submit_text(text_evidence(1))
            else:
                coordinator.submit_text(text_evidence(0))
                coordinator.submit_acoustic(acoustic_evidence(1))
            event_id = coordinator.active_incident.event_id
            self.clock.value = 5
            finalized = coordinator.tick()[0]
        return event_id, finalized

    def test_both_detectors_produce_one_incident_id_and_one_outbox_row(self):
        event_id, finalized = self.run_incident()
        records = self.outbox.list_records()
        self.assertEqual(len(records), 1)
        payload = json.loads(records[0].payload_json)
        self.assertEqual(event_id, finalized.event_id)
        self.assertEqual(payload["event_id"], event_id)
        self.assertEqual(payload["trigger_type"], "KEYWORD")
        self.assertTrue(payload["detailed_acoustic_evidence"]["yamnet"]["ran"])
        self.assertEqual(
            payload["detailed_acoustic_evidence"]["role"],
            "supporting_only",
        )

    def test_acoustic_opened_schema_stays_acoustic_with_text_support(self):
        event_id, _finalized = self.run_incident(acoustic_first=True)
        payload = json.loads(self.outbox.list_records()[0].payload_json)
        self.assertEqual(payload["event_id"], event_id)
        self.assertEqual(payload["trigger_type"], "ACOUSTIC")
        self.assertTrue(payload["monitored_word_detected"])
        self.assertIn("bobo", payload["monitored_terms"])
        self.assertEqual(
            payload["detailed_acoustic_evidence"]["role"],
            "primary_trigger",
        )

    def test_qualifying_acoustic_label_beats_higher_scoring_neutral_label(self):
        _event_id, finalized = self.run_incident()
        qualifying = acoustic_evidence(1, score=0.46)
        neutral = acoustic_evidence(2, qualifies=False, score=0.99)
        incident = replace(
            finalized,
            acoustic_evidence=(neutral, qualifying),
        )
        arguments = finalized_incident_send_arguments(
            incident,
            device_identifier=DEVICE_ID,
        )
        self.assertEqual(arguments["yamnet_class"], "Screaming")
        self.assertEqual(arguments["yamnet_score"], 0.46)

    def test_acoustic_opener_marks_tone_and_yamnet_as_primary(self):
        self.run_incident(acoustic_first=True)
        payload = json.loads(self.outbox.list_records()[0].payload_json)
        self.assertEqual(
            payload["detailed_acoustic_evidence"]["role"],
            "primary_trigger",
        )
        self.assertEqual(payload["tone_evidence"]["role"], "primary_trigger")

    def test_finalized_arguments_and_payload_contain_no_pcm_or_audio_path(self):
        _event_id, finalized = self.run_incident()
        arguments = finalized_incident_send_arguments(
            finalized,
            device_identifier=DEVICE_ID,
        )
        rendered = json.dumps(arguments, default=str).lower()
        for forbidden in (
            "raw_audio",
            "raw_pcm",
            "pcm_samples",
            "audio_base64",
            "audio_path",
            ".wav",
        ):
            self.assertNotIn(forbidden, rendered)
        self.assertLessEqual(len(arguments["waveform_snapshot"]), 40)
        self.assertTrue(all(
            isinstance(value, int)
            for value in arguments["waveform_snapshot"]
        ))
        self.assertEqual(list(Path(self.temp.name).glob("*.wav")), [])

    def test_outbox_still_rejects_raw_audio_fields(self):
        self.run_incident()
        payload = json.loads(self.outbox.list_records()[0].payload_json)
        payload["raw_pcm"] = [1, 2, 3]
        payload["event_id"] = "33333333-3333-4333-8333-333333333333"
        with self.assertRaises(OutboxSerializationError):
            self.outbox.enqueue(payload)

    def test_retry_identity_remains_event_id_and_idempotency_key(self):
        event_id, _finalized = self.run_incident()
        payload = json.loads(self.outbox.list_records()[0].payload_json)
        post = Mock(return_value=_Response())
        first = deliver_alert_payload(
            payload,
            api_url="http://127.0.0.1:8000",
            device_id=DEVICE_ID,
            device_key="not-a-real-key",
            post=post,
        )
        second = deliver_alert_payload(
            payload,
            api_url="http://127.0.0.1:8000",
            device_id=DEVICE_ID,
            device_key="not-a-real-key",
            post=post,
        )
        self.assertEqual(first.disposition, "DELIVERED")
        self.assertEqual(second.disposition, "DELIVERED")
        self.assertEqual(post.call_count, 2)
        for call in post.call_args_list:
            self.assertEqual(call.kwargs["headers"]["Idempotency-Key"], event_id)


if __name__ == "__main__":
    unittest.main()
