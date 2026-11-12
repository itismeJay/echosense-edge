import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from detection.coordinator import EventCoordinator
from sender.delivery import DeliverySettings, OutboxDeliveryWorker
from sender.http_client import deliver_alert_payload, send_alert
from sender.http_client import DeliveryResult
from sender.incident_delivery import (
    enqueue_finalized_incident,
    finalized_incident_send_arguments,
)
from sender.outbox import DELIVERED, AlertOutbox, OutboxSerializationError
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

    def run_incident(self, *, acoustic_first=False, dashboard_test_mode=False):
        coordinator = EventCoordinator(
            monotonic_clock=self.clock,
            finalizer=lambda incident: enqueue_finalized_incident(
                incident,
                alert_sender=send_alert,
                device_identifier=DEVICE_ID,
                dashboard_test_mode=dashboard_test_mode,
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

    def test_sound_first_still_produces_a_keyword_incident(self):
        # Sound never opens an incident; the later word does, and the earlier
        # sound is attached as supporting evidence.
        event_id, _finalized = self.run_incident(acoustic_first=True)
        payload = json.loads(self.outbox.list_records()[0].payload_json)
        self.assertEqual(payload["event_id"], event_id)
        self.assertEqual(payload["trigger_type"], "KEYWORD")
        self.assertTrue(payload["monitored_word_detected"])
        self.assertIn("bobo", payload["monitored_terms"])
        self.assertEqual(
            payload["detailed_acoustic_evidence"]["role"],
            "supporting_only",
        )

    def test_dashboard_test_mode_labels_genuine_detector_incident(self):
        event_id, finalized = self.run_incident(dashboard_test_mode=True)
        records = self.outbox.list_records()
        self.assertEqual(len(records), 1)
        payload = json.loads(records[0].payload_json)

        self.assertEqual(payload["event_id"], event_id)
        self.assertEqual(payload["trigger_type"], "TEST")
        self.assertTrue(payload["test_mode"])
        self.assertEqual(payload["severity"], finalized.severity)
        self.assertEqual(
            payload["severity_evidence"]["reasons"],
            list(finalized.severity_reasons),
        )
        self.assertTrue(payload["monitored_word_detected"])
        self.assertIn("bobo", payload["monitored_terms"])

    def test_dashboard_test_incident_reaches_delivery_and_visibility_check(self):
        event_id, _finalized = self.run_incident(dashboard_test_mode=True)
        verifier = Mock()
        deliver = Mock(return_value=DeliveryResult(
            "DELIVERED",
            http_status=201,
        ))
        settings = DeliverySettings(
            enabled=True,
            database_path=str(self.database_path),
            api_url="http://127.0.0.1:8000",
            device_id=DEVICE_ID,
            device_key="private-test-key",
            poll_interval_seconds=0.1,
            request_timeout_seconds=1.0,
            dashboard_test_mode=True,
            dashboard_test_reviewer_token="memory-only-token",
        )
        worker = OutboxDeliveryWorker(
            self.outbox,
            settings,
            deliver=deliver,
            dashboard_verifier=verifier,
            logger=lambda _line: None,
        )

        self.assertEqual(worker.run_once(), 1)
        record = self.outbox.get_record(event_id)
        self.assertEqual(record.status, DELIVERED)
        delivered_payload = deliver.call_args.args[0]
        self.assertEqual(delivered_payload["trigger_type"], "TEST")
        self.assertTrue(delivered_payload["test_mode"])
        verifier.schedule.assert_called_once()
        self.assertEqual(
            verifier.schedule.call_args.args[0]["event_id"],
            event_id,
        )

    def test_test_label_is_the_only_incident_argument_mode_change(self):
        _event_id, finalized = self.run_incident()
        normal = finalized_incident_send_arguments(
            finalized,
            device_identifier=DEVICE_ID,
            dashboard_test_mode=False,
        )
        test = finalized_incident_send_arguments(
            finalized,
            device_identifier=DEVICE_ID,
            dashboard_test_mode=True,
        )

        self.assertEqual(test.pop("trigger_type"), "TEST")
        self.assertTrue(test.pop("test_mode"))
        self.assertEqual(normal.pop("trigger_type"), finalized.trigger_type)
        self.assertFalse(normal.pop("test_mode"))
        self.assertEqual(test, normal)

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

    def test_acoustic_laughter_observation_reaches_structured_payload(self):
        coordinator = EventCoordinator(
            monotonic_clock=self.clock,
            finalizer=lambda incident: enqueue_finalized_incident(
                incident,
                alert_sender=send_alert,
                device_identifier=DEVICE_ID,
            ),
        )
        laughter = replace(
            acoustic_evidence(0.5, qualifies=False, score=0.82),
            yamnet_label="Laughter",
            laughter_detected=True,
            laughter_score=0.82,
        )
        service = _OutboxService(self.outbox)
        with patch(
            "sender.delivery.get_alert_delivery_service",
            return_value=service,
        ):
            coordinator.submit_text(text_evidence(0))
            coordinator.submit_acoustic(laughter)
            coordinator.submit_acoustic(acoustic_evidence(1))
            self.clock.value = 5
            coordinator.tick()

        payload = json.loads(self.outbox.list_records()[0].payload_json)
        self.assertTrue(payload["laughter_context"]["detected"])
        self.assertEqual(payload["laughter_context"]["score"], 0.82)
        self.assertEqual(payload["laughter_context"]["sources"], ["yamnet"])

    def test_sound_is_never_the_primary_trigger(self):
        self.run_incident(acoustic_first=True)
        payload = json.loads(self.outbox.list_records()[0].payload_json)
        self.assertEqual(
            payload["detailed_acoustic_evidence"]["role"],
            "supporting_only",
        )
        self.assertNotEqual(payload["tone_evidence"]["role"], "primary_trigger")

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

    def test_acoustic_features_do_not_claim_emotion_or_intent(self):
        _event_id, finalized = self.run_incident()
        aggressive_tone = replace(
            finalized.acoustic_evidence[0],
            tone_aggressive=True,
        )
        arguments = finalized_incident_send_arguments(replace(
            finalized,
            acoustic_evidence=(aggressive_tone,),
        ))
        self.assertEqual(arguments["emotion"], "unknown")
        self.assertTrue(arguments["tone_data"])

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
