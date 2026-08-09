import json
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from uuid import UUID, uuid4

from sender.http_client import send_alert
from sender.payload_v2 import (
    AlertPayloadValidationError,
    REVIEW_MESSAGE,
    build_alert_payload_v2,
)
from tests.schema_v2_fixtures import DEVICE_ID, schema_v2_payload


class AlertPayloadV2Tests(unittest.TestCase):
    def test_exact_contract_and_legacy_fields_are_absent(self):
        payload = schema_v2_payload()

        self.assertEqual(payload["schema_version"], 2)
        self.assertIs(type(payload["schema_version"]), int)
        self.assertEqual(payload["trigger_type"], "KEYWORD")
        self.assertEqual(payload["severity"], "MEDIUM")
        self.assertEqual(payload["review_message"], REVIEW_MESSAGE)
        self.assertEqual(payload["device_identifier"], DEVICE_ID)
        self.assertEqual(
            payload["transcribed_text"],
            "NON-PRODUCTION OUTBOX TEST — exact text.\n",
        )
        self.assertEqual(UUID(payload["event_id"]), UUID(payload["event_id"]))
        for legacy_field in (
            "location",
            "classroom",
            "school",
            "transcript",
            "detected_words",
            "categories",
            "hard_hits",
            "soft_hits",
            "yamnet_class",
            "yamnet_score",
            "yamnet_ran",
            "emotion",
            "rms",
            "energy_variance",
            "zero_crossing_rate",
            "peak_to_average",
            "duration",
            "duration_gate",
            "required_duration",
        ):
            self.assertNotIn(legacy_field, payload)

    def test_event_id_is_canonical_and_invalid_uuid_is_rejected(self):
        source = uuid4()
        payload = schema_v2_payload(event_id=str(source).upper())
        self.assertEqual(payload["event_id"], str(source))

        with self.assertRaises(AlertPayloadValidationError):
            schema_v2_payload(event_id="not-a-uuid")

    def test_trigger_type_is_keyword_and_invalid_trigger_is_rejected(self):
        self.assertEqual(schema_v2_payload()["trigger_type"], "KEYWORD")
        with self.assertRaises(AlertPayloadValidationError):
            schema_v2_payload(trigger_type="UNSUPPORTED")
        with self.assertRaises(AlertPayloadValidationError):
            schema_v2_payload(trigger_type="TEST")

    def test_severity_is_uppercase_and_invalid_value_is_rejected(self):
        for source, expected in (
            ("LOW", "LOW"),
            ("MEDIUM", "MEDIUM"),
            ("HIGH", "HIGH"),
        ):
            with self.subTest(source=source):
                evidence = {
                    "level": source,
                    "reasons": ["synthetic"],
                    "term_categories": {},
                    "supporting_evidence": [],
                }
                payload = schema_v2_payload(
                    severity=source,
                    severity_evidence=evidence,
                )
                self.assertEqual(payload["severity"], expected)
                self.assertNotEqual(payload["severity"], expected.lower())

        with self.assertRaises(AlertPayloadValidationError):
            schema_v2_payload(severity="CRITICAL")

    def test_timestamps_are_utc_aware_and_stable(self):
        end = datetime(2026, 8, 4, 8, 0, 2, tzinfo=timezone(timedelta(hours=8)))
        start = end - timedelta(seconds=2)
        first = schema_v2_payload(
            event_start_timestamp=start,
            event_end_timestamp=end,
        )
        second = schema_v2_payload(
            event_start_timestamp=start,
            event_end_timestamp=end,
        )

        self.assertEqual(first, second)
        self.assertEqual(first["event_end_timestamp"], "2026-08-04T00:00:02.000000Z")
        with self.assertRaises(AlertPayloadValidationError):
            schema_v2_payload(event_start_timestamp=datetime(2026, 8, 4))

    def test_current_evidence_is_structured_without_acoustic_trigger_claim(self):
        evidence = {
            "level": "HIGH",
            "reasons": ["term_category:threat_like_phrase"],
            "term_categories": {"threat_like_phrase": ["synthetic phrase"]},
            "supporting_evidence": [
                "direct_target_pattern",
                "repeated_monitored_evidence",
                "aggressive_acoustic_evidence",
            ],
        }
        payload = schema_v2_payload(
            severity="HIGH",
            severity_evidence=evidence,
            yamnet_ran=True,
            yamnet_class="Screaming",
            yamnet_score=0.81,
            emotion="angry",
            tone_data={
                "rms": 250,
                "energy_variance": 2000,
                "zero_crossing_rate": 0.12,
                "peak_to_average": 3.2,
            },
        )

        self.assertEqual(payload["severity_evidence"], evidence)
        self.assertEqual(
            payload["detailed_acoustic_evidence"]["role"],
            "supporting_only",
        )
        self.assertEqual(
            payload["detailed_acoustic_evidence"]["yamnet"],
            {"ran": True, "label": "Screaming", "score": 0.81},
        )
        self.assertEqual(payload["tone_evidence"]["emotion"], "angry")
        self.assertTrue(payload["repetition_evidence"]["detected"])
        self.assertTrue(payload["direct_address_evidence"]["detected"])
        self.assertNotIn("acoustic_trigger_evidence", payload)
        self.assertEqual(payload["trigger_type"], "KEYWORD")

    def test_waveform_is_sparse_only_and_raw_pcm_is_rejected(self):
        waveform = list(range(40))
        payload = schema_v2_payload(waveform_snapshot=waveform)
        self.assertEqual(payload["waveform_snapshot"], waveform)
        self.assertNotIn("audio", json.dumps(payload).lower())

        with self.assertRaises(AlertPayloadValidationError):
            schema_v2_payload(waveform_snapshot=bytes([1, 2, 3]))
        with self.assertRaises(AlertPayloadValidationError):
            build_alert_payload_v2(
                **{
                    **self._builder_values(),
                    "severity_evidence": {
                        "level": "MEDIUM",
                        "reasons": ["synthetic"],
                        "term_categories": {"raw_audio": ["forbidden"]},
                        "supporting_evidence": [],
                    },
                }
            )

    @staticmethod
    def _builder_values():
        end = datetime(2026, 8, 4, tzinfo=timezone.utc)
        return {
            "event_id": uuid4(),
            "trigger_type": "KEYWORD",
            "severity": "MEDIUM",
            "event_start_timestamp": end - timedelta(seconds=1),
            "event_end_timestamp": end,
            "device_identifier": DEVICE_ID,
            "detected_words": ["synthetic"],
        }

    @patch("sender.delivery.get_alert_delivery_service")
    def test_send_alert_persists_final_body_and_propagates_enqueue_result(
        self,
        get_service,
    ):
        service = get_service.return_value
        service.enqueue_payload.return_value = False
        end = datetime(2026, 8, 4, tzinfo=timezone.utc)

        with patch("sender.http_client.requests.post") as post:
            queued = send_alert(
                "HIGH",
                0.9,
                2.0,
                transcribed_text="synthetic phrase",
                detected_words=["synthetic phrase"],
                severity_evidence={
                    "level": "HIGH",
                    "reasons": ["synthetic"],
                    "term_categories": {},
                    "supporting_evidence": [],
                },
                event_id=str(uuid4()),
                event_start_timestamp=end - timedelta(seconds=2),
                event_end_timestamp=end,
                device_identifier=DEVICE_ID,
            )

        self.assertFalse(queued)
        post.assert_not_called()
        payload = service.enqueue_payload.call_args.args[0]
        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["severity"], "HIGH")
        self.assertEqual(payload["trigger_type"], "KEYWORD")
        service.enqueue_payload.assert_called_once()

    @patch("sender.delivery.get_alert_delivery_service")
    def test_invalid_payload_never_reaches_outbox(self, get_service):
        self.assertFalse(
            send_alert(
                "INVALID",
                0.9,
                2.0,
                event_id="not-a-uuid",
                device_identifier=DEVICE_ID,
            )
        )
        get_service.assert_not_called()


if __name__ == "__main__":
    unittest.main()
