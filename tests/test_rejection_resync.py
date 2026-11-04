"""P0-1 (edge): a rejected alert is logged loudly and retried once after a re-sync."""

import sqlite3
import unittest
from unittest.mock import patch

from model import monitored_terms
from sender import http_client
from sender.delivery import OutboxDeliveryWorker
from sender.http_client import (
    AUTHENTICATION as HTTP_AUTHENTICATION,
    DELIVERED as HTTP_DELIVERED,
    PERMANENT as HTTP_PERMANENT,
    DeliveryResult,
    device_auth_headers,
    rejection_detail,
)
from sender.outbox import DELIVERED, FAILED_PERMANENT, AlertOutbox
from tests.test_outbox import (
    DEFAULT_EVENT_ID,
    FakeResponse,
    OutboxTestCase,
    test_payload as outbox_payload,
)

REJECTED = DeliveryResult(
    HTTP_PERMANENT,
    http_status=422,
    error_type="http_422",
    detail="Matched text does not correspond to a monitored term",
)


def buanga_payload():
    payload = outbox_payload()
    payload["matched_terms"] = [{"match_type": "word", "term": "buanga"}]
    return payload


class SyncResponse(FakeResponse):
    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class RejectionResyncTests(OutboxTestCase):
    def worker(self, delivery, *, resync=None, refresh_payload=None, **settings):
        return OutboxDeliveryWorker(
            self.outbox,
            self.settings(**settings),
            deliver=delivery,
            clock=self.clock,
            random_uniform=lambda low, high: 0.0,
            logger=self.logs.append,
            resync=resync,
            refresh_payload=refresh_payload,
        )

    def test_rejection_resyncs_once_then_parks_with_a_loud_error(self):
        resyncs = []
        calls = []
        self.outbox.enqueue(buanga_payload())
        worker = self.worker(
            lambda payload: calls.append(payload) or REJECTED,
            resync=lambda: resyncs.append(1) or True,
        )

        worker.run_once()
        self.clock.advance(3600)
        worker.run_once()

        record = self.outbox.get_record(DEFAULT_EVENT_ID)
        self.assertEqual(len(calls), 2)
        self.assertEqual(resyncs, [1])
        self.assertEqual(record.status, FAILED_PERMANENT)
        self.assertEqual(record.rejection_retry_count, 1)
        self.assertEqual(record.last_http_status, 422)
        self.assertIn("does not correspond", record.last_error)
        errors = [line for line in self.logs if "ERROR" in line]
        self.assertTrue(any("re-syncing the dictionary" in line for line in errors))
        final = [line for line in errors if "NOT delivered" in line]
        self.assertEqual(len(final), 1)
        self.assertIn("http=422", final[0])
        self.assertIn(f"sender.outbox_admin retry {DEFAULT_EVENT_ID}", final[0])
        # The transcript never reaches the logs.
        self.assertFalse(any("buanga" in line for line in self.logs))

    def test_retry_after_resync_delivers_with_the_refreshed_term_id(self):
        calls = []
        results = [REJECTED, DeliveryResult(HTTP_DELIVERED, http_status=200)]
        self.outbox.enqueue(buanga_payload())

        def refresh(payload):
            refreshed = dict(payload)
            refreshed["matched_terms"] = [
                {"match_type": "word", "term": "buanga", "term_id": 77, "language": "ceb"}
            ]
            return refreshed

        worker = self.worker(
            lambda payload: calls.append(payload) or results.pop(0),
            resync=lambda: True,
            refresh_payload=refresh,
        )
        worker.run_once()

        self.assertEqual(self.outbox.get_record(DEFAULT_EVENT_ID).status, DELIVERED)
        self.assertNotIn("term_id", calls[0]["matched_terms"][0])
        self.assertEqual(calls[1]["matched_terms"][0]["term_id"], 77)

    def test_failed_resync_still_retries_the_original_payload_once(self):
        calls = []
        self.outbox.enqueue(buanga_payload())

        def broken_resync():
            raise OSError("offline")

        worker = self.worker(
            lambda payload: calls.append(payload) or REJECTED,
            resync=broken_resync,
            refresh_payload=lambda payload: self.fail("no refresh without a sync"),
        )
        worker.run_once()

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(self.outbox.get_record(DEFAULT_EVENT_ID).status, FAILED_PERMANENT)

    def test_manual_operator_retry_gets_a_fresh_resync_retry(self):
        calls = []
        self.outbox.enqueue(buanga_payload())
        worker = self.worker(lambda payload: calls.append(1) or REJECTED, resync=lambda: True)
        worker.run_once()
        self.assertTrue(self.outbox.retry_permanent(DEFAULT_EVENT_ID))
        self.assertEqual(self.outbox.get_record(DEFAULT_EVENT_ID).rejection_retry_count, 0)

        worker.run_once()

        self.assertEqual(len(calls), 4)

    def test_stuck_record_is_delivered_by_operator_retry(self):
        # The buanga case: parked by an old edge, delivered once the backend accepts it.
        self.outbox.enqueue(buanga_payload())
        old_backend = self.worker(lambda payload: REJECTED, max_rejection_retries=0)
        old_backend.run_once()
        self.assertEqual(self.outbox.get_record(DEFAULT_EVENT_ID).status, FAILED_PERMANENT)

        self.outbox.retry_permanent(DEFAULT_EVENT_ID)
        fixed_backend = self.worker(lambda payload: DeliveryResult(HTTP_DELIVERED, http_status=200))
        fixed_backend.run_once()

        self.assertEqual(self.outbox.get_record(DEFAULT_EVENT_ID).status, DELIVERED)

    def test_old_outbox_database_gains_the_retry_counter(self):
        legacy = self.database_path.with_name("legacy.sqlite3")
        connection = sqlite3.connect(legacy)
        connection.execute(
            "CREATE TABLE alert_outbox (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "event_id TEXT NOT NULL UNIQUE, payload_json TEXT NOT NULL)"
        )
        connection.commit()
        connection.close()

        outbox = AlertOutbox(legacy, busy_timeout_ms=100, clock=self.clock)
        outbox.initialize()
        outbox.enqueue(buanga_payload())

        self.assertEqual(outbox.get_record(DEFAULT_EVENT_ID).rejection_retry_count, 0)


class RejectionClassificationTests(unittest.TestCase):
    def deliver(self, response):
        return http_client.deliver_alert_payload(
            outbox_payload(),
            api_url="http://backend.test",
            device_id=outbox_payload()["device_identifier"],
            device_key="key",
            post=lambda *args, **kwargs: response,
        )

    def test_422_keeps_the_backends_reason(self):
        result = self.deliver(FakeResponse(422, {"detail": "Matched text is ambiguous"}))

        self.assertEqual(result.disposition, HTTP_PERMANENT)
        self.assertEqual(result.detail, "Matched text is ambiguous")

    def test_unassigned_device_409_is_kept_and_retried_not_dropped(self):
        result = self.deliver(
            FakeResponse(409, {"detail": "Device is not assigned to a classroom"})
        )

        self.assertEqual(result.disposition, HTTP_AUTHENTICATION)
        self.assertEqual(result.error_type, "device_not_ready_http_409")

    def test_event_conflict_409_stays_permanent(self):
        result = self.deliver(FakeResponse(409, {"detail": "Event payload conflict"}))

        self.assertEqual(result.disposition, HTTP_PERMANENT)

    def test_validation_detail_never_echoes_submitted_values(self):
        detail = rejection_detail(
            FakeResponse(
                422,
                {
                    "detail": [
                        {
                            "loc": ["body", "transcribed_text"],
                            "type": "string_too_long",
                            "msg": "secret transcript",
                            "input": "secret transcript",
                        }
                    ]
                },
            )
        )

        self.assertEqual(detail, "body.transcribed_text:string_too_long")


class DeviceCredentialsOnEveryCallTests(unittest.TestCase):
    def test_dictionary_sync_sends_device_credentials(self):
        captured = {}

        class Session:
            def get(self, url, **kwargs):
                captured.update(kwargs)
                return SyncResponse(200, [{"term_id": 5, "slur_text": "buang ka", "language": "ceb"}])

        with patch.object(
            http_client,
            "device_auth_headers",
            lambda: {"X-EchoSense-Device-Id": "pi-1", "X-EchoSense-Device-Key": "secret"},
        ):
            ok = monitored_terms.sync_backend_dictionary(session=Session())
        monitored_terms.replace_backend_terms([])

        self.assertTrue(ok)
        self.assertEqual(captured["headers"]["X-EchoSense-Device-Id"], "pi-1")

    def test_refused_dictionary_sync_keeps_current_terms(self):
        monitored_terms.replace_backend_terms([{"term_id": 1, "slur_text": "yawa"}])

        class Session:
            def get(self, url, **kwargs):
                return SyncResponse(401, {"detail": "Invalid device credentials"})

        self.assertFalse(monitored_terms.sync_backend_dictionary(session=Session()))
        self.assertEqual([term.term for term in monitored_terms.get_backend_terms()], ["yawa"])
        monitored_terms.replace_backend_terms([])

    def test_log_upload_sends_device_credentials(self):
        with patch.object(http_client.requests, "post") as post, patch.object(
            http_client,
            "device_auth_headers",
            lambda: {"X-EchoSense-Device-Id": "pi-1", "X-EchoSense-Device-Key": "secret"},
        ):
            http_client._send_logs_batch(["[AUDIO] line"])

        self.assertEqual(post.call_args.kwargs["headers"]["X-EchoSense-Device-Key"], "secret")

    def test_device_auth_headers(self):
        self.assertEqual(
            device_auth_headers(" pi-1 ", " key "),
            {"X-EchoSense-Device-Id": "pi-1", "X-EchoSense-Device-Key": "key"},
        )
        self.assertEqual(device_auth_headers("", "key"), {})


class AttachBackendTermIdsTests(unittest.TestCase):
    def tearDown(self):
        monitored_terms.replace_backend_terms([])

    def test_known_terms_get_their_backend_id_and_language(self):
        monitored_terms.replace_backend_terms(
            [{"term_id": 9, "slur_text": "Buang Ka", "language": "ceb"}]
        )
        payload = {
            "event_id": "x",
            "matched_terms": [
                {"term": "buang ka", "language": "fil", "match_type": "phrase"},
                {"term": "buanga", "match_type": "word"},
            ],
        }

        refreshed = monitored_terms.attach_backend_term_ids(payload)

        self.assertEqual(
            refreshed["matched_terms"],
            [
                {"term": "buang ka", "language": "ceb", "match_type": "phrase", "term_id": 9},
                {"term": "buanga", "match_type": "word"},
            ],
        )
        self.assertNotIn("term_id", payload["matched_terms"][0])


if __name__ == "__main__":
    unittest.main()
