import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import Mock

import requests

from sender.delivery import (
    AlertDeliveryService,
    DeliverySettings,
    OutboxDeliveryWorker,
)
from sender.http_client import (
    DELIVERED as HTTP_DELIVERED,
    PERMANENT as HTTP_PERMANENT,
    RETRYABLE as HTTP_RETRYABLE,
    DeliveryResult,
    deliver_alert_payload,
)
from sender.outbox import (
    DELIVERED,
    FAILED_PERMANENT,
    FAILED_RETRYABLE,
    PENDING,
    SENDING,
    AlertOutbox,
    OutboxSerializationError,
    OutboxStorageError,
    parse_utc_timestamp,
)


class MutableClock:
    def __init__(self):
        self.value = datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += timedelta(seconds=seconds)


class FakeResponse:
    def __init__(self, status_code, data=None, headers=None):
        self.status_code = status_code
        self._data = data
        self.headers = headers or {}

    def json(self):
        if isinstance(self._data, Exception):
            raise self._data
        return self._data


def test_payload(event_id="outbox-test-event"):
    transcript = "NON-PRODUCTION OUTBOX TEST — exact text.\n"
    return {
        "event_id": event_id,
        "severity": "low",
        "confidence": 0.5,
        "duration": 1.25,
        "transcript": transcript,
        "transcribed_text": transcript,
        "detected_words": [],
        "categories": [],
        "language": "en",
        "matched_terms": [],
        "yamnet_class": "NotRun",
        "yamnet_score": 0.0,
        "yamnet_ran": False,
        "waveform_snapshot": [0, 1, 0],
    }


class OutboxTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        self.database_path = Path(self.temp_directory.name) / "outbox.sqlite3"
        self.clock = MutableClock()
        self.logs = []
        self.outbox = AlertOutbox(
            self.database_path,
            busy_timeout_ms=100,
            clock=self.clock,
            logger=self.logs.append,
        )
        self.outbox.initialize()

    def settings(self, **overrides):
        values = {
            "enabled": True,
            "database_path": str(self.database_path),
            "api_url": "http://127.0.0.1:1",
            "poll_interval_seconds": 0.01,
            "request_timeout_seconds": 0.05,
            "initial_retry_delay_seconds": 5.0,
            "backoff_multiplier": 3.0,
            "max_retry_delay_seconds": 30.0,
            "jitter_percent": 0.20,
            "stale_sending_seconds": 10.0,
            "delivered_retention_days": 30.0,
            "max_records_per_cycle": 10,
            "busy_timeout_ms": 100,
            "health_log_interval_seconds": 60.0,
        }
        values.update(overrides)
        return DeliverySettings(**values)

    def worker(self, delivery, **settings):
        return OutboxDeliveryWorker(
            self.outbox,
            self.settings(**settings),
            deliver=delivery,
            clock=self.clock,
            random_uniform=lambda low, high: 0.0,
            logger=self.logs.append,
        )


class AlertOutboxStorageTests(OutboxTestCase):
    def test_enqueue_creates_one_persistent_record(self):
        result = self.outbox.enqueue(test_payload())

        self.assertTrue(result.inserted)
        records = self.outbox.list_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].status, PENDING)

    def test_complete_payload_and_exact_transcript_are_serialized(self):
        payload = test_payload()
        self.outbox.enqueue(payload)

        stored = json.loads(
            self.outbox.get_record(payload["event_id"]).payload_json
        )
        self.assertEqual(stored, payload)
        self.assertEqual(stored["transcript"], payload["transcript"])
        self.assertEqual(
            stored["transcribed_text"],
            payload["transcribed_text"],
        )

    def test_event_id_is_preserved(self):
        payload = test_payload("stable-event-id")
        self.outbox.enqueue(payload)

        record = self.outbox.get_record("stable-event-id")
        self.assertEqual(record.event_id, payload["event_id"])
        self.assertEqual(
            json.loads(record.payload_json)["event_id"],
            payload["event_id"],
        )

    def test_duplicate_event_id_does_not_create_another_row(self):
        first = self.outbox.enqueue(test_payload())
        second = self.outbox.enqueue(test_payload())

        self.assertTrue(first.inserted)
        self.assertFalse(second.inserted)
        self.assertEqual(first.record_id, second.record_id)
        self.assertEqual(len(self.outbox.list_records()), 1)

    def test_pending_record_survives_repository_reinitialization(self):
        self.outbox.enqueue(test_payload())
        reopened = AlertOutbox(
            self.database_path,
            clock=self.clock,
            logger=self.logs.append,
        )
        reopened.initialize()

        record = reopened.get_record("outbox-test-event")
        self.assertEqual(record.status, PENDING)

    def test_pending_record_survives_simulated_process_restart(self):
        payload = test_payload("restart-event")
        self.outbox.enqueue(payload)
        del self.outbox
        restarted = AlertOutbox(
            self.database_path,
            clock=self.clock,
            logger=self.logs.append,
        )
        restarted.initialize()

        self.assertEqual(
            json.loads(restarted.get_record("restart-event").payload_json),
            payload,
        )

    def test_stale_sending_record_is_recovered(self):
        self.outbox.enqueue(test_payload())
        claimed = self.outbox.claim_next(
            lease_seconds=10,
            now=self.clock(),
        )
        self.assertEqual(claimed.status, SENDING)
        self.clock.advance(11)

        recovered = self.outbox.recover_stale_sending(now=self.clock())

        self.assertEqual(recovered, 1)
        self.assertEqual(
            self.outbox.get_record(claimed.event_id).status,
            FAILED_RETRYABLE,
        )

    def test_raw_audio_field_is_not_serialized(self):
        payload = test_payload()
        payload["raw_audio"] = [0.1, 0.2]

        with self.assertRaises(OutboxSerializationError):
            self.outbox.enqueue(payload)
        self.assertEqual(self.outbox.list_records(), [])

    def test_database_lock_is_reported_without_losing_existing_record(self):
        self.outbox.enqueue(test_payload("existing-event"))
        lock = sqlite3.connect(
            str(self.database_path),
            timeout=0.1,
            isolation_level=None,
        )
        self.addCleanup(lock.close)
        lock.execute("BEGIN IMMEDIATE")
        try:
            with self.assertRaises(OutboxStorageError):
                self.outbox.enqueue(test_payload("locked-event"))
        finally:
            lock.execute("ROLLBACK")

        self.assertIsNotNone(self.outbox.get_record("existing-event"))
        self.assertIsNone(self.outbox.get_record("locked-event"))

    def test_database_write_failure_is_reported_by_service(self):
        invalid_parent = Path(self.temp_directory.name) / "not-a-directory"
        invalid_parent.write_text("file")
        service_logs = []
        service = AlertDeliveryService(
            self.settings(
                database_path=str(invalid_parent / "outbox.sqlite3")
            ),
            clock=self.clock,
            logger=service_logs.append,
        )

        self.assertFalse(service.enqueue_payload(test_payload()))
        self.assertTrue(
            any("not durably queued" in line for line in service_logs)
        )

    def test_additive_schema_migration_preserves_old_row(self):
        migration_path = Path(self.temp_directory.name) / "migration.sqlite3"
        connection = sqlite3.connect(migration_path)
        connection.execute(
            """
            CREATE TABLE alert_outbox (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO alert_outbox(event_id, payload_json) VALUES (?, ?)",
            ("old-event", json.dumps(test_payload("old-event"))),
        )
        connection.commit()
        connection.close()
        migrated = AlertOutbox(
            migration_path,
            clock=self.clock,
            logger=self.logs.append,
        )

        migrated.initialize()

        record = migrated.get_record("old-event")
        self.assertEqual(record.event_id, "old-event")
        self.assertEqual(record.status, PENDING)


class DeliveryClassificationTests(unittest.TestCase):
    def deliver(self, response=None, exception=None):
        def post(*_args, **_kwargs):
            if exception:
                raise exception
            return response

        return deliver_alert_payload(
            test_payload(),
            api_url="http://safe.test",
            timeout=0.1,
            post=post,
        )

    def test_any_2xx_is_delivered_even_with_invalid_json(self):
        result = self.deliver(FakeResponse(201, ValueError("invalid json")))
        self.assertEqual(result.disposition, HTTP_DELIVERED)
        self.assertEqual(result.http_status, 201)

    def test_http_500_is_retryable(self):
        result = self.deliver(FakeResponse(500))
        self.assertEqual(result.disposition, HTTP_RETRYABLE)

    def test_http_503_is_retryable(self):
        result = self.deliver(FakeResponse(503))
        self.assertEqual(result.disposition, HTTP_RETRYABLE)

    def test_http_429_is_retryable_and_retry_after_is_parsed(self):
        result = self.deliver(
            FakeResponse(429, headers={"Retry-After": "17"})
        )
        self.assertEqual(result.disposition, HTTP_RETRYABLE)
        self.assertEqual(result.retry_after_seconds, 17.0)

    def test_timeout_is_retryable(self):
        result = self.deliver(exception=requests.Timeout())
        self.assertEqual(result.disposition, HTTP_RETRYABLE)
        self.assertEqual(result.error_type, "timeout")

    def test_connection_refused_is_retryable(self):
        result = self.deliver(
            exception=requests.ConnectionError("Connection refused")
        )
        self.assertEqual(result.disposition, HTTP_RETRYABLE)
        self.assertEqual(result.error_type, "connection_refused")

    def test_dns_failure_is_retryable(self):
        result = self.deliver(
            exception=requests.ConnectionError(
                "Temporary failure in name resolution"
            )
        )
        self.assertEqual(result.disposition, HTTP_RETRYABLE)
        self.assertEqual(result.error_type, "dns_failure")

    def test_http_400_is_permanent(self):
        result = self.deliver(FakeResponse(400))
        self.assertEqual(result.disposition, HTTP_PERMANENT)

    def test_confirmed_duplicate_409_is_delivered(self):
        result = self.deliver(
            FakeResponse(
                409,
                {
                    "duplicate": True,
                    "event_id": "outbox-test-event",
                },
            )
        )
        self.assertEqual(result.disposition, HTTP_DELIVERED)
        self.assertTrue(result.duplicate_acknowledged)

    def test_unconfirmed_409_is_permanent(self):
        result = self.deliver(
            FakeResponse(409, {"detail": "conflict"})
        )
        self.assertEqual(result.disposition, HTTP_PERMANENT)


class OutboxWorkerTests(OutboxTestCase):
    def test_successful_2xx_marks_delivered(self):
        self.outbox.enqueue(test_payload())
        worker = self.worker(
            lambda _payload: DeliveryResult(
                HTTP_DELIVERED,
                http_status=204,
            )
        )

        self.assertEqual(worker.run_once(), 1)
        record = self.outbox.get_record("outbox-test-event")
        self.assertEqual(record.status, DELIVERED)
        self.assertEqual(record.attempt_count, 1)

    def test_retryable_statuses_schedule_retry(self):
        for status in (500, 503, 429):
            with self.subTest(status=status):
                event_id = f"retry-{status}"
                self.outbox.enqueue(test_payload(event_id))
                worker = self.worker(
                    lambda _payload, code=status: DeliveryResult(
                        HTTP_RETRYABLE,
                        http_status=code,
                        error_type=f"http_{code}",
                    )
                )
                worker.run_once(max_records=1)
                record = self.outbox.get_record(event_id)
                self.assertEqual(record.status, FAILED_RETRYABLE)
                self.assertGreater(
                    parse_utc_timestamp(record.next_attempt_at),
                    self.clock(),
                )

    def test_retry_after_is_respected(self):
        self.outbox.enqueue(test_payload())
        worker = self.worker(
            lambda _payload: DeliveryResult(
                HTTP_RETRYABLE,
                http_status=429,
                error_type="http_429",
                retry_after_seconds=20.0,
            )
        )
        worker.run_once()

        record = self.outbox.get_record("outbox-test-event")
        delay = (
            parse_utc_timestamp(record.next_attempt_at) - self.clock()
        ).total_seconds()
        self.assertEqual(delay, 20.0)

    def test_timeout_connection_and_dns_failures_schedule_retry(self):
        for error_type in ("timeout", "connection_refused", "dns_failure"):
            with self.subTest(error_type=error_type):
                event_id = f"network-{error_type}"
                self.outbox.enqueue(test_payload(event_id))
                worker = self.worker(
                    lambda _payload, kind=error_type: DeliveryResult(
                        HTTP_RETRYABLE,
                        error_type=kind,
                    )
                )
                worker.run_once(max_records=1)
                record = self.outbox.get_record(event_id)
                self.assertEqual(record.status, FAILED_RETRYABLE)
                self.assertEqual(record.last_error_type, error_type)

    def test_invalid_400_becomes_permanent_and_is_retained(self):
        self.outbox.enqueue(test_payload())
        worker = self.worker(
            lambda _payload: DeliveryResult(
                HTTP_PERMANENT,
                http_status=400,
                error_type="http_400",
            )
        )
        worker.run_once()

        record = self.outbox.get_record("outbox-test-event")
        self.assertEqual(record.status, FAILED_PERMANENT)
        self.assertIsNotNone(record.payload_json)

    def test_duplicate_acknowledgement_marks_delivered(self):
        self.outbox.enqueue(test_payload())
        worker = self.worker(
            lambda _payload: DeliveryResult(
                HTTP_DELIVERED,
                http_status=409,
                duplicate_acknowledged=True,
            )
        )
        worker.run_once()
        self.assertEqual(
            self.outbox.get_record("outbox-test-event").status,
            DELIVERED,
        )

    def test_delivered_record_is_not_sent_again(self):
        calls = []
        self.outbox.enqueue(test_payload())
        worker = self.worker(
            lambda payload: (
                calls.append(payload["event_id"])
                or DeliveryResult(HTTP_DELIVERED, http_status=200)
            )
        )
        worker.run_once()
        worker.run_once()

        self.assertEqual(calls, ["outbox-test-event"])

    def test_permanent_failure_is_not_automatically_retried(self):
        calls = []
        self.outbox.enqueue(test_payload())
        worker = self.worker(
            lambda payload: (
                calls.append(payload["event_id"])
                or DeliveryResult(HTTP_PERMANENT, http_status=400)
            )
        )
        worker.run_once()
        self.clock.advance(1000)
        worker.run_once()

        self.assertEqual(calls, ["outbox-test-event"])

    def test_retryable_record_is_retried_after_next_attempt_at(self):
        results = [
            DeliveryResult(HTTP_RETRYABLE, error_type="timeout"),
            DeliveryResult(HTTP_DELIVERED, http_status=200),
        ]
        calls = []
        self.outbox.enqueue(test_payload())
        worker = self.worker(
            lambda payload: (
                calls.append(payload["event_id"]) or results.pop(0)
            )
        )
        worker.run_once()
        worker.run_once()
        self.assertEqual(len(calls), 1)
        self.clock.advance(5)
        worker.run_once()

        self.assertEqual(len(calls), 2)
        self.assertEqual(
            self.outbox.get_record("outbox-test-event").status,
            DELIVERED,
        )

    def test_backoff_increases_and_is_capped(self):
        worker = self.worker(
            lambda _payload: DeliveryResult(HTTP_RETRYABLE)
        )
        self.assertEqual(worker.retry_delay_seconds(1), 5.0)
        self.assertEqual(worker.retry_delay_seconds(2), 15.0)
        self.assertEqual(worker.retry_delay_seconds(3), 30.0)
        self.assertEqual(worker.retry_delay_seconds(20), 30.0)

    def test_jitter_remains_in_configured_range(self):
        low_worker = OutboxDeliveryWorker(
            self.outbox,
            self.settings(),
            deliver=lambda _payload: DeliveryResult(HTTP_RETRYABLE),
            clock=self.clock,
            random_uniform=lambda low, high: low,
            logger=self.logs.append,
        )
        high_worker = OutboxDeliveryWorker(
            self.outbox,
            self.settings(),
            deliver=lambda _payload: DeliveryResult(HTTP_RETRYABLE),
            clock=self.clock,
            random_uniform=lambda low, high: high,
            logger=self.logs.append,
        )

        self.assertEqual(low_worker.retry_delay_seconds(1), 4.0)
        self.assertEqual(high_worker.retry_delay_seconds(1), 6.0)

    def test_attempt_count_and_error_persist_across_restart(self):
        self.outbox.enqueue(test_payload())
        worker = self.worker(
            lambda _payload: DeliveryResult(
                HTTP_RETRYABLE,
                error_type="timeout",
            )
        )
        worker.run_once()
        self.clock.advance(5)
        restarted_outbox = AlertOutbox(
            self.database_path,
            clock=self.clock,
            logger=self.logs.append,
        )
        restarted_outbox.initialize()
        restarted_worker = OutboxDeliveryWorker(
            restarted_outbox,
            self.settings(),
            deliver=lambda _payload: DeliveryResult(
                HTTP_RETRYABLE,
                error_type="http_503",
            ),
            clock=self.clock,
            random_uniform=lambda low, high: 0.0,
            logger=self.logs.append,
        )
        restarted_worker.run_once()

        record = restarted_outbox.get_record("outbox-test-event")
        self.assertEqual(record.attempt_count, 2)
        self.assertEqual(record.last_error_type, "http_503")
        self.assertEqual(record.event_id, "outbox-test-event")

    def test_corrupt_payload_is_retained_and_marked_for_review(self):
        self.outbox.enqueue(test_payload())
        connection = sqlite3.connect(self.database_path)
        connection.execute(
            "UPDATE alert_outbox SET payload_json = ?",
            ("{corrupt",),
        )
        connection.commit()
        connection.close()
        worker = self.worker(
            lambda _payload: self.fail("corrupt payload must not be sent")
        )

        worker.run_once()

        record = self.outbox.get_record("outbox-test-event")
        self.assertEqual(record.status, FAILED_PERMANENT)
        self.assertEqual(record.last_error_type, "corrupt_payload")
        self.assertEqual(record.payload_json, "{corrupt")

    def test_health_summary_values_are_safe_and_correct(self):
        payload = test_payload()
        self.outbox.enqueue(payload)
        health = self.outbox.health_summary(now=self.clock())
        serialized = json.dumps(health)

        self.assertEqual(health["database_status"], "OK")
        self.assertEqual(health["pending_count"], 1)
        self.assertEqual(health["sending_count"], 0)
        self.assertNotIn(payload["transcript"], serialized)
        self.assertNotIn("payload_json", serialized)

    def test_worker_starts_once_and_stops_cleanly(self):
        worker = self.worker(
            lambda _payload: DeliveryResult(HTTP_DELIVERED)
        )

        self.assertTrue(worker.start())
        self.assertFalse(worker.start())
        self.assertTrue(worker.running)
        self.assertTrue(worker.stop(join_timeout=1.0))
        self.assertFalse(worker.running)

    def test_enqueue_wakes_background_worker_and_returns_before_delivery(self):
        delivery_started = Mock()

        def slow_delivery(_payload):
            delivery_started()
            time.sleep(0.05)
            return DeliveryResult(HTTP_DELIVERED, http_status=200)

        service = AlertDeliveryService(
            self.settings(),
            deliver=slow_delivery,
            clock=self.clock,
            random_uniform=lambda low, high: 0.0,
            logger=self.logs.append,
        )
        self.assertTrue(service.start())
        started = time.monotonic()
        self.assertTrue(service.enqueue_payload(test_payload()))
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.05)
        deadline = time.monotonic() + 1
        while not delivery_started.called and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(delivery_started.called)
        service.stop()


if __name__ == "__main__":
    unittest.main()
