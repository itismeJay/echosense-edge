import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from config import device_credentials_from_environment
from tests.schema_v2_fixtures import (
    DEVICE_ID,
    DEVICE_KEY,
    event_id_for,
    schema_v2_payload,
)
from sender import outbox as outbox_module
from sender.delivery import DeliverySettings, OutboxDeliveryWorker
from sender.http_client import (
    AUTHENTICATION,
    DELIVERED,
    DeliveryResult,
    deliver_alert_payload,
)
from sender.outbox import (
    FAILED_RETRYABLE,
    AlertOutbox,
    parse_utc_timestamp,
)

DEFAULT_EVENT_ID = event_id_for("device-auth-test-event")


def test_payload(event_id=DEFAULT_EVENT_ID):
    return schema_v2_payload(event_id=event_id)


class MutableClock:
    def __init__(self):
        self.value = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += timedelta(seconds=seconds)


class FakeResponse:
    def __init__(self, status_code, data=None):
        self.status_code = status_code
        self._data = data
        self.headers = {}

    def json(self):
        return self._data


class DeviceConfigurationTests(unittest.TestCase):
    def test_credentials_load_from_environment(self):
        device_id, device_key = device_credentials_from_environment(
            {
                "ECHOSENSE_DEVICE_ID": f"  {DEVICE_ID}  ",
                "ECHOSENSE_DEVICE_KEY": f"  {DEVICE_KEY}  ",
            }
        )

        self.assertEqual(device_id, DEVICE_ID)
        self.assertEqual(device_key, DEVICE_KEY)
        self.assertEqual(
            device_credentials_from_environment({}),
            ("", ""),
        )

    def test_delivery_settings_repr_never_contains_key(self):
        settings = DeliverySettings(
            device_id=DEVICE_ID,
            device_key=DEVICE_KEY,
        )

        self.assertNotIn(DEVICE_KEY, repr(settings))


class AuthenticatedHttpDeliveryTests(unittest.TestCase):
    def test_device_headers_are_sent_on_alert_delivery(self):
        captured = {}

        def post(url, **kwargs):
            captured["url"] = url
            captured["kwargs"] = kwargs
            return FakeResponse(200)

        result = deliver_alert_payload(
            test_payload(),
            api_url="https://backend.invalid",
            timeout=1,
            device_id=DEVICE_ID,
            device_key=DEVICE_KEY,
            post=post,
        )

        self.assertEqual(result.disposition, DELIVERED)
        self.assertEqual(
            captured["kwargs"]["headers"],
            {
                "X-EchoSense-Device-Id": DEVICE_ID,
                "X-EchoSense-Device-Key": DEVICE_KEY,
                "Idempotency-Key": test_payload()["event_id"],
            },
        )
        self.assertNotIn("headers", captured["kwargs"]["json"])
        self.assertNotIn(DEVICE_KEY, json.dumps(captured["kwargs"]["json"]))

    def test_retry_reuses_identical_body_event_id_and_idempotency_key(self):
        payload = test_payload(event_id_for("stable-http-retry"))
        calls = []
        responses = [FakeResponse(503), FakeResponse(200)]

        def post(url, **kwargs):
            calls.append((url, kwargs))
            return responses.pop(0)

        first = deliver_alert_payload(
            payload,
            api_url="http://127.0.0.1:8000",
            device_id=DEVICE_ID,
            device_key=DEVICE_KEY,
            post=post,
        )
        second = deliver_alert_payload(
            payload,
            api_url="http://127.0.0.1:8000",
            device_id=DEVICE_ID,
            device_key=DEVICE_KEY,
            post=post,
        )

        self.assertEqual(first.disposition, "RETRYABLE")
        self.assertEqual(second.disposition, DELIVERED)
        self.assertEqual(calls[0][1]["json"], calls[1][1]["json"])
        self.assertEqual(calls[0][1]["json"]["event_id"], payload["event_id"])
        for _url, kwargs in calls:
            self.assertEqual(
                kwargs["headers"]["Idempotency-Key"],
                payload["event_id"],
            )

    def test_invalid_sender_configuration_fails_without_http(self):
        for overrides, error_type in (
            ({"api_url": ""}, "backend_url_invalid"),
            ({"api_url": "not-a-url"}, "backend_url_invalid"),
            ({"timeout": 0}, "request_timeout_invalid"),
        ):
            with self.subTest(error_type=error_type):
                post_called = False

                def post(*_args, **_kwargs):
                    nonlocal post_called
                    post_called = True
                    return FakeResponse(200)

                values = {
                    "api_url": "http://127.0.0.1:8000",
                    "timeout": 1,
                }
                values.update(overrides)
                result = deliver_alert_payload(
                    test_payload(),
                    device_id=DEVICE_ID,
                    device_key=DEVICE_KEY,
                    post=post,
                    **values,
                )

                self.assertEqual(result.error_type, error_type)
                self.assertFalse(post_called)

    def test_device_identity_mismatch_is_retained_without_http(self):
        with patch("sender.http_client.requests.post") as post:
            result = deliver_alert_payload(
                test_payload(),
                api_url="http://127.0.0.1:8000",
                device_id="different-device",
                device_key=DEVICE_KEY,
            )

        self.assertEqual(result.disposition, AUTHENTICATION)
        self.assertEqual(result.error_type, "device_identity_mismatch")
        post.assert_not_called()

    def test_missing_credentials_fails_without_http_request(self):
        post_called = False

        def post(*_args, **_kwargs):
            nonlocal post_called
            post_called = True
            return FakeResponse(200)

        result = deliver_alert_payload(
            test_payload(),
            api_url="https://backend.invalid",
            device_id="",
            device_key="",
            post=post,
        )

        self.assertEqual(result.disposition, AUTHENTICATION)
        self.assertEqual(result.error_type, "device_credentials_missing")
        self.assertFalse(post_called)

    def test_401_and_403_are_authentication_failures(self):
        for status in (401, 403):
            with self.subTest(status=status):
                result = deliver_alert_payload(
                    test_payload(),
                    api_url="https://backend.invalid",
                    device_id=DEVICE_ID,
                    device_key=DEVICE_KEY,
                    post=lambda *_args, code=status, **_kwargs: FakeResponse(
                        code
                    ),
                )

                self.assertEqual(result.disposition, AUTHENTICATION)
                self.assertEqual(result.http_status, status)


class AuthenticatedOutboxTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        self.database_path = (
            Path(self.temp_directory.name) / "alert_outbox.sqlite3"
        )
        self.clock = MutableClock()
        self.logs = []
        self.outbox = AlertOutbox(
            self.database_path,
            clock=self.clock,
            logger=self.logs.append,
        )
        self.outbox.initialize()

    def settings(self, **overrides):
        values = {
            "enabled": True,
            "database_path": str(self.database_path),
            "api_url": "https://backend.invalid",
            "device_id": DEVICE_ID,
            "device_key": DEVICE_KEY,
            "poll_interval_seconds": 1.0,
            "request_timeout_seconds": 1.0,
            "initial_retry_delay_seconds": 5.0,
            "backoff_multiplier": 3.0,
            "max_retry_delay_seconds": 300.0,
            "jitter_percent": 0.0,
            "stale_sending_seconds": 60.0,
            "delivered_retention_days": 30.0,
            "max_records_per_cycle": 10,
            "busy_timeout_ms": 100,
            "health_log_interval_seconds": 60.0,
        }
        values.update(overrides)
        return DeliverySettings(**values)

    def worker(self, **settings):
        return OutboxDeliveryWorker(
            self.outbox,
            self.settings(**settings),
            clock=self.clock,
            random_uniform=lambda _low, _high: 0.0,
            logger=self.logs.append,
        )

    def test_401_and_403_are_retained_with_backoff_and_redacted_logs(self):
        for status in (401, 403):
            with self.subTest(status=status):
                event_id = event_id_for(f"auth-{status}")
                self.outbox.enqueue(test_payload(event_id))
                with patch(
                    "sender.http_client.requests.post",
                    return_value=FakeResponse(status),
                ) as post:
                    worker = self.worker()
                    self.assertEqual(worker.run_once(max_records=1), 1)
                    self.assertEqual(worker.run_once(max_records=1), 0)

                record = self.outbox.get_record(event_id)
                self.assertEqual(record.status, FAILED_RETRYABLE)
                self.assertEqual(record.last_http_status, status)
                self.assertGreater(
                    parse_utc_timestamp(record.next_attempt_at),
                    self.clock(),
                )
                self.assertEqual(post.call_count, 1)

        serialized_logs = "\n".join(self.logs)
        self.assertNotIn(DEVICE_KEY, serialized_logs)
        self.assertNotIn("X-EchoSense-Device-Key", serialized_logs)
        self.assertNotIn("X-EchoSense-Device-Id", serialized_logs)

    def test_missing_credentials_are_retained_without_tight_loop(self):
        self.outbox.enqueue(test_payload())
        with patch("sender.http_client.requests.post") as post:
            worker = self.worker(device_id="", device_key="")
            self.assertEqual(worker.run_once(), 1)
            self.assertEqual(worker.run_once(), 0)

        record = self.outbox.get_record(DEFAULT_EVENT_ID)
        self.assertEqual(record.status, FAILED_RETRYABLE)
        self.assertEqual(
            record.last_error_type,
            "device_credentials_missing",
        )
        self.assertEqual(record.attempt_count, 1)
        post.assert_not_called()

    def test_authentication_retry_survives_restart_with_same_event_id(self):
        event_id = event_id_for("stable-auth-restart-event")
        self.outbox.enqueue(test_payload(event_id))
        first_worker = OutboxDeliveryWorker(
            self.outbox,
            self.settings(),
            deliver=lambda _payload: DeliveryResult(
                AUTHENTICATION,
                http_status=401,
                error_type="device_authentication_http_401",
            ),
            clock=self.clock,
            random_uniform=lambda _low, _high: 0.0,
            logger=self.logs.append,
        )
        first_worker.run_once()
        self.clock.advance(5)

        restarted = AlertOutbox(
            self.database_path,
            clock=self.clock,
            logger=self.logs.append,
        )
        restarted.initialize()
        delivered_event_ids = []
        restarted_worker = OutboxDeliveryWorker(
            restarted,
            self.settings(),
            deliver=lambda payload: (
                delivered_event_ids.append(payload["event_id"])
                or DeliveryResult(DELIVERED, http_status=200)
            ),
            clock=self.clock,
            random_uniform=lambda _low, _high: 0.0,
            logger=self.logs.append,
        )

        restarted_worker.run_once()

        record = restarted.get_record(event_id)
        self.assertEqual(delivered_event_ids, [event_id])
        self.assertEqual(record.event_id, event_id)
        self.assertEqual(record.status, outbox_module.DELIVERED)
        self.assertEqual(record.attempt_count, 2)


if __name__ == "__main__":
    unittest.main()
