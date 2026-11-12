import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from sender.dashboard_test import (
    DashboardTestConfigurationError,
    DashboardVisibilityVerifier,
    validate_dashboard_test_configuration,
)
from tests.schema_v2_fixtures import schema_v2_payload
from tools import dashboard_test_mode


class _Response:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body


class DashboardTestConfigurationTests(unittest.TestCase):
    def valid_values(self, **overrides):
        values = {
            "enabled": True,
            "api_url": "http://127.0.0.1:8000",
            "expected_backend_origin": "http://127.0.0.1:8000",
            "device_id": "dashboard-test-device",
            "expected_device_id": "dashboard-test-device",
            "device_key": "private-test-key",
            "outbox_enabled": True,
            "outbox_database_path": "/tmp/dashboard_test_outbox.sqlite3",
            "reviewer_token": "memory-only-token",
        }
        values.update(overrides)
        return values

    def test_complete_test_configuration_is_accepted(self):
        validate_dashboard_test_configuration(**self.valid_values())

    def test_mode_fails_closed_on_destination_device_or_outbox_mismatch(self):
        for override in (
            {"expected_backend_origin": "http://127.0.0.1:9000"},
            {"expected_device_id": "another-device"},
            {"outbox_database_path": "/var/lib/echosense/alerts.sqlite3"},
            {"reviewer_token": ""},
        ):
            with self.subTest(override=override):
                with self.assertRaises(DashboardTestConfigurationError):
                    validate_dashboard_test_configuration(
                        **self.valid_values(**override)
                    )

    def test_exactly_approved_production_destination_is_accepted(self):
        validate_dashboard_test_configuration(**self.valid_values(
            api_url="https://echosense-backend-75h3.onrender.com",
            expected_backend_origin=(
                "https://echosense-backend-75h3.onrender.com"
            ),
        ))


class DashboardVisibilityVerifierTests(unittest.TestCase):
    def test_test_event_is_confirmed_in_authenticated_dashboard_feed(self):
        payload = schema_v2_payload(trigger_type="TEST", test_mode=True)
        logs = []
        calls = []

        def get(url, **kwargs):
            calls.append((url, kwargs))
            return _Response(200, [{
                "id": 42,
                "event_id": payload["event_id"],
                "trigger_type": "TEST",
                "test_mode": True,
            }])

        verifier = DashboardVisibilityVerifier(
            "http://127.0.0.1:8000",
            "reviewer-token",
            get=get,
            wall_clock=lambda: datetime(2026, 8, 4, 0, 0, 3,
                                        tzinfo=timezone.utc),
            logger=logs.append,
        )

        self.assertTrue(verifier.verify(payload))
        self.assertEqual(
            calls[0][1]["params"]["event_id"],
            payload["event_id"],
        )
        self.assertEqual(
            calls[0][1]["headers"]["Authorization"],
            "Bearer reviewer-token",
        )
        self.assertIn("visible=true", logs[0])
        self.assertNotIn("reviewer-token", logs[0])
        self.assertNotIn(payload["transcribed_text"], logs[0])

    def test_non_test_payload_is_never_queried(self):
        get = mock.Mock()
        logs = []
        verifier = DashboardVisibilityVerifier(
            "http://127.0.0.1:8000",
            "reviewer-token",
            get=get,
            logger=logs.append,
        )

        self.assertFalse(verifier.verify(schema_v2_payload()))
        get.assert_not_called()
        self.assertIn("payload_not_labeled_test", logs[0])


class DashboardTestRunnerTests(unittest.TestCase):
    def test_runner_accepts_exactly_approved_production_destination(self):
        args = dashboard_test_mode.parse_args([
            "start",
            "--backend-url",
            "https://echosense-backend-75h3.onrender.com",
            "--approved-test-origin",
            "https://echosense-backend-75h3.onrender.com",
            "--credentials-file",
            "device.env",
            "--reviewer-env-file",
            "reviewer.env",
            "--i-approve-live-capture",
            "--i-approve-dashboard-test-delivery",
        ])
        self.assertEqual(
            args.backend_origin,
            "https://echosense-backend-75h3.onrender.com",
        )

    def test_nonlocal_backend_requires_exact_origin_approval(self):
        with self.assertRaises(ValueError):
            dashboard_test_mode.validate_test_destination(
                "https://test.example",
            )
        self.assertEqual(
            dashboard_test_mode.validate_test_destination(
                "https://test.example/path",
                "https://test.example",
            ),
            "https://test.example",
        )

    def test_runner_uses_dedicated_test_outbox_and_explicit_mode(self):
        args = argparse.Namespace(
            backend_origin="http://127.0.0.1:8000",
            outbox="data/dashboard_test_outbox.sqlite3",
            microphone="Test EMEET",
            whisper_model="models/test-model",
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            path = dashboard_test_mode.configure_environment(
                args,
                device_id="dashboard-test-device",
                device_key="private-key",
                reviewer_token="memory-token",
            )
            self.assertTrue(str(path).endswith("dashboard_test_outbox.sqlite3"))
            self.assertEqual(os.environ["ECHOSENSE_DASHBOARD_TEST_MODE"], "true")
            self.assertEqual(
                os.environ["ECHOSENSE_DASHBOARD_TEST_EXPECTED_DEVICE_ID"],
                "dashboard-test-device",
            )
            self.assertEqual(
                os.environ["ECHOSENSE_DASHBOARD_TEST_REVIEWER_TOKEN"],
                "memory-token",
            )

    def test_private_env_reader_supports_export_and_quotes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "private.env"
            path.write_text(
                "export ECHOSENSE_DEVICE_ID='test-device'\n"
                'ECHOSENSE_DEVICE_KEY="private-key"\n',
                encoding="utf-8",
            )
            self.assertEqual(
                dashboard_test_mode._load_device_credentials(path),
                ("test-device", "private-key"),
            )


if __name__ == "__main__":
    unittest.main()
