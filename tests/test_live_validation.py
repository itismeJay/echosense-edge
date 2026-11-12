import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from tools import live_validation


class LiveValidationTests(unittest.TestCase):
    def test_direct_script_adds_repository_root_without_pythonpath(self):
        script = Path(live_validation.__file__).resolve()
        code = (
            "import runpy; "
            f"runpy.run_path({str(script)!r}); "
            "import detection.transcript_quality"
        )
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory() as working_directory:
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=working_directory,
                env=environment,
                capture_output=True,
                text=True,
                timeout=10,
            )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_live_capture_requires_explicit_approval(self):
        with self.assertRaises(SystemExit):
            live_validation.parse_args([])

    def test_transcript_display_requires_adult_consent(self):
        with self.assertRaises(SystemExit):
            live_validation.parse_args([
                "--i-approve-live-capture",
                "--show-transcripts",
            ])

    def test_isolated_environment_overrides_external_integrations(self):
        args = argparse.Namespace(
            microphone="Test EMEET",
            whisper_model="models/test-model",
            show_transcripts=True,
        )
        with mock.patch.dict(os.environ, {
            "ECHOSENSE_BACKEND_BASE_URL": "https://production.invalid",
            "ECHOSENSE_OUTBOX_ENABLED": "true",
            "ECHOSENSE_DASHBOARD_TEST_MODE": "true",
            "ECHOSENSE_DASHBOARD_TEST_REVIEWER_TOKEN": "secret-token",
        }, clear=False):
            live_validation.configure_isolated_environment(args)
            self.assertEqual(
                os.environ["ECHOSENSE_BACKEND_BASE_URL"],
                "http://127.0.0.1:9",
            )
            self.assertEqual(os.environ["ECHOSENSE_OUTBOX_ENABLED"], "false")
            self.assertEqual(
                os.environ["ECHOSENSE_DASHBOARD_TEST_MODE"],
                "false",
            )
            self.assertEqual(
                os.environ["ECHOSENSE_DASHBOARD_TEST_REVIEWER_TOKEN"],
                "",
            )
            self.assertEqual(os.environ["ECHOSENSE_ENV_FILE"], "")
            self.assertEqual(
                os.environ["ECHOSENSE_PREFERRED_MICROPHONE"],
                "Test EMEET",
            )

    def test_realtimestt_file_logging_is_forced_off(self):
        import RealtimeSTT

        calls = []

        def fake_recorder(*args, **kwargs):
            calls.append((args, kwargs))
            return object()

        with mock.patch.object(RealtimeSTT, "AudioToTextRecorder", fake_recorder):
            live_validation.disable_realtimestt_log_file()
            RealtimeSTT.AudioToTextRecorder(no_log_file=False)
        self.assertTrue(calls[0][1]["no_log_file"])


if __name__ == "__main__":
    unittest.main()
