import tempfile
import unittest
from pathlib import Path

from config import detect_platform, load_edge_env, parse_env_file


class ParseEnvFileTests(unittest.TestCase):
    def test_parses_export_quotes_and_comments(self):
        values = parse_env_file(
            "# comment\n"
            "export ECHOSENSE_DEVICE_ID=room-1\n"
            "ECHOSENSE_PREFERRED_MICROPHONE=\"MacBook Pro\"\n"
            "ECHOSENSE_LOG_DIR='logs dir'\n"
            "\n"
            "not a setting\n"
            "ECHOSENSE_EMPTY=\n"
        )
        self.assertEqual(values, {
            "ECHOSENSE_DEVICE_ID": "room-1",
            "ECHOSENSE_PREFERRED_MICROPHONE": "MacBook Pro",
            "ECHOSENSE_LOG_DIR": "logs dir",
            "ECHOSENSE_EMPTY": "",
        })


class LoadEdgeEnvTests(unittest.TestCase):
    def test_exported_variables_win_over_file(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / "edge.env"
            env_file.write_text("A=file\nB=file\n", encoding="utf-8")
            environ = {"A": "shell"}
            load_edge_env(env_file, environ)
        self.assertEqual(environ, {"A": "shell", "B": "file"})

    def test_missing_file_and_empty_path_are_ignored(self):
        environ = {}
        self.assertEqual(load_edge_env("/nonexistent/edge.env", environ), {})
        self.assertEqual(load_edge_env("", environ), {})
        self.assertEqual(environ, {})

    def test_env_file_variable_selects_file(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / "pi.env"
            env_file.write_text("C=pi\n", encoding="utf-8")
            environ = {"ECHOSENSE_ENV_FILE": str(env_file)}
            load_edge_env(environ=environ)
        self.assertEqual(environ["C"], "pi")


class DetectPlatformTests(unittest.TestCase):
    def test_override_wins(self):
        self.assertEqual(
            detect_platform({"ECHOSENSE_PLATFORM": "RaspberryPi"}),
            "raspberrypi",
        )

    def test_detects_a_known_platform(self):
        self.assertIn(detect_platform({}), {"macos", "raspberrypi", "linux"})


if __name__ == "__main__":
    unittest.main()
