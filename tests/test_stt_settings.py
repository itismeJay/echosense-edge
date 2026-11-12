"""Whisper decode settings and the model version the edge reports."""

import unittest
from unittest import mock

import numpy as np

from model.language_restricted_engine import (
    apply_gain,
    decode_restricted,
    forced_language,
)
from model.stt_version import MAX_LENGTH, model_label, stt_model_version
from sender import http_client


class FakeModel:
    def __init__(self):
        self.calls = []

    def detect_language(self, audio):
        return "jw", 0.5, [("jw", 0.5), ("tl", 0.3), ("en", 0.2)]

    def transcribe(self, audio, **kwargs):
        self.calls.append(dict(kwargs, peak=float(np.max(np.abs(audio)))))
        return iter(()), None


class DecodeSettingsTests(unittest.TestCase):
    def test_language_modes(self):
        self.assertIsNone(forced_language("auto"))
        self.assertEqual(forced_language("tl"), "tl")
        self.assertEqual(forced_language("tl+en"), "tl")
        self.assertEqual(forced_language("en"), "en")

    def test_auto_detection_is_restricted_to_english_and_tagalog(self):
        model = FakeModel()
        _text, language, probability, _segments = decode_restricted(
            model, np.zeros(16000, np.float32),
        )
        self.assertEqual(language, "tl")
        self.assertAlmostEqual(probability, 0.6)

    def test_forced_tagalog_skips_detection_and_passes_the_neutral_prompt(self):
        model = FakeModel()
        decode_restricted(
            model, np.zeros(16000, np.float32),
            language_mode="tl", initial_prompt="Estorya sa klasrom.",
        )
        call = model.calls[0]
        self.assertEqual(call["language"], "tl")
        self.assertEqual(call["initial_prompt"], "Estorya sa klasrom.")
        self.assertFalse(call["condition_on_previous_text"])

    def test_no_prompt_by_default(self):
        model = FakeModel()
        decode_restricted(model, np.zeros(16000, np.float32))
        self.assertIsNone(model.calls[0]["initial_prompt"])

    def test_gain_modes(self):
        quiet = (np.sin(np.linspace(0, 200, 16000)) * 0.01).astype(np.float32)
        np.testing.assert_array_equal(apply_gain(quiet, "off"), quiet)
        self.assertAlmostEqual(float(np.max(np.abs(apply_gain(quiet, "peak")))), 0.95, places=3)
        boosted = apply_gain(quiet, "rms")
        self.assertGreater(float(np.max(np.abs(boosted))), 0.1)
        self.assertLessEqual(float(np.max(np.abs(boosted))), 1.0)
        loud = (np.sin(np.linspace(0, 200, 16000)) * 0.8).astype(np.float32)
        np.testing.assert_array_equal(apply_gain(loud, "rms"), loud)
        silence = np.zeros(16000, np.float32)
        np.testing.assert_array_equal(apply_gain(silence, "rms"), silence)


class ModelVersionTests(unittest.TestCase):
    def test_size_name_and_local_model_labels(self):
        self.assertEqual(model_label("small"), "whisper small")
        self.assertEqual(
            model_label("/srv/models/sapinsapin_whisper-small-pld-ceb-ct2-int8"),
            "whisper-small-pld-ceb",
        )

    def test_version_fits_the_backend_column(self):
        version = stt_model_version(
            "/srv/models/eemberda_phcodeswitch-ceb-dvo-ct2-int8",
            "tl+en", "a prompt", "rms",
        )
        self.assertLessEqual(len(version), MAX_LENGTH)
        self.assertTrue(version.startswith("phcodeswitch-ceb-dvo int8 tl+en"))

    def test_heartbeat_reports_the_real_models(self):
        captured = {}

        def fake_post(url, json=None, headers=None, timeout=None):
            captured["json"] = json
            raise SystemExit  # stop the loop after one heartbeat

        with mock.patch.object(http_client.time, "sleep"), \
                mock.patch.object(http_client.requests, "post", side_effect=fake_post):
            with self.assertRaises(SystemExit):
                http_client._heartbeat_loop(60, lambda: {"ip": "10.0.0.2"})
        body = captured["json"]
        self.assertEqual(body["ip"], "10.0.0.2")
        self.assertEqual(body["stt_model_version"], stt_model_version())
        self.assertIn("YAMNet", body["yamnet_version"])


if __name__ == "__main__":
    unittest.main()
