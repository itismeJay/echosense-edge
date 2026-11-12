import inspect
import math
from pathlib import Path
import unittest

import config
from config import (
    DetectionRuntimeConfig,
    detection_runtime_config_from_environment,
)
import main
import runtime


class AfterConfigurationTests(unittest.TestCase):
    def test_timing_defaults_match_development_starting_values(self):
        values = DetectionRuntimeConfig()
        self.assertEqual(values.pre_trigger_seconds, 2.0)
        self.assertEqual(values.post_trigger_seconds, 1.5)
        self.assertEqual(values.event_extension_seconds, 1.5)
        self.assertEqual(values.max_event_seconds, 6.0)
        self.assertEqual(values.duplicate_window_seconds, 10.0)
        self.assertEqual(values.tone_min_absolute_rms, 400.0)
        self.assertFalse(hasattr(values, "cooldown_seconds"))
        self.assertEqual(values.max_stt_utterance_seconds, 15.0)
        self.assertEqual(values.max_stt_utterance_samples, 240_000)
        self.assertEqual(values.max_stt_result_queue_pcm_bytes, 30_720_000)

    def test_environment_overrides_are_validated(self):
        values = detection_runtime_config_from_environment({
            "ECHOSENSE_PRE_TRIGGER_SECONDS": "1.5",
            "ECHOSENSE_POST_TRIGGER_SECONDS": "4",
            "ECHOSENSE_EVENT_EXTENSION_SECONDS": "2",
            "ECHOSENSE_MAX_EVENT_SECONDS": "9",
            "ECHOSENSE_DUPLICATE_WINDOW_SECONDS": "12",
            "ECHOSENSE_MAX_STT_UTTERANCE_SECONDS": "10",
        })
        self.assertEqual(values.pre_trigger_seconds, 1.5)
        self.assertEqual(values.max_event_seconds, 9.0)
        self.assertEqual(values.duplicate_window_seconds, 12.0)
        self.assertEqual(values.max_stt_utterance_seconds, 10.0)
        with self.assertRaises(ValueError):
            detection_runtime_config_from_environment({
                "ECHOSENSE_PRE_TRIGGER_SECONDS": "-1",
            })
        with self.assertRaises(ValueError):
            detection_runtime_config_from_environment({
                "ECHOSENSE_MAX_EVENT_SECONDS": "0",
            })

    def test_stt_retention_rejects_non_finite_and_unsafe_values(self):
        for value in ("0", "-1", "nan", "inf", "-inf", "3600"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    detection_runtime_config_from_environment({
                        "ECHOSENSE_MAX_STT_UTTERANCE_SECONDS": value,
                    })

    def test_after_resource_values_reject_non_finite_or_extreme_bounds(self):
        for name in (
            "pre_trigger_seconds",
            "post_trigger_seconds",
            "event_extension_seconds",
            "max_event_seconds",
            "duplicate_window_seconds",
        ):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    DetectionRuntimeConfig(**{name: math.inf})
        with self.assertRaises(ValueError):
            DetectionRuntimeConfig(max_event_seconds=61)
        with self.assertRaises(ValueError):
            DetectionRuntimeConfig(stt_result_queue_size=65)
        with self.assertRaises(ValueError):
            DetectionRuntimeConfig(frame_samples=15_601)
        with self.assertRaises(ValueError):
            DetectionRuntimeConfig(yamnet_window_samples=1_000_000)
        for name in (
            "ECHOSENSE_AUDIO_QUEUE_FRAMES",
            "ECHOSENSE_STT_RESULT_QUEUE_SIZE",
        ):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    detection_runtime_config_from_environment({name: "nan"})

    def test_audio_and_evidence_queue_bounds_are_positive(self):
        with self.assertRaises(ValueError):
            DetectionRuntimeConfig(audio_queue_frames=0)
        with self.assertRaises(ValueError):
            DetectionRuntimeConfig(stt_result_queue_size=0)
        with self.assertRaises(ValueError):
            DetectionRuntimeConfig(evidence_history_size=0)

    def test_yamnet_paths_are_configurable_and_repository_relative(self):
        root = Path(config.__file__).resolve().parent
        self.assertEqual(Path(config.YAMNET_MODEL_PATH).parent, root)
        self.assertEqual(Path(config.YAMNET_CLASSES_PATH).parent, root)
        self.assertNotIn("/home/echosense/", config.YAMNET_MODEL_PATH)

    def test_fatal_runtime_does_not_unconditionally_exit_success(self):
        source = inspect.getsource(main.main)
        self.assertNotIn("sys.exit(0)", source)
        self.assertIn("exit_code = 1", source)
        self.assertIn("return exit_code", source)

    def test_production_runtime_has_one_authoritative_after_path(self):
        source = inspect.getsource(runtime.EdgeRuntime.start)
        main_source = inspect.getsource(main.main)
        self.assertEqual(source.count("PyAudioSharedSource("), 1)
        self.assertNotIn("AggressionDetector", source)
        self.assertNotIn("capture_audio", source)
        self.assertNotIn("process_transcription_result", main_source)


if __name__ == "__main__":
    unittest.main()
