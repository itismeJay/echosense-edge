"""Integration coverage for EchoSense's real spawned transcription worker."""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path
import unittest

import numpy as np

from model.language_restricted_engine import RestrictedWhisperExecutor


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "sapinsapin_whisper-small-pld-ceb-ct2-int8"
)


@unittest.skipUnless(MODEL_PATH.is_dir(), "local Whisper integration model is absent")
class RealtimeSTTWorkerStartupTests(unittest.TestCase):
    def test_whisper_wrapper_initializes_realtimestt_without_a_microphone(self):
        from model import whisper_stt

        self.addCleanup(whisper_stt.stop_recorder)
        recorder = whisper_stt.get_recorder()
        self.assertIsNotNone(recorder)
        self.assertFalse(recorder.use_microphone.value)
        self.assertIsNone(recorder.transcript_process)
        self.assertTrue(whisper_stt._transcription_executor.worker_alive)

    def test_supported_executor_starts_real_spawned_worker(self):
        from RealtimeSTT.core.transcription import call_transcription_executor

        executor = RestrictedWhisperExecutor(
            model=str(MODEL_PATH),
            device="cpu",
            compute_type="int8",
            beam_size=1,
            vad_filter=False,
        )
        try:
            self.assertTrue(executor.worker_alive)
            self.assertNotEqual(executor.worker_pid, os.getpid())
            result = call_transcription_executor(
                executor,
                np.zeros(1600, dtype=np.float32),
                "tl",
                False,
            )
            self.assertEqual(result.info.language, "tl")
            self.assertEqual(result.info.language_probability, 1.0)
            self.assertIsInstance(result.metadata, dict)
            self.assertIn("no_speech_prob", result.metadata)
            self.assertIn("words", result.metadata)
        finally:
            executor.close()
        self.assertFalse(executor.worker_alive)

    def test_failed_worker_startup_leaves_no_child_process(self):
        before = {process.pid for process in multiprocessing.active_children()}
        with self.assertRaisesRegex(RuntimeError, "compute type"):
            RestrictedWhisperExecutor(
                model=str(MODEL_PATH),
                device="cpu",
                compute_type="not-a-real-compute-type",
                startup_timeout=15,
            )
        after = {
            process.pid
            for process in multiprocessing.active_children()
            if process.name == "echosense-restricted-whisper"
        }
        self.assertTrue(after.issubset(before))


if __name__ == "__main__":
    unittest.main()
