import contextlib
import io
import unittest

import numpy as np
import requests

from audio.audio_event import AudioEvent
from model.stt_provider import (
    ElevenLabsClient,
    HybridTranscriptionExecutor,
    ProviderError,
    RetryQueue,
    RetryWorker,
    STTSettings,
    load_keyterms,
    split_audio_tags,
)

AUDIO = np.zeros(1600, dtype=np.float32)


class FakeWhisper:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, language=None, use_prompt=True):
        from RealtimeSTT.transcription_engines.base import (
            TranscriptionInfo,
            TranscriptionResult,
        )

        self.calls.append(language)
        result = TranscriptionResult(
            text="whisper text",
            info=TranscriptionInfo(language="tl", language_probability=0.9),
        )
        result.metadata = {"avg_logprob": -0.2}
        return result


class FakeClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def transcribe(self, audio, sample_rate=16000):
        self.calls += 1
        outcome = self.outcomes.pop(0) if self.outcomes else ("ok", "ceb")
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def hybrid(outcomes, clock=None):
    clock = clock or Clock()
    settings = STTSettings(provider="elevenlabs", api_key="k")
    whisper = FakeWhisper()
    client = FakeClient(outcomes)
    executor = HybridTranscriptionExecutor(whisper, settings, client=client, clock=clock)
    return executor, whisper, client, clock


def event(seconds=1.0):
    return AudioEvent.from_float32_samples(np.zeros(int(16000 * seconds), dtype=np.float32))


def worker(outcomes):
    executor, whisper, client, clock = hybrid(outcomes)
    delivered = []
    retry = RetryWorker(executor, lambda *args: delivered.append(args))
    return executor, retry, whisper, client, clock, delivered


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        return fn(*args, **kwargs), out.getvalue()


class ProviderChoiceTests(unittest.TestCase):
    def test_defaults_to_elevenlabs_when_key_set(self):
        s, _ = quiet(STTSettings.from_environment, {"ELEVENLABS_API_KEY": "k"})
        self.assertEqual(s.provider, "elevenlabs")
        self.assertEqual((s.model, s.language, s.timeout_seconds), ("scribe_v3", "ceb", 15.0))
        self.assertEqual((s.outage_minutes, s.retry_seconds, s.probe_seconds), (30.0, 30.0, 60.0))
        self.assertEqual((s.queue_max_clips, s.queue_max_seconds), (200, 1800.0))

    def test_timeout_and_outage_from_environment(self):
        s, _ = quiet(STTSettings.from_environment, {
            "ELEVENLABS_API_KEY": "k", "ECHOSENSE_ELEVENLABS_TIMEOUT_S": "12",
            "ECHOSENSE_ELEVENLABS_OUTAGE_MINUTES": "5"})
        self.assertEqual((s.timeout_seconds, s.outage_minutes), (12.0, 5.0))

    def test_defaults_to_whisper_without_key(self):
        s, _ = quiet(STTSettings.from_environment, {})
        self.assertEqual(s.provider, "whisper")

    def test_explicit_whisper_wins_over_key(self):
        s, _ = quiet(STTSettings.from_environment,
                     {"ELEVENLABS_API_KEY": "k", "ECHOSENSE_STT_PROVIDER": "whisper"})
        self.assertEqual(s.provider, "whisper")

    def test_elevenlabs_without_key_falls_back_to_whisper(self):
        s, _ = quiet(STTSettings.from_environment, {"ECHOSENSE_STT_PROVIDER": "elevenlabs"})
        self.assertEqual(s.provider, "whisper")

    def test_whisper_provider_never_calls_elevenlabs(self):
        whisper, client = FakeWhisper(), FakeClient([])
        executor = HybridTranscriptionExecutor(
            whisper, STTSettings(provider="whisper"), client=client,
        )
        result, log = quiet(executor.transcribe, AUDIO)
        self.assertEqual(client.calls, 0)
        self.assertEqual(result.metadata["stt"]["provider"], "whisper")
        self.assertIn("[STT] provider=whisper fallback_reason=none", log)

    def test_keyterms_file_has_200_single_words(self):
        terms = load_keyterms()
        self.assertEqual(len(terms), 200)
        self.assertTrue(all(len(t.split()) == 1 for t in terms))
        self.assertIn("baho", terms)


class ElevenLabsResultTests(unittest.TestCase):
    def test_success_uses_elevenlabs_and_keeps_tags_as_context(self):
        executor, whisper, _, _ = hybrid([("Pangit ka (laughs) bai", "ceb")])
        result, log = quiet(executor.transcribe, AUDIO)
        self.assertEqual(result.text, "Pangit ka bai")
        self.assertEqual(result.info.language, "ceb")
        self.assertEqual(whisper.calls, [])
        stt = result.metadata["stt"]
        self.assertEqual((stt["provider"], stt["fallback_reason"]), ("elevenlabs", None))
        self.assertEqual(stt["audio_tags"], ["laughs"])
        self.assertIn("[STT] provider=elevenlabs fallback_reason=none latency_ms=", log)

    def test_split_audio_tags(self):
        self.assertEqual(split_audio_tags("[shouting] Hoy! (laughs)"), ("Hoy!", ["shouting", "laughs"]))
        self.assertEqual(split_audio_tags("baho ka"), ("baho ka", []))

    def test_language_retry_goes_to_whisper(self):
        executor, whisper, client, _ = hybrid([])
        result, _ = quiet(executor.transcribe, AUDIO, language="en")
        self.assertEqual(client.calls, 0)
        self.assertEqual(whisper.calls, ["en"])
        self.assertEqual(result.metadata["stt"]["fallback_reason"], "language_retry")


class FailureQueuesTests(unittest.TestCase):
    def test_failure_queues_without_whisper(self):
        for reason in ("timeout", "network:ConnectionError", "http_500", "quota:http_429"):
            executor, whisper, _, _ = hybrid([ProviderError(reason)])
            result, log = quiet(executor.transcribe, AUDIO)
            self.assertEqual(result.text, "")
            self.assertEqual(whisper.calls, [])
            stt = result.metadata["stt"]
            self.assertTrue(stt["queued"])
            self.assertEqual(stt["fallback_reason"], reason)
            self.assertIn(f"[STT] provider=elevenlabs queued=true reason={reason}", log)
            self.assertNotIn("xi-api-key", log)

    def test_unexpected_client_error_is_queued(self):
        executor, _, _, _ = hybrid([KeyError("x")])
        result, _ = quiet(executor.transcribe, AUDIO)
        self.assertEqual(result.metadata["stt"]["fallback_reason"], "error:KeyError")
        self.assertTrue(result.metadata["stt"]["queued"])

    def test_while_failing_new_utterances_queue_in_order_without_a_call(self):
        executor, _, client, _ = hybrid([ProviderError("timeout")])
        quiet(executor.transcribe, AUDIO)
        result, _ = quiet(executor.transcribe, AUDIO)
        self.assertEqual(client.calls, 1)
        self.assertEqual(result.metadata["stt"]["fallback_reason"], "awaiting_retry")


class FakeResponse:
    def __init__(self, status, body=None, text=""):
        self.status_code, self._body, self.text = status, body, text

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


class FakeSession:
    def __init__(self, outcome):
        self.outcome, self.kwargs = outcome, None

    def post(self, url, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class ClientTests(unittest.TestCase):
    def client(self, outcome):
        session = FakeSession(outcome)
        settings = STTSettings(provider="elevenlabs", api_key="k", timeout_seconds=4.0)
        return ElevenLabsClient(settings, ("baho", "pangit"), session=session), session

    def reason(self, outcome):
        client, _ = self.client(outcome)
        with self.assertRaises(ProviderError) as ctx:
            client.transcribe(AUDIO)
        return ctx.exception.reason

    def test_request_uses_setting_d(self):
        client, session = self.client(FakeResponse(200, {"text": "baho ka", "language_code": "ceb"}))
        self.assertEqual(client.transcribe(AUDIO), ("baho ka", "ceb"))
        data = session.kwargs["data"]
        self.assertIn(("model_id", "scribe_v3"), data)
        self.assertIn(("language_code", "ceb"), data)
        self.assertEqual([v for k, v in data if k == "keyterms"], ["baho", "pangit"])
        self.assertEqual(session.kwargs["timeout"], 4.0)

    def test_error_reasons(self):
        self.assertEqual(self.reason(requests.Timeout()), "timeout")
        self.assertEqual(self.reason(requests.ConnectionError()), "network:ConnectionError")
        self.assertEqual(self.reason(FakeResponse(503)), "http_503")
        self.assertEqual(self.reason(FakeResponse(401, text="invalid key")), "http_401")
        self.assertEqual(self.reason(FakeResponse(401, text="quota_exceeded")), "quota:http_401")
        self.assertEqual(self.reason(FakeResponse(429)), "quota:http_429")
        self.assertEqual(self.reason(FakeResponse(200, None)), "bad_response")


class RetryQueueTests(unittest.TestCase):
    def test_queue_retry_success_delivers_in_order_with_original_event(self):
        executor, retry, whisper, client, clock, delivered = worker([
            ProviderError("timeout"), ("una", "ceb"), ("ikaduha", "ceb"),
        ])
        first, second = event(), event()
        for audio_event in (first, second):
            result, _ = quiet(executor.transcribe, audio_event.samples)
            quiet(retry.enqueue, audio_event, result.metadata["stt"]["fallback_reason"])
        self.assertEqual(len(retry.queue), 2)

        clock.now = 29.0
        quiet(retry.step)
        self.assertEqual(delivered, [])
        clock.now = 30.0
        _, log = quiet(retry.step)
        self.assertEqual([d[0] for d in delivered], ["una", "ikaduha"])
        self.assertIs(delivered[0][3], first)
        self.assertIs(delivered[1][3], second)
        self.assertEqual(delivered[0][4]["stt"]["provider"], "elevenlabs")
        self.assertEqual(len(retry.queue), 0)
        self.assertEqual(whisper.calls, [])
        self.assertTrue(executor.health.healthy)
        self.assertIn("[STT] retry event=", log)
        self.assertIn("ok=true provider=elevenlabs", log)

    def test_failed_retry_keeps_clip_and_tries_again_in_30s(self):
        executor, retry, _, client, clock, delivered = worker([
            ProviderError("timeout"), ProviderError("http_503"), ("ok", "ceb"),
        ])
        quiet(executor.transcribe, AUDIO)
        quiet(retry.enqueue, event(), "timeout")
        clock.now = 30.0
        _, log = quiet(retry.step)
        self.assertIn("ok=false reason=http_503", log)
        self.assertEqual(len(retry.queue), 1)
        clock.now = 59.0
        quiet(retry.step)
        self.assertEqual(client.calls, 2)
        clock.now = 60.0
        quiet(retry.step)
        self.assertEqual([d[0] for d in delivered], ["ok"])

    def test_queue_is_bounded_by_clips_and_seconds(self):
        queue = RetryQueue(max_clips=3, max_seconds=1800.0)
        executor, _, _, _ = hybrid([])
        retry = RetryWorker(executor, lambda *a: None, queue=queue)
        events = [event() for _ in range(5)]
        logs = "".join(quiet(retry.enqueue, e, "timeout")[1] for e in events)
        self.assertEqual([c.audio_event for c in queue.snapshot()], events[2:])
        self.assertIn(f"queue_full dropped event={events[0].event_id}", logs)

        by_time = RetryQueue(max_clips=200, max_seconds=5.0)
        retry = RetryWorker(executor, lambda *a: None, queue=by_time)
        for _ in range(4):
            quiet(retry.enqueue, event(2.0), "timeout")
        self.assertEqual(len(by_time), 2)
        self.assertLessEqual(by_time.seconds, 5.0)

    def test_clip_rejected_three_times_goes_to_whisper(self):
        executor, retry, whisper, _, clock, delivered = worker([ProviderError("http_400")] * 4)
        quiet(executor.transcribe, AUDIO)
        self.assertTrue(executor.health.healthy)
        quiet(retry.enqueue, event(), "http_400")
        for now in (30.0, 60.0, 90.0):
            clock.now = now
            quiet(retry.step)
        self.assertEqual([d[0] for d in delivered], ["whisper text"])
        self.assertEqual(delivered[0][4]["stt"]["fallback_reason"], "clip_rejected")
        self.assertEqual(len(retry.queue), 0)


class OutageTests(unittest.TestCase):
    def fail_until(self, executor, retry, clock, minutes):
        quiet(executor.transcribe, AUDIO)
        quiet(retry.enqueue, event(), "timeout")
        logs = ""
        while clock.now < minutes * 60:
            clock.now += 30.0
            logs += quiet(retry.step)[1]
        return logs

    def test_outage_after_30_minutes_switches_to_whisper_and_drains_queue(self):
        executor, retry, whisper, client, clock, delivered = worker(
            [ProviderError("timeout")] * 200)
        logs = self.fail_until(executor, retry, clock, 29.5)
        self.assertFalse(executor.health.outage)
        self.assertEqual(whisper.calls, [])
        self.assertNotIn("outage_mode=on", logs)

        clock.now = 30 * 60.0
        _, log = quiet(retry.step)
        self.assertTrue(executor.health.outage)
        self.assertIn("[STT] outage_mode=on provider=whisper", log)
        self.assertEqual([d[0] for d in delivered], ["whisper text"])
        self.assertEqual(delivered[0][4]["stt"]["fallback_reason"], "outage_mode")
        self.assertEqual(len(retry.queue), 0)

        calls = client.calls
        result, _ = quiet(executor.transcribe, AUDIO)
        self.assertEqual(client.calls, calls)
        self.assertEqual(result.text, "whisper text")
        self.assertEqual(result.metadata["stt"]["fallback_reason"], "outage_mode")

    def test_probe_every_60s_and_recovery_switches_back(self):
        executor, retry, whisper, client, clock, _ = worker(
            [ProviderError("timeout")] * 62 + [("balik", "ceb"), ("bag-o", "ceb")])
        self.fail_until(executor, retry, clock, 30)
        self.assertTrue(executor.health.outage)
        calls = client.calls

        clock.now += 1.0
        quiet(retry.step)                  # schedules the first probe
        clock.now += 59.0
        quiet(retry.step)
        self.assertEqual(client.calls, calls)
        clock.now += 1.0
        _, log = quiet(retry.step)          # probe fails, outage stays on
        self.assertEqual(client.calls, calls + 1)
        self.assertIn("[STT] probe ok=false reason=timeout", log)
        self.assertTrue(executor.health.outage)

        clock.now += 60.0
        _, log = quiet(retry.step)          # probe succeeds
        self.assertIn("[STT] outage_mode=off provider=elevenlabs", log)
        self.assertIn("[STT] probe ok=true", log)
        self.assertFalse(executor.health.outage)
        self.assertTrue(executor.health.healthy)

        result, _ = quiet(executor.transcribe, AUDIO)
        self.assertEqual(result.text, "bag-o")
        self.assertEqual(result.metadata["stt"]["provider"], "elevenlabs")


class WhisperSttIntegrationTests(unittest.TestCase):
    def test_elevenlabs_transcript_skips_word_probability_filter(self):
        from model import whisper_stt

        whisper_stt.clear_transcription_results()
        metadata = {"stt": {"provider": "elevenlabs", "fallback_reason": None,
                            "latency_ms": 900, "audio_tags": ["laughs"]}}
        quiet(whisper_stt._on_text, "Baho ang imong baba, bay", "ceb", 1.0, None, metadata)
        result = whisper_stt.consume_transcription_result(timeout=1.0)
        self.assertEqual(result["stt_provider"], "elevenlabs")
        self.assertEqual(result["audio_tags"], ["laughs"])
        self.assertNotIn("low_word_probability", result["filter_reasons"])
        self.assertTrue(result["detected_words"])

    def test_whisper_low_probability_still_filtered(self):
        from model import whisper_stt

        whisper_stt.clear_transcription_results()
        metadata = {"no_speech_prob": 0.1, "avg_logprob": -0.3, "compression_ratio": 1.2,
                    "words": [("baho", 0.01), ("ang", 0.9), ("imong", 0.9), ("baba", 0.01)],
                    "segment_count": 1}
        quiet(whisper_stt._on_text, "baho ang imong baba", "tl", 0.9, None, metadata)
        result = whisper_stt.consume_transcription_result(timeout=1.0)
        self.assertEqual(result["stt_provider"], "whisper")
        self.assertIn("low_word_probability", result["filter_reasons"])


if __name__ == "__main__":
    unittest.main()
