"""Speech-to-text provider: ElevenLabs Scribe always, Whisper only after a long outage.

``HybridTranscriptionExecutor`` wraps the restricted Whisper executor and keeps
its ``transcribe(audio, language=None, use_prompt=True)`` contract, so
RealtimeSTT uses it unchanged. Policy:

- Every utterance goes to ElevenLabs (scribe_v3, language ceb, keyterms from
  lexicon/stt_keyterms.txt) with a 15 s timeout.
- A failure does NOT fall back to Whisper. The result comes back empty and
  marked ``queued``; whisper_stt puts the utterance's AudioEvent in the bounded
  in-memory ``RetryQueue`` (200 clips / 30 min of audio, oldest dropped first).
  While ElevenLabs is failing, new utterances are queued without a call so the
  order is kept. ``RetryWorker`` retries the queue in order every 30 s; a clip
  leaves the queue (and memory) as soon as it is transcribed, and its late text
  goes through detection with the original AudioEvent (original speech time).
- A clip ElevenLabs itself rejects (http_400/413/422) does not count as an
  outage; after 3 rejections that one clip is transcribed by Whisper.
- After 30 min of continuous failure (``outage_minutes``) outage mode starts:
  new utterances and the queued clips go to Whisper, and ElevenLabs is probed
  every 60 s with the last utterance's audio. One success ends outage mode.

Every result carries ``metadata["stt"] = {provider, fallback_reason,
latency_ms, audio_tags, words}`` (plus ``queued``) and, when a ToneAnalyzer is
attached, ``metadata["tone"]`` (emotion2vec + YAMNet shout/laughter + speech
RMS, computed in parallel with the ElevenLabs request). Log lines start with ``[STT]``:
provider, queued, retry, outage_mode on/off. Audio and the key are never logged.
Audio tags such as "(laughs)" are removed from the text used for matching and
kept as context.
"""

import io
import os
import re
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ELEVENLABS_URL = "https://api.elevenlabs.io/v1/speech-to-text"
KEYTERMS_PATH = Path(__file__).resolve().parents[1] / "lexicon" / "stt_keyterms.txt"
PROVIDERS = ("elevenlabs", "whisper")
# Rejections of one clip (bad audio), not a service outage.
CLIP_ERRORS = frozenset({"http_400", "http_413", "http_422"})
CLIP_ERROR_ATTEMPTS = 3

_AUDIO_TAG = re.compile(r"[\(\[]([^\)\]]{1,40})[\)\]]")


def load_keyterms(path=KEYTERMS_PATH):
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return ()
    return tuple(
        line.strip() for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    )


def split_audio_tags(text):
    """Return (text without tags, [tags]); "Pangit ka (laughs)" -> ("Pangit ka", ["laughs"])."""

    text = str(text or "")
    tags = [tag.strip().lower() for tag in _AUDIO_TAG.findall(text)]
    clean = " ".join(_AUDIO_TAG.sub(" ", text).split())
    return clean, tags


def float32_to_wav(audio, sample_rate=16000):
    samples = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    pcm = (samples * 32767.0).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


class ProviderError(Exception):
    """A failed ElevenLabs request; ``reason`` goes to the log."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def _number(env, names, default):
    for name in names:
        value = str(env.get(name) or "").strip()
        if value:
            return float(value)
    return float(default)


@dataclass(frozen=True)
class STTSettings:
    provider: str = "whisper"
    api_key: str = ""
    model: str = "scribe_v3"
    language: str = "ceb"
    timeout_seconds: float = 15.0
    outage_minutes: float = 30.0
    retry_seconds: float = 30.0
    probe_seconds: float = 60.0
    queue_max_clips: int = 200
    queue_max_seconds: float = 1800.0

    @classmethod
    def from_environment(cls, environ=None):
        env = os.environ if environ is None else environ
        api_key = str(env.get("ELEVENLABS_API_KEY") or "").strip()
        requested = str(env.get("ECHOSENSE_STT_PROVIDER") or "").strip().lower()
        if requested not in PROVIDERS:
            if requested:
                print(f"[STT] unknown ECHOSENSE_STT_PROVIDER={requested}, using default")
            requested = "elevenlabs" if api_key else "whisper"
        if requested == "elevenlabs" and not api_key:
            print("[STT] provider=elevenlabs requested but ELEVENLABS_API_KEY is empty; whisper only")
            requested = "whisper"
        return cls(
            provider=requested,
            api_key=api_key,
            model=str(env.get("ECHOSENSE_ELEVENLABS_MODEL") or "scribe_v3").strip(),
            language=str(env.get("ECHOSENSE_ELEVENLABS_LANGUAGE") or "ceb").strip(),
            timeout_seconds=_number(
                env, ("ECHOSENSE_ELEVENLABS_TIMEOUT_S", "ECHOSENSE_ELEVENLABS_TIMEOUT_SECONDS"), 15.0,
            ),
            outage_minutes=_number(env, ("ECHOSENSE_ELEVENLABS_OUTAGE_MINUTES",), 30.0),
            retry_seconds=_number(env, ("ECHOSENSE_ELEVENLABS_RETRY_S",), 30.0),
            probe_seconds=_number(env, ("ECHOSENSE_ELEVENLABS_PROBE_S",), 60.0),
            queue_max_clips=int(_number(env, ("ECHOSENSE_STT_QUEUE_MAX_CLIPS",), 200)),
            queue_max_seconds=_number(env, ("ECHOSENSE_STT_QUEUE_MAX_SECONDS",), 1800.0),
        )


class ElevenLabsHealth:
    """Continuous-failure tracking; outage mode after ``outage_seconds``."""

    def __init__(self, outage_seconds=1800.0, clock=time.monotonic):
        self.outage_seconds = float(outage_seconds)
        self._clock = clock
        self._lock = threading.Lock()
        self.failing_since = None
        self.outage = False

    @property
    def healthy(self):
        return self.failing_since is None

    def record_success(self):
        with self._lock:
            if self.outage:
                print("[STT] outage_mode=off provider=elevenlabs")
            self.failing_since = None
            self.outage = False

    def record_failure(self, reason):
        """Count one service failure; True when this one started outage mode."""

        with self._lock:
            now = self._clock()
            if self.failing_since is None:
                self.failing_since = now
            failing = now - self.failing_since
            if not self.outage and failing >= self.outage_seconds:
                self.outage = True
                print(
                    f"[STT] outage_mode=on provider=whisper reason={reason} "
                    f"failing_minutes={failing / 60:.1f}"
                )
                return True
            return False


@dataclass
class QueuedClip:
    audio_event: object
    reason: str
    queued_at: float
    attempts: int = 0
    # Tone signals measured when the clip was first heard (emotion etc.).
    tone: object = None

    @property
    def seconds(self):
        samples = getattr(self.audio_event, "samples", ())
        rate = getattr(self.audio_event, "sample_rate", 16000) or 16000
        return len(samples) / float(rate)


@dataclass
class RetryQueue:
    """Ordered, bounded in-memory clips; the oldest are dropped when full."""

    max_clips: int = 200
    max_seconds: float = 1800.0
    _items: deque = field(default_factory=deque)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __len__(self):
        return len(self._items)

    @property
    def seconds(self):
        return sum(item.seconds for item in self._items)

    def put(self, clip):
        """Append ``clip``; return the clips dropped to stay within bounds."""

        dropped = []
        with self._lock:
            self._items.append(clip)
            while len(self._items) > 1 and (
                len(self._items) > self.max_clips
                or sum(item.seconds for item in self._items) > self.max_seconds
            ):
                dropped.append(self._items.popleft())
        return dropped

    def snapshot(self):
        with self._lock:
            return list(self._items)

    def remove(self, clip):
        with self._lock:
            try:
                self._items.remove(clip)
            except ValueError:
                pass


class ElevenLabsClient:
    def __init__(self, settings, keyterms=(), session=None):
        self.settings = settings
        self.keyterms = tuple(keyterms)
        self.last_words = []
        if session is None:
            import requests

            session = requests.Session()
        self._session = session

    def transcribe(self, audio, sample_rate=16000):
        """Return (text, language_code); raise ProviderError on any failure."""

        import requests

        data = [
            ("model_id", self.settings.model),
            ("tag_audio_events", "true"),
        ]
        if self.settings.language:
            data.append(("language_code", self.settings.language))
        data.extend(("keyterms", term) for term in self.keyterms)
        files = {"file": ("utterance.wav", float32_to_wav(audio, sample_rate), "audio/wav")}
        try:
            response = self._session.post(
                ELEVENLABS_URL,
                headers={"xi-api-key": self.settings.api_key},
                data=data,
                files=files,
                timeout=self.settings.timeout_seconds,
            )
        except requests.Timeout:
            raise ProviderError("timeout") from None
        except requests.RequestException as exc:
            raise ProviderError(f"network:{type(exc).__name__}") from None
        status = int(getattr(response, "status_code", 0) or 0)
        if status == 429 or status == 402:
            raise ProviderError(f"quota:http_{status}")
        if status >= 500:
            raise ProviderError(f"http_{status}")
        if status >= 400:
            body = str(getattr(response, "text", "") or "").lower()
            if "quota" in body or "credit" in body:
                raise ProviderError(f"quota:http_{status}")
            raise ProviderError(f"http_{status}")
        try:
            body = response.json()
            text = str(body.get("text") or "")
            language = str(body.get("language_code") or self.settings.language or "")
        except (ValueError, AttributeError, TypeError):
            raise ProviderError("bad_response") from None
        self.last_words = _word_items(body)
        return text, language

    def transcribe_detailed(self, audio, sample_rate=16000):
        """(text, language, word items) — items as in ``_word_items``."""

        text, language = self.transcribe(audio, sample_rate)
        return text, language, self.last_words


MAX_WORD_ITEMS = 300


def _word_items(body):
    """ElevenLabs ``words`` without spacing: [{text, type, logprob}].

    ``type`` is "word" or "audio_event" (e.g. "(laughs)"). Timing and speaker
    fields are dropped; nothing here is audio.
    """

    items = []
    words = body.get("words") if isinstance(body, dict) else None
    for item in words or ():
        if not isinstance(item, dict):
            continue
        kind = str(item.get("type") or "word")
        if kind == "spacing":
            continue
        logprob = item.get("logprob")
        try:
            logprob = None if logprob is None else round(float(logprob), 4)
        except (TypeError, ValueError):
            logprob = None
        items.append({"text": str(item.get("text") or "")[:80], "type": kind[:40],
                      "logprob": logprob})
        if len(items) >= MAX_WORD_ITEMS:
            break
    return items


# ElevenLabs may answer with ISO 639-3 codes; keep the edge's ceb/fil/tl/en.
_ELEVENLABS_LANGUAGES = {"ceb": "ceb", "fil": "fil", "tl": "tl", "tgl": "tl",
                         "en": "en", "eng": "en"}


def _elevenlabs_language(detected, default):
    code = str(detected or "").strip().lower()
    return _ELEVENLABS_LANGUAGES.get(code) or str(default or "ceb")


def _result(text, language, probability, metadata):
    from RealtimeSTT.transcription_engines.base import (
        TranscriptionInfo,
        TranscriptionResult,
    )

    result = TranscriptionResult(
        text=text,
        info=TranscriptionInfo(language=language, language_probability=probability),
    )
    result.metadata = metadata
    return result


def _event_id(clip):
    return getattr(clip.audio_event, "event_id", "unavailable")


class HybridTranscriptionExecutor:
    """ElevenLabs always; the wrapped Whisper executor only in outage mode."""

    def __init__(self, whisper_executor, settings, *, client=None, health=None,
                 clock=time.monotonic, sample_rate=16000, tone_analyzer=None,
                 event_id_provider=None):
        self.whisper = whisper_executor
        # model/emotion_recognizer.ToneAnalyzer: emotion2vec + YAMNet shout/
        # laughter + speech RMS, run in parallel with the STT request.
        self.tone_analyzer = tone_analyzer
        # Returns the AudioEvent id of the clip being transcribed, so the
        # [EMOTION] log line names its utterance.
        self.event_id_provider = event_id_provider
        self.settings = settings
        self.sample_rate = sample_rate
        self._clock = clock
        self.client = client
        if self.client is None and settings.provider == "elevenlabs":
            self.client = ElevenLabsClient(settings, load_keyterms())
        self.health = health or ElevenLabsHealth(settings.outage_minutes * 60.0, clock=clock)
        # The recorder thread and the retry worker share one Whisper worker.
        self._whisper_lock = threading.Lock()
        self.last_audio = None

    # RealtimeSTT and whisper_stt use these executor attributes.
    def __getattr__(self, name):
        if name == "whisper":
            raise AttributeError(name)
        return getattr(self.whisper, name)

    def close(self, *args, **kwargs):
        return self.whisper.close(*args, **kwargs)

    @property
    def uses_elevenlabs(self):
        return self.settings.provider == "elevenlabs" and self.client is not None

    def transcribe(self, audio, language=None, use_prompt=True):
        # An explicit language is a re-decode of the same clip: no new tone.
        future = None
        if self.tone_analyzer is not None and not language:
            future = self.tone_analyzer.start(
                audio, self.sample_rate, event_id=self._current_event_id(),
            )
        result = self._transcribe(audio, language, use_prompt)
        if future is not None:
            tone = self.tone_analyzer.collect(future)
            metadata = getattr(result, "metadata", None)
            metadata = dict(metadata) if isinstance(metadata, dict) else {}
            if tone is not None:
                metadata["tone"] = tone
            result.metadata = metadata
        return result

    def _current_event_id(self):
        try:
            return str(self.event_id_provider() or "unavailable")
        except Exception:
            return "unavailable"

    def _transcribe(self, audio, language=None, use_prompt=True):
        started = self._clock()
        if not self.uses_elevenlabs:
            return self._whisper(audio, language, use_prompt, started, None)
        # An explicit language is whisper_stt's retry of a Whisper decode.
        if language:
            return self._whisper(audio, language, use_prompt, started, "language_retry")
        self.last_audio = audio
        if self.health.outage:
            return self._whisper(audio, None, use_prompt, started, "outage_mode")
        if not self.health.healthy:
            # Earlier clips are waiting for a retry; keep the order.
            return self._queued("awaiting_retry", started)
        try:
            return self.elevenlabs(audio, started)
        except ProviderError as exc:
            if exc.reason not in CLIP_ERRORS:
                self.health.record_failure(exc.reason)
            return self._queued(exc.reason, started)

    def elevenlabs(self, audio, started=None):
        """One ElevenLabs call; any failure is raised as ProviderError."""

        started = self._clock() if started is None else started
        words = []
        try:
            detailed = getattr(self.client, "transcribe_detailed", None)
            if callable(detailed):
                raw_text, detected, words = detailed(audio, self.sample_rate)
            else:
                raw_text, detected = self.client.transcribe(audio, self.sample_rate)
        except ProviderError:
            raise
        except Exception as exc:  # a client bug must not lose the clip
            raise ProviderError(f"error:{type(exc).__name__}") from None
        self.health.record_success()
        text, tags = split_audio_tags(raw_text)
        latency_ms = int((self._clock() - started) * 1000)
        stt = {
            "provider": "elevenlabs",
            "fallback_reason": None,
            "latency_ms": latency_ms,
            "audio_tags": tags,
            "words": list(words or ()),
        }
        print(
            f"[STT] provider=elevenlabs fallback_reason=none latency_ms={latency_ms}"
            + (f" audio_tags={tags}" if tags else "")
        )
        return _result(text, _elevenlabs_language(detected, self.settings.language),
                       1.0, {"stt": stt})

    def _queued(self, reason, started):
        latency_ms = int((self._clock() - started) * 1000)
        stt = {
            "provider": "elevenlabs",
            "fallback_reason": reason,
            "latency_ms": latency_ms,
            "audio_tags": [],
            "queued": True,
        }
        print(f"[STT] provider=elevenlabs queued=true reason={reason} latency_ms={latency_ms}")
        return _result("", self.settings.language, 0.0, {"stt": stt})

    def _whisper(self, audio, language, use_prompt, started, reason):
        with self._whisper_lock:
            result = self.whisper.transcribe(audio, language=language, use_prompt=use_prompt)
        latency_ms = int((self._clock() - started) * 1000)
        metadata = getattr(result, "metadata", None)
        metadata = dict(metadata) if isinstance(metadata, dict) else {}
        metadata["stt"] = {
            "provider": "whisper",
            "fallback_reason": reason,
            "latency_ms": latency_ms,
            "audio_tags": [],
        }
        result.metadata = metadata
        print(
            f"[STT] provider=whisper fallback_reason={reason or 'none'} "
            f"latency_ms={latency_ms}"
        )
        return result


class RetryWorker:
    """Retries queued clips in order; drains them with Whisper in outage mode.

    ``deliver(text, language, probability, audio_event, metadata)`` receives
    each late transcript with the clip's original AudioEvent.
    """

    def __init__(self, executor, deliver, *, queue=None, clock=None):
        settings = executor.settings
        self.executor = executor
        self.deliver = deliver
        if queue is None:
            queue = RetryQueue(settings.queue_max_clips, settings.queue_max_seconds)
        self.queue = queue
        self.retry_seconds = float(settings.retry_seconds)
        self.probe_seconds = float(settings.probe_seconds)
        self._clock = clock or executor._clock
        self._next_retry = None
        self._next_probe = None
        self._stop = threading.Event()
        self._thread = None
        self._lifecycle = threading.Lock()
        self._closed = False

    def enqueue(self, audio_event, reason, tone=None):
        if audio_event is None:
            print(f"[STT] queued=false reason={reason} lost=no_audio_event")
            return False
        now = self._clock()
        dropped = self.queue.put(QueuedClip(audio_event, reason, now, tone=tone))
        if self._next_retry is None:
            self._next_retry = now + self.retry_seconds
        print(
            f"[STT] queued event={audio_event.event_id} reason={reason} "
            f"queue_clips={len(self.queue)} queue_seconds={self.queue.seconds:.1f}"
        )
        for clip in dropped:
            print(f"[STT] queue_full dropped event={_event_id(clip)} queue_clips={len(self.queue)}")
        return True

    def step(self):
        """One scheduling tick (the thread calls it every second)."""

        now = self._clock()
        health = self.executor.health
        if health.outage:
            if self._next_probe is None:
                self._next_probe = now + self.probe_seconds
            elif now >= self._next_probe:
                self._next_probe = now + self.probe_seconds
                self._probe()
            if health.outage:
                self._drain_with_whisper("outage_mode")
                return
        self._next_probe = None
        if not len(self.queue):
            self._next_retry = None
            return
        if self._next_retry is None:
            self._next_retry = now + self.retry_seconds
        if now < self._next_retry:
            return
        self._next_retry = now + self.retry_seconds
        self._retry_round(now)

    def _retry_round(self, now):
        health = self.executor.health
        for clip in self.queue.snapshot():
            if self._stop.is_set():
                return
            clip.attempts += 1
            try:
                result = self.executor.elevenlabs(clip.audio_event.samples)
            except ProviderError as exc:
                print(
                    f"[STT] retry event={_event_id(clip)} ok=false reason={exc.reason} "
                    f"attempt={clip.attempts} queue_clips={len(self.queue)}"
                )
                if exc.reason in CLIP_ERRORS:
                    if clip.attempts >= CLIP_ERROR_ATTEMPTS:
                        self._whisper_clip(clip, "clip_rejected")
                    continue
                health.record_failure(exc.reason)
                if health.outage:
                    self._drain_with_whisper("outage_mode")
                return
            self.queue.remove(clip)
            print(
                f"[STT] retry event={_event_id(clip)} ok=true provider=elevenlabs "
                f"late_s={now - clip.queued_at:.1f} queue_clips={len(self.queue)}"
            )
            self._deliver(clip, result)

    def _probe(self):
        audio = self.executor.last_audio
        if audio is None:
            audio = np.zeros(self.executor.sample_rate, dtype=np.float32)
        try:
            self.executor.elevenlabs(audio)
        except ProviderError as exc:
            print(f"[STT] probe ok=false reason={exc.reason}")
            if exc.reason not in CLIP_ERRORS:
                self.executor.health.record_failure(exc.reason)
            return
        print("[STT] probe ok=true")

    def _drain_with_whisper(self, reason):
        for clip in self.queue.snapshot():
            if self._stop.is_set():
                return
            self._whisper_clip(clip, reason)

    def _whisper_clip(self, clip, reason):
        result = self.executor._whisper(
            clip.audio_event.samples, None, False, self._clock(), reason,
        )
        self.queue.remove(clip)
        print(
            f"[STT] retry event={_event_id(clip)} ok=true provider=whisper "
            f"reason={reason} queue_clips={len(self.queue)}"
        )
        self._deliver(clip, result)

    def _deliver(self, clip, result):
        analyzer = getattr(self.executor, "tone_analyzer", None)
        if analyzer is not None and getattr(clip, "tone", None) is None:
            try:
                clip.tone = analyzer.analyze(clip.audio_event.samples, event_id=_event_id(clip))
            except Exception as exc:
                print(f"[EMOTION] late_clip_error={type(exc).__name__}")
        if getattr(clip, "tone", None) is not None:
            metadata = getattr(result, "metadata", None)
            metadata = dict(metadata) if isinstance(metadata, dict) else {}
            metadata["tone"] = clip.tone
            result.metadata = metadata
        try:
            self.deliver(
                result.text,
                result.info.language,
                result.info.language_probability,
                clip.audio_event,
                getattr(result, "metadata", None),
            )
        except Exception as exc:
            print(f"[STT] late_delivery_error={type(exc).__name__}")

    def start(self):
        # stop_recorder() may run on another thread while the recorder loop is
        # still starting this worker: start and stop are serialized, and a
        # worker that was stopped first never starts.
        with self._lifecycle:
            if self._thread is not None or self._closed:
                return
            self._stop.clear()
            thread = threading.Thread(target=self._run, name="stt-retry", daemon=True)
            thread.start()
            self._thread = thread

    def _run(self):
        while not self._stop.wait(1.0):
            try:
                self.step()
            except Exception as exc:
                print(f"[STT] retry_worker_error={type(exc).__name__}")

    def stop(self):
        with self._lifecycle:
            self._closed = True
            self._stop.set()
            thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=20.0)
        if len(self.queue):
            print(f"[STT] stopped queue_clips_lost={len(self.queue)}")
