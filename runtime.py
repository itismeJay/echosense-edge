"""High-level lifecycle for the single-source AFTER detection runtime."""

from __future__ import annotations

import threading

from audio.led_indicator import LEDIndicator
from audio.shared_source import PyAudioSharedSource
from config import (
    ACOUSTIC_DEGRADED_MODE,
    DETECTION_RUNTIME,
    MICROPHONE_STRICT,
    PREFERRED_MICROPHONE_NAME,
)
from detection.acoustic_detector import AcousticDetector, AcousticDetectorWorker
from detection.coordinator import CoordinatorAudioWorker, EventCoordinator
from detection.text_detector import TextDetector, TextDetectorWorker
from model.monitored_terms import start_dictionary_sync
from model.whisper_stt import (
    RealtimeSTTFeedWorker,
    consume_transcription_result,
    get_recorder,
    stop_recorder,
)
from model.yamnet_infer import load_yamnet_bundle
from sender.delivery import start_alert_delivery, stop_alert_delivery
from sender.http_client import check_backend_connection, start_heartbeat
from sender.incident_delivery import enqueue_finalized_incident


class EdgeRuntime:
    """Initialize, run, and cleanly stop all bounded AFTER components."""

    def __init__(
        self,
        *,
        config=DETECTION_RUNTIME,
        source=None,
        led=None,
        stop_event=None,
        network_info_provider=None,
    ):
        self.config = config
        self.source = source
        self.led = led or LEDIndicator()
        self.stop_event = stop_event or threading.Event()
        self.network_info_provider = network_info_provider
        self.coordinator = None
        self.workers = []
        self.started = False

    def _finalize(self, incident):
        queued = enqueue_finalized_incident(incident)
        if queued:
            self.led.alert()
        return queued

    def start(self):
        self.led.startup()
        if not start_alert_delivery():
            print(
                "[OUTBOX] worker_not_started=true; enqueue will report "
                "durability failures explicitly"
            )
        check_backend_connection()
        start_dictionary_sync()
        start_heartbeat(
            interval=60,
            info_provider=self.network_info_provider,
        )

        yamnet = load_yamnet_bundle(degraded_mode=ACOUSTIC_DEGRADED_MODE)
        print(
            f"[ACOUSTIC] yamnet_available={str(yamnet.available).lower()} "
            f"reason={yamnet.reason}"
        )
        get_recorder()
        if self.source is None:
            self.source = PyAudioSharedSource(
                preferred_name=PREFERRED_MICROPHONE_NAME,
                strict=MICROPHONE_STRICT,
                frame_samples=self.config.frame_samples,
                pre_trigger_seconds=self.config.pre_trigger_seconds,
            )
        stt_frames = self.source.subscribe(
            "text-stt",
            self.config.audio_queue_frames,
        )
        acoustic_frames = self.source.subscribe(
            "acoustic",
            self.config.audio_queue_frames,
        )
        context_frames = self.source.subscribe(
            "incident-context",
            self.config.audio_queue_frames,
        )
        self.coordinator = EventCoordinator(
            config=self.config,
            pre_trigger_buffer=self.source.pre_trigger,
            finalizer=self._finalize,
        )
        text_detector = TextDetector(
            history_size=self.config.evidence_history_size,
        )
        acoustic_detector = AcousticDetector(
            interpreter=yamnet.interpreter,
            class_names=yamnet.class_names,
            unavailable_reason=yamnet.reason,
        )
        self.workers = [
            RealtimeSTTFeedWorker(
                stt_frames,
                max_internal_chunks=self.config.stt_internal_audio_chunks,
            ),
            AcousticDetectorWorker(
                acoustic_detector,
                acoustic_frames,
                self.coordinator.submit_acoustic,
                window_samples=self.config.yamnet_window_samples,
                hop_samples=self.config.yamnet_hop_samples,
            ),
            CoordinatorAudioWorker(self.coordinator, context_frames),
            TextDetectorWorker(
                text_detector,
                consume_transcription_result,
                self.coordinator.submit_text,
            ),
        ]
        for worker in self.workers:
            worker.start()
        self.source.start()
        self.led.listening_start()
        self.started = True
        print("[RUNTIME] AFTER detection architecture started")

    def run(self):
        self.start()
        while not self.stop_event.wait(0.1):
            self.coordinator.tick()
            error = getattr(self.source, "error", None)
            if error is None:
                error = next(
                    (
                        getattr(worker, "error", None)
                        for worker in self.workers
                        if getattr(worker, "error", None) is not None
                    ),
                    None,
                )
            if error is not None:
                raise RuntimeError(
                    f"detection worker failed: {type(error).__name__}"
                ) from error

    def stop(self):
        cleanup_errors = []

        def safely(name, callback):
            try:
                callback()
            except Exception as exc:
                cleanup_errors.append((name, exc))
                print(f"[RUNTIME] cleanup_error={name}:{type(exc).__name__}")

        if self.source is not None:
            safely("audio_source", self.source.stop)
        for worker in reversed(self.workers):
            safely(type(worker).__name__, worker.stop)
        safely("realtimestt", stop_recorder)
        if self.coordinator is not None:
            safely(
                "coordinator",
                lambda: self.coordinator.shutdown(finalize_active=True),
            )
        safely("outbox", stop_alert_delivery)
        safely("led", self.led.cleanup)
        self.started = False
        print(
            f"[RUNTIME] stopped cleanup_errors={len(cleanup_errors)}"
        )
