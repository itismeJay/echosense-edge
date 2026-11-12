"""High-level lifecycle for the single-source AFTER detection runtime."""

from __future__ import annotations

import threading

from audio.led_indicator import LEDIndicator
from audio.shared_source import PyAudioSharedSource
from config import (
    ACOUSTIC_DEGRADED_MODE,
    API_URL,
    DASHBOARD_TEST_EXPECTED_BACKEND_ORIGIN,
    DASHBOARD_TEST_EXPECTED_DEVICE_ID,
    DASHBOARD_TEST_MODE,
    DASHBOARD_TEST_REVIEWER_TOKEN,
    DETECTION_RUNTIME,
    ECHOSENSE_DEVICE_ID,
    ECHOSENSE_DEVICE_KEY,
    MICROPHONE_STRICT,
    OUTBOX_DATABASE_PATH,
    OUTBOX_ENABLED,
    PREFERRED_MICROPHONE_NAME,
    SEMANTIC_RUNTIME,
)
from detection.acoustic_detector import (
    AcousticDetector,
    AcousticDetectorWorker,
    NoiseFloorTracker,
)
from detection.coordinator import CoordinatorAudioWorker, EventCoordinator
from detection.latency import LATENCY
from detection.llm_semantic import SemanticGate
from detection.semantic import SemanticAnalyzer
from detection.text_detector import TextDetector, TextDetectorWorker
from model.school_config import start_school_config_sync
from model.whisper_stt import (
    RealtimeSTTFeedWorker,
    set_media_classifier,
    consume_transcription_result,
    get_recorder,
    stop_recorder,
)
from model.yamnet_infer import load_yamnet_bundle, top_classes_float32
from sender.delivery import start_alert_delivery, stop_alert_delivery
from sender.dashboard_test import (
    backend_origin,
    validate_dashboard_test_configuration,
)
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
        semantic_classifier=None,
    ):
        self.config = config
        self.source = source
        self.led = led or LEDIndicator()
        self.stop_event = stop_event or threading.Event()
        self.network_info_provider = network_info_provider
        self.semantic_classifier = semantic_classifier
        self.coordinator = None
        self.workers = []
        self.started = False

    def _finalize(self, incident):
        queued = enqueue_finalized_incident(incident)
        if queued:
            LATENCY.mark_incident(incident.event_id, "incident_enqueued")
            self.led.alert()
        return queued

    def start(self):
        validate_dashboard_test_configuration(
            enabled=DASHBOARD_TEST_MODE,
            api_url=API_URL,
            expected_backend_origin=DASHBOARD_TEST_EXPECTED_BACKEND_ORIGIN,
            device_id=ECHOSENSE_DEVICE_ID,
            expected_device_id=DASHBOARD_TEST_EXPECTED_DEVICE_ID,
            device_key=ECHOSENSE_DEVICE_KEY,
            outbox_enabled=OUTBOX_ENABLED,
            outbox_database_path=OUTBOX_DATABASE_PATH,
            reviewer_token=DASHBOARD_TEST_REVIEWER_TOKEN,
        )
        if DASHBOARD_TEST_MODE:
            print(
                "[DASHBOARD_TEST] enabled=true "
                f"backend_origin={backend_origin(API_URL)} "
                f"device={ECHOSENSE_DEVICE_ID} "
                f"outbox={OUTBOX_DATABASE_PATH} label=TEST"
            )
        self.led.startup()
        if not start_alert_delivery():
            print(
                "[OUTBOX] worker_not_started=true; enqueue will report "
                "durability failures explicitly"
            )
        check_backend_connection()
        start_school_config_sync()
        start_heartbeat(
            interval=60,
            info_provider=self.network_info_provider,
        )

        yamnet = load_yamnet_bundle(degraded_mode=ACOUSTIC_DEGRADED_MODE)
        print(
            f"[ACOUSTIC] yamnet_available={str(yamnet.available).lower()} "
            f"reason={yamnet.reason}"
        )
        # The media filter scores each finished utterance with its own YAMNet
        # interpreter; TFLite interpreters must not be shared across threads.
        media_yamnet = load_yamnet_bundle(degraded_mode=True)
        if media_yamnet.available:
            set_media_classifier(
                lambda samples, sample_rate: top_classes_float32(
                    samples,
                    sample_rate,
                    media_yamnet.interpreter,
                    media_yamnet.class_names,
                )
            )
        print(
            f"[FILTER] media_classifier={str(media_yamnet.available).lower()}"
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
        # Claude banter/bullying classifier (ECHOSENSE_SEMANTIC_MODE, default
        # all). Without ANTHROPIC_API_KEY it logs once and stays lexicon-only.
        self.semantic_gate = SemanticGate.from_environment()
        text_detector = TextDetector(
            history_size=self.config.evidence_history_size,
            semantic_gate=self.semantic_gate,
            semantic_analyzer=SemanticAnalyzer(
                classifier=self.semantic_classifier,
                enabled=SEMANTIC_RUNTIME.enabled,
                min_confidence=SEMANTIC_RUNTIME.min_confidence,
                max_context_turns=SEMANTIC_RUNTIME.max_context_turns,
                context_window_seconds=SEMANTIC_RUNTIME.context_window_seconds,
                max_context_chars=SEMANTIC_RUNTIME.max_context_chars,
            ),
        )
        print(
            "[SEMANTIC] enabled="
            f"{str(SEMANTIC_RUNTIME.enabled).lower()} "
            "available="
            f"{str(self.semantic_classifier is not None).lower()} "
            "mode=shadow_only"
        )
        acoustic_detector = AcousticDetector(
            interpreter=yamnet.interpreter,
            class_names=yamnet.class_names,
            unavailable_reason=yamnet.reason,
            noise_floor=NoiseFloorTracker.from_config(self.config),
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
        gate = getattr(self, "semantic_gate", None)
        if gate is not None:
            safely("semantic_llm", gate.shutdown)
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
