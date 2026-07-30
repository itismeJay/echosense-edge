"""Background delivery lifecycle for the persistent alert outbox."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import json
import random
import threading
import time
from typing import Callable

from config import (
    API_URL,
    OUTBOX_BACKOFF_MULTIPLIER,
    OUTBOX_BUSY_TIMEOUT_MS,
    OUTBOX_DATABASE_PATH,
    OUTBOX_DELIVERED_RETENTION_DAYS,
    OUTBOX_ENABLED,
    OUTBOX_HEALTH_LOG_INTERVAL_SECONDS,
    OUTBOX_INITIAL_RETRY_DELAY_SECONDS,
    OUTBOX_JITTER_PERCENT,
    OUTBOX_MAX_RECORDS_PER_CYCLE,
    OUTBOX_MAX_RETRY_DELAY_SECONDS,
    OUTBOX_POLL_INTERVAL_SECONDS,
    OUTBOX_REQUEST_TIMEOUT_SECONDS,
    OUTBOX_STALE_SENDING_SECONDS,
)
from sender.http_client import (
    DELIVERED as HTTP_DELIVERED,
    PERMANENT as HTTP_PERMANENT,
    RETRYABLE as HTTP_RETRYABLE,
    DeliveryResult,
    deliver_alert_payload,
)
from sender.outbox import (
    AlertOutbox,
    OutboxError,
    OutboxRecord,
    utc_now,
)


@dataclass(frozen=True, slots=True)
class DeliverySettings:
    enabled: bool = OUTBOX_ENABLED
    database_path: str = OUTBOX_DATABASE_PATH
    api_url: str = API_URL
    poll_interval_seconds: float = OUTBOX_POLL_INTERVAL_SECONDS
    request_timeout_seconds: float = OUTBOX_REQUEST_TIMEOUT_SECONDS
    initial_retry_delay_seconds: float = (
        OUTBOX_INITIAL_RETRY_DELAY_SECONDS
    )
    backoff_multiplier: float = OUTBOX_BACKOFF_MULTIPLIER
    max_retry_delay_seconds: float = OUTBOX_MAX_RETRY_DELAY_SECONDS
    jitter_percent: float = OUTBOX_JITTER_PERCENT
    stale_sending_seconds: float = OUTBOX_STALE_SENDING_SECONDS
    delivered_retention_days: float = OUTBOX_DELIVERED_RETENTION_DAYS
    max_records_per_cycle: int = OUTBOX_MAX_RECORDS_PER_CYCLE
    busy_timeout_ms: int = OUTBOX_BUSY_TIMEOUT_MS
    health_log_interval_seconds: float = (
        OUTBOX_HEALTH_LOG_INTERVAL_SECONDS
    )


class OutboxDeliveryWorker:
    """Single background worker that performs one HTTP attempt per claim."""

    def __init__(
        self,
        outbox: AlertOutbox,
        settings: DeliverySettings,
        *,
        deliver: Callable[[dict], DeliveryResult] | None = None,
        clock=utc_now,
        random_uniform=None,
        logger=print,
    ):
        self.outbox = outbox
        self.settings = settings
        self._clock = clock
        self._random_uniform = random_uniform or random.uniform
        self._logger = logger
        self._deliver = deliver or (
            lambda payload: deliver_alert_payload(
                payload,
                api_url=self.settings.api_url,
                timeout=self.settings.request_timeout_seconds,
            )
        )
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._thread = None
        self._stale_records_recovered = 0
        self._last_health_log_monotonic = 0.0

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    @property
    def stale_records_recovered(self) -> int:
        return self._stale_records_recovered

    def start(self) -> bool:
        with self._lifecycle_lock:
            if self.running:
                return False
            self._stop_event.clear()
            self._wake_event.clear()
            self._recover_stale()
            self._cleanup_delivered()
            self._thread = threading.Thread(
                target=self._run,
                name="echosense-outbox-worker",
                daemon=True,
            )
            self._thread.start()
        self._logger("[OUTBOX_WORKER] started")
        return True

    def stop(self, *, join_timeout=None) -> bool:
        with self._lifecycle_lock:
            thread = self._thread
            if thread is None:
                return True
            self._stop_event.set()
            self._wake_event.set()
        timeout = (
            float(join_timeout)
            if join_timeout is not None
            else self.settings.request_timeout_seconds + 2.0
        )
        thread.join(timeout=max(0.1, timeout))
        stopped = not thread.is_alive()
        if stopped:
            with self._lifecycle_lock:
                if self._thread is thread:
                    self._thread = None
            self._logger("[OUTBOX_WORKER] stopped")
        else:
            self._logger(
                "[OUTBOX_WORKER] stop timeout; leased record remains recoverable"
            )
        return stopped

    def wake(self) -> None:
        self._wake_event.set()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                processed = self.run_once()
                self._maybe_log_health()
            except OutboxError as exc:
                processed = 0
                error_code = getattr(exc, "code", type(exc).__name__)
                self._logger(
                    f"[OUTBOX] database error type={error_code}"
                )
            if processed:
                continue
            self._wake_event.wait(self.settings.poll_interval_seconds)
            self._wake_event.clear()

    def run_once(self, *, max_records=None) -> int:
        self._recover_stale()
        limit = max_records or self.settings.max_records_per_cycle
        processed = 0
        for _ in range(max(1, int(limit))):
            if self._stop_event.is_set():
                break
            record = self.outbox.claim_next(
                lease_seconds=self.settings.stale_sending_seconds,
                now=self._clock(),
            )
            if record is None:
                break
            self._process_record(record)
            processed += 1
        return processed

    def _process_record(self, record: OutboxRecord) -> None:
        try:
            payload = json.loads(record.payload_json)
            if not isinstance(payload, dict):
                raise ValueError("payload is not a JSON object")
            payload_event_id = str(payload.get("event_id") or "")
            if payload_event_id != record.event_id:
                raise ValueError("stored event_id mismatch")
        except (TypeError, ValueError, json.JSONDecodeError):
            self.outbox.mark_permanent_failure(
                record.id,
                error_type="corrupt_payload",
                error="stored payload requires review",
                now=self._clock(),
            )
            self._logger(
                f"[OUTBOX] permanent failure event={record.event_id} "
                "type=corrupt_payload"
            )
            return

        self._logger(
            f"[OUTBOX] delivery attempted event={record.event_id} "
            f"attempt={record.attempt_count}"
        )
        try:
            result = self._deliver(payload)
        except Exception as exc:
            result = DeliveryResult(
                HTTP_RETRYABLE,
                error_type=f"delivery_callable_{type(exc).__name__.lower()}",
            )

        if result.disposition == HTTP_DELIVERED:
            self.outbox.mark_delivered(
                record.id,
                http_status=result.http_status,
                now=self._clock(),
            )
            duplicate = " duplicate_ack=true" if result.duplicate_acknowledged else ""
            self._logger(
                f"[OUTBOX] delivered event={record.event_id} "
                f"http={result.http_status}{duplicate}"
            )
            return

        if result.disposition == HTTP_PERMANENT:
            self.outbox.mark_permanent_failure(
                record.id,
                error_type=result.error_type or "permanent_failure",
                error="backend rejected payload; operator review required",
                http_status=result.http_status,
                now=self._clock(),
            )
            self._logger(
                f"[OUTBOX] permanent failure event={record.event_id} "
                f"type={result.error_type or 'permanent_failure'} "
                f"http={result.http_status}"
            )
            return

        delay = self.retry_delay_seconds(
            record.attempt_count,
            retry_after_seconds=result.retry_after_seconds,
        )
        next_attempt_at = self._clock() + timedelta(seconds=delay)
        self.outbox.mark_retryable_failure(
            record.id,
            next_attempt_at=next_attempt_at,
            error_type=result.error_type or "retryable_failure",
            error="temporary delivery failure",
            http_status=result.http_status,
            now=self._clock(),
        )
        self._logger(
            f"[OUTBOX] retry scheduled event={record.event_id} "
            f"type={result.error_type or 'retryable_failure'} "
            f"delay_seconds={delay:.3f}"
        )

    def retry_delay_seconds(
        self,
        attempt_count: int,
        *,
        retry_after_seconds=None,
    ) -> float:
        exponent = max(0, int(attempt_count) - 1)
        base = min(
            self.settings.max_retry_delay_seconds,
            self.settings.initial_retry_delay_seconds
            * (self.settings.backoff_multiplier ** exponent),
        )
        spread = base * self.settings.jitter_percent
        jittered = base + self._random_uniform(-spread, spread)
        delay = max(self.settings.poll_interval_seconds, jittered)
        if retry_after_seconds is not None:
            delay = max(delay, max(0.0, float(retry_after_seconds)))
        return min(self.settings.max_retry_delay_seconds, delay)

    def _recover_stale(self) -> int:
        recovered = self.outbox.recover_stale_sending(now=self._clock())
        self._stale_records_recovered += recovered
        return recovered

    def _cleanup_delivered(self) -> int:
        retention_seconds = (
            self.settings.delivered_retention_days * 24 * 60 * 60
        )
        return self.outbox.cleanup_delivered(
            retention_seconds,
            now=self._clock(),
        )

    def health_summary(self) -> dict:
        summary = self.outbox.health_summary(now=self._clock())
        summary.update({
            "worker_running": self.running,
            "stale_records_recovered": self.stale_records_recovered,
        })
        return summary

    def _maybe_log_health(self) -> None:
        now = time.monotonic()
        if (
            now - self._last_health_log_monotonic
            < self.settings.health_log_interval_seconds
        ):
            return
        self._last_health_log_monotonic = now
        self._logger(
            "[OUTBOX_HEALTH] "
            + json.dumps(
                self.health_summary(),
                sort_keys=True,
                separators=(",", ":"),
            )
        )


class AlertDeliveryService:
    """Coordinates durable enqueueing and exactly one worker lifecycle."""

    def __init__(
        self,
        settings: DeliverySettings | None = None,
        *,
        deliver=None,
        clock=utc_now,
        random_uniform=None,
        logger=print,
    ):
        self.settings = settings or DeliverySettings()
        self._deliver = deliver
        self._clock = clock
        self._random_uniform = random_uniform
        self._logger = logger
        self._lock = threading.RLock()
        self._start_requested = False
        self._outbox = None
        self._worker = None
        self._last_database_error_type = None

    @property
    def outbox(self):
        return self._outbox

    @property
    def worker(self):
        return self._worker

    def _ensure_ready(self) -> bool:
        if not self.settings.enabled:
            return False
        with self._lock:
            if self._outbox is None:
                self._outbox = AlertOutbox(
                    self.settings.database_path,
                    busy_timeout_ms=self.settings.busy_timeout_ms,
                    clock=self._clock,
                    logger=self._logger,
                )
            if not self._outbox.initialized:
                try:
                    self._outbox.initialize()
                    self._last_database_error_type = None
                except OutboxError as exc:
                    error_code = getattr(exc, "code", type(exc).__name__)
                    self._last_database_error_type = error_code
                    self._logger(
                        "[OUTBOX] database initialization failed "
                        f"type={error_code}"
                    )
                    return False
            if self._worker is None:
                self._worker = OutboxDeliveryWorker(
                    self._outbox,
                    self.settings,
                    deliver=self._deliver,
                    clock=self._clock,
                    random_uniform=self._random_uniform,
                    logger=self._logger,
                )
            if self._start_requested and not self._worker.running:
                self._worker.start()
            return True

    def start(self) -> bool:
        with self._lock:
            if self._start_requested and self._worker and self._worker.running:
                return False
            self._start_requested = True
        return self._ensure_ready()

    def stop(self) -> bool:
        with self._lock:
            self._start_requested = False
            worker = self._worker
        if worker is None:
            return True
        stopped = worker.stop()
        if self._outbox is not None:
            self._outbox.close()
        return stopped

    def enqueue_payload(self, payload: dict) -> bool:
        if not self._ensure_ready():
            self._logger(
                "[OUTBOX] alert not durably queued "
                f"database_error_type={self._last_database_error_type or 'disabled'}"
            )
            return False
        try:
            result = self._outbox.enqueue(payload)
        except OutboxError as exc:
            error_code = getattr(exc, "code", type(exc).__name__)
            self._last_database_error_type = error_code
            self._logger(
                "[OUTBOX] alert not durably queued "
                f"database_error_type={error_code}"
            )
            return False
        if self._worker is not None:
            self._worker.wake()
        return bool(result.record_id)

    def health_summary(self) -> dict:
        if not self.settings.enabled:
            return {
                "database_status": "DISABLED",
                "worker_running": False,
                "stale_records_recovered": 0,
            }
        if not self._ensure_ready():
            return {
                "database_status": (
                    f"ERROR:{self._last_database_error_type or 'unknown'}"
                ),
                "worker_running": False,
                "stale_records_recovered": 0,
            }
        return self._worker.health_summary()


_DEFAULT_SERVICE = None
_DEFAULT_SERVICE_LOCK = threading.Lock()


def get_alert_delivery_service() -> AlertDeliveryService:
    global _DEFAULT_SERVICE
    with _DEFAULT_SERVICE_LOCK:
        if _DEFAULT_SERVICE is None:
            _DEFAULT_SERVICE = AlertDeliveryService()
        return _DEFAULT_SERVICE


def start_alert_delivery() -> bool:
    return get_alert_delivery_service().start()


def stop_alert_delivery() -> bool:
    return get_alert_delivery_service().stop()


def get_outbox_health() -> dict:
    return get_alert_delivery_service().health_summary()
