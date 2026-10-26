from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import math
import requests
import time
import threading
import uuid
from urllib.parse import urlsplit
from config import (
    API_URL,
    ECHOSENSE_DEVICE_ID,
    ECHOSENSE_DEVICE_KEY,
)
from sender.payload_v2 import (
    AlertPayloadValidationError,
    build_alert_payload_v2,
    normalize_event_id,
    validate_alert_payload_v2,
)

DELIVERED = "DELIVERED"
RETRYABLE = "RETRYABLE"
PERMANENT = "PERMANENT"
AUTHENTICATION = "AUTHENTICATION"
_RETRYABLE_HTTP_STATUSES = {408, 425, 429, 500, 502, 503, 504}


@dataclass(frozen=True, slots=True)
class DeliveryResult:
    disposition: str
    http_status: int | None = None
    error_type: str | None = None
    retry_after_seconds: float | None = None
    duplicate_acknowledged: bool = False


def _retry_after_seconds(response, *, now=None):
    value = str(response.headers.get("Retry-After") or "").strip()
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        retry_at = parsedate_to_datetime(value)
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        current = now or datetime.now(timezone.utc)
        return max(
            0.0,
            (retry_at.astimezone(timezone.utc) - current).total_seconds(),
        )
    except (TypeError, ValueError, OverflowError):
        return None


def _valid_backend_url(value: object) -> bool:
    try:
        parsed = urlsplit(str(value or "").strip())
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def deliver_alert_payload(
    payload,
    *,
    api_url=API_URL,
    timeout=10.0,
    device_id=ECHOSENSE_DEVICE_ID,
    device_key=ECHOSENSE_DEVICE_KEY,
    post=None,
) -> DeliveryResult:
    """Perform exactly one HTTP attempt and classify its acknowledgement."""

    normalized_device_id = str(device_id or "").strip()
    normalized_device_key = str(device_key or "").strip()
    if (
        not normalized_device_id
        or len(normalized_device_id) > 100
        or not normalized_device_key
        or len(normalized_device_key) > 256
    ):
        return DeliveryResult(
            AUTHENTICATION,
            error_type="device_credentials_missing",
        )
    if not _valid_backend_url(api_url):
        return DeliveryResult(
            RETRYABLE,
            error_type="backend_url_invalid",
        )
    try:
        request_timeout = float(timeout)
    except (TypeError, ValueError):
        request_timeout = 0.0
    if not math.isfinite(request_timeout) or request_timeout <= 0.0:
        return DeliveryResult(
            RETRYABLE,
            error_type="request_timeout_invalid",
        )
    try:
        validate_alert_payload_v2(payload)
        stable_event_id = normalize_event_id(payload.get("event_id"))
    except (AttributeError, AlertPayloadValidationError):
        return DeliveryResult(
            PERMANENT,
            error_type="invalid_schema_v2_payload",
        )
    if payload.get("device_identifier") != normalized_device_id:
        return DeliveryResult(
            AUTHENTICATION,
            error_type="device_identity_mismatch",
        )

    request_post = post or requests.post
    try:
        response = request_post(
            f"{api_url}/alerts/",
            json=payload,
            headers={
                "X-EchoSense-Device-Id": device_id,
                "X-EchoSense-Device-Key": device_key,
                "Idempotency-Key": stable_event_id,
            },
            timeout=request_timeout,
        )
    except requests.Timeout:
        return DeliveryResult(RETRYABLE, error_type="timeout")
    except requests.ConnectionError as exc:
        name = type(exc).__name__.lower()
        message = str(exc).lower()
        if "name resolution" in message or "getaddrinfo" in message:
            error_type = "dns_failure"
        elif "refused" in message:
            error_type = "connection_refused"
        else:
            error_type = name or "connection_error"
        return DeliveryResult(RETRYABLE, error_type=error_type)
    except requests.RequestException as exc:
        return DeliveryResult(
            RETRYABLE,
            error_type=type(exc).__name__.lower(),
        )

    status = int(response.status_code)
    if 200 <= status < 300:
        return DeliveryResult(DELIVERED, http_status=status)
    if status in {401, 403}:
        return DeliveryResult(
            AUTHENTICATION,
            http_status=status,
            error_type=f"device_authentication_http_{status}",
        )
    if status in _RETRYABLE_HTTP_STATUSES or status >= 500:
        return DeliveryResult(
            RETRYABLE,
            http_status=status,
            error_type=f"http_{status}",
            retry_after_seconds=(
                _retry_after_seconds(response)
                if status in {429, 503}
                else None
            ),
        )
    if 400 <= status < 500:
        return DeliveryResult(
            PERMANENT,
            http_status=status,
            error_type=f"http_{status}",
        )
    return DeliveryResult(
        RETRYABLE,
        http_status=status,
        error_type=f"unexpected_http_{status}",
    )


def send_alert(severity, confidence, duration,
               transcribed_text="", detected_words=None,
               categories=None, language="unknown",
               hard_hits=None, soft_hits=None,
               required_duration=None, duration_gate=None,
               yamnet_class="Unknown", yamnet_score=0.0,
               yamnet_ran=False,
               emotion="unknown", tone_data=None,
               waveform_snapshot=None, language_confidence=None,
               matched_terms=None, event_id=None, retries=None,
               severity_evidence=None,
               event_start_timestamp=None,
               event_end_timestamp=None,
               trigger_type="KEYWORD",
               acoustic_evidence_role="supporting_only",
               device_identifier=ECHOSENSE_DEVICE_ID):
    """Persist one complete alert payload before background HTTP delivery.

    The retained ``retries`` argument is ignored for call compatibility. Retry
    state is durable and managed by the outbox worker.
    """

    stable_event_id = str(event_id or uuid.uuid4())
    try:
        ended_at = event_end_timestamp or datetime.now(timezone.utc)
        started_at = event_start_timestamp
        if started_at is None:
            duration_seconds = float(duration)
            if not math.isfinite(duration_seconds) or duration_seconds < 0.0:
                raise AlertPayloadValidationError(
                    "duration must be finite and non-negative"
                )
            started_at = ended_at - timedelta(seconds=duration_seconds)
        payload = build_alert_payload_v2(
            event_id=stable_event_id,
            trigger_type=trigger_type,
            severity=severity,
            event_start_timestamp=started_at,
            event_end_timestamp=ended_at,
            device_identifier=device_identifier,
            transcribed_text=transcribed_text,
            detected_words=detected_words,
            confidence=confidence,
            language=language,
            language_confidence=language_confidence,
            matched_terms=matched_terms,
            severity_evidence=severity_evidence,
            yamnet_class=yamnet_class,
            yamnet_score=yamnet_score,
            yamnet_ran=yamnet_ran,
            emotion=emotion,
            tone_data=tone_data,
            waveform_snapshot=waveform_snapshot,
            acoustic_evidence_role=acoustic_evidence_role,
        )
    except (AlertPayloadValidationError, TypeError, ValueError, OverflowError) as exc:
        error_type = type(exc).__name__
        print(
            "[OUTBOX] alert not durably queued "
            f"event=invalid payload_error_type={error_type}"
        )
        return False
    from sender.delivery import get_alert_delivery_service

    return get_alert_delivery_service().enqueue_payload(payload)

def heartbeat_headers(device_id=ECHOSENSE_DEVICE_ID, device_key=ECHOSENSE_DEVICE_KEY):
    """Device credentials for the heartbeat, so the backend records this device's check-in.

    Without them the backend only updates its global status and the dashboard shows the
    device as Offline between alerts.
    """
    device_id = str(device_id or "").strip()
    device_key = str(device_key or "").strip()
    if not device_id or not device_key:
        return {}
    return {"X-EchoSense-Device-Id": device_id, "X-EchoSense-Device-Key": device_key}


def _heartbeat_loop(interval, info_provider=None):
    while True:
        time.sleep(interval)
        # Best-effort: include the Pi's live LAN IP/SSID so the device can be
        # located from the backend even when raspberrypi.local can't be reached
        # (e.g. a phone hotspot that blocks mDNS). Unknown fields are harmless;
        # if the backend ignores the body the heartbeat still works.
        info = {}
        try:
            info = info_provider() if info_provider else {}
        except Exception:
            info = {}
        try:
            response = requests.post(
                f"{API_URL}/system-settings/heartbeat",
                json=info or None,
                headers=heartbeat_headers(),
                timeout=10
            )
            ip = info.get("ip", "?")
            print(f"[HEARTBEAT] {response.status_code} ip={ip}")
        except Exception as e:
            print(f"[HEARTBEAT] Error: {e}")

def start_heartbeat(interval=60, info_provider=None):
    t = threading.Thread(
        target=_heartbeat_loop, args=(interval, info_provider), daemon=True
    )
    t.start()
    print(f"[HEARTBEAT] Started — posting every {interval}s")

def check_backend_connection():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            print(f"[SENDER] Backend connected!")
            return True
    except:
        print(f"[SENDER] Backend not reachable!")
        return False


# ── Remote log shipping ─────────────────────────────────────────────────────
# Buffers print() output and ships it to the backend so admins can read the
# Pi's logs from the dashboard without SSH. Best-effort: failures are swallowed
# so log shipping can never take down the detection loop.
import queue as _queue
import threading as _threading

_LOG_BUFFER = []
_LOG_LOCK = _threading.Lock()

def push_log_line(line: str) -> None:
    global _LOG_BUFFER
    if not line or not line.strip():
        return
    with _LOG_LOCK:
        _LOG_BUFFER.append(line.strip())
        if len(_LOG_BUFFER) >= 20:
            lines = _LOG_BUFFER.copy()
            _LOG_BUFFER.clear()
            _threading.Thread(
                target=_send_logs_batch,
                args=(lines,),
                daemon=True,
            ).start()

def _send_logs_batch(lines: list) -> None:
    try:
        requests.post(
            f"{API_URL}/system/logs",
            json={"lines": lines},
            timeout=5,
        )
    except Exception:
        pass  # logs are not critical — fail silently

def flush_logs() -> None:
    global _LOG_BUFFER
    with _LOG_LOCK:
        if _LOG_BUFFER:
            lines = _LOG_BUFFER.copy()
            _LOG_BUFFER.clear()
            _threading.Thread(
                target=_send_logs_batch,
                args=(lines,),
                daemon=True,
            ).start()

def start_log_flush_thread() -> None:
    import time as _time
    def _flush_loop():
        while True:
            _time.sleep(10)
            flush_logs()
    t = _threading.Thread(target=_flush_loop, daemon=True)
    t.start()
    print("[LOGS] Log sender started")
