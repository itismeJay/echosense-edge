"""Fail-closed configuration and visibility checks for Dashboard Test Mode."""

from __future__ import annotations

from datetime import datetime, timezone
import threading
import time
from urllib.parse import urlparse

import requests


class DashboardTestConfigurationError(RuntimeError):
    """Dashboard Test Mode is not safely or completely configured."""


def backend_origin(value: str) -> str:
    """Return a normalized HTTP(S) origin or an empty string."""

    parsed = urlparse(str(value or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return ""
    default_port = (
        parsed.scheme == "http" and parsed.port == 80
    ) or (
        parsed.scheme == "https" and parsed.port == 443
    )
    authority = parsed.hostname
    if parsed.port and not default_port:
        authority = f"{authority}:{parsed.port}"
    return f"{parsed.scheme}://{authority}"


def validate_dashboard_test_configuration(
    *,
    enabled: bool,
    api_url: str,
    expected_backend_origin: str,
    device_id: str,
    expected_device_id: str,
    device_key: str,
    outbox_enabled: bool,
    outbox_database_path: str,
    reviewer_token: str,
) -> None:
    """Reject unsafe test delivery before the worker can make a request."""

    if not enabled:
        return
    actual_origin = backend_origin(api_url)
    approved_origin = backend_origin(expected_backend_origin)
    if not actual_origin or actual_origin != approved_origin:
        raise DashboardTestConfigurationError(
            "backend origin does not match the explicitly approved test origin"
        )
    if not expected_device_id or device_id != expected_device_id:
        raise DashboardTestConfigurationError(
            "device identity does not match the explicitly approved test device"
        )
    if not device_key:
        raise DashboardTestConfigurationError("test device credentials are missing")
    if not outbox_enabled:
        raise DashboardTestConfigurationError("the test outbox must be enabled")
    if "test" not in str(outbox_database_path).casefold():
        raise DashboardTestConfigurationError(
            "Dashboard Test Mode requires a dedicated test-named outbox"
        )
    if not reviewer_token:
        raise DashboardTestConfigurationError(
            "a memory-only reviewer token is required for dashboard verification"
        )


class DashboardVisibilityVerifier:
    """Confirm a delivered TEST event appears in the dashboard's alert feed."""

    def __init__(
        self,
        api_url,
        reviewer_token,
        *,
        timeout_seconds=12.0,
        poll_seconds=0.5,
        request_timeout_seconds=5.0,
        get=None,
        clock=time.monotonic,
        wall_clock=None,
        sleeper=time.sleep,
        logger=print,
    ):
        self.api_url = str(api_url).rstrip("/")
        self._reviewer_token = str(reviewer_token)
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.poll_seconds = max(0.1, float(poll_seconds))
        self.request_timeout_seconds = max(
            0.5,
            float(request_timeout_seconds),
        )
        self._get = get or requests.get
        self._clock = clock
        self._wall_clock = wall_clock or (
            lambda: datetime.now(timezone.utc)
        )
        self._sleeper = sleeper
        self._logger = logger

    def schedule(self, payload: dict, *, backend_acknowledged_at=None) -> None:
        """Run the bounded reviewer-feed check outside the delivery worker."""

        acknowledged = (
            self._clock()
            if backend_acknowledged_at is None
            else float(backend_acknowledged_at)
        )
        thread = threading.Thread(
            target=self.verify,
            args=(dict(payload),),
            kwargs={"backend_acknowledged_at": acknowledged},
            name="echosense-dashboard-test-verifier",
            daemon=True,
        )
        thread.start()

    def verify(self, payload: dict, *, backend_acknowledged_at=None) -> bool:
        """Poll the same authenticated feed used by the dashboard."""

        event_id = str(payload.get("event_id") or "")
        if payload.get("trigger_type") != "TEST" or payload.get("test_mode") is not True:
            self._logger(
                f"[DASHBOARD_TEST] verification refused event={event_id or 'invalid'} "
                "reason=payload_not_labeled_test"
            )
            return False
        started = self._clock()
        acknowledged = started if backend_acknowledged_at is None else float(
            backend_acknowledged_at
        )
        deadline = started + self.timeout_seconds
        last_status = None
        while self._clock() <= deadline:
            try:
                response = self._get(
                    f"{self.api_url}/alerts/",
                    params={"event_id": event_id, "limit": 10},
                    headers={
                        "Authorization": f"Bearer {self._reviewer_token}",
                    },
                    timeout=self.request_timeout_seconds,
                )
                last_status = int(response.status_code)
                if 200 <= last_status < 300:
                    body = response.json()
                    alerts = body if isinstance(body, list) else []
                    visible = next(
                        (
                            item
                            for item in alerts
                            if isinstance(item, dict)
                            and str(item.get("event_id") or "") == event_id
                            and item.get("trigger_type") == "TEST"
                            and item.get("test_mode") is True
                        ),
                        None,
                    )
                    if visible is not None:
                        now = self._clock()
                        ack_to_feed_ms = max(
                            0.0,
                            (now - acknowledged) * 1000.0,
                        )
                        event_to_feed_ms = self._event_to_feed_ms(payload)
                        event_part = (
                            f" event_end_to_dashboard_feed_ms={event_to_feed_ms:.0f}"
                            if event_to_feed_ms is not None
                            else ""
                        )
                        self._logger(
                            f"[DASHBOARD_TEST] visible=true event={event_id} "
                            f"alert_id={visible.get('id', 'unknown')} "
                            f"backend_ack_to_dashboard_feed_ms={ack_to_feed_ms:.0f}"
                            f"{event_part}"
                        )
                        return True
                elif last_status in {401, 403}:
                    break
            except (requests.RequestException, TypeError, ValueError):
                last_status = None
            self._sleeper(self.poll_seconds)
        status = last_status if last_status is not None else "request_error"
        self._logger(
            f"[DASHBOARD_TEST] visible=false event={event_id} "
            f"last_http={status} timeout_seconds={self.timeout_seconds:.1f}"
        )
        return False

    def _event_to_feed_ms(self, payload):
        value = payload.get("event_end_timestamp")
        if not isinstance(value, str):
            return None
        try:
            event_end = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if event_end.tzinfo is None:
            return None
        return max(
            0.0,
            (self._wall_clock() - event_end.astimezone(timezone.utc)).total_seconds()
            * 1000.0,
        )
