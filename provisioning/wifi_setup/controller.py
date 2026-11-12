"""Fail-safe provisioning state machine for a single Wi-Fi interface."""

from __future__ import annotations

import hashlib
import threading
import time

from provisioning.wifi_setup.validation import (
    WifiInputError,
    validate_profile_uuid,
    validate_ssid,
    validate_wifi_password,
)


CONNECTED = "connected"
WAITING = "waiting_for_authorized_network"
SCANNING = "scanning"
HOTSPOT = "setup_hotspot"
CONNECTING = "connecting"
FAILED = "failed_recovered"


class ProvisioningBusyError(RuntimeError):
    pass


def network_fingerprint(ssid: str) -> str:
    return hashlib.sha256(ssid.encode("utf-8")).hexdigest()[:8]


class ProvisioningController:
    """Coordinates transitions without assuming AP/client concurrency."""

    def __init__(self, manager, credentials, *, logger=print, delay=time.sleep):
        self.manager = manager
        self.credentials = credentials
        self.logger = logger
        self.delay = delay
        self.state = WAITING
        self.last_error = ""
        self.networks = []
        self._operation_lock = threading.Lock()
        self._worker = None

    @property
    def operation_in_progress(self) -> bool:
        return bool(self._worker and self._worker.is_alive())

    def status(self) -> dict:
        connection = self.manager.connection_status()
        return {
            "state": self.state,
            "last_error": self.last_error,
            "connected": connection.client_connected,
            "ipv4_address": connection.ipv4_address,
            "current_network": connection.ssid or connection.connection_name,
            "networks": list(self.networks),
            "saved_networks": self.manager.list_saved(),
            "operation_in_progress": self.operation_in_progress,
        }

    def enter_setup(self) -> None:
        """Scan in client mode first, then create the fallback AP."""

        with self._operation_lock:
            self.state = SCANNING
            self.logger("[WIFI_SETUP] state=scanning mode=client")
            try:
                self.networks = self.manager.scan()
            except Exception as exc:
                self.networks = []
                self.logger(
                    f"[WIFI_SETUP] scan_error={getattr(exc, 'code', type(exc).__name__)}"
                )
            self.manager.start_hotspot(
                self.credentials.hotspot_ssid,
                self.credentials.hotspot_password,
            )
            self.state = HOTSPOT
            self.last_error = ""
            self.logger("[WIFI_SETUP] state=setup_hotspot access=local_only")

    def scan_connected(self) -> None:
        """Refresh scan results without disrupting a live client connection."""

        with self._operation_lock:
            connection = self.manager.connection_status()
            if not connection.client_connected:
                raise RuntimeError("client_connection_unavailable")
            self.state = SCANNING
            self.networks = self.manager.scan()
            self.state = CONNECTED
            self.last_error = ""
            self.logger("[WIFI_SETUP] state=connected scan=refreshed")

    def _start_worker(self, target) -> None:
        if self.operation_in_progress:
            raise ProvisioningBusyError("Another Wi-Fi operation is already running")
        self._worker = threading.Thread(
            target=target,
            name="echosense-wifi-operation",
            daemon=True,
        )
        self._worker.start()

    def _recover_after_failure(self) -> None:
        try:
            restored = self.manager.connect_saved_profiles(retries=1)
            if restored.client_connected:
                self.state = CONNECTED
                self.logger("[WIFI_SETUP] state=connected recovery=saved_network")
                return
        except Exception as exc:
            self.logger(
                "[WIFI_SETUP] saved_recovery_error="
                f"{getattr(exc, 'code', type(exc).__name__)}"
            )
        try:
            self.manager.start_hotspot(
                self.credentials.hotspot_ssid,
                self.credentials.hotspot_password,
            )
            self.state = FAILED
            self.logger("[WIFI_SETUP] state=failed_recovered setup_access=restored")
        except Exception as exc:
            self.state = FAILED
            self.last_error = (
                f"hotspot_{getattr(exc, 'code', type(exc).__name__)}"
            )
            self.logger(
                f"[WIFI_SETUP] state=failed recovery_error={self.last_error}"
            )

    def request_refresh(self) -> None:
        """Refresh in-place as a client, or temporarily leave AP mode.

        In setup mode the phone disconnects during the controlled single-radio
        transition and can rejoin the same setup hotspot afterward.
        """

        def refresh():
            self.delay(0.5)
            with self._operation_lock:
                self.state = SCANNING
                connection = self.manager.connection_status()
                if connection.client_connected:
                    self.logger("[WIFI_SETUP] state=scanning mode=client_connected")
                    try:
                        self.networks = self.manager.scan()
                        self.state = CONNECTED
                        self.last_error = ""
                    except Exception as exc:
                        self.state = CONNECTED
                        self.last_error = getattr(exc, "code", type(exc).__name__)
                        self.logger(f"[WIFI_SETUP] scan_error={self.last_error}")
                    return
                self.logger("[WIFI_SETUP] state=scanning transition=hotspot_to_client")
                try:
                    self.manager.stop_hotspot()
                    self.networks = self.manager.scan()
                    self.last_error = ""
                except Exception as exc:
                    self.last_error = getattr(exc, "code", type(exc).__name__)
                    self.logger(f"[WIFI_SETUP] scan_error={self.last_error}")
                finally:
                    try:
                        self.manager.start_hotspot(
                            self.credentials.hotspot_ssid,
                            self.credentials.hotspot_password,
                        )
                        self.state = HOTSPOT
                        self.logger("[WIFI_SETUP] state=setup_hotspot recovery=complete")
                    except Exception as exc:
                        self.state = FAILED
                        self.last_error = getattr(exc, "code", type(exc).__name__)
                        self.logger(
                            f"[WIFI_SETUP] state=failed hotspot_error={self.last_error}"
                        )

        self._start_worker(refresh)

    def request_connection(
        self,
        ssid,
        password,
        *,
        hidden=False,
        open_network=False,
        activation_delay=0.5,
        confirmed=False,
    ) -> None:
        if not confirmed:
            raise WifiInputError("Confirm the Wi-Fi change before connecting")
        clean_ssid = validate_ssid(ssid)
        clean_password = validate_wifi_password(
            password, open_network=bool(open_network)
        )
        try:
            transition_delay = max(0.5, min(15.0, float(activation_delay)))
        except (TypeError, ValueError):
            transition_delay = 0.5

        def connect():
            self.delay(transition_delay)
            with self._operation_lock:
                self.state = CONNECTING
                self.last_error = ""
                fingerprint = network_fingerprint(clean_ssid)
                self.logger(
                    f"[WIFI_SETUP] state=connecting network={fingerprint} "
                    f"hidden={str(bool(hidden)).lower()}"
                )
                try:
                    self.manager.stop_hotspot()
                    status = self.manager.connect(
                        clean_ssid,
                        clean_password,
                        hidden=bool(hidden),
                        open_network=bool(open_network),
                    )
                    if not status.client_connected:
                        raise RuntimeError("connection_not_verified")
                    self.state = CONNECTED
                    self.logger(
                        f"[WIFI_SETUP] state=connected network={fingerprint}"
                    )
                    return
                except Exception as exc:
                    self.last_error = getattr(exc, "code", type(exc).__name__)
                    self.logger(
                        f"[WIFI_SETUP] state=connection_failed network={fingerprint} "
                        f"error={self.last_error}"
                    )
                self._recover_after_failure()

        self._start_worker(connect)

    def request_saved_connection(self, profile_uuid, *, confirmed=False) -> None:
        if not confirmed:
            raise WifiInputError("Confirm the Wi-Fi change before connecting")
        clean_uuid = validate_profile_uuid(profile_uuid)

        def connect_saved():
            self.delay(0.5)
            with self._operation_lock:
                self.state = CONNECTING
                self.last_error = ""
                self.logger("[WIFI_SETUP] state=connecting source=saved_profile")
                try:
                    self.manager.stop_hotspot()
                    status = self.manager.activate_saved(clean_uuid)
                    if not status.client_connected:
                        raise RuntimeError("connection_not_verified")
                    self.state = CONNECTED
                    self.logger("[WIFI_SETUP] state=connected source=saved_profile")
                    return
                except Exception as exc:
                    self.last_error = getattr(exc, "code", type(exc).__name__)
                    self.logger(
                        "[WIFI_SETUP] state=connection_failed source=saved_profile "
                        f"error={self.last_error}"
                    )
                self._recover_after_failure()

        self._start_worker(connect_saved)

    def rejoin_saved_from_hotspot(self) -> bool:
        """Leave setup mode when a saved network is back in range.

        The single radio cannot scan while it is an access point, so the
        hotspot is stopped for one scan and restored unless a saved network
        is visible and its connection is verified.
        """

        with self._operation_lock:
            saved = [p for p in self.manager.list_saved() if p.autoconnect]
            if not saved:
                return False
            self.state = SCANNING
            self.logger("[WIFI_SETUP] state=scanning reason=hotspot_rejoin_check")
            connected = False
            try:
                self.manager.stop_hotspot()
                self.networks = self.manager.scan()
                visible = {network.ssid for network in self.networks}
                in_range = [p for p in saved if p.ssid in visible]
                if in_range:
                    self.logger(
                        "[WIFI_SETUP] saved_network_in_range="
                        + ",".join(network_fingerprint(p.ssid) for p in in_range)
                    )
                    status = self.manager.connect_saved_profiles(retries=1)
                    connected = status.client_connected
                    if connected:
                        self.state = CONNECTED
                        self.last_error = ""
                        self.logger(
                            "[WIFI_SETUP] state=connected source=hotspot_rejoin "
                            f"network={network_fingerprint(status.ssid or '')}"
                        )
                else:
                    self.logger("[WIFI_SETUP] saved_network_in_range=none")
            except Exception as exc:
                self.last_error = getattr(exc, "code", type(exc).__name__)
                self.logger(f"[WIFI_SETUP] rejoin_error={self.last_error}")
            finally:
                if not connected:
                    try:
                        self.manager.start_hotspot(
                            self.credentials.hotspot_ssid,
                            self.credentials.hotspot_password,
                        )
                        self.state = HOTSPOT
                    except Exception as exc:
                        self.state = FAILED
                        self.last_error = getattr(exc, "code", type(exc).__name__)
                        self.logger(
                            f"[WIFI_SETUP] state=failed hotspot_error={self.last_error}"
                        )
            return connected

    def set_priority(self, profile_uuid, priority) -> bool:
        if self.operation_in_progress:
            raise ProvisioningBusyError("Another Wi-Fi operation is already running")
        clean_uuid = validate_profile_uuid(profile_uuid)
        try:
            value = int(str(priority).strip())
        except (TypeError, ValueError):
            raise WifiInputError("Priority must be a whole number") from None
        if not -999 <= value <= 999:
            raise WifiInputError("Priority must be between -999 and 999")
        changed = self.manager.set_priority(clean_uuid, value)
        self.logger(f"[WIFI_SETUP] saved_network_priority_set={str(changed).lower()}")
        return changed

    def forget(self, profile_uuid, *, confirmed=False) -> bool:
        if self.operation_in_progress:
            raise ProvisioningBusyError("Another Wi-Fi operation is already running")
        if not confirmed:
            raise WifiInputError("Confirm before forgetting a saved network")
        clean_uuid = validate_profile_uuid(profile_uuid)
        removed = self.manager.forget(clean_uuid)
        self.logger(f"[WIFI_SETUP] saved_network_removed={str(removed).lower()}")
        return removed
