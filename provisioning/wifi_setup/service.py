"""Long-running recovery service; independent from EchoSense detection."""

from __future__ import annotations

import ipaddress
import os
import signal
import threading
import time

from provisioning.wifi_setup.controller import CONNECTED, ProvisioningController
from provisioning.wifi_setup.credentials import load_setup_credentials
from provisioning.wifi_setup.networkmanager import NetworkManager
from provisioning.wifi_setup.portal import make_portal_server


class WifiSetupService:
    def __init__(
        self,
        manager,
        credentials,
        *,
        recovery_seconds=45,
        poll_seconds=5,
        hotspot_scan_seconds=60,
        portal_factory=make_portal_server,
        logger=print,
        sleep=time.sleep,
        clock=time.monotonic,
    ):
        self.manager = manager
        self.credentials = credentials
        self.recovery_seconds = max(10.0, float(recovery_seconds))
        self.poll_seconds = max(1.0, float(poll_seconds))
        self.hotspot_scan_seconds = max(30.0, float(hotspot_scan_seconds))
        self.clock = clock
        self._hotspot_since = None
        self.portal_factory = portal_factory
        self.logger = logger
        self.sleep = sleep
        self.controller = ProvisioningController(
            manager, credentials, logger=logger, delay=sleep
        )
        self.stop_event = threading.Event()
        self.portal = None
        self.portal_thread = None
        self.portal_endpoint = None

    def stop(self):
        self.stop_event.set()
        self._stop_portal(reason="service_stopping")

    def _start_portal(self, status):
        if status.is_hotspot:
            endpoint = ("10.42.0.1", "10.42.0.0/24")
        elif status.client_connected and status.ipv4_address:
            try:
                local_network = str(ipaddress.ip_interface(status.ipv4_cidr).network)
            except ValueError:
                self.logger(
                    "[WIFI_SETUP] portal=not_started reason=invalid_client_subnet"
                )
                return
            endpoint = (status.ipv4_address, local_network)
        else:
            return
        if (
            self.portal_endpoint == endpoint
            and self.portal_thread
            and self.portal_thread.is_alive()
        ):
            return
        self._stop_portal(reason="network_changed")
        if status.client_connected:
            try:
                self.controller.scan_connected()
            except Exception as exc:
                self.logger(
                    f"[WIFI_SETUP] scan_error={getattr(exc, 'code', type(exc).__name__)}"
                )
        self.portal = self.portal_factory(
            self.controller,
            self.credentials,
            bind_address=endpoint[0],
            allowed_network=endpoint[1],
            captive=bool(status.is_hotspot),
        )
        self.portal_endpoint = endpoint
        self.portal_thread = threading.Thread(
            target=self.portal.serve_forever,
            kwargs={"poll_interval": 0.5},
            name="echosense-wifi-portal",
            daemon=True,
        )
        self.portal_thread.start()
        mode = "setup_hotspot" if status.is_hotspot else "client_lan"
        self.logger(f"[WIFI_SETUP] portal=started bind={mode}")

    def _stop_portal(self, *, reason):
        portal = self.portal
        if portal is None:
            return
        portal.shutdown()
        portal.server_close()
        self.portal = None
        self.portal_thread = None
        self.portal_endpoint = None
        self.logger(f"[WIFI_SETUP] portal=stopped reason={reason}")

    def _maybe_rejoin_saved(self):
        """While in setup mode, periodically look for a saved network."""

        now = self.clock()
        if self._hotspot_since is None:
            self._hotspot_since = now
            return
        if now - self._hotspot_since < self.hotspot_scan_seconds:
            return
        if self.controller.operation_in_progress:
            return
        in_use = getattr(self.portal, "in_use", None)
        if callable(in_use) and in_use():
            self.logger("[WIFI_SETUP] rejoin_check=skipped reason=portal_in_use")
            self._hotspot_since = now
            return
        self._hotspot_since = now
        if self.controller.rejoin_saved_from_hotspot():
            self._hotspot_since = None
            self._start_portal(self.manager.connection_status())

    def _connect_saved_or_start_setup(self):
        """Boundedly restore an authorized client before exposing setup."""

        self.logger(
            "[WIFI_SETUP] state=waiting_for_authorized_network retries=2"
        )
        try:
            status = self.manager.connect_saved_profiles(
                retries=2,
                retry_delay=min(10.0, self.recovery_seconds / 2.0),
            )
            if status.client_connected:
                self.controller.state = CONNECTED
                self.logger("[WIFI_SETUP] state=connected source=saved_profile")
                return
        except Exception as exc:
            self.logger(
                "[WIFI_SETUP] saved_connect_error="
                f"{getattr(exc, 'code', type(exc).__name__)}"
            )
        self.controller.enter_setup()
        self._start_portal(self.manager.connection_status())

    def run(self):
        preflight = self.manager.preflight()
        if not preflight.get("supported"):
            self.logger("[WIFI_SETUP] preflight=failed no_network_changes=true")
            return 2
        self.logger(
            "[WIFI_SETUP] preflight=passed platform=raspberrypi64 "
            "networkmanager=true"
        )
        reconciled = self.manager.reconcile_managed_profiles()
        self.logger(
            "[WIFI_SETUP] recovery_reconcile="
            f"finalized:{reconciled['finalized']},removed:{reconciled['removed']}"
        )
        while not self.stop_event.is_set():
            try:
                status = self.manager.connection_status()
            except Exception as exc:
                self.logger(
                    f"[WIFI_SETUP] status_error={getattr(exc, 'code', type(exc).__name__)}"
                )
                self.stop_event.wait(self.poll_seconds)
                continue

            if status.client_connected:
                self._hotspot_since = None
                self.controller.state = CONNECTED
                self._start_portal(status)
            elif status.is_hotspot:
                self._start_portal(status)
                self._maybe_rejoin_saved()
            elif not self.controller.operation_in_progress:
                try:
                    self._connect_saved_or_start_setup()
                except Exception as exc:
                    self.logger(
                        f"[WIFI_SETUP] hotspot_error={getattr(exc, 'code', type(exc).__name__)}"
                    )
            self.stop_event.wait(self.poll_seconds)
        self._stop_portal(reason="service_stopping")
        return 0


def service_from_environment(environ=None):
    source = os.environ if environ is None else environ
    interface = str(source.get("ECHOSENSE_WIFI_INTERFACE") or "wlan0").strip()
    credential_path = str(
        source.get("ECHOSENSE_WIFI_SETUP_CREDENTIALS")
        or "/etc/echosense/wifi-setup.json"
    )
    runtime_directory = str(
        source.get("ECHOSENSE_WIFI_RUNTIME_DIRECTORY")
        or "/run/echosense-wifi"
    )
    credentials = load_setup_credentials(credential_path)
    manager = NetworkManager(interface, runtime_directory=runtime_directory)
    return WifiSetupService(
        manager,
        credentials,
        recovery_seconds=float(
            source.get("ECHOSENSE_WIFI_RECOVERY_SECONDS") or 45
        ),
        poll_seconds=float(source.get("ECHOSENSE_WIFI_POLL_SECONDS") or 5),
        hotspot_scan_seconds=float(
            source.get("ECHOSENSE_WIFI_HOTSPOT_SCAN_SECONDS") or 60
        ),
    )


def run_service():
    service = service_from_environment()

    def stop(_signum, _frame):
        service.stop_event.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    return service.run()
