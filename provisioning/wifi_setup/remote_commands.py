"""Apply Wi-Fi changes made in the dashboard (device-key authenticated).

Every 60 s the box GETs ``/device/wifi/commands`` and POSTs one result per
command. Passwords arrive only in that response; they are handed to
NetworkManager on stdin and never logged or reported.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from urllib.parse import quote

import requests

from provisioning.wifi_setup.controller import network_fingerprint
from provisioning.wifi_setup.validation import (
    WifiInputError,
    validate_ssid,
    validate_wifi_password,
)


POLL_SECONDS = 60.0
ROLLBACK_SECONDS = 120.0
REACHABILITY_RETRY_SECONDS = 5.0
NOT_SUPPORTED = "not_supported_on_this_device"
ACTIONS = ("add", "update", "remove", "set_priority", "connect_now")


class CommandError(Exception):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _code(exc) -> str:
    return str(getattr(exc, "code", "") or type(exc).__name__)


class WifiCommandRunner:
    def __init__(
        self,
        manager,
        *,
        api_url: str,
        headers=dict,
        session=requests,
        logger=print,
        sleep=time.sleep,
        clock=time.monotonic,
        rollback_seconds=ROLLBACK_SECONDS,
        backend_reachable=None,
        timeout=10.0,
    ):
        self.manager = manager  # None: no NetworkManager (Mac) -> simulated
        self.api_url = str(api_url or "").rstrip("/")
        self.headers = headers
        self.session = session
        self.logger = logger
        self.sleep = sleep
        self.clock = clock
        self.rollback_seconds = float(rollback_seconds)
        self.backend_reachable = backend_reachable or self._backend_reachable
        self.timeout = timeout
        self._unreported = {}  # applied commands whose result POST failed
        self._missing_endpoint_logged = False

    # ---- backend ---------------------------------------------------------

    def _backend_reachable(self) -> bool:
        try:
            self.session.get(
                f"{self.api_url}/devices/me/config",
                headers=self.headers(),
                timeout=self.timeout,
            )
            return True
        except Exception:
            return False

    def fetch(self) -> list:
        try:
            response = self.session.get(
                f"{self.api_url}/device/wifi/commands",
                headers=self.headers(),
                timeout=self.timeout,
            )
        except Exception as exc:
            self.logger(f"[WIFI_REMOTE] poll unavailable ({type(exc).__name__})")
            return []
        status = getattr(response, "status_code", None)
        if status == 404:
            if not self._missing_endpoint_logged:
                self.logger("[WIFI_REMOTE] backend has no Wi-Fi commands endpoint (404)")
                self._missing_endpoint_logged = True
            return []
        if status in (401, 403):
            self.logger(f"[WIFI_REMOTE] ERROR poll refused ({status}): check device credentials")
            return []
        try:
            response.raise_for_status()
            commands = response.json().get("commands") or []
        except Exception as exc:
            self.logger(f"[WIFI_REMOTE] invalid poll response ({type(exc).__name__})")
            return []
        return [c for c in commands if isinstance(c, dict) and c.get("id") not in (None, "")]

    def report(self, command_id: str, result: dict) -> bool:
        body = {k: v for k, v in result.items() if v not in (None, "")}
        try:
            response = self.session.post(
                f"{self.api_url}/device/wifi/commands/{quote(command_id, safe='')}/result",
                json=body,
                headers=self.headers(),
                timeout=self.timeout,
            )
            ok = 200 <= int(getattr(response, "status_code", 0)) < 300
        except Exception:
            ok = False
        if not ok:
            self.logger(f"[WIFI_REMOTE] result not delivered id={command_id}; will retry")
        return ok

    def poll_once(self) -> list:
        """Fetch, apply each new command once, report each result."""

        results = []
        listed = set()
        for command in self.fetch():
            command_id = str(command["id"])
            listed.add(command_id)
            result = self._unreported.get(command_id)
            if result is None:
                result = self.apply(command)
            if self.report(command_id, result):
                self._unreported.pop(command_id, None)
            else:
                self._unreported[command_id] = result
            results.append((command_id, result))
        # Commands the backend no longer lists need no further report.
        for command_id in list(self._unreported):
            if command_id not in listed:
                self._unreported.pop(command_id, None)
        return results

    # ---- applying --------------------------------------------------------

    def apply(self, command: dict) -> dict:
        command_id = str(command.get("id"))
        action = str(command.get("action") or "")
        raw_ssid = str(command.get("ssid") or "")
        network = network_fingerprint(raw_ssid) if raw_ssid else "none"
        if self.manager is None:
            self.logger(
                f"[WIFI_REMOTE] wifi command received (simulated) id={command_id} "
                f"action={action} network={network}"
            )
            return {"status": "failed", "error": NOT_SUPPORTED}
        self.logger(
            f"[WIFI_REMOTE] command received id={command_id} action={action} network={network}"
        )
        try:
            if action not in ACTIONS:
                raise CommandError("unknown_action")
            try:
                ssid = validate_ssid(raw_ssid)
            except WifiInputError:
                raise CommandError("invalid_ssid") from None
            getattr(self, f"_{action}")(ssid, command)
            result = {"status": "applied"}
        except Exception as exc:
            result = {"status": "failed", "error": _code(exc)}
        result["current_ssid"] = self._current_ssid()
        self.logger(
            f"[WIFI_REMOTE] command id={command_id} action={action} "
            f"status={result['status']} error={result.get('error') or 'none'}"
        )
        return result

    def _current_ssid(self):
        try:
            status = self.manager.connection_status()
            return status.ssid if status.client_connected else None
        except Exception:
            return None

    @staticmethod
    def _password(command):
        password = command.get("password")
        if password in (None, ""):
            return None
        try:
            return validate_wifi_password(password)
        except WifiInputError:
            raise CommandError("invalid_password") from None

    @staticmethod
    def _priority(command, *, required=False):
        value = command.get("priority")
        if value is None:
            if required:
                raise CommandError("invalid_priority")
            return None
        if isinstance(value, bool):
            raise CommandError("invalid_priority")
        try:
            number = int(value)
        except (TypeError, ValueError):
            raise CommandError("invalid_priority") from None
        if number != value and str(number) != str(value).strip():
            raise CommandError("invalid_priority")
        if not -999 <= number <= 999:
            raise CommandError("invalid_priority")
        return number

    def _profile(self, ssid):
        profiles = self.manager.profiles_for_ssid(ssid)
        if not profiles:
            raise CommandError("network_not_saved")
        return profiles[0]

    def _add(self, ssid, command):
        password = self._password(command)
        priority = self._priority(command)
        existing = self.manager.profiles_for_ssid(ssid)
        if existing:
            self.manager.update_network(existing[0].uuid, password=password, priority=priority)
            return
        self.manager.save_network(
            ssid, password or "", hidden=bool(command.get("hidden")), priority=priority
        )

    def _update(self, ssid, command):
        password = self._password(command)
        priority = self._priority(command)
        if password is None and priority is None:
            raise CommandError("nothing_to_update")
        profile = self._profile(ssid)
        self.manager.update_network(profile.uuid, password=password, priority=priority)

    def _set_priority(self, ssid, command):
        priority = self._priority(command, required=True)
        profile = self._profile(ssid)
        if not self.manager.set_priority(profile.uuid, priority):
            raise CommandError("priority_not_saved")

    def _reachable_alternatives(self, ssid):
        visible = {network.ssid for network in self.manager.scan()}
        candidates = [
            profile for profile in self.manager.list_saved()
            if profile.autoconnect and profile.ssid != ssid and profile.ssid in visible
        ]
        return sorted(candidates, key=lambda p: (-p.priority, p.ssid.casefold()))

    def _remove(self, ssid, _command):
        profiles = self.manager.profiles_for_ssid(ssid)
        if not profiles:
            raise CommandError("network_not_saved")
        active = next((p for p in profiles if p.active), None)
        if active is not None:
            # Never drop the network in use without a reachable saved fallback.
            alternatives = self._reachable_alternatives(ssid)
            if not alternatives:
                raise CommandError("in_use_no_alternative")
            for alternative in alternatives:
                try:
                    self.manager.activate_saved(alternative.uuid)
                    break
                except Exception:
                    continue
            else:
                self._rollback(active.uuid)
                raise CommandError("in_use_switch_failed")
            self.logger(
                "[WIFI_REMOTE] switched before remove "
                f"network={network_fingerprint(alternative.ssid)}"
            )
        for profile in self.manager.profiles_for_ssid(ssid):
            if not self.manager.forget(profile.uuid):
                raise CommandError("remove_failed")

    def _connect_now(self, ssid, _command):
        profile = self._profile(ssid)
        if profile.active:
            return
        before = self.manager.connection_status()
        previous_uuid = before.connection_uuid if before.client_connected else None
        try:
            self.manager.activate_saved(profile.uuid)
        except Exception:
            self._rollback(previous_uuid)
            raise CommandError("connect_failed") from None
        deadline = self.clock() + self.rollback_seconds
        while True:
            if self.backend_reachable():
                return
            if self.clock() >= deadline:
                break
            self.sleep(REACHABILITY_RETRY_SECONDS)
        self.logger(
            f"[WIFI_REMOTE] backend unreachable for {int(self.rollback_seconds)}s; rolling back"
        )
        self._rollback(previous_uuid)
        raise CommandError("backend_unreachable_rolled_back")

    def _rollback(self, previous_uuid):
        try:
            if previous_uuid:
                self.manager.activate_saved(previous_uuid)
                self.logger("[WIFI_REMOTE] rollback=previous_network")
                return
        except Exception as exc:
            self.logger(f"[WIFI_REMOTE] rollback_error={_code(exc)}")
        try:
            self.manager.connect_saved_profiles(retries=1)
            self.logger("[WIFI_REMOTE] rollback=saved_networks")
        except Exception as exc:
            self.logger(f"[WIFI_REMOTE] rollback_error={_code(exc)}")


def build_runner():
    from config import API_URL, NETWORK_INTERFACE, PLATFORM
    from sender.http_client import device_auth_headers

    manager = None
    if PLATFORM in ("raspberrypi", "linux") and shutil.which("nmcli"):
        from provisioning.wifi_setup.networkmanager import NetworkManager

        manager = NetworkManager(
            NETWORK_INTERFACE,
            runtime_directory=os.environ.get("ECHOSENSE_WIFI_RUNTIME_DIRECTORY")
            or "/run/echosense-wifi",
        )
    # Late-bound print so main.py's log shipping sees these lines.
    return WifiCommandRunner(
        manager,
        api_url=API_URL,
        headers=device_auth_headers,
        logger=lambda line: print(line),
    )


def start_wifi_command_poll(stop_event=None, interval=None) -> threading.Thread:
    """Poll now, then every ``interval`` seconds in one daemon thread."""

    stop_event = stop_event or threading.Event()
    seconds = float(
        interval or os.environ.get("ECHOSENSE_WIFI_COMMAND_POLL_SECONDS") or POLL_SECONDS
    )
    runner = build_runner()

    def _loop():
        while not stop_event.is_set():
            try:
                runner.poll_once()
            except Exception as exc:  # never let the poll thread die
                print(f"[WIFI_REMOTE] poll_error={type(exc).__name__}")
            stop_event.wait(seconds)

    thread = threading.Thread(target=_loop, name="wifi-command-poll", daemon=True)
    thread.start()
    return thread
