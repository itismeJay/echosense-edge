"""Narrow, testable NetworkManager adapter.

Passwords are never placed in argv, command output, or logs.  NetworkManager's
documented ``passwd-file`` input is created owner-only in a volatile runtime
directory and unlinked immediately after activation.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import time
import uuid


HOTSPOT_PROFILE = "EchoSense Setup Hotspot"
# NetworkManager accepts connection.autoconnect-priority in this range.
MIN_PRIORITY = -999
MAX_PRIORITY = 999
MANAGED_PROFILE_PREFIX = "EchoSense WiFi "


class NetworkManagerError(RuntimeError):
    """A bounded NetworkManager operation failed with a privacy-safe code."""

    def __init__(self, code: str):
        self.code = str(code or "networkmanager_error")
        super().__init__(self.code)


@dataclass(frozen=True, slots=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True, slots=True)
class WifiNetwork:
    ssid: str
    signal: int
    security: str
    active: bool = False

    @property
    def secured(self) -> bool:
        return bool(self.security and self.security != "--")


@dataclass(frozen=True, slots=True)
class SavedNetwork:
    name: str
    uuid: str
    ssid: str
    active: bool = False
    autoconnect: bool = True
    priority: int = 0
    hidden: bool = False


@dataclass(frozen=True, slots=True)
class ConnectionStatus:
    state: str
    connection_name: str = ""
    connection_uuid: str = ""
    ipv4_address: str = ""
    is_hotspot: bool = False
    ssid: str = ""
    ipv4_cidr: str = ""

    @property
    def client_connected(self) -> bool:
        return (
            self.state.lower().startswith("100")
            and bool(self.ipv4_address)
            and not self.is_hotspot
        )


def _split_escaped(line: str, separator=":") -> list[str]:
    fields = []
    current = []
    escaped = False
    for character in line:
        if escaped:
            current.append(character)
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == separator:
            fields.append("".join(current))
            current = []
        else:
            current.append(character)
    if escaped:
        current.append("\\")
    fields.append("".join(current))
    return fields


class NetworkManager:
    def __init__(
        self,
        interface="wlan0",
        *,
        runtime_directory="/run/echosense-wifi",
        runner=None,
        sleep=time.sleep,
    ):
        self.interface = str(interface or "").strip()
        if not self.interface or not self.interface.replace("-", "").isalnum():
            raise ValueError("invalid network interface")
        self.runtime_directory = Path(runtime_directory)
        self._runner = runner or self._run_subprocess
        self._sleep = sleep

    @staticmethod
    def _run_subprocess(args, *, input_text=None, timeout=30) -> CommandResult:
        environment = os.environ.copy()
        environment.update({"LC_ALL": "C", "LANG": "C", "NO_COLOR": "1"})
        completed = subprocess.run(
            args,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=environment,
            check=False,
        )
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)

    def _nmcli(self, *arguments, timeout=30, check=True, input_text=None) -> CommandResult:
        if input_text is None:
            result = self._runner(
                ["nmcli", "--colors", "no", *arguments],
                timeout=timeout,
            )
        else:
            result = self._runner(
                ["nmcli", "--colors", "no", *arguments],
                timeout=timeout,
                input_text=input_text,
            )
        if check and result.returncode != 0:
            raise NetworkManagerError(f"nmcli_exit_{result.returncode}")
        return result

    def preflight(self) -> dict:
        """Read-only host capability report.  It never changes networking."""

        try:
            os_release = Path("/etc/os-release").read_text(encoding="utf-8")
        except OSError:
            os_release = ""
        try:
            model = Path("/proc/device-tree/model").read_bytes().rstrip(b"\0").decode(
                "utf-8", errors="replace"
            )
        except OSError:
            model = ""
        version = self._nmcli("--version", check=False)
        running = self._nmcli("-t", "-f", "RUNNING", "general", check=False)
        device = self._nmcli(
            "-t", "-f", "GENERAL.TYPE", "device", "show", self.interface,
            check=False,
        )
        capabilities = self._nmcli(
            "-t", "-f", "WIFI-PROPERTIES.AP", "device", "show", self.interface,
            check=False,
        )
        release_values = {}
        for line in os_release.splitlines():
            key, separator, value = line.partition("=")
            if separator:
                release_values[key] = value.strip().strip('"')
        architecture_64_bit = os.uname().machine in {"aarch64", "arm64"}
        os_supported = (
            release_values.get("ID", "").lower() in {"debian", "raspbian"}
            and release_values.get("VERSION_CODENAME", "").lower()
            in {"bookworm", "trixie"}
        )
        ap_capable = (
            capabilities.returncode == 0
            and capabilities.stdout.rsplit(":", 1)[-1].strip().lower() == "yes"
        )
        return {
            "model": model,
            "os_release": os_release,
            "nmcli_version": version.stdout.strip(),
            "networkmanager_running": running.returncode == 0
            and running.stdout.strip().lower() == "running",
            "interface": self.interface,
            "wifi_interface": device.returncode == 0
            and "wifi" in device.stdout.lower(),
            "architecture_64_bit": architecture_64_bit,
            "os_supported": os_supported,
            "access_point_capable": ap_capable,
            # Even when a driver advertises multiple virtual modes, reliability
            # varies by firmware/channel. The service deliberately transitions
            # one physical interface instead of relying on concurrency.
            "radio_strategy": "single_interface_safe_transitions",
            "supported": (
                "Raspberry Pi" in model
                and architecture_64_bit
                and os_supported
                and version.returncode == 0
                and running.returncode == 0
                and running.stdout.strip().lower() == "running"
                and device.returncode == 0
                and "wifi" in device.stdout.lower()
                and ap_capable
            ),
        }

    def scan(self) -> list[WifiNetwork]:
        result = self._nmcli(
            "-t", "--escape", "yes",
            "-f", "IN-USE,SSID,SIGNAL,SECURITY",
            "device", "wifi", "list", "--rescan", "yes",
            "ifname", self.interface,
            timeout=20,
        )
        by_ssid = {}
        for line in result.stdout.splitlines():
            fields = _split_escaped(line)
            if len(fields) != 4 or not fields[1]:
                continue
            try:
                signal = max(0, min(100, int(fields[2] or 0)))
            except ValueError:
                signal = 0
            network = WifiNetwork(
                ssid=fields[1],
                signal=signal,
                security=fields[3] or "--",
                active=fields[0].strip() == "*",
            )
            previous = by_ssid.get(network.ssid)
            if previous is None or network.signal > previous.signal:
                by_ssid[network.ssid] = network
        return sorted(by_ssid.values(), key=lambda item: (-item.signal, item.ssid.casefold()))

    def connection_status(self) -> ConnectionStatus:
        result = self._nmcli(
            "-t", "--escape", "yes",
            "-f", "GENERAL.STATE,GENERAL.CONNECTION,GENERAL.CON-UUID,IP4.ADDRESS",
            "device", "show", self.interface,
        )
        values = {}
        for line in result.stdout.splitlines():
            key, separator, value = line.partition(":")
            if separator:
                values[key] = value.replace("\\:", ":")
        name = values.get("GENERAL.CONNECTION", "")
        connection_uuid = values.get("GENERAL.CON-UUID", "")
        ssid = ""
        if connection_uuid and connection_uuid != "--":
            ssid_result = self._nmcli(
                "--escape", "no", "-g", "802-11-wireless.ssid",
                "connection", "show",
                "uuid", connection_uuid, check=False,
            )
            if ssid_result.returncode == 0:
                ssid = ssid_result.stdout.strip()
        ipv4_cidr = values.get("IP4.ADDRESS[1]", "")
        return ConnectionStatus(
            state=values.get("GENERAL.STATE", ""),
            connection_name=name,
            connection_uuid=connection_uuid,
            ipv4_address=ipv4_cidr.split("/", 1)[0],
            is_hotspot=name == HOTSPOT_PROFILE,
            ssid=ssid,
            ipv4_cidr=ipv4_cidr,
        )

    def list_saved(self) -> list[SavedNetwork]:
        result = self._nmcli(
            "-t", "--escape", "yes",
            "-f", "NAME,UUID,TYPE,ACTIVE,AUTOCONNECT",
            "connection", "show",
        )
        saved = []
        for line in result.stdout.splitlines():
            fields = _split_escaped(line)
            if len(fields) != 5 or fields[2] not in {"wifi", "802-11-wireless"}:
                continue
            if fields[0] == HOTSPOT_PROFILE:
                continue
            ssid_result = self._nmcli(
                "--escape", "no", "-g", "802-11-wireless.ssid",
                "connection", "show",
                "uuid", fields[1], check=False,
            )
            hidden_result = self._nmcli(
                "-g", "802-11-wireless.hidden", "connection", "show",
                "uuid", fields[1], check=False,
            )
            priority_result = self._nmcli(
                "-g", "connection.autoconnect-priority", "connection", "show",
                "uuid", fields[1], check=False,
            )
            try:
                priority = int(priority_result.stdout.strip() or 0)
            except ValueError:
                priority = 0
            saved.append(SavedNetwork(
                name=fields[0],
                uuid=fields[1],
                ssid=ssid_result.stdout.strip() or fields[0],
                active=fields[3].lower() == "yes",
                autoconnect=fields[4].lower() == "yes",
                priority=priority,
                hidden=hidden_result.stdout.strip().lower() == "yes",
            ))
        return saved

    def activate_saved(self, profile_uuid: str, *, timeout=20) -> ConnectionStatus:
        """Activate one already-authorized profile and verify the result."""

        profiles = {profile.uuid: profile for profile in self.list_saved()}
        profile = profiles.get(str(profile_uuid or ""))
        if profile is None or not profile.autoconnect:
            raise NetworkManagerError("saved_profile_unavailable")
        self._nmcli(
            "--wait", str(int(timeout)), "connection", "up", "uuid",
            profile.uuid, "ifname", self.interface, timeout=timeout + 5,
        )
        status = self.connection_status()
        if not status.client_connected or status.connection_uuid != profile.uuid:
            raise NetworkManagerError("saved_connection_not_verified")
        return status

    def connect_saved_profiles(
        self,
        *,
        retries=2,
        timeout=20,
        retry_delay=1.0,
    ) -> ConnectionStatus:
        """Try authorized profiles in configured priority order, boundedly.

        A live client connection is always retained. Visible profiles are tried
        before hidden profiles; unavailable visible profiles are skipped after
        a successful scan. If scanning itself fails, NetworkManager may try all
        authorized profiles using its cached radio information.
        """

        status = self.connection_status()
        if status.client_connected:
            return status
        try:
            attempt_count = max(1, min(5, int(retries)))
        except (TypeError, ValueError):
            attempt_count = 2
        profiles = [
            profile for profile in self.list_saved()
            if profile.autoconnect and not profile.active
        ]
        profiles.sort(key=lambda profile: (-profile.priority, profile.name.casefold()))
        if not profiles:
            return status

        for attempt in range(attempt_count):
            visible_ssids = None
            try:
                visible_ssids = {network.ssid for network in self.scan()}
            except Exception:
                pass
            if visible_ssids is None:
                candidates = profiles
            else:
                candidates = [
                    profile for profile in profiles
                    if profile.hidden or profile.ssid in visible_ssids
                ]
            for profile in candidates:
                current = self.connection_status()
                if current.client_connected:
                    return current
                try:
                    return self.activate_saved(profile.uuid, timeout=timeout)
                except Exception:
                    continue
            if attempt + 1 < attempt_count:
                self._sleep(max(0.0, min(10.0, float(retry_delay))))
        return self.connection_status()

    def reconcile_managed_profiles(self) -> dict:
        """Recover only interrupted profiles created by this service.

        An active candidate made it through NetworkManager activation and is
        safe to finalize. An inactive non-autoconnect candidate never completed
        verification and is removed. Existing operator-created profiles are
        outside this namespace and are never changed.
        """

        finalized = 0
        removed = 0
        for profile in self.list_saved():
            if not profile.name.startswith(MANAGED_PROFILE_PREFIX):
                continue
            if profile.active and not profile.autoconnect:
                result = self._nmcli(
                    "connection", "modify", "uuid", profile.uuid,
                    "connection.autoconnect", "yes",
                    "connection.autoconnect-priority", str(self.next_priority()),
                    check=False,
                )
                finalized += int(result.returncode == 0)
            elif not profile.active and not profile.autoconnect:
                result = self._nmcli(
                    "connection", "delete", "uuid", profile.uuid, check=False
                )
                removed += int(result.returncode == 0)
        return {"finalized": finalized, "removed": removed}

    @contextmanager
    def _password_file(self, password: str):
        self.runtime_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor, path = tempfile.mkstemp(
            prefix="nm-secret-", dir=self.runtime_directory
        )
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(f"802-11-wireless-security.psk:{password}\n")
                handle.flush()
                os.fsync(handle.fileno())
            yield path
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def _delete_profile(self, identifier: str) -> None:
        self._nmcli("connection", "delete", "id", identifier, check=False)

    def start_hotspot(self, ssid: str, password: str) -> None:
        self._delete_profile(HOTSPOT_PROFILE)
        self._nmcli(
            "connection", "add", "save", "yes", "type", "wifi",
            "ifname", self.interface, "con-name", HOTSPOT_PROFILE,
            "ssid", ssid,
            "802-11-wireless.mode", "ap",
            "802-11-wireless.band", "bg",
            "ipv4.method", "shared",
            "ipv4.addresses", "10.42.0.1/24",
            "ipv6.method", "disabled",
            "connection.autoconnect", "no",
            "802-11-wireless-security.key-mgmt", "wpa-psk",
        )
        try:
            with self._password_file(password) as password_file:
                self._nmcli(
                    "--wait", "20", "connection", "up", "id", HOTSPOT_PROFILE,
                    "ifname", self.interface, "passwd-file", password_file,
                    timeout=25,
                )
        except Exception:
            self._delete_profile(HOTSPOT_PROFILE)
            raise

    def stop_hotspot(self) -> None:
        self._nmcli("connection", "down", "id", HOTSPOT_PROFILE, check=False)

    def connect(
        self,
        ssid: str,
        password: str,
        *,
        hidden=False,
        open_network=False,
        timeout=35,
    ) -> ConnectionStatus:
        """Make one bounded attempt and persist only a verified client profile."""

        digest = hashlib.sha256(ssid.encode("utf-8")).hexdigest()[:10]
        profile_name = f"{MANAGED_PROFILE_PREFIX}{digest}-{uuid.uuid4().hex[:6]}"
        arguments = [
            "connection", "add", "save", "yes", "type", "wifi",
            "ifname", self.interface, "con-name", profile_name, "ssid", ssid,
            "connection.autoconnect", "no",
            "802-11-wireless.hidden", "yes" if hidden else "no",
            "ipv4.method", "auto", "ipv6.method", "auto",
        ]
        if not open_network:
            arguments.extend(("802-11-wireless-security.key-mgmt", "wpa-psk"))
        self._nmcli(*arguments)
        try:
            up_arguments = [
                "--wait", str(int(timeout)), "connection", "up", "id",
                profile_name, "ifname", self.interface,
            ]
            if open_network:
                self._nmcli(*up_arguments, timeout=timeout + 5)
            else:
                with self._password_file(password) as password_file:
                    self._nmcli(
                        *up_arguments, "passwd-file", password_file,
                        timeout=timeout + 5,
                    )
            status = self.connection_status()
            if not status.client_connected or status.connection_name != profile_name:
                raise NetworkManagerError("connection_not_verified")
            # Newest verified network wins by default; the portal can reorder.
            self._nmcli(
                "connection", "modify", "id", profile_name,
                "connection.autoconnect", "yes",
                "connection.autoconnect-priority",
                str(self.next_priority(exclude=profile_name)),
            )
            return status
        except Exception:
            self._delete_profile(profile_name)
            raise

    def profiles_for_ssid(self, ssid: str) -> list[SavedNetwork]:
        return [profile for profile in self.list_saved() if profile.ssid == ssid]

    def _store_psk(self, profile_uuid: str, password: str) -> None:
        """Save a WPA secret through nmcli's editor on stdin, never argv."""

        self._nmcli(
            "connection", "modify", "uuid", profile_uuid,
            "802-11-wireless-security.key-mgmt", "wpa-psk",
        )
        # "yes" answers the editor's autoconnect save prompt when it appears;
        # otherwise it is an ignored unknown command.
        result = self._nmcli(
            "connection", "edit", "uuid", profile_uuid,
            input_text=(
                f"set 802-11-wireless-security.psk {password}\n"
                "save persistent\nyes\nquit\n"
            ),
            check=False,
        )
        if result.returncode != 0 or "error" in (result.stdout or "").lower():
            raise NetworkManagerError("password_not_saved")

    def save_network(
        self, ssid: str, password: str = "", *, hidden=False, priority=None
    ) -> str:
        """Save a new autoconnect profile without switching networks."""

        digest = hashlib.sha256(ssid.encode("utf-8")).hexdigest()[:10]
        profile_name = f"{MANAGED_PROFILE_PREFIX}{digest}-{uuid.uuid4().hex[:6]}"
        value = self.next_priority() if priority is None else int(priority)
        # Created with autoconnect off so an interrupted add is cleaned up by
        # reconcile_managed_profiles; enabled only after the secret is stored.
        self._nmcli(
            "connection", "add", "save", "yes", "type", "wifi",
            "ifname", self.interface, "con-name", profile_name, "ssid", ssid,
            "connection.autoconnect", "no",
            "802-11-wireless.hidden", "yes" if hidden else "no",
            "ipv4.method", "auto", "ipv6.method", "auto",
        )
        try:
            profile_uuid = self._nmcli(
                "-g", "connection.uuid", "connection", "show", "id", profile_name,
            ).stdout.strip()
            if password:
                self._store_psk(profile_uuid, password)
            self._nmcli(
                "connection", "modify", "uuid", profile_uuid,
                "connection.autoconnect", "yes",
                "connection.autoconnect-priority",
                str(max(MIN_PRIORITY, min(MAX_PRIORITY, value))),
            )
            return profile_uuid
        except Exception:
            self._delete_profile(profile_name)
            raise

    def update_network(self, profile_uuid: str, *, password=None, priority=None) -> None:
        if password:
            self._store_psk(profile_uuid, password)
        if priority is not None and not self.set_priority(profile_uuid, priority):
            raise NetworkManagerError("priority_not_saved")

    def next_priority(self, *, exclude: str = "") -> int:
        """One above the highest saved autoconnect priority (newest first)."""

        priorities = [
            profile.priority for profile in self.list_saved()
            if profile.autoconnect and profile.name != exclude
        ]
        return max(MIN_PRIORITY, min(MAX_PRIORITY, max(priorities, default=-1) + 1))

    def set_priority(self, profile_uuid: str, priority: int) -> bool:
        """Persist autoconnect order for one saved (non-hotspot) profile."""

        profiles = {profile.uuid: profile for profile in self.list_saved()}
        if profile_uuid not in profiles:
            return False
        value = max(MIN_PRIORITY, min(MAX_PRIORITY, int(priority)))
        result = self._nmcli(
            "connection", "modify", "uuid", profile_uuid,
            "connection.autoconnect-priority", str(value), check=False,
        )
        return result.returncode == 0

    def forget(self, profile_uuid: str) -> bool:
        profiles = {profile.uuid: profile for profile in self.list_saved()}
        profile = profiles.get(profile_uuid)
        if profile is None or profile.active:
            return False
        result = self._nmcli(
            "connection", "delete", "uuid", profile_uuid, check=False
        )
        return result.returncode == 0
