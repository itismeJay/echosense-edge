"""Wi-Fi setup validation, NetworkManager boundaries, and recovery behavior."""

import contextlib
import http.client
import io
import os
from pathlib import Path
import re
import stat
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock
from urllib.parse import urlencode

import main as edge_main

from provisioning.wifi_setup.controller import (
    CONNECTED,
    FAILED,
    HOTSPOT,
    ProvisioningController,
)
from provisioning.wifi_setup.credentials import (
    create_setup_credentials,
    load_setup_credentials,
    save_setup_credentials,
)
from provisioning.wifi_setup.networkmanager import (
    CommandResult,
    ConnectionStatus,
    HOTSPOT_PROFILE,
    NetworkManager,
    NetworkManagerError,
    SavedNetwork,
    WifiNetwork,
    _split_escaped,
)
from provisioning.wifi_setup.portal import PortalSecurity, make_portal_server
from provisioning.wifi_setup.service import WifiSetupService
from provisioning.wifi_setup.validation import (
    WifiInputError,
    validate_profile_uuid,
    validate_ssid,
    validate_wifi_password,
)


def _mode(path):
    return stat.S_IMODE(Path(path).stat().st_mode)


class ValidationTests(unittest.TestCase):
    def test_ssid_accepts_utf8_up_to_32_bytes(self):
        self.assertEqual(validate_ssid("Grade 6 – A"), "Grade 6 – A")
        self.assertEqual(validate_ssid("a" * 32), "a" * 32)

    def test_ssid_rejects_blank_outer_spaces_control_and_oversize(self):
        for value in ("", " school", "school ", "bad\nssid", "é" * 17):
            with self.subTest(value=value), self.assertRaises(WifiInputError):
                validate_ssid(value)

    def test_password_rules_cover_wpa_hex_and_open_networks(self):
        self.assertEqual(validate_wifi_password("password123"), "password123")
        self.assertEqual(validate_wifi_password("a" * 64), "a" * 64)
        self.assertEqual(validate_wifi_password("", open_network=True), "")
        for value in ("short", "g" * 64, "line\nbreak"):
            with self.subTest(value=value), self.assertRaises(WifiInputError):
                validate_wifi_password(value)
        with self.assertRaises(WifiInputError):
            validate_wifi_password("unexpected", open_network=True)

    def test_profile_uuid_is_strict(self):
        value = "12345678-1234-4abc-8def-123456789abc"
        self.assertEqual(validate_profile_uuid(value.upper()), value)
        with self.assertRaises(WifiInputError):
            validate_profile_uuid("../../profile")


class CredentialTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "private" / "wifi.json"

    def tearDown(self):
        self.directory.cleanup()

    def test_credentials_are_unique_owner_only_and_repr_hides_secrets(self):
        first, password = create_setup_credentials("machine-a", "aa:bb:cc:dd:ee:ff")
        second, _ = create_setup_credentials("machine-a", "aa:bb:cc:dd:ee:ff")
        self.assertRegex(first.hotspot_ssid, r"^EchoSense-Setup-[0-9A-F]{4}$")
        self.assertNotEqual(first.hotspot_password, second.hotspot_password)
        self.assertNotIn(first.hotspot_password, repr(first))
        self.assertNotIn(password, repr(first))
        self.assertTrue(first.verify_admin_password(password))
        self.assertFalse(first.verify_admin_password("wrong-password"))

        save_setup_credentials(self.path, first)
        self.assertEqual(_mode(self.path.parent), 0o700)
        self.assertEqual(_mode(self.path), 0o600)
        self.assertEqual(load_setup_credentials(self.path), first)


class PortalSecurityTests(unittest.TestCase):
    def setUp(self):
        self.credentials, self.password = create_setup_credentials("m", "mac")

    def test_login_uses_expiring_sessions(self):
        now = [10.0]
        security = PortalSecurity(self.credentials, clock=lambda: now[0])
        self.assertIsNone(security.authenticate("10.42.0.2", "wrong"))
        token = security.authenticate("10.42.0.2", self.password)
        self.assertIsNotNone(token)
        self.assertIsNotNone(security.session(token))
        now[0] += 901
        self.assertIsNone(security.session(token))

    def test_repeated_failures_lock_one_client_without_exposing_password(self):
        now = [10.0]
        security = PortalSecurity(self.credentials, clock=lambda: now[0])
        for _ in range(5):
            self.assertIsNone(security.authenticate("10.42.0.2", "wrong"))
        self.assertIsNone(security.authenticate("10.42.0.2", self.password))
        now[0] += 61
        self.assertIsNotNone(security.authenticate("10.42.0.2", self.password))


class ParsingTests(unittest.TestCase):
    def test_nmcli_escaped_delimiters_are_parsed(self):
        self.assertEqual(
            _split_escaped(r"*:School\: Staff:91:WPA2"),
            ["*", "School: Staff", "91", "WPA2"],
        )


class DashboardWifiStatusTests(unittest.TestCase):
    def test_heartbeat_describes_local_only_wifi_management(self):
        with (
            mock.patch.object(edge_main, "get_ip", return_value="192.168.1.20"),
            mock.patch.object(edge_main, "get_ssid", return_value="School"),
            mock.patch.object(edge_main, "get_mac", return_value="aa:bb:cc:dd:ee:ff"),
            mock.patch.object(edge_main, "_remote_scan_supported", return_value=False),
        ):
            info = edge_main.get_network_info()
        self.assertEqual(info["wifi_status"], "connected")
        self.assertEqual(info["ssid"], "School")
        self.assertEqual(info["wifi_local_url"], "http://192.168.1.20/")
        self.assertFalse(info["wifi_remote_scan_supported"])

    def test_offline_guidance_uses_setup_hotspot_portal_address(self):
        with (
            mock.patch.object(edge_main, "get_ip", return_value="unknown"),
            mock.patch.object(edge_main, "get_ssid", return_value="unknown"),
            mock.patch.object(edge_main, "get_mac", return_value="unknown"),
        ):
            info = edge_main.get_network_info()
        self.assertEqual(info["wifi_status"], "disconnected")
        self.assertEqual(info["wifi_local_url"], "http://10.42.0.1/")


class RecordingRunner:
    def __init__(self):
        self.calls = []
        self.secret_file_contents = []
        self.fail_connection = False
        self.profile_name = ""

    def __call__(self, args, *, timeout=30, input_text=None):
        self.calls.append(list(args))
        if "con-name" in args:
            self.profile_name = args[args.index("con-name") + 1]
        if "passwd-file" in args:
            path = args[args.index("passwd-file") + 1]
            self.secret_file_contents.append(Path(path).read_text(encoding="utf-8"))
            if self.fail_connection and HOTSPOT_PROFILE not in args:
                return CommandResult(4, "", "activation failed")
        if "device" in args and "show" in args:
            return CommandResult(
                0,
                "GENERAL.STATE:100 (connected)\n"
                f"GENERAL.CONNECTION:{self.profile_name}\n"
                "GENERAL.CON-UUID:12345678-1234-4abc-8def-123456789abc\n"
                "IP4.ADDRESS[1]:192.168.10.8/24\n",
            )
        return CommandResult(0, "", "")


class NetworkManagerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.runner = RecordingRunner()
        self.manager = NetworkManager(
            "wlan0",
            runtime_directory=self.directory.name,
            runner=self.runner,
            sleep=lambda _seconds: None,
        )

    def tearDown(self):
        self.directory.cleanup()

    def test_scan_deduplicates_and_sorts_without_showing_hidden_entries(self):
        def runner(args, **_kwargs):
            return CommandResult(
                0,
                "*:School\\: Staff:73:WPA2\n:Guest:35:--\n:Guest:80:--\n::99:WPA2\n",
            )

        manager = NetworkManager("wlan0", runner=runner)
        networks = manager.scan()
        self.assertEqual([item.ssid for item in networks], ["Guest", "School: Staff"])
        self.assertEqual(networks[0].signal, 80)
        self.assertFalse(networks[0].secured)

    def test_preflight_is_read_only_and_requires_pi64_os_nm_and_ap_mode(self):
        calls = []

        def runner(args, **_kwargs):
            calls.append(args)
            joined = " ".join(args)
            if "--version" in args:
                return CommandResult(0, "nmcli tool, version 1.42.4\n")
            if "RUNNING" in args:
                return CommandResult(0, "running\n")
            if "GENERAL.TYPE" in args:
                return CommandResult(0, "GENERAL.TYPE:wifi\n")
            if "WIFI-PROPERTIES.AP" in args:
                return CommandResult(0, "WIFI-PROPERTIES.AP:yes\n")
            return CommandResult(1, "", "unexpected")

        manager = NetworkManager("wlan0", runner=runner)
        os_release = 'ID=debian\nVERSION_CODENAME=bookworm\nPRETTY_NAME="Debian GNU/Linux 12"\n'
        with (
            mock.patch.object(Path, "read_text", return_value=os_release),
            mock.patch.object(Path, "read_bytes", return_value=b"Raspberry Pi 5 Model B Rev 1.0\0"),
            mock.patch(
                "provisioning.wifi_setup.networkmanager.os.uname",
                return_value=SimpleNamespace(machine="aarch64"),
            ),
        ):
            report = manager.preflight()
        self.assertTrue(report["supported"])
        self.assertEqual(report["radio_strategy"], "single_interface_safe_transitions")
        all_arguments = " ".join(part for call in calls for part in call)
        self.assertNotIn(" connection ", f" {all_arguments} ")
        self.assertNotIn(" up ", f" {all_arguments} ")
        self.assertNotIn(" down ", f" {all_arguments} ")

    def test_hotspot_secret_uses_volatile_file_never_argv_and_is_unlinked(self):
        secret = "HOTSPOT-SECRET-123"
        self.manager.start_hotspot("EchoSense-Setup-1234", secret)
        flattened = " ".join(part for call in self.runner.calls for part in call)
        self.assertNotIn(secret, flattened)
        self.assertEqual(
            self.runner.secret_file_contents,
            [f"802-11-wireless-security.psk:{secret}\n"],
        )
        self.assertEqual(list(Path(self.directory.name).glob("nm-secret-*")), [])

    def test_client_secret_is_not_in_argv_and_verified_profile_autoconnects(self):
        secret = "CLIENT-SECRET-123"
        status = self.manager.connect("School WiFi", secret)
        self.assertTrue(status.client_connected)
        flattened = " ".join(part for call in self.runner.calls for part in call)
        self.assertNotIn(secret, flattened)
        self.assertTrue(any("connection.autoconnect-priority" in call for call in self.runner.calls))
        self.assertEqual(list(Path(self.directory.name).glob("nm-secret-*")), [])

    def test_failed_candidate_is_deleted_and_secret_file_removed(self):
        self.runner.fail_connection = True
        with self.assertRaises(NetworkManagerError):
            self.manager.connect("School WiFi", "CLIENT-SECRET-123")
        self.assertTrue(
            any("delete" in call and "EchoSense WiFi" in " ".join(call) for call in self.runner.calls)
        )
        self.assertEqual(list(Path(self.directory.name).glob("nm-secret-*")), [])

    def test_startup_reconciliation_only_changes_interrupted_managed_profiles(self):
        active_uuid = "12345678-1234-4abc-8def-123456789abc"
        stale_uuid = "22345678-1234-4abc-8def-123456789abc"
        unrelated_uuid = "32345678-1234-4abc-8def-123456789abc"
        self.manager.list_saved = mock.Mock(return_value=[
            SavedNetwork("EchoSense WiFi active", active_uuid, "A", True, False),
            SavedNetwork("EchoSense WiFi stale", stale_uuid, "B", False, False),
            SavedNetwork("School admin profile", unrelated_uuid, "C", False, False),
        ])
        result = self.manager.reconcile_managed_profiles()
        self.assertEqual(result, {"finalized": 1, "removed": 1})
        flattened_calls = [" ".join(call) for call in self.runner.calls]
        self.assertTrue(any(active_uuid in call and "modify" in call for call in flattened_calls))
        self.assertTrue(any(stale_uuid in call and "delete" in call for call in flattened_calls))
        self.assertFalse(any(unrelated_uuid in call for call in flattened_calls))

    def test_saved_profiles_use_visible_networks_and_configured_priority(self):
        low_uuid = "12345678-1234-4abc-8def-123456789abc"
        high_uuid = "22345678-1234-4abc-8def-123456789abc"
        disconnected = ConnectionStatus("30 (disconnected)")
        connected = ConnectionStatus(
            "100 (connected)", "high", high_uuid, "10.0.0.2"
        )
        self.manager.connection_status = mock.Mock(return_value=disconnected)
        self.manager.list_saved = mock.Mock(return_value=[
            SavedNetwork("low", low_uuid, "School", priority=10),
            SavedNetwork("high", high_uuid, "School", priority=80),
            SavedNetwork(
                "away",
                "32345678-1234-4abc-8def-123456789abc",
                "Unavailable",
                priority=100,
            ),
        ])
        self.manager.scan = mock.Mock(return_value=[
            WifiNetwork("School", 75, "WPA2")
        ])
        self.manager.activate_saved = mock.Mock(return_value=connected)
        result = self.manager.connect_saved_profiles(retries=2)
        self.assertEqual(result, connected)
        self.manager.activate_saved.assert_called_once_with(high_uuid, timeout=20)

    def test_new_network_gets_highest_priority_and_priority_is_settable(self):
        old_uuid = "22345678-1234-4abc-8def-123456789abc"
        saved = [SavedNetwork("Home", old_uuid, "Home", priority=7)]
        self.manager.list_saved = mock.Mock(return_value=saved)
        self.manager.connect("School WiFi", "CLIENT-SECRET-123")
        final = [c for c in self.runner.calls if "connection.autoconnect-priority" in c][-1]
        self.assertEqual(final[final.index("connection.autoconnect-priority") + 1], "8")

        self.runner.calls.clear()
        self.assertTrue(self.manager.set_priority(old_uuid, 5000))
        self.assertEqual(self.runner.calls[-1][-5:], [
            "modify", "uuid", old_uuid, "connection.autoconnect-priority", "999"
        ])
        self.assertFalse(self.manager.set_priority(
            "92345678-1234-4abc-8def-123456789abc", 1
        ))

    def test_live_client_is_never_switched_during_saved_profile_recovery(self):
        connected = ConnectionStatus(
            "100 (connected)", "current", "current-uuid", "10.0.0.2"
        )
        self.manager.connection_status = mock.Mock(return_value=connected)
        self.manager.list_saved = mock.Mock()
        self.assertEqual(self.manager.connect_saved_profiles(), connected)
        self.manager.list_saved.assert_not_called()


class FakeManager:
    def __init__(self):
        self.actions = []
        self.connect_error = None
        self.connected = False
        self.networks = [WifiNetwork("School", 85, "WPA2")]
        self.saved = []
        self.saved_recovery_status = None
        self.activate_saved_error = None

    def scan(self):
        self.actions.append("scan")
        return self.networks

    def start_hotspot(self, ssid, password):
        self.actions.append(("hotspot", ssid, password))

    def stop_hotspot(self):
        self.actions.append("stop_hotspot")

    def connect(self, ssid, password, **options):
        self.actions.append(("connect", ssid, password, options))
        if self.connect_error:
            raise self.connect_error
        self.connected = True
        return ConnectionStatus("100 (connected)", "client", "uuid", "10.0.0.2")

    def connection_status(self):
        if self.connected:
            return ConnectionStatus("100 (connected)", "client", "uuid", "10.0.0.2")
        return ConnectionStatus("100 (connected)", HOTSPOT_PROFILE, "hotspot", "10.42.0.1", True)

    def list_saved(self):
        return self.saved

    def forget(self, profile_uuid):
        self.actions.append(("forget", profile_uuid))
        return True

    def set_priority(self, profile_uuid, priority):
        self.actions.append(("priority", profile_uuid, priority))
        return True

    def connect_saved_profiles(self, **options):
        self.actions.append(("connect_saved", options))
        if self.saved_recovery_status is not None:
            return self.saved_recovery_status
        return ConnectionStatus("30 (disconnected)")

    def activate_saved(self, profile_uuid):
        self.actions.append(("activate_saved", profile_uuid))
        if self.activate_saved_error:
            raise self.activate_saved_error
        self.connected = True
        return ConnectionStatus(
            "100 (connected)", "saved", profile_uuid, "10.0.0.2"
        )


class ControllerRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.credentials, _ = create_setup_credentials("m", "mac")
        self.manager = FakeManager()
        self.logs = []
        self.controller = ProvisioningController(
            self.manager,
            self.credentials,
            logger=self.logs.append,
            delay=lambda _seconds: None,
        )
        self.controller.networks = list(self.manager.networks)

    def join_worker(self):
        self.controller._worker.join(timeout=2)
        self.assertFalse(self.controller.operation_in_progress)

    def test_enter_setup_scans_before_starting_single_radio_hotspot(self):
        self.controller.enter_setup()
        self.assertEqual(self.controller.state, HOTSPOT)
        self.assertEqual(self.manager.actions[0], "scan")
        self.assertEqual(self.manager.actions[1][0], "hotspot")

    def test_success_stops_hotspot_connects_once_and_disables_setup_state(self):
        self.controller.request_connection(
            "School", "valid-password", confirmed=True
        )
        self.join_worker()
        self.assertEqual(self.controller.state, CONNECTED)
        self.assertEqual(self.manager.actions[0], "stop_hotspot")
        connect_count = sum(
            action[0] == "connect"
            for action in self.manager.actions
            if isinstance(action, tuple)
        )
        self.assertEqual(connect_count, 1)

    def test_failure_restores_setup_hotspot_without_endless_retry(self):
        self.manager.connect_error = NetworkManagerError("bad_credentials")
        self.controller.request_connection(
            "School", "wrong-password", confirmed=True
        )
        self.join_worker()
        self.assertEqual(self.controller.state, FAILED)
        connect_actions = [
            action for action in self.manager.actions
            if isinstance(action, tuple) and action[0] == "connect"
        ]
        self.assertEqual(len(connect_actions), 1)
        self.assertEqual(self.manager.actions[-1][0], "hotspot")
        all_logs = "\n".join(self.logs)
        self.assertNotIn("wrong-password", all_logs)
        self.assertNotIn("School", all_logs)

    def test_refresh_stops_hotspot_scans_then_restores_it(self):
        self.controller.request_refresh()
        self.join_worker()
        self.assertEqual(
            [action if isinstance(action, str) else action[0] for action in self.manager.actions],
            ["stop_hotspot", "scan", "hotspot"],
        )
        self.assertEqual(self.controller.state, HOTSPOT)

    def test_refresh_on_client_lan_scans_without_starting_a_hotspot(self):
        self.manager.connected = True
        self.controller.request_refresh()
        self.join_worker()
        self.assertEqual(self.manager.actions, ["scan"])
        self.assertEqual(self.controller.state, CONNECTED)

    def test_forget_requires_inactive_known_profile_at_adapter_boundary(self):
        profile_uuid = "12345678-1234-4abc-8def-123456789abc"
        self.assertTrue(self.controller.forget(profile_uuid, confirmed=True))
        self.assertEqual(self.manager.actions[-1], ("forget", profile_uuid))

    def test_connect_and_forget_require_explicit_confirmation(self):
        with self.assertRaises(WifiInputError):
            self.controller.request_connection("School", "valid-password")
        with self.assertRaises(WifiInputError):
            self.controller.forget("12345678-1234-4abc-8def-123456789abc")
        with self.assertRaises(WifiInputError):
            self.controller.request_saved_connection(
                "12345678-1234-4abc-8def-123456789abc"
            )

    def test_confirmed_switch_activates_a_saved_profile(self):
        profile_uuid = "12345678-1234-4abc-8def-123456789abc"
        self.controller.request_saved_connection(profile_uuid, confirmed=True)
        self.join_worker()
        self.assertEqual(self.controller.state, CONNECTED)
        self.assertEqual(
            self.manager.actions,
            ["stop_hotspot", ("activate_saved", profile_uuid)],
        )

    def test_failed_change_restores_a_saved_client_before_hotspot(self):
        self.manager.connect_error = NetworkManagerError("bad_credentials")
        self.manager.saved_recovery_status = ConnectionStatus(
            "100 (connected)", "known", "known-uuid", "10.0.0.3"
        )
        self.controller.request_connection(
            "School", "wrong-password", confirmed=True
        )
        self.join_worker()
        self.assertEqual(self.controller.state, CONNECTED)
        self.assertEqual(self.manager.actions[-1][0], "connect_saved")
        self.assertFalse(any(
            isinstance(action, tuple) and action[0] == "hotspot"
            for action in self.manager.actions
        ))


class HotspotRejoinTests(unittest.TestCase):
    UUID = "12345678-1234-4abc-8def-123456789abc"

    def setUp(self):
        self.credentials, _ = create_setup_credentials("m", "mac")
        self.manager = FakeManager()
        self.logs = []
        self.controller = ProvisioningController(
            self.manager, self.credentials,
            logger=self.logs.append, delay=lambda _seconds: None,
        )

    def names(self):
        return [a if isinstance(a, str) else a[0] for a in self.manager.actions]

    def test_saved_network_in_range_stops_hotspot_and_connects(self):
        self.manager.saved = [SavedNetwork("School", self.UUID, "School")]
        self.manager.saved_recovery_status = ConnectionStatus(
            "100 (connected)", "School", self.UUID, "10.0.0.5", ssid="School"
        )
        self.assertTrue(self.controller.rejoin_saved_from_hotspot())
        self.assertEqual(self.names(), ["stop_hotspot", "scan", "connect_saved"])
        self.assertEqual(self.controller.state, CONNECTED)
        self.assertTrue(any("source=hotspot_rejoin" in line for line in self.logs))
        self.assertNotIn("School", "\n".join(self.logs))

    def test_no_saved_network_in_range_restores_hotspot(self):
        self.manager.saved = [SavedNetwork("Home", self.UUID, "Home")]
        self.assertFalse(self.controller.rejoin_saved_from_hotspot())
        self.assertEqual(self.names(), ["stop_hotspot", "scan", "hotspot"])
        self.assertEqual(self.controller.state, HOTSPOT)

    def test_failed_rejoin_restores_hotspot_and_no_saved_means_no_scan(self):
        self.manager.saved = [SavedNetwork("School", self.UUID, "School")]
        self.assertFalse(self.controller.rejoin_saved_from_hotspot())
        self.assertEqual(self.names()[-1], "hotspot")
        self.manager.actions.clear()
        self.manager.saved = []
        self.assertFalse(self.controller.rejoin_saved_from_hotspot())
        self.assertEqual(self.manager.actions, [])

    def test_priority_is_validated(self):
        self.assertTrue(self.controller.set_priority(self.UUID, "12"))
        self.assertEqual(self.manager.actions[-1], ("priority", self.UUID, 12))
        for bad in ("high", "1000", ""):
            with self.subTest(bad=bad), self.assertRaises(WifiInputError):
                self.controller.set_priority(self.UUID, bad)


class ServiceHotspotScanTests(unittest.TestCase):
    def make(self, in_use):
        credentials, _ = create_setup_credentials("m", "mac")
        self.manager = FakeManager()
        self.now = [0.0]

        class FakePortal:
            def serve_forever(self, **_kwargs):
                return

            def shutdown(self):
                return

            def server_close(self):
                return

            def in_use(self):
                return in_use

        service = WifiSetupService(
            self.manager, credentials,
            portal_factory=lambda *_a, **_k: FakePortal(),
            logger=lambda _line: None, sleep=lambda _seconds: None,
            clock=lambda: self.now[0],
        )
        service.controller.rejoin_saved_from_hotspot = mock.Mock(return_value=False)
        service._start_portal(self.manager.connection_status())
        return service

    def test_scans_every_60_seconds_while_portal_is_idle(self):
        service = self.make(in_use=False)
        service._maybe_rejoin_saved()
        self.now[0] = 59
        service._maybe_rejoin_saved()
        service.controller.rejoin_saved_from_hotspot.assert_not_called()
        self.now[0] = 61
        service._maybe_rejoin_saved()
        service.controller.rejoin_saved_from_hotspot.assert_called_once()

    def test_skips_scan_while_a_phone_is_using_the_portal(self):
        service = self.make(in_use=True)
        service._maybe_rejoin_saved()
        self.now[0] = 120
        service._maybe_rejoin_saved()
        service.controller.rejoin_saved_from_hotspot.assert_not_called()


class CaptivePortalTests(unittest.TestCase):
    def setUp(self):
        self.credentials, self.password = create_setup_credentials("m", "mac")
        self.controller = ProvisioningController(
            FakeManager(), self.credentials,
            logger=lambda _line: None, delay=lambda _seconds: None,
        )
        self.server = make_portal_server(
            self.controller, self.credentials, bind_address="127.0.0.1",
            port=0, allowed_network="127.0.0.0/8", captive=True,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def get(self, path, host):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
        connection.request("GET", path, headers={"Host": host})
        response = connection.getresponse()
        body = response.read().decode("utf-8")
        connection.close()
        return response, body

    def test_os_connectivity_checks_redirect_to_setup_page(self):
        portal = f"http://127.0.0.1:{self.port}/"
        for host, path in (
            ("connectivitycheck.gstatic.com", "/generate_204"),
            ("captive.apple.com", "/hotspot-detect.html"),
            ("www.msftconnecttest.com", "/connecttest.txt"),
            (f"127.0.0.1:{self.port}", "/generate_204"),
        ):
            with self.subTest(host=host, path=path):
                response, _ = self.get(path, host)
                self.assertEqual(response.status, 303)
                self.assertEqual(response.getheader("Location"), portal)
        self.assertFalse(self.server.in_use())
        response, body = self.get("/", f"127.0.0.1:{self.port}")
        self.assertEqual(response.status, 200)
        self.assertIn("Administrator password", body)
        self.assertTrue(self.server.in_use())

    def test_dnsmasq_answers_every_name_with_portal_ip(self):
        conf = Path("deploy/dnsmasq-shared.d/echosense-captive.conf").read_text()
        self.assertIn("address=/#/10.42.0.1", conf)
        self.assertIn('dhcp-option=114,"http://10.42.0.1/"', conf)


class PortalRouteTests(unittest.TestCase):
    def setUp(self):
        self.credentials, self.password = create_setup_credentials("m", "mac")
        self.manager = FakeManager()
        self.controller = ProvisioningController(
            self.manager,
            self.credentials,
            logger=lambda _line: None,
            delay=lambda _seconds: None,
        )
        self.controller.networks = list(self.manager.networks)
        self.server = make_portal_server(
            self.controller,
            self.credentials,
            bind_address="127.0.0.1",
            port=0,
            allowed_network="127.0.0.0/8",
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.connection = http.client.HTTPConnection(
            "127.0.0.1", self.server.server_address[1], timeout=2
        )

    def tearDown(self):
        self.connection.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, method, path, fields=None, headers=None):
        body = urlencode(fields) if fields is not None else None
        request_headers = dict(headers or {})
        if body is not None:
            request_headers["Content-Type"] = "application/x-www-form-urlencoded"
        self.connection.request(method, path, body=body, headers=request_headers)
        response = self.connection.getresponse()
        payload = response.read().decode("utf-8")
        return response, payload

    def test_login_status_and_connect_workflow_never_echoes_password(self):
        response, login_page = self.request("GET", "/")
        self.assertEqual(response.status, 200)
        self.assertIn("Administrator password", login_page)

        response, payload = self.request(
            "POST", "/login", {"password": self.password}
        )
        self.assertEqual(response.status, 303)
        self.assertNotIn(self.password, payload)
        cookie = response.getheader("Set-Cookie").split(";", 1)[0]

        response, status_page = self.request(
            "GET", "/", headers={"Cookie": cookie}
        )
        self.assertEqual(response.status, 200)
        self.assertIn("School", status_page)
        csrf = re.search(r"name='csrf' value='([^']+)'", status_page).group(1)

        response, connecting_page = self.request(
            "POST",
            "/connect",
            {
                "csrf": csrf,
                "manual_ssid": "New School",
                "password": "valid-password",
                "confirm_change": "yes",
            },
            headers={"Cookie": cookie},
        )
        self.assertEqual(response.status, 202)
        self.assertNotIn("valid-password", connecting_page)
        self.controller._worker.join(timeout=2)
        self.assertEqual(self.controller.state, CONNECTED)

    def test_saved_network_priority_can_be_set_from_the_page(self):
        profile_uuid = "12345678-1234-4abc-8def-123456789abc"
        self.manager.saved = [SavedNetwork("School", profile_uuid, "School", priority=3)]
        response, _ = self.request("POST", "/login", {"password": self.password})
        cookie = response.getheader("Set-Cookie").split(";", 1)[0]
        _, page = self.request("GET", "/", headers={"Cookie": cookie})
        self.assertIn("name='priority'", page)
        self.assertIn("value='3'", page)
        csrf = re.search(r"name='csrf' value='([^']+)'", page).group(1)
        response, _ = self.request(
            "POST", "/priority",
            {"csrf": csrf, "uuid": profile_uuid, "priority": "50"},
            headers={"Cookie": cookie},
        )
        self.assertEqual(response.status, 303)
        self.assertEqual(self.manager.actions[-1], ("priority", profile_uuid, 50))


class ServiceRecoveryTests(unittest.TestCase):
    def test_saved_profiles_are_attempted_before_setup_hotspot(self):
        credentials, _ = create_setup_credentials("m", "mac")
        manager = FakeManager()
        manager.saved_recovery_status = ConnectionStatus(
            "100 (connected)", "known", "known-uuid", "10.0.0.3"
        )
        service = WifiSetupService(
            manager,
            credentials,
            logger=lambda _line: None,
            sleep=lambda _seconds: None,
        )
        service._connect_saved_or_start_setup()
        self.assertEqual(service.controller.state, CONNECTED)
        self.assertEqual(manager.actions[0][0], "connect_saved")
        self.assertFalse(any(
            isinstance(action, tuple) and action[0] == "hotspot"
            for action in manager.actions
        ))

    def test_connected_portal_binds_only_to_the_current_client_subnet(self):
        credentials, _ = create_setup_credentials("m", "mac")
        manager = FakeManager()
        manager.connected = True
        captured = {}

        class FakePortal:
            def serve_forever(self, **_kwargs):
                return

            def shutdown(self):
                return

            def server_close(self):
                return

        def portal_factory(_controller, _credentials, **options):
            captured.update(options)
            return FakePortal()

        service = WifiSetupService(
            manager,
            credentials,
            portal_factory=portal_factory,
            logger=lambda _line: None,
            sleep=lambda _seconds: None,
        )
        status = ConnectionStatus(
            "100 (connected)",
            "School",
            "known-uuid",
            "192.168.50.8",
            False,
            "School",
            "192.168.50.8/24",
        )
        service._start_portal(status)
        self.assertEqual(captured["bind_address"], "192.168.50.8")
        self.assertEqual(captured["allowed_network"], "192.168.50.0/24")
        self.assertEqual(manager.actions, ["scan"])


if __name__ == "__main__":
    unittest.main()
