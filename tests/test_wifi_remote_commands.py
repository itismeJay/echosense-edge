"""Dashboard Wi-Fi commands applied on the box, with NetworkManager mocked."""

import unittest
from unittest import mock

from provisioning.wifi_setup.networkmanager import (
    CommandResult,
    ConnectionStatus,
    NetworkManager,
    NetworkManagerError,
    SavedNetwork,
    WifiNetwork,
)
from provisioning.wifi_setup.remote_commands import NOT_SUPPORTED, WifiCommandRunner

HOME = "11111111-1234-4abc-8def-123456789abc"
SCHOOL = "22222222-1234-4abc-8def-123456789abc"
SECRET = "Sup3r-Secret-PSK"


def connected(name, uuid):
    return ConnectionStatus("100 (connected)", name, uuid, "10.0.0.5", ssid=name)


class FakeResponse:
    def __init__(self, status_code=200, body=None):
        self.status_code = status_code
        self._body = body or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, commands):
        self.commands = commands
        self.posts = []

    def get(self, url, **_kwargs):
        return FakeResponse(200, {"commands": self.commands})

    def post(self, url, json=None, **_kwargs):
        self.posts.append((url, json))
        return FakeResponse(200)


def fake_manager(saved, current=None, visible=()):
    manager = mock.Mock()
    manager.list_saved.return_value = saved
    manager.profiles_for_ssid.side_effect = lambda ssid: [p for p in saved if p.ssid == ssid]
    manager.connection_status.return_value = current or ConnectionStatus("30 (disconnected)")
    manager.scan.return_value = [WifiNetwork(ssid, 70, "WPA2") for ssid in visible]
    manager.set_priority.return_value = True
    manager.forget.return_value = True
    return manager


class RemoteCommandTests(unittest.TestCase):
    def runner(self, manager, commands, **options):
        self.session = FakeSession(commands)
        self.logs = []
        return WifiCommandRunner(
            manager,
            api_url="http://backend",
            headers=lambda: {"X-EchoSense-Device-Id": "dev"},
            session=self.session,
            logger=self.logs.append,
            sleep=lambda _s: None,
            **options,
        )

    def test_add_saves_network_and_reports_without_logging_password(self):
        manager = fake_manager([], current=connected("Home", HOME))
        runner = self.runner(manager, [{
            "id": "c1", "action": "add", "ssid": "School", "password": SECRET, "priority": 20,
        }])
        runner.poll_once()
        manager.save_network.assert_called_once_with("School", SECRET, hidden=False, priority=20)
        url, body = self.session.posts[0]
        self.assertEqual(url, "http://backend/device/wifi/commands/c1/result")
        self.assertEqual(body, {"status": "applied", "current_ssid": "Home"})
        self.assertNotIn(SECRET, "\n".join(self.logs))
        self.assertNotIn(SECRET, str(self.session.posts))

    def test_remove_in_use_without_reachable_alternative_is_refused(self):
        saved = [
            SavedNetwork("Home", HOME, "Home", active=True),
            SavedNetwork("School", SCHOOL, "School"),
        ]
        manager = fake_manager(saved, current=connected("Home", HOME), visible=["Home"])
        runner = self.runner(manager, [{"id": "c2", "action": "remove", "ssid": "Home"}])
        runner.poll_once()
        manager.forget.assert_not_called()
        manager.activate_saved.assert_not_called()
        self.assertEqual(self.session.posts[0][1]["error"], "in_use_no_alternative")

    def test_remove_in_use_switches_to_reachable_network_first(self):
        saved = [
            SavedNetwork("Home", HOME, "Home", active=True),
            SavedNetwork("School", SCHOOL, "School"),
        ]
        manager = fake_manager(saved, visible=["Home", "School"])
        runner = self.runner(manager, [{"id": "c3", "action": "remove", "ssid": "Home"}])
        runner.poll_once()
        calls = [c[0] for c in manager.method_calls if c[0] in ("activate_saved", "forget")]
        self.assertEqual(calls, ["activate_saved", "forget"])
        manager.activate_saved.assert_called_once_with(SCHOOL)
        manager.forget.assert_called_once_with(HOME)
        self.assertEqual(self.session.posts[0][1]["status"], "applied")

    def test_set_priority(self):
        manager = fake_manager([SavedNetwork("School", SCHOOL, "School")])
        runner = self.runner(manager, [
            {"id": "c4", "action": "set_priority", "ssid": "School", "priority": 50},
            {"id": "c5", "action": "set_priority", "ssid": "School", "priority": 5000},
        ])
        runner.poll_once()
        manager.set_priority.assert_called_once_with(SCHOOL, 50)
        self.assertEqual(self.session.posts[0][1]["status"], "applied")
        self.assertEqual(self.session.posts[1][1]["error"], "invalid_priority")

    def test_connect_now_rolls_back_when_backend_unreachable_for_120s(self):
        saved = [
            SavedNetwork("Home", HOME, "Home", active=True),
            SavedNetwork("School", SCHOOL, "School"),
        ]
        manager = fake_manager(saved, current=connected("Home", HOME))
        now = [0.0]

        def sleep(seconds):
            now[0] += seconds

        runner = self.runner(
            manager,
            [{"id": "c6", "action": "connect_now", "ssid": "School"}],
            clock=lambda: now[0],
            backend_reachable=lambda: False,
        )
        runner.sleep = sleep
        runner.poll_once()
        self.assertEqual(
            [c.args[0] for c in manager.activate_saved.call_args_list], [SCHOOL, HOME]
        )
        self.assertGreaterEqual(now[0], 120)
        body = self.session.posts[0][1]
        self.assertEqual(body["status"], "failed")
        self.assertEqual(body["error"], "backend_unreachable_rolled_back")
        self.assertEqual(body["current_ssid"], "Home")

    def test_connect_now_keeps_new_network_when_backend_answers(self):
        saved = [
            SavedNetwork("Home", HOME, "Home", active=True),
            SavedNetwork("School", SCHOOL, "School"),
        ]
        manager = fake_manager(saved, current=connected("Home", HOME))
        runner = self.runner(
            manager,
            [{"id": "c7", "action": "connect_now", "ssid": "School"}],
            backend_reachable=lambda: True,
        )
        runner.poll_once()
        manager.activate_saved.assert_called_once_with(SCHOOL)
        self.assertEqual(self.session.posts[0][1]["status"], "applied")

    def test_failed_report_is_retried_without_reapplying(self):
        manager = fake_manager([SavedNetwork("School", SCHOOL, "School")])
        runner = self.runner(manager, [
            {"id": "c8", "action": "set_priority", "ssid": "School", "priority": 3},
        ])
        self.session.post = mock.Mock(side_effect=[RuntimeError("offline"), FakeResponse(200)])
        runner.poll_once()
        runner.poll_once()
        manager.set_priority.assert_called_once()
        self.assertEqual(self.session.post.call_count, 2)

    def test_scan_reports_known_deduped_strongest_first(self):
        saved = [SavedNetwork("School", SCHOOL, "School", active=True)]
        manager = fake_manager(saved, current=connected("School", SCHOOL))
        manager.scan.return_value = [
            WifiNetwork("Cafe", 40, "--"),
            WifiNetwork("School", 55, "WPA2"),
            WifiNetwork("", 99, "WPA2"),
            WifiNetwork("School", 80, "WPA2"),
            WifiNetwork("Lab", 80, "WPA1 WPA2"),
        ]
        runner = self.runner(manager, [{"id": "s1", "action": "scan"}])
        runner.poll_once()
        url, body = self.session.posts[0]
        self.assertTrue(url.endswith("/device/wifi/commands/s1/result"))
        self.assertEqual(body["status"], "applied")
        self.assertEqual(body["current_ssid"], "School")
        self.assertEqual(body["networks"], [
            {"ssid": "Lab", "signal": 80, "security": "WPA1 WPA2", "known": False},
            {"ssid": "School", "signal": 80, "security": "WPA2", "known": True},
            {"ssid": "Cafe", "signal": 40, "security": "open", "known": False},
        ])
        manager.rescan.assert_called_once()
        manager.scan.assert_called_once_with(rescan=False)
        manager.activate_saved.assert_not_called()
        manager.forget.assert_not_called()

    def test_scan_caps_at_fifty(self):
        manager = fake_manager([])
        manager.scan.return_value = [WifiNetwork(f"N{i:03}", i % 101, "WPA2") for i in range(80)]
        runner = self.runner(manager, [{"id": "s2", "action": "scan"}])
        runner.poll_once()
        networks = self.session.posts[0][1]["networks"]
        self.assertEqual(len(networks), 50)
        self.assertEqual(networks[0]["signal"], 79)

    def test_scan_failure_is_reported(self):
        manager = fake_manager([])
        manager.scan.side_effect = NetworkManagerError("nmcli_exit_10")
        runner = self.runner(manager, [{"id": "s3", "action": "scan"}])
        runner.poll_once()
        self.assertEqual(self.session.posts[0][1], {"status": "failed", "error": "nmcli_exit_10"})

    def test_mac_scan_reports_not_supported(self):
        runner = self.runner(None, [{"id": "s4", "action": "scan"}])
        runner.poll_once()
        self.assertEqual(
            self.session.posts[0][1], {"status": "failed", "error": NOT_SUPPORTED}
        )

    def test_mac_without_networkmanager_reports_not_supported(self):
        runner = self.runner(None, [
            {"id": "c9", "action": "add", "ssid": "School", "password": SECRET},
        ])
        runner.poll_once()
        self.assertTrue(any("wifi command received (simulated)" in l for l in self.logs))
        self.assertEqual(
            self.session.posts[0][1], {"status": "failed", "error": NOT_SUPPORTED}
        )
        self.assertNotIn(SECRET, "\n".join(self.logs))


class SaveNetworkAdapterTests(unittest.TestCase):
    def test_password_goes_to_stdin_never_argv(self):
        calls = []

        def runner(args, *, timeout=30, input_text=None):
            calls.append((list(args), input_text))
            if "-g" in args and "connection.uuid" in args:
                return CommandResult(0, SCHOOL + "\n")
            return CommandResult(0, "")

        manager = NetworkManager("wlan0", runtime_directory="/nonexistent", runner=runner)
        manager.list_saved = mock.Mock(return_value=[
            SavedNetwork("Home", HOME, "Home", priority=4),
        ])
        self.assertEqual(manager.save_network("School", SECRET), SCHOOL)
        self.assertFalse(any(SECRET in " ".join(args) for args, _ in calls))
        self.assertTrue(any(stdin and SECRET in stdin for _, stdin in calls))
        final = calls[-1][0]
        self.assertEqual(final[-4:], [
            "connection.autoconnect", "yes", "connection.autoconnect-priority", "5",
        ])

    def test_rescan_then_list_without_rescan_stays_connected(self):
        calls = []

        def runner(args, *, timeout=30, input_text=None):
            calls.append(list(args))
            if "rescan" in args and "list" not in args:
                return CommandResult(10, "", "Scanning not allowed while already scanning")
            return CommandResult(0, "*:School:80:WPA2\n:Ca\\:fe:40:\n::90:WPA2\n: :50:WPA2\n")

        manager = NetworkManager("wlan0", runtime_directory="/nonexistent", runner=runner)
        self.assertFalse(manager.rescan())
        networks = manager.scan(rescan=False)
        self.assertEqual(calls[0][-4:], ["wifi", "rescan", "ifname", "wlan0"])
        self.assertIn("no", calls[1][calls[1].index("--rescan") + 1:][:1])
        self.assertEqual([(n.ssid, n.signal, n.secured) for n in networks],
                         [("School", 80, True), ("Ca:fe", 40, False)])
        self.assertFalse(any(a in ("connect", "up", "down", "disconnect") for c in calls for a in c))


if __name__ == "__main__":
    unittest.main()
