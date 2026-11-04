"""QR setup on the box: identity files, credential choice, activation, labels."""

import contextlib
import io
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock

import requests

from provisioning.activation import (
    CONNECTED_MESSAGE,
    WAITING_MESSAGE,
    Speaker,
    activate_until_ready,
    speak_command,
)
from provisioning.identity import (
    BoxIdentity,
    IdentityError,
    load_device_key,
    load_identity,
    resolve_credentials,
    save_device_key,
    write_identity,
)

IDENTITY = BoxIdentity("es-7k3m", "ABCD-EFGH-JKMN", "ebs_secret-value-for-tests")
DEVICE_KEY = "edk_device-key-for-tests"
SECRETS = (IDENTITY.claim_code, IDENTITY.box_secret, DEVICE_KEY)


def _mode(path):
    return stat.S_IMODE(Path(path).stat().st_mode)


class FakeResponse:
    def __init__(self, status_code, body=None, headers=None):
        self.status_code = status_code
        self._body = body if body is not None else {}
        self.headers = headers or {}

    def json(self):
        return self._body


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds

    def __call__(self):
        return self.now


class IdentityFileTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_identity_round_trip_is_owner_only(self):
        path = write_identity(self.root / "box" / "identity.json", IDENTITY, backend_url="x")
        self.assertEqual(_mode(path), 0o600)
        self.assertEqual(_mode(path.parent), 0o700)
        self.assertEqual(load_identity(path), IDENTITY)
        self.assertNotIn(IDENTITY.box_secret, repr(load_identity(path)))

    def test_loose_permissions_are_tightened(self):
        path = write_identity(self.root / "identity.json", IDENTITY)
        os.chmod(path, 0o644)
        with contextlib.redirect_stdout(io.StringIO()):
            load_identity(path)
        self.assertEqual(_mode(path), 0o600)

    def test_missing_and_incomplete_identity(self):
        self.assertIsNone(load_identity(self.root / "absent.json"))
        broken = self.root / "identity.json"
        broken.write_text(json.dumps({"device_code": "es-7k3m"}))
        with self.assertRaises(IdentityError) as caught:
            load_identity(broken)
        self.assertIn("claim_code", str(caught.exception))

    def test_saved_key_only_for_the_same_box(self):
        path = save_device_key(self.root / "device_credentials.json", "es-7k3m", DEVICE_KEY)
        self.assertEqual(_mode(path), 0o600)
        self.assertEqual(load_device_key(path, "es-7k3m"), DEVICE_KEY)
        self.assertIsNone(load_device_key(path, "es-other"))


class CredentialChoiceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.identity = self.root / "identity.json"
        self.credentials = self.root / "device_credentials.json"

    def tearDown(self):
        self.directory.cleanup()

    def resolve(self, environ):
        return resolve_credentials(environ, self.identity, self.credentials)

    def test_without_identity_the_manual_key_flow_is_unchanged(self):
        result = self.resolve({"ECHOSENSE_DEVICE_ID": "room-1", "ECHOSENSE_DEVICE_KEY": "k"})
        self.assertEqual((result.device_id, result.device_key, result.source), ("room-1", "k", "environment"))

    def test_identity_without_key_needs_activation_and_ignores_other_device_env(self):
        write_identity(self.identity, IDENTITY)
        result = self.resolve({"ECHOSENSE_DEVICE_ID": "demo-macbook", "ECHOSENSE_DEVICE_KEY": "k"})
        self.assertEqual((result.device_id, result.device_key), ("es-7k3m", ""))
        self.assertEqual(result.source, "needs_activation")

    def test_identity_uses_saved_key(self):
        write_identity(self.identity, IDENTITY)
        save_device_key(self.credentials, "es-7k3m", DEVICE_KEY)
        result = self.resolve({})
        self.assertEqual((result.device_id, result.device_key, result.source), ("es-7k3m", DEVICE_KEY, "identity"))
        self.assertNotIn(DEVICE_KEY, repr(result))

    def test_manual_key_for_the_same_box_is_accepted(self):
        write_identity(self.identity, IDENTITY)
        result = self.resolve({"ECHOSENSE_DEVICE_ID": "ES-7K3M", "ECHOSENSE_DEVICE_KEY": "manual"})
        self.assertEqual((result.device_key, result.source), ("manual", "environment"))


class SpeakerTests(unittest.TestCase):
    def test_platform_commands(self):
        everything = lambda program: f"/usr/bin/{program}"  # noqa: E731
        nothing = lambda program: None  # noqa: E731
        self.assertEqual(speak_command("hi", "macos", which=everything), ["say", "hi"])
        self.assertEqual(speak_command("hi", "raspberrypi", which=everything), ["espeak-ng", "hi"])
        self.assertIsNone(speak_command("hi", "raspberrypi", which=nothing))
        self.assertEqual(speak_command("hi", "linux", "espeak-ng -s 140"), ["espeak-ng", "-s", "140", "hi"])
        self.assertIsNone(speak_command("hi", "macos", "off", which=everything))

    def test_speaker_never_raises(self):
        run = mock.Mock(side_effect=OSError("no audio"))
        speaker = Speaker("macos", run=run, which=lambda program: "/usr/bin/say")
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertFalse(speaker("hello"))
        self.assertIn("[VOICE] hello", output.getvalue())


class ActivationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.identity = write_identity(self.root / "identity.json", IDENTITY)
        self.credentials = self.root / "device_credentials.json"
        self.clock = FakeClock()
        self.spoken = []

    def tearDown(self):
        self.directory.cleanup()

    def run_activation(self, responses, max_polls=20, announce_seconds=180):
        post = mock.Mock(side_effect=responses)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            key = activate_until_ready(
                "http://backend",
                self.identity,
                self.credentials,
                speak=self.spoken.append,
                poll_seconds=10,
                announce_seconds=announce_seconds,
                post=post,
                sleep=self.clock.sleep,
                clock=self.clock,
                max_polls=max_polls,
            )
        for secret in SECRETS:
            self.assertNotIn(secret, output.getvalue())
        return key, post, output.getvalue()

    def test_waits_then_saves_key_and_says_connected(self):
        responses = [FakeResponse(202, {"status": "waiting"})] * 3 + [
            FakeResponse(200, {"status": "active", "device_key": DEVICE_KEY, "classroom_name": "Room 4"})
        ]
        key, post, output = self.run_activation(responses)

        self.assertEqual(key, DEVICE_KEY)
        self.assertEqual(self.spoken, [WAITING_MESSAGE, CONNECTED_MESSAGE])
        self.assertEqual(self.clock.sleeps, [10, 10, 10])
        self.assertEqual(load_device_key(self.credentials, "es-7k3m"), DEVICE_KEY)
        self.assertEqual(_mode(self.credentials), 0o600)
        url = post.call_args.args[0]
        body = post.call_args.kwargs["json"]
        self.assertEqual(url, "http://backend/devices/activate")
        self.assertEqual(body, {"device_code": "es-7k3m", "claim_code": IDENTITY.claim_code, "box_secret": IDENTITY.box_secret})
        self.assertIn("Room 4", output)

    def test_waiting_message_repeats_every_few_minutes(self):
        responses = [FakeResponse(202, {"status": "waiting"})] * 40
        key, _, _ = self.run_activation(responses, max_polls=40, announce_seconds=180)
        self.assertIsNone(key)
        # 40 polls x 10 s = 400 s: spoken at 0 s, 180 s and 360 s.
        self.assertEqual(self.spoken, [WAITING_MESSAGE] * 3)

    def test_rate_limit_refusal_and_network_errors_keep_waiting(self):
        responses = [
            FakeResponse(429, headers={"Retry-After": "45"}),
            requests.ConnectionError("down"),
            FakeResponse(401, {"detail": "Device setup code not accepted"}),
            FakeResponse(200, {"status": "active", "device_key": DEVICE_KEY}),
        ]
        key, _, output = self.run_activation(responses)
        self.assertEqual(key, DEVICE_KEY)
        self.assertEqual(self.clock.sleeps, [45, 10, 60])
        self.assertIn("not reachable", output)
        self.assertIn("not accepted", output)

    def test_replaced_identity_is_picked_up_without_restart(self):
        fresh = BoxIdentity("es-7k3m", "WXYZ-WXYZ-WXYZ", "ebs_new-secret")

        def post(url, json, timeout):
            if json["claim_code"] == fresh.claim_code:
                return FakeResponse(200, {"status": "active", "device_key": DEVICE_KEY})
            write_identity(self.identity, fresh)
            return FakeResponse(401, {})

        with contextlib.redirect_stdout(io.StringIO()):
            key = activate_until_ready(
                "http://backend", self.identity, self.credentials,
                speak=self.spoken.append, post=post, sleep=self.clock.sleep,
                clock=self.clock, max_polls=3,
            )
        self.assertEqual(key, DEVICE_KEY)


class MainActivationTests(unittest.TestCase):
    def test_restarts_only_when_activation_was_needed(self):
        import main
        from provisioning.identity import ResolvedCredentials

        activate = mock.Mock()
        restart = mock.Mock()
        ready = ResolvedCredentials("es-7k3m", DEVICE_KEY, "identity", IDENTITY)
        self.assertFalse(main.activate_if_needed(ready, activate, restart))
        activate.assert_not_called()

        waiting = ResolvedCredentials("es-7k3m", "", "needs_activation", IDENTITY)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(main.activate_if_needed(waiting, activate, restart))
        activate.assert_called_once()
        executable, argv = restart.call_args.args
        # The setup lines are flushed before exec replaces the process.
        flushed = []
        with mock.patch("sys.stdout") as stdout:
            stdout.flush.side_effect = lambda: flushed.append(restart.call_count)
            main.activate_if_needed(waiting, activate, restart)
        self.assertEqual(flushed[-1], 1)
        self.assertEqual(argv[0], executable)
        self.assertTrue(os.path.isabs(argv[1]))


try:
    import qrcode  # noqa: F401
except ImportError:  # the box itself does not need the label tool's dependency
    qrcode = None


@unittest.skipIf(qrcode is None, "qrcode not installed (tools/requirements-provision.txt)")
class ProvisionToolTests(unittest.TestCase):
    BOX = {
        "device_code": "es-7k3m",
        "claim_code": IDENTITY.claim_code,
        "box_secret": IDENTITY.box_secret,
        "qr_payload": f"echosense://claim?d=es-7k3m&c={IDENTITY.claim_code}",
        "device": {"id": "00000000-0000-0000-0000-000000000001"},
    }

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def session(self, provision_status=201, is_super_admin=True):
        session = mock.Mock()
        session.headers = {}
        session.post.side_effect = [
            FakeResponse(200, {"access_token": "token", "must_change_password": False,
                               "user": {"is_super_admin": is_super_admin}}),
            FakeResponse(provision_status, {"boxes": [self.BOX]}),
        ]
        return session

    def test_writes_identity_and_labels(self):
        from PIL import Image
        from tools import provision_box

        identity_copy = self.root / "box-data" / "identity.json"
        session = self.session()
        with contextlib.redirect_stdout(io.StringIO()) as output:
            code = provision_box.main(
                ["--email", "admin@echosense.app", "--out-dir", str(self.root / "labels"),
                 "--identity", str(identity_copy), "--backend", "http://backend/"],
                session=session, password_prompt=lambda prompt: "password",
            )
        self.assertEqual(code, 0)
        folder = self.root / "labels" / "es-7k3m"
        for path in (folder / "identity.json", identity_copy, folder / "label.png", folder / "label.pdf"):
            self.assertTrue(path.is_file(), path)
            self.assertEqual(_mode(path), 0o600)
        self.assertEqual(load_identity(identity_copy), IDENTITY)
        self.assertEqual(json.loads(identity_copy.read_text())["backend_url"], "http://backend")
        with Image.open(folder / "label.png") as label:
            self.assertEqual(label.size, provision_box.LABEL_SIZE)
        self.assertTrue((folder / "label.pdf").read_bytes().startswith(b"%PDF"))
        login_call, provision_call = session.post.call_args_list
        self.assertEqual(provision_call.args[0], "http://backend/devices/provision")
        self.assertEqual(session.headers["Authorization"], "Bearer token")
        for secret in (IDENTITY.claim_code, IDENTITY.box_secret, "password"):
            self.assertNotIn(secret, output.getvalue())

    def test_label_qr_decodes_to_the_payload_when_a_decoder_is_available(self):
        from tools import provision_box

        image = provision_box.render_label(self.BOX["qr_payload"], "es-7k3m", IDENTITY.claim_code)
        try:
            from pyzbar.pyzbar import decode
        except ImportError:
            self.skipTest("pyzbar not installed")
        decoded = [item.data.decode() for item in decode(image)]
        self.assertIn(self.BOX["qr_payload"], decoded)

    def test_session_token_skips_sign_in_but_still_checks_super_admin(self):
        from tools import provision_box

        session = mock.Mock()
        session.headers = {}
        session.get.return_value = FakeResponse(200, {"is_super_admin": True, "must_change_password": False})
        session.post.return_value = FakeResponse(201, {"boxes": [self.BOX]})
        prompt = mock.Mock()
        with mock.patch.dict(os.environ, {"ECHOSENSE_ADMIN_TOKEN": "session-token", "ECHOSENSE_BACKEND_BASE_URL": "http://localhost:8000"}):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                code = provision_box.main(["--out-dir", str(self.root)], session=session, password_prompt=prompt)
        self.assertEqual(code, 0)
        prompt.assert_not_called()
        self.assertEqual(session.get.call_args.args[0], "http://localhost:8000/auth/me")
        self.assertEqual(session.post.call_args.args[0], "http://localhost:8000/devices/provision")
        self.assertNotIn("session-token", output.getvalue())

        session.get.return_value = FakeResponse(200, {"is_super_admin": False})
        with mock.patch.dict(os.environ, {"ECHOSENSE_ADMIN_TOKEN": "session-token", "ECHOSENSE_BACKEND_BASE_URL": "http://localhost:8000"}):
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(provision_box.main(["--out-dir", str(self.root / "x")], session=session), 1)

    def test_refuses_non_super_admin(self):
        from tools import provision_box

        with contextlib.redirect_stderr(io.StringIO()) as errors:
            code = provision_box.main(
                ["--email", "school-admin@example.test", "--out-dir", str(self.root)],
                session=self.session(is_super_admin=False), password_prompt=lambda prompt: "x",
            )
        self.assertEqual(code, 1)
        self.assertIn("super admin", errors.getvalue())
        self.assertFalse(any(self.root.iterdir()))


if __name__ == "__main__":
    unittest.main()
