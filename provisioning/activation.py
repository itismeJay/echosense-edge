"""First start of a QR-set-up box: wait until an admin adds it to a classroom.

The box polls POST /devices/activate with its identity every few seconds. While it
waits it says "Waiting to be added to a classroom" (once, then every few minutes).
When an admin claims it, the backend returns the device key once; the box saves it
owner-only, says "EchoSense connected" and starts normally. Secrets are never printed.
"""

from __future__ import annotations

import shutil
import subprocess
import time

import requests

from provisioning.identity import IdentityError, load_identity, save_device_key


def print(*args, **kwargs):  # noqa: A001 - flushed: main.py exec()s right after activation
    import builtins

    kwargs.setdefault("flush", True)
    builtins.print(*args, **kwargs)

WAITING_MESSAGE = "Waiting to be added to a classroom"
CONNECTED_MESSAGE = "EchoSense connected"
REFUSED_MESSAGE = "Setup code not accepted. Ask EchoSense support to reset this box."
_SLOW_POLL_SECONDS = 60


def speak_command(text, platform, override=None, which=shutil.which):
    """The command that says ``text`` out loud, or None (then it is only printed).

    ECHOSENSE_SPEAK_COMMAND overrides (e.g. "espeak-ng -s 150"; "off" disables).
    Mac: say. Pi/Linux: espeak-ng, espeak or spd-say, whichever is installed.
    """

    if override is not None and str(override).strip():
        value = str(override).strip()
        if value.lower() in {"off", "none", "false", "0"}:
            return None
        return [*value.split(), text]
    if platform == "macos":
        return ["say", text] if which("say") else None
    for program in ("espeak-ng", "espeak", "spd-say"):
        if which(program):
            return [program, text]
    return None


class Speaker:
    """Says a status line without ever holding up activation.

    The speech command runs in the background (an audio device that is busy or missing
    can make `say` hang); a newer line stops one that is still playing.
    """

    def __init__(self, platform, override=None, popen=subprocess.Popen, which=shutil.which):
        self._platform = platform
        self._override = override
        self._popen = popen
        self._which = which
        self._current = None

    def __call__(self, text):
        print(f"[VOICE] {text}")
        command = speak_command(text, self._platform, self._override, self._which)
        if command is None:
            return False
        previous = self._current
        if previous is not None and previous.poll() is None:
            previous.terminate()
        try:
            self._current = self._popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:  # a missing speaker must never stop setup
            print(f"[VOICE] could not play ({type(exc).__name__})")
            return False
        return True


def _retry_after(response, default):
    try:
        return max(float(response.headers.get("Retry-After", default)), float(default))
    except (TypeError, ValueError):
        return float(default)


def activate_until_ready(
    api_url,
    identity_path,
    credentials_path,
    *,
    speak,
    poll_seconds=10.0,
    announce_seconds=180.0,
    post=requests.post,
    sleep=time.sleep,
    clock=time.monotonic,
    max_polls=None,
):
    """Poll until the backend hands over the device key; save it and return it.

    The identity file is re-read on every poll, so copying a fresh identity.json
    (after a super-admin reset) onto a waiting box takes effect without a restart.
    Returns None only when ``max_polls`` (tests) runs out.
    """

    if not api_url:
        raise RuntimeError("ECHOSENSE_BACKEND_BASE_URL is required to activate this box")
    last_spoken = {}
    last_state = None
    polls = 0

    def announce(message, every):
        now = clock()
        previous = last_spoken.get(message)
        if previous is None or now - previous >= every:
            speak(message)
            last_spoken[message] = now

    while max_polls is None or polls < max_polls:
        polls += 1
        delay = poll_seconds
        state = None
        try:
            identity = load_identity(identity_path)
            if identity is None:
                raise IdentityError("identity file is missing")
            response = post(
                f"{api_url}/devices/activate",
                json={
                    "device_code": identity.device_code,
                    "claim_code": identity.claim_code,
                    "box_secret": identity.box_secret,
                },
                timeout=10,
            )
            status = response.status_code
            if status == 200:
                body = response.json()
                device_key = str(body.get("device_key") or "").strip()
                if body.get("status") == "active" and device_key:
                    save_device_key(credentials_path, identity.device_code, device_key)
                    room = body.get("classroom_name") or "its classroom"
                    print(
                        f"[SETUP] Activated device={identity.device_code} room={room}; "
                        "key saved (owner-only)"
                    )
                    speak(CONNECTED_MESSAGE)
                    return device_key
                state = "unexpected"
                print("[SETUP] Activation answer had no key; retrying")
            elif status == 202:
                state = "waiting"
                if last_state != "waiting":
                    print(
                        f"[SETUP] device={identity.device_code} is waiting to be added to "
                        "a classroom (Classrooms > Add device)"
                    )
                announce(WAITING_MESSAGE, announce_seconds)
            elif status == 429:
                state = "throttled"
                delay = _retry_after(response, poll_seconds)
                print(f"[SETUP] Backend asked to slow down; next try in {int(delay)}s")
            elif status in (401, 403):
                state = f"refused-{status}"
                delay = max(poll_seconds, _SLOW_POLL_SECONDS)
                if last_state != state:
                    reason = (
                        "box disabled or school inactive"
                        if status == 403
                        else "setup codes not accepted (already set up or reset)"
                    )
                    print(f"[SETUP] Backend refused activation ({status}): {reason}")
                if status == 401:
                    announce(REFUSED_MESSAGE, announce_seconds)
            else:
                state = f"http-{status}"
                print(f"[SETUP] Activation call answered {status}; retrying")
        except IdentityError as exc:
            state = "identity"
            delay = max(poll_seconds, _SLOW_POLL_SECONDS)
            if last_state != state:
                print(f"[SETUP] Identity problem: {exc}")
        except ValueError:
            state = "unexpected"
            print("[SETUP] Activation answer was not JSON; retrying")
        except requests.RequestException as exc:
            state = "network"
            if last_state != state:
                print(f"[SETUP] Backend not reachable ({type(exc).__name__}); retrying")
        last_state = state
        sleep(delay)
    return None
