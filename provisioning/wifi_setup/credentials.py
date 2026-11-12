"""Owner-only setup credentials, separate from backend device identity."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import stat

from provisioning.identity import write_private_json


_SCRYPT_N = 2 ** 14
_SCRYPT_R = 8
_SCRYPT_P = 1


@dataclass(frozen=True, slots=True)
class SetupCredentials:
    hotspot_ssid: str
    hotspot_password: str
    admin_salt: str
    admin_digest: str

    def __repr__(self) -> str:
        return f"SetupCredentials(hotspot_ssid={self.hotspot_ssid!r})"

    def verify_admin_password(self, password: object) -> bool:
        try:
            salt = base64.b64decode(self.admin_salt, validate=True)
            expected = base64.b64decode(self.admin_digest, validate=True)
            actual = hashlib.scrypt(
                str(password or "").encode("utf-8"),
                salt=salt,
                n=_SCRYPT_N,
                r=_SCRYPT_R,
                p=_SCRYPT_P,
                dklen=len(expected),
            )
        except (TypeError, ValueError):
            return False
        return hmac.compare_digest(actual, expected)


def device_suffix(machine_id: object, mac_address: object) -> str:
    seed = f"{machine_id or ''}|{mac_address or ''}".encode("utf-8")
    return hashlib.sha256(seed).hexdigest()[:4].upper()


def create_setup_credentials(machine_id: object, mac_address: object):
    """Return credentials and the one-time administrator password.

    The hotspot PSK must remain available to NetworkManager after reboot.  It
    is stored only in the root-owned setup credential file.  The administrator
    password is stored as a scrypt digest and returned once for the setup card.
    """

    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    hotspot_password = "".join(secrets.choice(alphabet) for _ in range(16))
    admin_password = "".join(secrets.choice(alphabet) for _ in range(16))
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        admin_password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=32,
    )
    credentials = SetupCredentials(
        hotspot_ssid=f"EchoSense-Setup-{device_suffix(machine_id, mac_address)}",
        hotspot_password=hotspot_password,
        admin_salt=base64.b64encode(salt).decode("ascii"),
        admin_digest=base64.b64encode(digest).decode("ascii"),
    )
    return credentials, admin_password


def save_setup_credentials(path, credentials: SetupCredentials) -> Path:
    return write_private_json(
        path,
        {
            "version": 1,
            "hotspot_ssid": credentials.hotspot_ssid,
            "hotspot_password": credentials.hotspot_password,
            "admin_salt": credentials.admin_salt,
            "admin_digest": credentials.admin_digest,
        },
    )


def load_setup_credentials(path) -> SetupCredentials:
    target = Path(path).expanduser()
    try:
        mode = stat.S_IMODE(target.stat().st_mode)
        if mode & 0o077:
            os.chmod(target, 0o600)
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError("Wi-Fi setup credentials are unavailable") from exc
    if not isinstance(data, dict) or data.get("version") != 1:
        raise RuntimeError("Wi-Fi setup credential format is unsupported")
    values = {}
    for name in (
        "hotspot_ssid",
        "hotspot_password",
        "admin_salt",
        "admin_digest",
    ):
        value = str(data.get(name) or "")
        if not value:
            raise RuntimeError(f"Wi-Fi setup credentials are missing {name}")
        values[name] = value
    return SetupCredentials(**values)
