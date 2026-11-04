"""Box identity (QR setup) and the device key the box receives at activation.

identity.json is written by tools/provision_box.py and copied onto the box:

    {"device_code": "es-7k3m", "claim_code": "ABCD-EFGH-JKMN", "box_secret": "ebs_..."}

device_credentials.json is written by the box itself when the backend hands it its
device key (once). Both files are owner-only (0600) in an owner-only folder and are
never printed. This module has no dependency on config.py, so config can use it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat
import tempfile

OWNER_ONLY_FILE = 0o600
OWNER_ONLY_DIR = 0o700


class IdentityError(ValueError):
    """identity.json is missing fields or unreadable (message never includes secrets)."""


@dataclass(frozen=True)
class BoxIdentity:
    device_code: str
    claim_code: str
    box_secret: str

    def __repr__(self) -> str:  # never show the secrets in logs or tracebacks
        return f"BoxIdentity(device_code={self.device_code!r})"


def _tighten(path: Path) -> bool:
    """Make an existing file owner-only. True when it had to be fixed."""

    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        os.chmod(path, OWNER_ONLY_FILE)
        return True
    return False


def write_private_json(path, data: dict) -> Path:
    """Atomically write ``data`` as owner-only JSON (folder created owner-only)."""

    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True, mode=OWNER_ONLY_DIR)
    descriptor, temporary = tempfile.mkstemp(prefix=".tmp-", dir=target.parent)
    try:
        os.fchmod(descriptor, OWNER_ONLY_FILE)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return target


def _read_private_json(path: Path) -> dict:
    if _tighten(path):
        print(f"[SETUP] Tightened permissions on {path.name} to owner-only")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise IdentityError(f"{path.name} is not readable JSON") from exc
    if not isinstance(data, dict):
        raise IdentityError(f"{path.name} must hold a JSON object")
    return data


def load_identity(path) -> BoxIdentity | None:
    """The box identity, or None when there is no identity file (manual-key mode)."""

    target = Path(path).expanduser()
    if not target.is_file():
        return None
    data = _read_private_json(target)
    values = {}
    for field in ("device_code", "claim_code", "box_secret"):
        value = str(data.get(field) or "").strip()
        if not value:
            raise IdentityError(f"{target.name} is missing {field}")
        values[field] = value
    values["device_code"] = values["device_code"].lower()
    return BoxIdentity(**values)


def write_identity(path, identity: BoxIdentity, **extra) -> Path:
    return write_private_json(
        path,
        {
            "device_code": identity.device_code,
            "claim_code": identity.claim_code,
            "box_secret": identity.box_secret,
            **extra,
        },
    )


def load_device_key(path, device_code: str) -> str | None:
    """The saved key for ``device_code``; None if absent or saved for another box."""

    target = Path(path).expanduser()
    if not target.is_file():
        return None
    try:
        data = _read_private_json(target)
    except IdentityError:
        return None
    if str(data.get("device_code") or "").strip().lower() != device_code:
        return None
    key = str(data.get("device_key") or "").strip()
    return key or None


def save_device_key(path, device_code: str, device_key: str) -> Path:
    return write_private_json(
        path,
        {
            "device_code": device_code,
            "device_key": device_key,
            "activated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


@dataclass(frozen=True)
class ResolvedCredentials:
    device_id: str
    device_key: str
    # "environment": ECHOSENSE_DEVICE_ID/KEY (manual key flow);
    # "identity": identity.json plus the saved key; "needs_activation": identity, no key yet.
    source: str
    identity: BoxIdentity | None = None

    def __repr__(self) -> str:
        return f"ResolvedCredentials(device_id={self.device_id!r}, source={self.source!r})"


def resolve_credentials(environ, identity_path, credentials_path) -> ResolvedCredentials:
    """Pick the device credentials for this run.

    With an identity file the box is QR-set-up: its device code comes from identity.json
    and its key from device_credentials.json (saved at activation). A manually issued
    key in ECHOSENSE_DEVICE_KEY is used only when ECHOSENSE_DEVICE_ID names the same
    box. Without an identity file nothing changes: ECHOSENSE_DEVICE_ID/KEY as before.
    """

    env_id = str(environ.get("ECHOSENSE_DEVICE_ID") or "").strip()
    env_key = str(environ.get("ECHOSENSE_DEVICE_KEY") or "").strip()
    identity = load_identity(identity_path)
    if identity is None:
        return ResolvedCredentials(env_id, env_key, "environment")
    saved = load_device_key(credentials_path, identity.device_code)
    if saved:
        return ResolvedCredentials(identity.device_code, saved, "identity", identity)
    if env_key and env_id.lower() == identity.device_code:
        return ResolvedCredentials(identity.device_code, env_key, "environment", identity)
    return ResolvedCredentials(identity.device_code, "", "needs_activation", identity)
