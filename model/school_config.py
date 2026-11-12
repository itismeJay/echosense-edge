"""The school's detection config from the backend's /devices/me/config.

Fetched with the device credentials at startup and every 5 minutes, and cached on disk
so a device that starts without a network still applies its school's last known words
and sensitivity. It sets:

- the synchronized word list: global words minus the ones the school turned off, plus
  the school's own words (model/monitored_terms.py);
- the school policy: sensitivity and the turned-off words (detection/school_policy.py).

When the backend has no config endpoint (404) or the device is not assigned to a
school (409), the edge falls back to the global /dictionary/ sync with default rules.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Mapping, Optional

import requests

from config import API_URL, DATA_DIR, ECHOSENSE_DEVICE_ID
from detection import school_policy
from model.monitored_terms import (
    normalize_text,
    parse_config_terms,
    set_backend_terms,
    sync_backend_dictionary,
)

SYNC_INTERVAL_SECONDS = 300.0

# sync_school_config() outcomes.
APPLIED = "applied"
UNCHANGED = "unchanged"
CACHED = "cached"
DICTIONARY = "dictionary"
FAILED = "failed"

_lock = threading.Lock()
_applied_version: Optional[str] = None
_sync_thread: Optional[threading.Thread] = None


class SchoolConfigError(ValueError):
    """The config body is not a usable /devices/me/config response."""


def cache_path(api_url: str = API_URL, device_id: str = ECHOSENSE_DEVICE_ID) -> Path:
    """One cache file per backend and device, so local and live configs never mix."""

    configured = os.environ.get("ECHOSENSE_SCHOOL_CONFIG_PATH")
    if configured:
        return Path(configured)
    key = hashlib.sha256(f"{api_url}|{device_id}".encode("utf-8")).hexdigest()[:12]
    return Path(DATA_DIR) / f"school_config.{key}.json"


def applied_version() -> Optional[str]:
    with _lock:
        return _applied_version


def _validate(body: object) -> Mapping:
    if not isinstance(body, Mapping):
        raise SchoolConfigError("config is not an object")
    if body.get("schema_version") != 1:
        raise SchoolConfigError("unsupported config schema_version")
    if not isinstance(body.get("terms"), list):
        raise SchoolConfigError("config terms are not a list")
    disabled = body.get("disabled_terms", [])
    if not isinstance(disabled, list):
        raise SchoolConfigError("config disabled_terms is not a list")
    version = body.get("config_version")
    if not isinstance(version, str) or not version:
        raise SchoolConfigError("config_version is missing")
    return body


def apply_config(body: object, *, origin: str = "backend") -> str:
    """Apply a /devices/me/config body. Raises SchoolConfigError when it is invalid."""

    global _applied_version
    config = _validate(body)
    terms = parse_config_terms(config["terms"])
    disabled = {normalize_text(text) for text in config.get("disabled_terms") or ()}
    disabled.discard("")
    sensitivity = str(config.get("sensitivity") or school_policy.NORMAL)
    with _lock:
        set_backend_terms(terms)
        school_policy.set_policy(sensitivity, disabled)
        _applied_version = config["config_version"]
    school_words = sum(1 for item in terms if item.source == "school")
    print(
        f"[SCHOOL_CONFIG] applied origin={origin} sensitivity={school_policy.sensitivity()} "
        f"terms={len(terms)} school_terms={school_words} turned_off={len(disabled)} "
        f"version={config['config_version'][:12]}"
    )
    return config["config_version"]


def save_cache(body: Mapping, path: Optional[Path] = None) -> bool:
    """Write the config atomically next to the outbox. Never raises."""

    path = Path(path or cache_path())
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        cached = {"api_url": API_URL, "device_id": ECHOSENSE_DEVICE_ID, "config": dict(body)}
        handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".school_config.")
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(cached, stream, sort_keys=True, default=str)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        return True
    except OSError as exc:
        print(f"[SCHOOL_CONFIG] cache_write_failed={type(exc).__name__}")
        return False


def load_cache(path: Optional[Path] = None) -> Optional[Mapping]:
    """The cached config for this backend and device, or None."""

    path = Path(path or cache_path())
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(cached, Mapping):
        return None
    if cached.get("api_url") != API_URL or cached.get("device_id") != ECHOSENSE_DEVICE_ID:
        return None
    config = cached.get("config")
    return config if isinstance(config, Mapping) else None


def _apply_cache_if_nothing_applied() -> str:
    if applied_version() is not None:
        return FAILED
    cached = load_cache()
    if cached is None:
        return FAILED
    try:
        apply_config(cached, origin="cache")
    except SchoolConfigError:
        return FAILED
    return CACHED


def sync_school_config(session=requests, timeout: float = 10.0) -> str:
    """Fetch and apply the school config once. Failures keep the last good config."""

    from sender.http_client import device_auth_headers

    global _applied_version
    try:
        response = session.get(
            f"{API_URL}/devices/me/config",
            headers=device_auth_headers(),
            timeout=timeout,
        )
    except Exception as exc:
        print(f"[SCHOOL_CONFIG] unavailable ({type(exc).__name__}); using current config")
        return _apply_cache_if_nothing_applied()

    status = getattr(response, "status_code", None)
    if status in (404, 409):
        reason = "no config endpoint" if status == 404 else "device has no school"
        print(f"[SCHOOL_CONFIG] {reason} ({status}); using the global dictionary")
        with _lock:
            school_policy.reset_policy()
            _applied_version = None
        return DICTIONARY if sync_backend_dictionary(session=session) else FAILED
    if status in (401, 403):
        print(
            f"[SCHOOL_CONFIG] ERROR refused ({status}): check ECHOSENSE_DEVICE_ID and "
            "ECHOSENSE_DEVICE_KEY in edge.env; using current config"
        )
        return _apply_cache_if_nothing_applied()
    try:
        response.raise_for_status()
        body = response.json()
        version = _validate(body)["config_version"]
    except Exception as exc:
        print(f"[SCHOOL_CONFIG] invalid response ({type(exc).__name__}); using current config")
        return _apply_cache_if_nothing_applied()

    if version == applied_version():
        return UNCHANGED
    apply_config(body)
    save_cache(body)
    return APPLIED


def resync_terms() -> bool:
    """Re-fetch the word list now (after a backend rejection). True when it is current."""

    return sync_school_config() in (APPLIED, UNCHANGED, DICTIONARY)


def start_school_config_sync(interval: float = SYNC_INTERVAL_SECONDS) -> threading.Thread:
    """Sync now, then every ``interval`` seconds in one daemon thread."""

    global _sync_thread
    with _lock:
        if _sync_thread is not None and _sync_thread.is_alive():
            return _sync_thread

    sync_school_config()

    def _loop() -> None:
        while True:
            time.sleep(interval)
            try:
                sync_school_config()
            except Exception as exc:  # never let the sync thread die
                print(f"[SCHOOL_CONFIG] sync_error={type(exc).__name__}")

    thread = threading.Thread(target=_loop, name="school-config-sync", daemon=True)
    with _lock:
        _sync_thread = thread
    thread.start()
    return thread
