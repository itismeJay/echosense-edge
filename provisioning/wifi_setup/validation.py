"""Validation at the untrusted mobile-portal boundary."""

from __future__ import annotations

import re


class WifiInputError(ValueError):
    """A submitted Wi-Fi setting is not safe or supported."""


def validate_ssid(value: object) -> str:
    ssid = str(value or "")
    if not ssid or ssid != ssid.strip():
        raise WifiInputError("SSID must not be blank or have outer spaces")
    try:
        encoded = ssid.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise WifiInputError("SSID must be valid UTF-8") from exc
    if len(encoded) > 32:
        raise WifiInputError("SSID must be at most 32 UTF-8 bytes")
    if any(ord(character) < 32 or ord(character) == 127 for character in ssid):
        raise WifiInputError("SSID must not contain control characters")
    return ssid


def validate_wifi_password(value: object, *, open_network: bool = False) -> str:
    password = str(value or "")
    if open_network:
        if password:
            raise WifiInputError("Open networks must not include a password")
        return ""
    if len(password) == 64 and re.fullmatch(r"[0-9A-Fa-f]{64}", password):
        return password
    if not 8 <= len(password) <= 63:
        raise WifiInputError("Wi-Fi password must contain 8 to 63 characters")
    if any(ord(character) < 32 or ord(character) == 127 for character in password):
        raise WifiInputError("Wi-Fi password must not contain control characters")
    return password


def validate_profile_uuid(value: object) -> str:
    profile_uuid = str(value or "").strip().lower()
    if not re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        profile_uuid,
    ):
        raise WifiInputError("Invalid saved-network identifier")
    return profile_uuid
