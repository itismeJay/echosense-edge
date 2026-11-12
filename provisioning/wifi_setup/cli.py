"""Read-only preflight and explicitly invoked credential bootstrap commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from provisioning.wifi_setup.credentials import (
    create_setup_credentials,
    save_setup_credentials,
)
from provisioning.wifi_setup.networkmanager import NetworkManager
from provisioning.wifi_setup.service import run_service


def _machine_id():
    try:
        return Path("/etc/machine-id").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _mac_address(interface):
    try:
        return Path(f"/sys/class/net/{interface}/address").read_text(
            encoding="utf-8"
        ).strip()
    except OSError:
        return ""


def preflight(args):
    report = NetworkManager(args.interface).preflight()
    # Keep exact /etc/os-release content out of console output; expose only the
    # fields an operator needs to approve the target.
    release = {}
    for line in report.pop("os_release", "").splitlines():
        key, separator, value = line.partition("=")
        if separator and key in {"PRETTY_NAME", "VERSION_CODENAME", "ID"}:
            release[key.lower()] = value.strip().strip('"')
    report["os"] = release
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("supported") else 2


def bootstrap(args):
    target = Path(args.output).expanduser()
    if target.exists():
        raise RuntimeError("refusing to replace existing Wi-Fi setup credentials")
    credentials, admin_password = create_setup_credentials(
        _machine_id(), _mac_address(args.interface)
    )
    save_setup_credentials(target, credentials)
    print("Wi-Fi setup credentials created owner-only.")
    print(f"Setup hotspot: {credentials.hotspot_ssid}")
    print(f"Hotspot password (record once): {credentials.hotspot_password}")
    print(f"Administrator password (record once): {admin_password}")
    print("Store these on the physical device setup card; this output is not saved.")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description="EchoSense Wi-Fi setup service")
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser(
        "preflight", help="read-only Raspberry Pi OS and NetworkManager checks"
    )
    check.add_argument("--interface", default="wlan0")
    check.set_defaults(function=preflight)
    create = subparsers.add_parser(
        "bootstrap", help="create unique setup credentials (does not change networking)"
    )
    create.add_argument("--interface", default="wlan0")
    create.add_argument(
        "--output", default="/etc/echosense/wifi-setup.json"
    )
    create.set_defaults(function=bootstrap)
    service = subparsers.add_parser(
        "service", help="run the system service (after hardware approval)"
    )
    service.set_defaults(function=lambda _args: run_service())
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.function(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"wifi-setup: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
