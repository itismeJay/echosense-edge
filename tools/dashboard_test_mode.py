#!/usr/bin/env python3
"""Run genuine detection with explicitly approved TEST dashboard delivery."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from urllib.parse import urlparse

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_MICROPHONE = "EMEET M0 Plus"
DEFAULT_WHISPER_MODEL = "models/sapinsapin_whisper-small-pld-ceb-ct2-int8"


def _env_values(path):
    values = {}
    for raw_line in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key.isidentifier():
            values[key] = value
    return values


def _origin(value):
    parsed = urlparse(str(value or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return ""
    authority = parsed.hostname
    if parsed.port and not (
        (parsed.scheme == "http" and parsed.port == 80)
        or (parsed.scheme == "https" and parsed.port == 443)
    ):
        authority = f"{authority}:{parsed.port}"
    return f"{parsed.scheme}://{authority}"


def validate_test_destination(backend_url, approved_test_origin=None):
    """Return the origin, refusing known production before any request."""

    origin = _origin(backend_url)
    if not origin:
        raise ValueError("--backend-url must be a valid HTTP(S) URL")
    hostname = urlparse(origin).hostname
    if hostname not in {"127.0.0.1", "localhost", "::1"}:
        if _origin(approved_test_origin) != origin:
            raise ValueError(
                "non-local destinations require --approved-test-origin matching "
                "--backend-url exactly"
            )
    return origin


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run the real microphone/detector pipeline while labeling every "
            "delivered incident as TEST and verifying it in the dashboard feed."
        )
    )
    parser.add_argument("command", choices=("start",))
    parser.add_argument("--backend-url", required=True)
    parser.add_argument("--approved-test-origin")
    parser.add_argument("--credentials-file", required=True)
    parser.add_argument(
        "--reviewer-env-file",
        required=True,
        help=(
            "private env file containing reviewer email/password; values are "
            "used only to obtain an in-memory dashboard token"
        ),
    )
    parser.add_argument(
        "--reviewer-email-key",
        default="ECHOSENSE_DEV_ADMIN_EMAIL",
    )
    parser.add_argument(
        "--reviewer-password-key",
        default="ECHOSENSE_DEV_ADMIN_PASSWORD",
    )
    parser.add_argument(
        "--outbox",
        default="data/dashboard_test_outbox.sqlite3",
    )
    parser.add_argument("--microphone", default=DEFAULT_MICROPHONE)
    parser.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL)
    parser.add_argument(
        "--i-approve-live-capture",
        action="store_true",
        help="confirm that consenting adults approved live microphone capture",
    )
    parser.add_argument(
        "--i-approve-dashboard-test-delivery",
        action="store_true",
        help="confirm that TEST incidents may be written to this test backend",
    )
    args = parser.parse_args(argv)
    if not args.i_approve_live_capture:
        parser.error("--i-approve-live-capture is required")
    if not args.i_approve_dashboard_test_delivery:
        parser.error("--i-approve-dashboard-test-delivery is required")
    try:
        args.backend_origin = validate_test_destination(
            args.backend_url,
            args.approved_test_origin,
        )
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _load_device_credentials(path):
    values = _env_values(path)
    device_id = str(values.get("ECHOSENSE_DEVICE_ID") or "").strip()
    device_key = str(values.get("ECHOSENSE_DEVICE_KEY") or "").strip()
    if not device_id or not device_key:
        raise RuntimeError(
            "credentials file must contain ECHOSENSE_DEVICE_ID and "
            "ECHOSENSE_DEVICE_KEY"
        )
    return device_id, device_key


def _reviewer_token(args):
    values = _env_values(args.reviewer_env_file)
    email = str(values.get(args.reviewer_email_key) or "").strip()
    password = str(values.get(args.reviewer_password_key) or "")
    if not email or not password:
        raise RuntimeError(
            "reviewer env file does not contain the configured login keys"
        )
    try:
        health = requests.get(
            f"{args.backend_origin}/health",
            timeout=5,
        )
        health.raise_for_status()
        response = requests.post(
            f"{args.backend_origin}/auth/login",
            json={"email": email, "password": password},
            timeout=10,
        )
        response.raise_for_status()
        token = str(response.json().get("access_token") or "").strip()
    except (requests.RequestException, TypeError, ValueError) as exc:
        raise RuntimeError(
            "test backend health/login check failed"
        ) from exc
    if not token:
        raise RuntimeError("test backend login did not return a reviewer token")
    return token


def configure_environment(args, *, device_id, device_key, reviewer_token):
    """Set one-process overrides before importing any runtime module."""

    outbox_path = Path(args.outbox).expanduser()
    if not outbox_path.is_absolute():
        outbox_path = (PROJECT_ROOT / outbox_path).resolve()
    if "test" not in str(outbox_path).casefold():
        raise RuntimeError("--outbox must be a dedicated test-named path")
    os.environ.update({
        "ECHOSENSE_ENV_FILE": "",
        "ECHOSENSE_BACKEND_BASE_URL": args.backend_origin,
        "ECHOSENSE_DEVICE_ID": device_id,
        "ECHOSENSE_DEVICE_KEY": device_key,
        "ECHOSENSE_IDENTITY_FILE": "data/dashboard_test_no_identity.json",
        "ECHOSENSE_OUTBOX_ENABLED": "true",
        "ECHOSENSE_OUTBOX_DATABASE_PATH": str(outbox_path),
        "ECHOSENSE_DASHBOARD_TEST_MODE": "true",
        "ECHOSENSE_DASHBOARD_TEST_EXPECTED_BACKEND_ORIGIN": (
            args.backend_origin
        ),
        "ECHOSENSE_DASHBOARD_TEST_EXPECTED_DEVICE_ID": device_id,
        "ECHOSENSE_DASHBOARD_TEST_REVIEWER_TOKEN": reviewer_token,
        "ECHOSENSE_PREFERRED_MICROPHONE": args.microphone,
        "ECHOSENSE_MICROPHONE_STRICT": "true",
        "ECHOSENSE_WHISPER_MODEL": args.whisper_model,
        "ECHOSENSE_SHOW_TRANSCRIPT_TEXT": "false",
    })
    return outbox_path


def run(args):
    device_id, device_key = _load_device_credentials(args.credentials_file)
    # Login is the first non-health request and is made only to an explicitly
    # approved test origin.
    reviewer_token = _reviewer_token(args)
    outbox_path = configure_environment(
        args,
        device_id=device_id,
        device_key=device_key,
        reviewer_token=reviewer_token,
    )
    print(
        "[DASHBOARD_TEST] authorization confirmed "
        f"backend_origin={args.backend_origin} device={device_id} "
        f"outbox={outbox_path} label=TEST"
    )
    print(
        "[DASHBOARD_TEST] Use consenting adults only. No raw audio is stored "
        "or transmitted. Press Ctrl+C to stop."
    )
    import main as edge_main

    return edge_main.main()


def main(argv=None):
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
