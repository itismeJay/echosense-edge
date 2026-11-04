"""Provision EchoSense boxes: identity.json plus a printable QR label for each.

Run on the operator's Mac (or the Pi itself) as a super admin:

    venv/bin/pip install -r tools/requirements-provision.txt   # once: qrcode
    venv/bin/python tools/provision_box.py --backend http://localhost:8000 \\
        --email admin@echosense.app --count 1

The password is asked for (or read from ECHOSENSE_ADMIN_PASSWORD) and never printed.
For scripted runs, ECHOSENSE_ADMIN_TOKEN (an existing super-admin session token) skips
the sign-in; the account is still checked to be a super admin.
For each box this writes, under --out-dir (default labels/, owner-only):

    <device_code>/identity.json   copy to the box (data/identity.json, chmod 600)
    <device_code>/label.png       sticker with the QR, the device code and setup code
    <device_code>/label.pdf       same, for printing

--identity PATH also writes the (single) box's identity.json straight to PATH, e.g.
data/identity.json when provisioning the box you are sitting at.
--reset es-xxxx re-issues setup codes for an existing box (it is detached and its key
revoked) and writes a fresh identity and label.

The setup code and box secret are shown by the backend only once: keep the folder
private, print the label, copy identity.json to the box, then delete the folder.
"""

from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path
import sys

import requests

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from provisioning.identity import (  # noqa: E402
    OWNER_ONLY_DIR,
    BoxIdentity,
    write_identity,
)

LABEL_INSTRUCTION = "Scan in EchoSense → Classrooms → Add device"
LABEL_DPI = 300
LABEL_SIZE = (1050, 600)  # 3.5 x 2 in at 300 dpi
_FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


class ProvisionError(RuntimeError):
    pass


def _font(size):
    from PIL import ImageFont

    for candidate in _FONT_CANDIDATES:
        if Path(candidate).is_file():
            try:
                return ImageFont.truetype(candidate, size), True
            except OSError:
                continue
    return ImageFont.load_default(size=size), False


def render_label(qr_payload, device_code, claim_code):
    """A white sticker image: QR on the left, codes and the instruction on the right."""

    import qrcode
    from PIL import Image, ImageDraw

    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=10, border=2)
    qr.add_data(qr_payload)
    qr.make(fit=True)
    qr_image = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    side = LABEL_SIZE[1] - 60
    qr_image = qr_image.resize((side, side), Image.NEAREST)

    label = Image.new("RGB", LABEL_SIZE, "white")
    label.paste(qr_image, (30, 30))
    draw = ImageDraw.Draw(label)
    left = side + 70
    title, rich = _font(44)
    code_font, _ = _font(76)
    small, _ = _font(30)
    instruction = LABEL_INSTRUCTION if rich else LABEL_INSTRUCTION.replace("→", ">")
    draw.text((left, 50), "EchoSense box", fill="black", font=title)
    draw.text((left, 120), device_code, fill="black", font=code_font)
    draw.text((left, 240), "Setup code", fill="black", font=small)
    draw.text((left, 280), claim_code, fill="black", font=title)
    words = instruction.split(" ")
    lines, line = [], ""
    for word in words:
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=small) > LABEL_SIZE[0] - left - 30 and line:
            lines.append(line)
            line = word
        else:
            line = trial
    lines.append(line)
    for index, text in enumerate(lines):
        draw.text((left, 380 + index * 42), text, fill="black", font=small)
    draw.rectangle([(4, 4), (LABEL_SIZE[0] - 5, LABEL_SIZE[1] - 5)], outline="black", width=3)
    return label


def save_label(image, folder):
    png = folder / "label.png"
    pdf = folder / "label.pdf"
    image.save(png, dpi=(LABEL_DPI, LABEL_DPI))
    image.save(pdf, resolution=LABEL_DPI)
    for path in (png, pdf):
        os.chmod(path, 0o600)
    return png, pdf


def write_box(box, out_dir, backend_url, identity_path=None):
    identity = BoxIdentity(
        device_code=box["device_code"],
        claim_code=box["claim_code"],
        box_secret=box["box_secret"],
    )
    folder = Path(out_dir).expanduser() / identity.device_code
    folder.mkdir(parents=True, exist_ok=True, mode=OWNER_ONLY_DIR)
    os.chmod(folder, OWNER_ONLY_DIR)
    written = [write_identity(folder / "identity.json", identity, backend_url=backend_url)]
    if identity_path is not None:
        written.append(write_identity(identity_path, identity, backend_url=backend_url))
    png, pdf = save_label(
        render_label(box["qr_payload"], identity.device_code, identity.claim_code), folder
    )
    return {"identity": written, "png": png, "pdf": pdf}


def _detail(response):
    try:
        detail = response.json().get("detail")
    except Exception:
        detail = None
    return detail if isinstance(detail, str) else f"HTTP {response.status_code}"


def login(session, backend_url, email, password):
    response = session.post(
        f"{backend_url}/auth/login", json={"email": email, "password": password}, timeout=20
    )
    if response.status_code != 200:
        raise ProvisionError(f"Sign-in failed: {_detail(response)}")
    body = response.json()
    if body.get("must_change_password"):
        raise ProvisionError("Change this account's temporary password in the dashboard first")
    if not (body.get("user") or {}).get("is_super_admin"):
        raise ProvisionError("Only a super admin can provision boxes")
    session.headers["Authorization"] = f"Bearer {body['access_token']}"


def use_token(session, backend_url, token):
    session.headers["Authorization"] = f"Bearer {token}"
    response = session.get(f"{backend_url}/auth/me", timeout=20)
    if response.status_code != 200:
        raise ProvisionError(f"Session token not accepted: {_detail(response)}")
    body = response.json()
    if body.get("must_change_password"):
        raise ProvisionError("Change this account's temporary password in the dashboard first")
    if not body.get("is_super_admin"):
        raise ProvisionError("Only a super admin can provision boxes")


def provision(session, backend_url, count, display_name=None):
    payload = {"count": count}
    if display_name:
        payload["display_name"] = display_name
    response = session.post(f"{backend_url}/devices/provision", json=payload, timeout=60)
    if response.status_code != 201:
        raise ProvisionError(f"Provisioning failed: {_detail(response)}")
    return response.json()["boxes"]


def reset(session, backend_url, device_code):
    listed = session.get(f"{backend_url}/devices", timeout=20)
    if listed.status_code != 200:
        raise ProvisionError(f"Could not list devices: {_detail(listed)}")
    match = next((d for d in listed.json() if d.get("device_code") == device_code), None)
    if match is None:
        raise ProvisionError(f"No device {device_code}")
    response = session.post(f"{backend_url}/devices/{match['id']}/reset", timeout=20)
    if response.status_code != 200:
        raise ProvisionError(f"Reset failed: {_detail(response)}")
    return [response.json()]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--backend",
        default=os.environ.get("ECHOSENSE_BACKEND_BASE_URL") or "http://localhost:8000",
    )
    parser.add_argument("--email", default=os.environ.get("ECHOSENSE_ADMIN_EMAIL"))
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--display-name")
    parser.add_argument("--out-dir", default=str(_ROOT / "labels"))
    parser.add_argument("--identity", help="also write the box's identity.json here")
    parser.add_argument("--reset", metavar="DEVICE_CODE")
    args = parser.parse_args(argv)
    if not 1 <= args.count <= 50:
        parser.error("--count must be 1..50")
    if args.identity and (args.count != 1 and not args.reset):
        parser.error("--identity works with a single box")
    if not args.email and not os.environ.get("ECHOSENSE_ADMIN_TOKEN"):
        parser.error("--email (or ECHOSENSE_ADMIN_EMAIL) is required")
    return args


def main(argv=None, session=None, password_prompt=getpass.getpass):
    args = parse_args(argv)
    backend_url = args.backend.rstrip("/")
    token = os.environ.get("ECHOSENSE_ADMIN_TOKEN", "").strip()
    session = session or requests.Session()
    try:
        if token:
            use_token(session, backend_url, token)
        else:
            password = os.environ.get("ECHOSENSE_ADMIN_PASSWORD") or password_prompt(
                f"Password for {args.email}: "
            )
            login(session, backend_url, args.email, password)
        boxes = (
            reset(session, backend_url, args.reset.strip().lower())
            if args.reset
            else provision(session, backend_url, args.count, args.display_name)
        )
        identity_path = Path(args.identity).expanduser() if args.identity else None
        for box in boxes:
            files = write_box(box, args.out_dir, backend_url, identity_path)
            print(f"{box['device_code']}: label {files['png']} and {files['pdf'].name}")
            for path in files["identity"]:
                print(f"{box['device_code']}: identity {path} (owner-only)")
    except (ProvisionError, requests.RequestException) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(
        "Setup codes are shown only once. Print the labels, copy identity.json to each "
        "box, then delete the labels folder."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
