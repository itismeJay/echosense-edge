"""Dependency-free, mobile-friendly local setup portal."""

from __future__ import annotations

from dataclasses import dataclass
import html
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import secrets
import threading
import time
from urllib.parse import parse_qs, urlsplit

from provisioning.wifi_setup.controller import ProvisioningBusyError
from provisioning.wifi_setup.validation import WifiInputError


MAX_BODY_BYTES = 4096
SESSION_SECONDS = 15 * 60
MAX_LOGIN_FAILURES = 5
LOGIN_LOCKOUT_SECONDS = 60
# A page view or form post within this window means someone is using the portal.
ACTIVITY_SECONDS = 90


@dataclass(slots=True)
class Session:
    csrf_token: str
    expires_at: float


class PortalSecurity:
    def __init__(self, credentials, *, clock=time.monotonic):
        self.credentials = credentials
        self.clock = clock
        self.sessions = {}
        self.failures = {}
        self._lock = threading.Lock()

    def authenticate(self, client_ip: str, password: object) -> str | None:
        now = self.clock()
        with self._lock:
            attempts, locked_until = self.failures.get(client_ip, (0, 0.0))
            if locked_until > now:
                return None
        if not self.credentials.verify_admin_password(password):
            with self._lock:
                attempts += 1
                locked_until = (
                    now + LOGIN_LOCKOUT_SECONDS
                    if attempts >= MAX_LOGIN_FAILURES
                    else 0.0
                )
                self.failures[client_ip] = (attempts, locked_until)
            return None
        token = secrets.token_urlsafe(32)
        with self._lock:
            self.failures.pop(client_ip, None)
            self.sessions[token] = Session(
                csrf_token=secrets.token_urlsafe(24),
                expires_at=now + SESSION_SECONDS,
            )
        return token

    def has_active_session(self) -> bool:
        now = self.clock()
        with self._lock:
            return any(session.expires_at > now for session in self.sessions.values())

    def session(self, token: object) -> Session | None:
        now = self.clock()
        with self._lock:
            session = self.sessions.get(str(token or ""))
            if session is None or session.expires_at <= now:
                self.sessions.pop(str(token or ""), None)
                return None
            return session


_STYLE = """
body{font-family:system-ui,sans-serif;background:#f3f6f8;color:#17212b;margin:0}
main{max-width:42rem;margin:auto;padding:1rem}section{background:white;border-radius:14px;
padding:1rem;margin:1rem 0;box-shadow:0 2px 12px #0001}label{display:block;margin:.8rem 0}
input,select,button{box-sizing:border-box;width:100%;font:inherit;padding:.8rem;border-radius:9px;
border:1px solid #9aa7b2}button{background:#0c6b58;color:white;border:0;font-weight:700}
.secondary{background:#43515d}.danger{background:#9f2f2f}.muted{color:#5d6872;font-size:.92rem}
.network{display:flex;gap:.6rem;align-items:center}.network input{width:auto}.status{font-weight:700}
"""


def _page(title: str, body: str) -> bytes:
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{_STYLE}</style></head>"
        f"<body><main><h1>{html.escape(title)}</h1>{body}</main></body></html>"
    ).encode("utf-8")


def _cookie_token(header: str) -> str:
    jar = cookies.SimpleCookie()
    try:
        jar.load(header or "")
    except cookies.CookieError:
        return ""
    morsel = jar.get("ESSESSION")
    return morsel.value if morsel else ""


class PortalServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self, address, controller, credentials, *, allowed_network,
        captive=False, clock=time.monotonic,
    ):
        self.controller = controller
        self.clock = clock
        self.security = PortalSecurity(credentials, clock=clock)
        self.allowed_network = ipaddress.ip_network(allowed_network)
        self.captive = bool(captive)
        self.last_activity = None
        super().__init__(address, PortalHandler)
        host, port = self.server_address[:2]
        self.portal_url = f"http://{host}/" if port == 80 else f"http://{host}:{port}/"

    def mark_activity(self):
        self.last_activity = self.clock()

    def in_use(self) -> bool:
        """True while a signed-in session exists or a page was used recently."""

        if self.security.has_active_session():
            return True
        last = self.last_activity
        return last is not None and self.clock() - last < ACTIVITY_SECONDS


class PortalHandler(BaseHTTPRequestHandler):
    server_version = "EchoSenseSetup/1"
    sys_version = ""

    def log_message(self, _format, *_args):
        # Request paths, form fields and cookies must not enter routine logs.
        return

    def _local_client(self) -> bool:
        try:
            return ipaddress.ip_address(self.client_address[0]) in self.server.allowed_network
        except ValueError:
            return False

    def _send(self, status, body, *, content_type="text/html; charset=utf-8", cookie=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; base-uri 'none'",
        )
        self.send_header("Referrer-Policy", "no-referrer")
        if cookie:
            self.send_header(
                "Set-Cookie",
                f"ESSESSION={cookie}; Path=/; HttpOnly; SameSite=Strict; Max-Age={SESSION_SECONDS}",
            )
        self.end_headers()
        self.wfile.write(body)

    def _redirect(self, location, *, cookie=None):
        self.send_response(303)
        self.send_header("Location", location)
        self.send_header("Cache-Control", "no-store")
        if cookie:
            self.send_header(
                "Set-Cookie",
                f"ESSESSION={cookie}; Path=/; HttpOnly; SameSite=Strict; Max-Age={SESSION_SECONDS}",
            )
        self.end_headers()

    def _session(self):
        return self.server.security.session(_cookie_token(self.headers.get("Cookie", "")))

    def _form(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY_BYTES:
            raise WifiInputError("Invalid form size")
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0]
        if content_type != "application/x-www-form-urlencoded":
            raise WifiInputError("Unsupported form type")
        raw = self.rfile.read(length).decode("utf-8", errors="strict")
        return {key: values[-1] for key, values in parse_qs(raw, keep_blank_values=True).items()}

    def _require_post_session(self, form):
        session = self._session()
        if session is None:
            raise PermissionError("login required")
        if not secrets.compare_digest(str(form.get("csrf") or ""), session.csrf_token):
            raise PermissionError("invalid request token")
        return session

    def do_GET(self):  # noqa: N802
        if not self._local_client():
            self._send(403, b"Forbidden", content_type="text/plain")
            return
        path = urlsplit(self.path).path
        if self.server.captive:
            # DNS answers every name with the portal IP, so OS connectivity
            # checks (generate_204, hotspot-detect.html, connecttest.txt, ...)
            # land here; redirecting them makes phones open the setup page.
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
            if host != self.server.server_address[0] or path not in {"/", "/login"}:
                self._redirect(self.server.portal_url)
                return
        elif path not in {"/", "/login", "/generate_204", "/hotspot-detect.html"}:
            self._send(404, b"Not found", content_type="text/plain")
            return
        self.server.mark_activity()
        session = self._session()
        if session is None:
            body = _page(
                "EchoSense Wi-Fi Setup",
                "<section><p>Enter the administrator password from this device's setup card.</p>"
                "<form method='post' action='/login'><label>Administrator password"
                "<input type='password' name='password' autocomplete='current-password' required>"
                "</label><button type='submit'>Sign in</button></form></section>",
            )
            self._send(200, body)
            return
        self._render_status(session)

    def _render_status(self, session):
        status = self.server.controller.status()
        network_rows = []
        for index, network in enumerate(status["networks"]):
            checked = " checked" if index == 0 else ""
            label = f"{html.escape(network.ssid)} — {network.signal}%"
            if network.secured:
                label += " — secured"
            network_rows.append(
                "<label class='network'>"
                f"<input type='radio' name='ssid' value='{html.escape(network.ssid, quote=True)}'"
                f"{checked}><span>{label}</span></label>"
            )
        if not network_rows:
            network_rows.append("<p class='muted'>No scan results. Use manual SSID entry below.</p>")
        saved_rows = []
        saved_networks = sorted(
            status["saved_networks"],
            key=lambda item: (-item.priority, item.ssid.casefold()),
        )
        for profile in saved_networks:
            saved_rows.append(
                "<form method='post' action='/priority'>"
                f"<input type='hidden' name='csrf' value='{session.csrf_token}'>"
                "<input type='hidden' name='uuid' value='"
                f"{html.escape(profile.uuid, quote=True)}'>"
                f"<p><strong>{html.escape(profile.ssid)}</strong>"
                + (" (current)" if profile.active else "")
                + "</p><label>Priority (higher connects first)"
                "<input type='number' name='priority' min='-999' max='999' "
                f"value='{int(profile.priority)}' required></label>"
                "<button class='secondary' type='submit'>Save priority</button>"
                "</form>"
            )
            if profile.active:
                continue
            saved_rows.append(
                "<form method='post' action='/connect-saved'>"
                f"<input type='hidden' name='csrf' value='{session.csrf_token}'>"
                "<input type='hidden' name='uuid' value='"
                f"{html.escape(profile.uuid, quote=True)}'>"
                "<label><input style='width:auto' type='checkbox' "
                "name='confirm_change' value='yes' required> "
                "Confirm switching to this saved network</label>"
                "<button type='submit'>Switch to this network</button>"
                "</form>"
                "<form method='post' action='/forget'>"
                f"<input type='hidden' name='csrf' value='{session.csrf_token}'>"
                "<input type='hidden' name='uuid' value='"
                f"{html.escape(profile.uuid, quote=True)}'>"
                "<label><input style='width:auto' type='checkbox' "
                "name='confirm_forget' value='yes' required> "
                "Confirm forgetting this network</label>"
                "<button class='danger' type='submit'>Forget</button>"
                "</form>"
            )
        error = ""
        if status["last_error"]:
            error = (
                "<p>Last switch failed. A usable saved connection or setup access "
                "was restored; check the password and retry.</p>"
            )
        if status["connected"]:
            transition_note = (
                "Refresh keeps the current connection. Connect switches the Pi "
                "to the selected network, so reopen this page at its new address."
            )
        else:
            transition_note = (
                "Refresh and Connect briefly stop this hotspot because the Pi has "
                "one Wi-Fi radio. Reconnect to the same setup hotspot if the page closes."
            )
        body = (
            "<section><p class='status'>Status: "
            + html.escape(status["state"].replace("_", " "))
            + "</p>"
            + "<p>Current network: "
            + html.escape(status.get("current_network") or "Not connected")
            + "</p>"
            + error
            + "<p class='muted'>This authenticated page is available only from "
            + "the device's current local network.</p></section>"
            + "<section><h2>Connect to Wi-Fi</h2><form method='post' action='/connect'>"
            + f"<input type='hidden' name='csrf' value='{session.csrf_token}'>"
            + "".join(network_rows)
            + "<label>Manual or hidden SSID<input name='manual_ssid' maxlength='32'></label>"
            + "<label><input style='width:auto' type='checkbox' name='hidden' value='yes'> Hidden network</label>"
            + "<label><input style='width:auto' type='checkbox' name='open_network' value='yes'> Open network (no password)</label>"
            + "<label><input style='width:auto' type='checkbox' name='same_phone' value='yes'> This phone will provide the destination hotspot (10-second start delay)</label>"
            + "<label>Wi-Fi password<input type='password' name='password' autocomplete='new-password'></label>"
            + "<label><input style='width:auto' type='checkbox' "
            + "name='confirm_change' value='yes' required> "
            + "Confirm switching this device to the selected network</label>"
            + "<button type='submit'>Connect</button></form>"
            + "<form method='post' action='/refresh'>"
            + f"<input type='hidden' name='csrf' value='{session.csrf_token}'>"
            + "<button class='secondary' type='submit'>Refresh network list</button></form>"
            + f"<p class='muted'>{transition_note}</p></section>"
        )
        if saved_rows:
            body += "<section><h2>Saved networks</h2>" + "".join(saved_rows) + "</section>"
        self._send(200, _page("EchoSense Wi-Fi Setup", body))

    def do_POST(self):  # noqa: N802
        if not self._local_client():
            self._send(403, b"Forbidden", content_type="text/plain")
            return
        path = urlsplit(self.path).path
        self.server.mark_activity()
        try:
            form = self._form()
            if path == "/login":
                token = self.server.security.authenticate(
                    self.client_address[0], form.get("password", "")
                )
                if token is None:
                    self._send(403, _page("Sign-in failed", "<p>Check the setup card and try again later.</p>"))
                else:
                    self._redirect("/", cookie=token)
                return
            self._require_post_session(form)
            if path == "/connect":
                ssid = form.get("manual_ssid", "").strip() or form.get("ssid", "")
                self.server.controller.request_connection(
                    ssid,
                    form.get("password", ""),
                    hidden=form.get("hidden") == "yes",
                    open_network=form.get("open_network") == "yes",
                    activation_delay=10 if form.get("same_phone") == "yes" else 0.5,
                    confirmed=form.get("confirm_change") == "yes",
                )
                self._send(202, _page(
                    "Connecting",
                    "<p>The device will leave its current network during the "
                    "attempt. Join the selected Wi-Fi, then check the dashboard. "
                    "If it fails, use the restored saved network or setup hotspot.</p>",
                ))
                return
            if path == "/refresh":
                self.server.controller.request_refresh()
                self._send(202, _page(
                    "Refreshing",
                    "<p>The network list is refreshing. If you are using the setup "
                    "hotspot, reconnect after it restarts.</p>",
                ))
                return
            if path == "/connect-saved":
                self.server.controller.request_saved_connection(
                    form.get("uuid", ""),
                    confirmed=form.get("confirm_change") == "yes",
                )
                self._send(202, _page(
                    "Switching network",
                    "<p>The device is switching to the saved network. Reopen "
                    "the local portal at its new address.</p>",
                ))
                return
            if path == "/priority":
                self.server.controller.set_priority(
                    form.get("uuid", ""), form.get("priority", "")
                )
                self._redirect("/")
                return
            if path == "/forget":
                self.server.controller.forget(
                    form.get("uuid", ""),
                    confirmed=form.get("confirm_forget") == "yes",
                )
                self._redirect("/")
                return
            self._send(404, b"Not found", content_type="text/plain")
        except PermissionError:
            self._send(403, b"Forbidden", content_type="text/plain")
        except (UnicodeDecodeError, WifiInputError) as exc:
            self._send(400, _page("Invalid Wi-Fi settings", f"<p>{html.escape(str(exc))}</p>"))
        except ProvisioningBusyError as exc:
            self._send(409, _page("Wi-Fi setup is busy", f"<p>{html.escape(str(exc))}</p>"))


def make_portal_server(
    controller,
    credentials,
    *,
    bind_address="10.42.0.1",
    port=80,
    allowed_network="10.42.0.0/24",
    captive=False,
):
    return PortalServer(
        (bind_address, int(port)),
        controller,
        credentials,
        allowed_network=allowed_network,
        captive=captive,
    )
