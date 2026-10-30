"""Live hub: the phone as a live controller (`groundtrack live-hub`, and Live mode in the app).

Two listeners, one asyncio loop (in a background thread when started from the app):

- **Phone (LAN):** HTTPS + WSS on `port` (default 8766). It serves the phone page
  (`live_phone/`) and `/ws/phone`, and nothing else: no file access, no other routes. Every
  request needs the pairing code in the URL, `https://<lan-ip>:8766/?k=<code>`; anything without
  it is refused (403). The certificate is self-signed (made once with `cryptography`); Safari's
  warning is accepted once on the phone (motion sensors need a secure page, and the WSS uses the
  same certificate).
- **Recorder (LAN, realtime only; plans/live_controller/10):** `/ws/record` on the same port,
  only while the laptop is in the realtime mode (`set_recorder`; else 409), one page at a time.
  The one exception to "never video over the link": its camera frames come to this machine
  (and only here) for the realtime tracker. Never relayed, never logged.
- **View (this machine only):** `ws://127.0.0.1:<view_port>/ws/view` (default 8767), for the
  Live view. A browser page may connect only from this machine (Origin 127.0.0.1 / localhost,
  or no Origin for local programs).

The certificate (`cert.pem`, `key.pem`) and the pairing code (`token.txt`) are kept per user in
`%APPDATA%\\groundtrack\\live\\` (never in the project), so a phone's Home Screen icon keeps
working between sessions until the code is renewed. Only connections are logged, never what
they carry.

Python API (for the app):

    hub = LiveHub(port=8766, view_port=8767, host="0.0.0.0", home=None, log=print)
    info = hub.start()      # listens in a background thread; returns hub.info()
    hub.info()              # {"running", "port", "view_port", "url", "phone_urls": [{"url",
                            #   "ip", "name", "likely", "qr_svg"}], "view_url", "fingerprint",
                            #   "phones", "views"}
    hub.rotate_token()      # a new pairing code; connected phones are closed (code 4001)
    hub.stop()

`url` / `phone_urls` carry the code (show them as the QR, `qr_svg`). `view_url` is the Live
view's socket. With `host="0.0.0.0"` the phone URLs list every LAN address, the Wi-Fi /
hotspot one first (`likely`).

Messages (JSON text frames; pose v1, see plans/live_controller/00_README.md):

    phone -> hub   pose, zero, control  -> checked, rebuilt with known fields only, the
                                           phone's "player" and "spot" added, sent to every
                                           view (anything else is dropped)
                   control spot         -> a spot claim, kept here (not relayed)
                   {"type": "ping", "id": n}  -> answered {"type": "pong", "id": n}
    view  -> hub   state                -> checked the same way, kept; each phone gets its own
                                           copy (on every state, claim, and when it connects)
    view  -> hub   {"type": "release", "spot": name}  -> the camera left the scene: the phone
                                           holding it lets it go (stays connected, picks again)
    hub -> view    {"type": "hub", "phones": n, "views": m,
                    "players": [{"player": 1, "spot": "North"}, ...]}   on every change
    hub -> phone   {"type": "hub", "phones": n, "views": m, "player": k}  on every change

A pose is relayed as {"type": "pose", "v": 1, "src", "seq", "t", "q", "p", "status", "player",
"spot"} (q normalised), a zero as {"type": "zero", "player", "spot"}, a control as {"type":
"control", "action", "value", "player", "spot"}: the actions play, pause, seek, record,
section, gain, scale, space (true: the clutch held, false: let go) and level (no value: the
space back to level). Phones never switch the clip or the site (the laptop does): a "clip"
control from a phone is dropped.

Players and spots (cameras; plans/live_controller/08_space_and_multiplayer.md, b):
- Phones are numbered 1-4 (the lowest free number; a page that comes back gets its number
  back when it's free). A fifth is closed with 4003.
- The spots are the view's, every camera of the site: state "spots" = [{"name", "clip",
  "group", "in_scene", "gain", "scale", "space_held", "space_tilted"}] (all but name
  optional), "groups" = [{"name", "spots": [names]}] (areas; empty for a small site) and
  "start_spot" (the camera the view opened on). A state without "spots" keeps the last ones
  (so a view starting up doesn't drop the claims); "spots": [] clears them. "site" is the
  site's name (e.g. "trafalgar"); a state without it keeps the last one.
- A phone claims a spot with {"type": "control", "action": "spot", "value": "<name>" | null};
  a spot is held by one phone at a time (a claim on a taken or unlisted spot is refused; the
  phone's state shows what it holds). A phone that joins without a spot takes the start spot
  when it's free (the first phone gets the start camera); so does one whose spot the view no
  longer lists. Every other phone picks.
- A new site (the laptop switched: "site" changed, and only that; the first "site" seen just
  sets it) releases every camera; the first phone (by joining) gets the start spot, the
  others pick again. Within one site (a new start spot, cameras added or dropped) the claims
  still listed stay.
- While the view lists spots, what acts on a spot (pose, zero, gain, scale, space, level) is
  dropped for a phone that holds none; play, pause, seek (and record, section) are shared.
  With no spots (one clip) everything goes through, "spot": null.
- SPACE is relayed as presses and let-goes only (a let-go without a press is dropped). The
  hub lets go for the phone (space false, its player and spot) when it leaves, is replaced,
  changes spot or loses its spot, so the clutch never sticks.
- Each phone's state: the view's shared fields (clip, clips, playing, time, duration,
  recording, section, src, status), its "player" and "spot", "spots" = [{"name", "clip",
  "group", "player": holder or null}], "groups", "start_spot", "site" (null before the
  view names one), and gain, scale, space_held,
  space_tilted of its own spot (with spots and no spot of its own: none of these four; with
  no spots: the view's).

The phone page connects with its own id (`/ws/phone?k=<code>&id=<page id>`): a second
connection with the same id replaces the first (closed with 4002) and keeps its player and
spot, so a page never keeps two.
"""

from __future__ import annotations

import datetime
import hashlib
import hmac
import ipaddress
import json
import logging
import math
import re
import secrets
import socket
import ssl
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

PHONE_WEB = Path(__file__).with_name("live_phone")
PORT = 8766                      # 8765 is `groundtrack ui`
VIEW_PORT = 8767
# the only files the phone can fetch; the html and the manifest carry the code (__K__)
STATIC = {"/": "index.html", "/index.html": "index.html", "/phone.js": "phone.js",
          "/record.js": "record.js",
          "/phone.css": "phone.css", "/manifest.json": "manifest.json", "/icon.png": "icon.png",
          "/fonts/Geist-Variable.woff2": "fonts/Geist-Variable.woff2",
          "/fonts/SplineSansMono-Variable.woff2": "fonts/SplineSansMono-Variable.woff2"}
TEMPLATED = {"index.html", "manifest.json"}
CTYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
          ".css": "text/css; charset=utf-8", ".json": "application/manifest+json",
          ".png": "image/png", ".woff2": "font/woff2"}
CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; "
       "font-src 'self'; manifest-src 'self'; connect-src 'self' wss:; "
       "media-src 'self' blob: mediastream:; base-uri 'none'; "      # the recorder's preview
       "form-action 'none'; frame-ancestors 'none'")
SOURCES = {"web", "webcam", "freed", "arkit", "mouse", "replay"}
STATUSES = {"ok", "limited", "lost"}
# control actions and the value each one carries (space: the clutch, held / let go; level:
# the space back to level, a double-tap; spot: a spot claim, a name or null). No "clip": only
# the laptop chooses the clip / site.
ACTIONS = {"play": None, "pause": None, "seek": "number", "record": "bool",
           "section": "bool", "gain": "number", "scale": "number", "space": "bool",
           "level": None, "spot": "name"}
# what acts on the phone's own spot (the rest is shared: anyone can press it)
SPOT_ACTIONS = {"gain", "scale", "space", "level"}
STATE_FIELDS = {"clip": "text", "clips": "texts", "playing": "bool", "time": "number",
                "duration": "number", "recording": "bool", "gain": "number", "scale": "number",
                "src": "text", "status": "text", "section": "bool", "space_held": "bool",
                "space_tilted": "bool", "spots": "spots", "groups": "groups",
                "start_spot": "name", "site": "text"}
SITE_FIELDS = ("spots", "groups", "start_spot", "site")         # the site: kept apart
SPOT_FIELDS = {"clip": "text", "group": "text", "in_scene": "bool", "gain": "number",
               "scale": "number", "space_held": "bool", "space_tilted": "bool"}
TO_PHONES = ("clip", "group")                                   # of a spot, besides its holder
OWN_FIELDS = ("gain", "scale", "space_held", "space_tilted")    # a phone's: its own spot's
MAX_PLAYERS = 4
MAX_PHONE_MSG = 4096             # a phone's message (pose, control): larger ones close it (1009)
MAX_FRAME = 4_000_000            # a recorder's frame: [u32 header length][header JSON][JPEG]
MAX_FRAME_HEAD = 1024
MAX_SIDE = 4096                  # a frame's width and height, at most
CLOSE_RECORDER = 4004            # a recorder (another page) is connected already
CLOSE_REALTIME_OFF = 4005        # the laptop left the realtime mode
MAX_SPOTS = 64
NAME_MAX = 40
CLOSE_NEW_CODE = 4001            # the phone was closed because the pairing code changed
CLOSE_REPLACED = 4002            # the same page (its id) connected again: the older one goes
CLOSE_FULL = 4003                # four phones are connected already
PHONE_ID = re.compile(r"[A-Za-z0-9_-]{8,64}")
LOCAL_ORIGIN = re.compile(r"^https?://(127\.0\.0\.1|localhost|\[::1\])(:\d+)?$")
WIFI_HINTS = ("wi-fi", "wifi", "wlan", "wireless")
VIRTUAL_HINTS = ("vethernet", "virtualbox", "vmware", "hyper-v", "bluetooth", "loopback",
                 "tailscale", "zerotier", "vpn")


def _need(module: str):
    """Import one of the `.[live]` extras, or say how to install them."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise RuntimeError(f"Live mode needs {module.split('.')[0]}: "
                           f"pip install -e \".[live]\"") from e


# --------------------------------------------------------------------------- pairing + TLS
def live_dir() -> Path:
    """%APPDATA%\\groundtrack\\live (beside the app's own settings), never the project."""
    from .desktop_links import settings_dir

    return settings_dir() / "live"


def load_token(home: Path) -> str:
    """The pairing code, made once and kept (a Home Screen icon keeps working)."""
    f = Path(home) / "token.txt"
    if f.exists():
        tok = f.read_text(encoding="utf-8").strip()
        if re.fullmatch(r"[A-Za-z0-9_-]{16,64}", tok):
            return tok
    return new_token(home)


def new_token(home: Path) -> str:
    """A new pairing code (phones must scan the QR again)."""
    Path(home).mkdir(parents=True, exist_ok=True)
    tok = secrets.token_urlsafe(12)
    (Path(home) / "token.txt").write_text(tok + "\n", encoding="utf-8")
    return tok


def ensure_certificate(home: Path, ips: list[str] | None = None) -> tuple[Path, Path]:
    """cert.pem + key.pem, self-signed, made once (again only when missing or expired).

    Names localhost, 127.0.0.1 and the LAN addresses of the time it is made; Safari's
    warning is accepted once either way."""
    home = Path(home)
    cert, key = home / "cert.pem", home / "key.pem"
    x509 = _need("cryptography.x509")
    if cert.exists() and key.exists():
        try:
            c = x509.load_pem_x509_certificate(cert.read_bytes())
            if c.not_valid_after_utc > datetime.datetime.now(datetime.timezone.utc) + \
                    datetime.timedelta(days=7):
                return cert, key
        except ValueError:
            pass
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    home.mkdir(parents=True, exist_ok=True)
    pk = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "groundtrack live hub")])
    alt = [x509.DNSName("localhost")]
    for ip in dict.fromkeys(["127.0.0.1", *(ips or [])]):
        try:
            alt.append(x509.IPAddress(ipaddress.ip_address(ip)))
        except ValueError:
            pass
    now = datetime.datetime.now(datetime.timezone.utc)
    c = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
         .public_key(pk.public_key()).serial_number(x509.random_serial_number())
         .not_valid_before(now - datetime.timedelta(days=1))
         .not_valid_after(now + datetime.timedelta(days=825))     # iOS' limit for TLS certs
         .add_extension(x509.SubjectAlternativeName(alt), critical=False)
         .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
         .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
         .sign(pk, hashes.SHA256()))
    key.write_bytes(pk.private_bytes(serialization.Encoding.PEM,
                                     serialization.PrivateFormat.TraditionalOpenSSL,
                                     serialization.NoEncryption()))
    cert.write_bytes(c.public_bytes(serialization.Encoding.PEM))
    return cert, key


def fingerprint(cert: Path) -> str:
    """SHA-256 of the certificate, as Safari shows it under 'View Certificate'."""
    der = ssl.PEM_cert_to_DER_cert(Path(cert).read_text(encoding="ascii"))
    h = hashlib.sha256(der).hexdigest().upper()
    return " ".join(h[i:i + 2] for i in range(0, len(h), 2))


# --------------------------------------------------------------------------- addresses + QR
def lan_addresses() -> list[dict]:
    """This machine's IPv4 addresses a phone could reach, the likely one (default route,
    Wi-Fi / hotspot adapter) first: [{"ip", "name", "likely"}]."""
    default = None
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.0.2.1", 9))              # no packet is sent: only picks the route
        default = s.getsockname()[0]
    except OSError:
        pass
    finally:
        s.close()
    found: dict[str, str] = {}
    try:
        import psutil

        stats = psutil.net_if_stats()
        for name, addrs in psutil.net_if_addrs().items():
            if name in stats and not stats[name].isup:
                continue
            for a in addrs:
                if a.family == socket.AF_INET:
                    found.setdefault(a.address, name)
    except ImportError:
        try:
            for ai in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
                found.setdefault(ai[4][0], "")
        except OSError:
            pass
    if default:
        found.setdefault(default, "")
    out = []
    for ip, name in found.items():
        a = ipaddress.ip_address(ip)
        if a.is_loopback or a.is_link_local or a.is_unspecified:
            continue
        low = name.lower()
        wifi = any(h in low for h in WIFI_HINTS)
        virtual = any(h in low for h in VIRTUAL_HINTS)
        rank = (0 if ip == default else 1 if wifi else 3 if virtual else 2)
        out.append((rank, ip, name))
    out.sort()
    return [{"ip": ip, "name": name, "likely": rank <= 1 and rank == out[0][0]}
            for rank, ip, name in out]


def _qr_matrix(text: str, border: int = 4) -> list[list[bool]]:
    qrcode = _need("qrcode")
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, border=border)
    qr.add_data(text)
    qr.make(fit=True)
    return qr.get_matrix()


def qr_svg(text: str, dark: str = "#000", light: str = "#fff") -> str:
    """A QR code as a small SVG (dark on light, with its quiet zone; no width / height, so it
    takes the size of its box)."""
    m = _qr_matrix(text)
    n = len(m)
    d = []
    for y, row in enumerate(m):
        x = 0
        while x < n:
            if row[x]:
                run = 1
                while x + run < n and row[x + run]:
                    run += 1
                d.append(f"M{x} {y}h{run}v1h-{run}z")
                x += run
            else:
                x += 1
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {n} {n}" '
            f'shape-rendering="crispEdges"><rect width="{n}" height="{n}" fill="{light}"/>'
            f'<path fill="{dark}" d="{"".join(d)}"/></svg>')


def qr_text(text: str) -> str:
    """The QR code in half-block characters, for a dark terminal."""
    m = _qr_matrix(text, border=2)
    if len(m) % 2:
        m.append([False] * len(m[0]))
    chars = {(False, False): "█", (True, False): "▄", (False, True): "▀", (True, True): " "}
    return "\n".join("".join(chars[(m[y][x], m[y + 1][x])] for x in range(len(m[0])))
                     for y in range(0, len(m), 2))


# --------------------------------------------------------------------------- messages
def _num(v) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def _vec(v, n: int) -> list[float] | None:
    if not isinstance(v, list) or len(v) != n:
        return None
    out = [_num(x) for x in v]
    return None if any(x is None for x in out) else out


def _value(v, kind):
    """v as `kind` ("number", "text", "texts", "bool", "name", "spots"), or ... when it isn't
    one (text and name may be None)."""
    if kind == "number":
        x = _num(v)
        return ... if x is None else x
    if kind == "bool":
        return v if isinstance(v, bool) else ...
    if kind == "text":
        return v[:200] if isinstance(v, str) else None if v is None else ...
    if kind == "name":
        return v if isinstance(v, str) and 0 < len(v) <= NAME_MAX else None if v is None else ...
    if kind == "texts":
        if not isinstance(v, list) or not all(isinstance(s, str) for s in v):
            return ...
        return [s[:200] for s in v[:500]]
    if kind == "spots":
        return _spots(v)
    if kind == "groups":
        return _groups(v)
    return ...


def _groups(v):
    """The view's areas: [{"name", "spots": [names]}], at most MAX_SPOTS; an entry without a
    good name, with a name seen before, or with no spot names is left out."""
    if not isinstance(v, list):
        return ...
    out, names = [], set()
    for g in v:
        if len(out) == MAX_SPOTS:
            break
        name = _value(g.get("name"), "name") if isinstance(g, dict) else None
        spots = g.get("spots") if isinstance(g, dict) else None
        if not isinstance(name, str) or name in names or not isinstance(spots, list):
            continue
        spots = list(dict.fromkeys(s for s in spots[:MAX_SPOTS]
                                   if isinstance(_value(s, "name"), str)))
        if spots:
            names.add(name)
            out.append({"name": name, "spots": spots})
    return out


def _spots(v):
    """The view's spot list: [{"name", and what of SPOT_FIELDS is valid}], at most MAX_SPOTS;
    an entry without a good name, or with a name seen before, is left out."""
    if not isinstance(v, list):
        return ...
    out, names = [], set()
    for s in v:
        if len(out) == MAX_SPOTS:
            break
        name = _value(s.get("name"), "name") if isinstance(s, dict) else None
        if not isinstance(name, str) or name in names:
            continue
        names.add(name)
        e = {"name": name}
        for k, kind in SPOT_FIELDS.items():
            if k in s:
                x = _value(s[k], kind)
                if x is not ... and x is not None:
                    e[k] = x
        out.append(e)
    return out


def clean_phone_message(m) -> dict | None:
    """A pose, zero or control rebuilt with its known fields only (None: drop it)."""
    if not isinstance(m, dict):
        return None
    kind = m.get("type")
    if kind == "zero":
        return {"type": "zero"}
    if kind == "control":
        action = m.get("action")
        if action not in ACTIONS:
            return None
        kind_v = ACTIONS[action]
        value = None if kind_v is None else _value(m.get("value"), kind_v)
        if value is ... or (kind_v == "text" and value is None) or \
                (action in ("gain", "scale") and not value > 0):
            return None
        return {"type": "control", "action": action, "value": value}
    if kind != "pose" or m.get("v") != 1:
        return None
    q = _vec(m.get("q"), 4)
    p = m.get("p")
    p = None if p is None else _vec(p, 3) or ...
    seq, t = m.get("seq"), _num(m.get("t"))
    if q is None or p is ... or t is None or isinstance(seq, bool) or not isinstance(seq, int):
        return None
    norm = math.sqrt(sum(v * v for v in q))
    if not 0.9 < norm < 1.1 or m.get("src") not in SOURCES:
        return None
    status = m.get("status", "ok")
    if status not in STATUSES:
        return None
    return {"type": "pose", "v": 1, "src": m["src"], "seq": seq, "t": t,
            "q": [v / norm for v in q], "p": p, "status": status}


def clean_state(m) -> dict | None:
    """A view's state with its known fields only (None: not a state)."""
    if not isinstance(m, dict) or m.get("type") != "state":
        return None
    out = {"type": "state"}
    for k, kind in STATE_FIELDS.items():
        if k in m:
            v = _value(m[k], kind)
            if v is not ...:
                out[k] = v
    if not out.get("site"):                      # no name (or none): the last one stays
        out.pop("site", None)
    if "spots" not in out:                      # areas and the start: only with their spots
        out.pop("groups", None)
        out.pop("start_spot", None)
        return out
    names = {s["name"] for s in out["spots"]}
    if "groups" in out:
        out["groups"] = [{"name": g["name"], "spots": [s for s in g["spots"] if s in names]}
                         for g in out["groups"] if any(s in names for s in g["spots"])]
    if out.get("start_spot") not in names:
        out.pop("start_spot", None)
    return out


def clean_frame(data) -> tuple[dict, bytes] | None:
    """A recorder's frame, [u32 little-endian header length][header JSON][JPEG], checked: the
    header rebuilt with its known fields only ({seq, t, w, h, q, skip}: q a pose v1 quaternion,
    normalised, or None; skip the frames the phone did not send so far, or None) and the JPEG's
    start. None: drop it."""
    if not isinstance(data, (bytes, bytearray)) or not 4 < len(data) <= MAX_FRAME:
        return None
    n = int.from_bytes(data[:4], "little")
    if not 2 <= n <= MAX_FRAME_HEAD or len(data) < 4 + n + 4:
        return None
    try:
        m = json.loads(bytes(data[4:4 + n]).decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    jpeg = bytes(data[4 + n:])
    if not isinstance(m, dict) or jpeg[:2] != b"\xff\xd8":
        return None
    seq, t, w, h = _num(m.get("seq")), _num(m.get("t")), m.get("w"), m.get("h")
    if seq is None or seq < 0 or t is None:
        return None
    if not all(isinstance(v, int) and not isinstance(v, bool) and 16 <= v <= MAX_SIDE for v in (w, h)):
        return None
    q = _vec(m.get("q"), 4) if m.get("q") is not None else None
    if m.get("q") is not None and q is None:
        return None
    if q is not None:
        n_q = math.sqrt(sum(v * v for v in q))
        if not 0.5 < n_q < 2:
            return None
        q = [v / n_q for v in q]
    skip = m.get("skip")                             # a count, for the stats only: else left out
    skip = skip if isinstance(skip, int) and not isinstance(skip, bool) and 0 <= skip < 2 ** 31 else None
    return {"seq": int(seq), "t": float(t), "w": w, "h": h, "q": q, "skip": skip}, jpeg


def _parse(raw) -> dict | None:
    if not isinstance(raw, str) or len(raw) > 65536:
        return None
    try:
        m = json.loads(raw)
    except ValueError:
        return None
    return m if isinstance(m, dict) else None


def _peer(conn) -> str:
    try:
        return str(conn.remote_address[0])
    except (TypeError, IndexError, AttributeError):
        return "?"


def _ws_logger() -> logging.Logger:
    """websockets' own log, kept at warnings: its debug lines would show requests and frames."""
    lg = logging.getLogger("groundtrack.live_hub.ws")
    lg.setLevel(logging.WARNING)
    return lg


# --------------------------------------------------------------------------- hub
class _Player:
    """One phone: its number (1-4), the spot it holds (a name of the view's state.spots, or
    None), whether its SPACE is held (as last relayed to the views), whether it has had its
    chance at the start spot (placed), and when it joined (since: the first phone goes first)."""

    __slots__ = ("held", "num", "placed", "since", "spot")
    _joins = iter(range(1 << 62))

    def __init__(self, num: int):
        self.num, self.spot, self.held, self.placed = num, None, False, False
        self.since = next(_Player._joins)


class LiveHub:
    """The phone listener (LAN, TLS, pairing code) and the view listener (this machine)."""

    def __init__(self, port: int = PORT, view_port: int = VIEW_PORT, host: str = "0.0.0.0",
                 home: Path | None = None, log=print):
        self.port, self.view_port, self.host = int(port), int(view_port), host
        self.home = Path(home) if home else live_dir()
        self._log = log
        self.token = load_token(self.home)
        self.cert: Path | None = None
        self.phones: set = set()                     # every phone socket (closing ones too)
        self.players: dict = {}                      # a numbered phone's socket -> its _Player
        self.phone_ids: dict = {}                    # the page's id -> its connection
        self.seats: dict = {}                        # a page id that left -> (number, spot)
        self.views: set = set()
        self.state: dict | None = None               # the view's last state, without the site
        self.site: dict = {"spots": [], "groups": [], "start_spot": None}   # its cameras
        self.site_name: str | None = None            # the view's site (a change: a new site)
        # realtime (plans/live_controller/10): the tracker frames go to while the laptop is in
        # the realtime mode (live_track.LiveTracker; None: off, /ws/record refused), and the
        # one recorder page connected (its socket and page id)
        self.recorder = None
        self.rec_ws = None
        self.rec_id: str | None = None
        self.bound: dict[str, int] = {}
        self._loop = None
        self._stop = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None

    def log(self, msg: str) -> None:
        self._log(f"{time.strftime('%H:%M:%S')} live hub: {msg}")

    # ---- life
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and self._ready.is_set()

    def start(self, timeout: float = 15.0) -> dict:
        """Listen in a background thread; returns info() once both listeners are up."""
        if self.running:
            return self.info()
        _need("websockets")
        self.cert, self._key = ensure_certificate(self.home, [a["ip"] for a in lan_addresses()])
        self._ready.clear()
        self._error = None
        self._thread = threading.Thread(target=self._run, name="live-hub", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            raise RuntimeError("live hub did not start")
        if self._error is not None:
            e, self._error = self._error, None
            if isinstance(e, OSError):
                raise RuntimeError(f"live hub: port {self.port} or {self.view_port} is not "
                                   f"free ({e.strerror or e}); choose others with --port / "
                                   f"--view-port") from e
            raise RuntimeError(f"live hub: {e}") from e
        return self.info()

    def stop(self, timeout: float = 10.0) -> None:
        if self._thread is None:
            return
        if self._loop is not None and self._stop is not None:
            try:
                self._loop.call_soon_threadsafe(self._stop.set)
            except RuntimeError:                     # the loop has already ended
                pass
        self._thread.join(timeout)
        self._thread = None

    def wait(self) -> None:
        """Block until stopped; Ctrl+C stops it (`groundtrack live-hub`)."""
        try:
            while self._thread is not None and self._thread.is_alive():
                self._thread.join(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def rotate_token(self) -> dict:
        """A new pairing code. Phones still connected with the old one are closed."""
        self.token = new_token(self.home)
        if self.running:
            import asyncio

            asyncio.run_coroutine_threadsafe(self._close_phones(), self._loop).result(10)
        self.log("new pairing code: phones must scan the QR code again")
        return self.info()

    def info(self) -> dict:
        port = self.bound.get("lan", self.port)
        view_port = self.bound.get("view", self.view_port)
        if self.host in ("", "0.0.0.0", "::"):
            addrs = lan_addresses()
        else:
            addrs = [{"ip": self.host, "name": "", "likely": True}]
        urls = []
        for a in addrs:
            host = f"[{a['ip']}]" if ":" in a["ip"] else a["ip"]
            url = f"https://{host}:{port}/?k={self.token}"
            urls.append({**a, "url": url, "qr_svg": qr_svg(url)})
        return {"running": self.running, "port": port, "view_port": view_port,
                "url": urls[0]["url"] if urls else None, "phone_urls": urls,
                "view_url": f"ws://127.0.0.1:{view_port}/ws/view",
                "fingerprint": fingerprint(self.cert) if self.cert else None,
                "phones": len(self.players), "views": len(self.views),
                "realtime": self.recorder is not None, "recorder": self.rec_ws is not None}

    def set_recorder(self, sink) -> None:
        """The realtime mode on (sink: what takes the frames, live_track.LiveTracker) or off
        (None: the recorder page is closed). Phones are told (their Record button)."""
        self.recorder = sink
        if self.running:
            self._loop.call_soon_threadsafe(self._recorder_changed)

    def _recorder_changed(self) -> None:
        import asyncio

        if self.recorder is None and self.rec_ws is not None:
            asyncio.ensure_future(self.rec_ws.close(CLOSE_REALTIME_OFF, "realtime is off"))
        self._announce()

    # ---- loop
    def _run(self) -> None:
        import asyncio

        try:
            asyncio.run(self._main())
        except BaseException as e:                   # noqa: BLE001  (reported by start())
            self._error = e
        finally:
            self._ready.set()

    async def _main(self) -> None:
        import asyncio

        from websockets.asyncio.server import serve

        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.load_cert_chain(self.cert, self._key)
        ctx.set_alpn_protocols(["http/1.1"])
        common = {"compression": None, "server_header": None, "logger": _ws_logger(),
                  "ping_interval": 10, "ping_timeout": 10}
        # max_size: a recorder's frame; a phone's messages are held to MAX_PHONE_MSG in _phone
        async with serve(self._phone, self.host, self.port, ssl=ctx,
                         process_request=self._lan_request, max_size=MAX_FRAME, **common) as lan, \
                serve(self._view, "127.0.0.1", self.view_port,
                      process_request=self._view_request, max_size=2 ** 16, **common) as view:
            self.bound = {"lan": lan.sockets[0].getsockname()[1],
                          "view": view.sockets[0].getsockname()[1]}
            self.log(f"phone page on https://{self.host}:{self.bound['lan']}/ (pairing code "
                     f"needed), view on ws://127.0.0.1:{self.bound['view']}/ws/view")
            self._ready.set()
            await self._stop.wait()
            self.log("stopping")
        self.bound = {}

    # ---- HTTP (before the socket handshake)
    def _token_ok(self, query: str) -> bool:
        k = parse_qs(query).get("k", [""])[0]
        return hmac.compare_digest(k.encode(), self.token.encode())

    def _lan_request(self, conn, request):
        url = urlsplit(request.path)
        if not self._token_ok(url.query):
            self.log(f"refused {_peer(conn)} {url.path[:60]} (no or old pairing code)")
            return conn.respond(403, "Pairing code missing or out of date: scan the QR code in "
                                     "the motus Live view again.\n")
        if url.path == "/ws/phone":
            return None                               # on to the socket handshake
        if url.path == "/ws/record":                  # the recorder: only in the realtime mode
            if self.recorder is None:
                return conn.respond(409, "Realtime is off: pick Realtime in the motus Live view.\n")
            return None
        name = STATIC.get(url.path)
        if name is None:
            return conn.respond(404, "Not found.\n")
        return self._file(name)

    def _file(self, name: str):
        from websockets.datastructures import Headers
        from websockets.http11 import Response

        body = (PHONE_WEB / name).read_bytes()
        if name in TEMPLATED:
            body = body.replace(b"__K__", self.token.encode())
        headers = Headers([("Content-Type", CTYPES[Path(name).suffix]),
                           ("Content-Length", str(len(body))), ("Connection", "close"),
                           ("Cache-Control", "no-store"), ("Referrer-Policy", "no-referrer"),
                           ("X-Content-Type-Options", "nosniff"),
                           ("Content-Security-Policy", CSP)])
        return Response(200, "OK", headers, body)

    def _view_request(self, conn, request):
        origin = request.headers.get("Origin")
        if urlsplit(request.path).path != "/ws/view":
            return conn.respond(404, "Not found.\n")
        if origin is not None and not LOCAL_ORIGIN.match(origin):
            self.log(f"refused a view from another site ({origin[:60]})")
            return conn.respond(403, "Only pages on this machine.\n")
        return None

    # ---- sockets
    async def _phone(self, ws) -> None:
        import asyncio

        from websockets.exceptions import ConnectionClosed

        if urlsplit(ws.request.path).path == "/ws/record":
            return await self._record(ws)
        ip, t0 = _peer(ws), time.monotonic()
        poses = dropped = 0
        pid = parse_qs(urlsplit(ws.request.path).query).get("id", [""])[0]
        pid = pid if PHONE_ID.fullmatch(pid) else None
        old = self.phone_ids.get(pid) if pid else None
        pl = self.players.pop(old, None) if old is not None else None
        if pl is None:                               # a new phone (not the same page again)
            seat = self.seats.get(pid) if pid else None
            num = self._free_number(seat[0] if seat else None)
            if num is None:
                self.log(f"refused a phone from {ip}: {MAX_PLAYERS} phones are connected")
                await ws.close(CLOSE_FULL, f"{MAX_PLAYERS} phones are connected")
                return
            self.seats.pop(pid, None)
            pl = _Player(num)
            if seat and seat[1] is not None and self._claim(pl, seat[1]):
                pl.placed = True                     # back on its spot: it was still free
        if pid:
            self.phone_ids[pid] = ws
        self.phones.add(ws)
        self.players[ws] = pl
        self._auto_place()                           # the first phone: the start camera
        self.log(f"player {pl.num} connected from {ip} ({len(self.players)} connected)")
        if old is not None:                          # one socket per page, whatever the page did
            self.log(f"player {pl.num}: its older connection is replaced")
            self._let_go(pl)                         # the newer one starts let go
            asyncio.ensure_future(old.close(CLOSE_REPLACED, "replaced by a newer connection"))
        self._announce()
        self._send_states(None if pl.spot else ws)   # who holds which, when it came back on one
        try:
            async for raw in ws:
                if len(raw) > MAX_PHONE_MSG:          # as websockets' own max_size did
                    await ws.close(1009, "message too big")
                    break
                m = _parse(raw)
                if m is not None and m.get("type") == "ping":
                    await ws.send(json.dumps({"type": "pong", "id": _num(m.get("id"))}))
                    continue
                out = clean_phone_message(m)
                if out is None:
                    dropped += 1
                    continue
                poses += out["type"] == "pose"
                if self.players.get(ws) is pl:       # (not when replaced by a newer one)
                    self._from_phone(ws, pl, out)
        except ConnectionClosed:
            pass
        finally:
            self.phones.discard(ws)
            if pid and self.phone_ids.get(pid) is ws:
                del self.phone_ids[pid]
            mine = self.players.get(ws) is pl
            if mine:
                del self.players[ws]
                self._let_go(pl)                     # never stuck: a lost phone lets go
                if pid:
                    self.seats[pid] = (pl.num, pl.spot)
                    while len(self.seats) > 32:
                        self.seats.pop(next(iter(self.seats)))
            dt = max(time.monotonic() - t0, 1e-6)
            self.log(f"player {pl.num} ({ip}) left after {dt:.0f} s: {poses} poses "
                     f"({poses / dt:.1f} Hz)"
                     + (f", {dropped} unknown messages dropped" if dropped else ""))
            self._announce()
            if mine and pl.spot is not None:
                self._send_states()                  # its spot is free again

    async def _record(self, ws) -> None:
        """The recorder (plans/live_controller/10): one page at a time. On connecting it gets
        {"type": "rec", "hello": true, "long_side"} (the laptop's frame size; it starts after).
        Text: {"type": "rec", "action": "start", "w", "h", "resume", "height_m", "tilt_offset"
        (both optional: the phone's calibration)} / {"type": "rec", "action": "stop"} / ping;
        binary: frames (clean_frame) to the realtime tracker. It gets {"type":
        "rec", ...the tracker's status} twice a second. Only connections and recordings are
        logged, never frames."""
        import asyncio

        from websockets.exceptions import ConnectionClosed

        ip, t0 = _peer(ws), time.monotonic()
        pid = parse_qs(urlsplit(ws.request.path).query).get("id", [""])[0]
        pid = pid if PHONE_ID.fullmatch(pid) else None
        old = self.rec_ws
        if old is not None and (pid is None or pid != self.rec_id):
            self.log(f"refused a recorder from {ip}: one is connected")
            await ws.close(CLOSE_RECORDER, "a recorder is connected")
            return
        self.rec_ws, self.rec_id = ws, pid
        if old is not None:                          # the same page again: the older one goes
            asyncio.ensure_future(old.close(CLOSE_REPLACED, "replaced by a newer connection"))
        self.log(f"recorder connected from {ip}")
        self._announce()
        frames = dropped = 0
        recording = False
        sink = self.recorder
        try:
            await ws.send(json.dumps({"type": "rec", "hello": True,
                                      "long_side": getattr(sink, "long_side", 1920)}))
        except ConnectionClosed:
            pass

        async def report():
            try:
                while True:
                    await asyncio.sleep(0.5)
                    sink = self.recorder
                    if sink is not None:
                        await ws.send(json.dumps({"type": "rec", **sink.status()}))
            except ConnectionClosed:
                pass

        ticker = asyncio.ensure_future(report())
        try:
            async for raw in ws:
                sink = self.recorder
                if sink is None:
                    break
                if isinstance(raw, (bytes, bytearray)):
                    fr = clean_frame(raw)
                    if fr is None or not recording or not sink.frame(*fr):
                        dropped += 1
                        continue
                    frames += 1
                    continue
                m = _parse(raw)
                if m is None:
                    dropped += 1
                elif m.get("type") == "ping":
                    await ws.send(json.dumps({"type": "pong", "id": _num(m.get("id"))}))
                elif m.get("type") == "rec" and m.get("action") == "start":
                    w, h = m.get("w"), m.get("h")
                    try:
                        if not all(isinstance(v, int) and not isinstance(v, bool) for v in (w, h)):
                            raise ValueError("frame size: two whole numbers")
                        kw = {}
                        for key in ("height_m", "tilt_offset"):     # the phone's calibration (step 2)
                            if m.get(key) is not None:
                                if _num(m[key]) is None:
                                    raise ValueError(f"{key}: a number")
                                kw[key] = _num(m[key])
                        res = sink.begin(w, h, resume=m.get("resume") is True, **kw)
                    except (RuntimeError, ValueError) as e:
                        await ws.send(json.dumps({"type": "rec", "error": str(e)}))
                        continue
                    recording = True
                    self.log(("recording resumed" if res["resumed"] else "recording started")
                             + f": {res['id']}")
                    await ws.send(json.dumps({"type": "rec", "started": res["id"],
                                              "resumed": res["resumed"]}))
                elif m.get("type") == "rec" and m.get("action") == "stop":
                    if recording:
                        sink.end()
                        recording = False
                        self.log("recording stopped by the recorder")
                    await ws.send(json.dumps({"type": "rec", "stopped": True}))
                else:
                    dropped += 1
        except ConnectionClosed:
            pass
        finally:
            ticker.cancel()
            mine = self.rec_ws is ws
            if mine:
                self.rec_ws = None
                if recording and self.recorder is not None:
                    self.recorder.lost()             # it may come back (live_track.LOST_S)
            dt = max(time.monotonic() - t0, 1e-6)
            self.log(f"recorder ({ip}) left after {dt:.0f} s: {frames} frames ({frames / dt:.1f} "
                     f"fps)" + (f", {dropped} dropped" if dropped else ""))
            if mine:
                self._announce()

    def _from_phone(self, ws, pl: _Player, out: dict) -> None:
        """One checked message of a numbered phone: a claim, or relayed to the views."""
        action = out.get("action")
        if action == "spot":
            if self._claim(pl, out["value"]):
                self._announce()
                self._send_states()
            else:                                    # refused or no change: show it the truth
                self._send_states(ws)
            return
        if (out["type"] in ("pose", "zero") or action in SPOT_ACTIONS) and pl.spot is None \
                and self._spot_names():
            return                                   # spots, and none is this phone's
        if action == "space":
            if out["value"] == pl.held:
                return                               # presses and let-goes only
            pl.held = out["value"]
        out["player"], out["spot"] = pl.num, pl.spot
        self._to_views(out)

    async def _view(self, ws) -> None:
        from websockets.exceptions import ConnectionClosed

        self.views.add(ws)
        self.log(f"view connected ({len(self.views)} connected)")
        self._announce()
        try:
            async for raw in ws:
                m = _parse(raw)
                if m is not None and m.get("type") == "release":
                    self._release(_value(m.get("spot"), "name"))
                    continue
                st = clean_state(m)
                if st is not None:
                    self._view_state(st)
        except ConnectionClosed:
            pass
        finally:
            self.views.discard(ws)
            self.log(f"view left ({len(self.views)} connected)")
            self._announce()

    def _view_state(self, st: dict) -> None:
        """A checked state of the view: kept (the site only when it lists its spots), the
        claims checked against the spots, the start camera handed out, each phone told."""
        site = {k: st.pop(k) for k in SITE_FIELDS if k in st}
        self.state = st
        changed = False
        name = site.get("site")                      # only its name says the site changed
        switched = None not in (name, self.site_name) and name != self.site_name
        if name is not None:
            self.site_name = name
        if "spots" in site:
            self.site = {"spots": site["spots"], "groups": site.get("groups", []),
                         "start_spot": site.get("start_spot")}
        names = set(self._spot_names())
        for p in self.players.values():
            if switched or (p.spot is not None and p.spot not in names):
                changed = changed or p.spot is not None
                self._claim(p, None)                 # a new site, or a camera the view dropped
                p.placed = False                     # not its choice: it may get the start
        if switched:                                 # pages coming back: their number only
            self.seats = {k: (num, None) for k, (num, _) in self.seats.items()}
        changed = self._auto_place() or changed
        if changed:
            self._announce()
        self._send_states()

    def _release(self, name) -> None:
        """The laptop took the camera `name` out of the scene: its phone (if any) lets it go,
        stays connected, and picks again (no start camera for it)."""
        pl = next((p for p in self.players.values() if name and p.spot == name), None)
        if pl is None:
            return                                   # no one holds it (or not a name)
        self._claim(pl, None)
        self._announce()
        self._send_states()

    # ---- players and spots
    def _spot_names(self) -> list[str]:
        return [s["name"] for s in self.site["spots"]]

    def _auto_place(self) -> bool:
        """Phones not placed yet (just joined, or the view dropped their camera), first come
        first: one without a spot takes the start spot when it's free; the others pick. Once
        the view lists spots, each phone gets this one chance. Whether a claim changed."""
        start = self.site["start_spot"]
        if not self._spot_names():
            return False
        changed = False
        for pl in sorted(self.players.values(), key=lambda p: p.since):
            if not pl.placed:
                pl.placed = True
                if pl.spot is None and start is not None and self._claim(pl, start):
                    changed = True
        return changed

    def _free_number(self, prefer: int | None = None) -> int | None:
        taken = {p.num for p in self.players.values()}
        if prefer is not None and prefer not in taken:
            return prefer
        return next((n for n in range(1, MAX_PLAYERS + 1) if n not in taken), None)

    def _claim(self, pl: _Player, name: str | None) -> bool:
        """pl takes the spot `name` (None: lets its spot go). One phone per spot: a taken or
        unlisted spot is refused. Whether anything changed."""
        if name == pl.spot:
            return False
        if name is not None and (name not in self._spot_names()
                                 or any(p.spot == name for p in self.players.values())):
            return False
        self._let_go(pl)                             # SPACE never follows it to another spot
        pl.spot = name
        return True

    def _let_go(self, pl: _Player) -> None:
        if pl.held:
            pl.held = False
            self._to_views({"type": "control", "action": "space", "value": False,
                            "player": pl.num, "spot": pl.spot})

    def _phone_state(self, pl: _Player) -> dict:
        """The view's state as this phone sees it: its number, its spot and its spot's own
        fields, every spot with who holds it, the areas, the start spot and the site."""
        st = dict(self.state)
        spots = self.site["spots"]
        holder = {p.spot: p.num for p in self.players.values() if p.spot is not None}
        st.update(player=pl.num, spot=pl.spot,
                  spots=[{"name": s["name"], **{k: s[k] for k in TO_PHONES if k in s},
                          "player": holder.get(s["name"])} for s in spots],
                  groups=self.site["groups"], start_spot=self.site["start_spot"],
                  site=self.site_name)
        if spots:
            own = next((s for s in spots if s["name"] == pl.spot), {})
            for k in OWN_FIELDS:
                st.pop(k, None)
                if k in own:
                    st[k] = own[k]
        return st

    def _send_states(self, only=None) -> None:
        """Each phone (or only that one) its own state, once the view has sent one."""
        from websockets.asyncio.server import broadcast

        if self.state is None:
            return
        for ws, pl in list(self.players.items()):
            if only is None or ws is only:
                broadcast([ws], json.dumps(self._phone_state(pl)))

    def _to_views(self, msg: dict) -> None:
        from websockets.asyncio.server import broadcast

        broadcast(self.views, json.dumps(msg))

    def _announce(self) -> None:
        from websockets.asyncio.server import broadcast

        base = {"type": "hub", "phones": len(self.players), "views": len(self.views)}
        players = [{"player": p.num, "spot": p.spot}
                   for p in sorted(self.players.values(), key=lambda p: p.num)]
        broadcast(self.views, json.dumps({**base, "players": players}))
        rt = {"realtime": True} if self.recorder is not None else {}    # the phone's Record button
        for ws, pl in list(self.players.items()):
            broadcast([ws], json.dumps({**base, "player": pl.num, **rt}))

    async def _close_phones(self) -> None:
        for ws in list(self.phones):
            await ws.close(CLOSE_NEW_CODE, "new pairing code")
