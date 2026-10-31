"""Live hub (`groundtrack live-hub`): the pairing code, the relay phone -> view, the state
fan-out view -> phones, what is logged, the QR code, and the phone page's orientation maths
(run in node on hand-worked poses). Everything listens on 127.0.0.1 only."""

import json
import math
import re
import shutil
import ssl
import subprocess
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

pytest.importorskip("websockets")
pytest.importorskip("cryptography")
pytest.importorskip("qrcode")

from websockets.exceptions import ConnectionClosed, InvalidStatus
from websockets.sync.client import connect

from groundtrack import live_hub
from groundtrack.live_hub import LiveHub, qr_svg, qr_text

REAL_LIVE_DIR = live_hub.live_dir


@pytest.fixture(autouse=True)
def _not_the_real_appdata(tmp_path, monkeypatch):
    """The hub keeps its code and certificate per user: never touch the real %APPDATA%."""
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setattr(live_hub, "live_dir", lambda: tmp_path / "appdata" / "groundtrack" / "live")


@pytest.fixture
def hub(tmp_path):
    logs = []
    h = LiveHub(port=0, view_port=0, host="127.0.0.1", home=tmp_path / "live", log=logs.append)
    h.logs = logs
    h.start()
    yield h
    h.stop()


def tls(h):
    """Trusts the hub's own certificate. At most TLS 1.2: on TLS 1.3, websockets' sync client
    (it reads in a thread while the caller writes) leaves about 1 connect in 100-300 waiting
    for its handshake response for good, against any TLS 1.3 server (a plain threaded one too).
    The hub serves TLS 1.3 fine: 0 in 1500 with the asyncio client."""
    ctx = ssl.create_default_context(cafile=str(h.cert))
    ctx.maximum_version = ssl.TLSVersion.TLSv1_2
    return ctx


def phone(h, token=None, pid=None):
    k = h.token if token is None else token
    return connect(f"wss://127.0.0.1:{h.bound['lan']}/ws/phone?k={k}"
                   + (f"&id={pid}" if pid else ""), ssl=tls(h), open_timeout=5)


def mine(n=1, spot=None):
    """What the hub adds to everything it relays from a phone."""
    return {"player": n, "spot": spot}


def until(fn, timeout=3.0):
    """Wait until fn() is true (the hub works in its own thread)."""
    end = time.time() + timeout
    while time.time() < end:
        if fn():
            return True
        time.sleep(0.02)
    return False


def logged(h, pattern, timeout=3.0):
    return until(lambda: any(re.search(pattern, line) for line in h.logs), timeout)


def view(h, origin=None):
    return connect(f"ws://127.0.0.1:{h.bound['view']}/ws/view", origin=origin, open_timeout=5)


def get(h, path):
    """(status, headers, body) of an HTTPS GET on the phone listener."""
    try:
        with urllib.request.urlopen(f"https://127.0.0.1:{h.bound['lan']}{path}", context=tls(h),
                                    timeout=5) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def recv(ws, kind, timeout=3.0, pred=lambda m: True):
    """The next message of this type (and pred); others, e.g. hub counts, are skipped."""
    end = time.time() + timeout
    while True:
        m = json.loads(ws.recv(timeout=max(0.01, end - time.time())))
        if m["type"] == kind and pred(m):
            return m


def pose(seq=1, q=(1, 0, 0, 0), **extra):
    return {"type": "pose", "v": 1, "src": "web", "seq": seq, "t": 12.5, "q": list(q),
            "p": None, "status": "ok", **extra}


# --------------------------------------------------------------------------- code + certificate
def test_code_and_certificate_are_kept_beside_each_other_outside_the_project(tmp_path):
    home = tmp_path / "live"
    a = LiveHub(home=home)
    assert re.fullmatch(r"[A-Za-z0-9_-]{16}", a.token)
    old = a.token
    assert LiveHub(home=home).token == old                        # kept between sessions
    a.rotate_token()
    assert a.token != old and LiveHub(home=home).token == a.token
    cert, _ = live_hub.ensure_certificate(home, ["10.0.0.7"])
    first = cert.read_bytes()
    assert live_hub.ensure_certificate(home)[0].read_bytes() == first    # made once
    from cryptography import x509

    c = x509.load_pem_x509_certificate(first)
    san = c.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert {str(ip) for ip in san.get_values_for_type(x509.IPAddress)} >= {"127.0.0.1",
                                                                         "10.0.0.7"}
    assert "localhost" in san.get_values_for_type(x509.DNSName)
    assert sorted(p.name for p in home.iterdir()) == ["cert.pem", "key.pem", "token.txt"]
    assert str(live_hub.live_dir()).startswith(str(tmp_path / "appdata"))


def test_default_folder_is_appdata_groundtrack_live(tmp_path, monkeypatch):
    from groundtrack import desktop_links

    monkeypatch.setattr(desktop_links.sys, "platform", "win32")
    assert REAL_LIVE_DIR() == tmp_path / "appdata" / "groundtrack" / "live"


# --------------------------------------------------------------------------- phone listener
def test_page_and_files_need_the_pairing_code(hub):
    k = hub.token
    st, hd, body = get(hub, f"/?k={k}")
    assert st == 200 and hd["Content-Type"].startswith("text/html")
    assert f"phone.js?k={k}".encode() in body and b"__K__" not in body
    assert hd["Cache-Control"] == "no-store" and "default-src 'none'" in hd["Content-Security-Policy"]
    assert hd["Server"] is None
    st, hd, body = get(hub, f"/phone.js?k={k}")
    assert st == 200 and "javascript" in hd["Content-Type"] and b"cameraQuaternion" in body
    st, _, body = get(hub, f"/manifest.json?k={k}")
    assert st == 200 and json.loads(body)["start_url"] == f"./?k={k}"
    st, hd, body = get(hub, f"/icon.png?k={k}")
    assert st == 200 and body[:8] == b"\x89PNG\r\n\x1a\n"
    for font in ("Geist-Variable", "SplineSansMono-Variable"):
        st, hd, body = get(hub, f"/fonts/{font}.woff2?k={k}")
        assert st == 200 and hd["Content-Type"] == "font/woff2" and body[:4] == b"wOF2"
        assert "font-src 'self'" in hd["Content-Security-Policy"]
    for path in ("/", "/?k=", f"/?k={k}x", "/phone.js", "/manifest.json", "/live_hub.py",
                 f"/?K={k}", "/ws/phone", "/fonts/Geist-Variable.woff2"):
        assert get(hub, path)[0] == 403, path
    for path in ("/live_hub.py", "/../live_hub.py", "/ws/view", "/fonts/x.woff2",
                 "/%2e%2e/cli.py"):
        assert get(hub, f"{path}?k={k}")[0] == 404, path
    assert not any(k in line for line in hub.logs)


def test_sockets_refuse_everything_without_the_code(hub):
    for bad in ("", "nope", hub.token[:-1]):
        with pytest.raises(InvalidStatus) as e:
            phone(hub, bad)
        assert e.value.response.status_code == 403
    with pytest.raises(InvalidStatus) as e:                      # no view socket on the LAN
        connect(f"wss://127.0.0.1:{hub.bound['lan']}/ws/view?k={hub.token}", ssl=tls(hub))
    assert e.value.response.status_code == 404
    with pytest.raises(InvalidStatus) as e:                      # and no phone on the view port
        connect(f"ws://127.0.0.1:{hub.bound['view']}/ws/phone?k={hub.token}")
    assert e.value.response.status_code == 404
    with pytest.raises(InvalidStatus) as e:                      # a web page from elsewhere
        view(hub, origin="https://example.com")
    assert e.value.response.status_code == 403
    with view(hub, origin="http://127.0.0.1:5173") as v:          # the app's own page
        assert recv(v, "hub")["views"] == 1


def test_junk_requests_are_refused_quietly(hub, capfd):
    import socket

    def raw(data, tls_on=True):
        s = socket.create_connection(("127.0.0.1", hub.bound["lan"]), timeout=3)
        try:
            if tls_on:
                s = tls(hub).wrap_socket(s, server_hostname="127.0.0.1")
            s.sendall(data)
            out = b""
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    return out
                out += chunk
        except (OSError, ssl.SSLError):
            return out
        finally:
            s.close()

    k = hub.token
    assert not raw(f"POST /?k={k} HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n\r\nhi".encode()
                   ).startswith(b"HTTP/1.1 200")                    # dropped, nothing served
    assert raw(f"GET /ws/phone?k={k} HTTP/1.1\r\nHost: x\r\n\r\n".encode()
               ).startswith(b"HTTP/1.1 426")
    raw(b"\x00\xff garbage \r\n\r\n")
    assert not raw(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n", tls_on=False).startswith(b"HTTP/1.1 200")
    with phone(hub) as p:                                         # still serving
        assert recv(p, "hub")["phones"] == 1
    err = capfd.readouterr().err
    assert err == "" and not any(k in line for line in hub.logs)


# --------------------------------------------------------------------------- relay
def test_phone_poses_and_remote_reach_the_view_and_nothing_else(hub):
    with view(hub) as v:
        assert recv(v, "hub") == {"type": "hub", "phones": 0, "views": 1, "players": []}
        with phone(hub) as p:
            assert recv(v, "hub") == {"type": "hub", "phones": 1, "views": 1,
                                      "players": [{"player": 1, "spot": None}]}
            assert recv(p, "hub") == {"type": "hub", "phones": 1, "views": 1, "player": 1}
            p.send(json.dumps(pose(7, (1.02, 0, 0, 0.0), frame="AAAA", img=[1, 2],
                                   player=3, spot="North")))
            m = recv(v, "pose")
            assert m == {**pose(7, (1, 0, 0, 0)), **mine()}      # known fields only, q unit
            junk = [{"type": "video", "data": "AAAA"}, pose(q=(1, 0, 0)), pose(v=2),
                    pose(src="camera"), pose(q=(0, 0, 0, 0)), pose(status="great"),
                    pose(t="now"), pose(p=[1, 2]), pose(seq=True),
                    {"type": "control", "action": "seek", "value": "abc"},
                    {"type": "control", "action": "delete", "value": None},
                    {"type": "control", "action": "gain", "value": -1},
                    {"type": "control", "action": "clip", "value": None},
                    # only the laptop chooses the clip / site: a phone's clip is dropped
                    {"type": "control", "action": "clip", "value": "richmond_6462"},
                    [1, 2], "x"]
            for j in junk:
                p.send(json.dumps(j))
            p.send(b"\x00binary")
            p.send("{not json")
            p.send(json.dumps({"type": "zero", "extra": 1}))
            p.send(json.dumps({"type": "control", "action": "gain", "value": 1.5, "x": 2}))
            p.send(json.dumps({"type": "control", "action": "play", "value": "whatever"}))
            p.send(json.dumps(pose(8, (0, 1, 0, 0), p=[1.0, 2.0, 0.5])))
            got = [json.loads(v.recv(timeout=3)) for _ in range(4)]
            assert got == [{"type": "zero", **mine()},
                           {"type": "control", "action": "gain", "value": 1.5, **mine()},
                           {"type": "control", "action": "play", "value": None, **mine()},
                           {**pose(8, (0, 1, 0, 0), p=[1.0, 2.0, 0.5]), **mine()}]
            p.send(json.dumps({"type": "ping", "id": 41, "secret": "x"}))
            assert recv(p, "pong") == {"type": "pong", "id": 41}
        assert recv(v, "hub")["phones"] == 0
        with pytest.raises(TimeoutError):                         # the ping never reached it
            recv(v, "ping", timeout=0.3)


def test_many_poses_arrive_in_order_and_fast(hub):
    with view(hub) as v, phone(hub) as p:
        recv(v, "hub")
        for i in range(300):
            p.send(json.dumps(pose(i)))
        seqs = [recv(v, "pose")["seq"] for _ in range(300)]
        assert seqs == list(range(300))
        lat = []
        for i in range(40):                                       # one at a time: hub delay
            t0 = time.perf_counter()
            p.send(json.dumps(pose(1000 + i)))
            recv(v, "pose")
            lat.append(time.perf_counter() - t0)
        assert float(np.median(lat)) < 0.02


def test_state_fans_out_to_every_phone_and_waits_for_new_ones(hub):
    state = {"type": "state", "clip": "richmond_6462", "clips": ["richmond_6462", "trafalgar"],
             "playing": True, "time": 3.25, "duration": 64.0, "recording": False, "gain": 1.0,
             "scale": 1.0, "src": "web", "status": "ok", "section": False}
    alone = {"spot": None, "spots": [], "groups": [], "start_spot": None,
             "site": None}                                          # one clip
    with phone(hub) as p1, phone(hub) as p2, view(hub) as v:
        v.send(json.dumps({**state, "frames": ["AAAA"], "tracks": [[1, 2]], "video": "b64"}))
        assert recv(p1, "state") == {**state, **alone, "player": 1}
        assert recv(p2, "state") == {**state, **alone, "player": 2}
        v.send(json.dumps(pose(3)))                               # a view sends state only
        v.send(json.dumps({"type": "state", "time": "soon", "clips": [1, 2], "playing": 1,
                           "clip": "trafalgar"}))
        assert recv(p1, "state") == {"type": "state", "clip": "trafalgar", **alone, "player": 1}
        with pytest.raises(TimeoutError):
            recv(p1, "pose", timeout=0.3)
        with phone(hub) as p3:                                    # joins later: the last state
            assert recv(p3, "state") == {"type": "state", "clip": "trafalgar", **alone,
                                         "player": 3}


def test_new_code_closes_connected_phones_and_refuses_the_old_one(hub):
    old = hub.token
    with phone(hub) as p:
        recv(p, "hub")
        info = hub.rotate_token()
        assert hub.token != old and hub.token in info["url"]
        with pytest.raises(ConnectionClosed) as e:
            while True:
                p.recv(timeout=3)
        assert e.value.rcvd.code == live_hub.CLOSE_NEW_CODE
    with pytest.raises(InvalidStatus):
        phone(hub, old)
    assert get(hub, f"/?k={old}")[0] == 403
    with phone(hub) as p:
        assert recv(p, "hub")["phones"] == 1


def test_log_shows_connections_never_payloads(hub):
    with view(hub) as v, phone(hub) as p:
        p.send(json.dumps(pose(5, (0.123456, 0, 0, 0.992350))))
        p.send(json.dumps({"type": "control", "action": "clip", "value": "SECRET_CLIP"}))
        v.send(json.dumps({"type": "state", "clips": ["OTHER_SECRET"]}))
        recv(v, "pose")
        recv(p, "state")
    assert logged(hub, r"player 1 \(127\.0\.0\.1\) left after \d+ s: 1 poses")
    text = "\n".join(hub.logs)
    assert "player 1 connected from 127.0.0.1" in text and "view connected" in text
    for secret in ("SECRET_CLIP", "OTHER_SECRET", "0.123456", "0.99235", hub.token):
        assert secret not in text


def test_info_restart_and_busy_port(hub, tmp_path):
    info = hub.info()
    assert info["running"] and info["phone_urls"][0]["ip"] == "127.0.0.1"
    assert info["url"] == f"https://127.0.0.1:{hub.bound['lan']}/?k={hub.token}"
    assert info["view_url"] == f"ws://127.0.0.1:{hub.bound['view']}/ws/view"
    assert info["phone_urls"][0]["qr_svg"].startswith("<svg")
    assert re.fullmatch(r"([0-9A-F]{2} ){31}[0-9A-F]{2}", info["fingerprint"])
    other = LiveHub(port=hub.bound["lan"], view_port=0, host="127.0.0.1", home=tmp_path / "o",
                    log=lambda s: None)
    with pytest.raises(RuntimeError, match="not free"):
        other.start()
    ports = dict(hub.bound)
    hub.stop()
    assert not hub.running and not hub.info()["running"]
    hub.port, hub.view_port = ports["lan"], ports["view"]
    hub.start()                                                   # same ports again
    with view(hub) as v:
        assert recv(v, "hub")["views"] == 1


# --------------------------------------------------------------------------- space clutch
def test_space_and_level_reach_the_view_bad_values_do_not(hub):
    with view(hub) as v, phone(hub) as p:
        for m in ({"type": "control", "action": "space", "value": True},
                  {"type": "control", "action": "space", "value": "yes"},
                  {"type": "control", "action": "space", "value": 1},
                  {"type": "control", "action": "space"},
                  {"type": "control", "action": "map", "value": "hinge"},
                  {"type": "control", "action": "level", "value": "now"},
                  {"type": "control", "action": "level"},
                  {"type": "control", "action": "space", "value": False}):
            p.send(json.dumps(m))
        got = [recv(v, "control") for _ in range(4)]
        assert got == [{"type": "control", "action": "space", "value": True, **mine()},
                       {"type": "control", "action": "level", "value": None, **mine()},
                       {"type": "control", "action": "level", "value": None, **mine()},
                       {"type": "control", "action": "space", "value": False, **mine()}]
        alone = {"player": 1, "spot": None, "spots": [], "groups": [], "start_spot": None,
                 "site": None}
        v.send(json.dumps({"type": "state", "space_held": True, "space_tilted": False}))
        assert recv(p, "state") == {"type": "state", "space_held": True, "space_tilted": False,
                                    **alone}
        v.send(json.dumps({"type": "state", "space_held": "yes", "space_tilted": 1, "clip": "c"}))
        assert recv(p, "state") == {"type": "state", "clip": "c", **alone}


def test_space_is_presses_and_let_goes_only(hub):
    space = lambda on: json.dumps({"type": "control", "action": "space", "value": on})
    with view(hub) as v, phone(hub) as p:
        for on in (False, True, True, False, False, True):        # a page's let-go on connect too
            p.send(space(on))
        p.send(json.dumps({"type": "control", "action": "play"}))
        got = [recv(v, "control") for _ in range(4)]
        assert [(m["action"], m["value"]) for m in got] == [("space", True), ("space", False),
                                                            ("space", True), ("play", None)]


def test_a_phone_that_leaves_while_holding_lets_go(hub):
    with view(hub) as v:
        with phone(hub) as p:
            p.send(json.dumps({"type": "control", "action": "space", "value": True}))
            assert recv(v, "control")["value"] is True
        assert recv(v, "control") == {"type": "control", "action": "space", "value": False,
                                      **mine()}
        with phone(hub) as p:                                     # let go before leaving: once
            p.send(json.dumps({"type": "control", "action": "space", "value": True}))
            p.send(json.dumps({"type": "control", "action": "space", "value": False}))
            assert [recv(v, "control")["value"] for _ in range(2)] == [True, False]
        with pytest.raises(TimeoutError):
            recv(v, "control", timeout=0.4)


def test_one_socket_per_page_id_and_four_players(hub):
    with view(hub) as v, phone(hub, pid="page-aaaaaaaa") as first:
        assert recv(first, "hub")["player"] == 1
        first.send(json.dumps({"type": "control", "action": "space", "value": True}))
        assert recv(v, "control")["value"] is True
        with phone(hub, pid="page-aaaaaaaa") as again:            # the same page again
            with pytest.raises(ConnectionClosed) as e:
                while True:
                    first.recv(timeout=3)
            assert e.value.rcvd.code == live_hub.CLOSE_REPLACED
            assert recv(v, "control") == {"type": "control", "action": "space", "value": False,
                                          **mine()}
            assert recv(again, "hub")["player"] == 1               # keeps its number
            time.sleep(0.2)
            assert hub.info()["phones"] == 1
            with phone(hub, pid="page-bbbbbbbb") as other, phone(hub) as no_id, phone(hub) as no_id2:
                assert [recv(ws, "hub")["player"] for ws in (other, no_id, no_id2)] == [2, 3, 4]
                time.sleep(0.2)
                assert hub.info()["phones"] == 4                 # other pages and old pages stay
                again.send(json.dumps(pose(9)))
                assert recv(v, "pose") == {**pose(9), **mine(1)}
                no_id2.send(json.dumps(pose(10)))
                assert recv(v, "pose")["player"] == 4
                with phone(hub, pid="page-cccccccc") as fifth:    # four is the most
                    with pytest.raises(ConnectionClosed) as e:
                        while True:
                            fifth.recv(timeout=3)
                    assert e.value.rcvd.code == live_hub.CLOSE_FULL
                for ws in (other, no_id, no_id2):
                    ws.ping().wait(2)
                assert hub.info()["phones"] == 4
            assert logged(hub, r"refused a phone from 127\.0\.0\.1: 4 phones")
        assert until(lambda: not hub.players)
        with phone(hub) as late:                                  # the lowest free number
            assert recv(late, "hub")["player"] == 1
    assert not any("page-aaaaaaaa" in line for line in hub.logs)


# --------------------------------------------------------------------------- spots (several phones)
SPOTS = [{"name": "North", "clip": "trafalgar6354"}, {"name": "East", "clip": "trafalgar6376"},
         {"name": "Fountain", "clip": "trafalgar6363"}, {"name": "South", "clip": "trafalgar6367"}]
SHARED = {"type": "state", "clip": "trafalgar6354", "playing": False, "time": 1.5,
          "duration": 60.0, "gain": 1.0, "space_held": False, "space_tilted": False}


def claim(ws, spot):
    ws.send(json.dumps({"type": "control", "action": "spot", "value": spot}))


def holders(st):
    return {s["name"]: s["player"] for s in st["spots"]}


def test_spots_one_phone_each_and_each_phone_its_own_state(hub):
    with view(hub) as v, phone(hub) as a, phone(hub) as b, phone(hub) as c:
        claim(a, "North")                                         # no spots yet: refused
        v.send(json.dumps({**SHARED, "spots": [
            {**SPOTS[0], "gain": 2.0, "space_tilted": True, "frames": ["AAAA"]},
            {**SPOTS[1], "space_held": True}, *SPOTS[2:],
            {"name": "North"}, {"name": ""}, {"name": 7}, "South", {"clip": "x"}]}))
        st = recv(a, "state")
        assert st["spot"] is None and [s["name"] for s in st["spots"]] == [s["name"] for s in SPOTS]
        assert st["spots"][0] == {"name": "North", "clip": "trafalgar6354", "player": None}
        assert not set(st) & {"gain", "space_held", "space_tilted"}   # no spot: none of its own
        assert st["clip"] == "trafalgar6354" and st["time"] == 1.5   # shared
        claim(a, "North")
        on_north = lambda m: {"player": 1, "spot": "North"} in m["players"]
        assert recv(v, "hub", pred=on_north)["players"] == [{"player": 1, "spot": "North"},
                                                             {"player": 2, "spot": None},
                                                             {"player": 3, "spot": None}]
        st = recv(a, "state")
        assert (st["player"], st["spot"], st["gain"], st["space_tilted"]) == (1, "North", 2.0, True)
        assert "space_held" not in st                             # North's own fields only
        st = recv(b, "state", pred=lambda m: holders(m)["North"] is not None)
        assert holders(st) == {"North": 1, "East": None, "Fountain": None, "South": None}
        claim(b, "North")                                         # taken: refused
        st = recv(b, "state")
        assert st["spot"] is None and holders(st)["North"] == 1
        claim(b, "Nowhere")                                       # not a spot: refused
        assert recv(b, "state")["spot"] is None
        claim(b, "East")
        st = recv(b, "state")
        assert (st["spot"], st["space_held"], "gain" in st) == ("East", True, False)
        # what acts on a spot carries it; c has none, so its spot-wise messages are dropped
        for ws in (c, a, b):
            ws.send(json.dumps(pose(5)))
            ws.send(json.dumps({"type": "zero"}))
            ws.send(json.dumps({"type": "control", "action": "gain", "value": 1.5}))
            ws.send(json.dumps({"type": "control", "action": "space", "value": True}))
            ws.send(json.dumps({"type": "control", "action": "level"}))
            ws.send(json.dumps({"type": "control", "action": "seek", "value": 4.0}))
        got = []
        while len(got) < 13:
            m = json.loads(v.recv(timeout=3))
            if m["type"] != "hub":
                got.append((m["type"], m.get("action"), m["player"], m["spot"]))
        assert [g for g in got if g[2] == 3] == [("control", "seek", 3, None)]   # shared: anyone
        for n, spot in ((1, "North"), (2, "East")):
            own = [g for g in got if g[2] == n]
            assert [g[:2] for g in own] == [("pose", None), ("zero", None), ("control", "gain"),
                                            ("control", "space"), ("control", "level"),
                                            ("control", "seek")]
            assert {g[3] for g in own} == {spot}
        # a's SPACE is held on North: moving to Fountain lets go there first
        claim(a, "Fountain")
        assert recv(v, "control") == {"type": "control", "action": "space", "value": False,
                                      **mine(1, "North")}
        assert recv(v, "hub", pred=lambda m: {"player": 1, "spot": "Fountain"} in m["players"])
        # b leaves while holding East: it lets go, and East is free again
        b.close()
        assert recv(v, "control") == {"type": "control", "action": "space", "value": False,
                                      **mine(2, "East")}
        recv(c, "state", pred=lambda m: holders(m) == {"North": None, "East": None,
                                                       "Fountain": 1, "South": None})
        # the view no longer lists Fountain: a's claim goes
        v.send(json.dumps({**SHARED, "spots": SPOTS[:2]}))
        st = recv(a, "state", pred=lambda m: len(m["spots"]) == 2)
        assert st["spot"] is None and [s["name"] for s in st["spots"]] == ["North", "East"]
        assert recv(v, "hub", pred=lambda m: {"player": 1, "spot": None} in m["players"]
                    )["players"] == [{"player": 1, "spot": None}, {"player": 3, "spot": None}]
        # a state without spots (a view starting up) keeps them and the claims
        v.send(json.dumps({**SHARED, "time": 2.0}))
        st = recv(c, "state", pred=lambda m: m["time"] == 2.0)
        assert [s["name"] for s in st["spots"]] == ["North", "East"]
        # spots: [] is one clip again: everything goes through, spot null
        v.send(json.dumps({**SHARED, "spots": []}))
        st = recv(c, "state", pred=lambda m: m["spots"] == [])
        assert st["gain"] == 1.0 and st["space_held"] is False and st["spot"] is None
        c.send(json.dumps({"type": "zero"}))
        assert recv(v, "zero") == {"type": "zero", **mine(3)}


def test_a_page_that_comes_back_gets_its_number_and_spot(hub):
    with view(hub) as v:
        v.send(json.dumps({**SHARED, "spots": SPOTS}))
        with phone(hub) as p1, phone(hub, pid="page-aaaaaaaa") as p2:
            claim(p2, "South")
            st = recv(p2, "state", pred=lambda m: m["spot"] == "South")
            assert st["player"] == 2
            p1.close()
            p2.close()
        recv(v, "hub", pred=lambda m: m["players"] == [])         # both gone
        with phone(hub) as other:                                 # the lowest free number
            assert recv(other, "hub")["player"] == 1
            with phone(hub, pid="page-aaaaaaaa") as back:         # its number and spot again
                assert recv(back, "hub")["player"] == 2
                st = recv(back, "state")
                assert (st["player"], st["spot"]) == (2, "South")
                assert holders(recv(other, "state", pred=lambda m: holders(m)["South"]))[
                    "South"] == 2
                recv(v, "hub", pred=lambda m: m["players"] == [{"player": 1, "spot": None},
                                                               {"player": 2, "spot": "South"}])


# a site of 16 cameras in 4 areas (as Trafalgar), the view opened on 6354
AREAS = {"North": ["6354", "6355", "6356", "6357"], "North-east": ["6376", "6377", "6378"],
         "South": ["6363", "6364", "6366", "6367", "6369"], "West": ["6358", "6359", "6360", "6361"]}
SITE = {"spots": [{"name": n, "clip": f"trafalgar{n}", "group": g, "in_scene": n == "6354",
                   "gain": 1.0, "scale": 1.0, "space_held": False, "space_tilted": False}
                  for g, names in AREAS.items() for n in names],
        "groups": [{"name": g, "spots": names} for g, names in AREAS.items()],
        "start_spot": "6354", "site": "trafalgar"}


def test_the_first_phone_gets_the_start_camera_the_others_pick(hub):
    with phone(hub) as a:                                         # before the view is open
        assert recv(a, "hub")["player"] == 1
        with view(hub) as v:
            v.send(json.dumps({**SHARED, **SITE}))
            st = recv(a, "state")
            assert (st["player"], st["spot"], st["start_spot"]) == (1, "6354", "6354")
            assert st["groups"] == SITE["groups"] and len(st["spots"]) == 16
            assert st["spots"][0] == {"name": "6354", "clip": "trafalgar6354", "group": "North",
                                      "player": 1}               # in_scene stays with the view
            assert st["gain"] == 1.0 and st["space_tilted"] is False
            recv(v, "hub", pred=lambda m: m["players"] == [{"player": 1, "spot": "6354"}])
            with phone(hub) as b:                                 # the start is taken: b picks
                st = recv(b, "state")
                assert (st["player"], st["spot"]) == (2, None)
                claim(b, "6366")
                assert recv(b, "state")["spot"] == "6366"
                a.close()
                recv(v, "hub", pred=lambda m: m["players"] == [{"player": 2, "spot": "6366"}])
                st = recv(b, "state", pred=lambda m: holders(m)["6354"] is None)
                assert st["spot"] == "6366"                       # b keeps its pick
                with phone(hub) as c:                             # the next to join: the start
                    st = recv(c, "state")
                    assert (st["player"], st["spot"]) == (1, "6354")
                    claim(c, None)                                # let go by choice: stays so
                    assert recv(c, "state")["spot"] is None
                    # a view starting up (no spots yet) keeps the cameras and the claims
                    v.send(json.dumps({**SHARED, "time": 3.0}))
                    st = recv(b, "state", pred=lambda m: m["time"] == 3.0)
                    assert (st["spot"], len(st["spots"]), st["start_spot"]) == ("6366", 16, "6354")
                    assert recv(c, "state", pred=lambda m: m["time"] == 3.0)["spot"] is None
                    # another site (6 cameras or fewer: no areas): every camera goes, the
                    # first phone by joining (b) gets the new start, c picks
                    v.send(json.dumps({**SHARED, "spots": [{"name": "A1"}, {"name": "A2"}],
                                       "start_spot": "A1", "site": "kew"}))
                    st = recv(b, "state", pred=lambda m: len(m["spots"]) == 2)
                    assert (st["spot"], st["groups"], st["start_spot"]) == ("A1", [], "A1")
                    assert recv(c, "state", pred=lambda m: len(m["spots"]) == 2)["spot"] is None
                    recv(v, "hub", pred=lambda m: m["players"] == [{"player": 1, "spot": None},
                                                                   {"player": 2, "spot": "A1"}])


RICHMOND = {"spots": [{"name": n, "clip": f"richmond{n}", "group": g}
                      for g, names in (("Bridge", ["6460", "6461", "6462", "6463"]),
                                       ("Green", ["6470", "6471", "6472", "6473"])) for n in names],
            "groups": [{"name": "Bridge", "spots": ["6460", "6461", "6462", "6463"]},
                       {"name": "Green", "spots": ["6470", "6471", "6472", "6473"]}],
            "start_spot": "6462", "site": "richmond"}


def test_a_new_site_lets_every_camera_go_and_the_first_phone_gets_its_start(hub):
    on = lambda *pairs: (lambda m: m["players"] == [{"player": n, "spot": s} for n, s in pairs])
    with view(hub) as v:
        v.send(json.dumps({**SHARED, **SITE}))
        with phone(hub) as a, phone(hub) as b, phone(hub, pid="page-cccccccc") as c:
            claim(b, "6366")
            claim(c, "6376")
            recv(v, "hub", pred=on((1, "6354"), (2, "6366"), (3, "6376")))
            c.send(json.dumps({"type": "control", "action": "space", "value": True}))
            assert recv(v, "control")["value"] is True
            # the laptop switches to Richmond: c's SPACE lets go on 6376, every camera goes,
            # a (the first to join) gets Richmond's start, b and c pick
            v.send(json.dumps({**SHARED, **RICHMOND}))
            assert recv(v, "control") == {"type": "control", "action": "space", "value": False,
                                          **mine(3, "6376")}
            recv(v, "hub", pred=on((1, "6462"), (2, None), (3, None)))
            assert recv(a, "state", pred=lambda m: m["start_spot"] == "6462")["spot"] == "6462"
            st = recv(b, "state", pred=lambda m: m["start_spot"] == "6462")
            assert (st["spot"], st["site"]) == (None, "richmond")
            assert [g["name"] for g in st["groups"]] == ["Bridge", "Green"]
            assert holders(st) == {**{s["name"]: None for s in RICHMOND["spots"]}, "6462": 1}
            claim(b, "6470")
            claim(c, "6471")
            recv(v, "hub", pred=on((1, "6462"), (2, "6470"), (3, "6471")))
            # within one site nothing is let go: the same list, one camera more, a new start
            # camera, a state without "site" (or an empty one)
            v.send(json.dumps({**SHARED, **RICHMOND, "time": 7.0}))
            assert recv(b, "state", pred=lambda m: m["time"] == 7.0)["spot"] == "6470"
            more = {**RICHMOND, "spots": [*RICHMOND["spots"], {"name": "6480", "group": "Green"}]}
            v.send(json.dumps({**SHARED, **more, "time": 8.0}))
            st = recv(c, "state", pred=lambda m: m["time"] == 8.0)
            assert (st["spot"], len(st["spots"])) == ("6471", 9)
            v.send(json.dumps({**SHARED, **more, "start_spot": "6470", "time": 9.0}))
            st = recv(b, "state", pred=lambda m: m["time"] == 9.0)
            assert (st["spot"], st["start_spot"]) == ("6470", "6470")
            no_name = {k: x for k, x in more.items() if k != "site"}
            v.send(json.dumps({**SHARED, **no_name, "site": "", "time": 10.0}))
            st = recv(c, "state", pred=lambda m: m["time"] == 10.0)
            assert (st["spot"], st["site"]) == ("6471", "richmond")
            c.close()                                             # leaves on 6471
            recv(v, "hub", pred=on((1, "6462"), (2, "6470")))
            # another site with the very same camera names: its name decides, every camera goes
            v.send(json.dumps({**SHARED, **more, "site": "richmond-2", "time": 11.0}))
            recv(v, "hub", pred=on((1, "6462"), (2, None)))
            assert recv(b, "state", pred=lambda m: m["time"] == 11.0)["spot"] is None
            with phone(hub, pid="page-cccccccc") as back:          # its number, not its camera
                st = recv(back, "state")
                assert (st["player"], st["spot"], holders(st)["6471"]) == (3, None, None)


def test_the_laptop_releases_a_camera_its_phone_stays_and_picks(hub):
    with view(hub) as v:
        v.send(json.dumps({**SHARED, **SITE}))
        with phone(hub) as a, phone(hub) as b:
            claim(b, "6366")
            recv(v, "hub", pred=lambda m: m["players"] == [{"player": 1, "spot": "6354"},
                                                           {"player": 2, "spot": "6366"}])
            b.send(json.dumps({"type": "control", "action": "space", "value": True}))
            recv(v, "control")
            for junk in ({"type": "release", "spot": "6355"}, {"type": "release", "spot": 7},
                         {"type": "release"}):                    # no one holds it: nothing
                v.send(json.dumps(junk))
            with pytest.raises(TimeoutError):
                recv(v, "hub", timeout=0.3)
            v.send(json.dumps({"type": "release", "spot": "6366"}))
            assert recv(v, "control") == {"type": "control", "action": "space", "value": False,
                                          **mine(2, "6366")}
            assert recv(v, "hub")["players"] == [{"player": 1, "spot": "6354"},
                                                 {"player": 2, "spot": None}]
            st = recv(b, "state", pred=lambda m: m["spot"] is None)
            assert (st["player"], holders(st)["6366"], holders(st)["6354"]) == (2, None, 1)
            assert recv(a, "state", pred=lambda m: holders(m)["6366"] is None)["spot"] == "6354"
            claim(b, "6366")                                      # it may pick it again
            assert recv(b, "state")["spot"] == "6366"


def test_a_camera_the_laptop_locked_is_refused_and_the_phones_see_it(hub):
    locked = {**SITE, "spots": [{**s, "locked": True} if s["name"] == "6366" else s for s in SITE["spots"]]}
    with view(hub) as v:
        v.send(json.dumps({**SHARED, **locked}))
        with phone(hub) as a:
            st = recv(a, "state", pred=lambda m: m["spot"] == "6354")
            by = {s["name"]: s for s in st["spots"]}
            assert by["6366"]["locked"] is True and "locked" not in by["6354"]
            recv(v, "hub", pred=lambda m: m["players"] == [{"player": 1, "spot": "6354"}])
            claim(a, "6366")                                      # locked: refused
            with pytest.raises(TimeoutError):
                recv(v, "hub", timeout=0.3)
            v.send(json.dumps({**SHARED, **SITE}))                # unlocked: it may take it
            claim(a, "6366")
            assert recv(v, "hub")["players"] == [{"player": 1, "spot": "6366"}]


def test_a_phone_on_a_camera_the_laptop_locks_lets_it_go_without_the_release(hub):
    """31 Oct: the padlock sends the hub a release, which is lost when the link is down then. The
    state marking the camera locked lets its phone go as well (it stays and picks again)."""
    locked = {**SITE, "spots": [{**s, "locked": True} if s["name"] == "6366" else s for s in SITE["spots"]]}
    with view(hub) as v:
        v.send(json.dumps({**SHARED, **SITE}))
        with phone(hub) as a, phone(hub) as b:
            claim(b, "6366")
            recv(v, "hub", pred=lambda m: m["players"] == [{"player": 1, "spot": "6354"},
                                                           {"player": 2, "spot": "6366"}])
            v.send(json.dumps({**SHARED, **locked}))              # no release: the state alone
            assert recv(v, "hub")["players"] == [{"player": 1, "spot": "6354"},
                                                 {"player": 2, "spot": None}]
            lk = lambda m: {s["name"]: s for s in m["spots"]}["6366"].get("locked") is True
            st = recv(b, "state", pred=lk)                        # (states from before its claim skipped)
            assert (st["player"], st["spot"], holders(st)["6366"]) == (2, None, None)
            v.send(json.dumps({**SHARED, **locked}))              # again: nothing more to let go
            with pytest.raises(TimeoutError):
                recv(v, "hub", timeout=0.3)
            claim(b, "6366")                                      # still refused
            with pytest.raises(TimeoutError):
                recv(v, "hub", timeout=0.3)


def test_a_zero_carries_the_pose_it_is_taken_at(hub):
    """31 Oct: a phone whose hold changes at its Zero sends the pose in the new hold inside the zero,
    so Live takes both in one step (a pose sent before the zero could be drawn for a frame, turned
    ~90 deg). The pose is checked as a pose; a bad one leaves a plain zero."""
    with view(hub) as v:
        with phone(hub) as p:
            recv(v, "hub", pred=lambda m: m["phones"] == 1)
            p.send(json.dumps({"type": "zero", "pose": pose(9, (0, 0, 1.02, 0), extra=1)}))
            assert recv(v, "zero") == {"type": "zero", "pose": pose(9, (0, 0, 1, 0)), **mine()}
            for bad in (pose(q=(0, 0, 0, 0)), pose(src="camera"), {"type": "zero"}, "x", [1]):
                p.send(json.dumps({"type": "zero", "pose": bad}))
                assert recv(v, "zero") == {"type": "zero", **mine()}


def test_the_first_site_name_lets_nothing_go(hub):
    with view(hub) as v:
        v.send(json.dumps({**SHARED, **{k: x for k, x in SITE.items() if k != "site"}}))
        with phone(hub) as a, phone(hub) as b:
            claim(b, "6366")
            recv(v, "hub", pred=lambda m: m["players"] == [{"player": 1, "spot": "6354"},
                                                           {"player": 2, "spot": "6366"}])
            v.send(json.dumps({**SHARED, **SITE, "time": 4.0}))   # the name, first seen
            st = recv(a, "state", pred=lambda m: m["time"] == 4.0)
            assert (st["site"], st["spot"]) == ("trafalgar", "6354")
            assert recv(b, "state", pred=lambda m: m["time"] == 4.0)["spot"] == "6366"


def test_areas_and_the_start_are_kept_only_with_their_cameras():
    st = live_hub.clean_state({
        "type": "state", "spots": [{"name": "a"}, {"name": "b", "in_scene": "yes", "group": "N"}],
        "groups": [{"name": "N", "spots": ["a", "nope", "a", 7]}, {"name": "X", "spots": ["nope"]},
                   {"name": "N", "spots": ["b"]}, {"spots": ["a"]}, "bad",
                   {"name": "Y", "spots": "a"}],
        "start_spot": "nope"})
    assert st == {"type": "state", "spots": [{"name": "a"}, {"name": "b", "group": "N"}],
                  "groups": [{"name": "N", "spots": ["a"]}]}
    assert live_hub.clean_state({"type": "state", "groups": [{"name": "N", "spots": ["a"]}],
                                 "start_spot": "a"}) == {"type": "state"}
    for bad in ("", None, 7, ["trafalgar"]):                      # no name: the last one stays
        assert live_hub.clean_state({"type": "state", "site": bad}) == {"type": "state"}
    assert live_hub.clean_state({"type": "state", "site": "trafalgar"}) == \
        {"type": "state", "site": "trafalgar"}                   # a name needs no spots
    many = [{"name": f"c{i}"} for i in range(100)]
    assert len(live_hub.clean_state({"type": "state", "spots": many})["spots"]) == \
        live_hub.MAX_SPOTS


# the phone page itself, run in node with a stub DOM and a fake socket: one socket per page
# (a retry suspended in the background and the page coming back used to open two at once),
# the SPACE clutch: press / let go on lift, cancel, sliding off, hiding, the link dropping,
# ("spots") the plain camera list, "player N · camera", and four phones at most, and
# ("groups") a large site: the start camera shown at once, an area then a camera, back, and
# the laptop switching the site: the picker opens by itself when this phone has no camera
PAGE_HARNESS = r"""
const [phoneUrl, mode] = [process.argv[2], process.argv[3]];
const sockets = [], sent = [], out = {};
class FakeWS {
  constructor(url) { this.url = url; this.readyState = 0; this.bufferedAmount = 0; sockets.push(this); }
  send(s) { if (this.readyState !== 1) throw new Error("send on a closed socket"); sent.push(JSON.parse(s)); }
  close() { this.readyState = 3; }
  open() { this.readyState = 1; this.onopen && this.onopen({}); }
  drop(code = 1006) { this.readyState = 3; this.onclose && this.onclose({code}); }
  message(m) { this.onmessage && this.onmessage({data: JSON.stringify(m)}); }
}
class ClassList {
  constructor() { this.s = new Set(); }
  add(c) { this.s.add(c); } remove(c) { this.s.delete(c); } contains(c) { return this.s.has(c); }
  toggle(c, on) { on = on === undefined ? !this.s.has(c) : !!on; on ? this.s.add(c) : this.s.delete(c); return on; }
}
class El {
  constructor(id) { Object.assign(this, {id, on: {}, classList: new ClassList(), hidden: false,
    textContent: "", value: "0", options: [], children: [], disabled: false, offsetWidth: 0,
    max: 1}); }
  addEventListener(t, f) { (this.on[t] ||= []).push(f); }
  fire(t) { for (const f of this.on[t] || []) f({type: t, pointerId: 1, preventDefault() {}}); }
  setAttribute() {} replaceChildren(...c) { this.options = this.children = c; }
  append(...c) { this.children.push(...c); }
  hasPointerCapture() { return true; } releasePointerCapture() {}
}
const els = {}, docOn = {}, winOn = {}, store = new Map();
globalThis.document = {visibilityState: "visible", getElementById: id => els[id] ||= new El(id),
  createElement: () => new El(""), addEventListener: (t, f) => (docOn[t] ||= []).push(f),
  fonts: {add() {}}};
globalThis.window = globalThis;
window.isSecureContext = true;
window.addEventListener = (t, f) => (winOn[t] ||= []).push(f);
globalThis.location = {search: "?k=testcode1234567", host: "127.0.0.1:1"};
globalThis.localStorage = {getItem: k => store.get(k) ?? null, setItem: (k, v) => store.set(k, v)};
globalThis.WebSocket = FakeWS;
globalThis.FontFace = class { load() { return Promise.resolve(); } };
globalThis.requestAnimationFrame = f => setTimeout(f, 0);
let hubDown = false;
globalThis.fetch = async () => { if (hubDown) throw new Error("unreachable"); return {status: 200}; };
const wait = ms => new Promise(r => setTimeout(r, ms));
const live = () => sockets.filter(s => s.readyState <= 1).length;
const visible = v => { document.visibilityState = v ? "visible" : "hidden";
                       for (const f of docOn.visibilitychange || []) f(); };
const cut = () => sent.splice(0).filter(m => m.type !== "ping").map(
  m => m.type === "control" ? `${m.action}:${m.value}` : m.type);

document.getElementById("picker").hidden = true;   // as in index.html
await import(phoneUrl);
if (mode === "spots") await spots();
else if (mode === "groups") await groups();
else await clutch();
console.log(JSON.stringify(out));
process.exit(0);

async function clutch() {
sockets[0].open();
// the hub goes away while the phone is in the background; it comes back to the page
hubDown = true;
visible(false);
sockets[0].drop();
await wait(20);                    // the code check fails: one retry is waiting
visible(true);                     // coming back connects at once
await wait(900);                   // the waiting retry would fire by now
out.race = {made: sockets.length, live: live(), url: sockets.at(-1).url};
if (mode === "all") {
  const sp = els.space, last = () => sockets.at(-1);
  cut();
  last().open();
  out.open = cut();
  const hold = async end => { sp.fire("pointerdown"); out.held = sp.classList.contains("held");
                              await wait(300); await end(); return cut(); };
  out.lift = await hold(() => sp.fire("pointerup"));
  out.slide = await hold(() => sp.fire("pointerleave"));
  out.cancel = await hold(() => sp.fire("pointercancel"));
  out.hidden = await hold(() => visible(false));
  visible(true);
  out.socketsAfterHidden = sockets.length;
  hubDown = false;
  out.dropped = await hold(() => last().drop());
  out.heldAfterDrop = sp.classList.contains("held");
  await wait(500);                 // reconnects by itself
  last().open();
  out.reconnect = cut();
  sp.fire("pointerdown"); await wait(60); sp.fire("pointerup"); await wait(80);
  sp.fire("pointerdown"); sp.fire("pointerup");
  out.doubleTap = cut();
  await wait(700);
  last().message({type: "state", space_tilted: true, space_held: false});
  out.turned = [sp.classList.contains("turned"), sp.classList.contains("held"), els.spacest.textContent];
  last().message({type: "state", space_tilted: false, space_held: true});
  out.elsewhere = [sp.classList.contains("turned"), els.spacest.textContent];
  last().message({type: "state", space_tilted: false, space_held: false});
  out.level = [sp.classList.contains("turned"), els.spacest.textContent];
  last().drop(4002);               // the same page id connected elsewhere: no fight back
  await wait(800);
  out.replaced = {made: sockets.length, conn: els.conn.textContent};
  visible(true);                   // this page in use again: it takes over
  out.takeOver = sockets.length;
}
}

async function spots() {
  const sp = els.space, last = () => sockets.at(-1), off = () => sp.classList.contains("off");
  const opts = () => els.spot.options.map(o => [o.value, o.textContent, !!o.disabled]);
  const north = {name: "North", clip: "trafalgar6354", player: 1};
  const east = p => ({name: "East", clip: "trafalgar6376", player: p});
  last().open();
  cut();
  last().message({type: "hub", phones: 2, views: 1, player: 2});
  out.who0 = [els.who.hidden, els.who.textContent];
  last().message({type: "state", player: 2, spot: null, spots: [north, east(null)], playing: false});
  out.pick = {spotbox: els.spotbox.hidden, disabled: els.spot.disabled, options: opts(),
              who: els.who.textContent, zero: els.zero.disabled, gain: els.gain.disabled,
              off: off(), st: els.spacest.textContent};
  sp.fire("pointerdown"); await wait(50); sp.fire("pointerup");
  out.offPress = cut();            // no spot of its own: SPACE does nothing
  els.spot.value = "East"; els.spot.fire("change");
  out.claim = cut();
  last().message({type: "state", player: 2, spot: "East", spots: [north, east(2)],
                  space_tilted: true, space_held: false, gain: 2});
  out.own = {who: els.who.textContent, value: els.spot.value, options: opts(), off: off(),
             zero: els.zero.disabled, st: els.spacest.textContent, gain: els.gainval.textContent};
  sp.fire("pointerdown");
  out.press = cut();
  last().message({type: "state", player: 2, spot: null, spots: [north, east(null)]});
  out.lost = {sent: cut(), held: sp.classList.contains("held"), off: off()};   // spot gone: let go
  sp.fire("pointerup");
  out.liftAfter = cut();
  last().message({type: "state", player: 2, spot: "East", spots: [north, east(2)]});
  cut();
  await wait(400);                 // (not a double-tap)
  sp.fire("pointerdown");          // held, and a second finger picks no spot: let go first
  els.spot.value = ""; els.spot.fire("change");
  out.switch = cut();
  sp.fire("pointerup");
  last().message({type: "state", player: 2, spot: null, spots: []});    // one clip
  out.single = {spotbox: els.spotbox.hidden, disabled: els.spot.disabled, options: opts(),
                who: els.who.textContent, off: off(), zero: els.zero.disabled};
  const made = sockets.length;
  last().drop(4003);               // four phones already: says so, tries again now and then
  out.full = {conn: els.conn.textContent, note: els.note.textContent, whoHidden: els.who.hidden};
  await wait(1000);
  out.fullSoon = sockets.length - made;
  await wait(2400);
  out.fullLater = sockets.length - made;
  last().open();
  last().message({type: "hub", phones: 4, views: 1, player: 4});
  out.joined = {noteHidden: els.note.hidden, who: els.who.textContent};
}

async function groups() {
  const sp = els.space, last = () => sockets.at(-1);
  const items = () => els.plist.children.map(b => [b.children[0].textContent,
    b.children[1].textContent, !!b.disabled, b.classList.contains("mine")]);
  const tap = i => els.plist.children[i].fire("click");
  const field = () => ({name: els.camname.textContent, area: els.camarea.textContent,
                        go: els.camgo.textContent, who: els.who.textContent});
  const AREAS = {North: ["6354", "6355"], South: ["6363", "6366", "6367"]};
  const site = (hold, own, extra = {}) => ({type: "state", player: 1, spot: own, start_spot: "6354", site: "trafalgar",
    groups: Object.entries(AREAS).map(([name, spots]) => ({name, spots})),
    spots: Object.entries(AREAS).flatMap(([g, ns]) => ns.map(n => ({name: n, clip: "trafalgar" + n,
                                                                    group: g, player: hold[n] ?? null}))),
    ...extra});
  last().open();
  cut();
  last().message({type: "hub", phones: 2, views: 1, player: 1});
  last().message(site({"6354": 1, "6366": 3}, "6354"));    // the first phone: the start camera
  out.auto = {cam: els.cam.hidden, spotbox: els.spotbox.hidden, picker: els.picker.hidden, off: sp.classList.contains("off"), ...field()};
  sp.fire("pointerdown");
  els.cam.fire("click");           // change: SPACE is let go under the sheet
  out.openLetsGo = cut();
  out.areas = {picker: els.picker.hidden, title: els.ptitle.textContent, back: els.pback.hidden,
               none: els.pnone.hidden, items: items()};
  const first = els.plist.children[0];
  last().message(site({"6354": 1, "6366": 3}, "6354", {time: 5}));     // a time tick
  out.kept = els.plist.children[0] === first;             // nothing rebuilt under a finger
  tap(1);                                                  // South
  out.south = {title: els.ptitle.textContent, back: els.pback.hidden, items: items()};
  els.pback.fire("click");
  out.back = {title: els.ptitle.textContent, back: els.pback.hidden, n: items().length};
  tap(1); tap(2);                                          // South, 6367
  out.claim = {sent: cut(), picker: els.picker.hidden};
  last().message(site({"6366": 3, "6367": 1}, "6367"));
  out.claimed = field();
  els.cam.fire("click"); tap(1); tap(0);                   // South, 6363: free when shown...
  cut();
  last().message(site({"6366": 3, "6367": 1}, "6367"));   // (a state still on its way)
  out.inFlight = els.note.textContent;
  last().message(site({"6363": 4, "6366": 3, "6367": 1}, "6367"));   // ...taken first
  out.refused = {hidden: els.note.hidden, note: els.note.textContent, name: els.camname.textContent};
  els.cam.fire("click");
  els.pnone.fire("click");                                 // no camera
  out.none = {sent: cut(), picker: els.picker.hidden};
  last().message(site({"6363": 4, "6366": 3}, null));
  out.noCamera = {...field(), off: sp.classList.contains("off"), noteHidden: els.note.hidden,
                  picker: els.picker.hidden};                // let go by choice: stays shut
  els.cam.fire("click");
  last().message(site({}, null, {groups: []}));            // a small site: the plain list
  out.small = {picker: els.picker.hidden, cam: els.cam.hidden, spotbox: els.spotbox.hidden};
  els.spot.value = "6355"; els.spot.fire("change");         // asked for, and then...
  cut();
  // ...the laptop switches the site: no answer to that claim, and no camera here: the picker
  const RICH = {Bridge: ["6460", "6461", "6462"], Green: ["6470", "6471"]};
  const rich = (start, own, hold, extra = {}) => ({type: "state", player: 1, spot: own, start_spot: start, site: "richmond",
    groups: Object.entries(RICH).map(([name, spots]) => ({name, spots})),
    spots: Object.entries(RICH).flatMap(([g, ns]) => ns.map(n => ({name: n, group: g, player: hold[n] ?? null}))),
    ...extra});
  last().message(rich("6462", null, {"6462": 2}));
  out.newSite = {picker: els.picker.hidden, title: els.ptitle.textContent, items: items(),
                 noteHidden: els.note.hidden, ...field()};
  els.pclose.fire("click");
  last().message(rich("6462", null, {"6462": 2}, {time: 6}));          // the same site: stays shut
  last().message(rich("6470", null, {"6462": 2}, {site: undefined}));  // a new start, no name: same
  out.sameSite = els.picker.hidden;
  last().message(rich("6470", "6470", {"6470": 1}, {site: "kew"}));   // a new site: this phone's start
  out.startHere = {picker: els.picker.hidden, ...field()};
  last().message(rich("6470", null, {"6470": 2}, {site: "richmond"}));  // and back: no camera
  out.again = {picker: els.picker.hidden, ...field()};
  els.pclose.fire("click");
  last().message(rich("6470", "6471", {"6470": 2, "6471": 1}));
  last().message(rich("6470", null, {"6470": 2}));         // the laptop took 6471 out
  out.released = {picker: els.picker.hidden, ...field()};
  const lk = rich("6470", null, {"6460": 2});                  // 6461 locked on the laptop
  lk.spots.find(s => s.name === "6461").locked = true;
  last().message(lk);
  out.lockedArea = {items: items(), area: els.camarea.textContent};
}
"""


def _run_page(tmp_path, phone_js, mode="all"):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    harness = tmp_path / "page_harness.mjs"
    harness.write_text(PAGE_HARNESS, encoding="utf-8")
    r = subprocess.run([node, str(harness), phone_js.as_uri(), mode], capture_output=True,
                       encoding="utf-8", timeout=60, check=False)     # node writes UTF-8
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_page_keeps_one_socket_and_never_sticks_the_clutch(tmp_path):
    out = _run_page(tmp_path, live_hub.PHONE_WEB / "phone.js")
    assert out["race"]["live"] == 1 and out["race"]["made"] == 2         # one, not two
    assert re.search(r"/ws/phone\?k=testcode1234567&id=[0-9a-f]{24}$", out["race"]["url"])
    assert out["open"] == ["space:false"]                                 # a new link: let go
    assert out["held"] is True
    for how in ("lift", "slide", "cancel", "hidden"):
        assert out[how] == ["space:true", "space:false"], how
    assert out["socketsAfterHidden"] == 2                                 # coming back: no 2nd
    assert out["dropped"] == ["space:true"] and out["heldAfterDrop"] is False
    assert out["reconnect"] == ["space:false"]
    assert out["doubleTap"] == ["space:true", "space:false", "level:null"]
    assert out["turned"] == [True, False, "turned · double-tap: level"]
    assert out["elsewhere"] == [True, "held elsewhere"]
    assert out["level"] == [False, "hold: turn the space"]
    assert out["replaced"] == {"made": 3, "conn": "in use on another page"}
    assert out["takeOver"] == 4


def test_page_spot_picker_player_and_four_phones(tmp_path):
    out = _run_page(tmp_path, live_hub.PHONE_WEB / "phone.js", "spots")
    assert out["who0"] == [False, "player 2"]
    assert out["pick"] == {"spotbox": False, "disabled": False,
                           "options": [["", "–", False], ["North", "North · player 1", True],
                                       ["East", "East", False]],
                           "who": "player 2 · pick a camera", "zero": True, "gain": True,
                           "off": True, "st": "pick a camera"}
    assert out["offPress"] == []
    assert out["claim"] == ["spot:East"]
    assert out["own"] == {"who": "player 2 · East", "value": "East",
                          "options": [["", "–", False], ["North", "North · player 1", True],
                                      ["East", "East", False]],
                          "off": False, "zero": False, "st": "turned · double-tap: level",
                          "gain": "2.00×"}
    assert out["press"] == ["space:true"]
    assert out["lost"] == {"sent": ["space:false"], "held": False, "off": True}
    assert out["liftAfter"] == []
    assert out["switch"] == ["space:true", "space:false", "spot:null"]
    # one clip: no clip list on the phone (the laptop chooses), the camera list shown empty
    assert out["single"] == {"spotbox": False, "disabled": True, "options": [["", "–", False]],
                             "who": "player 2", "off": False, "zero": False}
    assert out["full"] == {"conn": "four phones connected", "whoHidden": True,
                           "note": "Four phones are connected: this one joins when one of them "
                                   "leaves."}
    assert (out["fullSoon"], out["fullLater"]) == (0, 1)
    assert out["joined"] == {"noteHidden": True, "who": "player 4"}


def test_page_large_site_area_then_camera(tmp_path):
    out = _run_page(tmp_path, live_hub.PHONE_WEB / "phone.js", "groups")
    assert out["auto"] == {"cam": False, "spotbox": True, "picker": True,
                           "off": False, "name": "6354", "area": "North", "go": "change",
                           "who": "player 1 · 6354"}         # the start camera, shown at once
    assert out["openLetsGo"] == ["space:true", "space:false"]
    assert out["areas"] == {"picker": False, "title": "Pick an area", "back": True, "none": False,
                            "items": [["North", "yours · 2 cameras", False, True],
                                      ["South", "3 cameras · 1 taken", False, False]]}
    assert out["kept"] is True
    assert out["south"] == {"title": "South", "back": False,
                            "items": [["6363", "free", False, False],
                                      ["6366", "player 3", True, False],     # taken: greyed
                                      ["6367", "free", False, False]]}
    assert out["back"] == {"title": "Pick an area", "back": True, "n": 2}
    assert out["claim"] == {"sent": ["spot:6367"], "picker": True}
    assert out["claimed"] == {"name": "6367", "area": "South", "go": "change",
                              "who": "player 1 · 6367"}
    assert out["inFlight"] == ""
    assert out["refused"] == {"hidden": False, "note": "6363 is taken: pick another camera.",
                              "name": "6367"}
    assert out["none"] == {"sent": ["spot:null"], "picker": True}
    assert out["noCamera"] == {"name": "pick a camera", "area": "3 free", "go": "",
                               "who": "player 1 · pick a camera", "off": True,
                               "noteHidden": True, "picker": True}
    assert out["small"] == {"picker": True, "cam": True, "spotbox": False}
    assert out["newSite"] == {"picker": False, "title": "Pick an area",       # opened by itself
                              "items": [["Bridge", "3 cameras · 1 taken", False, False],
                                        ["Green", "2 cameras", False, False]],
                              "noteHidden": True, "name": "pick a camera", "area": "4 free",
                              "go": "", "who": "player 1 · pick a camera"}
    assert out["sameSite"] is True
    assert out["startHere"] == {"picker": True, "name": "6470", "area": "Green", "go": "change",
                                "who": "player 1 · 6470"}
    assert out["released"] == {"picker": False, "name": "pick a camera", "area": "4 free",
                               "go": "", "who": "player 1 · pick a camera"}
    assert out["again"] == {"picker": False, "name": "pick a camera", "area": "4 free", "go": "",
                            "who": "player 1 · pick a camera"}
    assert out["lockedArea"] == {"items": [["Bridge", "3 cameras · 1 taken · 1 locked", False, False],
                                           ["Green", "2 cameras", False, False]],
                                 "area": "3 free"}                 # locked: told apart from taken


# --------------------------------------------------------------------------- QR, addresses, CLI
def test_qr_code_reads_back_as_the_url():
    cv2 = pytest.importorskip("cv2")
    url = "https://172.20.10.3:8766/?k=AbCdEfGhIjKlMnOp"
    svg = qr_svg(url)
    n = int(re.search(r'viewBox="0 0 (\d+) \1"', svg).group(1))
    img = np.full((n, n), 255, np.uint8)
    for x, y, run in re.findall(r"M(\d+) (\d+)h(\d+)v1h-\3z", svg):
        img[int(y), int(x):int(x) + int(run)] = 0
    img = cv2.resize(img, (n * 8, n * 8), interpolation=cv2.INTER_NEAREST)
    text, _, _ = cv2.QRCodeDetector().detectAndDecode(img)
    assert text == url
    rows = qr_text(url).splitlines()
    assert len(rows) >= 12 and len(set(map(len, rows))) == 1


def test_lan_addresses_are_ones_a_phone_could_reach():
    import ipaddress

    addrs = live_hub.lan_addresses()
    for a in addrs:
        ip = ipaddress.ip_address(a["ip"])
        assert not ip.is_loopback and not ip.is_link_local
    assert sum(a["likely"] for a in addrs) <= 1 or all(a["likely"] for a in addrs[:2])


def test_cli_live_hub_defaults():
    from groundtrack.cli import build_parser, cmd_live_hub

    a = build_parser().parse_args(["live-hub"])
    assert (a.func, a.port, a.view_port, a.host, a.new_code) == (cmd_live_hub, 8766, 8767,
                                                                 "0.0.0.0", False)


# --------------------------------------------------------------------------- orientation maths
def _cam(alpha, beta, gamma, angle):
    """Independent reference: W3C R = Rz(a) Rx(b) Ry(g) as matrices, the screen's angle, then
    the back camera in OpenCV axes (x right, y down, z out of the back)."""
    def rz(d):
        c, s = math.cos(math.radians(d)), math.sin(math.radians(d))
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def rx(d):
        c, s = math.cos(math.radians(d)), math.sin(math.radians(d))
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def ry(d):
        c, s = math.cos(math.radians(d)), math.sin(math.radians(d))
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    return rz(alpha) @ rx(beta) @ ry(gamma) @ rz(-angle) @ np.diag([1.0, -1.0, -1.0])


S30, C30, S20, C20 = 0.5, math.sqrt(3) / 2, math.sin(math.radians(20)), math.cos(math.radians(20))
# (alpha, beta, gamma, screen angle) -> camera x, y, z in the world (x east, y north, z up),
# worked by hand from the phone's axes (x right, y top, z out of the glass, portrait)
HAND = [
    ("flat on a table, face up", (0, 0, 0, 0), [(1, 0, 0), (0, -1, 0), (0, 0, -1)]),
    ("upright, landscape, top to the left, looking north", (90, 0, -90, 90),
     [(1, 0, 0), (0, 0, -1), (0, 1, 0)]),
    ("upright, landscape, top to the right, looking north", (270, 0, 90, 270),
     [(1, 0, 0), (0, 0, -1), (0, 1, 0)]),
    ("the same with window.orientation = -90", (-90, 0, 90, -90),
     [(1, 0, 0), (0, 0, -1), (0, 1, 0)]),
    ("turned 90 deg right: looking east", (0, 0, -90, 90), [(0, -1, 0), (0, 0, -1), (1, 0, 0)]),
    ("turned 90 deg right, other hold", (180, 0, 90, 270), [(0, -1, 0), (0, 0, -1), (1, 0, 0)]),
    ("tilted 30 deg up", (270, -180, 60, 90), [(1, 0, 0), (0, S30, -C30), (0, C30, S30)]),
    ("tilted 30 deg down", (90, 0, -60, 90), [(1, 0, 0), (0, -S30, -C30), (0, C30, -S30)]),
    ("rolled 20 deg clockwise", (90, 20, -90, 90), [(C20, 0, -S20), (-S20, 0, -C20), (0, 1, 0)]),
]


def test_phone_orientation_to_camera_quaternion_in_node():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    rng = np.random.default_rng(4)
    rand = [(float(a), float(b), float(g), int(s)) for a, b, g, s in
            zip(rng.uniform(0, 360, 40), rng.uniform(-180, 180, 40), rng.uniform(-90, 90, 40),
                rng.choice([0, 90, 180, 270], 40))]
    cases = [c for _, c, _ in HAND] + rand
    src = (live_hub.PHONE_WEB / "phone.js").as_uri()
    code = (f'import {{cameraQuaternion, axes, tiltRoll}} from "{src}";\n'
            f"const cases = {json.dumps(cases)};\n"
            "const qs = cases.map(c => cameraQuaternion(...c));\n"
            "console.log(JSON.stringify({qs, axes: qs.map(axes), tr: qs.map(tiltRoll)}));")
    r = subprocess.run([node, "--input-type=module", "-e", code], capture_output=True,
                       encoding="utf-8", timeout=60, check=False)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    for (name, _, want), got in zip(HAND, out["axes"]):
        assert np.allclose(np.array(got), np.array(want, float), atol=1e-9), name
    for c, q, ax in zip(cases, out["qs"], out["axes"]):
        assert abs(np.linalg.norm(q) - 1) < 1e-12
        assert np.allclose(np.array(ax).T, _cam(*c), atol=1e-9), c       # columns = axes
    tilt_up, roll_cw = out["tr"][6][0], out["tr"][8][1]
    assert tilt_up == pytest.approx(30) and roll_cw == pytest.approx(20)
    assert out["tr"][1] == pytest.approx([0, 0], abs=1e-9)


def test_the_hold_comes_from_gravity_not_the_screen_lock():
    """30 Oct: an iPhone filming sideways with its rotation lock on reported screen angle 0, so its
    camera came out rolled 90 deg. holdAngle takes the angle from gravity: the hand-worked holds
    give their own angle, the locked iPhone gives 90 (level again), a phone near flat keeps the
    last one, a hold kept to landscape picks the side that puts the picture's up up, and an open
    hold changes only past the switch (no flapping at 45 deg). 31 Oct: a phone looking down on the
    plan keeps its hold as it wobbles across flat or tips sideways (Live's camera spun 90 / 180
    deg); turned over or upright it still changes."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    rng = np.random.default_rng(7)
    rand = [(float(a), float(b), float(g)) for a, b, g in
            zip(rng.uniform(0, 360, 60), rng.uniform(-180, 180, 60), rng.uniform(-90, 90, 60))]
    hand = [c for name, c, _ in HAND if not name.startswith("flat")]
    src = (live_hub.PHONE_WEB / "phone.js").as_uri()
    code = (f'import {{holdAngle, cameraQuaternion, tiltRoll}} from "{src}";\n'
            f"const hand = {json.dumps(hand)}, rand = {json.dumps(rand)};\n"
            "console.log(JSON.stringify({\n"
            "  hand: hand.map(c => [holdAngle(c[0], c[1], c[2]), holdAngle(c[0], c[1], c[2], true, 0)]),\n"
            "  locked: [0, 90].map(a => tiltRoll(cameraQuaternion(90, 0, -60, a))),\n"
            "  held: holdAngle(90, 0, -60, true, 0),\n"
            "  flat: [holdAngle(0, 0, 0, null, 270), holdAngle(0, 5, 10, true, 90)],\n"
            "  rand: rand.map(c => [holdAngle(c[0], c[1], c[2], true, 90), holdAngle(c[0], c[1], c[2], false, 0)]),\n"
            "  diag: [[45, 90], [50, 90], [65, 90], [45, 0]].map(([r, p]) => holdAngle(90, r, -90, null, p)),\n"
            "  plan: [[-30, 0, 0], [30, 0, 180], [10, 30, 0], [10, -30, 0], [-60, 0, 0], [90, 0, 180]]\n"
            "        .map(([b, g, p]) => holdAngle(0, b, g, null, p)),\n"
            "  fit: holdAngle(0, 10, -30, true, 0),\n"
            "}));")
    r = subprocess.run([node, "--input-type=module", "-e", code], capture_output=True,
                       encoding="utf-8", timeout=60, check=False)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    for c, (free, land) in zip(hand, out["hand"]):
        assert free == land == c[3] % 360, c                   # the angle the hand-worked hold has
    (tilt0, roll0), (tilt90, roll90) = out["locked"]           # tilted 30 deg down, lock on: 0
    assert abs(roll0) == pytest.approx(90, abs=1e-6) and roll90 == pytest.approx(0, abs=1e-6)
    assert tilt0 == pytest.approx(-30) and tilt90 == pytest.approx(-30) and out["held"] == 90
    assert out["flat"] == [270, 90]                             # near flat: the last one kept
    for c, (land, port) in zip(rand, out["rand"]):
        down = {a: _cam(*c, a)[2, 1] for a in (0, 90, 180, 270)}   # the picture's down, its z
        if np.hypot(*_cam(*c, 0)[2, :2]) < 0.7:
            continue                                             # short of upright: the last one
        assert down[land] == min(down[90], down[270]) and down[port] == min(down[0], down[180]), c
    d = out["diag"]                                             # landscape rolled 45, 50, 65 deg
    assert d[:2] == [90, 90] and d[2] in (0, 180)               # it changes past the switch only
    assert d[3] in (0, 180)                                     # at 45 deg each hold keeps itself
    assert out["plan"] == [0, 180, 0, 0, 180, 0]                # 30 deg past flat, 60 down, tipped
    assert out["fit"] == 90                                     # sideways: kept; over, upright: changed


def test_a_steering_phone_rolled_about_its_screen_turns_smoothly():
    """31 Oct, the user: rolling the phone about its screen's axis (a steering wheel) snapped Live's
    camera where the hold turned to landscape (~56 deg): the hold changed by 90 deg mid-roll. The
    steering page keeps the hold it settled at the start (and at Zero): rolled 0 -> 120 deg in 5 deg
    steps, every step turns the camera 5 deg, while a hold decided at each reading jumps."""
    from scipy.spatial.transform import Rotation
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    start = Rotation.from_euler("ZXY", [30, 60, 0], degrees=True)        # upright, looking 30 deg down
    rolls = [(start * Rotation.from_euler("z", r, degrees=True)).as_euler("ZXY", degrees=True).tolist()
             for r in range(0, 125, 5)]
    src = (live_hub.PHONE_WEB / "phone.js").as_uri()
    code = (f'import {{holdAngle, cameraQuaternion}} from "{src}";'
            f"const rolls = {json.dumps(rolls)};"
            "const h0 = holdAngle(...rolls[0], null, 0);"
            "let prev = h0; const open = rolls.map(e => (prev = holdAngle(...e, null, prev)));"
            "console.log(JSON.stringify({h0, open, kept: rolls.map(e => cameraQuaternion(...e, h0)),"
            "  each: rolls.map((e, i) => cameraQuaternion(...e, open[i]))}));")
    r = subprocess.run([node, "--input-type=module", "-e", code], capture_output=True,
                       encoding="utf-8", timeout=60, check=False)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    step = lambda qs: [math.degrees(2 * math.acos(min(1, abs(np.dot(a, b))))) for a, b in zip(qs, qs[1:])]
    assert out["h0"] == 0 and len(set(out["open"])) > 1                  # decided each reading: it turns
    assert step(out["kept"]) == pytest.approx([5] * 24, abs=1e-6)         # kept: 5 deg a step, no jump
    assert max(step(out["each"])) > 80                                    # (the snap the user saw)
    page = (live_hub.PHONE_WEB / "phone.js").read_text(encoding="utf-8")
    assert "if (held == null) held = holdAngle(e.alpha, e.beta, e.gamma, null, screenAngle());" in page
    assert "held = holdAngle(e.alpha, e.beta, e.gamma, null, held" not in page   # not at each reading
    zero = page.split('ui.zero.addEventListener("click"')[1].split("});")[0]
    # Live's zero in the new hold: the pose rides in the zero (one message, one step in Live); the
    # hold changes only when it was sent
    assert 'if (send({type: "zero", pose: poseMsg(q, "ok")})) { held = h; lastQ = q; sent++; draw(); }' in zero
    assert "sendPose(" not in zero
    live = (live_hub.PHONE_WEB.parent / "desktop_web" / "live" / "live.js").read_text(encoding="utf-8")
    on_zero = live.split("onZero: (m) => {")[1].split("onHub:")[0]
    assert on_zero.index("if (m.pose) onPose(") < on_zero.index("zero(s.inp.phoneSrc, s)")


def test_phone_gain_runs_0_1_to_3_and_its_ticks_sit_on_their_values():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    src = (live_hub.PHONE_WEB / "phone.js").as_uri()
    code = (f'import {{gainOf, sliderOf}} from "{src}";\n'
            "console.log(JSON.stringify({ends: [gainOf(0), gainOf(1000)],"
            " at: [0.1, 0.25, 0.5, 1, 2, 3].map(sliderOf),"
            " back: [0, 677, 1000].map(s => sliderOf(gainOf(s)))}));")
    r = subprocess.run([node, "--input-type=module", "-e", code], capture_output=True,
                       encoding="utf-8", timeout=60, check=False)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["ends"] == pytest.approx([0.1, 3])
    assert out["back"] == [0, 677, 1000]
    html = (live_hub.PHONE_WEB / "index.html").read_text(encoding="utf-8")
    css = (live_hub.PHONE_WEB / "phone.css").read_text(encoding="utf-8")
    labels = re.findall(r'<span class="(g\d+)">([^<]*)</span>', html)
    assert [t for _, t in labels] == ["0.1", "0.25", "0.5", "1", "2", "3×"]
    for (cls, _), slider in zip(labels, out["at"]):           # each label on its own value
        p = float(re.search(rf"\.{cls} \{{ --p: ([\d.]+); \}}", css).group(1))
        assert p == pytest.approx(slider / 1000, abs=0.001), cls


def test_phone_sends_no_record_and_its_record_button_is_realtime_only():
    """Every pass records by itself (the laptop): the phone never sends a record control. Its
    Record button is the realtime recorder's (record.js, plans/live_controller/10), hidden
    unless the hub says the laptop is in the realtime mode."""
    html = (live_hub.PHONE_WEB / "index.html").read_text(encoding="utf-8")
    js = (live_hub.PHONE_WEB / "phone.js").read_text(encoding="utf-8")
    assert '"record"' not in js
    assert re.search(r'<button id="rec"[^>]*\bhidden\b[^>]*>Record</button>', html)
    assert "showRealtime(m.realtime === true)" in js


# --------------------------------------------------------------------------- realtime recorder
def frame_bytes(head=None, jpeg=b"\xff\xd8\xff\xe0jpegdata", raw_head=None):
    h = raw_head if raw_head is not None else json.dumps(
        {"seq": 1, "t": 3.5, "w": 640, "h": 360, "q": [1.5, 0, 0, 0], **(head or {})}).encode()
    return len(h).to_bytes(4, "little") + h + jpeg


def test_recorder_frames_are_checked():
    head, jpeg = live_hub.clean_frame(frame_bytes({"extra": "x", "seq": 7}))
    assert head == {"seq": 7, "t": 3.5, "w": 640, "h": 360, "q": [1.0, 0, 0, 0], "skip": None}
    assert jpeg == b"\xff\xd8\xff\xe0jpegdata"
    assert live_hub.clean_frame(frame_bytes({"q": None}))[0]["q"] is None
    assert live_hub.clean_frame(frame_bytes({"skip": 12}))[0]["skip"] == 12   # the phone's count
    for odd in (-1, 2 ** 31, 1.5, True, "3", [1]):                # odd counts: left out, not refused
        assert live_hub.clean_frame(frame_bytes({"skip": odd}))[0]["skip"] is None
    bad = [frame_bytes(jpeg=b"GIF89a"), frame_bytes({"w": 8}), frame_bytes({"h": 5000}),
           frame_bytes({"w": 640.0}), frame_bytes({"w": True}), frame_bytes({"seq": -1}),
           frame_bytes({"t": "now"}), frame_bytes({"t": None}), frame_bytes({"q": [1, 0, 0]}),
           frame_bytes({"q": [0, 0, 0, 0.1]}), frame_bytes({"q": [5, 0, 0, 0]}),
           frame_bytes(raw_head=b"{not json"), frame_bytes(raw_head=b"[1, 2]"),
           frame_bytes(raw_head=b"x" * 2000), b"\x00\x00", "text",
           (10 ** 6).to_bytes(4, "little") + b"{}" + b"\xff\xd8",
           frame_bytes(jpeg=b"\xff\xd8" + b"0" * live_hub.MAX_FRAME)]
    for b in bad:
        assert live_hub.clean_frame(b) is None, b[:40]


class FakeSink:
    """What the hub hands frames to (live_track.LiveTracker's calls)."""

    long_side = 1280                                              # the laptop's frame size (720p)

    def __init__(self):
        self.calls, self.frames = [], []

    def begin(self, w, h, resume=False, **kw):
        if w < 64:
            raise ValueError("frame size out of range")
        self.calls.append(("begin", w, h, resume, *([kw] if kw else [])))
        return {"id": "20261030-120000", "resumed": resume}

    def frame(self, head, jpeg):
        self.frames.append((head, jpeg))
        return True

    def end(self):
        self.calls.append(("end",))

    def lost(self):
        self.calls.append(("lost",))

    def status(self):
        return {"mode": "ready", "recording": bool(self.calls) and self.calls[-1][0] == "begin"}


def recorder(h, pid="rec-page-aaaa"):
    return connect(f"wss://127.0.0.1:{h.bound['lan']}/ws/record?k={h.token}&id={pid}",
                   ssl=tls(h), open_timeout=5, max_size=2 ** 20)


def closed_with(ws):
    with pytest.raises(ConnectionClosed) as e:
        while True:
            ws.recv(timeout=3)
    return e.value.rcvd.code


def test_recorder_only_in_the_realtime_mode_and_phones_are_told(hub):
    with pytest.raises(InvalidStatus) as e:
        recorder(hub)
    assert e.value.response.status_code == 409
    with pytest.raises(InvalidStatus) as e:                       # the code first
        connect(f"wss://127.0.0.1:{hub.bound['lan']}/ws/record?k=nope", ssl=tls(hub))
    assert e.value.response.status_code == 403
    assert hub.info()["realtime"] is False
    with phone(hub) as p:
        assert "realtime" not in recv(p, "hub")
        hub.set_recorder(FakeSink())
        assert recv(p, "hub", pred=lambda m: m.get("realtime") is True)["player"] == 1
        assert hub.info()["realtime"] is True and hub.info()["recorder"] is False
        with recorder(hub) as r:
            assert until(lambda: hub.info()["recorder"])
            hub.set_recorder(None)                                # the laptop leaves realtime
            assert closed_with(r) == live_hub.CLOSE_REALTIME_OFF
        assert recv(p, "hub", pred=lambda m: "realtime" not in m)
        assert until(lambda: not hub.info()["recorder"])
    with pytest.raises(InvalidStatus) as e:
        recorder(hub)
    assert e.value.response.status_code == 409


def test_recorder_start_frames_stop_and_drop_out(hub):
    sink = FakeSink()
    hub.set_recorder(sink)
    with view(hub) as v, recorder(hub) as r:
        assert recv(r, "rec") == {"type": "rec", "hello": True, "long_side": 1280}   # the size first
        r.send(frame_bytes())                                     # before start: not taken
        r.send(json.dumps({"type": "rec", "action": "start", "w": 10, "h": 10}))
        assert recv(r, "rec", pred=lambda m: "error" in m)["error"] == "frame size out of range"
        r.send(json.dumps({"type": "rec", "action": "start", "w": 640.5, "h": 360}))
        assert "error" in recv(r, "rec", pred=lambda m: "error" in m)
        r.send(json.dumps({"type": "rec", "action": "start", "w": 640, "h": 360}))
        assert recv(r, "rec", pred=lambda m: "started" in m) == {
            "type": "rec", "started": "20261030-120000", "resumed": False}
        for i in range(5):
            r.send(frame_bytes({"seq": i, "q": [0, 0, 0, 1]}))
        r.send(frame_bytes(jpeg=b"nope"))
        r.send(b"\x00")
        r.send("{not json")
        r.send(json.dumps({"type": "ping", "id": 3}))
        assert recv(r, "pong") == {"type": "pong", "id": 3}
        assert [h["seq"] for h, _ in sink.frames] == [0, 1, 2, 3, 4]
        assert sink.frames[0][0]["q"] == [0, 0, 0, 1.0]
        st = recv(r, "rec", pred=lambda m: "mode" in m)           # the tracker's status
        assert st == {"type": "rec", "mode": "ready", "recording": True}
        r.send(json.dumps({"type": "rec", "action": "stop"}))
        assert recv(r, "rec", pred=lambda m: "stopped" in m)["stopped"] is True
        assert sink.calls[-1] == ("end",)
        r.send(frame_bytes())                                     # after stop: not taken
        r.send(json.dumps({"type": "rec", "action": "start", "w": 640, "h": 360, "resume": True}))
        recv(r, "rec", pred=lambda m: "started" in m)
        r.send(json.dumps({"type": "rec", "action": "stop"}))       # recalibrate: the phone's height
        recv(r, "rec", pred=lambda m: "stopped" in m)
        r.send(json.dumps({"type": "rec", "action": "start", "w": 640, "h": 360,
                           "height_m": "tall", "tilt_offset": 0}))
        assert recv(r, "rec", pred=lambda m: "error" in m)["error"] == "height_m: a number"
        r.send(json.dumps({"type": "rec", "action": "start", "w": 640, "h": 360,
                           "height_m": 1.65, "tilt_offset": -2, "extra": 1}))
        recv(r, "rec", pred=lambda m: "started" in m)
        with pytest.raises(TimeoutError):                         # nothing reaches the view
            recv(v, "pose", timeout=0.3)
    assert until(lambda: ("lost",) in sink.calls)                 # left while recording
    assert sink.calls == [("begin", 640, 360, False), ("end",), ("begin", 640, 360, True),
                          ("end",), ("begin", 640, 360, False, {"height_m": 1.65, "tilt_offset": -2.0}),
                          ("lost",)]
    assert len(sink.frames) == 5
    assert logged(hub, r"recorder \(127\.0\.0\.1\) left after \d+ s: 5 frames")
    text = "\n".join(hub.logs)
    assert "recording started: 20261030-120000" in text and "jpegdata" not in text
    assert hub.token not in text and "rec-page-aaaa" not in text


def test_one_recorder_at_a_time_the_same_page_takes_over(hub):
    hub.set_recorder(FakeSink())
    with recorder(hub, "rec-page-aaaa") as first:
        assert until(lambda: hub.info()["recorder"])
        with recorder(hub, "rec-page-bbbb") as other:
            assert closed_with(other) == live_hub.CLOSE_RECORDER
        with recorder(hub, "rec-page-aaaa") as again:             # the same page, back
            assert closed_with(first) == live_hub.CLOSE_REPLACED
            again.send(json.dumps({"type": "ping", "id": 1}))
            assert recv(again, "pong")["id"] == 1
    assert logged(hub, "refused a recorder from 127.0.0.1: one is connected")


def test_a_phone_message_over_4096_bytes_closes_it(hub):
    with phone(hub) as p:
        recv(p, "hub")
        p.send(json.dumps({"type": "zero", "pad": "x" * 5000}))
        assert closed_with(p) == 1009


def test_recorder_frame_header_and_size_in_node():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    src = (live_hub.PHONE_WEB / "record.js").as_uri()
    code = (f'import {{frameHeader, frameSize, cleanHeight, horizon}} from "{src}";\n'
            "const h = frameHeader({seq: 3, t: 1.25, w: 1920, h: 1080, q: null, skip: 4});\n"
            "console.log(JSON.stringify({head: [...h], sizes: [[1920, 1080], [3840, 2160],"
            " [1080, 1920], [1280, 720], [1001, 563], [4032, 3024]]"
            ".map(([w, h]) => frameSize(w, h)),"
            " small: [[1920, 1080, 1280], [1280, 720, 1280], [1080, 1920, 1280], [1920, 1080, 4000],"
            " [1920, 1080, 10], [1920, 1080, 'x']].map(([w, h, l]) => frameSize(w, h, l)),"
            " heights: ['1.73', null, '', 'x', 9, 0.1, 1.5 + 0.05 + 0.05].map(cleanHeight),"
            " level: [horizon(-10, 2), horizon(0, 0), horizon(18.684, -3)]}));")
    r = subprocess.run([node, "--input-type=module", "-e", code], capture_output=True,
                       encoding="utf-8", timeout=60, check=False)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    head, jpeg = live_hub.clean_frame(bytes(out["head"]) + b"\xff\xd8\xff\xd9")
    assert head == {"seq": 3, "t": 1.25, "w": 1920, "h": 1080, "q": None, "skip": 4}
    assert jpeg == b"\xff\xd8\xff\xd9"
    assert out["sizes"] == [[1920, 1080], [1920, 1080], [1080, 1920], [1280, 720], [1002, 564],
                            [1920, 1440]]
    # the laptop's choice (720p); never above 1080p, never below 320; nonsense: 1080p
    assert out["small"] == [[1280, 720], [1280, 720], [720, 1280], [1920, 1080], [320, 180],
                            [1920, 1080]]
    # the remembered height (step 2): in range, to the cm; nothing or nonsense: 1.5 m
    assert out["heights"] == [1.73, 1.5, 1.5, 1.5, 5, 0.3, 1.6]
    # the level: 10 deg down puts the horizon above the middle (tan 10 / tan of the half height);
    # tilted up by the half height, it is at the bottom edge; rolled clockwise, it turns back
    a, b, c = out["level"]
    assert a["turn"] == -2 and a["up"] == pytest.approx(math.tan(math.radians(10)) * 1597.7 / 540)
    assert b["up"] == 0 and c["turn"] == 3 and c["up"] == pytest.approx(-1, abs=1e-3)
