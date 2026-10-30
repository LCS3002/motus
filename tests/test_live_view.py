"""Live view (plans/live_controller/03, `motus app` -> Live): the app's Live routes (the page, a
run's scene package through the usual checks, the phone hub endpoints) and the view's modules in
node: the package parser, performed trails, the mouse head (pan / tilt as the 02 tests want
them), the hub link (pose.js) with a fake socket and against the real LiveHub with a fake phone."""

import json
import queue
import shutil
import socket
import ssl
import subprocess
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from groundtrack import desktop_links as links
from groundtrack.desktop import WEB, DesktopApp, make_desktop_handler
from groundtrack.live_export import TRACK_DTYPE
from groundtrack.ui import _load, _run_root

LIVE = WEB / "live"
THREE = WEB / "vendor" / "three"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")


# --------------------------------------------------------------------------- a project with a package
def tracks_bin():
    """Two tracks: 1 on frames 0-5 (frame 3 predicted), 2 on frames 2, 3 and 7 (a 4-frame gap)."""
    rows = [(1, f, 0 if f != 3 else 1) for f in range(6)] + [(2, f, 0) for f in (2, 3, 7)]
    rows.sort(key=lambda r: (r[1], r[0]))
    a = np.zeros(len(rows), TRACK_DTYPE)
    for i, (tid, f, src) in enumerate(rows):
        a[i] = (tid, f, f / 30.0, 0.5 * f, 10.0 + tid, 0.0, 1.5, 0, src)
    return a


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """The demo site with one run that has a scene package (and one that has none)."""
    from groundtrack.demo import make_demo_site

    root = tmp_path_factory.mktemp("proj")
    make_demo_site(root)
    runs = _run_root(_load(root / "sites" / "demo.yaml"))
    live = runs / "r1" / "live"
    live.mkdir(parents=True)
    meta = {"version": 1, "clip": "demo", "run": "r1", "frames": 8, "fps": 30.0, "duration_s": 0.267,
            "tracks": {"n_tracks": 2}, "video": {"file": "video.mp4", "blurred": True},
            "origin": {"E": 1000.0, "N": 2000.0, "datum_z": 5.0}, "camera": {"C": [0.5, -0.5, 1.6]}}
    (live / "scene.json").write_text(json.dumps(meta), encoding="utf-8")
    (live / "terrain.f32").write_bytes(np.zeros(16, "<f4").tobytes())
    (live / "aerial.jpg").write_bytes(b"\xff\xd8\xff\xe0 not really a jpeg")
    (live / "tracks.bin").write_bytes(tracks_bin().tobytes())
    (live / "tracks_index.json").write_text("{}", encoding="utf-8")
    (live / "video.mp4").write_bytes(bytes(range(256)) * 40)
    (live / "check.png").write_bytes(b"\x89PNG not for the view")
    (runs / "r2").mkdir(parents=True)
    return root, runs


@pytest.fixture()
def app(project, tmp_path, monkeypatch):
    from groundtrack import live_hub

    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))          # never the real hub home
    monkeypatch.setattr(live_hub, "live_dir", lambda: tmp_path / "appdata" / "groundtrack" / "live")
    a = DesktopApp(project[0], links.Settings(tmp_path / "desktop.json"))
    a.hub_options = {"port": 0, "view_port": 0, "host": "127.0.0.1", "home": tmp_path / "hub"}
    yield a
    a.stop_hub()


@pytest.fixture()
def server(app):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), None)
    port = srv.server_address[1]
    srv.RequestHandlerClass = make_desktop_handler(app, port)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()
    srv.server_close()


def http(url, headers=None, data=None, method=None):
    """(status, headers, body); errors are returned, not raised."""
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def q(path):
    return urllib.request.quote(str(path))


# --------------------------------------------------------------------------- routes
def test_page_and_its_modules_are_served_offline(server):
    st, h, body = http(server + "/live")
    assert st == 200 and h["Content-Type"].startswith("text/html")
    page = body.decode()
    imap = json.loads(page.split('<script type="importmap">')[1].split("</script>")[0])
    assert imap["imports"] == {"three": "/static/vendor/three/three.module.min.js",
                               "three/addons/": "/static/vendor/three/addons/"}
    assert "Cross-Origin-Opener-Policy" not in h and "Cross-Origin-Embedder-Policy" not in h
    for path in ("live/live.js", "live/pose.js", "live/projection.js", "live/projection_worker.js",
                 "live/spots.js", "live/panels.js", "live/passes.js", "live/record.js", "live/replay.js",
                 "vendor/three/three.module.min.js", "vendor/three/three.core.min.js",
                 "vendor/three/addons/controls/OrbitControls.js",
                 "vendor/three/addons/controls/TransformControls.js",
                 "vendor/three/addons/lines/Line2.js", "vendor/three/addons/lines/LineMaterial.js",
                 "vendor/three/addons/lines/LineSegments2.js", "live/live.css", "tokens.css"):
        st, h, _ = http(f"{server}/static/{path}")
        assert st == 200, path
        if path.endswith(".js"):
            assert h["Content-Type"].startswith("application/javascript"), path
    # everything is local: no CDN, no remote fonts (only the hub's socket on this machine)
    for name in ("live.js", "pose.js", "scene.js", "tracks.js", "lines.js", "timeline.js", "section.js",
                 "spots.js", "panels.js", "passes.js", "live.html", "live.css"):
        text = (LIVE / name).read_text(encoding="utf-8").replace("http://www.w3.org/2000/svg", "")
        for bad in ("https://", "http://", "//cdn", "fonts.googleapis"):
            assert bad not in text, (name, bad)


def test_three_is_pinned_and_licensed():
    assert "three.js 0.185.1" in (THREE / "VERSION").read_text(encoding="utf-8")
    assert (THREE / "LICENSE").read_text(encoding="utf-8").startswith("The MIT License")
    core = (THREE / "three.core.min.js").read_text(encoding="utf-8")
    assert "REVISION" in core and '="185"' in core


def test_package_files_through_the_checks(server, project, tmp_path):
    root, runs = project
    run = runs / "r1"
    st, _, body = http(f"{server}/live/pkg/scene.json?run={q(run)}")
    assert st == 200 and json.loads(body)["clip"] == "demo"
    st, _, body = http(f"{server}/live/pkg/tracks.bin?run={q(run)}")
    assert st == 200 and len(body) == 30 * len(tracks_bin())
    # the video in pieces (seeking): a range request
    st, h, body = http(f"{server}/live/pkg/video.mp4?run={q(run)}", {"Range": "bytes=100-199"})
    assert st == 206 and h["Content-Range"] == "bytes 100-199/10240" and body == bytes(range(100, 200))
    # only the package's own files, by name
    for name in ("check.png", "../scene.json", "..%2F..%2Fsites%2Fdemo.yaml", "live.log", ""):
        assert http(f"{server}/live/pkg/{name}?run={q(run)}")[0] == 404, name
    assert http(f"{server}/live/pkg/scene.json?run={q(runs / 'r2')}")[0] == 404     # no package
    assert http(f"{server}/live/pkg/scene.json")[0] == 400                           # no run
    elsewhere = tmp_path / "elsewhere"
    shutil.copytree(run, elsewhere / "r1")
    assert http(f"{server}/live/pkg/scene.json?run={q(elsewhere / 'r1')}")[0] == 403  # outside


def test_info_and_clips(server, project):
    _, runs = project
    st, _, body = http(f"{server}/api/live/info?run={q(runs / 'r1')}")
    info = json.loads(body)
    assert st == 200 and info["live"] and info["clip"] == "demo" and info["name"] == "r1"
    assert info["frames"] == 8 and info["n_tracks"] == 2 and info["blurred"] is True
    assert json.loads(http(f"{server}/api/live/info?run={q(runs / 'r2')}")[2]) == {"live": False}
    clips = json.loads(http(f"{server}/api/live/clips")[2])["clips"]
    assert clips == [{"name": f"{runs.name}/r1", "run": str(runs / "r1")}]


def test_site_route_lists_the_cameras_of_the_runs_site(server, project):
    """/api/live/site: the cameras a Live view on this run can show (live_site; the demo site has
    one clip, so one camera, named by its number), through the usual checks."""
    _, runs = project
    st, _, body = http(f"{server}/api/live/site?run={q(runs / 'r1')}")
    site = json.loads(body)
    assert st == 200 and site["site"] == "demo" and site["groups"] == []
    assert [c["name"] for c in site["cameras"]] == ["demo"] and site["start_spot"] == "demo"
    cam = site["cameras"][0]
    assert cam["run"] == str((runs / "r1").resolve()) and cam["run_name"] == "r1" and cam["clip"] == "demo"
    assert cam["E"] == pytest.approx(1000.5) and cam["N"] == pytest.approx(1999.5)
    assert http(f"{server}/api/live/site?run={q(runs / 'r2')}")[0] == 404                 # no package
    assert http(f"{server}/api/live/site?run={q(Path.home())}")[0] in (403, 404)           # outside


def test_sites_route_lists_the_sites_with_packages(server, project):
    """/api/live/sites: the site picker (live_site.site_list): the demo site, its one camera the start."""
    _, runs = project
    st, _, body = http(f"{server}/api/live/sites")
    sites = json.loads(body)["sites"]
    assert st == 200 and [s["site"] for s in sites] == ["demo"]
    assert sites[0]["cameras"] == 1 and sites[0]["start_run"] == str((runs / "r1").resolve())


def test_pass_routes_save_list_load_delete(server, project, tmp_path):
    """08 (b) passes: saved under <run>/live/passes/<id>/ only (pass.json + a .bin per camera,
    16 bytes a point), listed for the site, read back, saved again under the same id, deleted;
    only with the app's header, only inside the project's runs/, sizes checked."""
    import base64

    _, runs = project
    run = runs / "r1"
    n = len(tracks_bin())
    blob = bytes(range(256)) * (16 * n // 256) + bytes(16 * n % 256)
    cam = {"name": "demo", "id": "demo", "clip": "demo", "run_name": "r1", "run": str(run), "points": n,
           "file": "demo.bin", "origin": {"E": 1000.0, "N": 2000.0, "datum_z": 5.0}, "offset": [0, 0, 0],
           "fps": 30.0, "frames": 8, "performed_to": 7, "tint": "#9b82d9", "players": [1]}
    manifest = {"site": "demo", "start_spot": "demo", "started": "2026-10-28T19:00:00", "ended": "2026-10-28T19:00:41",
                "complete": True, "clock_s": 0.267, "duration_s": 0.267, "origin": cam["origin"], "cameras": [cam]}
    hdr = {"X-Groundtrack": "1", "Content-Type": "application/json"}

    def post(body, headers=hdr):
        st, _, b = http(f"{server}/api/live/pass", headers, json.dumps(body).encode(), "POST")
        return st, json.loads(b) if b[:1] == b"{" else b

    save = {"action": "save", "run": str(run), "manifest": manifest, "data": {"demo.bin": base64.b64encode(blob).decode()}}
    assert post(save, {"Content-Type": "application/json"})[0] == 403          # no app header
    st, r = post(save)
    assert st == 200 and r["cameras"] == ["demo"] and r["complete"] is True, r
    pid = r["id"]
    folder = run / "live" / "passes" / pid
    assert sorted(p.name for p in folder.iterdir()) == ["demo.bin", "pass.json"]
    assert (folder / "demo.bin").read_bytes() == blob
    m = json.loads((folder / "pass.json").read_text(encoding="utf-8"))
    assert m["format"] == "motus-live-pass" and m["version"] == 1 and m["id"] == pid and "layout" in m
    assert m["cameras"][0]["run"] == f"runs/{runs.name}/r1"                    # project-relative
    lst = json.loads(http(f"{server}/api/live/passes?run={q(run)}")[2])["passes"]
    assert [p["id"] for p in lst] == [pid] and lst[0]["bytes"] > 16 * n and lst[0]["run"] == str(run.resolve())
    got = json.loads(http(f"{server}/api/live/pass?run={q(run)}&id={pid}")[2])
    assert got["cameras"][0]["path"] == str(run.resolve())                     # absolute, for the view
    st, _, b = http(f"{server}/live/passdata?run={q(run)}&id={pid}&file=demo.bin")
    assert st == 200 and b == blob
    assert http(f"{server}/live/passdata?run={q(run)}&id={pid}&file=pass.json")[0] == 404   # only its .bin files
    assert http(f"{server}/live/passdata?run={q(run)}&id=..%2F..&file=demo.bin")[0] == 400
    # the same pass finished again: the same id, the new drawing
    st, r2 = post({**save, "id": pid, "data": {"demo.bin": base64.b64encode(blob[::-1]).decode()}})
    assert st == 200 and r2["id"] == pid and (folder / "demo.bin").read_bytes() == blob[::-1]
    # checked: sizes, files, ids, where
    assert post({**save, "data": {"demo.bin": base64.b64encode(blob[:-1]).decode()}})[0] == 400
    assert post({**save, "data": {**save["data"], "x.bin": ""}})[0] == 400
    assert post({**save, "id": "20990101-000000"})[0] == 400        # no such pass
    assert post({**save, "id": "../../x"})[0] == 400
    bad = {**manifest, "cameras": [{**cam, "file": "../evil.bin"}]}
    assert post({**save, "manifest": bad, "data": {"../evil.bin": save["data"]["demo.bin"]}})[0] == 400
    elsewhere = tmp_path / "proj_lib" / "r1"
    shutil.copytree(run, elsewhere)
    assert post({**save, "run": str(elsewhere)})[0] == 403                     # not the project's runs/
    # deleted: its files and folder
    assert post({"action": "delete", "run": str(run), "id": pid}) == (200, {"ok": True})
    assert not folder.exists() and json.loads(http(f"{server}/api/live/passes?run={q(run)}")[2])["passes"] == []
    assert post({"action": "delete", "run": str(run), "id": pid})[0] == 400
    (run / "live" / "passes").rmdir()


def envelope(head, files):
    """live.js postFiles: [u32 header length][JSON header][files]."""
    h = json.dumps({**head, "files": [{"file": k, "bytes": len(v)} for k, v in files.items()]}).encode()
    return len(h).to_bytes(4, "little") + h + b"".join(files.values())


def test_recorded_pass_saved_as_bytes_rendered_and_exported(server, project, app, monkeypatch):
    """05: a pass with its recording saved as bytes (/api/live/pass/save); rendered frame by frame
    into <run>/live/renders/<id>_<view>.mp4 (every frame once, counted by ffprobe, the PNGs deleted);
    exported as a job of the app (Rhino open: asked first)."""
    import cv2

    from groundtrack import live_passes as lp

    _, runs = project
    run = runs / "r1"
    n = len(tracks_bin())
    clock = np.zeros(4, lp.CLOCK_DTYPE)
    clock["t"] = clock["T"] = np.arange(4) / 30
    cam = np.zeros(2, lp.CAM_DTYPE)
    cam["R_wc"] = cam["S"] = np.eye(3)
    cam["k"], cam["flags"] = [0, 2], lp.PERFORMED
    rec = {"started": "2026-10-29T10:00:00", "seconds": 0.1, "clock": {"file": "clock.rec", "rows": 4},
           "events": {"file": "events.jsonl", "n": 1}, "cameras": [{"name": "demo", "file": "demo.rec", "rows": 2, "run": str(run)}]}
    manifest = {"site": "demo", "start_spot": "demo", "complete": True, "clock_s": 0.1, "recording": rec,
                "cameras": [{"name": "demo", "file": "demo.bin", "points": n, "run": str(run), "performed_to": 3}]}
    files = {"demo.bin": bytes(16 * n), "clock.rec": clock.tobytes(), "demo.rec": cam.tobytes(),
             "events.jsonl": b'{"k": 0, "t": 0, "T": 0, "type": "pass_start"}\n'}
    body = envelope({"action": "save", "run": str(run), "manifest": manifest}, files)
    url, hdr = f"{server}/api/live/pass/save", {"X-Groundtrack": "1", "Content-Type": "application/octet-stream"}
    assert http(url, {"Content-Type": "application/octet-stream"}, body, "POST")[0] == 403        # no app header
    assert http(url, hdr, body[:-1], "POST")[0] == 400
    st, _, b = http(url, hdr, body, "POST")
    r = json.loads(b)
    assert st == 200 and r["recorded"] is True, r
    pid = r["id"]
    folder = run / "live" / "passes" / pid
    assert (folder / "demo.rec").read_bytes() == cam.tobytes() and json.loads((folder / "pass.json").read_text())["version"] == 2
    st, _, b = http(f"{server}/live/passdata?run={q(run)}&id={pid}&file=clock.rec")
    assert st == 200 and b == clock.tobytes()

    def post(route, obj):
        st, _, b = http(server + route, {"X-Groundtrack": "1", "Content-Type": "application/json"}, json.dumps(obj).encode(), "POST")
        return st, json.loads(b)

    # render: 5 frames of 32 x 16 at 10 fps
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        st, b = post("/api/live/render", {"action": "begin", "run": str(run), "id": pid, "fps": 10, "width": 32, "height": 16,
                                          "frames": 5, "view": "plan"})
        assert st == 200, b
        tok, frames_dir = b["token"], Path(b["dir"])
        assert frames_dir.parent == run / "live" / "renders" and frames_dir.is_dir()

        def frame(j, w=32, h=16):
            img = np.full((h, w, 3), 40 * j, np.uint8)
            return http(f"{server}/api/live/render/frame?token={tok}&frame={j}", hdr, cv2.imencode(".png", img)[1].tobytes(), "POST")

        assert frame(0, 30)[0] == 400 and frame(5)[0] == 400                        # its size, its range
        assert http(f"{server}/api/live/render/frame?token={tok}&frame=1", hdr, b"not a png", "POST")[0] == 400
        for j in (4, 0, 2, 1):                                                       # in any order
            assert frame(j)[0] == 200
        st, b = post("/api/live/render", {"action": "finish", "token": tok})
        assert st == 400 and "missing" in b["error"]                                 # frame 3
        assert frame(3)[0] == 200
        st, b = post("/api/live/render", {"action": "finish", "token": tok})
        assert st == 200, b
        out = run / "live" / "renders" / f"{pid}_plan.mp4"
        assert b["path"] == str(out) and out.is_file() and b["frames"] == 5 and b["seconds"] == 0.5 and b["fps"] == 10
        assert b["codec"] == "h264" and (b["width"], b["height"]) == (32, 16) and not frames_dir.exists()
        st, b = post("/api/live/render", {"action": "begin", "run": str(run), "id": pid, "fps": 10, "width": 32, "height": 16, "frames": 2})
        d2 = Path(b["dir"])
        assert post("/api/live/render", {"action": "cancel", "token": b["token"]}) == (200, {"ok": True}) and not d2.exists()
        assert post("/api/live/render", {"action": "finish", "token": b["token"]})[0] == 400
        assert post("/api/live/render", {"action": "begin", "run": str(run), "id": pid, "fps": 10, "width": 31, "height": 16,
                                         "frames": 2})[0] == 400                    # yuv420p: even sizes
        shutil.rmtree(run / "live" / "renders")
    assert post("/api/live/render", {"action": "begin", "run": str(run), "id": "20990101-000000", "frames": 2})[0] == 400

    # export: a job of the app; Rhino open: asked first
    started = []

    class FakeJob:
        def __init__(self, kind, args, cwd, label):
            self.id, self.kind, self.args = "j1", kind, args
            started.append((kind, args, label))

        def summary(self):
            return {"id": self.id, "kind": self.kind}

    monkeypatch.setattr(app, "job_class", FakeJob)
    monkeypatch.setattr(links, "running_apps", lambda: {"rhino": True, "houdini": False})
    st, b = post("/api/live/export", {"run": str(run), "id": pid, "rhino": True})
    assert st == 409 and b["confirm"] is True and not started
    st, b = post("/api/live/export", {"run": str(run), "id": pid, "rhino": True, "force": True})
    assert st == 200 and b["id"] == "j1"
    assert started[-1][:2] == ("live-pass-export", ["live-pass-export", str(run.resolve()), pid, "--rhino"])
    assert post("/api/live/export", {"run": str(run), "id": pid, "houdini": True, "no_build": True})[0] == 200
    assert started[-1][1][3:] == ["--houdini", "--no-build"]
    assert post("/api/live/export", {"run": str(run), "id": "../x"})[0] == 400

    # the collage / a camera's layer: each layer's drawing in, prepared in <run>/live/exports/, then a job
    lay = {"camera": "demo", "camera_id": "demo", "package": str(run), "run": str(run), "id": pid, "key": 1,
           "kind": "play", "stacked": False, "up_to": 3, "file": "1_demo.bin"}
    url = f"{server}/api/live/export/layers"
    st, _, b = http(url, hdr, envelope({"site": "demo", "layers": [lay]}, {"1_demo.bin": bytes(16 * n)}), "POST")
    assert st == 409 and json.loads(b)["confirm"] is True and not (run / "live" / "exports").exists()
    st, _, b = http(url, hdr, envelope({"site": "demo", "layers": [lay], "force": True}, {"1_demo.bin": bytes(16 * n)}), "POST")
    r = json.loads(b)
    out = Path(r["dir"])
    assert st == 200 and r["id"] == "j1" and out.parent == run.resolve() / "live" / "exports" and out.name.endswith("-collage")
    assert started[-1][:2] == ("live-layers-export", ["live-layers-export", str(out), "--rhino", "--houdini"])
    meta = json.loads((out / "layers.json").read_text(encoding="utf-8"))
    assert [(ly["camera"], ly["id"], ly["file"]) for ly in meta["layers"]] == [("demo", pid, "layers/001_demo.bin")]
    monkeypatch.setattr(links, "running_apps", lambda: {"rhino": False, "houdini": False})
    st, _, b = http(url, hdr, envelope({"camera": "demo", "layers": [lay]}, {"1_demo.bin": bytes(16 * n)}), "POST")
    assert st == 200 and Path(json.loads(b)["dir"]).name.endswith("-demo")
    assert http(url, hdr, envelope({"layers": [lay]}, {"1_demo.bin": bytes(16 * n - 16)}), "POST")[0] == 400   # not its package's
    assert http(url, hdr, envelope({"layers": [{**lay, "camera": "other"}]}, {"1_demo.bin": bytes(16 * n)}), "POST")[0] == 400
    shutil.rmtree(run / "live" / "exports")
    assert post("/api/live/pass", {"action": "delete", "run": str(run), "id": pid}) == (200, {"ok": True})
    assert not folder.exists()
    (run / "live" / "passes").rmdir()


def test_rules_route_gives_the_runs_own_rules(server, project):
    """/api/live/rules: the run's config (_working/config_used.yaml) as rules.js takes it; a run
    without one gets groundtrack's defaults, said so ("source": "defaults", the HUD shows it)."""
    import yaml

    from groundtrack.config import default_config
    from groundtrack.live_rules import rules_config

    root, runs = project
    run = runs / "r1"
    keys = {"track_buffer_s", "max_gap_s", "stitch", "stitch_max_gap_s", "groups"}
    st, _, body = http(f"{server}/api/live/rules?run={q(run)}")                # no config_used.yaml
    r = json.loads(body)
    assert st == 200 and r == {**rules_config(default_config()), "source": "defaults"}
    assert keys <= set(r) and r["groups"]
    assert http(f"{server}/api/live/rules?run={q(runs / 'r2')}")[0] == 404     # no package
    assert http(f"{server}/api/live/rules")[0] == 400                          # no run
    cfg = _load(root / "sites" / "demo.yaml")
    cfg.data["cleaning"]["stitch_max_gap_s"] = 7.5                             # the run's own, not the defaults
    work = run / "_working"
    work.mkdir()
    try:
        (work / "config_used.yaml").write_text(yaml.safe_dump(cfg.data, sort_keys=False), encoding="utf-8")
        st, _, body = http(f"{server}/api/live/rules?run={q(run)}")
        r = json.loads(body)
        assert st == 200 and r == {**rules_config(cfg), "source": "run"} and r["stitch_max_gap_s"] == 7.5
    finally:
        shutil.rmtree(work)
    # the view says so: "rules · defaults" on the HUD's Drawn row
    assert '"defaults" ? " · rules · defaults"' in (LIVE / "live.js").read_text(encoding="utf-8")


def test_hub_endpoints_start_info_new_code_stop(server, app):
    def post(body, headers=None):
        st, _, data = http(server + "/api/live/hub", {"X-Groundtrack": "1"} if headers is None else headers,
                           json.dumps(body).encode(), "POST")
        return st, json.loads(data)

    assert json.loads(http(server + "/api/live/hub")[2]) == {"running": False}
    assert http(server + "/api/live/hub", {}, b'{"action": "start"}', "POST")[0] == 403   # the app's header only
    st, info = post({"action": "start"})
    assert st == 200 and info["running"] and info["phones"] == 0
    u = info["phone_urls"][0]
    assert u["url"].startswith("https://127.0.0.1:") and "?k=" in u["url"] and u["qr_svg"].startswith("<svg")
    assert info["view_url"] == f"ws://127.0.0.1:{info['view_port']}/ws/view"
    again = post({"action": "start"})[1]                               # one hub per app
    assert again["port"] == info["port"] and again["url"] == info["url"]
    assert json.loads(http(server + "/api/live/hub")[2])["url"] == info["url"]
    renewed = post({"action": "new-code"})[1]
    assert renewed["url"] != info["url"] and renewed["port"] == info["port"]
    assert not post({"action": "stop"})[1]["running"]
    app.stop_hub()                                                     # stopping twice is fine


# --------------------------------------------------------------------------- node: the view's modules
HOOKS = """
const V = %s;
export async function resolve(spec, ctx, next) {
  if (spec === "three") return { url: V + "three.module.min.js", shortCircuit: true };
  if (spec.startsWith("three/addons/")) return { url: V + "addons/" + spec.slice(13), shortCircuit: true };
  return next(spec, ctx);
}
"""

PRELUDE = """
import { register } from "node:module";
import { pathToFileURL } from "node:url";
register(pathToFileURL(process.argv[2] + "/hooks.mjs").href);
// a canvas for the heads' dot texture (no DOM in node)
globalThis.document = { createElement: () => ({ width: 0, height: 0, getContext: () => ({
  beginPath() {}, arc() {}, fill() {}, fillRect() {} }) }) };
const LIVE = %s;
const P = await import(LIVE + "projection.js");
const M = await import(LIVE + "pose.js");
const T = await import(LIVE + "tracks.js");
const out = (k, v) => console.log(JSON.stringify([k, v]));
const D = Math.PI / 180;
const mousePose = (pan, tilt, roll = 0) => ({ type: "pose", v: 1, src: "mouse", seq: 1, t: 0,
  q: M.matToQuat(M.headMatrix(pan * D, tilt * D, roll * D)), p: [0, 0, 0], status: "ok" });
const bearing = (sc, X, i) => Math.atan2(X[3 * i] - sc.C[0], X[3 * i + 1] - sc.C[1]) / D;
const dist = (sc, X, i) => Math.hypot(X[3 * i] - sc.C[0], X[3 * i + 1] - sc.C[1]);
"""


def flat_meta(yaw=30.0, tilt=4.0, h=1.5, f=1500.0, W=1920, H=1080, R_max=1e4, frames=20):
    p, t = np.radians(yaw), np.radians(tilt)
    fwd = np.array([np.sin(p) * np.cos(t), np.cos(p) * np.cos(t), -np.sin(t)])
    right = np.array([np.cos(p), -np.sin(p), 0.0])
    R_cw = np.column_stack([right, np.cross(fwd, right), fwd]).T
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    C = np.array([0.0, 0.0, h])
    return {"version": 1, "image_size": [W, H], "R_max": R_max, "frames": frames, "fps": 30.0,
            "camera": {"P": (K @ np.column_stack([R_cw, -R_cw @ C])).tolist(), "K": K.tolist(),
                       "R_cw": R_cw.tolist(), "C": C.tolist()}, "terrain": None}


def run_node(tmp_path, body, job):
    (tmp_path / "hooks.mjs").write_text(HOOKS % json.dumps(THREE.as_uri() + "/"), encoding="utf-8")
    (tmp_path / "job.json").write_text(json.dumps(job), encoding="utf-8")
    script = PRELUDE % json.dumps(LIVE.as_uri() + "/") + \
        'const job = JSON.parse((await import("node:fs")).readFileSync(process.argv[2] + "/job.json", "utf8"));\n' + body
    (tmp_path / "run.mjs").write_text(script, encoding="utf-8")
    r = subprocess.run([NODE, "run.mjs", str(tmp_path)], cwd=tmp_path, capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, r.stderr
    return {k: v for k, v in (json.loads(line) for line in r.stdout.splitlines() if line.startswith("["))}


@needs_node
def test_package_parser_in_node(tmp_path):
    (tmp_path / "tracks.bin").write_bytes(tracks_bin().tobytes())
    res = run_node(tmp_path, """
const b = (await import("node:fs")).readFileSync(process.argv[2] + "/tracks.bin");
const d = T.parseTracks(Uint8Array.from(b).buffer, { frames: 8 });
out("n", d.n); out("tracks", d.nTracks); out("frameStart", Array.from(d.frameStart));
const p = d.pairs, pairs = [];
for (let s = 0; s < p.n; s++) pairs.push([d.tid[p.A[s]], d.frame[p.A[s]], d.frame[p.B[s]], p.near[s], p.guess[s]]);
out("pairs", pairs); out("pairStart", Array.from(p.start)); out("x5", Array.from(d.X.slice(15, 18)));
""", {})
    a = tracks_bin()
    assert res["n"] == len(a) and res["tracks"] == 2
    assert res["frameStart"] == [int(np.searchsorted(a["frame"], f)) for f in range(9)]
    # consecutive rows per track, in the frame order of their end; near = gap <= 2, guess = not seen
    assert sorted(res["pairs"], key=lambda r: (r[2], r[0])) == res["pairs"]
    assert res["pairs"] == [[1, 0, 1, 1, 0], [1, 1, 2, 1, 0], [1, 2, 3, 1, 1], [2, 2, 3, 1, 0],
                            [1, 3, 4, 1, 1], [1, 4, 5, 1, 0], [2, 3, 7, 0, 0]]
    assert res["pairStart"][7] == 6 and res["pairStart"][8] == res["pairStart"][9] == 7
    assert res["x5"] == pytest.approx([a["x"][5], a["y"][5], a["z"][5]])


@needs_node
def test_mouse_head_turns_the_plan_like_the_phone(tmp_path):
    """The mouse controller's pose through projection.js: pan 30 right swirls the plan 30 the
    other way and keeps distances; tilt up pulls in, tilt down throws out (flat-ground law on the
    view's bearing); the HUD's camera delta reads back what was set."""
    res = run_node(tmp_path, """
const sc = P.prepareScene(job.meta, null), zero = mousePose(0, 0);
const yaw = P.heading(sc.R_wc) / D, n = 600;
for (const pan of [30, -30, 10]) {
  // ground points where the turned camera looks (true positions: flat ground, z = 0)
  const X = new Float64Array(3 * n);
  for (let i = 0; i < n; i++) { const b = (yaw + pan - 15 + 30 * (i % 30) / 29) * D, r = 6 + 4 * (i / 30 | 0);
    X[3 * i] = sc.C[0] + r * Math.sin(b); X[3 * i + 1] = sc.C[1] + r * Math.cos(b); }
  const cam = P.poseToCamera(sc, mousePose(pan, 0), zero), o = P.project(sc, cam, X), sh = [], dd = [];
  for (let i = 0; i < n; i++) if (o.state[i] === P.DRAWN) {
    sh.push(((bearing(sc, o.X, i) - bearing(sc, X, i) + 540) % 360) - 180); dd.push(dist(sc, o.X, i) - dist(sc, X, i)); }
  out("pan" + pan, { n: sh.length, min: Math.min(...sh), max: Math.max(...sh), dd: Math.max(...dd.map(Math.abs)),
                     delta: M.cameraDelta(sc, cam) });
}
const h = sc.C[2], ds = [3, 5, 8, 12, 20, 35, 60, 100, 200];
const Y = new Float64Array(3 * ds.length);
ds.forEach((r, i) => { Y[3 * i] = sc.C[0] + r * Math.sin(yaw * D); Y[3 * i + 1] = sc.C[1] + r * Math.cos(yaw * D); });
for (const t of [3, -4]) {
  const cam = P.poseToCamera(sc, mousePose(0, t), zero), o = P.project(sc, cam, Y), rows = [];
  ds.forEach((r, i) => rows.push([r, o.state[i], o.state[i] === P.DRAWN ? dist(sc, o.X, i) : null]));
  out("tilt" + t, { rows, h, delta: M.cameraDelta(sc, cam), K: Array.from(sc.K), H: sc.H, t0: job.tilt });
}
const cam = P.poseToCamera(sc, mousePose(12, -3, 5), zero);
out("delta", M.cameraDelta(sc, cam));
const R = M.headMatrix(0.3, -0.2, 0.1), R2 = P.quatToMatrix(M.matToQuat(R));
out("roundtrip", Math.max(...R.map((v, i) => Math.abs(v - R2[i]))));
""", {"meta": flat_meta(yaw=30.0, tilt=4.0), "tilt": 4.0})
    for pan in (30, -30, 10):
        r = res[f"pan{pan}"]
        assert r["n"] > 300
        assert r["min"] == pytest.approx(-pan, abs=1e-6) and r["max"] == pytest.approx(-pan, abs=1e-6)
        assert r["dd"] < 1e-6
        assert r["delta"]["pan"] == pytest.approx(pan, abs=1e-6) and abs(r["delta"]["tilt"]) < 1e-6
    for t in (3, -4):
        r = res[f"tilt{t}"]
        assert r["delta"]["tilt"] == pytest.approx(t, abs=1e-6) and abs(r["delta"]["pan"]) < 1e-6
        half = np.degrees(np.arctan(r["H"] / 2 / r["K"][4]))
        for d, st, d2 in r["rows"]:
            dep = np.degrees(np.arctan(r["h"] / d))
            a = dep + t
            if abs(dep - r["t0"] + t) >= half - 1e-3:
                assert st == 2, (t, d)                          # out of view
            elif a <= 0:
                assert st == 3, (t, d)                          # thrown past the horizon: lost
            else:
                assert st == 0 and d2 == pytest.approx(r["h"] / np.tan(np.radians(a)), rel=1e-6)
                assert (d2 < d) if t > 0 else (d2 > d)          # up pulls in, down throws out
    assert res["tilt-4"]["rows"][-1][1] == 3
    # turns together: the roll is about the phone's level forward, so they mix a little
    assert res["delta"]["pan"] == pytest.approx(12, abs=0.5)
    assert res["delta"]["tilt"] == pytest.approx(-3, abs=0.5)
    assert res["delta"]["roll"] == pytest.approx(5, abs=0.5)
    assert res["roundtrip"] < 1e-12


REAL_PKG = Path(__file__).resolve().parents[1] / "runs" / "trafalgar6376" / "v1" / "live"
REFRAME = """
const S = await import(LIVE + "scene.js");
const fs = await import("node:fs");
let sc, X;
if (job.pkg) {                                   // a real package: its scene and its track points
  const meta = JSON.parse(fs.readFileSync(job.pkg + "/scene.json", "utf8"));
  const tb = fs.readFileSync(job.pkg + "/terrain.f32");
  sc = P.prepareScene(meta, new Float32Array(Uint8Array.from(tb).buffer));
  const d = T.parseTracks(Uint8Array.from(fs.readFileSync(job.pkg + "/tracks.bin")).buffer, meta);
  X = d.X;
} else {
  sc = P.prepareScene(job.meta, null);
  const yaw = P.heading(sc.R_wc) / D, n = 2000; X = new Float64Array(3 * n);
  for (let i = 0; i < n; i++) { const b = (yaw - 40 + 80 * Math.random()) * D, r = 4 + 80 * Math.random();
    X[3 * i] = sc.C[0] + r * Math.sin(b); X[3 * i + 1] = sc.C[1] + r * Math.cos(b); X[3 * i + 2] = 1.7 * Math.random(); }
}
const pix = (K, R, C, i) => {                    // pinhole projection of point i: [u, v, z]
  const a = X[3 * i] - C[0], b = X[3 * i + 1] - C[1], c = X[3 * i + 2] - C[2];
  const x = R[0] * a + R[1] * b + R[2] * c, y = R[3] * a + R[4] * b + R[5] * c, z = R[6] * a + R[7] * b + R[8] * c;
  return [(K[0] * x + K[1] * y + K[2] * z) / z, (K[3] * x + K[4] * y + K[5] * z) / z, z];
};
const inside = (p) => p[2] > 0 && p[0] >= 0 && p[0] <= sc.W && p[1] >= 0 && p[1] <= sc.H;
const zero = mousePose(0, 0), n = X.length / 3;
for (const [name, pan, tilt, roll] of [["identity", 0, 0, 0], ["tilt_down_4", 0, -4, 0], ["pan_right_20", 20, 0, 0],
                                       ["turned", 20, -4, 3]]) {
  const cam = P.poseToCamera(sc, mousePose(pan, tilt, roll), zero, 1, 1), H = S.reframeMatrix(sc, cam);
  let m = 0, worst = 0, up = 0, left = 0;
  for (let i = 0; i < n; i++) {
    const x0 = pix(sc.K, sc.R_cw, sc.C, i), xv = pix(sc.K, cam.R_cw, cam.C, i);  // recording, virtual camera
    if (!inside(x0) || !inside(xv)) continue;
    // the plane's pixel xv shows the recording at H xv (the shader's lookup)
    const h = [0, 1, 2].map((r) => H[3 * r] * xv[0] + H[3 * r + 1] * xv[1] + H[3 * r + 2]);
    worst = Math.max(worst, Math.hypot(h[0] / h[2] - x0[0], h[1] / h[2] - x0[1]));
    m++; up += xv[1] < x0[1] - 1e-6; left += xv[0] < x0[0] - 1e-6;
  }
  out(name, { n: m, worst, up, left, H: Array.from(H), moved: Math.hypot(cam.C[0] - sc.C[0], cam.C[1] - sc.C[1], cam.C[2] - sc.C[2]) });
}
"""


@needs_node
@pytest.mark.parametrize("where", ["synthetic", "trafalgar6376"])
def test_reframed_video_puts_each_person_where_the_virtual_camera_sees_them(tmp_path, where):
    """The re-framed video (the recording fixed in the world, the violet frame a window over it):
    a person's pixel in the recording, mapped into the virtual camera's frame, lands where the
    virtual camera itself projects the true point (< 0.5 px). Tilt down: they slide up; turn
    right: they slide left (as the drawn tracks). At the start pose it is the glued video."""
    if where != "synthetic" and not (REAL_PKG / "scene.json").is_file():
        pytest.skip("the Trafalgar 6376 package is not here")
    job = {"meta": flat_meta(yaw=30.0, tilt=4.0)} if where == "synthetic" else {"pkg": str(REAL_PKG)}
    res = run_node(tmp_path, REFRAME, job)
    assert np.allclose(res["identity"]["H"], np.eye(3).ravel(), atol=1e-12)
    for name in ("identity", "tilt_down_4", "pan_right_20", "turned"):
        r = res[name]
        assert r["moved"] < 1e-9 and r["n"] > 200, (name, r["n"])
        assert r["worst"] < 0.5, (name, r["worst"])
    assert res["tilt_down_4"]["up"] == res["tilt_down_4"]["n"]         # people slide up
    assert res["pan_right_20"]["left"] == res["pan_right_20"]["n"]     # and left


# the rules config (live_rules.rules_config's shape): class 0 in no group (no cleaning), class 1
# "walker" cleaned (3 m/s, 0.5 m), stitched within 2 m
RULES = {"version": 1, "track_buffer_s": 1.0, "max_gap_s": 1.0, "stitch": True, "stitch_max_gap_s": 2.0,
         "groups": {"walkers": {"classes": ["walker"], "max_speed": 3.0, "jump_tolerance_m": 0.5,
                                "stitch_radius_m": 2.0}}}


@needs_node
def test_performed_trails_keep_the_pose_they_were_drawn_with(tmp_path):
    """Performed mode: each frame is drawn with the pose of the moment it was played; a pose
    change redraws only the current frame; scrubbing back clears what lies after."""
    a = np.zeros(20, TRACK_DTYPE)
    for f in range(20):
        a[f] = (1, f, f / 30.0, 0.0, 10.0 + 0.5 * f, 0.0, 1.0, 0, 0)
    (tmp_path / "tracks.bin").write_bytes(a.tobytes())
    res = run_node(tmp_path, """
const sc = P.prepareScene(job.meta, null), zero = mousePose(0, 0), I = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
const b = (await import("node:fs")).readFileSync(process.argv[2] + "/tracks.bin");
const d = T.parseTracks(Uint8Array.from(b).buffer, job.meta);
const tv = new T.TrackView(sc, d, job.rules, {});
const c0 = P.poseToCamera(sc, mousePose(0, 0), zero), c10 = P.poseToCamera(sc, mousePose(10, 0), zero, 1, 1);
const snap = () => Array.from({ length: d.n }, (_, i) => [tv.perf.state[i], Number.isNaN(tv.perf.X[3 * i]) ? null : +bearing(sc, tv.perf.X, i).toFixed(6)]);
tv.perform(9, c0, I, false); out("a", snap());
tv.perform(9, c10, I, true); out("b", snap());
tv.perform(14, c10, I, false); out("c", snap());
tv.build(14); out("drawn14", tv.solid.n);
tv.perform(5, c10, I, false); out("d", snap()); out("counts", [tv.counts(5), tv.counts(6)]);
out("kinds", Array.from(tv.out.kind));
tv.build(5); out("drawn5", tv.solid.n);
""", {"meta": flat_meta(yaw=0.0, tilt=3.0, frames=20), "rules": RULES})
    b0 = res["a"][0][1]
    assert all(s == 0 and v == pytest.approx(b0, abs=1e-6) for s, v in res["a"][:10])
    assert all(s == 255 and v is None for s, v in res["a"][10:])                 # not yet played
    assert all(v == pytest.approx(b0, abs=1e-6) for _, v in res["b"][:9])       # kept
    assert res["b"][9][1] == pytest.approx(b0 - 10, abs=1e-6)                   # the current frame
    assert all(v == pytest.approx(b0 - 10, abs=1e-6) for _, v in res["c"][10:15])
    assert all(s == 255 for s, _ in res["c"][15:])
    assert res["drawn14"] == 14                       # every pair up to frame 14 (both ends drawn)
    assert all(s == 255 for s, _ in res["d"][6:])                                 # cleared
    assert res["d"][5][1] == pytest.approx(b0 - 10, abs=1e-6)
    assert res["counts"] == [{"placed": 1, "guessed": 0, "dropped": 0}, {"placed": 0, "guessed": 0, "dropped": 0}]
    assert res["kinds"] == [0] * 6 + [255] * 14                       # the rules' verdict: cleared too
    assert res["drawn5"] == 5


@needs_node
def test_space_clutch_through_the_pose_input(tmp_path):
    """08 (a): held, the whole space turns about C0 with the hand (tilt down: the far side goes
    down) while the camera stays; let go, the hand steers the camera inside the turned space; no
    jump at either; held while any holder holds; a double-tap levels in ~0.3 s; zero resets only
    the camera."""
    res = run_node(tmp_path, """
const sc = P.prepareScene(job.meta, null), inp = new M.PoseInput(sc);
const ang = (R) => { const r = P.matrixToRotvec(R); return Math.hypot(r[0], r[1], r[2]) / D; };
const diff = (A, B) => Math.max(...Array.from(A, (v, i) => Math.abs(v - B[i])));
const far = (S) => P.applySpace(S, sc.C, Float64Array.from([sc.C[0] + 50 * Math.sin(sc.yaw), sc.C[1] + 50 * Math.cos(sc.yaw), 0]))[2];
inp.pose(mousePose(0, 0));
const c0 = Float64Array.from(inp.camera(0).R_wc);
out("press", [inp.press("key", 0), inp.held, ang(inp.space), diff(inp.camera(0).R_wc, c0)]);
inp.pose(mousePose(0, -15)); inp.camera(0.1);
out("held", [ang(inp.space), diff(inp.cam.R_wc, c0), far(inp.space), inp.tilted]);
out("release", [inp.release("key", 0.2), inp.held, ang(inp.space), diff(inp.camera(0.2).R_wc, c0)]);
inp.pose(mousePose(20, -15));
out("steer", [M.cameraDelta(sc, inp.camera(0.3)), ang(inp.space)]);
out("holders", [inp.press("phone 1", 0.4), inp.press("key", 0.4), inp.release("key", 0.4), inp.held,
                inp.release("phone 1", 0.4), inp.held, inp.press("phone 2", 0.5),
                inp.releaseWhere((w) => w.startsWith("phone"), 0.5), inp.held, ang(inp.space)]);
const cBefore = Float64Array.from(inp.camera(0.9).R_wc);
inp.level(1.0); out("level", [inp.levelling, ang(inp.space)]);
inp.camera(1.15); const mid = ang(inp.space);
inp.camera(1.35); out("levelled", [mid, ang(inp.space), inp.tilted, inp.levelling, diff(inp.cam.R_wc, cBefore)]);
inp.press("key", 2); inp.pose(mousePose(20, -25)); inp.camera(2.05); inp.release("key", 2.1);
const a10 = ang(inp.space); inp.zero("mouse");
out("zero", [a10, ang(inp.space), M.cameraDelta(sc, inp.camera(2.2))]);
""", {"meta": flat_meta(yaw=30.0, tilt=4.0)})
    ok, held, a, jump = res["press"]
    assert ok is True and held is True and a < 1e-9 and jump < 1e-12                # no jump
    a, frozen, z_far, tilted = res["held"]
    assert a == pytest.approx(15, abs=1e-6) and frozen < 1e-12 and tilted is True
    assert z_far < -10                                                # with the hand: the far side goes down
    ok, held, a, jump = res["release"]
    assert ok is True and held is False and a == pytest.approx(15, abs=1e-6) and jump < 1e-9
    delta, a = res["steer"]                                           # the hand steers the camera now
    assert delta["pan"] == pytest.approx(20, abs=0.5) and abs(delta["tilt"]) < 0.5 and a == pytest.approx(15, abs=1e-6)
    assert res["holders"] == [True, False, False, True, True, False, True, True, False, pytest.approx(15, abs=1e-6)]
    assert res["level"][0] is True
    mid, a, tilted, levelling, cam_moved = res["levelled"]
    assert 5 < mid < 10 and a < 1e-9 and tilted is False and levelling is False
    assert cam_moved < 1e-9                                           # levelling leaves the camera
    a10, a_after, delta = res["zero"]
    assert a10 == pytest.approx(10, abs=1e-6) and a_after == pytest.approx(a10, abs=1e-9)   # S kept
    assert max(abs(delta["pan"]), abs(delta["tilt"]), abs(delta["roll"])) < 1e-6            # the camera reset


@needs_node
def test_gain_range_stack_toggle_and_no_record(tmp_path):
    """29 Oct: the gain goes from 0.1x to 3x (the pose input clamps it there; the G keys step 0.1,
    0.25, 0.5, 0.75, 1 .. 3; the panel's slider the same range); the Passes section has Stack
    passes (K), off by default; the view has no Record any more."""
    res = run_node(tmp_path, """
const sc = P.prepareScene(job.meta, null), inp = new M.PoseInput(sc);
const g = (v) => { inp.setGain(v); return inp.gain; };
out("gain", [g(0.05), g(0.1), g(0.15), g(5), g(0), M.GAIN_MIN, M.GAIN_MAX, M.GAINS]);
""", {"meta": flat_meta(yaw=0.0, tilt=3.0)})
    assert res["gain"] == [0.1, 0.1, 0.15, 3, 1, 0.1, 3, [0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]]
    page = (LIVE / "live.html").read_text(encoding="utf-8")
    assert 'id="gain" min="0.1" max="3"' in page
    stack = page.split('id="pass-stack"')[0].rsplit("<input", 1)[1] + page.split('id="pass-stack"')[1].split(">")[0]
    assert "checked" not in stack and "Stack passes" in page
    assert 'id="record"' not in page and ">Record<" not in page
    js = (LIVE / "live.js").read_text(encoding="utf-8")
    assert 'k === "k" || k === "K"' in js and 'params.get("stack") === "1"' in js
    assert 'action === "record"' not in js and "recording: true" in js          # 05: every pass records itself


def test_passes_menu_one_camera_alone_and_the_collage():
    """29 Oct (b): the saved passes in their own menu (L, a button in the Passes section; Esc
    closes it), Stack, Undo and Clear stay in the side panel; one camera alone in the main view, the
    same again the whole collage (with one camera in play too; Solo since 29 Oct (4), see the next
    test); only the active cameras are in play (the phones', else the one kept
    for the mouse). Sandbox / collage (29 Oct c): a camera out of play with a layer stays frozen
    (parked); X or its menu clears its layer; collageLayers() gives the collage as data (09).
    Cameras vs Scene (29 Oct (3)): the panel's two parts; a camera stays until removed (its menu;
    a phone holding it is released through the hub)."""
    page = (LIVE / "live.html").read_text(encoding="utf-8")
    panel = page.split('id="panel"')[1].split("</aside>")[0]
    menu = page.split('id="pass-menu"')[1].split("</section>")[0]
    assert 'id="pass-list"' in menu and 'id="pass-list"' not in panel
    assert all(f'id="{i}"' in panel for i in ("pass-undo", "pass-clear", "pass-stack", "pass-menu-btn"))
    assert 'id="pass-menu-close"' in menu and page.split('id="pass-menu"')[0].endswith('class="pmenu" ')
    js = (LIVE / "live.js").read_text(encoding="utf-8")
    assert 'k === "l" || k === "L"' in js and 'k === "o" || k === "O"' in js
    assert "soloHide(true);\n  renderer.render(scene3, camera);\n  soloHide(false);" in js   # the main view only
    assert "function syncScene()" in js and "if (L.keep) w.add(L.keep);" in js                 # the mouse's camera
    assert "L.spots.length > 1 && L.spots.includes(s)" not in js                      # alone: one camera in play too
    assert 'k === "x" || k === "X"' in js and "function clearCamera(name)" in js and "function park(s)" in js
    assert "function collageLayers()" in js and all(f in js.split("window.live = ")[1] for f in ("collageLayers", "clearCamera", "setSolo"))
    assert "keepLength" not in (LIVE / "timeline.js").read_text(encoding="utf-8")       # the clock: the cameras in play
    assert ".spot-m" in (LIVE / "live.css").read_text(encoding="utf-8")
    # 29 Oct (3), Cameras vs Scene: the panel in two parts; each camera's menu; cameras stay until removed
    cams, scene = panel.split('id="part-scene"')
    assert 'id="spot-list"' in cams and 'id="cam-menu"' in cams and 'id="phone"' in cams
    assert all(f'id="{i}"' in cams for i in ("cm-focus", "cm-clear", "cm-remove", "cm-sheet", "cm-exports", "zero", "gain", "scale"))
    assert all(f'id="{i}"' in scene for i in ("sites", "passes", "pass-menu-btn", "collage-sheet", "help"))
    assert all(f'data-group="{g}"' in scene for g in ("trails", "ground", "under", "look", "section"))
    assert 'type: "release", spot: name' in js and "function removeCamera(name)" in js and "function markCamera(s)" in js
    assert "sendMessage(m)" in (LIVE / "pose.js").read_text(encoding="utf-8")
    assert (LIVE / "panels.js").read_text(encoding="utf-8").count("this.onFocus(e.spot, ev)") == 1


def test_focus_flies_the_view_whole_scene_and_solo():
    """29 Oct (4), after the phone test: Focus (O, Shift+click a camera in the list or on its panel,
    its menu) flies the view (plan or perspective) to frame that camera and its traces, the others
    stay visible; Whole scene (W, a button under the cameras) flies back out to the collage; the old
    focus (the others hidden) is Solo (I, its menu), the same again: all."""
    page = (LIVE / "live.html").read_text(encoding="utf-8")
    cams = page.split('id="part-cameras"')[1].split('id="part-scene"')[0]
    assert all(f'id="{i}"' in cams for i in ("cm-focus", "cm-solo", "whole-scene"))
    assert 'Focus <span class="key">O</span>' in cams and 'Solo <span class="key">I</span>' in cams
    assert 'Whole scene <span class="key">W</span>' in cams
    assert "O focus · W whole scene · I solo" in page.split('id="help"')[1]
    js = (LIVE / "live.js").read_text(encoding="utf-8")
    keys = js.split('window.addEventListener("keydown"')[-1].split("window.addEventListener")[0]
    assert 'k === "o" || k === "O") toggleFocus(L.focus)' in keys
    assert 'k === "w" || k === "W") wholeScene()' in keys
    assert 'k === "i" || k === "I") setSolo(L.solo ? null : L.focus)' in keys      # the old O
    for k in "zgGkKlLxXsShHpPfFvVmMcCuUbBiIwW":                                     # no binding taken over
        assert keys.count(f'k === "{k}"') == 1, k
    # the view: the framed camera (or the one alone) through the existing framing; the collage otherwise
    vp = js.split("function viewPose(v)")[1].split("function setView(")[0]
    assert "const one = L.solo || (L.framed && all.includes(L.framed) ? L.framed : null);" in vp
    assert "if (all.length <= 1 || one) {" in vp and "fs = one || fs;" in vp
    fr = js.split("function setFramed(s, notify = true)")[1].split("function setSolo(")[0]
    assert "setView(S.view === 1 || S.view === 2 ? S.view : lastOverview);" in fr   # flies, animated
    assert "if (!L.framed) L.solo = null;" in fr                                     # whole scene: all shown
    assert "const toggleFocus = (s) => setFramed(L && L.framed === s ? null : s);" in fr
    assert "const wholeScene = () => setFramed(null);" in fr
    assert "soloHide" not in fr                                                       # Focus hides nothing
    # the triggers: Shift+click in the list and on a panel, the menu's buttons
    assert "if (e.shiftKey) toggleFocus(r.s); else markCamera(r.s);" in js
    assert "if (e && e.shiftKey) toggleFocus(s); else setFocus(s);" in js
    assert '$("#cm-focus").onclick = () => { const s = menuSpot(); if (s) toggleFocus(s); };' in js
    assert '$("#cm-solo").onclick = () => { const s = menuSpot(); if (s) toggleSolo(s); };' in js
    assert '$("#whole-scene").onclick = () => wholeScene();' in js
    assert "setFramed" in js.split("window.live = ")[1]
    # a camera gone: framed no more
    assert js.count("if (L.framed === s) L.framed = null;") == 2


def test_video_boxes_only_on_request():
    """29 Oct (5), the user: the players' video boxes (panels) no longer open when a phone takes a
    camera; none by default; a camera's menu toggles its box (Video), B all the cameras in play."""
    page = (LIVE / "live.html").read_text(encoding="utf-8")
    cams = page.split('id="part-cameras"')[1].split('id="part-scene"')[0]
    assert 'id="cm-video"' in cams and 'Video <span class="key">B</span>' in cams
    assert "B video boxes" in page.split('id="help"')[1]
    js = (LIVE / "live.js").read_text(encoding="utf-8")
    assert "boxes: new Set()" in js                                                   # none open
    rp = js.split("function refreshPlayers()")[1].split("\n}\n")[0]
    assert "L.boxes.has(p.spot)" in rp                                                # only asked-for boxes
    assert "L.boxes.delete(s.name);" in js.split("function removeSpot(s)")[1].split("\n}\n")[0]   # leaving play: closed
    keys = js.split('window.addEventListener("keydown"')[-1].split("window.addEventListener")[0]
    assert keys.count('k === "b"') == 1 and 'k === "b" || k === "B") toggleBoxes()' in keys
    assert '$("#cm-video").onclick = () => { const s = menuSpot(); if (s) toggleBox(s); };' in js
    assert "toggleBoxes" in js.split("window.live = ")[1]


def lexicon_tracks():
    """1 walks away (seen throughout); 2 stands, frames 8-10 the footage's own guesses; 3 stands
    left (lost once the camera pans 20 right); 4 a walker that jumps 10 m at frame 10 (a new
    identity)."""
    rows = []
    for f in range(20):
        rows += [(1, f, 0.0, 10.0 + 0.05 * f, 0, 0), (2, f, 2.0, 15.0, 0, 1 if 8 <= f <= 10 else 0),
                 (3, f, -7.0, 15.0, 0, 0), (4, f, 7.0, 15.0 if f < 10 else 25.0, 1, 0)]
    a = np.zeros(len(rows), TRACK_DTYPE)
    for i, (tid, f, x, y, cls, src) in enumerate(rows):
        a[i] = (tid, f, f / 30.0, x, y, 0.0, 1.0, cls, src)
    return a


LEXICON = """
const Ru = await import(LIVE + "rules.js");
const sc = P.prepareScene(job.meta, null), zero = mousePose(0, 0), I = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
const fs = await import("node:fs");
const d = T.parseTracks(Uint8Array.from(fs.readFileSync(process.argv[2] + "/tracks.bin")).buffer, job.meta);
const tv = new T.TrackView(sc, d, job.rules, job.classes);
const c0 = P.poseToCamera(sc, zero, zero, 1, 1), c20 = P.poseToCamera(sc, mousePose(20, 0), zero, 1, 1);
const R = (o) => Array.from({ length: d.n }, (_, i) => [d.tid[i], d.frame[i], o.kind[i], o.cause[i], o.part[i]]);
const segs = (s) => Array.from({ length: s.n }, (_, k) => [...Array.from(s.pos.slice(6 * k, 6 * k + 6)), ...Array.from(s.col.slice(6 * k, 6 * k + 3))]);
const nanless = (A) => Array.from(A, (v) => (Number.isNaN(v) ? null : v));
out("tails", [tv.tails === undefined, "TAIL" in T, "FADE_S" in T]);
tv.perform(9, c0, I, false); tv.perform(19, c20, I, false);
out("perf", R(tv.out));
tv.build(19); out("white", { solid: tv.solid.n, guess: segs(tv.guess), opacity: tv.solid.material.opacity });
tv.setCause(true); tv.build(19);
out("cause", { solid: tv.solid.n, guess: segs(tv.guess), opacity: tv.solid.material.opacity,
               heads: Array.from({ length: tv.nHeads }, (_, k) => [...Array.from(tv.headPos.slice(3 * k, 3 * k + 3)), ...Array.from(tv.headCol.slice(3 * k, 3 * k + 3))]) });
out("counts", tv.counts(19));
tv.setCause(false);
// the space turned: what was drawn stays where it was drawn, the current frame is drawn turned
const S = P.rotvecToMatrix([-10 * D, 0, 0]);
const before = nanless(tv.perf.X);
tv.perform(19, c20, S, true);
out("turned", { before, after: nanless(tv.perf.X), want: nanless(P.applySpace(S, sc.C, P.project(sc, c20, d.X).X)),
                frame: Array.from(d.frame) });
// below the ground (frame 19 is turned under it): "dim" darkens the ends below, "veil" shows the crowd again through it
const segs6 = (s) => Array.from({ length: s.n }, (_, k) => [...Array.from(s.pos.slice(6 * k, 6 * k + 6)), ...Array.from(s.col.slice(6 * k, 6 * k + 6))]);
tv.build(19); const same = segs6(tv.solid);
tv.setUnder("dim"); tv.build(19); const dimmed = segs6(tv.solid);
tv.setUnder("veil");
out("under", { same, dimmed, ghosts: tv.ghost.map((g) => [g.visible, g.material.depthTest, g.parent === tv.crowd]),
               crowd: [tv.solid.obj, tv.guess.obj, tv.heads].every((o) => o.parent === tv.crowd),
               refApart: tv.ref.obj.parent === tv.group && tv.crowd.parent === tv.group });
tv.setUnder("same");
// the reference: the machine at rest, thin grey, up to the frame shown
tv.computeReference(); tv.build(9);
out("ref9", { n: tv.ref.n, cols: [...new Set(segs(tv.ref).map((s) => s.slice(6).join(",")))] });
// whole clip: judged in the space's own frame, then turned = turned, then judged
tv.setMode("whole");
const local = P.project(sc, c20, d.X, { X: new Float32Array(3 * d.n), state: new Uint8Array(d.n) });
tv.judgeWhole(local); tv.turnWhole(S); tv.build(19);
const R2 = new Ru.Rules(d, job.rules, job.classes, { rest: tv.restDraw });
R2.all(P.applySpace(S, sc.C, local.X), local.state);
out("whole", { a: R(tv.whole), b: R(R2.out), xa: nanless(tv.whole.X), xb: nanless(R2.out.X), ref: tv.ref.n });
"""


@needs_node
def test_lexicon_the_machines_rules_draw_the_trails(tmp_path):
    """08 (c): coloured solid = placed, white dashes = the machine's guesses (the footage's and
    the ones Live causes), a new identity breaks the line, nothing where dropped, no fading tail;
    cause mode: Live's guesses violet, the footage's white, the solid dimmed; a turned space
    leaves performed trails where they were drawn; the whole clip judged then turned is the same
    as turned then judged; the reference is the machine at rest, thin grey."""
    (tmp_path / "tracks.bin").write_bytes(lexicon_tracks().tobytes())
    res = run_node(tmp_path, LEXICON, {"meta": flat_meta(yaw=0.0, tilt=3.0, frames=20), "rules": RULES,
                                       "classes": {"0": "thing", "1": "walker"}})
    assert res["tails"] == [True, False, False]                       # no fading tail
    perf = {(t, f): (k, c, p) for t, f, k, c, p in res["perf"]}
    assert all(perf[1, f] == (0, 0, 0) for f in range(20))            # placed
    assert all(perf[2, f][1] == 1 for f in (8, 9, 10))                # the footage's guesses
    assert all(perf[3, f] == (0, 0, 0) for f in range(10))
    assert all(perf[3, f] == (1, 2, 0) for f in range(10, 20))        # lost by the pan: predicted, Live's
    assert [perf[4, f][2] for f in range(20)] == [0] * 10 + [1] * 10  # the jump: a new identity
    assert all(perf[4, f][0] == 0 for f in range(20))
    violet = pytest.approx([0x9B / 255, 0x82 / 255, 0xD9 / 255], abs=1e-6)
    w, c = res["white"], res["cause"]
    # solid: 1: 19, 2: 19 - 4 (frames 7-11 touch a guess), 3: 9 (then predicted), 4: 18 (the break)
    assert w["solid"] == c["solid"] == 61 and w["opacity"] == 1 and c["opacity"] == pytest.approx(0.3)
    assert len(w["guess"]) == len(c["guess"]) == 14
    assert all(s[6:] == [1, 1, 1] for s in w["guess"])                # default: every guess white
    live = [s for s in c["guess"] if s[0] < -5]                       # track 3 (x = -7)
    assert len(live) == 10 and all(s[6:] == violet for s in live)     # Live's: violet
    assert all(s[6:] == [1, 1, 1] for s in c["guess"] if s[0] >= -5)  # the footage's: white
    heads = c["heads"]
    assert len(heads) == 4 and sum(h[3:] == violet for h in heads) == 1
    assert res["counts"] == {"placed": 3, "guessed": 1, "dropped": 0}
    t = res["turned"]
    for i, fr in enumerate(t["frame"]):
        got, was, want = t["after"][3 * i:3 * i + 3], t["before"][3 * i:3 * i + 3], t["want"][3 * i:3 * i + 3]
        if fr < 19:
            assert got == was                                         # stays in the world where drawn
        elif want[0] is None:
            assert got == [None] * 3
        else:
            assert got == pytest.approx(want, abs=1e-4)               # the current frame: turned
    assert any(t["after"][3 * i + 2] < -1 for i, fr in enumerate(t["frame"])
               if fr == 19 and t["after"][3 * i] is not None)
    u = res["under"]
    assert u["crowd"] and u["refApart"]                               # only the crowd turns; the reference apart
    assert u["ghosts"] == [[True, False, True]] * 3                   # veil: the crowd again, through the ground
    assert len(u["same"]) == len(u["dimmed"]) and any(s[5] < -1 for s in u["dimmed"])
    for s, d in zip(u["same"], u["dimmed"]):
        assert s[:6] == d[:6]
        for z, c0, c1 in ((s[2], s[6:9], d[6:9]), (s[5], s[9:12], d[9:12])):
            assert c1 == pytest.approx([0.32 * v for v in c0] if z < 0.03 else c0, abs=1e-6)   # (lines sit 0.05 up)
    r9 = res["ref9"]
    assert r9["n"] == 36 and len(r9["cols"]) == 1
    assert [float(v) for v in r9["cols"][0].split(",")] == pytest.approx([0.45] * 3, abs=1e-6)   # grey
    wh = res["whole"]
    assert wh["a"] == wh["b"]                                         # kinds, causes, parts
    xa, xb = wh["xa"], wh["xb"]
    assert [v is None for v in xa] == [v is None for v in xb]
    assert np.allclose([v for v in xa if v is not None], [v for v in xb if v is not None], atol=1e-3)
    assert wh["ref"] == 75                                            # the whole reference


RECORD = """
const Rc = await import(LIVE + "record.js");
const Rp = await import(LIVE + "replay.js");
const Ps = await import(LIVE + "passes.js");
const fs = await import("node:fs");
const sc = P.prepareScene(job.meta, null), zero = mousePose(0, 0), I = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
const d = T.parseTracks(Uint8Array.from(fs.readFileSync(process.argv[2] + "/tracks.bin")).buffer, job.meta);
// live: one camera panned right from frame 10, it leaves at 13 and joins again at 15 (a new stint),
// the clock paused at the end (two frames alike: nothing written), then a last turn
const rec = new Rc.Recorder();
rec.begin(1000, "2026-10-29T10:00:00");
rec.event("pass_start", {}, 1000, 0);
let tv = new T.TrackView(sc, d, job.rules, job.classes), now = 1000, prev = null, joined = true, left = null;
const join = (f) => { rec.camera("cam", { name: "cam", id: "cam", points: d.n }); rec.event("camera_join", { camera: "cam", player: 1, value: { join_frame: f } }, now, f / 30); };
join(0);
const plan = [];
for (let f = 0; f < 20; f++) plan.push([f, f < 10 ? 0 : 2 * (f - 9)]);
plan.push([19, 20], [19, 20], [19, 23]);
let rows = 0, first = true;
for (const [f, pan] of plan) {
  now += 33.3;
  rec.frame(now, f / 30);
  if (f === 13 && joined) { left = tv.snapshot(); rec.event("camera_leave", { camera: "cam", player: 1 }); joined = false; }
  if (f === 15 && !joined) { tv = new T.TrackView(sc, d, job.rules, job.classes); tv.perfFrame = f - 1; join(f); joined = true; first = true; prev = null; }
  if (joined) {
    const cam = P.poseToCamera(sc, mousePose(pan, 0), zero, 1, 1), changed = prev !== null && prev !== pan;
    tv.perform(f, cam, I, changed);
    prev = pan;
    const c = Rc.machineCounts(d, tv.out, tv.perf.state, f);
    if (rec.row("cam", { cam, S: I, frame: f, pose: mousePose(pan, 0), gain: 1, scale: 1, counts: c, player: 1, src: "mouse",
                         status: "ok", flags: Rc.PERFORMED | (changed ? Rc.CHANGED : 0) | (first ? Rc.JOINED : 0) })) rows++;
    first = false;
  }
  rec.commit(1, Rc.PLAYING);
}
rec.event("pass_end", { value: { reason: "test", complete: true } }, now, 19 / 30);
const live = Ps.mergeSnaps(left, tv.snapshot());
const snap = rec.snapshot((n) => n + ".rec");
for (const [k, v] of Object.entries(snap.files)) fs.writeFileSync(process.argv[2] + "/" + k, typeof v === "string" ? v : Buffer.from(v));
fs.writeFileSync(process.argv[2] + "/live.bin", Buffer.from(Ps.packPass(live)));
fs.writeFileSync(process.argv[2] + "/rec.json", JSON.stringify(snap.meta));
// replayed: a fresh TrackView driven by the recording alone
const data = Rp.readRecording(snap.meta, snap.files["clock.rec"], { "cam.rec": snap.files["cam.rec"] }, snap.files["events.jsonl"]);
const play = new Rp.Replay(data, new Map([["cam", new T.TrackView(sc, d, job.rules, job.classes)]]));
const counts = [];
play.onRow = (name, rws, r, t) => counts.push([rws.view(r).counts, Array.from(Rc.machineCounts(d, t.out, t.perf.state, rws.frame(r)))]);
const types = [];
play.onEvent = (e) => types.push(e.type);
play.run();
out("cmp", Rp.compareDrawings(play.drawings().get("cam"), live));
out("n", [snap.meta.clock.rows, snap.meta.cameras[0].rows, rows, plan.length, data.events.length, play.done, play.seconds]);
out("counts", counts);
out("types", types);
out("view0", data.cams.get("cam").view(0));
"""


@needs_node
def test_a_pass_records_itself_and_its_replay_draws_it_again(tmp_path):
    """05: the recording (record.js) keeps what changed only, every row read back; the pass replayed
    from its recording alone (replay.js) draws exactly what the live pass drew, a camera that left
    and came back included, and the machine's counts per frame are those recorded. The files are
    the format live_passes.py reads (numpy dtypes)."""
    from groundtrack import live_passes as lp

    (tmp_path / "tracks.bin").write_bytes(lexicon_tracks().tobytes())
    res = run_node(tmp_path, RECORD, {"meta": flat_meta(yaw=0.0, tilt=3.0, frames=20), "rules": RULES,
                                      "classes": {"0": "thing", "1": "walker"}})
    cmp = res["cmp"]
    assert cmp["points"] > 40 and cmp["max_m"] == 0 and cmp["kind"] == 0 and cmp["nan"] == 0, cmp
    clock_rows, cam_rows, written, frames, n_events, done, secs = res["n"]
    # 20 frames less the two it was away (13, 14), the first paused frame (its "changed" flag goes),
    # the last turn; the second paused frame is the same as the first: not written
    assert cam_rows == written == 20 and frames == 23 and clock_rows == 22 and done is True
    assert n_events == 5 and res["types"] == ["pass_start", "camera_join", "camera_leave", "camera_join", "pass_end"]
    assert all(a == b for a, b in res["counts"]) and len(res["counts"]) == 20
    v0 = res["view0"]
    assert v0["frame"] == 0 and v0["src"] == "mouse" and v0["player"] == 1 and v0["flags"] & 32 and v0["gain"] == 1
    # the same bytes through numpy
    clock = np.fromfile(tmp_path / "clock.rec", lp.CLOCK_DTYPE)
    cam = np.fromfile(tmp_path / "cam.rec", lp.CAM_DTYPE)
    assert len(clock) == clock_rows and len(cam) == cam_rows and secs == pytest.approx(clock["t"][-1])
    assert np.all(np.diff(clock["t"]) > 0) and np.all(clock["flags"] == lp.CLOCK_PLAYING)
    assert list(cam["frame"][:3]) == [0, 1, 2] and cam["k"][0] == 0 and np.all(np.diff(cam["k"].astype(int)) > 0)
    assert np.allclose([np.linalg.det(R) for R in cam["R_wc"]], 1) and np.all(cam["src"] == 2)
    assert int(cam["flags"][13]) & lp.JOINED and cam["frame"][13] == 15                 # the second stint
    assert np.all(cam["flags"] & lp.PERFORMED) and int(cam["flags"][-1]) & lp.CHANGED
    ev = [json.loads(x) for x in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert ev[3]["value"] == {"join_frame": 15} and ev[2]["k"] == 13


RECORD_SEEK = """
const Rc = await import(LIVE + "record.js");
const Rp = await import(LIVE + "replay.js");
const fs = await import("node:fs");
const I = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]), zero = mousePose(0, 0);
const buf = Uint8Array.from(fs.readFileSync(process.argv[2] + "/tracks.bin")).buffer;
const cams = job.metas.map((meta, n) => { const sc = P.prepareScene(meta, null), d = T.parseTracks(buf, meta);
  return { name: "cam" + n, sc, d, tv: new T.TrackView(sc, d, job.rules, job.classes), prev: null, off: 3 * n }; });
// two cameras, the second's clip 3 frames later; played to 16, scrubbed back to 5, played to the end
// while both pan, then paused at the end while they still turn (the last frame drawn again)
const plan = [];
for (let T = 0; T <= 16; T++) plan.push([T, T < 8 ? 0 : 3 * (T - 7)]);
for (let T = 5; T <= 22; T++) plan.push([T, -2 * (T - 5)]);
for (let k = 0; k < 6; k++) plan.push([22, -34 + 5 * k]);
const rec = new Rc.Recorder();
rec.begin(1000, "2026-10-29T10:00:00");
rec.event("pass_start", {}, 1000, 0);
for (const c of cams) { rec.camera(c.name, { name: c.name, points: c.d.n }); rec.event("camera_join", { camera: c.name, value: { join_frame: 0 } }, 1000, 0); }
let now = 1000, lastT = 0;
for (const [T, pan] of plan) {
  now += 33.3;
  rec.frame(now, T / 30);
  if (T < lastT) rec.event("seek", { value: T / 30 });
  lastT = T;
  for (const c of cams) {
    const f = Math.min(Math.max(T - c.off, 0), c.d.frames - 1), p = pan * (c.off ? -1 : 1);
    const cam = P.poseToCamera(c.sc, mousePose(p, 0), zero, 1, 1), changed = c.prev !== null && c.prev !== p;
    c.tv.perform(T - c.off, cam, I, changed);
    c.prev = p;
    rec.row(c.name, { cam, S: I, frame: T - c.off, pose: null, gain: 1, scale: 1, counts: Rc.machineCounts(c.d, c.tv.out, c.tv.perf.state, f),
                      player: 0, src: "mouse", status: "ok", flags: Rc.PERFORMED | (changed ? Rc.CHANGED : 0) });
  }
  rec.commit(1, Rc.PLAYING);
}
const snap = rec.snapshot((n) => n + ".rec");
const bufs = Object.fromEntries(cams.map((c) => [c.name + ".rec", snap.files[c.name + ".rec"]]));
const play = new Rp.Replay(Rp.readRecording(snap.meta, snap.files["clock.rec"], bufs, snap.files["events.jsonl"]),
                           new Map(cams.map((c) => [c.name, new T.TrackView(c.sc, c.d, job.rules, job.classes)])));
const counts = [];
play.onRow = (name, rws, r, t) => counts.push(rws.view(r).counts.join() === Array.from(Rc.machineCounts(t.d, t.out, t.perf.state, Math.min(Math.max(rws.frame(r), 0), t.d.frames - 1))).join());
play.run();
const dr = play.drawings();
out("cmp", cams.map((c) => Rp.compareDrawings(dr.get(c.name), c.tv.snapshot())));
out("counts", [counts.length, counts.every(Boolean)]);
out("frames", cams.map((c) => c.tv.perfFrame));
"""


@needs_node
def test_replay_is_exact_after_a_scrub_back_with_two_cameras(tmp_path):
    """05: two cameras in the scene (the second's clip 3 frames later), the clock scrubbed back while
    both draw, then played on and paused at the end while both still turn: the replay of the
    recording draws exactly what each camera drew, and the counts per row are the recorded ones."""
    (tmp_path / "tracks.bin").write_bytes(lexicon_tracks().tobytes())
    res = run_node(tmp_path, RECORD_SEEK, {"metas": [flat_meta(yaw=0.0, tilt=3.0, frames=20), flat_meta(yaw=10.0, tilt=4.0, frames=20)],
                                           "rules": RULES, "classes": {"0": "thing", "1": "walker"}})
    for c in res["cmp"]:
        assert c["points"] > 40 and c["max_m"] == 0 and c["kind"] == 0 and c["nan"] == 0, res["cmp"]
    assert res["counts"][1] is True and res["counts"][0] > 70
    assert res["frames"] == [19, 19]                                  # both drew to their clip's end


SMOOTH = """
const sc = P.prepareScene(job.meta, null), z = mousePose(0, 0);
const same = (a, b) => a.length === b.length && Array.from(a).every((v, i) => Object.is(v, b[i]));
const tr = (c) => same(c.R_cw, [0, 3, 6, 1, 4, 7, 2, 5, 8].map((i) => c.R_wc[i]));
// off (the default): the mapped camera exactly, ticks change nothing
const off = new M.PoseInput(sc);
off.pose(z); off.pose(mousePose(12, -5, 3));
const v0 = off.version; off.tick(1); off.tick(2);
out("off", [off.smooth, same(off.camera(2).R_wc, P.poseToCamera(sc, mousePose(12, -5, 3), z).R_wc), off.version === v0]);
// a 10 deg pan from rest just after the frame at t = 0: the same at 30 and 120 fps
const step = (fps) => {
  const inp = new M.PoseInput(sc);
  inp.pose(z); inp.camera(-1);
  const before = Float64Array.from(inp.cam.R_wc);
  inp.setSmooth(true, -1);
  let t = 0, jump = 0;
  for (let k = -fps; k <= 0; k++) { t = k / fps; inp.tick(t); inp.camera(t); if (!k) jump = Math.max(...Array.from(inp.cam.R_wc, (v, i) => Math.abs(v - before[i]))); }
  inp.pose(mousePose(10, 0));
  let exact = true;
  for (let k = 1; k <= 0.3 * fps; k++) { t = k / fps; inp.tick(t); if (inp.version !== inp._v) { inp._v = inp.version; inp.camera(t); } exact = exact && tr(inp.cam); }
  const pan = M.cameraDelta(sc, inp.cam).pan;
  let n = 0;                                                      // settles: the mapped camera exactly, then at rest
  while (inp.sm.moving && n < 1000) { t += 1 / fps; inp.tick(t); inp.camera(t); n++; }
  const v = inp.version; inp.tick(t + 1); inp.tick(t + 2);
  return [pan, jump, exact, n, same(inp.cam.R_wc, P.poseToCamera(sc, mousePose(10, 0), z).R_wc) && tr(inp.cam), inp.version === v];
};
out("step", [step(30), step(120)]);
// a steady 30 deg/s pan at 60 Hz: the lag; then +-0.5 deg noise: the jitter left
const lag = new M.PoseInput(sc);
lag.pose(z); lag.setSmooth(true, 0);
let t = 0;
for (let k = 1; k <= 90; k++) { t = k / 60; lag.pose(mousePose(30 * t, 0)); lag.tick(t); lag.camera(t); }
const lagDeg = 30 * t - M.cameraDelta(sc, lag.cam).pan;
let seed = 3; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5);
const raw = [], sm = [];
for (let k = 1; k <= 240; k++) {
  t += 1 / 60; const tilt = rnd(); lag.pose(mousePose(45, tilt)); lag.tick(t); lag.camera(t);
  if (k > 60) { raw.push(tilt); sm.push(M.cameraDelta(sc, lag.cam).tilt); }
}
const sd = (a) => { const m = a.reduce((s, v) => s + v, 0) / a.length; return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length); };
out("lag", [lagDeg, sd(sm) / sd(raw), M.SMOOTH_S]);
// off again: the mapped camera at once
lag.setSmooth(false); lag.camera(t + 0.01);
out("offAgain", same(lag.cam.R_wc, P.poseToCamera(sc, mousePose(45, raw[raw.length - 1]), z).R_wc));
"""


@needs_node
def test_smooth_eases_the_camera_off_by_default(tmp_path):
    """08, 29 Oct (4), the user: Smooth per camera, off by default (off: the camera follows the pose
    exactly, ticks change nothing). On: exponential smoothing of the virtual camera, rotation by slerp,
    position by lerp, one fixed strength (SMOOTH_S = 0.1 s): no jump when switched on, frame-rate
    independent (30 and 120 fps agree), a steady turn trails by about SMOOTH_S, the jitter cut, and
    at rest it is the mapped camera exactly (R_cw = R_wcᵀ exactly: as a replay rebuilds it)."""
    res = run_node(tmp_path, SMOOTH, {"meta": flat_meta(yaw=30.0, tilt=4.0)})
    assert res["off"] == [False, True, True]
    a, b = res["step"]
    for pan, jump, exact, n, mapped, rest in (a, b):
        assert jump < 1e-12 and exact is True and mapped is True and rest is True and 0 < n < 1000
        assert pan == pytest.approx(10 * (1 - np.exp(-3)), abs=1e-6)        # 95 % there after 0.3 s
    assert a[0] == pytest.approx(b[0], abs=1e-9)
    lag, jitter, tau = res["lag"]
    assert tau == 0.1 and 2.6 < lag < 3.4 and jitter < 0.4                   # 30 deg/s: ~3 deg behind
    assert res["offAgain"] is True


RECORD_SMOOTH = """
const Rc = await import(LIVE + "record.js");
const Rp = await import(LIVE + "replay.js");
const fs = await import("node:fs");
const sc = P.prepareScene(job.meta, null), I = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
const d = T.parseTracks(Uint8Array.from(fs.readFileSync(process.argv[2] + "/tracks.bin")).buffer, job.meta);
// live: a jittery pan from the phone at uneven frame times; Smooth on at frame 4, off at 16
const inp = new M.PoseInput(sc), tv = new T.TrackView(sc, d, job.rules, job.classes), key = new Float64Array(12).fill(NaN);
const rec = new Rc.Recorder();
rec.begin(1000, "2026-10-29T10:00:00");
rec.event("pass_start", {}, 1000, 0);
rec.camera("cam", { name: "cam", id: "cam", points: d.n });
rec.event("camera_join", { camera: "cam", player: 1, value: { join_frame: 0 } }, 1000, 0);
let now = 1000, seed = 5, off = 0, first = true;
const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5);
inp.pose({ ...mousePose(0, 0), src: "web" });
for (let f = 0; f < 20; f++) {
  for (let s = 0; s < 2; s++) {                                   // two frames a clip frame, uneven
    now += 12 + 10 * Math.abs(rnd());
    const t = now / 1000;
    rec.frame(now, f / 30);
    if (f === 4 && s === 0) { rec.event("smooth", { camera: "cam", player: 1, value: true }); inp.setSmooth(true, t); }
    if (f === 16 && s === 0) { rec.event("smooth", { camera: "cam", player: 1, value: false }); inp.setSmooth(false, t); }
    const m = { ...mousePose(f < 18 ? 2.5 * f + rnd() : 40, rnd() * 0.6), src: "web" };
    inp.pose(m); inp.tick(t);
    const cam = inp.camera(t);
    let changed = false;
    for (let i = 0; i < 9; i++) if (key[i] !== cam.R_cw[i]) { key[i] = cam.R_cw[i]; changed = true; }
    for (let i = 0; i < 3; i++) if (key[9 + i] !== cam.C[i]) { key[9 + i] = cam.C[i]; changed = true; }
    tv.perform(f, cam, I, changed);
    const raw = P.poseToCamera(sc, m, { ...mousePose(0, 0), src: "web" });
    const a = P.matrixToRotvec(Float64Array.from([0, 1, 2].flatMap((i) => [0, 1, 2].map((j) => cam.R_wc[3 * i] * raw.R_wc[3 * j] + cam.R_wc[3 * i + 1] * raw.R_wc[3 * j + 1] + cam.R_wc[3 * i + 2] * raw.R_wc[3 * j + 2]))));
    off = Math.max(off, Math.hypot(...a) / D);                    // the recorded camera against the raw mapping
    rec.row("cam", { cam, S: I, frame: f, pose: m, gain: 1, scale: 1, counts: Rc.machineCounts(d, tv.out, tv.perf.state, f), player: 1,
                     src: "web", status: "ok", flags: Rc.PERFORMED | (changed ? Rc.CHANGED : 0) | (first ? Rc.JOINED : 0) | (inp.smooth ? Rc.SMOOTH : 0) });
    first = false;
    rec.commit(1, Rc.PLAYING);
  }
}
rec.event("pass_end", { value: { reason: "test", complete: true } }, now, 19 / 30);
const snap = rec.snapshot((n) => n + ".rec");
for (const [k, v] of Object.entries(snap.files)) fs.writeFileSync(process.argv[2] + "/" + k, typeof v === "string" ? v : Buffer.from(v));
const play = new Rp.Replay(Rp.readRecording(snap.meta, snap.files["clock.rec"], { "cam.rec": snap.files["cam.rec"] }, snap.files["events.jsonl"]),
                           new Map([["cam", new T.TrackView(sc, d, job.rules, job.classes)]]));
play.run();
out("cmp", Rp.compareDrawings(play.drawings().get("cam"), tv.snapshot()));
out("off", [off, Rc.SMOOTH]);
"""


@needs_node
def test_replay_is_exact_with_a_smoothed_camera(tmp_path):
    """Smooth is recorded with the pass (a "smooth" event, the SMOOTH flag on the camera's rows) and
    the rows keep the eased camera (not the raw mapping), so the replay draws exactly what the pass
    drew; rows without the flag (Smooth off, older passes) are the camera as it followed the pose."""
    from groundtrack import live_passes as lp

    (tmp_path / "tracks.bin").write_bytes(lexicon_tracks().tobytes())
    res = run_node(tmp_path, RECORD_SMOOTH, {"meta": flat_meta(yaw=0.0, tilt=3.0, frames=20), "rules": RULES,
                                             "classes": {"0": "thing", "1": "walker"}})
    cmp = res["cmp"]
    assert cmp["points"] > 40 and cmp["max_m"] == 0 and cmp["kind"] == 0 and cmp["nan"] == 0, cmp
    eased, flag = res["off"]
    assert eased > 0.5 and flag == lp.SMOOTH == 64                     # the eased camera was recorded
    cam = np.fromfile(tmp_path / "cam.rec", lp.CAM_DTYPE)
    on = (cam["flags"] & lp.SMOOTH) > 0
    assert not on[:8].any() and on[8:32].all() and not on[32:].any()  # on from frame 4 to 16 (2 rows a frame)
    ev = [json.loads(x) for x in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(e["type"], e["value"], e["k"]) for e in ev if e["type"] == "smooth"] == [("smooth", True, 8), ("smooth", False, 32)]
    assert "smooth" in lp.EVENTS


def test_smooth_switch_in_the_camera_menu_and_key_j():
    """Smooth: a switch in each camera's menu (off by default), J on the marked camera (a free key:
    no binding taken over), in the help; the view records it and shows it per camera."""
    page = (LIVE / "live.html").read_text(encoding="utf-8")
    menu = page.split('id="cam-menu"')[1].split("</section>")[0]
    sw = menu.split('id="cm-smooth"')[0].rsplit("<input", 1)[1] + menu.split('id="cm-smooth"')[1].split(">")[0]
    assert 'type="checkbox"' in sw and "checked" not in sw and 'Smooth <span class="key">J</span>' in menu
    assert menu.index('id="cm-exports"') < menu.index('id="cm-smooth"') < menu.index('id="cm-mouse"')   # for phones too
    assert "J smooth" in page.split('id="help"')[1]
    js = (LIVE / "live.js").read_text(encoding="utf-8")
    keys = js.split('window.addEventListener("keydown"')[-1].split("window.addEventListener")[0]
    assert keys.count('k === "j"') == 1 and 'k === "j" || k === "J") setSmooth(!L.focus.inp.smooth)' in keys
    for k in "zgGkKlLxXsShHpPfFvVmMcCuUbBiIoOwW":
        assert keys.count(f'k === "{k}"') == 1, k
    assert '$("#cm-smooth").onchange = (e) => { const s = menuSpot(); if (s) setSmooth(e.target.checked, s);' in js
    assert '$("#cm-smooth").checked = s.inp.smooth;' in js and 'recAt("smooth", s, !!on)' in js
    assert "(inp.smooth ? SMOOTH : 0)" in js and "inp.tick(clock());" in js and "setSmooth" in js.split("window.live = ")[1]
    assert "export const SMOOTH = 64;" in (LIVE / "record.js").read_text(encoding="utf-8")


SPOTS = """
const Sp = await import(LIVE + "spots.js");
const Ps = await import(LIVE + "passes.js");
const fs = await import("node:fs");
// a tint each (the violet's hue, distinct, never orange); a camera joining gets a free one
out("tints", Sp.TINTS.map((t) => [t, Sp.hue(Sp.hexRGB(t))]));
out("free", [Sp.freeTint([]), Sp.freeTint([0]), Sp.freeTint([0, 2]), Sp.freeTint([0, 1, 2, 3]), Sp.freeTint([0, 0, 1, 2, 3, 3])]);
// placed by the packages' origins, the start camera's being the scene's
const o0 = { E: 530000, N: 180400, datum_z: 4.5 };
out("offsets", [Sp.offsetFrom({ E: 530030, N: 180338, datum_z: 4.26 }, o0), Sp.offsetFrom(o0, o0), Sp.offsetFrom(null, o0),
                Sp.offsetFrom({ E: 1, N: 2, datum_z: 3 }, null), Sp.offsetFrom({ E: 530010, N: 180400 }, o0)]);
// a word of the URL (players=, focus=): a camera's name, number, clip or clip/run
const cams = [{ name: "North \\u00b7 6354", id: "6354", clip: "trafalgar6354", run_name: "v1" },
              { name: "North-west \\u00b7 6363", id: "6363", clip: "trafalgar6363", run_name: "v1" }];
const id = (c) => (c ? c.id : null);
out("find", [id(Sp.findCamera("6363", cams)), id(Sp.findCamera("north \\u00b7 6354", cams)), id(Sp.findCamera("trafalgar6363", cams)),
             id(Sp.findCamera("TRAFALGAR6354/v1", cams)), id(Sp.findCamera("6999", cams)), id(Sp.findCamera("", cams)),
             id(Sp.findCamera(null, cams))]);
// the shared clock: past its end a clip holds its last frame; a seek is clamped per clip
out("clock", [Sp.clipTime(7, 5.87, false), Sp.clipTime(3, 5.87, false), Sp.clipTime(-1, 5, false), Sp.clipTime(3, 0, false),
              Sp.loopDiff(2, 1, 0), Sp.loopDiff(0.1, 5.8, 5.87)]);
out("rects", { none: Sp.panelRects(0, 1600, 900), two: Sp.panelRects(2, 1600, 900, { top: 52, bottom: 56, right: 248 }),
               many: Sp.panelRects(6, 1600, 600, { top: 52, bottom: 56, right: 0 }) });
// the Timeline with fake videos (no DOM): one clock from 0 to the longest clip's end, no looping
const el = () => ({ setAttribute() {}, addEventListener() {}, value: "", textContent: "", title: "", disabled: false });
const bar = { _q: {}, set innerHTML(v) {}, querySelector(sel) { const k = sel.split('"')[1]; return (this._q[k] = this._q[k] || el()); },
              classList: { add() {}, remove() {} }, matches: () => false, addEventListener() {} };
globalThis.window = { addEventListener() {} };
globalThis.HTMLVideoElement = class {};
const Tl = await import(LIVE + "timeline.js");
const vid = () => ({ currentTime: 0, paused: true, seeking: false, ended: false, playbackRate: 1, loop: true,
                     play() { this.paused = false; return Promise.resolve(); }, pause() { this.paused = true; } });
const vs = [vid(), vid(), vid()], keys = [{ n: "a" }, { n: "b" }, { n: "c" }];
const events = [];
let passes = 0;
const tl = new Tl.Timeline(bar, (w) => events.push(w), () => passes++);
tl.add(keys[0], { video: vs[0], fps: 30, frames: 176 });
tl.add(keys[1], { video: vs[1], fps: 30, frames: 1228 });
tl.add(keys[2], { video: vs[2], fps: 30, frames: 300 });
tl.seekTime(10);
tl.poll();
const seek = { T: tl.time, frames: keys.map((k) => tl.frameOf(k)), dur: tl.duration, loops: vs.map((v) => v.loop),
               max: bar._q.scrub.max, held: tl.timeOf(keys[0]) };
tl.setFocus(keys[1]); const focus = [tl.frame, tl.frames];
tl.seekTime(0); tl.play();
tl.wall0 -= 7000; tl.c.forEach((k) => { k.fixAt = 0; });           // 7 s later
tl.poll();
const play = { T: tl.time, times: vs.map((v) => v.currentTime), playing: vs.map((v) => !v.paused) };
tl.wall0 -= 60000;                                                // past the longest clip's end
tl.poll();
const end = { T: tl.time, playing: tl.playing, atEnd: tl.atEnd, paused: vs.map((v) => v.paused), events: events.slice(), passes,
              frames: keys.map((k) => tl.frameOf(k)) };
tl.play();                                                        // a new pass: every clip from 0
const again = { T: tl.time, passes, playing: vs.map((v) => !v.paused), times: vs.map((v) => v.currentTime) };
tl.remove(keys[1]);                                               // the long one leaves
const removed = { dur: tl.duration, max: bar._q.scrub.max, n: tl.c.size };
tl.pause(); tl.seekTime(3);
const v3 = vid(), k3 = { n: "d" };
tl.add(k3, { video: v3, fps: 30, frames: 60 });                   // joins at 3 s, 2 s long: its last frame
const joined = { frame: tl.frameOf(k3), time: v3.currentTime, paused: v3.paused, dur: tl.duration };
const k4 = { n: "e" };
tl.add(k4, { video: vid(), fps: 30, frames: 900 });               // a 30 s clip; at 20 s it leaves:
tl.seekTime(20);
tl.remove(k4);
const left = { dur: tl.duration, T: tl.time, n: tl.c.size };      // the clock follows the cameras in play
out("timeline", { seek, focus, play, end, again, removed, joined, left });
// a camera's tint: Live's own guesses in cause mode, its heads too
const sc = P.prepareScene(job.meta, null), zero = mousePose(0, 0), I = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
const d = T.parseTracks(Uint8Array.from(fs.readFileSync(process.argv[2] + "/tracks.bin")).buffer, job.meta);
const tint = Sp.hexRGB(Sp.TINTS[2]);
const tv = new T.TrackView(sc, d, job.rules, job.classes, { tint });
const before = tv.snapshot();
tv.perform(9, P.poseToCamera(sc, zero, zero, 1, 1), I, false); tv.perform(19, P.poseToCamera(sc, mousePose(20, 0), zero, 1, 1), I, false);
tv.setCause(true); tv.build(19);
out("tinted", { tint, guess: Array.from({ length: tv.guess.n }, (_, k) => [tv.guess.pos[6 * k], ...Array.from(tv.guess.col.slice(6 * k, 6 * k + 3))]),
                heads: Array.from({ length: tv.nHeads }, (_, k) => Array.from(tv.headCol.slice(3 * k, 3 * k + 3))) });
// passes: the drawing kept, saved and read back exactly; drawn again as the performed trails
const snap = tv.snapshot();
const buf = Ps.packPass(snap), back = Ps.unpackPass(buf, d.n, snap.upTo);
const same = (a, b) => a.length === b.length && Array.from(a).every((v, i) => v === b[i] || (v !== v && b[i] !== b[i]));
let bad = null;
try { Ps.unpackPass(buf.slice(1), d.n); } catch (e) { bad = e.message; }
const past = new T.Past();
past.setCause(true);
past.add(7, "cam", d, back, [10, 20, 0.5], tint);
const it = past.items[0];
const segs = (S, k) => Array.from({ length: S.n }, (_, i) => Array.from(S.pos.slice(6 * i, 6 * i + 6)).concat(Array.from(S.col.slice(6 * i, 6 * i + 3))));
const pastOut = { solid: it.solid.n, guess: it.guess.n, tvSolid: tv.solid.n, tvGuess: tv.guess.n,
                  sameSolid: JSON.stringify(segs(it.solid)) === JSON.stringify(segs(tv.solid)),
                  sameGuess: JSON.stringify(segs(it.guess)) === JSON.stringify(segs(tv.guess)),
                  at: it.obj.position.toArray(), mats: past.materials().length, segments: past.segments };
past.setCause(false);
pastOut.white = Array.from({ length: it.guess.n }, (_, i) => Array.from(it.guess.col.slice(6 * i, 6 * i + 3))).every((c) => c.every((v) => v === 1));
pastOut.veil = [past.items[0].ghosts.map((g) => g.visible), (past.setUnder("veil"), past.items[0].ghosts.map((g) => g.visible))];
past.add(8, "cam", d, back, [0, 0, 0], tint);
past.remove(7);
pastOut.after = [past.items.length, past.has(7), past.has(8)];
const it8 = past.items[0];
past.drop(it8); past.drop(it8);                                   // one drawing (a camera's layer); twice: nothing
pastOut.dropped = [past.items.length, past.has(8), past.group.children.length];
past.clear();
pastOut.cleared = past.items.length;
out("passes", { before, upTo: snap.upTo, bytes: buf.byteLength, n: d.n, X: same(back.X, snap.X), kind: same(back.kind, snap.kind),
                cause: same(back.cause, snap.cause), part: same(back.part, snap.part), bad, past: pastOut,
                drawn: [Ps.drawn(null), Ps.drawn(snap)], b64: Ps.toBase64(new Uint8Array([0, 1, 2, 250, 255]).buffer),
                iso: /^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}$/.test(Ps.localIso()),
                label: [Ps.passLabel({ id: "x", ended: "2026-10-28T19:14:02", cameras: ["a", "b", "c"], clock_s: 40.96, complete: true })
                          === "28 Oct 19:14 \\u00b7 3 cameras \\u00b7 0:41",
                        Ps.passLabel({ id: "y", started: "2026-10-28T09:05:00", cameras: ["a"], clock_s: 12.2, complete: false })
                          === "28 Oct 09:05 \\u00b7 1 camera \\u00b7 0:12 \\u00b7 stopped"] });
"""


@needs_node
def test_cameras_one_scene_one_clock_and_passes(tmp_path):
    """08 (b): cameras join the scene at their packages' origins (the start camera's is the
    scene's), each with a free tint of the violet (distinct, never orange), found by a URL word;
    one shared clock from 0 to the end of the longest clip: no looping, a shorter clip holds its
    last frame, at the end everything stops and play starts a new pass from 0; cameras join and
    leave the clock; the players' panels in a column that shrinks to fit; a camera's tint for
    Live's own guesses in cause mode; a pass's drawing saved and read back exactly and drawn again
    as the performed trails were."""
    (tmp_path / "tracks.bin").write_bytes(lexicon_tracks().tobytes())
    res = run_node(tmp_path, SPOTS, {"meta": flat_meta(yaw=0.0, tilt=3.0, frames=20), "rules": RULES,
                                     "classes": {"0": "thing", "1": "walker"}})
    tints = res["tints"]
    assert tints[0][0] == "#9b82d9"                                   # the first: the controller's violet
    assert len({t for t, _ in tints}) == 4 and all(230 <= h <= 300 for _, h in tints)
    hues = sorted(h for _, h in tints)
    assert min(b - a for a, b in zip(hues, hues[1:])) > 5             # told apart
    assert res["free"] == [0, 1, 1, 0, 1]                             # the first free; all taken: the least used
    assert res["offsets"] == [[30, -62, pytest.approx(-0.24)], [0, 0, 0], [0, 0, 0], [0, 0, 0], [10, 0, -4.5]]
    assert res["find"] == ["6363", "6354", "6363", "6354", None, None, None]
    c = res["clock"]
    assert c[0] == pytest.approx(5.87 - 1e-3) and c[1] == 3 and c[2] == 0 and c[3] == 0
    assert c[4] == 1 and c[5] == pytest.approx(0.17)
    r = res["rects"]
    assert r["none"] == []
    two = r["two"]
    assert len(two) == 2 and two[0]["w"] == 384 and two[0]["h"] == 216     # 0.24 W at 16:9
    assert two[0]["x"] + two[0]["w"] == 1600 - 248 - 12 and two[0]["y"] == 52
    assert two[1]["y"] == 52 + 216 + 8 and two[1]["x"] == two[0]["x"]
    many = r["many"]
    assert len(many) == 6 and many[-1]["y"] + many[-1]["h"] <= 600 - 56 + 1   # shrunk to fit
    assert all(abs(m["w"] / m["h"] - 16 / 9) < 0.05 for m in many)
    t = res["timeline"]
    sk = t["seek"]
    assert sk["T"] == 10 and sk["dur"] == pytest.approx(1228 / 30) and sk["max"] == str(round(1228 / 30 * 30))
    assert sk["frames"] == [175, 300, 299]                            # clamped per clip; the long one at 10 s
    assert sk["loops"] == [False, False, False] and sk["held"] == pytest.approx(176 / 30 - 1e-3)
    assert t["focus"] == [300, 1228]
    play = t["play"]
    assert play["T"] == pytest.approx(7, abs=0.05) and play["playing"] == [False, True, True]   # 5.9 s: held
    assert play["times"][0] == pytest.approx(175.5 / 30)              # on its last frame
    assert play["times"][1] == pytest.approx(7, abs=0.05) and play["times"][2] == pytest.approx(7, abs=0.05)
    end = t["end"]
    assert end["T"] == pytest.approx(1228 / 30) and end["playing"] is False and end["atEnd"] is True
    assert end["paused"] == [True, True, True] and end["events"][-1] == "end" and end["passes"] == 0
    assert end["frames"] == [175, 1227, 299]                          # every clip on its last frame
    again = t["again"]
    assert again["T"] == 0 and again["passes"] == 1 and again["playing"] == [True, True, True] and again["times"] == [0, 0, 0]
    assert t["removed"] == {"dur": pytest.approx(10), "max": "300", "n": 2}
    j = t["joined"]
    assert j["frame"] == 59 and j["paused"] is True and j["time"] == pytest.approx(59.5 / 30) and j["dur"] == pytest.approx(10)
    assert t["left"] == {"dur": pytest.approx(10), "T": pytest.approx(10), "n": 3}   # the clock: the active cameras'
    tn = res["tinted"]
    live = [g for g in tn["guess"] if g[0] < -5]                      # track 3: Live's guesses
    assert len(live) == 10 and all(g[1:] == pytest.approx(tn["tint"], abs=1e-6) for g in live)
    assert all(g[1:] == [1, 1, 1] for g in tn["guess"] if g[0] >= -5)
    assert sum(h == pytest.approx(tn["tint"], abs=1e-6) for h in tn["heads"]) == 1
    ps = res["passes"]
    assert ps["before"] is None and ps["upTo"] == 19                  # nothing drawn: no snapshot
    assert ps["bytes"] == 16 * ps["n"] and ps["X"] and ps["kind"] and ps["cause"] and ps["part"]
    assert "bytes" in ps["bad"]
    pa = ps["past"]
    assert pa["solid"] == pa["tvSolid"] > 0 and pa["guess"] == pa["tvGuess"] > 0
    assert pa["sameSolid"] and pa["sameGuess"]                        # the same lines, the same colours
    assert pa["at"] == [10, 20, 0.5] and pa["mats"] == 4 and pa["segments"] == pa["solid"] + pa["guess"]
    assert pa["white"] is True                                        # cause off: every guess white
    assert pa["veil"] == [[False, False], [True, True]]
    assert pa["after"] == [1, False, True] and pa["cleared"] == 0
    assert pa["dropped"] == [0, False, 0]
    assert ps["drawn"] == [False, True] and ps["b64"] == "AAEC+v8=" and ps["iso"]
    assert ps["label"] == [True, True]          # "28 Oct 19:14 \u00b7 3 cameras \u00b7 0:41", "... \u00b7 0:12 \u00b7 stopped"


@needs_node
def test_pose_link_dispatch_throttle_and_reconnect(tmp_path):
    """PoseLink with a fake socket: messages reach their handlers, the state goes out at once
    when something changes and at most 4 times a second when only the time moves, and a lost
    socket is reopened with a growing pause."""
    res = run_node(tmp_path, """
const socks = [];
class FakeWS { constructor(url) { this.url = url; this.readyState = 0; this.sent = []; socks.push(this); }
  send(s) { this.sent.push(JSON.parse(s)); } close() { this.readyState = 3; this.onclose && this.onclose(); }
  open() { this.readyState = 1; this.onopen(); } msg(o) { this.onmessage({ data: typeof o === "string" ? o : JSON.stringify(o) }); } }
const got = [], status = [];
const link = new M.PoseLink({ url: "ws://127.0.0.1:1/ws/view", WebSocketImpl: FakeWS,
  onPose: (m) => got.push(["pose", m.seq]), onZero: () => got.push(["zero"]), onControl: (a, v) => got.push(["control", a, v]),
  onHub: (m) => got.push(["hub", m.phones]), onStatus: (s) => status.push(s) });
const ws = socks[0];
link.sendState({ clip: "a/b", clips: ["a/b", "c/d"], playing: false, time: 0, duration: 9, recording: false,
                 gain: 1, scale: 1, src: "mouse", status: "ok", section: false, junk: 1 });
out("before_open", ws.sent.length);
ws.open();
out("on_open", ws.sent);
ws.msg({ type: "pose", v: 1, seq: 7, q: [1, 0, 0, 0] }); ws.msg({ type: "zero" });
ws.msg({ type: "control", action: "gain", value: 2 }); ws.msg({ type: "hub", phones: 1, views: 1 });
ws.msg("not json"); ws.msg({ type: "other" });
out("got", got);
const t0 = Date.now(); let n0 = ws.sent.length;
await new Promise((r) => { const iv = setInterval(() => { link.sendState({ time: (Date.now() - t0) / 1000 });
  if (Date.now() - t0 > 1000) { clearInterval(iv); r(); } }, 16); });
await new Promise((r) => setTimeout(r, 300));
out("time_sends", ws.sent.length - n0);
n0 = ws.sent.length;
out("same_again", link.sendState({ playing: false }));
link.sendState({ playing: true }); out("change_at_once", ws.sent.length - n0);
out("last", ws.sent[ws.sent.length - 1].playing);
const until = (k) => new Promise((r) => { const iv = setInterval(() => { if (socks.length >= k) { clearInterval(iv); r(); } }, 5); });
ws.close();
let t1 = Date.now(); await until(2); const d1 = Date.now() - t1;
socks[1].close();                                   // that attempt fails too
t1 = Date.now(); await until(3); const d2 = Date.now() - t1;
out("reopen_ms", [socks[1].url === ws.url, d1, d2]);
socks[2].open(); out("resent_on_reopen", socks[2].sent.length === 1 && socks[2].sent[0].clip === "a/b");
out("status", status);
link.close(); await new Promise((r) => setTimeout(r, 700)); out("after_close", socks.length);
""", {})
    assert res["before_open"] == 0
    st = res["on_open"]
    assert len(st) == 1 and st[0]["type"] == "state" and "junk" not in st[0]
    assert set(st[0]) == {"type", "clip", "clips", "playing", "time", "duration", "recording", "gain", "scale",
                          "src", "status", "section"}
    assert res["got"] == [["pose", 7], ["zero"], ["control", "gain", 2], ["hub", 1]]
    assert 3 <= res["time_sends"] <= 5
    assert res["same_again"] is False and res["change_at_once"] == 1 and res["last"] is True
    # closed: tried again after 0.5 s, then (that attempt failing) after 1 s
    ok, d1, d2 = res["reopen_ms"]
    assert ok and 400 <= d1 <= 900 and 900 <= d2 <= 1500
    assert res["resent_on_reopen"]
    assert res["status"][:3] == ["connecting", "connected", "off"] and res["status"][-1] == "connected"
    assert res["after_close"] == 3


# --------------------------------------------------------------------------- pose.js with the real hub
VIEW = """
const url = process.argv[3];
const job = JSON.parse((await import("node:fs")).readFileSync(process.argv[2] + "/job.json", "utf8"));
const inp = new M.PoseInput(P.prepareScene(job.meta, null));      // the clutch, as live.js holds it
const state = { clip: "site/a", clips: ["site/a", "site/b"], playing: false, time: 0, duration: 12,
                recording: false, gain: 1, scale: 1, src: "none", status: "ok", section: false,
                space_held: false, space_tilted: false };
let timer = null;
const set = (o) => { Object.assign(state, o); link.sendState(o); };     // as live.js: the state follows the view
const space = () => set({ space_held: inp.held, space_tilted: inp.tilted });
const link = new M.PoseLink({ url,
  onStatus: (s) => { out("status", s); if (s === "connected") link.sendState(state);
                     else inp.releaseWhere((w) => w.startsWith("phone")); },
  onPose: (m) => { out("pose", [m.seq, m.src, m.q]); inp.pose(m); inp.camera(); set({ src: m.src, status: m.status }); space(); },
  onZero: () => out("zero", 1),
  onHub: (m) => out("hub", [m.phones, m.views]),
  onControl: (a, v, player, spot) => {
    out("control", [a, v]); out("who", [a, player, spot]);
    if (a === "play") { set({ playing: true }); const t0 = Date.now();
      timer = setInterval(() => set({ time: (Date.now() - t0) / 1000 }), 16); }
    else if (a === "pause") { clearInterval(timer); set({ playing: false }); }
    else if (a === "gain") set({ gain: v });
    else if (a === "section") set({ section: !!v });
    else if (a === "record") set({ recording: false });                 // recording comes with job 05
    else if (a === "space") { if (v === true) inp.press(`phone ${player}`); else inp.release(`phone ${player}`); space(); }
    else if (a === "level") { inp.level(); setTimeout(() => { inp.camera(); space(); }, 450); }   // the glide: 0.3 s
  } });
process.stdin.on("data", () => { link.close(); process.exit(0); });
"""


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@needs_node
def test_pose_link_against_the_hub_with_a_fake_phone(app, tmp_path):
    pytest.importorskip("websockets")
    pytest.importorskip("cryptography")
    pytest.importorskip("qrcode")
    from websockets.sync.client import connect

    app.hub_options = {**app.hub_options, "port": free_port(), "view_port": free_port()}
    info = app.live_hub({"action": "start"})
    hub = app._hub
    (tmp_path / "hooks.mjs").write_text(HOOKS % json.dumps(THREE.as_uri() + "/"), encoding="utf-8")
    (tmp_path / "view.mjs").write_text(PRELUDE % json.dumps(LIVE.as_uri() + "/") + VIEW, encoding="utf-8")
    (tmp_path / "job.json").write_text(json.dumps({"meta": flat_meta(yaw=30.0, tilt=4.0)}), encoding="utf-8")
    proc = subprocess.Popen([NODE, "view.mjs", str(tmp_path), info["view_url"]], cwd=tmp_path, text=True,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    events = queue.Queue()
    threading.Thread(target=lambda: [events.put(json.loads(line)) for line in proc.stdout
                                     if line.startswith("[")], daemon=True).start()

    def wait(kind, pred=lambda v: True, timeout=8.0):
        end = time.time() + timeout
        while time.time() < end:
            try:
                k, v = events.get(timeout=max(0.01, end - time.time()))
            except queue.Empty:
                break
            if k == kind and pred(v):
                return v
        raise AssertionError(f"no {kind} from the view")

    tls = ssl.create_default_context(cafile=str(hub.cert))
    tls.maximum_version = ssl.TLSVersion.TLSv1_2                  # (a TLS 1.3 close can race the test)

    def phone():
        return connect(f"wss://127.0.0.1:{info['port']}/ws/phone?k={hub.token}", ssl=tls, open_timeout=5)

    def recv(ws, pred, timeout=4.0):
        end = time.time() + timeout
        while True:
            m = json.loads(ws.recv(timeout=max(0.01, end - time.time())))
            if m["type"] == "state" and pred(m):
                return m

    try:
        wait("status", lambda s: s == "connected")
        with phone() as ws:
            st = recv(ws, lambda m: "clips" in m)                   # the view's state, at once
            assert st["clip"] == "site/a" and st["clips"] == ["site/a", "site/b"] and st["recording"] is False
            wait("hub", lambda v: v == [1, 1])
            for i in range(1, 31):
                ws.send(json.dumps({"type": "pose", "v": 1, "src": "web", "seq": i, "t": i / 60,
                                    "q": [1, 0, 0, 0], "p": [0, 0, 0], "status": "ok"}))
            seqs = [wait("pose")[0] for _ in range(30)]
            assert seqs == list(range(1, 31))
            assert recv(ws, lambda m: m.get("src") == "web")
            ws.send(json.dumps({"type": "zero"}))
            wait("zero")
            ws.send(json.dumps({"type": "control", "action": "gain", "value": 2}))
            assert wait("control", lambda v: v[0] == "gain") == ["gain", 2]
            assert recv(ws, lambda m: m.get("gain") == 2)
            ws.send(json.dumps({"type": "control", "action": "play"}))
            recv(ws, lambda m: m.get("playing") is True)
            t0, n = time.time(), 0
            while time.time() - t0 < 1.5:
                try:
                    recv(ws, lambda m: m.get("time", 0) > 0, timeout=max(0.01, 1.5 - (time.time() - t0)))
                    n += 1
                except TimeoutError:
                    break
            assert 4 <= n <= 8, n                                   # the time: ~4 a second
            ws.send(json.dumps({"type": "control", "action": "pause"}))
            recv(ws, lambda m: m.get("playing") is False)
            ws.send(json.dumps({"type": "control", "action": "record", "value": True}))
            assert wait("control", lambda v: v[0] == "record") == ["record", True]
            ws.send(json.dumps({"type": "control", "action": "section", "value": True}))
            assert recv(ws, lambda m: m.get("section") is True)["recording"] is False
            # SPACE (08 a): held, the hand turns the space; let go; a double-tap levels
            ws.send(json.dumps({"type": "control", "action": "space", "value": True}))
            player = wait("who", lambda v: v[0] == "space")[1]
            assert isinstance(player, int) and player >= 1
            assert recv(ws, lambda m: m.get("space_held") is True)["space_tilted"] is False
            c, s = np.cos(np.radians(5)), np.sin(np.radians(5))      # the phone turned 10 degrees
            ws.send(json.dumps({"type": "pose", "v": 1, "src": "web", "seq": 31, "t": 0.6,
                                "q": [c, s, 0, 0], "p": [0, 0, 0], "status": "ok"}))
            assert recv(ws, lambda m: m.get("space_tilted") is True)["space_held"] is True
            ws.send(json.dumps({"type": "control", "action": "space", "value": False}))
            assert wait("who", lambda v: v[0] == "space") == ["space", player, None]   # one clip: no spot
            assert recv(ws, lambda m: m.get("space_held") is False)["space_tilted"] is True
            ws.send(json.dumps({"type": "control", "action": "level"}))
            assert wait("who", lambda v: v[0] == "level") == ["level", player, None]
            recv(ws, lambda m: m.get("space_tilted") is False)
            ws.send(json.dumps({"type": "control", "action": "space", "value": True}))
            recv(ws, lambda m: m.get("space_held") is True)
        # the phone left holding SPACE: the hub lets go for it
        assert wait("who", lambda v: v[0] == "space") == ["space", player, None]    # the press
        assert wait("control", lambda v: v[0] == "space") == ["space", False]      # and the let-go
        # the hub goes away and comes back on the same ports: the view reconnects by itself
        app.live_hub({"action": "stop"})
        wait("status", lambda s: s == "off")
        assert app.live_hub({"action": "start"})["running"]
        wait("status", lambda s: s == "connected", timeout=12)
        with phone() as ws:
            st = recv(ws, lambda m: True)
            assert st["gain"] == 2 and st["section"] is True and st["playing"] is False
            assert st["space_held"] is False
    finally:
        try:
            proc.stdin.write("quit\n")
            proc.stdin.flush()
            proc.wait(5)
        except (OSError, subprocess.TimeoutExpired):
            proc.kill()
    assert proc.returncode == 0, proc.stderr.read()
