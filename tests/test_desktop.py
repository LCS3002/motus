"""Desktop app (`groundtrack app`): the calibration and area editors' server side, the Houdini /
Rhino / Grasshopper links, and the same guards as the browser UI."""

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from groundtrack import desktop_cal as cal
from groundtrack import desktop_links as links
from groundtrack.desktop import DesktopApp, make_desktop_handler


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    """The synthetic demo site: a video, a GeoTIFF and known calibration pairs."""
    from groundtrack.demo import make_demo_site

    root = tmp_path_factory.mktemp("proj")
    site = make_demo_site(root)
    return root, site


@pytest.fixture()
def app(demo, tmp_path):
    root, _ = demo
    return DesktopApp(root, links.Settings(tmp_path / "desktop.json"))


def test_live_fit_matches_the_known_camera(demo, app):
    site = app.cal_site("sites/demo.yaml")
    st = cal.state(site)
    assert len(st["pairs"]) >= 6 and st["video"]["width"] == 1280
    assert st["map"]["epsg"] == 27700 and st["saved"] is None
    from groundtrack.homography import fit_homography

    p = st["pairs"][0]
    r = cal.fit(site, st["pairs"], query=[p["u"], p["v"]])
    # the CLI picker's fit on the same pairs (the demo clicks carry 0.7 px of noise)
    h = fit_homography([[q["u"], q["v"]] for q in st["pairs"]],
                       [[q["E"], q["N"]] for q in st["pairs"]], image_size=(1280, 720))
    assert r["ok"] and r["method"] == "homography" and r["rmse_m"] == pytest.approx(
        h.rmse_m, abs=1e-3) and r["rmse_m"] < 0.3
    assert len(r["points"]) == len(st["pairs"]) and all(q["proj"] for q in r["points"])
    assert r["grid"]["lines"] and len(r["footprint"]) >= 3
    assert r["guess"]["E"] == pytest.approx(r["points"][0]["proj"][0], abs=0.01)
    assert r["guess"]["N"] == pytest.approx(r["points"][0]["proj"][1], abs=0.01)
    few = cal.fit(site, st["pairs"][:3])                  # a plain homography needs 5
    assert not few["ok"] and few["need"] == 5


def test_saving_pairs_backs_up_and_calibrates_like_the_cli(demo, app):
    root, _ = demo
    site = app.cal_site("sites/demo.yaml")
    pairs = cal.state(site)["pairs"]
    with pytest.raises(ValueError):
        cal.save_pairs(site, pairs[:2])
    job = app.cal_save({"config": "sites/demo.yaml", "frame": 0, "pairs": pairs})
    t0 = time.time()
    while app.jobs[job["id"]].status == "running" and time.time() - t0 < 120:
        time.sleep(0.3)
    j = app.jobs[job["id"]]
    assert j.status == "done", "\n".join(j.lines[-20:])
    assert (root / "calibration" / "demo_homography.json").exists()
    assert list((root / "calibration" / "backup").glob("demo_homography_points.*.csv"))
    saved = cal.state(app.cal_site("sites/demo.yaml"))["saved"]
    live = cal.fit(app.cal_site("sites/demo.yaml"), pairs)
    assert saved["rmse_m"] == pytest.approx(live["rmse_m"], abs=1e-3)   # what was shown
    assert saved["n_points"] == len(pairs)


def test_area_is_saved_where_groundtrack_roi_puts_it(demo, app):
    root, _ = demo
    site = app.cal_site("sites/demo.yaml")
    with pytest.raises(ValueError):
        cal.save_roi(site, [[0, 0], [10, 0]])
    p = cal.save_roi(site, [[0, 0], [100.04, 0], [100, 80]])
    assert p == root / "sites" / "demo_roi.json"
    assert json.loads(p.read_text())[1] == [100.0, 0.0]
    assert cal.state(app.cal_site("sites/demo.yaml"))["roi"] == [[0, 0], [100, 0], [100, 80]]
    assert cal.clear_roi(site).parent.name == "backup" and not p.exists()


def test_kits_models_and_composites_are_found(tmp_path):
    lib = tmp_path / "lib"
    kit = lib / "houdini" / "plaza"
    (kit / "fields").mkdir(parents=True)
    (kit / "modules.json").write_text(json.dumps(
        {"site": "plaza", "modules": [{"name": "plaza_whole"}, {"name": "plaza1"}]}))
    (kit / "hou_pop_build.py").write_text("# builder")
    (kit / "plaza_pop.hipnc").write_bytes(b"x")
    gh = lib / "gh kits" / "_lacuna_"                     # Houdini Export from Grasshopper
    gh.mkdir(parents=True)
    for f in ("modules.json", "hou_pop_build.py", "_lacuna__field.geo"):
        (gh / f).write_text("{}")
    comp = lib / "square" / "composite"
    (comp / "houdini").mkdir(parents=True)
    (comp / "houdini" / "houdini_field.py").write_text("ORIGIN = (0, 0)")
    (comp / "houdini" / "vector_field_master.csv").write_text("x")
    run = lib / "square" / "clip1"                        # a run: not a combined plan
    (run / "houdini").mkdir(parents=True)
    (run / "houdini" / "houdini_field.py").write_text("ORIGIN = (0, 0)")
    (lib / "model.3dm").write_bytes(b"x")
    (lib / "def.gh").write_bytes(b"x")
    (lib / "images").mkdir()
    (lib / "images" / "hidden.3dm").write_bytes(b"x")     # output folders are not walked
    s = links.scan([lib])
    kits = {k["site"]: k for k in s["kits"]}
    assert kits["plaza"]["source"] == "motus" and kits["plaza"]["modules"] == [
        "plaza_whole", "plaza1"]
    assert kits["plaza"]["scene"].endswith("plaza_pop.hipnc")
    assert next(k for k in s["kits"] if k["name"] == "_lacuna_")["source"] == "Grasshopper"
    assert [c["path"] for c in s["composites"]] == [str(comp)]
    assert [m["name"] for m in s["models"]] == ["model.3dm"]
    assert [d["name"] for d in s["definitions"]] == ["def.gh"]
    assert s["scenes"] == []                              # the kit's scene is with the kit


def test_shell_line_matches_the_grasshopper_plugin():
    line = links.shell_line(r"C:\kits\plaza")
    assert line == ('import os; os.environ["GT_KIT"] = r"C:\\kits\\plaza"; '
                    'exec(open(r"C:\\kits\\plaza\\hou_pop_build.py").read())').replace("\\\\", "\\")


def test_plugin_version_from_the_binary(tmp_path):
    key = "ProductVersion\0".encode("utf-16-le")
    blob = b"MZ" + b"\0" * 30 + key
    blob += b"\0" * ((-len(blob)) % 4) + "0.5.2+abc\0".encode("utf-16-le") + b"\0" * 8
    (tmp_path / "x.gha").write_bytes(blob)
    assert links.pe_version(tmp_path / "x.gha") == "0.5.2"
    assert links.pe_version(tmp_path / "missing.gha") is None


def test_library_folders_are_kept_per_user(tmp_path):
    s = links.Settings(tmp_path / "s.json")
    with pytest.raises(ValueError):
        s.add_library(str(tmp_path / "nope"))
    s.add_library(str(tmp_path))
    assert links.Settings(tmp_path / "s.json").libraries == [str(tmp_path.resolve())]
    s.remove_library(str(tmp_path.resolve()))
    assert links.Settings(tmp_path / "s.json").libraries == []


def test_houdini_kits_are_guarded(app, tmp_path_factory):
    with pytest.raises(ValueError):
        app.houdini_build({"site": "../x", "composite": "", "out": ""})
    with pytest.raises(ValueError):
        app.houdini_build({"site": "demo", "composite": str(tmp_path_factory.mktemp("c"))})
    with pytest.raises(PermissionError):
        app._kit_out(str(Path(tmp_path_factory.mktemp("elsewhere")) / "kit"))
    with pytest.raises((PermissionError, FileNotFoundError, ValueError)):
        app.houdini_rebuild({"kit": str(tmp_path_factory.mktemp("notakit"))})
    with pytest.raises(PermissionError):                  # only files in allowed folders
        app.open_file({"path": str(tmp_path_factory.mktemp("o") / "a.3dm")})


def test_http_routes_and_guards(app):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), None)
    port = srv.server_address[1]
    srv.RequestHandlerClass = make_desktop_handler(app, port)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"

    def post(path, body, headers=None):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(), method="POST",
                                     headers=headers if headers is not None else
                                     {"X-Groundtrack": "1"})
        return json.loads(urllib.request.urlopen(req).read())

    try:
        with urllib.request.urlopen(base + "/") as r:
            assert b"motus" in r.read()
        with urllib.request.urlopen(base + "/static/app.css") as r:
            assert r.headers["Content-Type"].startswith("text/css")
        for bad in ("/static/../desktop.py", "/static/..%2Fdesktop.py"):
            with pytest.raises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(base + bad)
            assert e.value.code == 404
        with urllib.request.urlopen(base + "/api/app/overview") as r:
            assert [s["name"] for s in json.loads(r.read())["sites"]] == ["demo"]
        with urllib.request.urlopen(base + "/api/cal/frame?config=sites/demo.yaml&w=320") as r:
            assert r.headers["Content-Type"] == "image/jpeg" and r.read()[:2] == b"\xff\xd8"
        st = json.loads(urllib.request.urlopen(base + "/api/cal/state?config=sites/demo.yaml")
                        .read())
        assert post("/api/cal/fit", {"config": "sites/demo.yaml", "pairs": st["pairs"]})["ok"]
        with pytest.raises(urllib.error.HTTPError) as e:     # no custom header: refused
            post("/api/cal/fit", {"config": "sites/demo.yaml", "pairs": []}, headers={})
        assert e.value.code == 403
        req = urllib.request.Request(base + "/api/app/overview", headers={"Host": "evil.example"})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req)
        assert e.value.code == 403
        with pytest.raises(urllib.error.HTTPError) as e:     # only known configs
            post("/api/roi/save", {"config": "../x.yaml", "points": [[0, 0], [1, 0], [1, 1]]})
        assert e.value.code == 400
    finally:
        srv.shutdown()
        srv.server_close()


def _fake_sites(tmp_path, n=4):
    (tmp_path / "sites").mkdir()
    for i in range(n):
        (tmp_path / "sites" / f"s{i}.yaml").write_text(f"site: s{i}\n", encoding="utf-8")


def test_allowed_roots_refresh_once_for_concurrent_calls(tmp_path, monkeypatch):
    from groundtrack import ui

    _fake_sites(tmp_path)
    calls = []

    class Cfg:
        def __init__(self, f):
            self.f = f

        def path(self, key):
            return (tmp_path / f"out_{self.f.stem}") if key == "output_dir" else None

    def slow_load(f):
        calls.append(f.name)
        time.sleep(0.05)
        return Cfg(f)

    monkeypatch.setattr(ui, "_load", slow_load)
    app = DesktopApp(tmp_path, links.Settings(tmp_path / "desktop.json"))
    expect = app.allowed_roots()                  # first fill
    assert len(calls) == 4 and (tmp_path / "out_s0").resolve() in expect
    calls.clear()
    app._roots = (time.time() - 10, app._roots[1])  # expired
    out = []
    ts = [threading.Thread(target=lambda: out.append(app.allowed_roots())) for _ in range(20)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(calls) == 4                         # each config parsed once
    assert all(sorted(map(str, r)) == sorted(map(str, expect)) for r in out)


def test_allowed_roots_pick_up_a_changed_config(tmp_path, monkeypatch):
    _fake_sites(tmp_path, 1)
    app = DesktopApp(tmp_path, links.Settings(tmp_path / "desktop.json"))
    before = app.allowed_roots()
    (tmp_path / "sites" / "s0.yaml").write_text(
        "site: s0\noutput_dir: elsewhere/out\n", encoding="utf-8")
    assert app.allowed_roots() == before           # still cached
    app._roots = (time.time() - 10, app._roots[1])
    assert (tmp_path / "sites" / "elsewhere" / "out").resolve() in app.allowed_roots()
    # invalidation never serves the old roots
    app._roots = (0.0, [])
    assert (tmp_path / "sites" / "elsewhere" / "out").resolve() in app.allowed_roots()


def test_webview_busy_error_is_a_clean_message(tmp_path, monkeypatch, capsys):
    import sys
    import types

    from groundtrack import desktop

    def fake_start(*a, **k):
        raise RuntimeError("COM error HRESULT 0x800700AA")

    fake = types.SimpleNamespace(
        create_window=lambda *a, **k: types.SimpleNamespace(
            events=types.SimpleNamespace(closing=[], loaded=[], maximized=[], restored=[]),
            show=lambda: None),
        start=fake_start)

    class Ev(list):
        def __iadd__(self, f):
            return self

    fake.create_window = lambda *a, **k: types.SimpleNamespace(
        events=types.SimpleNamespace(closing=Ev(), loaded=Ev(), maximized=Ev(), restored=Ev()),
        show=lambda: None)
    monkeypatch.setitem(sys.modules, "webview", fake)
    monkeypatch.setattr(desktop.DesktopApp, "check_device", lambda self: None)
    desktop.serve(tmp_path, port=0, window=True)   # returns, no traceback
    err = capsys.readouterr().err
    assert "another motus window is still open" in err and "0x800700AA" in err

    fake.start = lambda *a, **k: (_ for _ in ()).throw(OSError(-2147024726, "busy"))
    desktop.serve(tmp_path, port=0, window=True)

    fake.start = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("other"))
    with pytest.raises(RuntimeError, match="other"):
        desktop.serve(tmp_path, port=0, window=True)
