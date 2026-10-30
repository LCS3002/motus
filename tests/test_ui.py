"""Local web UI: the server only talks to this machine and only acts inside the project."""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest
import yaml

from groundtrack.locate import camera_from_place, write_camera_position
from groundtrack.templates import TEMPLATES
from groundtrack.ui import App, make_handler


@pytest.fixture()
def project(tmp_path):
    (tmp_path / "sites").mkdir()
    (tmp_path / "sites" / "plaza9.yaml").write_text(
        TEMPLATES["plaza"].format(site="plaza9", fname="sites/plaza9.yaml"), encoding="utf-8")
    return tmp_path


def test_sites_and_detail(project):
    app = App(project)
    s = app.sites()
    assert [x["name"] for x in s] == ["plaza9"]
    assert s[0]["video_ok"] is False and s[0]["calibrated"] is False
    d = app.site("sites/plaza9.yaml")
    assert d["runs"] == [] and d["calibration"]["ok"] is False


def test_new_site_and_paths(project):
    app = App(project)
    with pytest.raises(ValueError):
        app.new_site({"site": "../evil", "template": "plaza"})
    r = app.new_site({"site": "road1", "template": "motorway", "video": r"C:\clips\road1.mov"})
    cfg = yaml.safe_load((project / r["config"]).read_text(encoding="utf-8"))
    assert cfg["site"] == "road1" and cfg["video"] == "C:/clips/road1.mov"
    with pytest.raises(ValueError):
        app.new_site({"site": "road1", "template": "motorway"})        # no overwriting


def test_only_known_configs_and_project_files(project, tmp_path_factory):
    app = App(project)
    with pytest.raises(ValueError):
        app.resolve_config("../outside.yaml")
    outside = tmp_path_factory.mktemp("elsewhere") / "secret.txt"
    outside.write_text("x")
    with pytest.raises(PermissionError):
        app.check_file(str(outside))
    with pytest.raises(ValueError):
        app.start_job({"kind": "rm", "config": "sites/plaza9.yaml"})
    with pytest.raises(ValueError):                                     # run names are checked
        app.start_job({"kind": "package", "config": "sites/plaza9.yaml", "run": "../../x"})


def test_camera_position_written_into_config(project):
    app = App(project)
    hit = {"E": 537548.3, "N": 180245.5, "size_m": 40.0, "name": "somewhere"}
    cam = camera_from_place(hit, floor=10)
    assert cam["height_m"] == pytest.approx(10 * 3.1 + 1.5)
    assert cam["position_tol_m"] == 20.0
    app.set_camera({"config": "sites/plaza9.yaml", **cam, "hfov_deg": 40})
    app.set_camera({"config": "sites/plaza9.yaml", "E": 1, "N": 2, "height_m": 3})
    cfg = yaml.safe_load((project / "sites" / "plaza9.yaml").read_text(encoding="utf-8"))
    assert cfg["camera_position"]["E"] == 1 and cfg["camera_position"]["hfov_deg"] == 40
    assert cfg["detection"]["model"]                                   # the rest is untouched
    text = (project / "sites" / "plaza9.yaml").read_text(encoding="utf-8")
    assert text.count("camera_position:") == 1


def test_adopt_automatic_calibration_keeps_a_backup(project):
    import numpy as np

    from groundtrack.homography import Homography

    app = App(project)
    cal = project / "calibration"
    cal.mkdir()
    mine = Homography(H=np.eye(3), origin=(0.0, 0.0), image_size=(10, 10))
    auto = Homography(H=np.diag([2.0, 2.0, 1.0]), origin=(0.0, 0.0), image_size=(10, 10))
    mine.save(cal / "plaza9_homography.json")
    auto.save(cal / "plaza9_homography_auto.json")
    assert "auto_candidate" in app.site("sites/plaza9.yaml")["calibration"]
    app.adopt_auto({"config": "sites/plaza9.yaml"})
    assert Homography.load(cal / "plaza9_homography.json").H[0, 0] == 2.0
    assert Homography.load(cal / "plaza9_homography.before_auto.json").H[0, 0] == 1.0
    assert not (cal / "plaza9_homography_auto.json").exists()
    assert "auto_candidate" not in app.site("sites/plaza9.yaml")["calibration"]


def test_write_camera_position_keeps_following_keys(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("site: a\ncamera_position:\n  E: 1\n  N: 2\nprojection:\n  max_range_m: 30\n")
    write_camera_position(p, {"E": 5, "N": 6, "height_m": 7})
    cfg = yaml.safe_load(p.read_text())
    assert cfg["camera_position"] == {"E": 5, "N": 6, "height_m": 7}
    assert cfg["projection"]["max_range_m"] == 30


def test_http_guards(project):
    app = App(project)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), None)
    port = srv.server_address[1]
    srv.RequestHandlerClass = make_handler(app, port)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        with urllib.request.urlopen(base + "/api/sites") as r:
            assert json.loads(r.read())["sites"][0]["name"] == "plaza9"
        # another website pointing a hostname at this port (DNS rebinding) is refused
        req = urllib.request.Request(base + "/api/sites", headers={"Host": "evil.example"})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req)
        assert e.value.code == 403
        # a plain cross-site form POST (no custom header) cannot start jobs
        req = urllib.request.Request(base + "/api/job", data=b'{"kind": "device"}',
                                     method="POST")
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req)
        assert e.value.code == 403
    finally:
        srv.shutdown()
        srv.server_close()


def test_json_sends_infinity_and_nan_as_null(project, monkeypatch):
    """The browser's JSON.parse refuses Infinity / NaN (e.g. the RMSE of an exact 5-pair
    homography): they go out as null and the rest is unchanged."""
    app = App(project)
    real = app.sites
    monkeypatch.setattr(app, "sites", lambda: [
        {**s, "rmse_m": float("inf"), "errs": [float("nan"), 1.5], "t": (float("-inf"), 2)} for s in real()])
    srv = ThreadingHTTPServer(("127.0.0.1", 0), None)
    port = srv.server_address[1]
    srv.RequestHandlerClass = make_handler(app, port)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    def strict(c):
        raise ValueError(f"{c} is not JSON")

    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/sites") as r:
            s = json.loads(r.read(), parse_constant=strict)["sites"][0]
        assert s["rmse_m"] is None and s["errs"] == [None, 1.5] and s["t"] == [None, 2]
        assert s["name"] == "plaza9"
    finally:
        srv.shutdown()
        srv.server_close()


def test_combined_plans_are_listed_and_combining_is_guarded(project):
    app = App(project)
    assert app.composites() == []
    c = project / "runs" / "composites" / "square"
    for sub in ("images", "labels", "rhino"):
        (c / sub).mkdir(parents=True)
    (c / "images" / "plan_tracks.png").write_bytes(b"x")
    (c / "images" / "plan_tracks_layer.png").write_bytes(b"x")
    (c / "labels" / "title.txt").write_text("Square\nall clips\n", encoding="utf-8")
    (c / "rhino" / "composite.3dm").write_bytes(b"x")
    assert app.composites() == [{"name": "square", "images": 2}]
    d = app.composite_detail("square")
    assert d["title"][0] == "Square" and len(d["images"]) == 1 and len(d["layers"]) == 1
    assert len(d["rhino"]) == 1
    with pytest.raises(ValueError):
        app.composite_detail("../square")
    with pytest.raises(ValueError):                       # a name that would leave the folder
        app.start_job({"kind": "composite", "configs": ["sites/plaza9.yaml"],
                       "options": {"name": "../evil"}})
    with pytest.raises(ValueError):                       # only known site configs
        app.start_job({"kind": "composite", "configs": ["../other.yaml"],
                       "options": {"name": "ok"}})
