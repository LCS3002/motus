"""Live's point cloud (plans/live_controller/11, option A; groundtrack/live_cloud.py, live/cloud.js).
A separate function, off by default: the tests also prove that nothing else loads it."""
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import test_live_view as V
from groundtrack import live_cloud as LC

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "groundtrack"
LIVE = PKG / "desktop_web" / "live"


def _text(p):
    return p.read_text(encoding="utf-8").replace("\r\n", "\n")


# --------------------------------------------------------------------------- the maths
def test_the_ground_fit_finds_the_affine_map_past_occluders():
    rng = np.random.default_rng(3)
    y = 1 / rng.uniform(3, 25, 4000)                         # the ground's true inverse depth
    x = (y - 0.004) / 2.5                                    # the model's: an affine map of it
    occ = rng.random(4000) < 0.55                            # more than half hidden: nearer things
    y_seen = np.where(occ, y * rng.uniform(1.3, 4, 4000), y)
    a, b, share = LC.fit_ground(x, y_seen)
    assert a == pytest.approx(2.5, rel=1e-3) and b == pytest.approx(0.004, abs=1e-5)
    assert share == pytest.approx(0.45, abs=0.03)
    a0, b0, s0 = LC.fit_ground(x[:10], y[:10])               # too few pixels
    assert math.isnan(a0) and math.isnan(b0) and s0 == 0.0
    assert LC.fit_ground(-x, y)[2] == 0.0                     # nearer must stay nearer: no fit


def test_ground_depth_on_flat_and_sloped_terrain():
    K = np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]])
    rays = LC.grid_rays(K, (640, 480), step=8)
    t = math.radians(20)                                     # looking north, 20 deg down
    fwd = np.array([0, math.cos(t), -math.sin(t)]); right = np.array([1.0, 0, 0])
    R_wc = np.column_stack([right, np.cross(fwd, right), fwd])
    C = np.array([0, 0, 2.0])
    dirs = rays @ R_wc.T
    d = LC.ground_depth(dirs, C, lambda x, y: np.zeros_like(x))
    X = C + d[..., None] * dirs
    ok = np.isfinite(d)
    assert ok.mean() > 0.5 and np.abs(X[ok][:, 2]).max() < 1e-6     # every ray that comes down lands on z = 0
    assert np.isnan(d[0]).all()                              # the top rows look over the horizon
    slope = LC.Terrain(np.tile((np.arange(40) * 0.05)[::-1, None], (1, 40)), -20, 40)   # 5 %, rising northwards
    d2 = LC.ground_depth(dirs, C, slope)
    X2 = C + d2[..., None] * dirs
    ok2 = np.isfinite(d2) & (dirs[..., 2] < -0.15)           # (grazing rays converge slower: beyond the fit's reach)
    assert ok2.mean() > 0.3
    assert np.abs(X2[ok2][:, 2] - slope(X2[ok2][:, 0], X2[ok2][:, 1])).max() < 0.02


def test_voxels_merge_by_mean_and_count():
    v = LC.VoxelCloud(0.5)
    v.add(np.array([[0.1, 0.1, 0.1], [0.3, 0.2, 0.4], [5.0, 5.0, 5.0]]), np.array([[0, 0, 0], [100, 50, 20], [9, 9, 9]]))
    v.add(np.array([[0.2, 0.3, 0.1], [-0.2, -0.2, -0.2]]), np.array([[200, 100, 40], [1, 2, 3]]))
    P, rgb, n = v.points()
    by = {tuple(np.floor(p / 0.5).astype(int)): (p, c, k) for p, c, k in zip(P, rgb, n)}
    p, c, k = by[(0, 0, 0)]
    assert k == 3 and p == pytest.approx([0.2, 0.2, 0.2]) and c.tolist() == [100, 50, 20]
    assert by[(-1, -1, -1)][2] == 1 and by[(10, 10, 10)][2] == 1 and len(v) == 3


def test_the_files_round_trip_and_the_ply_reads(tmp_path):
    P = np.array([[1.5, -2.0, 0.25], [10.0, 20.0, 3.0]]); rgb = np.array([[10, 20, 30], [200, 210, 220]], np.uint8)
    meta = LC.write_cloud(tmp_path, P, rgb, np.array([3, 400]), {"voxel_m": 0.1, "source": "test"})
    assert meta["n"] == 2 and meta["record_bytes"] == 16 == LC.CLOUD_DTYPE.itemsize
    m, rec = LC.read_cloud(tmp_path)
    assert m == meta and rec["x"].tolist() == [1.5, 10.0] and rec["n"].tolist() == [3, 255]
    assert [rec["r"][1], rec["g"][1], rec["b"][1]] == [200, 210, 220]
    raw = (tmp_path / "cloud.ply").read_bytes()
    head, body = raw.split(b"end_header\n")
    assert b"element vertex 2" in head and len(body) == 2 * 15
    assert not list(tmp_path.glob("*.part"))
    LC.remove_cloud(tmp_path)
    assert not list(tmp_path.glob("cloud.*"))


# --------------------------------------------------------------------------- a site camera
def _package(tmp, people=()):
    """A still camera 2 m up looking north 12 deg down over flat ground, a wall across at y = 18 m,
    its video (a few frames), people standing (x, y) as tracks."""
    W, H, f = 320, 180, 260.0
    t = math.radians(12)
    fwd = np.array([0, math.cos(t), -math.sin(t)]); right = np.array([1.0, 0, 0])
    R_wc = np.column_stack([right, np.cross(fwd, right), fwd])
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    live = tmp / "run" / "live"
    live.mkdir(parents=True)
    meta = {"version": 1, "clip": "test", "run": "v1", "image_size": [W, H], "frames": 40, "fps": 10.0, "R_max": 30.0,
            "camera": {"K": K.tolist(), "R_cw": R_wc.T.tolist(), "C": [0.0, 0.0, 2.0]},
            "terrain": {"file": "terrain.f32", "nx": 80, "ny": 80, "x0": -40.0, "y1": 40.0, "cell": 1.0},
            "tracks": {"file": "tracks.bin"}, "video": {"file": "video.mp4", "blurred": True, "stabilized": True}}
    (live / "scene.json").write_text(json.dumps(meta))
    np.zeros((80, 80), "<f4").tofile(live / "terrain.f32")
    tr = np.zeros(len(people) * 40, LC.TRACK_DTYPE)
    for i, (x, y) in enumerate(people):
        s = tr[i * 40:(i + 1) * 40]
        s["id"], s["frame"], s["x"], s["y"], s["z"] = i + 1, np.arange(40), x, y, 0.0
    tr.tofile(live / "tracks.bin")
    vw = cv2.VideoWriter(str(live / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (W, H))
    for k in range(40):
        img = np.full((H, W, 3), 90, np.uint8)
        img[: H // 3] = (200, 160, 120)
        vw.write(img)
    vw.release()
    return live, K, R_wc, np.array([0.0, 0.0, 2.0])


class FakeDepth:
    """The scene's true inverse depth (ground; the wall at y = 18), as an affine map: like the model."""
    name = "fake"

    def __init__(self, K, R_wc, C, a=0.4, b=0.02):
        self.K, self.R_wc, self.C, self.a, self.b, self.calls = K, R_wc, C, a, b, 0

    def __call__(self, rgb, hw):
        self.calls += 1
        rays = LC.grid_rays(self.K, (rgb.shape[1], rgb.shape[0]), step=LC.STEP)
        assert rays.shape[:2] == tuple(hw)
        dirs = rays @ self.R_wc.T
        dg = LC.ground_depth(dirs, self.C, lambda x, y: np.zeros_like(x))
        dw = np.where(dirs[..., 1] > 1e-6, (18.0 - self.C[1]) / np.maximum(dirs[..., 1], 1e-6), np.inf)
        d = np.fmin(np.nan_to_num(dg, nan=np.inf), dw)
        inv = np.where(np.isfinite(d), 1 / d, 0.0)
        return ((inv - self.b) / self.a).astype(np.float32)


def test_a_site_cloud_lands_on_its_ground_and_wall_and_leaves_people_out(tmp_path):
    live, K, R_wc, C = _package(tmp_path, people=[(1.0, 8.0)])
    fake = FakeDepth(K, R_wc, C)
    meta = LC.make_site_cloud(live.parent, frames=6, voxel=0.1, depth=fake, log=lambda *_: None)
    assert fake.calls == 6 and meta["frames"] == 6 and meta["source"] == "site" and meta["model"] == "fake"
    assert all(s > 0.9 for s in meta["fit"]["inliers"])      # the fake is exact: the ground fits
    _, rec = LC.read_cloud(live)
    P = np.c_[rec["x"], rec["y"], rec["z"]]
    ground, wall = P[P[:, 1] < 17.5], P[P[:, 1] >= 17.5]
    assert len(ground) > 500 and np.abs(ground[:, 2]).max() < 0.08
    # within 2 %: the wall's foot passes for the ground behind it (inside the fit's tolerance), ~1 % off
    assert len(wall) > 100 and np.abs(wall[:, 1] - 18).max() < 0.02 * 18
    near = (np.abs(P[:, 0] - 1.0) < 0.35) & (np.abs(P[:, 1] - 8.0) < 0.35)
    assert not near.any()                                    # the tracked person's ground is not in it
    assert np.linalg.norm(P[:, :2], axis=1).max() < 30 + LC.MARGIN_M


def test_a_site_cloud_refuses_realtime_and_unblurred_packages(tmp_path):
    live, K, R_wc, C = _package(tmp_path)
    m = json.loads((live / "scene.json").read_text())
    (live / "scene.json").write_text(json.dumps({**m, "realtime": True}))
    with pytest.raises(ValueError, match="Realtime"):
        LC.make_site_cloud(live.parent, depth=FakeDepth(K, R_wc, C), log=lambda *_: None)
    (live / "scene.json").write_text(json.dumps({**m, "video": {"file": "video.mp4", "blurred": False}}))
    with pytest.raises(ValueError, match="not blurred"):
        LC.make_site_cloud(live.parent, depth=FakeDepth(K, R_wc, C), log=lambda *_: None)
    assert not (live / "cloud.json").exists()


# --------------------------------------------------------------------------- the switch: off by default
def test_nothing_loads_the_cloud_unless_it_is_asked_for():
    """Off: no module imports live_cloud at load (only inside the CLI's command and Realtime's
    switch), and live_cloud itself loads neither torch nor transformers until a cloud is made."""
    imports = ("from .live_cloud", "import live_cloud", "groundtrack.live_cloud", "from . import live_cloud")
    users = sorted(p.name for p in PKG.glob("*.py") if p.name != "live_cloud.py"
                   and any(i in p.read_text(encoding="utf-8") for i in imports))
    assert users == ["cli.py", "live_track.py"]
    cli, lt = _text(PKG / "cli.py"), _text(PKG / "live_track.py")
    assert cli.count("from .live_cloud import") == 1 == lt.count("from .live_cloud import")
    assert "from .live_cloud import" in cli.split("def cmd_live_cloud(args):")[1].split("\ndef ")[0]
    assert "from .live_cloud import" in lt.split("    def set_cloud(self, on) -> None:")[1].split("\n    def ")[0]
    py = lambda code: subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=180)
    r = py("import sys, groundtrack.live_cloud; print('torch' in sys.modules, 'transformers' in sys.modules)")
    assert r.returncode == 0 and r.stdout.split() == ["False", "False"], r.stderr
    r = py("import sys, groundtrack.cli, groundtrack.desktop, groundtrack.live_hub, groundtrack.live_track; "
           "print('groundtrack.live_cloud' in sys.modules, 'transformers' in sys.modules)")
    assert r.returncode == 0 and r.stdout.split() == ["False", "False"], r.stderr


def test_live_shows_the_cloud_only_while_its_switch_is_on():
    js, html = _text(LIVE / "live.js"), _text(LIVE / "live.html")
    assert 'cloud: params.get("cloud") === "1" }' in js                # off unless asked
    assert '<input type="checkbox" data-layer="cloud"> Point cloud' in html   # not checked
    sc = js.split("function syncCloud(s) {")[1].split("\n}\n")[0]
    assert sc.lstrip().startswith("if (!S.layers.cloud) { dropCloud(s); return; }")
    # its code too: loaded only once on (here, and in Realtime's poll, which runs only while it is on)
    assert js.count('"./cloud.js"') == js.count('await import("./cloud.js")') == 2
    assert 'await import("./cloud.js")' in sc
    assert "  if (S.layers.cloud) { if (L) rtCloud(); syncCloudPanel(); }" in js
    rc = js.split("async function rtCloud() {")[1].split("\n}\n")[0]
    assert 'c.state !== "live"' in rc and 'await import("./cloud.js")' in rc and "/api/live/realtime/cloud" in rc
    assert js.count("/api/live/realtime/cloud") == 1
    assert 'cloud: rtCloudWanted() });' in js and '=== "1"; } catch { return false; } };' in js   # off unless kept on
    assert js.count('pkg(s.run, "cloud.') == 2 and 'pkg(s.run, "cloud.json")' in sc and 'pkg(s.run, "cloud.bin")' in sc
    assert "/api/live/cloud" in js.split("async function makeClouds() {")[1].split("\n}\n")[0]
    assert js.count("/api/live/cloud") == 1
    assert "dropCloud(s);\n  scene3.remove(s.group);" in js                   # a camera leaving takes its cloud
    assert "L.section.register(view.material);" in sc                         # the section cuts it too
    assert 'id="cloud-row" hidden' in html


def test_the_app_serves_the_cloud_and_makes_it_as_a_job():
    d = _text(PKG / "desktop.py")
    assert '"cloud.json", "cloud.bin"}' in d and '"/api/live/cloud": app.live_cloud,' in d
    body = d.split("def live_cloud(self, body: dict) -> dict:")[1].split("\n    def ")[0]
    assert 'args = ["live-cloud", str(live.parent)]' in body and 'if meta.get("realtime"):' in body
    # Realtime: the switch (start takes it too) and the view's feed while it records
    rt = d.split("    def live_realtime(self, body: dict) -> dict:")[1].split("\n    def ")[0]
    assert 'self._realtime.set_cloud(body["cloud"] is True)' in rt and 'self._realtime.set_cloud(body.get("on") is True)' in rt
    assert 'elif u.path == "/api/live/realtime/cloud":' in d and "rt.cloud_since(rid, v0)" in d
    rh, rj = _text(LIVE / "realtime.html"), _text(LIVE / "realtime.js")
    assert '<button data-on="0" role="tab" aria-selected="true">Off</button>' in rh
    assert 'localStorage.getItem(CLOUD_KEY) === "1"; } catch { return false; } })();' in rj
    assert '{ action: "start", long_side: longSide, cloud: cloudOn }' in rj
    from groundtrack.cli import build_parser
    a = build_parser().parse_args(["live-cloud", "runs/x/v1", "--frames", "12", "--remove"])
    assert (a.frames, a.voxel, a.remove, a.func.__name__) == (12, 0.10, True, "cmd_live_cloud")


@V.needs_node
def test_the_view_reads_cloud_bin(tmp_path):
    rec = np.zeros(2, LC.CLOUD_DTYPE)
    rec["x"], rec["y"], rec["z"] = [1.5, -3.25], [2.0, 40.0], [0.5, 12.0]
    rec["r"], rec["g"], rec["b"], rec["n"] = [10, 250], [20, 240], [30, 230], [1, 9]
    (tmp_path / "cloud.bin").write_bytes(rec.tobytes())
    res = V.run_node(tmp_path, """
const Cl = await import(LIVE + "cloud.js");
const b = (await import("node:fs")).readFileSync(process.argv[2] + "/cloud.bin");
const p = Cl.parseCloud(Uint8Array.from(b).buffer);
out("n", p.n); out("pos", Array.from(p.positions)); out("col", Array.from(p.colors));
const v = new Cl.CloudView(p, "grey");
out("grey", Array.from(v.points.geometry.getAttribute("color").array));
v.setMode("colour"); out("colour", Array.from(v.points.geometry.getAttribute("color").array));
v.setLook("projector"); out("size", [v.material.size, v.material.opacity]);
let bad = ""; try { Cl.parseCloud(new ArrayBuffer(17)); } catch (e) { bad = e.message; } out("bad", bad);
""", {})
    assert res["n"] == 2 and res["pos"] == [1.5, 2.0, 0.5, -3.25, 40.0, 12.0]
    assert res["col"] == [10, 20, 30, 250, 240, 230] and res["colour"] == res["col"]
    assert res["grey"][:3] == [129, 129, 129] and len(set(res["grey"][3:])) == 1 and res["grey"][3] > 230
    assert res["size"] == [0.13, 0.8] and "whole records" in res["bad"]


# --------------------------------------------------------------------------- Realtime (step 2): a rough cloud while it records
def test_voxels_merged_frame_by_frame_are_the_voxels_of_all_at_once():
    rng = np.random.default_rng(5)
    P = rng.uniform(-3, 3, (6000, 3)); rgb = rng.integers(0, 255, (6000, 3))
    one, many = LC.VoxelCloud(0.5), LC.VoxelCloud(0.5)
    one.add(P, rgb)
    for i in range(0, 6000, 700):
        many.add(P[i:i + 700], rgb[i:i + 700])
    a, b = one.points(), many.points()
    assert len(one) == len(many) and np.allclose(a[0], b[0]) and (a[1] == b[1]).all() and (a[2] == b[2]).all()
    assert np.all(np.diff(many.keys) > 0)


def test_the_box_mask_covers_each_box_grown():
    m = LC.box_mask(np.array([[100, 40, 140, 120], [-50, -50, 10, 10]]), (90, 160), step=2, pad=0.25)
    rows, cols = np.nonzero(m[:, 20:])                       # the first box: 90-150 x 20-140 px grown
    assert (cols.min() + 20, cols.max() + 20, rows.min(), rows.max()) == (45, 74, 10, 69)
    assert m[:13, :13].all() and not m[13:, 13:40].any() and not m[13:, :13].any()   # the second: clipped


def _rec(heading=123.0, tilt=20.0, nudge=0.0, size=(640, 360)):
    from groundtrack.live_track import sensor_camera
    from test_live_track import phone_pose, to_quat

    qs = [to_quat(phone_pose(heading, tilt, 0))] * 30
    h, K, R_cw, C = sensor_camera(qs, size, 1.5, tilt_offset=nudge)
    logs = []
    return SimpleNamespace(h=h, K=K, R_cw=R_cw, C=C, cal_qs=qs, tilt_offset=nudge, size=size, log=logs.append,
                           logs=logs)


class GroundDepth:
    """The flat ground's inverse depth for a camera turned R_wc (an affine map of it, as the model)."""
    name = "fake-ground"

    def __init__(self, rec, R_wc, a=0.3, b=0.01):
        self.rec, self.R, self.a, self.b, self.calls = rec, R_wc, a, b, 0

    def __call__(self, rgb, hw):
        self.calls += 1
        step = max(1, round(max(self.rec.size) / LC.RT_GRID))
        dirs = LC.grid_rays(self.rec.K, self.rec.size, step) @ self.R.T
        assert dirs.shape[:2] == tuple(hw)
        d = LC.ground_depth(dirs, self.rec.C, lambda x, y: np.zeros_like(x))
        return ((np.where(np.isfinite(d), 1 / d, 0.0) - self.b) / self.a).astype(np.float32)


@pytest.mark.parametrize("nudge", [0.0, 5.0])
def test_the_phones_turn_since_the_calibration_turns_the_cloud(nudge):
    from groundtrack.live_track import level_camera
    from test_live_track import phone_pose, rot_z, to_quat

    rec = _rec(nudge=nudge)
    b = LC.CloudBuilder(SimpleNamespace(state="ready"), rec)
    assert np.allclose(b.camera(None), rec.R_cw.T)                          # no pose: the calibrated
    assert np.allclose(b.camera(to_quat(phone_pose(123, 20, 0))), rec.R_cw.T, atol=1e-6)
    turned = b.camera(to_quat(phone_pose(123 + 40, 20 - 8, 0)))             # 40 right, 8 up from the start
    assert np.allclose(turned, rot_z(-40) @ level_camera(20 - 8 + nudge), atol=1e-6)


def test_a_depth_frame_lands_on_the_ground_where_the_phone_looked_without_the_boxes():
    from test_live_track import phone_pose, to_quat

    rec = _rec()
    b = LC.CloudBuilder(SimpleNamespace(state="ready", error=None), rec)
    R = b.camera(to_quat(phone_pose(123 + 60, 25, 0)))                      # turned 60 deg right
    img = np.full((360, 640, 3), (40, 90, 160), np.uint8)
    b.add(GroundDepth(rec, R), 7, img, np.zeros((0, 4)), R)
    _, data = b.snapshot()
    rec0 = np.frombuffer(data, LC.CLOUD_DTYPE)
    P = np.c_[rec0["x"], rec0["y"], rec0["z"]]
    assert len(P) > 1000 and np.abs(P[:, 2]).max() < 0.03                   # on the flat ground
    bearing = np.degrees(np.arctan2(P[:, 0], P[:, 1]))
    assert 50 < np.median(bearing) < 70 and np.abs(bearing - 60).max() < 45  # where it looked
    assert [rec0["r"][0], rec0["g"][0], rec0["b"][0]] == [160, 90, 40]      # (BGR in, RGB kept)
    head, same = b.snapshot(head0 := b.version)
    assert head["same"] and same == b"" and head["n"] == len(P) and head0 == 1
    # a box (anyone detected) in the middle of the next frame: its ground stays out
    b2 = LC.CloudBuilder(SimpleNamespace(state="ready", error=None), rec)
    b2.add(GroundDepth(rec, R), 8, img, np.array([[280, 200, 360, 330]]), R)
    P2 = np.frombuffer(b2.snapshot()[1], LC.CLOUD_DTYPE)
    x = (rec.K @ R.T @ (np.c_[P2["x"], P2["y"], P2["z"]] - rec.C).T)
    u, v = x[0] / x[2], x[1] / x[2]
    assert len(P2) < len(P) and not ((u > 284) & (u < 356) & (v > 204) & (v < 326)).any()
    assert b2.fits[0] > 0.9 and b2.model == "fake-ground"


def test_frames_are_offered_now_and_then_once_the_phone_turned():
    from test_live_track import phone_pose, to_quat

    rec = _rec()
    got = []
    w = SimpleNamespace(state="ready", offer=got.append)
    b = LC.CloudBuilder(w, rec)
    img = np.zeros((360, 640, 3), np.uint8)
    for k in range(8):
        b.pose(k, to_quat(phone_pose(123 + (10 if k >= 3 else 0), 20, 0)))
    b.offer(0, img, np.zeros((0, 4)))
    b.offer(1, img, np.zeros((0, 4)))                        # too soon
    b._last = (b._last[0] - LC.RT_EVERY_S - 0.01, b._last[1])
    b.offer(2, img, np.zeros((0, 4)))                        # not turned
    b.offer(3, img, [[1, 2, 3, 4]])                          # turned 10 deg
    b._last = (b._last[0] - LC.RT_AGAIN_S - 0.01, b._last[1])
    b.offer(4, img, np.zeros((0, 4)))                        # not turned, but a while later
    assert [g[1] for g in got] == [0, 3, 4] and got[1][3].tolist() == [[1, 2, 3, 4]]
    assert all(g[0] is b for g in got)
    rec.h = None                                              # (before the calibration: none)
    b._last = None
    b.offer(5, img, np.zeros((0, 4)))
    w.state, rec.h = "error", True                            # (the model did not load: none)
    b.offer(6, img, np.zeros((0, 4)))
    assert len(got) == 3
    w.state = "ready"
    b.offer(7, None, "not boxes")                             # never into tracking
    assert b.errors == 1 and "a frame left out" in rec.logs[0]


def test_the_worker_makes_frames_in_its_own_thread_and_stop_writes_the_files(tmp_path):
    rec = _rec()
    R = rec.R_cw.T
    depth = GroundDepth(rec, R)
    w = LC.RealtimeCloud(log=lambda m: None, depth=depth)
    try:
        assert w.thread.name == "rt-cloud" and w.thread.is_alive() and w.state == "ready"
        b = w.builder(rec)
        b.offer(3, np.zeros((360, 640, 3), np.uint8), np.zeros((0, 4)))
        assert V_until(lambda: b.frames == 1)
        m = b.finish(tmp_path)
        assert m["source"] == "realtime" and m["rough"] is True and m["frames"] == 1
        assert m["model"] == "fake-ground" and m["voxel_m"] == LC.RT_VOXEL_M
        assert {p.name for p in tmp_path.iterdir()} == {"cloud.bin", "cloud.json", "cloud.ply"}
        b.add(depth, 4, np.zeros((360, 640, 3), np.uint8), np.zeros((0, 4)), R)   # closed: no more
        assert b.frames == 1
        assert w.builder(rec).finish(tmp_path / "none") is None            # nothing made: no files
    finally:
        w.close()
    assert not w.thread.is_alive() and w.depth is None


def V_until(fn, timeout=10.0):
    import time

    end = time.time() + timeout
    while time.time() < end:
        if fn():
            return True
        time.sleep(0.01)
    return False


def test_a_depth_model_that_does_not_load_leaves_tracking_alone(monkeypatch):
    def broken(*a, **k):
        raise ImportError("no transformers")

    monkeypatch.setattr(LC, "DepthModel", broken)
    w = LC.RealtimeCloud(log=lambda m: None)
    w.thread.join(10)
    assert w.state == "error" and "no transformers" in w.error
    b = w.builder(_rec())
    b.offer(1, np.zeros((360, 640, 3), np.uint8), np.zeros((0, 4)))   # ignored, no raise
    assert b.frames == 0 and w._item is None
    w.close()


def test_a_recording_with_the_cloud_on_gets_its_cloud_and_off_gets_nothing(tmp_path, monkeypatch):
    import threading

    import test_live_track as T
    from groundtrack.config import default_config
    from groundtrack.live_track import LiveTracker, sensor_camera

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
    monkeypatch.setattr(LC, "RT_EVERY_S", 0.0)
    monkeypatch.setattr(LC, "RT_AGAIN_S", 0.0)
    noise = np.random.default_rng(1).integers(0, 255, (T.H, T.W, 3), dtype=np.uint8)

    def record(rt):
        rid = rt.begin(T.W, T.H)["id"]
        rec = rt.rec
        for n, i in enumerate(range(T.N), 1):
            assert rt.frame({"seq": i, "t": 3.0 + i / T.FPS, "w": T.W, "h": T.H, "q": T.Q_CAL}, T.picture(i, noise))
            assert T.until(lambda n=n: rec.n_tracked == n, 120 if n == 1 else 10)
        return rid, rec

    # off (the default): nothing of it
    rt = LiveTracker(tmp_path / "off", cfg=default_config(), model=T.FakeYOLO(), log=lambda m: None)
    rid, rec = record(rt)
    assert rt.cloud_on is False and rt.cloud is None and rec.cloud is None and "cloud" not in rt.status()
    assert not any(t.name == "rt-cloud" for t in threading.enumerate())
    with pytest.raises(FileNotFoundError):
        rt.cloud_since(rid)
    rt.end()
    assert T.until(lambda: rt.saved is not None, 60) and rt.saved["complete"]
    assert not list((tmp_path / "off").rglob("cloud.*"))

    # on: a depth frame now and then, in its own thread; the files at Stop
    rt = LiveTracker(tmp_path / "on", cfg=default_config(), model=T.FakeYOLO(), log=lambda m: None)
    _, _, R_cw, _ = sensor_camera([T.Q_CAL], (T.W, T.H))
    rt.cloud_depth = GroundDepth(SimpleNamespace(size=(T.W, T.H), K=sensor_camera([T.Q_CAL], (T.W, T.H))[1],
                                                 C=np.array([0, 0, 1.5])), R_cw.T)
    with pytest.raises(ValueError):
        rt.set_cloud("yes")
    rt.set_cloud(True)
    worker = rt.cloud
    st = rt.status()["cloud"]
    assert st == {"on": True, "state": "ready", "error": None, "points": 0, "frames": 0}
    rid, rec = record(rt)
    assert T.until(lambda: rt.cloud_since(rid)[0]["n"] > 0)
    head, data = rt.cloud_since(rid)
    assert head["rough"] and len(data) == 16 * head["n"] and rt.status()["cloud"]["points"] == head["n"]
    assert rt.cloud_depth.calls >= 1
    rt.end()
    assert T.until(lambda: rt.saved is not None, 60)
    assert rt.saved["complete"] and rt.saved["notes"] == []
    live = tmp_path / "on" / "runs" / "live" / rid / "live"
    meta, recs = LC.read_cloud(live)
    assert meta["source"] == "realtime" and meta["n"] == len(recs) > 100
    assert np.abs(recs["z"]).max() < 0.05                   # the flat ground
    assert "point cloud:" in (live.parent / "realtime_log.txt").read_text(encoding="utf-8")
    rt.set_cloud(False)                                     # off: the thread ends, the model goes
    assert rt.cloud is None and "cloud" not in rt.status()
    assert T.until(lambda: not worker.thread.is_alive()) and worker.depth is None
    rt.close()
