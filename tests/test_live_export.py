"""live-export: the scene package of a run for the Live view (synthetic demo site, no YOLO)."""

import json

import cv2
import numpy as np
import pandas as pd
import pytest

from groundtrack.config import load_config
from groundtrack.debug_video import HOLD_S, privacy_boxes, raw_all
from groundtrack.demo import ORIGIN, SyntheticCamera, make_demo_site
from groundtrack.homography import fit_homography, snapshot
from groundtrack.layout import Run
from groundtrack.live_export import (
    TRACK_DTYPE,
    camera_from_P,
    export_live,
    project,
    terrain_grid,
)

FPS = 25.0
EXTRA_BOX = (30, 600.0, 380.0, 660.0, 520.0)     # an untracked person (privacy_boxes.csv)


def _walkers(truth: dict, t: float) -> dict:
    """The demo's two people at time t (local m): A stands 1 s, walks east, stands; B walks
    south (demo.make_demo_site)."""
    a0, a1 = np.array(truth["a_start"]) - ORIGIN, np.array(truth["a_end"]) - ORIGIN
    s = np.clip((t - 1.0) / truth["a_walk_s"], 0, 1)
    return {1: a0 + (a1 - a0) * s, 2: np.array([-7.0, 14.0 - truth["b_speed"] * t])}


def _raw_tracks(cam: SyntheticCamera, truth: dict, n: int) -> pd.DataFrame:
    """Tracker rows of the demo's people, boxes as the camera sees a 1.75 m person."""
    rows = []
    for k in range(n):
        for tid, p in _walkers(truth, k / FPS).items():
            en = ORIGIN + p
            foot = cam.ground_to_pixel(en)[0]
            head = cam.project(np.r_[en, 1.75])[0]
            w = 0.38 * (foot[1] - head[1])
            rows.append({"track_id": tid, "class": "person", "class_id": 0, "frame": k,
                         "time_s": round(k / FPS, 4), "x1": foot[0] - w / 2, "y1": head[1],
                         "x2": foot[0] + w / 2, "y2": foot[1], "confidence": 0.9,
                         "predicted": k % 40 == 39})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def demo_export(tmp_path_factory):
    from groundtrack.exports import write_points_csv
    from groundtrack.trajectories import process_tracks

    root = tmp_path_factory.mktemp("live")
    site = make_demo_site(root)
    cfg = load_config(site["config"])
    cam, n = site["camera"], site["n_frames"]
    world = ORIGIN + np.array([[-13.5, 13.6], [-10.5, 14.4], [10.4, 19.5], [-7.25, 31.25],
                               [15.25, 35.25], [0.3, 0.3], [-9.7, 2.3], [12.75, 32.75]])
    h = fit_homography(cam.ground_to_pixel(world), world, image_size=(cam.width, cam.height))
    h.save(cfg.path("homography"))

    run = Run(root / "runs" / "demo" / "live_test").make()
    snapshot(cfg.path("homography"), run.homography_used)
    raw = _raw_tracks(cam, site["truth"], n)
    raw.to_csv(run.raw_tracks, index=False)
    f, x1, y1, x2, y2 = EXTRA_BOX
    pd.DataFrame([{"frame": f, "class": "person", "x1": x1, "y1": y1, "x2": x2, "y2": y2}]) \
        .to_csv(run.privacy, index=False)
    run.meta.write_text(json.dumps({"video": str(site["video"]), "fps": FPS, "width": cam.width,
                                    "height": cam.height, "n_frames": n, "start_frame": 0,
                                    "end_frame": n, "vid_stride": 1, "reference_frame": 0}))
    points, summary = process_tracks(raw, cfg, h, FPS, log=lambda *_: None)
    write_points_csv(points, run.points)
    summary.to_csv(run.tracks, index=False)
    res = export_live(run.root, cfg, detect=False)
    scene = json.loads((run.root / "live" / "scene.json").read_text(encoding="utf-8"))
    return {"site": site, "cfg": cfg, "run": run, "h": h, "raw": raw, "res": res,
            "scene": scene, "live": run.root / "live"}


def test_scene_header(demo_export):
    s, live, n = demo_export["scene"], demo_export["live"], demo_export["site"]["n_frames"]
    assert set(s) == {"version", "clip", "run", "crs", "origin", "fps", "frames", "duration_s",
                      "image_size", "camera", "terrain", "aerial", "tracks", "video", "R_max",
                      "speed_ramp", "source_runs"}
    assert (s["version"], s["clip"], s["run"]) == (1, "demo", "live_test")
    assert s["fps"] == FPS and s["frames"] == n and s["duration_s"] == pytest.approx(n / FPS)
    assert s["image_size"] == [1280, 720]
    assert 25.0 <= s["R_max"] <= 110.0
    assert s["video"] == {"file": "video.mp4", "blurred": True, "stabilized": False}
    for k in ("scene.json", "terrain.f32", "aerial.jpg", "tracks.bin", "tracks_index.json",
              "video.mp4", "check.png"):
        assert (live / k).exists(), k
    t = s["terrain"]
    assert (live / "terrain.f32").stat().st_size == t["nx"] * t["ny"] * 4
    # the terrain covers R_max + 20 m around the camera, the aerial the same extent
    cx, cy = s["camera"]["C"][:2]
    assert t["x0"] <= cx - s["R_max"] - 20 and t["x0"] + t["nx"] >= cx + s["R_max"] + 20
    assert t["y1"] >= cy + s["R_max"] + 20 and t["y1"] - t["ny"] <= cy - s["R_max"] - 20
    assert s["aerial"]["extent"] == [t["x0"], t["x0"] + t["nx"], t["y1"] - t["ny"], t["y1"]]
    aer = cv2.imread(str(live / "aerial.jpg"), cv2.IMREAD_UNCHANGED)
    assert aer.ndim == 2 and max(aer.shape) == 2048
    # flat calibration: the ground is the datum
    assert not np.fromfile(live / "terrain.f32", "<f4").any()


def test_camera_round_trip(demo_export):
    cam_s, h = demo_export["scene"]["camera"], demo_export["h"]
    P, K, R, C = (np.array(cam_s[k]) for k in ("P", "K", "R_cw", "C"))
    assert np.allclose(np.tril(K, -1), 0) and (np.diag(K) > 0).all() and K[2, 2] == 1
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9) and np.linalg.det(R) == pytest.approx(1)
    assert np.allclose(P, h._rays()[0])
    X = np.random.default_rng(0).uniform([-20, 0, -1], [20, 60, 3], (50, 3))
    uv, depth = project(K, R, C, X)
    ph = np.column_stack([X, np.ones(len(X))]) @ P.T
    assert (depth > 0).all() and (ph[:, 2] > 0).all()
    assert np.allclose(uv, ph[:, :2] / ph[:, 2:3], atol=1e-6)
    # the synthetic camera it was calibrated from (OpenCV axes, local frame)
    truth = demo_export["site"]["camera"]
    assert np.allclose(C, truth.C - np.r_[h.origin, 0], atol=0.05)
    assert np.degrees(np.arccos(np.clip((np.trace(R @ truth.R.T) - 1) / 2, -1, 1))) < 0.1
    assert cam_s["height_m"] == pytest.approx(10.0, abs=0.05)
    assert cam_s["yaw_deg"] == pytest.approx(8.0, abs=0.1)
    assert cam_s["tilt_deg"] == pytest.approx(20.0, abs=0.1)


def test_camera_from_P_exact_and_mirrored():
    cam = SyntheticCamera(yaw_deg=40.0, pitch_deg=15.0, roll_deg=3.0, cam_local=(2.0, -5.0),
                          origin=np.zeros(2))
    P = cam.K @ np.column_stack([cam.R, -cam.R @ cam.C])
    K, R, C = camera_from_P(2.5 * P)
    assert np.allclose(K, cam.K) and np.allclose(R, cam.R) and np.allclose(C, cam.C)
    with pytest.raises(ValueError):          # points in front at negative depth: mirrored
        camera_from_P(-P)


def test_terrain_grid_matches_terrain_height(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    from groundtrack.posefit import pose_homography
    from groundtrack.terrain import load_terrain

    # a 1 m DTM, 60 x 40 m: a slope with a 3 m terrace in its north half
    rows, cols = np.mgrid[0:40, 0:60]
    z = (20.0 + 0.05 * cols + np.where(rows < 18, 3.0, 0.0)).astype(np.float32)
    left, top = ORIGIN[0] - 20.0, ORIGIN[1] + 30.0
    path = tmp_path / "dtm.tif"
    with rasterio.open(path, "w", driver="GTiff", height=40, width=60, count=1, dtype="float32",
                       crs="EPSG:27700", transform=from_origin(left, top, 1.0, 1.0)) as ds:
        ds.write(z, 1)
    terrain = load_terrain(path)
    h = pose_homography(ORIGIN[0], ORIGIN[1] - 10, 2.0, 0.0, 5.0, 1000.0, (1280, 720))
    h.set_terrain(terrain, 21.0)
    centre = np.array([0.3, -2.6]) + ORIGIN - np.asarray(h.origin)
    grid, x0, y1 = terrain_grid(h, centre, 30.0)
    ny, nx = grid.shape
    assert grid.dtype == np.float32 and x0 == np.floor(centre[0] - 30) and y1 == np.ceil(centre[1] + 30)
    assert x0 + nx >= centre[0] + 30 and y1 - ny <= centre[1] - 30
    gx, gy = np.meshgrid(x0 + np.arange(nx) + 0.5, y1 - np.arange(ny) - 0.5)
    world = np.column_stack([gx.ravel(), gy.ravel()]) + np.asarray(h.origin)
    want = terrain.height(world) - 21.0
    inside = np.isfinite(want)
    assert inside.any() and (~inside).any()          # the grid reaches past the model
    assert np.allclose(grid.ravel()[inside], want[inside], atol=1e-5)
    assert not grid.ravel()[~inside].any()           # past the model: the datum
    # row 0 is north: the terrace (3 m higher) is at the top
    assert grid[1, nx // 2] > grid[-2, nx // 2] + 2


def test_tracks_match_points_csv(demo_export):
    run, h, s = demo_export["run"], demo_export["h"], demo_export["scene"]
    points = pd.read_csv(run.points)
    rec = np.fromfile(demo_export["live"] / "tracks.bin", TRACK_DTYPE)
    assert TRACK_DTYPE.itemsize == 30
    assert len(rec) == len(points) == s["tracks"]["n_points"]
    assert s["tracks"]["n_tracks"] == points["track_id"].nunique()
    assert np.all(np.diff(rec["frame"].astype(np.int64) * 10_000 + rec["track_id"]) > 0)
    p = points.sort_values(["frame", "track_id"]).reset_index(drop=True)
    assert np.array_equal(rec["track_id"], p["track_id"]) and np.array_equal(rec["frame"], p["frame"])
    assert np.allclose(rec["x"], p["x"] - h.origin[0], atol=1e-3)
    assert np.allclose(rec["y"], p["y"] - h.origin[1], atol=1e-3)
    assert np.allclose(rec["t"], p["frame"] / FPS, atol=1e-5)
    assert np.allclose(rec["speed"], p["speed"], atol=1e-4)
    assert not rec["z"].any()
    src = p["source"].map({"detected": 0, "predicted": 1, "stitched": 2, "interpolated": 3})
    assert np.array_equal(rec["source"], src) and (rec["source"] == 1).any()
    assert s["tracks"]["classes"] == {"0": "person"} and not rec["class"].any()
    index = json.loads((demo_export["live"] / "tracks_index.json").read_text(encoding="utf-8"))
    assert set(index) == {str(t) for t in points["track_id"].unique()}
    for tid, g in points.groupby("track_id"):
        e = index[str(tid)]
        assert (e["class"], e["group"]) == ("person", "people")
        assert (e["first_frame"], e["last_frame"]) == (g["frame"].min(), g["frame"].max())


def test_check_reprojects_onto_the_feet(demo_export):
    res = demo_export["res"]
    assert res["reproj_median_px"] < 1.0                # steady walkers: smoothing changes ~nothing
    img = cv2.imread(str(res["check"]))
    assert img.shape[:2] == (720, 1280)


def test_every_frame_blurred_where_privacy_boxes_say(demo_export):
    from groundtrack.debug_video import blur_boxes

    run, live, n = demo_export["run"], demo_export["live"], demo_export["site"]["n_frames"]
    boxes = privacy_boxes(raw_all(run.root, demo_export["raw"]), True, True,
                          int(round(HOLD_S * FPS)))
    raw = demo_export["raw"]
    assert (boxes["frame"] == EXTRA_BOX[0]).sum() > (raw["frame"] == EXTRA_BOX[0]).sum()
    by_frame = {f: g[["x1", "y1", "x2", "y2"]].to_numpy() for f, g in boxes.groupby("frame")}
    src = cv2.VideoCapture(str(demo_export["site"]["video"]))
    out = cv2.VideoCapture(str(live / "video.mp4"))
    assert int(out.get(cv2.CAP_PROP_FRAME_COUNT)) == n
    for f in range(n):
        ok_a, a = src.read()
        ok_b, b = out.read()
        assert ok_a and ok_b
        want = a.copy()
        blur_boxes(want, by_frame.get(f, np.zeros((0, 4))))
        to_want = np.abs(b.astype(int) - want).max(axis=2)
        to_src = np.abs(b.astype(int) - a).max(axis=2)
        # the output is the footage blurred in exactly these boxes (H.264 aside) ...
        assert np.median(to_want) <= 2, f
        # ... and where the blur changed the picture, it follows the blurred version
        changed = np.abs(want.astype(int) - a).max(axis=2) > 12
        assert changed.sum() > 200, f                   # people in every frame
        assert to_want[changed].mean() < 0.4 * to_src[changed].mean(), f


def test_video_steadied_by_the_registration(demo_export, tmp_path):
    """With a camera-motion registration each frame is blurred, then warped into the
    calibration frame (black where the frame no longer reaches)."""
    import shutil

    from groundtrack.debug_video import blur_boxes
    from groundtrack.live_export import write_video
    from groundtrack.registration import save_registration

    site, n, raw = demo_export["site"], demo_export["site"]["n_frames"], demo_export["raw"]
    run = Run(tmp_path / "runs" / "demo" / "steady").make()
    for p in (demo_export["run"].raw_tracks, demo_export["run"].privacy):
        shutil.copy2(p, run.work / p.name)
    # the camera drifting: frame f's pixels sit (0.4 f, -0.2 f) px off the calibration frame's
    Hs = [np.array([[1, 0, 0.4 * f], [0, 1, -0.2 * f], [0, 0, 1.0]]) for f in range(n)]
    save_registration(run.registration, range(n), Hs, [100] * n, 0)
    out = tmp_path / "steady.mp4"
    k, kept, steadied = write_video(demo_export["cfg"], run, site["video"], out, range(n), FPS,
                                    keep=150, detect=False, log=lambda *_: None)
    assert steadied and k == n
    boxes = privacy_boxes(raw_all(run.root, raw), True, True, int(round(HOLD_S * FPS)))
    by_frame = {f: g[["x1", "y1", "x2", "y2"]].to_numpy() for f, g in boxes.groupby("frame")}
    src, res = cv2.VideoCapture(str(site["video"])), cv2.VideoCapture(str(out))
    for f in range(n):
        ok_a, a = src.read()
        ok_b, b = res.read()
        assert ok_a and ok_b
        H, W = a.shape[:2]
        blurred = a.copy()
        blur_boxes(blurred, by_frame.get(f, np.zeros((0, 4))))
        want = cv2.warpPerspective(blurred, Hs[f], (W, H), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT)
        plain = cv2.warpPerspective(a, Hs[f], (W, H), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT)
        if f == 150:
            assert np.array_equal(kept, want)
        to_want = np.abs(b.astype(int) - want).max(axis=2)
        assert np.median(to_want) <= 4, f           # H.264 on noise moving by sub-pixels
        # the people are blurred where they stand after the warp (blurred first, then warped)
        changed = np.abs(want.astype(int) - plain).max(axis=2) > 12
        to_plain = np.abs(b.astype(int) - plain).max(axis=2)
        assert to_want[changed].mean() < 0.4 * to_plain[changed].mean(), f
        if f >= 30:                                 # black where the camera turned away
            assert b[:, :int(0.4 * f) - 4].mean() < 6 and b[H - int(0.2 * f) + 4:].mean() < 6
