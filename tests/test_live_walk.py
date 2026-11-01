"""live_walk (plans/live_controller/11, step 3): a Realtime recording's cloud placed sharp from its
walk, worked out from its video (VGGT), for the cloud only. A stand-in VGGT renders a known street
(flat ground, a wall, a parked car, a person) from a known walk, each batch in its own frame and scale
(as VGGT's: the batch's first camera, any scale), so the joins, the pinning to the calibration and the
ground's scale are checked against the truth."""

import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from groundtrack import live_cloud as LC
from groundtrack import live_walk as LW
from groundtrack.live_track import POSE_DTYPE, level_camera

ROOT = Path(__file__).resolve().parents[1]
W, H, F_FULL = 320, 180, 266.0              # the recording's frames and focal (px)
N, CAL_END = 120, 30                         # frames; still (the calibration's second) up to here
WALL_Y, CAR, PERSON = 15.0, ((2.0, 7.0, 0.0), (5.0, 9.0, 1.4)), ((-2.4, 8.8, 0.0), (-1.6, 9.2, 1.8))
R_CAL = level_camera(20.0)                   # world_from_camera: north, 20 deg down


def rz(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])


def true_camera(k):
    """(R_wc, C) at frame k: still at the calibration, then a walk north-east, turning a little."""
    if k <= CAL_END:
        return R_CAL, np.array([0.0, 0.0, 1.5])
    u = (k - CAL_END) / (N - CAL_END)
    return rz(-12 * math.sin(3 * u)) @ R_CAL, np.array([2.0 * u, 4.0 * u, 1.5])


def _box_hit(C, d, lo, hi):
    lo, hi = np.array(lo), np.array(hi)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (lo - C) / d, (hi - C) / d
    tn = np.nanmax(np.minimum(t1, t2), axis=-1)
    tf = np.nanmin(np.maximum(t1, t2), axis=-1)
    return np.where((tf >= tn) & (tn > 0), tn, np.inf)


def render(K, hw, R_wc, C):
    """The street's depth (camera z) at every pixel; inf where the sky is."""
    h, w = hw
    rays = LW.pixel_rays(K, hw)
    d = rays @ R_wc.T
    with np.errstate(divide="ignore", invalid="ignore"):
        tg = np.where(d[..., 2] < -1e-6, -C[2] / d[..., 2], np.inf)
        tw = np.where(d[..., 1] > 1e-6, (WALL_Y - C[1]) / d[..., 1], np.inf)
    zw = C[2] + tw * d[..., 2]
    tw = np.where((zw >= 0) & (zw <= 8), tw, np.inf)
    t = np.minimum.reduce([tg, tw, _box_hit(C, d, *CAR), _box_hit(C, d, *PERSON)])
    return t


class FakeVGGT:
    """VGGT's answers for a batch: each frame's camera in the batch's frame (its first camera, the
    batch's own scale), its depth in that scale, K at the frames' size; the sky unconfident."""
    name = "fake-vggt"

    def __init__(self):
        self.calls = []
        self.scales = iter([0.37, 2.9, 1.3, 0.8, 5.0, 0.6])

    def __call__(self, frames, keys):
        self.calls.append(list(keys))
        h, w = frames[0].shape[:2]
        K = np.array([[F_FULL * w / W, 0, w / 2], [0, F_FULL * h / H, h / 2], [0, 0, 1.0]])
        sb = next(self.scales)
        R0, C0 = true_camera(keys[0])
        Rs, ts, Ds, Cs = [], [], [], []
        for k in keys:
            R_wc, C = true_camera(k)
            R_cw = R_wc.T
            Rs.append(R_cw @ R0)                              # batch frame = the first camera's, scaled
            ts.append(sb * R_cw @ (C0 - C))
            t = render(K, (h, w), R_wc, C)
            sky = ~np.isfinite(t)
            Ds.append(np.where(sky, 1000.0, sb * t))
            Cs.append(np.where(sky, 1.0, 6.0))
        n = len(keys)
        return np.array(Rs), np.array(ts), np.repeat(K[None], n, 0), np.array(Ds, np.float32), np.array(Cs, np.float32)


def person_detector(rgb, k):
    """The person's box in frame k (full frame pixels), from the true camera."""
    R_wc, C = true_camera(k)
    lo, hi = PERSON
    P = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    K = np.array([[F_FULL, 0, W / 2], [0, F_FULL, H / 2], [0, 0, 1.0]])
    q = (K @ (R_wc.T @ (P - C).T)).T
    if (q[:, 2] <= 0.1).any():
        return np.zeros((0, 4))
    u, v = q[:, 0] / q[:, 2], q[:, 1] / q[:, 2]
    return np.array([[u.min(), v.min(), u.max(), v.max()]])


def recording(root, growing=False, realtime=True):
    """A Realtime recording's folder: scene.json, a video of fresh frames, the recorder's poses."""
    live = root / "live"
    live.mkdir(parents=True)
    K = [[F_FULL, 0, W / 2], [0, F_FULL, H / 2], [0, 0, 1]]
    (live / "scene.json").write_text(json.dumps({
        "version": 1, "realtime": realtime, "growing": growing, "fps": 30.0, "frames": N, "image_size": [W, H],
        "camera": {"K": K, "R_cw": R_CAL.T.tolist(), "C": [0, 0, 1.5], "height_m": 1.5}, "R_max": 50.0,
        "video": {"file": "video.mp4", "blurred": True}, "origin": {"E": 0, "N": 0, "datum_z": None}}))
    (root / "recording.json").write_text(json.dumps({"id": "test", "tilt_offset_deg": 0.0}))
    vw = cv2.VideoWriter(str(live / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
    rng = np.random.default_rng(1)
    for _ in range(N):
        vw.write(rng.integers(0, 255, (H, W, 3), dtype=np.uint8))
    vw.release()
    p = np.zeros(N, POSE_DTYPE)
    p["k"], p["t"] = np.arange(N), np.arange(N) / 30.0
    for k in range(N):
        a = 0.0 if k <= CAL_END else math.radians(k - CAL_END)
        p["q"][k] = [math.cos(a / 2), 0, 0, math.sin(a / 2)]
    p.tofile(live / "recorder_pose.rec")
    return live


# --------------------------------------------------------------------------- the pieces
def test_a_similarity_is_found_past_outliers():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(2000, 3)) * 3
    R = rz(37) @ np.array([[1, 0, 0], [0, math.cos(0.4), -math.sin(0.4)], [0, math.sin(0.4), math.cos(0.4)]])
    B = 2.5 * A @ R.T + [1, -2, 0.5]
    B[:400] += rng.normal(size=(400, 3)) * 5                  # a fifth of them wrong
    s, R2, t, err = LW.robust_similarity(A, B, np.ones(len(A)))
    assert s == pytest.approx(2.5, rel=1e-3) and np.allclose(R2, R, atol=1e-3) and np.allclose(t, [1, -2, 0.5], atol=1e-2)
    assert err < 0.01


def test_the_calibrations_second_is_found_as_the_recorder_found_it():
    p = np.zeros(90, POSE_DTYPE)
    p["k"], p["t"] = np.arange(90), np.arange(90) / 30.0
    for k in range(90):                                       # turning until frame 20, then still
        a = math.radians(min(k, 20) * 2.0)
        p["q"][k] = [math.cos(a / 2), 0, 0, math.sin(a / 2)]
    ks = LW.still_window(p)
    assert ks[0] == 20 and 29 <= len(ks) <= 31
    assert LW.still_window(np.zeros(0, POSE_DTYPE)) == []


def test_frames_picked_at_the_rate_with_the_calibrations_and_the_right_size():
    ks = LW.pick_times(300, 30.0, anchor=17, rate=2.0)
    assert 17 in ks and ks[:3] == [0, 15, 17] and len(ks) == 21
    assert len(LW.pick_times(30 * 600, 30.0, 0, 2.0, most=240)) <= 241           # a long one: fewer a second
    assert LW.small_size(1920, 1080) == (518, 294) and LW.small_size(1080, 1920) == (294, 518)
    assert LW.small_size(1280, 720) == (518, 294)


def test_a_repeated_frame_is_skipped_for_the_next_fresh_one(tmp_path):
    vw = cv2.VideoWriter(str(tmp_path / "v.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (64, 48))   # (frame by frame)
    rng = np.random.default_rng(2)
    last = None
    for k in range(40):
        if k in (10, 11, 12) and last is not None:            # no frame came: the last one again
            img = last
        else:                                                 # (smooth, as a picture: noise re-encodes)
            img = cv2.resize(rng.integers(0, 255, (6, 8, 3), dtype=np.uint8), (64, 48), interpolation=cv2.INTER_CUBIC)
        vw.write(img)
        last = img
    vw.release()
    got = LW.read_frames(tmp_path / "v.avi", [0, 10, 20], (28, 14), 30.0)
    assert [k for k, _ in got] == [0, 13, 20] and got[0][1].shape == (14, 28, 3)


def test_the_ground_is_the_lowest_big_level_plane_not_a_table():
    rng = np.random.default_rng(3)
    table = np.c_[rng.uniform(-1, 1, 6000), rng.uniform(1, 2, 6000), np.full(6000, -0.5)]
    floor = np.c_[rng.uniform(-4, 4, 3000), rng.uniform(0.5, 6, 3000), np.full(3000, -1.5)]
    noise = rng.uniform(-3, 3, (300, 3)) - [0, 0, 2.5]         # a few stray points below everything
    n, d, inl = LW.fit_ground(np.concatenate([table, floor, noise]), tol=0.02)
    assert d == pytest.approx(1.5, abs=0.01) and n[2] > 0.999 and inl[6000:9000].mean() > 0.95
    n, d, _ = LW.fit_ground(table, tol=0.02)
    assert d == pytest.approx(0.5, abs=0.01)
    assert LW.fit_ground(np.c_[rng.uniform(-1, 1, (500, 2)), rng.uniform(0.1, 1, 500)], tol=0.02) is None   # nothing below


def test_a_stretch_that_cannot_be_joined_is_left_out_and_the_calibrations_kept():
    fake = FakeVGGT()
    frames = [(k, np.zeros((42, 70, 3), np.uint8)) for k in range(0, 120, 4)]
    batches = LW.run_batches(frames, fake, chunk=12, overlap=4, log=lambda *_: None)
    assert [len(b["ks"]) for b in batches] == [12, 12, 12, 6] and batches[1]["ks"][:4] == batches[0]["ks"][-4:]
    batches[2]["conf"][:] = 1.0                               # nothing VGGT is sure of
    logs = []
    cams, errs, dropped = LW.join_batches(batches, anchor=8, log=logs.append)
    assert sorted(cams) == [k for k, _ in frames[:20]] and len(errs) == 1 and errs[0] < 1e-3
    assert dropped == [(b["ks"][0], b["ks"][-1]) for b in batches[2:]] and "nothing to hold on to" in logs[0]
    assert LW.join_batches(batches, anchor=100, log=logs.append)[2][0] == (0, 76)          # the anchor's stretch kept
    # joined: every frame's camera centre at its true place, up to the first batch's scale
    c0 = true_camera(0)[1]
    for k in (0, 40, 76):
        R_cw, t = cams[k][:2]
        c = -R_cw.T @ t / 0.37                                # (the first batch's frame: camera 0's)
        assert np.allclose(R_CAL @ c, true_camera(k)[1] - c0, atol=1e-6)


# --------------------------------------------------------------------------- the whole
def test_the_walk_cloud_lands_where_the_street_is(tmp_path, monkeypatch):
    monkeypatch.setattr(LW, "SIDE", 140)
    live = recording(tmp_path / "rec")
    LC.write_cloud(live, np.zeros((3, 3)), np.zeros((3, 3)), np.ones(3), {"source": "realtime", "rough": True})
    fake, logs = FakeVGGT(), []
    meta = LW.make_walk_cloud(tmp_path / "rec", rate=10.0, model=fake, detector=person_detector, chunk=16,
                              log=logs.append)
    assert len(fake.calls) >= 3 and meta["source"] == "walk" and meta["rough"] is False
    w = meta["walk"]
    assert w["batches"] == len(fake.calls) and max(w["join_err"]) < 1e-3 and w["left_out"] == []
    assert w["tilt_fix_deg"] < 0.2 and w["ground_rms_m"] < 0.02 and 15 <= w["anchor_frame"] <= 16
    assert w["walked_m"] == pytest.approx(math.hypot(2, 4), rel=0.05) and w["people"] == "blurred"
    _, rec = LC.read_cloud(live)
    P = np.c_[rec["x"], rec["y"], rec["z"]].astype(float)
    wall = P[P[:, 1] > WALL_Y - 0.3]
    assert len(wall) > 100 and np.median(wall[:, 1]) == pytest.approx(WALL_Y, abs=0.06)
    on_car = (P[:, 0] > 2.1) & (P[:, 0] < 4.9) & (P[:, 1] > 7.1) & (P[:, 1] < 8.9)
    by_car = (P[:, 0] > 1.85) & (P[:, 0] < 5.15) & (P[:, 1] > 6.85) & (P[:, 1] < 9.15)           # its roof and sides
    ground = P[(P[:, 1] < WALL_Y - 0.5) & ~by_car & (np.abs(P[:, 0] + 2) > 1) & (P[:, 2] < 1.0)]
    assert len(ground) > 500 and np.quantile(np.abs(ground[:, 2]), 0.95) < 0.06             # the ground at z = 0
    assert on_car.any() and np.median(P[on_car, 2]) == pytest.approx(1.4, abs=0.06)          # the car's roof: not the ground
    person = (np.abs(P[:, 0] + 2) < 0.45) & (np.abs(P[:, 1] - 9) < 0.25) & (P[:, 2] > 0.3) & (P[:, 2] < 1.7)
    assert not person.any()                                                                   # left out
    # the rough cloud kept beside it, and back on asking
    assert json.loads((live / "cloud_rough.json").read_text())["source"] == "realtime"
    assert LW.restore_rough(live) and LC.read_cloud(live)[0]["source"] == "realtime"
    assert not (live / "cloud_rough.json").exists()


def test_the_walk_cloud_refuses_what_it_is_not_for(tmp_path):
    recording(tmp_path / "a", realtime=False)
    with pytest.raises(ValueError, match="not a Realtime"):
        LW.make_walk_cloud(tmp_path / "a", model=FakeVGGT(), detector=False)
    recording(tmp_path / "b", growing=True)
    with pytest.raises(ValueError, match="still recording"):
        LW.make_walk_cloud(tmp_path / "b", model=FakeVGGT(), detector=False)
    assert not list(tmp_path.rglob("cloud.*"))


def test_nothing_loads_the_walk_unless_asked():
    """Off: no module imports live_walk at load (the CLI's command only), and it loads neither torch
    nor VGGT until a walk cloud is made."""
    users = sorted(p.name for p in (ROOT / "groundtrack").glob("*.py") if p.name != "live_walk.py"
                   and any(i in p.read_text(encoding="utf-8") for i in ("from .live_walk", "import live_walk")))
    assert users == ["cli.py"]
    cli = (ROOT / "groundtrack" / "cli.py").read_text(encoding="utf-8")
    assert "from .live_walk import" in cli.split("def cmd_live_cloud(args):")[1].split("\ndef ")[0]
    r = subprocess.run([sys.executable, "-c", "import sys, groundtrack.live_walk, groundtrack.cli, groundtrack.desktop; "
                        "print('torch' in sys.modules, 'vggt' in sys.modules)"], cwd=ROOT, capture_output=True,
                       text=True, timeout=180)
    assert r.returncode == 0 and r.stdout.split() == ["False", "False"], r.stderr
