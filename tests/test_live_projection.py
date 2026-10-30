"""Live controller projection core: the laws of each camera move, a filmed clip, and JS = Python."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from groundtrack import live_projection as lp

REPO = Path(__file__).resolve().parents[1]
LIVE = REPO / "groundtrack" / "desktop_web" / "live"
JS = LIVE / "projection.js"
RICHMOND = Path.home() / "Downloads" / "richmond_analysis" / "results" / "compare"
CM = 0.01
BISECTED = lp.STEP_M / 2 ** lp.BISECT      # the raycast's resolution along the ground (1.95 mm)


# --------------------------------------------------------------------------- scenes and poses
def cam_R_wc(yaw, tilt, roll=0.0):
    """world_from_camera (OpenCV axes): forward at heading yaw (deg, clockwise from north), tilt
    deg down, roll deg about the forward axis (right side down)."""
    p, t = np.radians(yaw), np.radians(tilt)
    fwd = np.array([np.sin(p) * np.cos(t), np.cos(p) * np.cos(t), -np.sin(t)])
    right = np.array([np.cos(p), -np.sin(p), 0.0])
    R_wc = np.column_stack([right, np.cross(fwd, right), fwd])
    return R_wc @ Rot.from_rotvec([0, 0, np.radians(roll)]).as_matrix()


def make_meta(C=(0.0, 0.0, 1.5), yaw=30.0, tilt=2.0, roll=0.0, f=1500.0, size=(1920, 1080),
              R_max=1e4, terrain=None):
    """scene.json of a synthetic camera; terrain = (heights (ny, nx), x0, y1) or None."""
    W, H = size
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    R_cw = cam_R_wc(yaw, tilt, roll).T
    C = np.asarray(C, float)
    P = K @ np.column_stack([R_cw, -R_cw @ C])
    meta = {"version": 1, "image_size": [W, H], "R_max": R_max,
            "camera": {"P": P.tolist(), "K": K.tolist(), "R_cw": R_cw.tolist(), "C": C.tolist()},
            "terrain": None}
    heights = None
    if terrain is not None:
        z, x0, y1 = terrain
        meta["terrain"] = {"file": "terrain.f32", "nx": z.shape[1], "ny": z.shape[0], "x0": x0,
                           "y1": y1, "cell": 1.0}
        heights = np.asarray(z, np.float32).ravel()
    return meta, heights


def bumpy(x0=-200, y1=200, n=400):
    """Gentle hills, a ramp up to a 3 m terrace in the north-east: heights at 1 m cell centres."""
    xs = x0 + 0.5 + np.arange(n)
    ys = y1 - 0.5 - np.arange(n)
    X, Y = np.meshgrid(xs, ys)
    z = 0.8 * np.sin(X / 23.0) * np.cos(Y / 31.0) + 0.4 * np.sin((X + Y) / 11.0)
    z += np.clip((X + Y - 60) / 8.0, 0, 1) * 3.0
    return z, float(x0), float(y1)


def pit(x0=-200, y1=200, n=400):
    """bumpy() with a deep low spot: 20 x 20 m, 8 m below the datum (like Trafalgar's -8 m), in front
    of the camera of terrain_meta("pit")."""
    z, x0, y1 = bumpy(x0, y1, n)
    X, Y = np.meshgrid(x0 + 0.5 + np.arange(n), y1 - 0.5 - np.arange(n))
    return np.where((np.abs(X + 14) < 10) & (np.abs(Y - 20) < 10), -8.0, z), x0, y1


def terrain_meta(kind):
    """The scenes of the skip and JS tests: bumpy ground, the same with a deep pit (camera at
    Trafalgar's 1.96 m), flat."""
    z = {"bumpy": bumpy, "pit": pit}.get(kind)
    return make_meta(C=(3.0, -4.0, 1.96 if kind == "pit" else 1.7), yaw=-35.0, tilt=3.0, roll=0.6,
                     R_max=110.0, terrain=z() if z else None)


def pose(R_wc_G, p=None, t=0.0):
    x, y, z, w = Rot.from_matrix(R_wc_G).as_quat()
    return {"type": "pose", "v": 1, "q": [w, x, y, z], "p": None if p is None else list(map(float, p)), "t": t}


def ground_points(scene, bearings, dists):
    """Ground points at bearings (deg, absolute) and distances from the start camera."""
    b = np.radians(np.asarray(bearings, float))
    d = np.asarray(dists, float)
    x = scene.C[0] + d * np.sin(b)
    y = scene.C[1] + d * np.cos(b)
    return np.column_stack([x, y, lp.ground_height(scene, x, y)])


def polar(scene, X):
    e = X[:, :2] - scene.C[:2]
    return np.hypot(e[:, 0], e[:, 1]), np.degrees(np.arctan2(e[:, 0], e[:, 1]))


def turn(axis, deg):
    return Rot.from_rotvec(np.asarray(axis, float) / np.linalg.norm(axis) * np.radians(deg)).as_matrix()


# the phone: held at another heading and tilt than the clip's camera, in its own gravity frame
PHONE0 = cam_R_wc(yaw=200.0, tilt=12.0)


def phone_scene(**kw):
    meta, heights = make_meta(**kw)
    return lp.prepare_scene(meta, heights)


# --------------------------------------------------------------------------- 1. identity
@pytest.mark.parametrize("terrain", [False, True])
def test_identity_pose_draws_every_visible_point_where_it_is(terrain):
    meta, heights = make_meta(C=(5.0, -3.0, 8.0 if terrain else 1.6), yaw=40.0, tilt=3.0, roll=0.8,
                              R_max=150.0, terrain=bumpy() if terrain else None)
    scene = lp.prepare_scene(meta, heights)
    rng = np.random.default_rng(1)
    X = ground_points(scene, 40 + rng.uniform(-30, 30, 2000), rng.uniform(2, 140, 2000))
    zero = pose(PHONE0, p=[1.0, 2.0, 0.3])
    for gain in (1.0, 2.5):
        cam = lp.pose_to_camera(scene, zero, zero, gain=gain, scale=3.0)
        Xp, st = lp.project(scene, cam, X)
        u = (X - scene.C) @ (scene.K @ scene.R_cw).T
        u = u[:, :2] / u[:, 2:3]
        inview = (u[:, 0] >= 0) & (u[:, 0] <= scene.W) & (u[:, 1] >= 0) & (u[:, 1] <= scene.H)
        drawn = st == lp.DRAWN
        if terrain:          # ground hidden behind a hill is drawn on the hill: compare the rest
            assert drawn[inview].mean() > 0.9
        else:
            assert drawn[inview].all()
        err = np.linalg.norm(Xp[drawn] - X[drawn], axis=1)
        hidden = err > 0.05
        assert hidden.mean() < (0.05 if terrain else 1e-9)
        assert err[~hidden].max() < CM
        assert (st[~inview] == lp.OUT_OF_VIEW).all()
    # no zero yet: the start camera
    cam = lp.pose_to_camera(scene, zero, None)
    assert np.allclose(cam.R_cw, scene.R_cw) and np.allclose(cam.C, scene.C)


# --------------------------------------------------------------------------- 2. pan
@pytest.mark.parametrize("theta", [10.0, -10.0, 30.0, -30.0, 46.0, -46.0])
def test_pan_turns_the_plan_the_other_way(theta):
    scene = phone_scene(yaw=30.0, tilt=4.0, roll=0.0, R_max=1e4)
    zero = pose(PHONE0)
    Q = turn([0, 0, 1], -theta) @ PHONE0                        # pan right: heading + theta
    assert np.isclose((np.degrees(lp.heading(Q) - lp.heading(PHONE0)) + 180) % 360 - 180, theta)
    cam = lp.pose_to_camera(scene, pose(Q), zero)
    rng = np.random.default_rng(2)
    X = ground_points(scene, 30 + theta + rng.uniform(-25, 25, 500), rng.uniform(3, 80, 500))
    Xp, st = lp.project(scene, cam, X)
    ok = st == lp.DRAWN
    assert ok.mean() > 0.5
    d, b = polar(scene, X[ok])
    d2, b2 = polar(scene, Xp[ok])
    shift = (b2 - b + 180) % 360 - 180
    assert np.abs(shift + theta).max() * np.pi / 180 * d.max() < CM
    assert np.abs(d2 - d).max() < CM


# --------------------------------------------------------------------------- 3. tilt
@pytest.mark.parametrize("t", [2.0, 5.0, 10.0])
def test_tilt_pulls_in_and_throws_out_along_the_view(t):
    h, t0, yaw = 1.5, 2.0, 30.0
    scene = phone_scene(C=(0, 0, h), yaw=yaw, tilt=t0, roll=0.0, R_max=1e4)
    zero = pose(PHONE0)
    right = PHONE0[:, 0]
    dists = np.array([3, 4, 5, 6, 7, 8, 10, 15, 25, 40, 60, 90, 150, 300, 1000.0])
    X = ground_points(scene, np.full(len(dists), yaw), dists)
    dep = np.arctan(h / dists)
    for sign, name in ((+1, "up"), (-1, "down")):
        Q = turn(right, sign * t) @ PHONE0
        assert (Q[2, 2] > PHONE0[2, 2]) == (sign > 0)              # up raises the forward
        Xp, st = lp.project(scene, lp.pose_to_camera(scene, pose(Q), zero), X)
        # in view: the point's angle below the new view axis within the half-height of the frame
        half = np.arctan(scene.H / 2 / scene.K[1, 1])
        below = dep - np.radians(t0) + sign * np.radians(t)
        inview = np.abs(below) < half - 1e-3
        assert (st[~inview] == lp.OUT_OF_VIEW).all()
        a = dep + sign * np.radians(t)
        past = a <= 0
        assert (st[inview & past] == lp.PAST_HORIZON).all()
        ok = inview & ~past
        assert (st[ok] == lp.DRAWN).all() and ok.sum() >= 3, name
        pred = h / np.tan(a[ok])
        d2, b2 = polar(scene, Xp[ok])
        assert np.abs(d2 - pred).max() < CM, name
        assert np.abs((b2 - yaw + 180) % 360 - 180).max() < 1e-6
    # thrown past R_max: lost as well
    scene2 = phone_scene(C=(0, 0, h), yaw=yaw, tilt=t0, roll=0.0, R_max=100.0)
    Q = turn(right, -t) @ PHONE0
    Xp, st = lp.project(scene2, lp.pose_to_camera(scene2, pose(Q), zero), X)
    a = dep - np.radians(t)
    pred = np.where(a > 0, h / np.tan(np.maximum(a, 1e-9)), np.inf)
    inview = np.abs(dep - np.radians(t0) - np.radians(t)) < np.arctan(scene.H / 2 / scene.K[1, 1]) - 1e-3
    assert (st[inview & (pred > 100.5)] == lp.PAST_HORIZON).all()
    assert (st[inview & (pred < 99.5)] == lp.DRAWN).all()


# --------------------------------------------------------------------------- 4. raise and step
def test_raise_and_lower_scale_the_plan_about_the_camera():
    h = 1.5
    scene = phone_scene(C=(2, 1, h), yaw=-60.0, tilt=6.0, roll=0.4, R_max=1e4)
    p0 = np.array([3.0, -2.0, 1.2])
    zero = pose(PHONE0, p=p0)
    rng = np.random.default_rng(4)
    X = ground_points(scene, -60 + rng.uniform(-25, 25, 400), rng.uniform(4, 60, 400))
    for h2, scale in ((3.0, 1.0), (6.0, 2.0), (1.0, 1.0), (2.5, 0.5)):
        src = pose(PHONE0, p=p0 + [0, 0, (h2 - h) / scale])
        cam = lp.pose_to_camera(scene, src, zero, scale=scale)
        assert np.isclose(cam.C[2], h2)
        Xp, st = lp.project(scene, cam, X)
        ok = st == lp.DRAWN
        assert ok.mean() > 0.4
        pred = scene.C[:2] + (X[ok, :2] - scene.C[:2]) * h / h2      # d' = d h / h', same bearing
        assert np.linalg.norm(Xp[ok, :2] - pred, axis=1).max() < CM


def test_a_step_shifts_the_plan_the_other_way():
    scene = phone_scene(C=(0, 0, 1.5), yaw=110.0, tilt=5.0, roll=0.0, R_max=1e4)
    p0 = np.array([0.5, 0.5, 1.0])
    zero = pose(PHONE0, p=p0)
    fwd_G = PHONE0[:, 2].copy()
    fwd_G[2] = 0
    fwd_G /= np.linalg.norm(fwd_G)
    right_G = np.array([fwd_G[1], -fwd_G[0], 0.0])
    yaw = np.radians(110.0)
    fwd = np.array([np.sin(yaw), np.cos(yaw), 0.0])
    right = np.array([np.cos(yaw), -np.sin(yaw), 0.0])
    rng = np.random.default_rng(5)
    X = ground_points(scene, 110 + rng.uniform(-25, 25, 400), rng.uniform(6, 60, 400))
    for s, dirG, dirS in ((3.0, fwd_G, fwd), (-2.0, fwd_G, fwd), (1.5, right_G, right)):
        cam = lp.pose_to_camera(scene, pose(PHONE0, p=p0 + s * dirG), zero)
        assert np.allclose(cam.C, scene.C + s * dirS)
        Xp, st = lp.project(scene, cam, X)
        ok = st == lp.DRAWN
        assert ok.mean() > 0.4
        assert np.linalg.norm(Xp[ok] - (X[ok] - s * dirS), axis=1).max() < CM
    # no position from the source: the camera stays where it stood
    cam = lp.pose_to_camera(scene, pose(PHONE0, p=None), zero)
    assert np.allclose(cam.C, scene.C)


# --------------------------------------------------------------------------- 5. roll
@pytest.mark.parametrize("r", [2.0, 5.0])
def test_roll_skews_the_plan(r):
    yaw, tilt = 30.0, 4.0
    scene = phone_scene(C=(0, 0, 1.5), yaw=yaw, tilt=tilt, roll=0.0, R_max=1e4)
    phone = cam_R_wc(200.0, tilt)                      # held like the camera: a pure roll
    zero = pose(phone)
    beta = np.array([5.0, 10.0, 15.0, 20.0])
    dists = np.array([8.0, 15.0, 25.0, 40.0])
    B, D = np.meshgrid(beta, dists)
    left = ground_points(scene, yaw - B.ravel(), D.ravel())
    right = ground_points(scene, yaw + B.ravel(), D.ravel())

    def drawn(deg):
        Q = turn(phone[:, 2], deg) @ phone                    # about the forward: right side down
        cam = lp.pose_to_camera(scene, pose(Q), zero)
        (pl, sl), (pr, sr) = lp.project(scene, cam, left), lp.project(scene, cam, right)
        assert (sl == lp.DRAWN).all() and (sr == lp.DRAWN).all()
        return pl, pr

    pl, pr = drawn(+r)
    dl, _ = polar(scene, pl)
    dr, _ = polar(scene, pr)
    d = D.ravel()
    assert (dl < d).all() and (dr > d).all()                     # left pulled in, right thrown out
    ml, mr = drawn(-r)                                           # the mirror image
    mirror = np.diag([-1.0, 1.0])
    yr = np.radians(yaw)
    axes = np.column_stack([[np.cos(yr), -np.sin(yr)], [np.sin(yr), np.cos(yr)]])   # right, forward

    def to_cam(P):
        return (P[:, :2] - scene.C[:2]) @ axes

    assert np.linalg.norm(to_cam(ml) - to_cam(pr) @ mirror, axis=1).max() < CM
    assert np.linalg.norm(to_cam(mr) - to_cam(pl) @ mirror, axis=1).max() < CM


# --------------------------------------------------------------------------- 6. out of view
def test_out_of_view_and_behind_are_lost():
    yaw = 30.0
    scene = phone_scene(C=(0, 0, 1.5), yaw=yaw, tilt=3.0, R_max=1e4)
    zero = pose(PHONE0)
    X = ground_points(scene, [yaw, yaw + 50, yaw - 50, yaw + 180, yaw - 20, yaw + 20], [20] * 6)
    Xp, st = lp.project(scene, lp.start_camera(scene), X)
    assert list(st) == [lp.DRAWN, lp.OUT_OF_VIEW, lp.OUT_OF_VIEW, lp.BEHIND, lp.DRAWN, lp.DRAWN]
    assert np.isnan(Xp[1:4]).all() and np.isfinite(Xp[[0, 4, 5]]).all()
    cam = lp.pose_to_camera(scene, pose(turn([0, 0, 1], -46.0) @ PHONE0), zero)   # pan right 46
    Xp, st = lp.project(scene, cam, X)
    assert st[4] == lp.OUT_OF_VIEW and np.isnan(Xp[4]).all()   # the left of the start view: lost
    assert st[1] == lp.DRAWN                                   # the right comes into view
    assert np.isnan(Xp[st != lp.DRAWN]).all()


# --------------------------------------------------------------------------- 7. a filmed clip
def scene_from_homography(h, R_max, pad=20.0):
    """A scene (as the export writes it) from a calibration: K R_cw C from P, the DTM on 1 m cells."""
    from scipy.linalg import rq

    P, _, C = h._rays()
    K, R = rq(P[:, :3])
    S = np.diag(np.sign(np.diag(K)))
    K, R = K @ S, S @ R
    assert np.linalg.det(R) > 0
    K = K / K[2, 2]
    meta = {"version": 1, "image_size": list(h.image_size), "R_max": R_max,
            "origin": {"E": h.origin[0], "N": h.origin[1], "datum_z": h.ground_z_m},
            "camera": {"P": P.tolist(), "K": K.tolist(), "R_cw": R.tolist(), "C": C.tolist()},
            "terrain": None}
    heights = None
    if h.terrain is not None:
        e = R_max + pad
        x0, x1 = np.floor(C[0] - e), np.ceil(C[0] + e)
        y0, y1 = np.floor(C[1] - e), np.ceil(C[1] + e)
        nx, ny = int(x1 - x0), int(y1 - y0)
        X, Y = np.meshgrid(x0 + 0.5 + np.arange(nx), y1 - 0.5 - np.arange(ny))
        z = h.terrain.height(np.column_stack([X.ravel() + h.origin[0], Y.ravel() + h.origin[1]]))
        z = np.where(np.isfinite(z), z - h.ground_z_m, 0.0)
        meta["terrain"] = {"file": "terrain.f32", "nx": nx, "ny": ny, "x0": float(x0), "y1": float(y1),
                           "cell": 1.0}
        heights = z.astype(np.float32)
    return meta, heights


def clip_rotations(h, reg):
    """Per registered frame, the camera's turn in the start camera (rp_drag_plans.Clip): K^-1 H K
    made a rotation, smoothed over time (median 9, then mean 15 frames)."""
    import pandas as pd

    cp = h.camera_params
    W, H = h.image_size
    K = np.array([[cp["focal_px"], 0, W / 2], [0, cp["focal_px"], H / 2], [0, 0, 1.0]])
    Kinv = np.linalg.inv(K)
    rots = {}
    for k, Hk in reg.items():
        d = np.linalg.det(Hk)
        if not np.isfinite(d) or d == 0:
            continue
        U, _, Vt = np.linalg.svd(Kinv @ (Hk / np.cbrt(d)) @ K)
        rots[int(k)] = U @ Vt
    frames = np.array(sorted(rots))
    rv = Rot.from_matrix(np.stack([rots[k] for k in frames])).as_rotvec()
    rv = pd.DataFrame(rv).rolling(9, center=True, min_periods=1).median() \
        .rolling(15, center=True, min_periods=1).mean().to_numpy(copy=True)
    return dict(zip(frames.tolist(), Rot.from_rotvec(rv).as_matrix())), \
        dict(zip(frames.tolist(), (rots[k] for k in frames)))


def replay_6432(smoothed=True):
    import pandas as pd

    from groundtrack.homography import Homography
    from groundtrack.registration import load_registration

    cal = REPO / "calibration" / "richmond6432_homography.json"
    reg_path = REPO / "runs" / "richmond6432" / "moved" / "_working" / "registration.npz"
    pts = RICHMOND / "6432_points.csv"
    for p in (cal, reg_path, pts):
        if not p.exists():
            pytest.skip(f"{p} is missing (gitignored / outside the repo)")
    h = Homography.load(cal)
    try:
        h.terrain
    except FileNotFoundError:
        pytest.skip("the Richmond DTM is missing")
    meta, heights = scene_from_homography(h, R_max=150.0)       # rp_compare's MAX_M
    scene = lp.prepare_scene(meta, heights)
    reg, _, _ = load_registration(reg_path)
    smooth, raw = clip_rotations(h, reg)
    rots = smooth if smoothed else raw
    P = pd.read_csv(pts)
    o = np.asarray(h.origin)
    G = turn([0, 0, 1], 37.0)                                # the source's frame: another heading
    zero = pose(G @ scene.R_wc)
    Xp = np.full((len(P), 3), np.nan)
    st = np.full(len(P), 255, np.uint8)
    for k, idx in P.groupby("frame").indices.items():
        if int(k) not in rots:
            continue
        cam = lp.pose_to_camera(scene, pose(G @ scene.R_wc @ rots[int(k)]), zero)
        x, y = P.E_stab.to_numpy()[idx] - o[0], P.N_stab.to_numpy()[idx] - o[1]
        X = np.column_stack([x, y, lp.ground_height(scene, x, y)])
        Xp[idx], st[idx] = lp.project(scene, cam, X)
    fixed = np.column_stack([P.E_fixed - o[0], P.N_fixed - o[1]])
    return P, Xp, st, fixed


def test_richmond_6432_replay_matches_the_fixed_plan():
    P, Xp, st, fixed = replay_6432(smoothed=True)
    both = (~P.lost.to_numpy()) & (st == lp.DRAWN)
    gap = np.linalg.norm(Xp[both, :2] - fixed[both], axis=1)
    lost_agree = ((st != lp.DRAWN) == P.lost.to_numpy()).mean()
    print(f"\n6432 replay (smoothed turns): {both.sum()} of {len(P)} positions drawn by both, "
          f"median gap {np.median(gap):.3f} m, p90 {np.percentile(gap, 90):.3f} m, "
          f"lost / drawn agree {100 * lost_agree:.1f} %")
    assert both.sum() > 0.5 * len(P)
    assert np.median(gap) < 0.5


def test_richmond_6432_replay_unsmoothed_is_the_fixed_plan():
    """With the registration's own turns the drawn positions are rp_compare's fixed ones: the
    same frames and signs. What is left is the registration's non-rotational part."""
    P, Xp, st, fixed = replay_6432(smoothed=False)
    both = (~P.lost.to_numpy()) & (st == lp.DRAWN)
    gap = np.linalg.norm(Xp[both, :2] - fixed[both], axis=1)
    print(f"\n6432 replay (raw turns): median gap {np.median(gap):.3f} m, "
          f"p90 {np.percentile(gap, 90):.3f} m")
    assert np.median(gap) < 0.5


# --------------------------------------------------------------------------- speed and colour
def test_speed_as_drawn_is_rp_drag_plans_speed():
    import pandas as pd

    rng = np.random.default_rng(8)
    t = np.arange(40) / 30.0
    xy = np.column_stack([1.3 * t, 0.2 * t]) + rng.normal(0, 0.05, (40, 2))
    xy[[5, 6, 20]] = np.nan
    Xp = np.column_stack([xy, np.zeros(40)])
    v = lp.speed_as_drawn(Xp, t, np.zeros(40, int))
    s = pd.DataFrame(xy, columns=["x", "y"]).rolling(7, center=True, min_periods=2).mean().to_numpy()
    dt = np.gradient(t)
    ref = np.nan_to_num(np.hypot(np.gradient(s[:, 0]), np.gradient(s[:, 1])) / np.where(dt > 0, dt, np.nan))
    assert np.allclose(v, ref)
    c = lp.ramp_color([0.5, 1.0, 6.3, 40.0, 400.0])
    assert np.allclose(c[0], c[1]) and np.allclose(c[3], c[4])


# --------------------------------------------------------------------------- 8. JS = Python
RUNNER = r"""
import { readFileSync, writeFileSync } from "node:fs";
import * as L from "./projection.mjs";
const dir = process.argv[2];
const rd = (f, T) => { const b = readFileSync(dir + "/" + f); return new T(Uint8Array.from(b).buffer); };
const wr = (f, a) => writeFileSync(dir + "/" + f, Buffer.from(a.buffer));
const job = JSON.parse(readFileSync(dir + "/job.json", "utf8"));
const scene = L.prepareScene(job.meta, job.terrain ? rd("terrain.f32", Float32Array) : null);
const X = rd("points.f32", Float32Array), n = job.n_points;
const cams = new Float64Array(12 * job.poses.length), xp = new Float64Array(X.length);
const st = new Uint8Array(X.length / 3);
const cam = { R_cw: new Float64Array(9), R_wc: new Float64Array(9), C: new Float64Array(3) };
const out = { X: xp, state: st };
job.poses.forEach((ps, k) => {
  L.poseToCamera(scene, ps.src, ps.zero, ps.gain, ps.scale, { out: cam });
  cams.set(cam.R_cw, 12 * k); cams.set(cam.C, 12 * k + 9);
  L.project(scene, cam, X, out, k * n, (k + 1) * n);
});
wr("cams.f64", cams); wr("xp.f64", xp); wr("state.u8", st);
const plain = L.prepareScene(job.meta, job.terrain ? rd("terrain.f32", Float32Array) : null, { skip: false });
const outP = { X: new Float64Array(X.length), state: new Uint8Array(X.length / 3) };
job.poses.forEach((ps, k) => {
  L.poseToCamera(plain, ps.src, ps.zero, ps.gain, ps.scale, { out: cam });
  L.project(plain, cam, X, outP, k * n, (k + 1) * n);
});
wr("xp_plain.f64", outP.X); wr("state_plain.u8", outP.state);
const outH = { X: new Float64Array(X.length), state: new Uint8Array(X.length / 3) };
job.poses.forEach((ps, k) => {
  L.poseToCamera(scene, ps.src, ps.zero, ps.gain, ps.scale, { out: cam });
  L.project(scene, cam, X, outH, k * n, (k + 1) * n, "hinge");
});
wr("xp_hinge.f64", outH.X); wr("state_hinge.u8", outH.state);
const sm = new L.PoseSmoother(job.smooth.min_cutoff, job.smooth.beta, job.smooth.d_cutoff);
const sc = new Float64Array(12 * job.smooth.poses.length);
job.smooth.poses.forEach((ps, k) => {
  L.poseToCamera(scene, ps.src, ps.zero, ps.gain, ps.scale, { out: cam, smoother: sm });
  sc.set(cam.R_cw, 12 * k); sc.set(cam.C, 12 * k + 9);
});
wr("smooth.f64", sc);
const tid = rd("tid.u32", Uint32Array), tt = rd("t.f32", Float32Array), tx = rd("txp.f64", Float64Array);
wr("speed.f64", L.speedAsDrawn(tx, tt, L.trackIndex(tid)));
wr("speed3.f64", L.speedAsDrawn(tx, tt, L.trackIndex(tid), null, 7, 3));
const rc = new Float32Array(3 * job.ramp.length);
job.ramp.forEach((v, i) => L.rampColor(v, rc, 3 * i));
wr("ramp.f32", rc);
wr("footprint.f64", L.footprint(scene));
wr("drag.f64", L.dragGrid(scene));
"""


def random_case(rng, scene, n_poses, n_pts):
    """Random phone poses (another heading, any tilt, turns up to ~60 deg, moves of a few m, gains,
    scales) and ground points across the start view."""
    poses, pts = [], []
    yaw = np.degrees(lp.heading(scene.R_wc))
    for _ in range(n_poses):
        Q0 = cam_R_wc(rng.uniform(0, 360), rng.uniform(-10, 30), rng.uniform(-5, 5))
        Q = Rot.from_rotvec(rng.normal(0, 0.35, 3)).as_matrix() @ Q0
        p0 = rng.uniform(-2, 2, 3)
        has_p = rng.random() < 0.7
        poses.append({"src": pose(Q, p=p0 + rng.normal(0, 1.5, 3) if has_p else None),
                      "zero": pose(Q0, p=p0 if has_p else None),
                      "gain": float(rng.uniform(0.25, 3.0)), "scale": float(rng.uniform(0.5, 3.0))})
        b, d = yaw + rng.uniform(-50, 50, n_pts), rng.uniform(1, 170, n_pts)
        pts.append(ground_points(scene, b, d))
    pts = np.vstack(pts).astype(np.float32)
    pts[rng.random(len(pts)) < 0.01] = np.nan          # missing coordinates: behind
    return poses, pts


@pytest.mark.parametrize("kind", ["bumpy", "pit", "flat"])
def test_js_equals_python(tmp_path, kind):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    meta, heights = terrain_meta(kind)
    scene = lp.prepare_scene(meta, heights)
    rng = np.random.default_rng({"bumpy": 11, "flat": 12, "pit": 13}[kind])
    n_poses, n_pts = 1000, 30
    poses, pts = random_case(rng, scene, n_poses, n_pts)
    # a smoothed sequence: one zero, a wobbling turn and move at 60 Hz
    Q0 = cam_R_wc(150.0, 8.0)
    seq = []
    for i in range(240):
        t = i / 60.0
        Q = Rot.from_rotvec([0.3 * np.sin(t), 0.1 * np.sin(3 * t) + rng.normal(0, 0.01), 0.2 * t]).as_matrix() @ Q0
        seq.append({"src": pose(Q, p=[np.sin(t), 0.5 * t, rng.normal(0, 0.02)] if i % 50 < 40 else None, t=t),
                    "zero": pose(Q0, p=[0, 0, 0]), "gain": 1.5, "scale": 2.0})
    smooth = {"min_cutoff": 0.8, "beta": 0.3, "d_cutoff": 1.0, "poses": seq}
    # tracks drawn: a few tracks, interleaved rows, lost rows
    m = 600
    tid = rng.integers(1, 9, m).astype(np.uint32)
    tt = np.sort(rng.uniform(0, 20, m)).astype(np.float32)
    txp = np.column_stack([np.cumsum(rng.normal(0, 0.3, m)), np.cumsum(rng.normal(0, 0.3, m)),
                           np.cumsum(rng.normal(0, 0.1, m))])
    txp[rng.random(m) < 0.1] = np.nan
    ramp = [0.0, 0.3, 1.0, 1.7, 4.2, 9.9, 25.0, 39.99, 40.0, 1e3]

    (tmp_path / "projection.mjs").write_bytes(JS.read_bytes())
    (tmp_path / "runner.mjs").write_text(RUNNER, encoding="utf-8")
    job = {"meta": meta, "terrain": heights is not None, "poses": poses, "n_points": n_pts,
           "smooth": smooth, "ramp": ramp}
    (tmp_path / "job.json").write_text(json.dumps(job), encoding="utf-8")
    if heights is not None:
        heights.astype("<f4").tofile(tmp_path / "terrain.f32")
    pts.astype("<f4").tofile(tmp_path / "points.f32")
    tid.astype("<u4").tofile(tmp_path / "tid.u32")
    tt.astype("<f4").tofile(tmp_path / "t.f32")
    txp.astype("<f8").tofile(tmp_path / "txp.f64")
    r = subprocess.run([node, "runner.mjs", str(tmp_path)], cwd=tmp_path, capture_output=True, text=True,
                       timeout=300)
    assert r.returncode == 0, r.stderr

    X = pts.astype(float).reshape(n_poses, n_pts, 3)
    cams_js = np.fromfile(tmp_path / "cams.f64").reshape(n_poses, 12)
    xp_js = np.fromfile(tmp_path / "xp.f64").reshape(n_poses, n_pts, 3)
    st_js = np.fromfile(tmp_path / "state.u8", np.uint8).reshape(n_poses, n_pts)
    xp_py = np.full_like(xp_js, np.nan)
    st_py = np.zeros_like(st_js)
    cam_d = 0.0
    for k, ps in enumerate(poses):
        cam = lp.pose_to_camera(scene, ps["src"], ps["zero"], ps["gain"], ps["scale"])
        cam_d = max(cam_d, np.abs(cam.R_cw.ravel() - cams_js[k, :9]).max(), np.abs(cam.C - cams_js[k, 9:]).max())
        xp_py[k], st_py[k] = lp.project(scene, cam, X[k])
    same = st_js == st_py
    drawn = same & (st_py == lp.DRAWN)
    d = np.linalg.norm(xp_js[drawn] - xp_py[drawn], axis=1)
    print(f"\nJS = Python ({kind}): {n_poses} poses x {n_pts} points, "
          f"{drawn.sum()} drawn, states differ {(~same).sum()}, max |X'| difference {d.max() * 1000:.2e} mm, "
          f"max camera difference {cam_d:.2e}")
    assert cam_d < 1e-9
    assert (~same).sum() <= 2                     # only a point on a frame edge or a grazing ray
    assert d.max() < CM
    assert np.isnan(xp_js[st_js != lp.DRAWN]).all()
    # JS: the skip = the plain march
    xp_pl = np.fromfile(tmp_path / "xp_plain.f64").reshape(n_poses, n_pts, 3)
    st_pl = np.fromfile(tmp_path / "state_plain.u8", np.uint8).reshape(n_poses, n_pts)
    assert (st_pl == st_js).all()
    dd = np.linalg.norm(xp_pl[st_pl == lp.DRAWN, :2] - xp_js[st_js == lp.DRAWN, :2], axis=1)
    print(f"JS skip = JS plain march ({kind}): states equal, max |X'| difference {dd.max() * 1000:.2e} mm")
    assert dd.max() <= BISECTED
    # hinge
    xp_h = np.fromfile(tmp_path / "xp_hinge.f64").reshape(n_poses, n_pts, 3)
    st_h = np.fromfile(tmp_path / "state_hinge.u8", np.uint8).reshape(n_poses, n_pts)
    xh_py = np.full_like(xp_h, np.nan)
    sh_py = np.zeros_like(st_h)
    for k, ps in enumerate(poses):
        cam = lp.pose_to_camera(scene, ps["src"], ps["zero"], ps["gain"], ps["scale"])
        xh_py[k], sh_py[k] = lp.project(scene, cam, X[k], mode="hinge")
    assert (st_h == sh_py).all()
    dh = np.linalg.norm(xp_h[st_h == lp.DRAWN] - xh_py[sh_py == lp.DRAWN], axis=1)
    print(f"JS = Python, hinge ({kind}): {(st_h == lp.DRAWN).sum()} drawn, states equal, "
          f"max |X'| difference {dh.max() * 1000:.2e} mm")
    assert dh.max() < CM and np.isnan(xp_h[st_h != lp.DRAWN]).all()

    sm = lp.PoseSmoother(smooth["min_cutoff"], smooth["beta"], smooth["d_cutoff"])
    sc = np.fromfile(tmp_path / "smooth.f64").reshape(-1, 12)
    for k, ps in enumerate(seq):
        cam = lp.pose_to_camera(scene, ps["src"], ps["zero"], ps["gain"], ps["scale"], smoother=sm)
        assert np.abs(cam.R_cw.ravel() - sc[k, :9]).max() < 1e-9
        assert np.abs(cam.C - sc[k, 9:]).max() < 1e-9

    v_js = np.fromfile(tmp_path / "speed.f64")
    v_py = lp.speed_as_drawn(txp, tt.astype(float), tid)
    assert np.allclose(v_js, v_py, rtol=1e-9, atol=1e-9)
    v3_js = np.fromfile(tmp_path / "speed3.f64")
    assert np.allclose(v3_js, lp.speed_as_drawn(txp, tt.astype(float), tid, dims=3), rtol=1e-9, atol=1e-9)
    assert (v3_js >= v_js - 1e-9).all() and (v3_js > v_js + 1e-6).any()
    c_js = np.fromfile(tmp_path / "ramp.f32", np.float32).reshape(-1, 3)
    assert np.abs(c_js - lp.ramp_color(ramp)).max() < 1 / 255
    fp_js = np.fromfile(tmp_path / "footprint.f64").reshape(-1, 3)
    assert np.abs(fp_js - lp.footprint(scene)).max() < CM
    g_js = np.fromfile(tmp_path / "drag.f64").reshape(-1, 3)
    g_py = lp.drag_grid(scene)
    assert len(g_py) >= 10 and g_js.shape == g_py.shape and np.abs(g_js - g_py).max() < CM


# --------------------------------------------------------------------------- the skip = the plain march
@pytest.mark.parametrize("kind", ["bumpy", "pit"])
def test_skip_equals_the_plain_march(kind):
    """Jumping the tiles whose ground is certainly below the ray changes nothing: the same states,
    X' within the bisection's resolution, on the random cases and on a fan of grazing rays."""
    meta, heights = terrain_meta(kind)
    fast, plain = lp.prepare_scene(meta, heights), lp.prepare_scene(meta, heights, skip=False)
    rng = np.random.default_rng({"bumpy": 21, "pit": 22}[kind])
    poses, pts = random_case(rng, fast, 300, 30)          # (JS: both marches on 1000 poses, below)
    X = pts.astype(float).reshape(300, 30, 3)
    worst, deep = 0.0, 0

    def in_pit(P):         # on the pit's far wall, below any of bumpy()'s ground (-1.2 m)
        return int(((np.abs(P[:, 0] + 14) < 10.6) & (np.abs(P[:, 1] - 20) < 10.6) & (P[:, 2] < -1.5)).sum())

    for k, ps in enumerate(poses):
        cam = lp.pose_to_camera(fast, ps["src"], ps["zero"], ps["gain"], ps["scale"])
        a, sa = lp.project(fast, cam, X[k])
        b, sb = lp.project(plain, cam, X[k])
        assert (sa == sb).all()
        ok = sa == lp.DRAWN
        if ok.any():
            worst = max(worst, np.linalg.norm(a[ok, :2] - b[ok, :2], axis=1).max())
            deep += in_pit(a[ok])
    # rays from the start camera at every depression from grazing to steep, across the view
    yaw = lp.heading(fast.R_wc)
    B, E = np.meshgrid(yaw + np.radians(np.linspace(-45, 45, 181)), -np.radians(np.geomspace(0.05, 40, 120)))
    D = np.column_stack([np.sin(B.ravel()) * np.cos(E.ravel()), np.cos(B.ravel()) * np.cos(E.ravel()),
                         np.sin(E.ravel())])
    ha, hb = lp.raycast(fast, fast.C, D), lp.raycast(plain, plain.C, D)
    assert (np.isfinite(ha[:, 0]) == np.isfinite(hb[:, 0])).all()
    hit = np.isfinite(ha[:, 0])
    fan = np.linalg.norm(ha[hit, :2] - hb[hit, :2], axis=1).max()
    deep += in_pit(ha[hit])
    print(f"\nskip = plain march ({kind}): states equal, max |X'| difference {max(worst, fan) * 1000:.2e} mm "
          f"({hit.sum()} of {len(D)} fan rays hit; {deep} hits in the pit)")
    assert worst <= BISECTED and fan <= BISECTED
    if kind == "pit":
        assert deep > 50


POOL_SHIM = r"""
import { parentPort, workerData } from "node:worker_threads";
globalThis.self = { postMessage: (m, t) => parentPort.postMessage(m, t) };
await import(workerData);
parentPort.on("message", (d) => self.onmessage({ data: d }));
"""

POOL_RUNNER = r"""
import { readFileSync } from "node:fs";
import { Worker } from "node:worker_threads";
import { pathToFileURL } from "node:url";
const [dir, live] = process.argv.slice(2);
const L = await import(pathToFileURL(live + "/projection.js").href);
const P = await import(pathToFileURL(live + "/projection_worker.js").href);
const shim = pathToFileURL(dir + "/shim.mjs"), file = pathToFileURL(live + "/projection_worker.js").href;
const spawn = () => {
  const w = new Worker(shim, { workerData: file });
  return { postMessage: (m, t) => w.postMessage(m, t), set onmessage(f) { w.on("message", (d) => f({ data: d })); },
           terminate: () => w.terminate() };
};
const rd = (f, T) => { const b = readFileSync(dir + "/" + f); return new T(Uint8Array.from(b).buffer); };
const job = JSON.parse(readFileSync(dir + "/job.json", "utf8"));
const scene = L.prepareScene(job.meta, rd("terrain.f32", Float32Array));
const X = rd("points.f32", Float32Array), n = X.length / 3;
const same = (a, b) => a === b || (a !== a && b !== b);
const res = {};
for (const [shared, mode] of [[true, "ground"], [false, "ground"], [true, "hinge"], [false, "hinge"]]) {
  const pool = P.createProjectionPool(scene, 3, mode === "hinge" ? { shared, spawn, mode } : { shared, spawn });
  let differ = 0, drawn = 0;
  const mine = { X: new Float32Array(3 * n), state: new Uint8Array(n) };
  for (const [k, ps] of job.poses.entries()) {
    const cam = L.poseToCamera(scene, ps.src, ps.zero, ps.gain, ps.scale);
    const ref = L.project(scene, cam, X, { X: new Float32Array(3 * n), state: new Uint8Array(n) }, 0, -1, mode);
    const got = await pool.project(cam, X, k % 2 ? mine : null);
    for (let i = 0; i < n; i++) {
      if (got.state[i] !== ref.state[i]) differ++;
      drawn += ref.state[i] === 0;
      for (let c = 0; c < 3; c++) if (!same(got.X[3 * i + c], ref.X[3 * i + c])) differ++;
    }
  }
  pool.close();
  res[(shared ? "shared" : "transfer") + " " + mode] = { differ, drawn };
}
console.log(JSON.stringify(res));
"""


def test_worker_pool_equals_one_thread(tmp_path):
    """projection_worker.js (node worker_threads standing in for browser workers): shared and
    transferred results are exactly those of one thread."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    meta, heights = terrain_meta("pit")
    scene = lp.prepare_scene(meta, heights)
    poses, pts = random_case(np.random.default_rng(31), scene, 6, 3000)
    pts = pts.reshape(6, -1, 3)[0]                        # one set of points, six poses
    (tmp_path / "shim.mjs").write_text(POOL_SHIM, encoding="utf-8")
    (tmp_path / "pool.mjs").write_text(POOL_RUNNER, encoding="utf-8")
    (tmp_path / "job.json").write_text(json.dumps({"meta": meta, "poses": poses}), encoding="utf-8")
    heights.astype("<f4").tofile(tmp_path / "terrain.f32")
    pts.astype("<f4").tofile(tmp_path / "points.f32")
    r = subprocess.run([node, "pool.mjs", str(tmp_path), str(LIVE)], cwd=tmp_path, capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    res = json.loads(r.stdout.strip().splitlines()[-1])
    print(f"\nworker pool (3 workers, 6 poses x {len(pts)} points): {res}")
    assert len(res) == 4
    for key, r_ in res.items():
        assert r_["differ"] == 0 and r_["drawn"] > 1000, key


# --------------------------------------------------------------------------- the hinge mode
def elevation(scene, X):
    e = X - scene.C
    return np.degrees(np.arctan2(e[:, 2], np.hypot(e[:, 0], e[:, 1])))


@pytest.mark.parametrize("terrain", [False, True])
def test_hinge_identity_draws_every_seen_point_where_it_is(terrain):
    meta, heights = make_meta(C=(5.0, -3.0, 8.0 if terrain else 1.6), yaw=40.0, tilt=3.0, roll=0.8,
                              R_max=50.0, terrain=bumpy() if terrain else None)
    scene = lp.prepare_scene(meta, heights)
    rng = np.random.default_rng(41)
    X = ground_points(scene, 40 + rng.uniform(-30, 30, 2000), rng.uniform(2, 140, 2000))
    zero = pose(PHONE0, p=[1.0, 2.0, 0.3])
    cam = lp.pose_to_camera(scene, zero, zero, gain=2.5, scale=3.0)
    Xp, st = lp.project(scene, cam, X, mode="hinge")
    _, st_g = lp.project(scene, cam, X)
    assert (st == np.where(st_g == lp.PAST_HORIZON, lp.DRAWN, st_g)).all()   # beyond R_max: drawn too
    ok = st == lp.DRAWN
    assert ok.sum() > 1000 and (np.hypot(*(X[ok, :2] - scene.C[:2]).T) > 50).any()
    assert np.abs(Xp[ok] - X[ok]).max() < 1e-9
    assert np.isnan(Xp[~ok]).all()


@pytest.mark.parametrize("theta", [10.0, -30.0, 46.0])
def test_hinge_pan_swirls_like_the_ground_mode(theta):
    scene = phone_scene(yaw=30.0, tilt=4.0, roll=0.0, R_max=1e4)
    cam = lp.pose_to_camera(scene, pose(turn([0, 0, 1], -theta) @ PHONE0), pose(PHONE0))
    rng = np.random.default_rng(42)
    X = ground_points(scene, 30 + theta + rng.uniform(-25, 25, 500), rng.uniform(3, 80, 500))
    Xp, st = lp.project(scene, cam, X, mode="hinge")
    Xg, sg = lp.project(scene, cam, X)
    ok = st == lp.DRAWN
    assert ok.mean() > 0.5 and (sg[ok] == lp.DRAWN).all()
    d, b = polar(scene, X[ok])
    d2, b2 = polar(scene, Xp[ok])
    assert np.abs((b2 - b + 180) % 360 - 180 + theta).max() * np.pi / 180 * d.max() < 1e-6
    assert np.abs(d2 - d).max() < 1e-9 and np.abs(Xp[ok, 2] - X[ok, 2]).max() < 1e-9
    assert np.linalg.norm(Xp[ok] - Xg[ok], axis=1).max() < CM          # flat ground: as drawn on it


@pytest.mark.parametrize("t", [2.0, 5.0, 10.0])
def test_hinge_tilt_swings_the_plan_like_a_drawbridge(t):
    h, t0, yaw = 1.5, 2.0, 30.0
    scene = phone_scene(C=(0, 0, h), yaw=yaw, tilt=t0, roll=0.0, R_max=1e4)
    zero = pose(PHONE0)
    dists = np.array([3, 4, 5, 6, 8, 10, 15, 25, 40, 60, 90, 150.0])
    X = ground_points(scene, np.full(len(dists), yaw), dists)
    for sign, name in ((-1, "down"), (+1, "up")):
        Q = turn(PHONE0[:, 0], sign * t) @ PHONE0
        Xp, st = lp.project(scene, lp.pose_to_camera(scene, pose(Q), zero), X, mode="hinge")
        ok = st == lp.DRAWN
        assert ok.sum() >= 3 and (st[~ok] == lp.OUT_OF_VIEW).all(), name
        # tilt down: the plan swings up about the camera (the points in front rise); up: down
        assert np.abs(elevation(scene, Xp[ok]) - (elevation(scene, X[ok]) - sign * t)).max() < 1e-9, name
        assert (np.sign(Xp[ok, 2] - X[ok, 2]) == -sign).all(), name
        assert np.abs(np.linalg.norm(Xp[ok] - scene.C, axis=1) - np.linalg.norm(X[ok] - scene.C, axis=1)).max() < 1e-9
        _, b2 = polar(scene, Xp[ok])
        assert np.abs((b2 - yaw + 180) % 360 - 180).max() < 1e-6


def test_hinge_roll_banks_the_plan():
    yaw, tilt, r = 30.0, 4.0, 5.0
    scene = phone_scene(C=(0, 0, 1.5), yaw=yaw, tilt=tilt, roll=0.0, R_max=1e4)
    phone = cam_R_wc(200.0, tilt)
    B, D = np.meshgrid([5.0, 10.0, 15.0, 20.0], [8.0, 15.0, 25.0, 40.0])
    left = ground_points(scene, yaw - B.ravel(), D.ravel())
    right = ground_points(scene, yaw + B.ravel(), D.ravel())

    def drawn(deg):
        cam = lp.pose_to_camera(scene, pose(turn(phone[:, 2], deg) @ phone), pose(phone))
        (pl, sl), (pr, sr) = lp.project(scene, cam, left, mode="hinge"), lp.project(scene, cam, right, mode="hinge")
        assert (sl == lp.DRAWN).all() and (sr == lp.DRAWN).all()
        return pl, pr

    pl, pr = drawn(+r)                                   # the camera's right side down
    assert (pr[:, 2] > right[:, 2]).all() and (pl[:, 2] < left[:, 2]).all()    # the plan banks the other way
    ml, mr = drawn(-r)
    yr = np.radians(yaw)
    axes = np.column_stack([[np.cos(yr), -np.sin(yr), 0], [np.sin(yr), np.cos(yr), 0], [0, 0, 1]])

    def to_cam(P):
        return (P - scene.C) @ axes

    mirror = np.array([-1.0, 1.0, 1.0])
    assert np.abs(to_cam(ml) - to_cam(pr) * mirror).max() < 1e-9
    assert np.abs(to_cam(mr) - to_cam(pl) * mirror).max() < 1e-9


def test_hinge_moves_shift_the_plan_and_keep_distances():
    scene = phone_scene(C=(0, 0, 1.5), yaw=110.0, tilt=5.0, roll=0.3, R_max=1e4)
    p0 = np.array([0.5, 0.5, 1.0])
    zero = pose(PHONE0, p=p0)
    rng = np.random.default_rng(43)
    X = ground_points(scene, 110 + rng.uniform(-25, 25, 400), rng.uniform(6, 60, 400))
    for dp in ([0, 0, 2.0], [0, 0, -0.5], 3.0 * PHONE0[:, 2] * [1, 1, 0]):
        cam = lp.pose_to_camera(scene, pose(PHONE0, p=p0 + dp), zero)
        Xp, st = lp.project(scene, cam, X, mode="hinge")
        ok = st == lp.DRAWN
        assert ok.mean() > 0.4
        assert np.abs(Xp[ok] - (X[ok] - (cam.C - scene.C))).max() < 1e-9    # raised: the plan drops
    # any turn and move: the distance from the camera is kept; only what it does not see is lost
    meta, heights = terrain_meta("pit")
    scene = lp.prepare_scene(meta, heights)
    poses, pts = random_case(np.random.default_rng(44), scene, 300, 30)
    X = pts.astype(float).reshape(300, 30, 3)
    for k, ps in enumerate(poses):
        cam = lp.pose_to_camera(scene, ps["src"], ps["zero"], ps["gain"], ps["scale"])
        Xp, st = lp.project(scene, cam, X[k], mode="hinge")
        _, sg = lp.project(scene, cam, X[k])
        assert (st == np.where(sg == lp.PAST_HORIZON, lp.DRAWN, sg)).all()
        assert not (st == lp.PAST_HORIZON).any()
        ok = st == lp.DRAWN
        if ok.any():
            assert np.abs(np.linalg.norm(Xp[ok] - scene.C, axis=1)
                          - np.linalg.norm(X[k, ok] - cam.C, axis=1)).max() < 1e-9
    with pytest.raises(ValueError):
        lp.project(scene, cam, X[0], mode="sky")


def test_speed_as_drawn_in_space():
    import pandas as pd

    t = np.arange(30) / 30.0
    Xp = np.column_stack([np.full(30, 4.0), np.full(30, -2.0), 2.0 * t])     # rising, 2 m/s
    Xp[[7, 8]] = np.nan
    assert np.allclose(lp.speed_as_drawn(Xp, t, np.zeros(30, int)), 0.0)
    v = lp.speed_as_drawn(Xp, t, np.zeros(30, int), dims=3)
    s = pd.DataFrame(Xp).rolling(7, center=True, min_periods=2).mean().to_numpy()
    ref = np.sqrt(sum(np.gradient(s[:, k]) ** 2 for k in range(3))) / np.gradient(t)
    assert np.allclose(v, np.nan_to_num(ref)) and np.allclose(v[15:24], 2.0)


def test_scene_package_round_trip(tmp_path):
    meta, heights = make_meta(terrain=bumpy(n=50), R_max=40.0)
    (tmp_path / "scene.json").write_text(json.dumps(meta), encoding="utf-8")
    heights.astype("<f4").tofile(tmp_path / "terrain.f32")
    scene = lp.load_scene(tmp_path)
    assert scene.grid is not None and scene.grid.z.shape == (50, 50)
    x, y = np.array([-199.5, -190.0, 500.0]), np.array([199.5, 180.0, 0.0])
    z = scene.grid.height(x, y)
    assert np.isclose(z[0], heights[0]) and z[2] == 0.0
    bad = json.loads(json.dumps(meta))
    bad["camera"]["C"][0] += 1.0
    with pytest.raises(ValueError):
        lp.prepare_scene(bad, heights)


# --------------------------------------------------------------------------- the space clutch (08 (a))
def phone_delta(scene, Q, zero, gain=1.0):
    return lp.pose_to_camera(scene, pose(Q), zero, gain=gain, with_delta=True)[1]


def drawn_in_space(scene, cl, X):
    """X' = C0 + S (G_Dc(X) - C0): the clutch's camera, the ground mode, the turned space."""
    Xp, st = lp.project(scene, lp.turn_camera(scene, cl.Dc), X)
    return lp.apply_space(cl.S, scene.C, Xp), st


def test_clutch_hold_turns_the_space_with_the_hand():
    yaw = 30.0
    scene = phone_scene(C=(0, 0, 1.5), yaw=yaw, tilt=3.0, R_max=1e4)
    zero = pose(PHONE0)
    far = ground_points(scene, [yaw], [40.0])
    cl = lp.Clutch().press(phone_delta(scene, PHONE0, zero))
    down = turn(PHONE0[:, 0], -10.0) @ PHONE0                     # the phone tilted down 10 deg
    cl.move(phone_delta(scene, down, zero))
    assert cl.held and cl.tilted and np.isclose(np.degrees(cl.angle), 10.0)
    assert np.allclose(cl.Dc, np.eye(3))                           # held: the camera does not move
    Xs = lp.apply_space(cl.S, scene.C, far)
    assert np.isclose(Xs[0, 2] - far[0, 2], -40.0 * np.sin(np.radians(10.0)), atol=0.1)   # far side down
    right = turn([0, 0, 1], -20.0) @ PHONE0                        # the phone turned right 20 deg
    cl.move(phone_delta(scene, right, zero))
    _, b = polar(scene, lp.apply_space(cl.S, scene.C, far))
    assert np.isclose(b[0], yaw + 20.0)                            # the space turns right
    # the drawn crowd is the ground-mode drawing carried by the space
    X = ground_points(scene, yaw + np.linspace(-20, 20, 9), np.full(9, 15.0))
    Xd, st = drawn_in_space(scene, cl, X)
    assert (st == lp.DRAWN).all()
    assert np.abs(Xd - lp.apply_space(cl.S, scene.C, X)).max() < CM
    M = lp.space_matrix(cl.S, scene.C)
    assert np.allclose((np.c_[X, np.ones(9)] @ M.T)[:, :3], lp.apply_space(cl.S, scene.C, X))


def test_clutch_release_steers_the_camera_inside_the_turned_space():
    yaw = 30.0
    scene = phone_scene(C=(0, 0, 1.5), yaw=yaw, tilt=4.0, R_max=1e4)
    zero = pose(PHONE0)
    cl = lp.Clutch()
    pre = turn([0, 0, 1], -7.0) @ PHONE0                           # a pan before the hold
    cl.move(phone_delta(scene, pre, zero))
    Dc0 = cl.Dc.copy()
    cl.press(phone_delta(scene, pre, zero))
    tilted = turn(pre[:, 0], -12.0) @ pre
    cl.move(phone_delta(scene, tilted, zero)).release(phone_delta(scene, tilted, zero))
    assert not cl.held and np.allclose(cl.Dc, Dc0)                 # the camera goes on from the press
    theta = 15.0
    after = turn([0, 0, 1], -theta) @ tilted                       # then pan right: the camera turns
    cl.move(phone_delta(scene, after, zero))
    rng = np.random.default_rng(51)
    X = ground_points(scene, yaw + 7 + theta + rng.uniform(-20, 20, 300), rng.uniform(4, 60, 300))
    Xd, st = drawn_in_space(scene, cl, X)
    ok = st == lp.DRAWN
    assert ok.mean() > 0.5
    G = lp.apply_space(cl.S.T, scene.C, Xd[ok])                    # back in the space's own frame
    d, b = polar(scene, X[ok])
    d2, b2 = polar(scene, G)
    assert np.abs((b2 - b + 180) % 360 - 180 + 7 + theta).max() * np.pi / 180 * d.max() < CM
    assert np.abs(d2 - d).max() < CM
    assert np.abs(G[:, 2] - X[ok, 2]).max() < CM                   # on the (turned) ground
    assert np.abs(Xd[ok, 2] - X[ok, 2]).max() > 1.0                # and that ground is tilted


def test_clutch_has_no_jumps_at_press_release_and_zero():
    scene = phone_scene(C=(0, 0, 1.5), yaw=30.0, tilt=3.0, R_max=1e4)
    zero = pose(PHONE0)
    rng = np.random.default_rng(52)
    cl = lp.Clutch()
    Q = PHONE0
    for k in range(400):
        Q = Rot.from_rotvec(rng.normal(0, 0.02, 3)).as_matrix() @ Q
        D = phone_delta(scene, Q, zero, gain=1.7)
        cl.move(D, t=k / 60.0)
        S, Dc = cl.S.copy(), cl.Dc.copy()
        op = rng.choice(["move"] * 6 + ["press", "release", "zero"])
        if op == "press":
            cl.press(D)
        elif op == "release":
            cl.release(D)
        elif op == "zero":
            cl.zero()
            zero, Dc = pose(Q), np.eye(3)                          # zero resets the camera only
            D = phone_delta(scene, Q, zero, gain=1.7)
            assert np.allclose(D, np.eye(3))
        assert np.allclose(cl.S, S, atol=1e-12) and np.allclose(cl.Dc, Dc, atol=1e-12), op
        cl.move(D, t=k / 60.0)                                     # the same pose again: still no jump
        assert np.allclose(cl.S, S, atol=1e-12) and np.allclose(cl.Dc, Dc, atol=1e-12), op
    assert cl.tilted


def test_clutch_level_glides_back_in_0_3_s():
    scene = phone_scene(C=(0, 0, 1.5), yaw=30.0, tilt=3.0, R_max=1e4)
    zero = pose(PHONE0)
    cl = lp.Clutch().press(phone_delta(scene, PHONE0, zero))
    Q = turn([1.0, 0.4, 0.2], 25.0) @ PHONE0
    D = phone_delta(scene, Q, zero)
    cl.move(D).release(D)
    axis = lp.matrix_to_rotvec(cl.S) / cl.angle
    a0 = cl.angle
    cl.level(10.0)
    ts = 10.0 + np.linspace(0, 0.36, 37)
    ang = []
    for t in ts:
        cl.move(D, t)
        ang.append(cl.angle)
        if cl.angle > 1e-9:
            assert np.allclose(lp.matrix_to_rotvec(cl.S) / cl.angle, axis)       # a slerp: the same axis
    ang = np.array(ang)
    assert np.all(np.diff(ang) <= 1e-12) and np.isclose(ang[0], a0)
    assert np.isclose(ang[np.argmin(np.abs(ts - 10.15))], a0 / 2)               # eased: half at half time
    assert (ang[ts >= 10.3 - 1e-9] < 1e-12).all() and ang[ts < 10.29].min() > 0
    assert not cl.tilted and np.allclose(cl.Dc, D @ cl.K)                       # the camera untouched
    # a hold during a glide: the hand does not turn the space; after the glide it does, from level
    cl.press(D).move(D, 20.0)
    cl.move(phone_delta(scene, turn([0, 0, 1], -10.0) @ Q, zero), 20.0)
    assert np.isclose(np.degrees(cl.angle), 10.0)
    cl.level(30.0).move(D, 30.1)
    assert 0 < cl.angle < np.radians(10.0)
    cl.move(phone_delta(scene, turn([0, 0, 1], 5.0) @ Q, zero), 30.2)            # turned while gliding
    cl.move(phone_delta(scene, turn([0, 0, 1], 5.0) @ Q, zero), 30.4)            # the glide ends: level
    assert not cl.tilted
    cl.move(phone_delta(scene, turn([0, 0, 1], -5.0) @ Q, zero), 30.5)           # the hold goes on from there
    assert np.isclose(np.degrees(cl.angle), 10.0)


CLUTCH_RUNNER = r"""
import { readFileSync, writeFileSync } from "node:fs";
import * as L from "./projection.mjs";
const dir = process.argv[2];
const job = JSON.parse(readFileSync(dir + "/clutch.json", "utf8"));
const scene = L.prepareScene(job.meta, null);
const cl = new L.Clutch(), D = new Float64Array(9), res = [];
for (const ev of job.events) {
  L.poseToCamera(scene, ev.src, ev.zero, ev.gain, 1, { delta: D });
  if (ev.op === "press") cl.press(D);
  else if (ev.op === "release") cl.release(D);
  else if (ev.op === "level") cl.level(ev.t, ev.duration);
  else if (ev.op === "zero") cl.zero();
  else cl.move(D, ev.t);
  res.push([...D, ...cl.S, ...cl.K, ...cl.Dc, cl.held ? 1 : 0, cl.angle]);
}
const S = Float64Array.from(job.S), C0 = Float64Array.from(job.C0), X = Float64Array.from(job.X);
const Xs = L.applySpace(S, C0, X), M = L.spaceMatrix(S, C0);
const into = new Float64Array(X.length).fill(7);
L.applySpace(S, C0, X, into, 2, 5);
writeFileSync(dir + "/out.json", JSON.stringify({ res, Xs: Array.from(Xs), M: Array.from(M), into: Array.from(into) }));
"""


def test_clutch_js_equals_python(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    meta, _ = make_meta(C=(1.0, -2.0, 1.7), yaw=-35.0, tilt=3.0, roll=0.6, R_max=110.0)
    scene = lp.prepare_scene(meta)
    rng = np.random.default_rng(53)
    Q, zero, gain, events = cam_R_wc(150.0, 8.0), None, 1.0, []
    for k in range(600):
        t = k / 60.0
        Q = Rot.from_rotvec(rng.normal(0, 0.03, 3)).as_matrix() @ Q
        op = str(rng.choice(["move"] * 8 + ["press", "release", "level", "zero", "gain"]))
        if op == "zero" or zero is None:
            zero = pose(Q)
        if op == "gain":
            gain, op = float(rng.uniform(0.5, 3.0)), "move"
        events.append({"op": op, "t": t, "duration": float(rng.choice([0.3, 0.0, 0.5])),
                       "src": pose(Q), "zero": zero, "gain": gain})
    S = Rot.from_rotvec(rng.normal(0, 0.5, 3)).as_matrix()
    X = rng.normal(0, 30, (20, 3))
    (tmp_path / "projection.mjs").write_bytes(JS.read_bytes())
    (tmp_path / "runner.mjs").write_text(CLUTCH_RUNNER, encoding="utf-8")
    job = {"meta": meta, "events": events, "S": S.ravel().tolist(), "C0": scene.C.tolist(), "X": X.ravel().tolist()}
    (tmp_path / "clutch.json").write_text(json.dumps(job), encoding="utf-8")
    r = subprocess.run([node, "runner.mjs", str(tmp_path)], cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    out = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    cl, worst, held = lp.Clutch(), 0.0, 0
    for ev, js in zip(events, out["res"]):
        _, D = lp.pose_to_camera(scene, ev["src"], ev["zero"], ev["gain"], with_delta=True)
        if ev["op"] == "press":
            cl.press(D)
        elif ev["op"] == "release":
            cl.release(D)
        elif ev["op"] == "level":
            cl.level(ev["t"], ev["duration"])
        elif ev["op"] == "zero":
            cl.zero()
        else:
            cl.move(D, ev["t"])
        py = np.concatenate([D.ravel(), cl.S.ravel(), cl.K.ravel(), cl.Dc.ravel(), [float(cl.held), cl.angle]])
        worst = max(worst, np.abs(np.asarray(js) - py).max())
        held += cl.held
    print(f"\nclutch JS = Python: {len(events)} events ({held} held), max difference {worst:.2e}")
    assert worst < 1e-9 and 50 < held < len(events) - 50
    assert np.abs(np.reshape(out["Xs"], (-1, 3)) - lp.apply_space(S, scene.C, X)).max() < 1e-9
    assert np.abs(np.reshape(out["M"], (4, 4)) - lp.space_matrix(S, scene.C)).max() < 1e-12
    into = np.reshape(out["into"], (-1, 3))
    assert np.allclose(into[2:5], lp.apply_space(S, scene.C, X[2:5])) and (into[[0, 1, 5]] == 7).all()
