import numpy as np
import pandas as pd
import pytest

from groundtrack.config import Group
from groundtrack.homography import fit_homography
from groundtrack.trajectories import (
    heading_deg, process_tracks, remove_spikes, savgol_window, smooth_and_differentiate,
    split_segments, vehicle_centre,
)

FPS = 30.0
DT = 1 / FPS


def test_constant_velocity_line():
    """1.4 m/s heading 60 deg with 5 cm position noise -> speed within 2 %, heading within 2 deg."""
    rng = np.random.default_rng(0)
    t = np.arange(0, 10, DT)
    hd = np.radians(60)
    x = 1.4 * np.sin(hd) * t + rng.normal(0, 0.05, len(t))
    y = 1.4 * np.cos(hd) * t + rng.normal(0, 0.05, len(t))
    xs, ys, vx, vy = smooth_and_differentiate(x, y, DT, 1.0)
    speed = np.hypot(vx, vy)
    inner = slice(15, -15)
    assert abs(speed[inner].mean() - 1.4) / 1.4 < 0.02
    assert np.abs(speed[inner] - 1.4).max() < 0.35
    assert abs(np.median(heading_deg(vx, vy, 0.3)) - 60) < 2
    # smoothing reduces the noise
    true_x = 1.4 * np.sin(hd) * t
    assert np.std(xs - true_x) < 0.5 * np.std(x - true_x)


def test_circle_speed_and_heading():
    """Walking a 10 m radius circle anticlockwise at 1.2 m/s."""
    t = np.arange(0, 20, DT)
    w = 1.2 / 10
    x, y = 10 * np.cos(w * t), 10 * np.sin(w * t)
    _, _, vx, vy = smooth_and_differentiate(x, y, DT, 1.0)
    speed = np.hypot(vx, vy)
    assert np.allclose(speed[5:-5], 1.2, atol=0.01)
    hd = heading_deg(vx, vy, 0.3)
    # anticlockwise motion: heading = direction of the tangent (-sin, cos)
    true = (np.degrees(np.arctan2(-np.sin(w * t), np.cos(w * t))) + 360) % 360
    diff = (hd - true + 180) % 360 - 180
    assert np.abs(diff[5:-5]).max() < 1.0


def test_heading_held_when_stationary():
    vx = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    vy = np.zeros(7)
    hd = heading_deg(vx, vy, 0.3)
    assert np.allclose(hd[:6], 90)
    assert hd[6] == pytest.approx(270)


def test_savgol_window():
    assert savgol_window(1000, 1.0, DT, 2) == 31
    assert savgol_window(1000, 0.6, DT, 2) == 19
    assert savgol_window(10, 1.0, DT, 2) == 9
    assert savgol_window(3, 1.0, DT, 2) == 3
    assert savgol_window(2, 1.0, DT, 2) == 0


def test_spike_removed():
    t = np.arange(20) * DT
    x = 1.4 * t
    y = np.zeros(20)
    x2 = x.copy()
    x2[10] += 5.0  # 5 m spike in one frame
    keep = remove_spikes(t, x2, y, vmax=6, tol=1.0)
    assert not keep[10] and keep.sum() == 19
    assert remove_spikes(t, x, y, vmax=6, tol=1.0).all()


def test_id_switch_splits_track():
    t = np.arange(40) * DT
    x = np.where(np.arange(40) < 20, 1.4 * t, 30 + 1.4 * t)  # jumps 30 m at sample 20
    segs = split_segments(t, x, np.zeros(40), vmax=6, tol=1.0, max_gap_s=2.0)
    assert [len(s) for s in segs] == [20, 20]


def test_vehicle_offset_travel():
    g = Group("vehicles", ["car"], (0, 35), 70, 0.6, 2.2, "travel")
    x, y = vehicle_centre(np.array([100.0]), np.array([50.0]), np.array([90.0]), g, "car")
    assert x[0] == pytest.approx(97.8) and y[0] == pytest.approx(50.0)
    # unknown heading (never moved): no shift
    x, y = vehicle_centre(np.array([100.0]), np.array([50.0]), np.array([np.nan]), g, "car")
    assert x[0] == 100.0
    # per-class offsets
    g2 = Group("vehicles", ["car", "bus"], (0, 35), 70, 0.6, {"default": 2.2, "bus": 6}, "travel")
    x, y = vehicle_centre(np.array([0.0]), np.array([0.0]), np.array([0.0]), g2, "bus")
    assert y[0] == pytest.approx(-6)


def test_vehicle_offset_view_moves_away_from_camera(cam, ground_points):
    """Camera looks north: the centre must be north of (further than) the foot point,
    whichever way the car drives."""
    h = fit_homography(cam.ground_to_pixel(ground_points), ground_points,
                       image_size=(cam.width, cam.height))
    g = Group("vehicles", ["car"], (0, 35), 70, 0.6, 2.2, "view", vehicle_width_m=1.8)
    foot = ground_points[[2]]
    uv = cam.ground_to_pixel(foot)
    away = foot[0] - cam.C[:2]
    away /= np.linalg.norm(away)
    for heading in (0.0, 180.0, 90.0, 270.0, 30.0):
        x, y = vehicle_centre(foot[:, 0], foot[:, 1], np.array([heading]), g, "car", h,
                              uv[:, 0], uv[:, 1])
        shift = np.array([x[0], y[0]]) - foot[0]
        # always exactly away from the camera, whichever way the car drives
        assert shift / np.linalg.norm(shift) == pytest.approx(away, abs=1e-6)
        # by the rectangle's support distance: half length end-on, half width side-on
        travel = np.array([np.sin(np.radians(heading)), np.cos(np.radians(heading))])
        c = abs(travel @ away)
        expected = 2.2 * c + 0.9 * np.sqrt(1 - c * c)
        assert np.linalg.norm(shift) == pytest.approx(expected, abs=1e-6)


def test_camera_recovered_from_homography(cam, ground_points):
    h = fit_homography(cam.ground_to_pixel(ground_points), ground_points,
                       image_size=(cam.width, cam.height))
    c = h.camera()
    assert c["focal_px"] == pytest.approx(1400, rel=1e-3)
    assert c["height_m"] == pytest.approx(12.0, abs=0.01)
    assert c["E"] == pytest.approx(cam.C[0], abs=0.01)
    assert c["N"] == pytest.approx(cam.C[1], abs=0.01)


@pytest.mark.parametrize("yaw,roll,pitch,f", [(25, 0, 30, 1400), (-40, 3, 45, 1100),
                                              (10, -2, 20, 2600)])
def test_camera_recovered_rotated(ground_points, yaw, roll, pitch, f):
    from conftest import ORIGIN, SyntheticCamera

    cam = SyntheticCamera(f=f, yaw_deg=yaw, roll_deg=roll, pitch_deg=pitch, height_m=9.0,
                          cam_local=(-3.0, -25.0))
    # keep only calibration points that are actually in view
    uv = cam.ground_to_pixel(ground_points)
    vis = (uv[:, 0] > 0) & (uv[:, 0] < 1920) & (uv[:, 1] > 0) & (uv[:, 1] < 1080)
    extra = ORIGIN + np.array([[-10, -5], [10, -8], [0, 10], [-20, 25], [20, 20], [5, 40]])
    pts = np.vstack([ground_points[vis], extra])
    h = fit_homography(cam.ground_to_pixel(pts), pts, image_size=(1920, 1080))
    c = h.camera()
    assert c["focal_px"] == pytest.approx(f, rel=1e-3)
    assert c["height_m"] == pytest.approx(9.0, abs=0.02)
    assert np.hypot(c["E"] - cam.C[0], c["N"] - cam.C[1]) < 0.05


def test_view_offset_fallback_without_camera(cam, ground_points):
    """No image size -> no camera estimate -> 'image up' approximation, still away-ish."""
    h = fit_homography(cam.ground_to_pixel(ground_points), ground_points)
    assert h.camera() is None
    g = Group("vehicles", ["car"], (0, 35), 70, 0.6, 2.2, "view")
    foot = ground_points[[2]]
    uv = cam.ground_to_pixel(foot)
    x, y = vehicle_centre(foot[:, 0], foot[:, 1], np.array([0.0]), g, "car", h, uv[:, 0], uv[:, 1])
    shift = np.array([x[0], y[0]]) - foot[0]
    away = (foot[0] - cam.C[:2]) / np.linalg.norm(foot[0] - cam.C[:2])
    assert shift @ away / np.linalg.norm(shift) > 0.95


def _synthetic_raw(cam, tracks):
    """tracks: list of (track_id, class, start_xy_local, velocity, n_frames, predicted_frames)."""
    from conftest import ORIGIN

    rows = []
    for tid, cls, p0, vel, n, pred in tracks:
        for k in range(n):
            en = ORIGIN + np.asarray(p0) + np.asarray(vel) * k * DT
            u, v = cam.ground_to_pixel(en)[0]
            w = 40.0 if cls == "person" else 120.0
            hgt = 100.0 if cls == "person" else 60.0
            rows.append((tid, cls, 0 if cls == "person" else 2, k, k * DT, u - w / 2, v - hgt,
                         u + w / 2, v, np.nan if k in pred else 0.8, k in pred))
    return pd.DataFrame(rows, columns=["track_id", "class", "class_id", "frame", "time_s", "x1",
                                       "y1", "x2", "y2", "confidence", "predicted"])


def test_process_tracks_end_to_end(cam, ground_points, cfg):
    from conftest import ORIGIN

    h = fit_homography(cam.ground_to_pixel(ground_points), ground_points)
    raw = _synthetic_raw(cam, [
        (1, "person", (-5, 10), (1.4, 0.0), 150, set(range(60, 75))),  # occluded 0.5 s
        (2, "person", (5, 30), (0.0, -1.0), 15, set()),               # 0.5 s: short, kept
        (3, "car", (-20, 40), (25.0, 0.0), 45, set()),
    ])
    points, summary = process_tracks(raw, cfg, h, FPS, 1, log=lambda *_: None)
    assert set(summary["track_id"]) == {1, 2, 3}       # min_track_s 0: short pieces stay, as in Live
    cfg.data["cleaning"]["min_track_s"] = 1.0
    assert set(process_tracks(raw, cfg, h, FPS, 1, log=lambda *_: None)[1]["track_id"]) == {1, 3}
    cfg.data["cleaning"]["min_track_s"] = 0.0
    p1 = points[points["track_id"] == 1]
    assert abs(p1["speed"].iloc[10:-10].mean() - 1.4) < 0.02
    assert p1["predicted"].sum() == 15
    assert (p1.loc[p1["predicted"], "source"] == "predicted").all()
    assert np.allclose(p1["y"], ORIGIN[1] + 10, atol=0.01)
    s1 = summary.set_index("track_id").loc[1]
    assert s1["straightness"] == pytest.approx(1.0, abs=1e-3)
    assert s1["path_length_m"] == pytest.approx(1.4 * 149 * DT, rel=0.01)
    # car: centre is 2.2 m behind the foot point along travel (east)
    p3 = points[points["track_id"] == 3]
    assert abs(p3["speed"].mean() - 25) < 0.1
    true_x0 = ORIGIN[0] - 20 - 2.2
    assert p3["x"].iloc[0] == pytest.approx(true_x0, abs=0.05)
    assert set(points.columns[:11]) == {"track_id", "class", "frame", "time_s", "x", "y", "vx",
                                        "vy", "speed", "heading", "predicted"}


def test_class_majority_vote(cam, ground_points, cfg):
    h = fit_homography(cam.ground_to_pixel(ground_points), ground_points)
    raw = _synthetic_raw(cam, [(7, "car", (0, 20), (20.0, 0.0), 60, set())])
    raw.loc[raw.index[:10], "class"] = "truck"  # flicker
    points, summary = process_tracks(raw, cfg, h, FPS, 1, log=lambda *_: None)
    assert summary["class"].tolist() == ["car"]
    assert (points["class"] == "car").all()


def test_config_extends(tmp_path):
    from groundtrack.config import load_config

    (tmp_path / "base.yaml").write_text(
        "site: motorway\nvideo: a.mov\nhomography: a.json\n"
        "detection: {imgsz: 1920, conf: 0.3}\n"
        "groups:\n  vehicles: {classes: [car, truck], offset_mode: view}\n")
    (tmp_path / "clip2.yaml").write_text(
        "extends: base.yaml\nvideo: b.mov\nhomography: b.json\ndetection: {conf: 0.2}\n")
    c = load_config(tmp_path / "clip2.yaml")
    assert c.site == "motorway"
    assert c.path("video") == (tmp_path / "b.mov").resolve()
    assert c["detection"]["imgsz"] == 1920 and c["detection"]["conf"] == 0.2
    assert c.classes == ["car", "truck"] and c.groups["vehicles"].offset_mode == "view"
    assert c.groups["vehicles"].max_speed == 70.0  # defaults still fill in


def test_parked_vehicles_dropped_but_standing_people_kept(cam, ground_points, cfg):
    h = fit_homography(cam.ground_to_pixel(ground_points), ground_points)
    raw = _synthetic_raw(cam, [
        (1, "car", (-10, 30), (0.0, 0.0), 90, set()),     # parked for 3 s
        (2, "car", (-20, 40), (20.0, 0.0), 45, set()),    # driving
        (3, "person", (0, 20), (0.0, 0.0), 90, set()),    # someone standing: real data
    ])
    _, summary = process_tracks(raw, cfg, h, FPS, 1, log=lambda *_: None)
    assert set(summary["track_id"]) == {2, 3}


def _box_track(n=90, cut=None):
    """A person walking down the image: box 100 px tall; `cut` = (frames, 'bottom'|'top', px)."""
    rows = []
    for k in range(n):
        y2 = 500.0 + 2.0 * k
        y1 = y2 - 100.0
        if cut and k in cut[0]:
            if cut[1] == "bottom":
                y2 -= cut[2]
            else:
                y1 += cut[2]
        rows.append((1, "person", 0, k, k / 30, 300.0, y1, 340.0, y2, 0.8, False))
    return pd.DataFrame(rows, columns=["track_id", "class", "class_id", "frame", "time_s", "x1",
                                       "y1", "x2", "y2", "confidence", "predicted"])


def test_feet_recovered_when_lower_body_hidden():
    from groundtrack.trajectories import recover_feet

    hidden = set(range(40, 55))
    raw = _box_track(cut=(hidden, "bottom", 40))   # railing hides 40 px of legs
    uv, rec = recover_feet(raw, rows_per_window=60)
    true_y2 = 500.0 + 2.0 * np.arange(90)
    assert rec.sum() == len(hidden)
    assert np.abs(uv[:, 1] - true_y2).max() < 1.0  # feet back where they really are


def test_top_occlusion_keeps_real_feet():
    from groundtrack.trajectories import recover_feet

    raw = _box_track(cut=(set(range(40, 55)), "top", 40))  # e.g. an umbrella / gantry above
    uv, rec = recover_feet(raw, rows_per_window=60)
    assert not rec.any()
    assert np.allclose(uv[:, 1], raw["y2"])


def test_unoccluded_track_untouched():
    from groundtrack.trajectories import recover_feet

    raw = _box_track()
    uv, rec = recover_feet(raw, rows_per_window=60)
    assert not rec.any() and np.allclose(uv[:, 1], raw["y2"])


def test_camera_calibration_with_known_position(cam, ground_points):
    """3 clicks + the camera's position recover the full calibration; a bad click shows up."""
    from groundtrack.posefit import camera_calibration

    px = cam.ground_to_pixel(ground_points)
    prior = {"E": cam.C[0] + 4, "N": cam.C[1] - 3, "height_m": 12.5, "position_tol_m": 10,
             "height_tol_m": 3, "hfov_deg": 60}
    # exact prior + 3 clicks: exact
    exact = dict(prior, E=cam.C[0], N=cam.C[1], height_m=12.0)
    h = camera_calibration(px[[0, 3, 6]], ground_points[[0, 3, 6]], (cam.width, cam.height), exact)
    test = ground_points[[1, 2, 4, 5, 7]]
    err = np.linalg.norm(h.to_world(cam.ground_to_pixel(test)) - test, axis=1)
    assert err.max() < 0.05
    # prior 5 m / 0.5 m off + 5 clicks: the clicks pull it back
    h = camera_calibration(px[:5], ground_points[:5], (cam.width, cam.height), prior)
    err = np.linalg.norm(h.to_world(cam.ground_to_pixel(ground_points[5:])) - ground_points[5:],
                         axis=1)
    assert err.max() < 0.5
    bad = ground_points.copy()
    bad[2] += [6.0, -4.0]
    h2 = camera_calibration(px, bad, (cam.width, cam.height), prior)
    assert not h2.points[2]["inlier"] and sum(p["inlier"] for p in h2.points) == 7


def test_far_samples_kept_and_flagged(cam, ground_points, cfg):
    """Past the precise range (projection.max_range_m) samples are kept, flagged beyond_limit,
    up to FAR_FACTOR x the range; further ones are dropped."""
    from groundtrack.trajectories import FAR_FACTOR

    from conftest import ORIGIN

    h = fit_homography(cam.ground_to_pixel(ground_points), ground_points)
    h.camera_params = {"E": ORIGIN[0], "N": ORIGIN[1] - 20.0, "height_m": 12.0}  # the camera
    cfg.data["projection"]["max_range_m"] = 30
    raw = _synthetic_raw(cam, [
        (1, "person", (-5, 0), (1.3, 0.0), 90, set()),      # ~20 m away: precise
        (2, "person", (-5, 30), (1.3, 0.0), 90, set()),     # ~50 m: kept, flagged
        (3, "person", (-5, 30 * FAR_FACTOR), (1.3, 0.0), 90, set()),   # > 2.2 x 30 m: dropped
    ])
    points, _ = process_tracks(raw, cfg, h, FPS, 1, log=lambda *_: None)
    by = points.groupby("track_id")["beyond_limit"].agg(["min", "max"])
    assert set(by.index) == {1, 2}
    assert not by.loc[1, "max"] and by.loc[2, "min"]


def test_radial_smoothing_removes_depth_wobble():
    """A walker 25 m from a 2 m high camera, foot points wobbling 0.3 m in depth (about a
    pixel; box jitter lasts several frames, so the wobble is correlated) and 0.03 m sideways:
    smoothing depth more than sideways gets the speed right, the plain smoothing does not."""
    from scipy.ndimage import gaussian_filter1d

    from groundtrack.trajectories import smooth_radial

    rng = np.random.default_rng(0)
    t = np.arange(0, 4, DT)
    cam = {"E": 0.0, "N": 0.0, "height_m": 2.0}
    x, y = -2.8 + 1.4 * t, np.full_like(t, 25.0)           # walking across the view, east
    r, th = np.hypot(x, y), np.arctan2(y, x)
    wobble = gaussian_filter1d(rng.normal(0, 1, len(t)), 6)
    r = r + wobble * 0.3 / wobble.std()
    th = th + rng.normal(0, 0.03 / 25, len(t))
    xn, yn = r * np.cos(th), r * np.sin(th)
    _, _, vx0, vy0 = smooth_and_differentiate(xn, yn, DT, 1.0)
    _, _, vx1, vy1 = smooth_radial(xn, yn, DT, 1.0, cam)
    err0 = np.abs(np.hypot(vx0, vy0) - 1.4)[15:-15].mean()
    err1 = np.abs(np.hypot(vx1, vy1) - 1.4)[15:-15].mean()
    assert err1 < 0.3 * err0 and err1 < 0.05


def test_passengers_inside_a_bus_are_dropped():
    """A person whose box sits inside a bus box nearly all the time is a passenger; one who
    walks past the bus (inside briefly) is a pedestrian."""
    from groundtrack.trajectories import passenger_tracks

    rows = []
    for f in range(30):
        rows.append((9, "bus", f, 100 + 5 * f, 200, 400 + 5 * f, 420))          # moving bus
        rows.append((1, "person", f, 200 + 5 * f, 230, 230 + 5 * f, 300))       # in a window
        x = 50 + 40 * f                                                       # walks past
        rows.append((2, "person", f, x, 300, x + 30, 380))
    raw = pd.DataFrame(rows, columns=["track_id", "class", "frame", "x1", "y1", "x2", "y2"])
    assert passenger_tracks(raw) == {1}


def test_people_in_front_of_a_vehicle_are_kept():
    """Inside a vehicle's box the whole time, but not riding in it: someone standing by a
    parked car, someone walking across the front of a moving bus, and someone walking just
    ahead of a car crawling at walking pace (feet below the car's bottom edge)."""
    from groundtrack.trajectories import passenger_tracks

    rows = []
    for f in range(30):
        rows.append((7, "car", f, 600, 300, 800, 400))                            # parked
        rows.append((3, "person", f, 680, 280, 710, 380))                         # stands by it
        rows.append((9, "bus", f, 100 + 6 * f, 200, 700 + 6 * f, 420))            # moving bus
        x = 150 + 15 * f                                                        # walks across
        rows.append((4, "person", f, x, 300, x + 30, 380))
        rows.append((8, "car", f, 900 + 4 * f, 300, 1100 + 4 * f, 400))         # crawls
        rows.append((5, "person", f, 980 + 4 * f, 290, 1010 + 4 * f, 410))      # walks ahead of it
    raw = pd.DataFrame(rows, columns=["track_id", "class", "frame", "x1", "y1", "x2", "y2"])
    assert passenger_tracks(raw) == set()


def test_feet_hidden_behind_a_nearer_person():
    """In a crowd seen from low down the far person's legs are behind the near one: flagged.
    Two people side by side, or one in front of a car, are not."""
    from groundtrack.trajectories import feet_hidden

    raw = pd.DataFrame([
        (1, "person", 0, 100, 100, 140, 200),     # far, legs covered by 2
        (2, "person", 0, 95, 150, 145, 300),      # near, reaches lower
        (3, "person", 0, 400, 100, 440, 200),     # side by side with 4
        (4, "person", 0, 460, 110, 500, 210),
        (5, "person", 0, 700, 100, 740, 200),     # in front of a car: only people hide feet
        (6, "car", 0, 650, 160, 900, 260),
    ], columns=["track_id", "class", "frame", "x1", "y1", "x2", "y2"])
    assert feet_hidden(raw).tolist() == [True, False, False, False, False, False]


def test_hidden_feet_bridged(cam, ground_points, cfg):
    """hidden_feet: bridge - frames where a nearer person covers someone's legs are left out
    and interpolated from the frames around them; the walk keeps its speed."""
    h = fit_homography(cam.ground_to_pixel(ground_points), ground_points)
    raw = _synthetic_raw(cam, [(1, "person", (-5, 10), (1.4, 0.0), 120, set())])
    walker = raw.set_index("frame")
    front = []
    for f in range(40, 60):                       # someone nearer, over the walker's legs
        b = walker.loc[f]
        front.append((9, "person", 0, f, b["time_s"], b["x1"], b["y2"] - 50, b["x2"],
                      b["y2"] + 60, 0.8, False))
    raw = pd.concat([raw, pd.DataFrame(front, columns=raw.columns)], ignore_index=True)
    cfg["cleaning"]["hidden_feet"] = "bridge"
    points, _ = process_tracks(raw, cfg, h, FPS, 1, log=lambda *_: None)
    p1 = points[points["track_id"] == 1].set_index("frame")
    assert (p1.loc[40:59, "source"] == "interpolated").all()
    assert (p1.loc[:39, "source"] == "detected").all()
    assert abs(p1["speed"].iloc[10:-10].mean() - 1.4) < 0.02
