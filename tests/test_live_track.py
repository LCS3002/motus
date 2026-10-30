"""Realtime (plans/live_controller/10, steps 1-3): the camera fixed from a still second of the
recorder's poses, and a recording made by LiveTracker from JPEG frames (a stand-in model, the real
ByteTrack): the same tracks as tracking those detections frame by frame, the package Live opens,
the raw and blurred videos at a constant frame rate, the poses, a recorder that drops out. The
phone's calibration (step 2): the still second, its height and tilt nudge, the people check; the
fixed camera drags the plan as the recorder turns. The Live view's feed (step 3): the points so
far, whole frames only, and the blurred preview."""

import csv
import json
import math
import shutil
import subprocess
import time
from collections import deque
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from groundtrack import live_track
from groundtrack.config import default_config
from groundtrack.homography import Homography
from groundtrack.live_export import TRACK_DTYPE, _angles
from groundtrack.live_track import (
    CAL_S,
    CAL_TIMEOUT_S,
    FPS,
    POSE_DTYPE,
    R_MAX_M,
    STATS_COLUMNS,
    LiveTracker,
    Recording,
    level_camera,
    list_recordings,
    package_tracks,
    quat_matrix,
    recording_title,
    sensor_camera,
)


def until(fn, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if fn():
            return True
        time.sleep(0.01)
    return False


def rot_z(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])


def to_quat(R):
    """Rotation matrix -> [w, x, y, z] (pose v1 q)."""
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w > 1e-6:
        return [w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w),
                (R[1, 0] - R[0, 1]) / (4 * w)]
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(max(0.0, 1 + R[i, i] - R[j, j] - R[k, k])) * 2
    q = [0.0, 0.0, 0.0, 0.0]
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = s / 4
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


def phone_pose(heading, tilt, roll):
    """world_from_camera (OpenCV axes) of a phone facing `heading` (clockwise from the sensor
    frame's north), tilt down, rolled about its view axis."""
    return rot_z(-heading) @ level_camera(tilt) @ rot_z(roll)


# --------------------------------------------------------------------------- the camera
@pytest.mark.parametrize("heading", [0, 37, 200, 315])
def test_sensor_camera_keeps_tilt_and_roll_and_faces_north(heading):
    R = phone_pose(heading, 20, 5)
    rng = np.random.default_rng(heading)
    qs = []
    for _ in range(30):                                   # a hand that shakes a little
        q = np.array(to_quat(R)) + rng.normal(0, 0.002, 4)
        qs.append(list(q / np.linalg.norm(q)))
    h, K, R_cw, C = sensor_camera(qs, (1920, 1080))
    cp = h.camera_params
    assert cp["tilt_deg"] == pytest.approx(20, abs=0.3)
    assert abs(cp["roll_deg"]) == pytest.approx(5, abs=0.3)
    assert cp["yaw_deg"] == pytest.approx(0, abs=1e-6)                  # the start is north
    assert np.allclose(R_cw, phone_pose(0, 20, 5).T, atol=0.01)          # whatever the heading
    assert cp["focal_px"] == pytest.approx(1597.7) and cp["height_m"] == 1.5
    assert C.tolist() == [0, 0, 1.5] and K[0, 2] == 960 and K[1, 2] == 540
    assert h.method == "realtime sensors" and h.camera_params["E"] == 0.0
    P = K @ R_cw @ np.column_stack([np.eye(3), -C])
    for X in ([3.0, 12.0], [-4.0, 25.0], [0.5, 6.0]):                   # ground -> pixel -> ground
        x = P @ np.r_[X, 0.0, 1.0]
        assert x[2] > 0
        assert np.allclose(h.to_world(np.array([x[:2] / x[2]]))[0], X, atol=1e-6)
    P_h = h._rays()[0]                                    # as the rest of groundtrack builds it
    assert np.allclose(P_h / np.linalg.norm(P_h), P / np.linalg.norm(P), atol=1e-6)
    assert _angles(R_cw)[1] == pytest.approx(cp["tilt_deg"])


def test_sensor_camera_without_poses_is_level_and_ten_degrees_down():
    h, _, _, _ = sensor_camera([], (1280, 720))
    cp = h.camera_params
    assert (cp["tilt_deg"], cp["roll_deg"], cp["yaw_deg"]) == pytest.approx((10, 0, 0), abs=1e-9)
    assert cp["focal_px"] == pytest.approx(1597.7 * 1280 / 1920)
    assert "no poses" in h.method_info["note"]
    _, K2, _, _ = sensor_camera(None, (1080, 1920))        # portrait: the long side sets it
    assert K2[0, 0] == pytest.approx(1597.7)


# --------------------------------------------------------------------------- a recording
W, H = 640, 360
Q_CAL = to_quat(phone_pose(123, 20, 0))                     # 20 deg down, facing anywhere
SKIP = 20                                                  # this frame never arrives
N = 45                                                     # 1.5 s: calibrated at 1 s


def picture(i, noise):
    """Frame i: a fixed noise (what blurring shows), its number as two grey blocks top left
    (base 16, steps wide enough for the video's colour conversion)."""
    img = noise.copy()
    img[:48, :48] = 20 + 14 * (i // 16)
    img[:48, 48:96] = 20 + 14 * (i % 16)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()


def number(img):
    hi, lo = (round((float(img[8:40, x:x + 32].mean()) - 20) / 14) for x in (8, 56))
    return 16 * hi + lo


def boxes(i):
    """The stand-in model's boxes of frame i: two people walking (sure), one it is not sure of
    (blurred, never tracked), a bicycle with no rider."""
    a, b = 200 + 4 * i, 450 - 3 * i
    xyxy = np.array([[a - 15, 220, a + 15, 300], [b - 14, 200, b + 14, 280], [80, 150, 130, 260],
                     [560, 250, 620, 320]], np.float32)
    return xyxy, np.array([0.9, 0.8, 0.15, 0.7], np.float32), np.array([0, 0, 0, 1], np.float32)


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = xyxy, conf, cls

    def cpu(self):
        return self

    def numpy(self):
        return self


class _Result:
    def __init__(self, b):
        self.boxes = b


class FakeYOLO:
    """YOLO's predict: the boxes of the frame whose number the picture carries, at conf."""

    def __init__(self):
        self.seen = []

    def predict(self, img, conf=0.25, classes=None, **kw):
        i = number(img)
        self.seen.append(i)
        xyxy, c, cls = boxes(i)
        keep = (c >= conf) & np.isin(cls, classes)
        return [_Result(_Boxes(xyxy[keep], c[keep], cls[keep]))]


@pytest.fixture
def ffmpeg():
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")


def frames_of(path):
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def replay(cfg, sent, h):
    """The tracks of the same detections, frame by frame (detect.FrameTracker, a new ByteTrack),
    put on the ground through h: what the package must hold."""
    from groundtrack.detect import FrameTracker, make_tracker

    det = cfg["detection"]
    tracker, _ = make_tracker(det["tracker"], FPS, det.get("track_buffer_s"), None, lambda m: None)
    ft = FrameTracker(det, tracker, set(cfg.class_ids), None, record_predicted=False)
    out = []
    blank = np.zeros((H, W, 3), np.uint8)
    for i in sent:
        xyxy, conf, cls = boxes(i)
        sure = conf >= float(det["conf"])
        for r in sorted(ft.step(i, i / FPS, blank, xyxy[sure], conf[sure], cls[sure].astype(int)),
                        key=lambda r: r[0]):
            x, y = h.to_world(np.array([[(r[5] + r[7]) / 2, r[8]]]))[0] - np.asarray(h.origin)
            out.append((r[0], r[3], x, y))
    return out


def test_a_recording_becomes_a_live_package(tmp_path, ffmpeg, key_frames):
    cfg = default_config()
    model = FakeYOLO()
    rt = LiveTracker(tmp_path, cfg=cfg, model=model, log=lambda m: None)
    assert rt.status()["mode"] == "ready" and rt.status()["recording"] is False
    rid = rt.begin(W, H)["id"]
    rec = rt.rec
    with pytest.raises(RuntimeError):                      # one at a time
        rt.begin(W, H)
    noise = np.random.default_rng(1).integers(0, 255, (H, W, 3), dtype=np.uint8)
    sent = [i for i in range(N) if i != SKIP]
    for n, i in enumerate(sent, 1):
        assert rt.frame({"seq": i, "t": 7.25 + i / FPS, "w": W, "h": H, "q": Q_CAL},
                        picture(i, noise))
        assert until(lambda n=n: rec.n_tracked == n, 120 if n == 1 else 10), (i, rec.n_tracked)
        # (one at a time: none skipped by latest-wins; the first one imports ultralytics)
        if i == 5:
            st = rt.status()
            assert st["recording"] and st["id"] == rid and st["calibrated"] is False
            assert st["people"] == 2 and st["tracks"] == 0          # placed once calibrated
    assert rt.frame({"seq": 99, "t": 7.25, "w": W, "h": H, "q": None}, picture(0, noise)) is False
    assert rt.frame({"seq": 99, "t": 9.0, "w": 320, "h": 180, "q": None}, picture(0, noise)) is False
    st = rt.status()
    assert st["calibrated"] and st["camera"]["tilt_deg"] == pytest.approx(20, abs=0.1)
    assert st["frames"] == N - 1 and st["tracked"] == N - 1 and st["tracks"] == 3
    growing = json.loads((tmp_path / "runs" / "live" / rid / "live" / "scene.json").read_text(encoding="utf-8"))
    rt.end()
    assert until(lambda: rt.saved is not None, 60)
    assert rt.rec is None and rt.status()["saved"]["complete"] is True
    assert model.seen == sent

    root = tmp_path / "runs" / "live" / rid
    meta = json.loads((root / "recording.json").read_text(encoding="utf-8"))
    assert meta["complete"] and meta["notes"] == []
    assert (meta["frames_received"], meta["frames_tracked"], meta["package_frames"]) == (N - 1, N - 1, N)
    assert meta["frames_dropped"] == 1 and meta["frames_bad"] == 1
    assert meta["seconds"] == N / FPS and meta["tracks"] == 3          # two people, a bicycle

    live = root / "live"
    assert sorted(p.name for p in live.iterdir()) == ["aerial.jpg", "recorder_pose.rec",
                                                       "scene.json", "terrain.f32", "tracks.bin",
                                                       "tracks_index.json", "video.mp4"]
    scene = json.loads((live / "scene.json").read_text(encoding="utf-8"))
    assert scene["realtime"] is True and scene["growing"] is False and scene["crs"] is None
    assert scene["clip"] == "realtime"
    assert (scene["fps"], scene["frames"], scene["image_size"]) == (FPS, N, [W, H])
    assert scene["aerial"]["blank"] is True and scene["video"]["blurred"] is True
    # the classes it tracks, named from the first write (the view's rules then = a reopened one's)
    assert growing["growing"] is True and growing["tracks"]["classes"] == scene["tracks"]["classes"]
    assert set(scene["tracks"]["classes"].values()) == set(cfg.classes) >= {"person", "bicycle"}
    tr = scene["terrain"]
    assert np.fromfile(live / "terrain.f32", "<f4").shape == (tr["nx"] * tr["ny"],)

    # the tracks: as tracking the same detections frame by frame, through the fixed camera
    h = Homography.load(root / "_working" / "homography_used.json")
    assert h.camera_params["tilt_deg"] == pytest.approx(20, abs=0.1)
    got = np.fromfile(live / "tracks.bin", TRACK_DTYPE)
    want = replay(cfg, sent, h)
    assert len(got) == len(want) == scene["tracks"]["n_points"] > 2 * (N - 5)
    assert [(int(r["track_id"]), int(r["frame"])) for r in got] == [w[:2] for w in want]
    assert np.allclose(got["x"], [w[2] for w in want], atol=1e-3)
    assert np.allclose(got["y"], [w[3] for w in want], atol=1e-3)
    assert np.all(np.diff(got["frame"].astype(int)) >= 0) and SKIP not in got["frame"]
    assert np.allclose(got["t"], got["frame"] / FPS) and np.all(got["z"] == 0)
    assert set(got["class"]) == {0, 1} and np.all(np.isfinite(got["speed"]))
    firsts = {}
    for r in got:
        firsts.setdefault(int(r["track_id"]), float(r["speed"]))
    assert set(firsts.values()) == {0.0}                   # no speed before a second point
    index = json.loads((live / "tracks_index.json").read_text(encoding="utf-8"))
    assert sorted(index) == sorted(str(t) for t in firsts)
    assert sorted(v["class"] for v in index.values()) == ["bicycle", "person", "person"]
    P = np.array(scene["camera"]["P"])
    for r in got[::7]:                                      # the scene's camera sees each point
        x = P @ [r["x"], r["y"], 0, 1]                      # where its foot was
        tid, i = int(r["track_id"]), int(r["frame"])
        foots = [((a + c) / 2, d) for a, _, c, d in boxes(i)[0][[0, 1, 3]]]
        assert min(math.dist(x[:2] / x[2], f) for f in foots) < 3, (tid, i)    # ByteTrack's box

    # the videos: one frame per slot, the missing one repeated; blurred where anyone may be
    raw, vid = frames_of(root / "raw.mkv"), frames_of(live / "video.mp4")
    assert len(raw) == len(vid) == N
    assert [number(f) for f in raw] == [i if i != SKIP else SKIP - 1 for i in range(N)]
    assert [number(f) for f in vid] == [i if i != SKIP else SKIP - 1 for i in range(N)]
    for (x1, y1, x2, y2) in boxes(10)[0][:3].astype(int):  # the unsure person blurred too
        assert vid[10][y1:y2, x1:x2].std() < 0.4 * raw[10][y1:y2, x1:x2].std()
    assert vid[10][300:350, 300:400].std() > 0.8 * raw[10][300:350, 300:400].std()  # nobody here
    kf = key_frames(live / "video.mp4")                     # a keyframe every second: Live seeks it
    assert kf[0] == 0 and max(np.diff(kf + [N])) <= 30 < N

    poses = np.fromfile(live / "recorder_pose.rec", POSE_DTYPE)
    assert poses["k"].tolist() == sent and poses["seq"].tolist() == sent
    assert np.allclose(poses["q"], Q_CAL, atol=1e-6)
    assert (root / "_working" / "config_used.yaml").is_file()
    log = (root / "realtime_log.txt").read_text(encoding="utf-8")
    assert "calibrated from 29 poses (a still second)" in log and "package complete" in log

    [listed] = list_recordings(tmp_path)
    assert listed["id"] == rid and listed["complete"] and not listed["growing"]
    assert listed["tracks"] == 3 and listed["run"] == str(root)
    assert listed["title"] == recording_title(rid)

    # in Live: a site of its own, the camera called by the recording's time
    from groundtrack.live_site import site_cameras

    info = site_cameras(root, [])
    assert info["site"] == "Realtime" and info["start_spot"] == recording_title(rid)
    assert [c["name"] for c in info["cameras"]] == [recording_title(rid)]
    assert info["cameras"][0]["id"] == "realtime"            # file names: the id, never the title


def test_a_short_recording_is_still_calibrated_and_saved(tmp_path, ffmpeg):
    rt = LiveTracker(tmp_path, cfg=default_config(), model=FakeYOLO(), log=lambda m: None)
    rid = rt.begin(W, H)["id"]
    noise = np.zeros((H, W, 3), np.uint8)
    for i in range(6):                                      # 0.2 s: shorter than the calibration
        rt.frame({"seq": i, "t": i / FPS, "w": W, "h": H, "q": None}, picture(i, noise))
        assert until(lambda i=i: rt.rec.n_tracked == i + 1)
    rt.end()
    assert until(lambda: rt.saved is not None, 60)
    assert rt.saved["complete"] is True and rt.saved["calibration"]["tilt_deg"] == 10.0
    assert json.loads((tmp_path / "runs/live" / rid / "live/scene.json").read_text())["frames"] == 6


def test_a_recording_measures_itself(tmp_path, ffmpeg, monkeypatch):
    monkeypatch.setattr(live_track, "STATS_S", 0.2)
    rt = LiveTracker(tmp_path, cfg=default_config(), model=FakeYOLO(), log=lambda m: None)
    rid = rt.begin(W, H)["id"]
    noise = np.zeros((H, W, 3), np.uint8)
    for i in range(12):                                     # the phone has skipped 2 a frame
        rt.frame({"seq": i, "t": i / FPS, "w": W, "h": H, "q": None, "skip": 2 * i},
                 picture(i, noise))
        assert until(lambda i=i: rt.rec.n_tracked == i + 1, 120 if i == 0 else 10)
        time.sleep(0.06)
    rt.end()
    assert until(lambda: rt.saved is not None, 60)
    root = tmp_path / "runs" / "live" / rid
    with open(root / "stats.csv", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        assert rd.fieldnames == STATS_COLUMNS
        rows = list(rd)
    assert len(rows) >= 2
    assert [float(r["t_s"]) for r in rows] == sorted(float(r["t_s"]) for r in rows)
    assert (int(rows[-1]["frames_in"]), int(rows[-1]["frames_tracked"])) == (12, 12)
    assert sum(int(r["phone_skipped"]) for r in rows if r["phone_skipped"]) == 22
    assert any(float(r["fps_tracked"]) > 0 and r["lag_ms"] != "" for r in rows)
    st = json.loads((root / "recording.json").read_text(encoding="utf-8"))["stats"]
    assert st["file"] == "stats.csv" and st["every_s"] == 0.2 and st["rows"] == len(rows)
    assert st["phone_skipped"] == 22 and st["fps_tracked"] > 0 and st["lag_ms"] is not None
    [listed] = list_recordings(tmp_path)
    assert listed["fps_tracked"] == st["fps_tracked"]


def test_the_frame_size_and_the_recordings_title(tmp_path):
    rt = LiveTracker(tmp_path, cfg=default_config(), model=FakeYOLO(), log=lambda m: None)
    assert rt.long_side == 1920 and rt.status()["long_side"] == 1920
    rt.set_long_side(1280)                                  # 720p: the next recording
    assert rt.status()["long_side"] == 1280
    for bad in (1000, 3840, True, "1280", None):
        with pytest.raises(ValueError):
            rt.set_long_side(bad)
    assert rt.long_side == 1280
    rt.close()
    assert recording_title("20261030-110119") == "Rec 30 Oct 11:01"
    assert recording_title("odd") == "Rec odd"


def test_nothing_tracked_leaves_no_package(tmp_path, ffmpeg):
    rt = LiveTracker(tmp_path, cfg=default_config(), model=FakeYOLO(), log=lambda m: None)
    rid = rt.begin(W, H)["id"]
    rt.end()
    assert until(lambda: rt.saved is not None, 60)
    assert rt.saved["complete"] is False
    assert any("nothing was tracked" in n for n in rt.saved["notes"])
    assert not (tmp_path / "runs/live" / rid / "live/scene.json").exists()
    [listed] = list_recordings(tmp_path)
    assert not listed["complete"] and not listed["growing"]


def test_a_recorder_that_drops_out_comes_back_or_the_recording_ends(tmp_path, ffmpeg, monkeypatch):
    monkeypatch.setattr(live_track, "LOST_S", 0.5)
    rt = LiveTracker(tmp_path, cfg=default_config(), model=FakeYOLO(), log=lambda m: None)
    rid = rt.begin(W, H)["id"]
    noise = np.zeros((H, W, 3), np.uint8)
    for i in range(3):
        rt.frame({"seq": i, "t": i / FPS, "w": W, "h": H, "q": None}, picture(i, noise))
        assert until(lambda i=i: rt.rec.n_tracked == i + 1)
    rt.lost()
    assert rt.status()["lost"] is True
    assert rt.begin(W, H, resume=True) == {"id": rid, "resumed": True}
    assert rt.status()["lost"] is False
    time.sleep(0.6)                                         # its timer finds it back: no end
    assert rt.rec is not None and not rt.rec.closing
    # the phone's clock may have jumped: frame numbers go on from where the laptop's clock stands
    rt.frame({"seq": 3, "t": 500.0, "w": W, "h": H, "q": None}, picture(3, noise))
    assert until(lambda: rt.rec.n_tracked == 4)
    assert rt.rec.k_raw > 2
    rt.lost()                                               # and this time it does not come back
    assert until(lambda: rt.saved is not None, 30)
    assert rt.saved["id"] == rid and rt.saved["frames_received"] == 4
    with pytest.raises(ValueError):
        rt.begin(10, 10)
    assert rt.begin(W, H)["id"] != rid                      # a new one, not the old
    rt.close()
    assert rt.mode == "off" and rt.rec is None


def test_exports_of_a_recording_say_local_metres(tmp_path):
    """A recording's camera (saved in its _working/) has no CRS: its pass exports say "local
    metres" and their GeoJSON carries no EPSG:27700 (live_pass_export._crs)."""
    from groundtrack.live_pass_export import _crs

    h, *_ = sensor_camera([], (1920, 1080))
    h.save(tmp_path / "homography_used.json")
    assert _crs(Homography.load(tmp_path / "homography_used.json")) == "local metres"
    assert _crs(Homography(H=np.eye(3), origin=(500000.0, 170000.0))) == "EPSG:27700"


def test_the_phones_own_pose_gives_the_camera():
    """The recorder's q as phone.js makes it (deviceorientation -> pose v1), through
    sensor_camera: the tilt the phone shows, its roll, whichever way it faced."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    from groundtrack import live_hub

    # (alpha, beta, gamma, screen angle): landscape, 30 deg down facing north / east; level,
    # rolled 20 deg clockwise; the other landscape hold, 12 deg down
    cases = [(90, 0, -60, 90), (0, 0, -60, 90), (90, 20, -90, 90), (270, 0, 78, 270)]
    src = (live_hub.PHONE_WEB / "phone.js").as_uri()
    code = (f'import {{cameraQuaternion, tiltRoll}} from "{src}";\n'
            f"const qs = {json.dumps(cases)}.map(c => cameraQuaternion(...c));\n"
            "console.log(JSON.stringify({qs, tr: qs.map(tiltRoll)}));")
    r = subprocess.run([node, "--input-type=module", "-e", code], capture_output=True,
                       encoding="utf-8", timeout=60, check=False)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    want = [(30, 0), (30, 0), (0, 20), (12, 0)]
    for q, (tilt_up, roll_cw), (down, roll) in zip(out["qs"], out["tr"], want):
        assert (-tilt_up, abs(roll_cw)) == pytest.approx((down, roll), abs=1e-6)   # the phone's
        cp = sensor_camera([q] * 5, (1920, 1080))[0].camera_params
        assert cp["tilt_deg"] == pytest.approx(down, abs=1e-6)
        assert abs(cp["roll_deg"]) == pytest.approx(roll, abs=1e-6) and cp["yaw_deg"] == 0


# --------------------------------------------------------------------------- the phone's calibration
def turned(q, deg, axis):
    """q turned by deg about the camera's own axis (0: x, its tilt; 2: z, its roll)."""
    a = math.radians(deg)
    c, s_ = math.cos(a), math.sin(a)
    T = {0: [[1, 0, 0], [0, c, -s_], [0, s_, c]], 2: [[c, -s_, 0], [s_, c, 0], [0, 0, 1]]}[axis]
    return to_quat(quat_matrix(q) @ np.array(T, float))


def still_state():
    return SimpleNamespace(n_q=0, win=deque(), still_s=0.0, cal_qs=[], cal_ready=False,
                           base=(0.0, 0), closing=False)


def test_the_calibration_waits_for_a_still_second():
    """The poses of a still second (each within STILL_DEG of their mean) calibrate; a move
    starts the second again; the hand's shake is still."""
    rec = still_state()
    rng = np.random.default_rng(3)
    t = 0.0
    for i in range(20):                                     # framing: 1 deg a frame
        t = i / FPS
        Recording._still(rec, t, turned(Q_CAL, i, 0))
        assert rec.still_s < 0.15 and not rec.cal_ready         # a few frames, at most
        assert not Recording.cal_due(rec, t)
    q_still = turned(Q_CAL, 19, 0)
    for i in range(20, 20 + 25):                            # still (a shake of 0.2 deg) ...
        t = i / FPS
        Recording._still(rec, t, turned(turned(q_still, rng.normal(0, 0.2), 0), rng.normal(0, 0.2), 2))
    assert 0.6 < rec.still_s < CAL_S and not rec.cal_ready
    Recording._still(rec, t + 1 / FPS, turned(q_still, 2.5, 0))     # ... moved again: from here
    assert rec.still_s == 0 and not rec.cal_ready
    q_still = turned(q_still, 2.5, 0)
    t0 = t + 1 / FPS
    for j in range(1, 40):
        Recording._still(rec, t0 + j / FPS, q_still)
        if rec.cal_ready:
            break
    assert rec.cal_ready and rec.still_s == pytest.approx(CAL_S, abs=1.6 / FPS)
    assert Recording.cal_due(rec, t0 + j / FPS)
    assert all(np.allclose(q, q_still) for q in rec.cal_qs) and len(rec.cal_qs) >= 29
    h = sensor_camera(rec.cal_qs, (1920, 1080))[0]
    assert h.camera_params["tilt_deg"] == pytest.approx(20 - 21.5, abs=0.01)    # 21.5 deg up


def test_the_calibration_does_not_wait_forever():
    rec = still_state()
    assert not Recording.cal_due(rec, CAL_S - 0.01)
    assert Recording.cal_due(rec, CAL_S)                    # no poses at all: no motion sensors
    rec.n_q = 5                                             # poses, never still
    assert not Recording.cal_due(rec, CAL_TIMEOUT_S - 0.01)
    assert Recording.cal_due(rec, CAL_TIMEOUT_S)
    rec.closing = True                                      # the recording ends first
    assert Recording.cal_due(rec, 0.1)


@pytest.mark.parametrize("roll", [0, 6])
def test_the_tilt_nudge_and_the_height(roll):
    """The phone's tilt nudge turns the camera about its own x axis (+: more down); the height
    puts the camera that high; a synthetic gravity (the pose) gives the tilt and roll it has."""
    q = to_quat(phone_pose(40, 12, roll))
    for nudge in (0, 2, -3.5):
        h, K, R_cw, C = sensor_camera([q] * 10, (1920, 1080), height_m=2.25, tilt_offset=nudge)
        cp = h.camera_params
        assert cp["tilt_deg"] == pytest.approx(12 + nudge, abs=0.05)
        assert abs(cp["roll_deg"]) == pytest.approx(roll, abs=0.1)     # rolled: nearly
        assert cp["height_m"] == 2.25 and C.tolist() == [0, 0, 2.25]
        assert h.method_info["tilt_offset_deg"] == nudge
        x = K @ R_cw @ (np.array([0.0, 10.0, 0.0]) - C)     # still on the ground, round trip
        assert np.allclose(h.to_world(np.array([x[:2] / x[2]]))[0], [0, 10], atol=1e-6)


@pytest.mark.parametrize("pan", [5, 20, -25])
def test_turning_the_recorder_drags_the_plan(pan):
    """The camera is fixed at the start: the recorder turned by pan (clockwise, deg) puts a
    standing person -pan deg round the camera, as far away (plans/live_controller/10: about
    -1 deg of plan per deg of pan)."""
    q0 = to_quat(phone_pose(123, 15, 0))
    h, K, R_cw, C = sensor_camera([q0] * 10, (1920, 1080))
    Rz = R_cw.T @ quat_matrix(q0).T                        # the sensors' frame -> the plan's
    R_pan = (Rz @ quat_matrix(to_quat(phone_pose(123 + pan, 15, 0)))).T
    for X in ([0.0, 10.0], [2.0, 14.0], [-1.0, 7.0]):
        x = K @ R_pan @ (np.r_[X, 0.0] - C)
        got = h.to_world(np.array([x[:2] / x[2]]))[0]
        turn = math.degrees(math.atan2(got[0], got[1]) - math.atan2(X[0], X[1]))
        assert turn == pytest.approx(-pan, abs=1e-6)
        assert math.hypot(*got) == pytest.approx(math.hypot(*X), abs=1e-6)


def people_rows(cam_height, tall, tilt=15.0, n=40):
    """Person boxes (RAW_COLUMNS) of people `tall` m standing 6 to 20 m from a camera
    cam_height m up, tilt deg down."""
    _, K, R_cw, C = sensor_camera([to_quat(phone_pose(0, tilt, 0))] * 3, (1920, 1080),
                                  height_m=cam_height)
    rows = []
    for i in range(n):
        X = np.array([-3 + 6 * (i % 7) / 6, 6 + 14 * i / (n - 1), 0.0])
        f, t = (K @ R_cw @ (X + [0, 0, dz] - C) for dz in (0.0, tall))
        (u, v), (_, v_top) = f[:2] / f[2], t[:2] / t[2]
        w = 0.25 * (v - v_top)
        rows.append((i + 1, "person", 0, i, i / FPS, u - w, v_top, u + w, v, 0.9, False))
    return rows


def check_of(rows, height_m, logs):
    rec = SimpleNamespace(person_rows=deque(rows), h=sensor_camera(
        [to_quat(phone_pose(0, 15, 0))] * 3, (1920, 1080), height_m=height_m)[0],
        height_m=height_m, check=None, _check_failed=False, log=logs.append)
    Recording.people_check(rec)
    return rec


def test_the_people_check_offers_the_height_that_fits():
    logs = []
    rec = check_of(people_rows(1.5, 1.7), 1.5, logs)        # right: people come out 1.7 m
    assert rec.check["ok"] and rec.check["median_m"] == pytest.approx(1.7, abs=0.02)
    assert rec.check["height_m"] == pytest.approx(1.5, abs=0.02) and rec.check["n"] == 40
    assert len(logs) == 1 and "people 1.70 m tall" in logs[0]
    Recording.people_check(rec)                             # the same verdict: not logged again
    assert len(logs) == 1
    rec = check_of(people_rows(2.0, 1.7), 1.5, logs)        # the phone is higher than it says
    assert not rec.check["ok"] and rec.check["error_pct"] == pytest.approx(-25, abs=1)
    assert rec.check["height_m"] == pytest.approx(2.0, abs=0.03)
    assert "the camera nearer 1.9" in logs[-1] or "the camera nearer 2.0" in logs[-1]
    assert check_of(people_rows(1.5, 1.7, n=20), 1.5, logs).check is None    # too few people


def test_the_phones_height_and_tilt_and_what_the_live_view_gets(tmp_path, ffmpeg):
    """begin takes the phone's height and tilt nudge; the status shows the still second; the
    Live view gets the points so far (whole frames only: none at or past "frames"), the same
    bytes as tracks.bin, from any point on; a blurred preview while it records; the package's
    points once saved."""
    rt = LiveTracker(tmp_path, cfg=default_config(), model=FakeYOLO(), log=lambda m: None)
    for bad in ({"height_m": 9.0}, {"height_m": 0.1}, {"tilt_offset": 25.0}):
        with pytest.raises(ValueError):
            rt.begin(W, H, **bad)
    rid = rt.begin(W, H, height_m=2.0, tilt_offset=3.0)["id"]
    rec = rt.rec
    noise = np.random.default_rng(2).integers(0, 255, (H, W, 3), dtype=np.uint8)
    for i in range(40):
        rt.frame({"seq": i, "t": i / FPS, "w": W, "h": H, "q": Q_CAL}, picture(i, noise))
        assert until(lambda i=i: rec.n_tracked == i + 1, 120 if i == 0 else 10)
        if i == 15:
            st = rt.status()
            assert st["calibrated"] is False and st["still_s"] == pytest.approx(0.5, abs=0.05)
            assert (st["height_m"], st["tilt_offset_deg"], st["check"]) == (2.0, 3.0, None)
            head, data = rt.tracks_since(rid)
            assert (head["n"], head["frames"], data) == (0, 0, b"")     # nothing placed yet
    st = rt.status()
    assert st["calibrated"] and st["camera"]["tilt_deg"] == pytest.approx(23, abs=0.1)
    assert st["camera"]["height_m"] == 2.0 and st["still_s"] == pytest.approx(CAL_S, abs=0.05)
    head, data = rt.tracks_since(rid, 0)
    got = np.frombuffer(data, TRACK_DTYPE)
    assert head == {"id": rid, "n0": 0, "n": len(got), "frames": 40, "done": False}
    assert len(got) > 40 and got["frame"].max() < head["frames"]
    assert data == (tmp_path / "runs/live" / rid / "live/tracks.bin").read_bytes()
    head2, tail = rt.tracks_since(rid, head["n"] - 3)
    assert head2["n0"] == head["n"] - 3 and tail == data[-3 * TRACK_DTYPE.itemsize:]
    assert rt.tracks_since(rid, 10 ** 6)[0]["n0"] == head["n"]
    scene = json.loads((tmp_path / "runs/live" / rid / "live/scene.json").read_text())
    assert scene["growing"] is True and scene["R_max"] == R_MAX_M     # fixed from the start
    jpg = rt.preview(rid)
    assert jpg[:2] == b"\xff\xd8"
    assert cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR).shape == (H, W, 3)
    assert rt.preview("20200101-000000") is None
    for bad in ("../x", "", "20261030-1200"):
        with pytest.raises(ValueError):
            rt.tracks_since(bad)
    rt.end()
    assert until(lambda: rt.saved is not None, 60)
    assert rt.preview(rid) is None
    head, data = rt.tracks_since(rid)                      # from the package now
    live = tmp_path / "runs/live" / rid / "live"
    assert head["done"] is True and head["frames"] == 40
    assert data == (live / "tracks.bin").read_bytes() and head["n"] == len(data) // TRACK_DTYPE.itemsize
    meta = json.loads((tmp_path / "runs/live" / rid / "recording.json").read_text())
    assert (meta["height_m"], meta["tilt_offset_deg"]) == (2.0, 3.0)
    rt.close()
    assert package_tracks(tmp_path, rid) == (head, data)
    odd = tmp_path / "runs/live/20200101-000000/live"
    odd.mkdir(parents=True)
    (odd / "scene.json").write_text(json.dumps({"growing": True, "frames": 3}))
    with pytest.raises(FileNotFoundError):                 # left growing (a crash): no package
        package_tracks(tmp_path, "20200101-000000")
