"""Realtime (plans/live_controller/10, steps 1-3): one phone films, this machine tracks.

The recorder (live_phone/record.js) sends its frames as JPEG through the live hub (/ws/record).
Here YOLO + ByteTrack track them as they come (detect.FrameTracker: the same detections as
`groundtrack process`), every foot point goes onto the ground through the camera fixed at the
start, and each recording becomes a scene package that today's Live view plays:

    runs/live/<YYYYMMDD-HHMMSS>/          one recording (a site of one camera)
      raw.mkv                 the frames as filmed (JPEG, not blurred) at FPS, kept for an
                              offline re-track; never served by the app, never sent anywhere
      recording.json          what happened: times, sizes, frames received / tracked, the
                              calibration, the model, the stats' medians
      stats.csv               every STATS_S: frames in, tracked, lag, people, the phone's skipped
                              frames (STATS_COLUMNS): a long recording measures itself
      realtime_log.txt        what the tracker did
      _working/homography_used.json   the fixed camera (the sensor calibration below)
      _working/config_used.yaml       the detection and the machine's rules used
      live/                   the scene package (live_export.py), written as it goes
        scene.json            "realtime": true; "growing": true until Stop; tracks.classes:
                              every class it tracks, from the start
        tracks.bin            appended frame by frame (seen points only: guessing is rules.js's);
                              the same points are kept in memory for the Live view
                              (LiveTracker.tracks_since), whole frames only
        tracks_index.json     written at Stop
        video.mp4             blurred (blurred.mkv while recording, remuxed at Stop)
        terrain.f32           flat
        aerial.jpg            black: there is no aerial (open it with ground=C, the grid)
        recorder_pose.rec     the recorder's pose with every frame it sent (POSE_DTYPE)

Frames: the package's frame k is the recorder's capture time t (its own clock) k / FPS after its
first frame; a slot no frame arrives for repeats the last frame in the videos, so the package
plays at a constant FPS. A decoder thread decodes every frame as it comes; tracking takes the
newest decoded one whenever it is free (latest wins), so decoding the next frame overlaps YOLO.
A frame tracking skips is in raw.mkv but has no tracks, and video.mp4 repeats the last tracked
frame instead (blurred), so no frame is ever shown unblurred. The recorder's frame size is the
laptop's choice (LiveTracker.long_side: 1080p or 720p), told to the phone when it connects.

Calibration (step 2: no button): the recorder's pose (pose v1 q: world_from_camera, OpenCV axes,
in a gravity frame) averaged over its first still second (CAL_S, every pose within STILL_DEG of
their mean; never still for CAL_TIMEOUT_S: the last second); the start direction is "north" (yaw
0); the ground flat, the phone's height (remembered on the phone, HEIGHT_M by default) below the
camera; a tilt nudge from the phone added; the focal length FOCAL_1920 px at 1920 wide (the
Richmond calibrations: check it once against Safari's stream). Fixed from then on: the
recorder's later moves are not corrected (plans/live_controller/10, "Calibrated once, then
kept"); a new height or tilt is a new recording. Without poses (no motion sensors):
DEFAULT_TILT_DEG down, level. Then, in the background, selfcal.people_check: people off 1.7 m by
more than CHECK_TOL and the phone offers a height (and tilt) nudge. The reach is R_MAX_M, fixed
from the start (offline it comes from every point, which a recording does not have yet).

The Live view (step 3) draws a recording as it grows: tracks_since (the points from n on and the
frames tracked so far) and preview (the newest blurred picture, PREVIEW_W wide), both from
memory, served by the app on this machine only.

The hub calls begin / frame / end / lost / status from its own thread; nothing here blocks it
(frames go to the tracking thread, latest wins, and to the raw writer's queue).
"""

from __future__ import annotations

import json
import math
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from .config import COCO_IDS, Config, default_config
from .homography import Homography
from .live_export import (
    CELL_M,
    KEYFRAME_S,
    SPEED_RAMP,
    TERRAIN_MARGIN_M,
    TRACK_DTYPE,
    TRACK_LAYOUT,
    _angles,
)

FPS = 30.0                  # the package's frame rate (the recorder films at 30)
CAL_S = 1.0                 # the poses averaged for the calibration: a still second
STILL_DEG = 1.5             # still: every pose of that second within this of their mean
CAL_TIMEOUT_S = 15.0        # never still this long: calibrated from the last second anyway
HEIGHT_M = 1.50             # the camera above the ground, unless the phone says (a handheld phone)
HEIGHT_RANGE = (0.3, 5.0)   # m: the heights a recorder may give
TILT_NUDGE_MAX = 20.0       # deg: the phone's tilt nudge (+: more down), at most
R_MAX_M = 50.0              # m: a recording's reach, fixed at the start
PREVIEW_W = 640             # the Live view's picture: this long side (blurred) ...
PREVIEW_S = 0.1             # ... at most this often
CHECK_EVERY_S = 3.0         # the people check this often, once enough people were seen
CHECK_ROWS = 3000           # the people rows it keeps (the newest)
CHECK_TOL = 0.15            # people off 1.7 m by more than this: the phone offers nudges
FOCAL_1920 = 1597.7         # px at 1920 wide: every Richmond calibration (iPhone, 1x)
DEFAULT_TILT_DEG = 10.0     # no poses: the camera this far down, level
PRIVACY_CONF = 0.10         # boxes this sure are blurred; the tracker takes the config's conf
BLUR_PAD = 0.15             # each blurred box grown by this much of its size
HOLD_S = 0.3                # a box stays blurred this long after it was last found
SPEED_ALPHA = 0.35          # the drawn speed: an exponential average of the step speeds
LOST_S = 10.0               # a recorder that drops out may come back this long
RAW_QUEUE = 240             # frames waiting for raw.mkv (8 s): more are dropped, and counted
STATS_S = 5.0               # a stats.csv row this often
LONG_SIDES = (1920, 1280)   # the recorder's frame size, long side: 1080p (default) or 720p
STATS_COLUMNS = ["t_s", "frames_in", "frames_tracked", "fps_in", "fps_tracked", "lag_ms",
                 "lag_ms_max", "people", "tracks", "phone_skipped", "lost"]
MIN_SIDE, MAX_SIDE = 64, 4096
POSE_DTYPE = np.dtype([("k", "<u4"), ("seq", "<u4"), ("t", "<f8"), ("q", "<f4", (4,))])
MODES = ("off", "loading", "ready", "error")
RECORDING_ID = re.compile(r"\d{8}-\d{6}(-\d+)?")


def recordings_dir(project: Path) -> Path:
    return Path(project) / "runs" / "live"


# --------------------------------------------------------------------------- the camera
def mean_rotation(Rs) -> np.ndarray:
    """The rotation nearest the mean of rotation matrices (n x 3 x 3)."""
    U, _, Vt = np.linalg.svd(np.asarray(Rs, float).reshape(-1, 3, 3).sum(0))
    return U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt


def quat_matrix(q) -> np.ndarray:
    """[w, x, y, z] -> its rotation matrix."""
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def level_camera(tilt_deg: float = DEFAULT_TILT_DEG) -> np.ndarray:
    """world_from_camera (OpenCV axes) of a level camera facing north, tilt_deg down."""
    a = math.radians(tilt_deg)
    right, fwd = np.array([1.0, 0, 0]), np.array([0.0, math.cos(a), -math.sin(a)])
    return np.column_stack([right, np.cross(fwd, right), fwd])


def sensor_camera(qs, size, height_m: float = HEIGHT_M, focal_1920: float = FOCAL_1920,
                  tilt_offset: float = 0.0):
    """The fixed camera from the recorder's poses (pose v1 q, world_from_camera in a gravity
    frame, heading arbitrary): (Homography pixel -> local ground, K, R_cw, C). The start
    direction becomes north; the ground is flat, height_m below the camera at (0, 0);
    tilt_offset (deg, + down) turns the camera about its own x axis (the phone's nudge)."""
    W, Hh = int(size[0]), int(size[1])
    f = focal_1920 * max(W, Hh) / 1920.0            # the lens's focal at this frame's long side
    qs = [q for q in (qs or []) if q is not None and np.all(np.isfinite(q))]
    R = mean_rotation([quat_matrix(q) for q in qs]) if qs else level_camera()
    if tilt_offset:
        a = math.radians(tilt_offset)               # forward towards the camera's down (+y)
        R = R @ np.array([[1.0, 0, 0], [0, math.cos(a), math.sin(a)], [0, -math.sin(a), math.cos(a)]])
    fwd = R[:, 2]
    look = fwd if math.hypot(fwd[0], fwd[1]) > 1e-3 else -R[:, 1]     # straight down: its top
    psi = math.atan2(look[0], look[1])                                 # heading, from north
    Rz = np.array([[math.cos(psi), -math.sin(psi), 0], [math.sin(psi), math.cos(psi), 0],
                   [0, 0, 1.0]])
    R_cw = (Rz @ R).T
    C = np.array([0.0, 0.0, float(height_m)])
    K = np.array([[f, 0, W / 2], [0, f, Hh / 2], [0, 0, 1.0]])
    G = K @ np.column_stack([R_cw[:, 0], R_cw[:, 1], -R_cw @ C])      # ground (x, y, 1) -> pixel
    H = np.linalg.inv(G)
    H /= H[2, 2]
    w_sign = 1.0
    for b in np.radians(np.arange(0, 360, 15)):                       # a ground point in front
        X = np.array([10 * math.sin(b), 10 * math.cos(b), 0.0])
        x = K @ R_cw @ (X - C)
        if x[2] > 0.1:
            w_sign = float(np.sign((H @ (x / x[2]))[2])) or 1.0
            break
    h = Homography(H=H, origin=(0.0, 0.0), image_size=(W, Hh), crs="local")
    h.w_sign = w_sign
    yaw, tilt, roll = _angles(R_cw)
    yaw = 0.0 if min(yaw, 360 - yaw) < 1e-6 else yaw                  # north, not 360
    h.camera_params = {"focal_px": f, "yaw_deg": yaw, "tilt_deg": tilt, "roll_deg": roll,
                       "hfov_deg": float(np.degrees(2 * np.arctan(W / (2 * f)))),
                       "height_m": float(height_m), "E": 0.0, "N": 0.0}
    h.method = "realtime sensors"
    h.method_info = {"poses": len(qs), "focal_1920": focal_1920, "tilt_offset_deg": float(tilt_offset),
                     "note": "the recorder's pose over a still second; heading = the start "
                             "direction; flat ground" + ("" if qs else
                                                         f"; no poses: {DEFAULT_TILT_DEG:g} deg "
                                                         "down, level")}
    return h, K, R_cw, C


def realtime_config() -> Config:
    """groundtrack's defaults (the detection and the machine's rules); the model is the repo's
    models/ copy, never downloaded."""
    from .live_export import _still_model

    cfg = default_config()
    _still_model(cfg)
    return cfg


# --------------------------------------------------------------------------- ffmpeg pipes
class _Pipe:
    """An ffmpeg process fed through stdin; its messages kept for errors."""

    def __init__(self, args: list[str]):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("realtime needs ffmpeg on the PATH")
        self._err = tempfile.TemporaryFile()  # noqa: SIM115  (open while ffmpeg runs)
        self.proc = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-hide_banner", *args],
                                     stdin=subprocess.PIPE, stderr=self._err,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.failed: str | None = None

    def write(self, data: bytes) -> bool:
        if self.failed:
            return False
        try:
            self.proc.stdin.write(data)
            return True
        except OSError:
            self.failed = self._says()
            return False

    def close(self) -> str | None:
        """None when ffmpeg finished well, else what it said."""
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        code = self.proc.wait(60)
        if code and not self.failed:
            self.failed = self._says()
        return self.failed

    def _says(self) -> str:
        try:
            self.proc.wait(5)
            self._err.seek(0)
            return self._err.read().decode("utf-8", "replace").strip()[-400:] or "no message"
        except Exception as e:  # noqa: BLE001
            return str(e)


def _remux(src: Path, out: Path) -> str | None:
    """blurred.mkv -> video.mp4 (+faststart), no re-encode. None when it worked."""
    r = subprocess.run([shutil.which("ffmpeg") or "ffmpeg", "-y", "-loglevel", "error",
                        "-hide_banner", "-i", str(src), "-c", "copy", "-movflags", "+faststart",
                        str(out)], capture_output=True, text=True, timeout=600, check=False,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if r.returncode == 0:
        return None
    return r.stderr.strip()[-400:] or f"exit {r.returncode}"


# --------------------------------------------------------------------------- a recording
class Recording:
    """One recording: its folder, the writers, the tracking state. Made by LiveTracker.begin."""

    def __init__(self, root: Path, size: tuple[int, int], cfg: Config, log,
                 height_m: float = HEIGHT_M, tilt_offset: float = 0.0):
        self.root, self.size, self.cfg = root, (int(size[0]), int(size[1])), cfg
        self.height_m, self.tilt_offset = float(height_m), float(tilt_offset)
        self.live = root / "live"
        self.live.mkdir(parents=True)
        (root / "_working").mkdir()
        self._logf = open(root / "realtime_log.txt", "a", encoding="utf-8")  # noqa: SIM115  (closed in finish)
        self._log = log
        self.id = root.name
        self.started = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.t_start = time.monotonic()
        self.ended: str | None = None
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.latest = None                   # (k, t, rx time, jpeg): the newest frame not decoded
        self.ready = None                    # (k, t, rx time, image): the newest decoded, not tracked
        self.decoded_all = False             # closing, and every frame received is decoded
        self.wake_tr = threading.Event()     # a decoded frame is ready (wake: a frame came)
        self.closing = False
        self.lost_at: float | None = None
        self.reanchor = False
        self.base: tuple[float, int] | None = None     # (recorder t, k) frame numbers count from
        self.k_raw = -1                      # the last frame number received
        self.last_rx = 0.0
        self.n_in = self.n_dropped = self.n_bad = self.n_tracked = self.n_raw_dropped = 0
        self.n_points = 0
        self.rx_times: deque = deque(maxlen=90)
        self.tr_times: deque = deque(maxlen=90)
        self.lag_ms: deque = deque(maxlen=30)
        self.people = 0
        self.phone_skip: int | None = None   # frames the phone did not send (its count, in the header)
        # stats.csv
        self.stats_f = open(root / "stats.csv", "w", encoding="utf-8", newline="")  # noqa: SIM115
        self.stats_f.write(",".join(STATS_COLUMNS) + "\n")
        self.stats_rows: list[dict] = []
        self._win = (self.t_start, 0, 0, None)       # this row's start: (time, in, tracked, phone skip)
        self._win_lags: list[float] = []
        # calibration: a still second (_still)
        self.cal_qs: list = []
        self.win: deque = deque()            # (t, q): the poses since the recorder was last moved
        self.n_q = 0                         # frames with a pose
        self.cal_ready = False               # a still second came: cal_qs are its poses
        self.still_s = 0.0                   # how long it has been still (the phone's ring)
        self.h: Homography | None = None
        self.K = self.R_cw = self.C = None
        self.pending: list[tuple] = []       # rows tracked before the calibration
        # tracks
        self.tracks_f = open(self.live / "tracks.bin", "wb")  # noqa: SIM115
        self.index: dict[str, dict] = {}
        self.last: dict[int, tuple] = {}     # track -> (t, x, y, speed)
        # the Live view's: every point so far, the frames tracked, the newest small picture
        self.records = bytearray()
        self.upto = 0                        # frames below this are whole (tracked, or skipped)
        self.preview: tuple[int, bytes] | None = None
        self._preview_t = 0.0
        # the people check (selfcal.people_check)
        self.person_rows: deque = deque(maxlen=CHECK_ROWS)
        self.check: dict | None = None
        self._check_t = 0.0
        self._check_failed = False
        # videos and poses
        self.raw_q: queue.Queue = queue.Queue(RAW_QUEUE)
        self.raw = _Pipe(["-f", "image2pipe", "-framerate", f"{FPS:g}", "-c:v", "mjpeg", "-i", "-",
                          "-c:v", "copy", "-an", str(root / "raw.mkv")])
        W, Hh = self.size[0] - self.size[0] % 2, self.size[1] - self.size[1] % 2
        self.vsize = (W, Hh)
        self.blur = _Pipe(["-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{Hh}", "-r", f"{FPS:g}",
                           "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
                           "-g", str(round(KEYFRAME_S * FPS)),
                           "-pix_fmt", "yuv420p", "-an", str(self.live / "blurred.mkv")])
        self.k_blur = -1                     # the last frame number written to the blurred video
        self.last_blur: np.ndarray | None = None
        self.held: deque = deque()           # (k, boxes): boxes blurred until HOLD_S later
        self.poses = open(self.live / "recorder_pose.rec", "wb")  # noqa: SIM115
        self.raw_thread = threading.Thread(target=self._raw_loop, name=f"rt-raw-{self.id}",
                                           daemon=True)
        self.raw_thread.start()
        self.dec_thread = threading.Thread(target=self._decode_loop, name=f"rt-dec-{self.id}",
                                           daemon=True)
        self.dec_thread.start()
        self.log(f"recording {self.id}: {self.size[0]}x{self.size[1]} at {FPS:g} fps -> {root}")

    def log(self, msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} realtime: {msg}"
        try:
            self._logf.write(line + "\n")
            self._logf.flush()
        except (OSError, ValueError):
            pass
        self._log(line)

    # ---- receiving (the hub's thread: quick, never blocks)
    def receive(self, head: dict, jpeg: bytes) -> bool:
        if self.closing:
            return False
        if (int(head["w"]), int(head["h"])) != self.size:
            self.n_bad += 1
            return False
        t, now = float(head["t"]), time.monotonic()
        with self.lock:
            if self.base is None:
                self.base = (t, 0)
            elif self.reanchor:              # back after a drop: on from where the clock stands
                self.base = (t, self.k_raw + max(1, round((now - self.last_rx) * FPS)))
                self.reanchor = False
            k = self.base[1] + round((t - self.base[0]) * FPS)
            if k <= self.k_raw:              # two frames in one slot (or the clock went back)
                self.n_dropped += 1
                return False
            self.k_raw, self.last_rx = k, now
            self.n_in += 1
            self.rx_times.append(now)
            q = head.get("q")
            if head.get("skip") is not None:
                self.phone_skip = int(head["skip"])
            if self.h is None and q is not None and not self.cal_ready:
                self._still(t, q)
            self.latest = (k, t, now, jpeg)
        rec = np.zeros(1, POSE_DTYPE)
        rec["k"], rec["seq"], rec["t"] = k, int(head.get("seq", 0)) & 0xFFFFFFFF, t
        rec["q"] = q if q is not None else np.nan
        try:
            self.poses.write(rec.tobytes())
        except ValueError:                   # closed meanwhile
            pass
        try:
            self.raw_q.put_nowait((k, jpeg))
        except queue.Full:
            self.n_raw_dropped += 1
        self.wake.set()
        return True

    def _still(self, t: float, q) -> None:
        """The calibration waits for a still second: the poses since the recorder last moved
        (more than STILL_DEG from their mean: the second starts again), once they span CAL_S,
        become cal_qs. still_s: how long it has been still. (Under self.lock.)"""
        self.n_q += 1
        w = self.win
        w.append((t, np.asarray(q, float)))
        while len(w) > 1 and w[0][0] < t - CAL_S:
            w.popleft()
        Q = np.array([x for _, x in w])
        Q *= np.where(Q @ Q[0] < 0, -1.0, 1.0)[:, None]     # one hemisphere: q and -q are one turn
        m = Q.sum(0)
        m /= np.linalg.norm(m)
        if 2 * math.degrees(math.acos(min(1.0, float(np.abs(Q @ m).min())))) > STILL_DEG:
            w.clear()                                       # it moved: from this pose on
            w.append((t, np.asarray(q, float)))
        self.still_s = w[-1][0] - w[0][0]
        if self.still_s >= CAL_S - 1.5 / FPS:
            self.cal_qs = [x.tolist() for _, x in w]
            self.cal_ready = True

    def cal_due(self, t: float) -> bool:
        """Whether to calibrate at the recorder's time t: a still second came, the recording
        ends, no pose at all after CAL_S (no motion sensors), or never still for CAL_TIMEOUT_S."""
        dt = t - self.base[0]
        return (self.cal_ready or self.closing or (self.n_q == 0 and dt >= CAL_S)
                or dt >= CAL_TIMEOUT_S)

    def _raw_loop(self) -> None:
        last_k, last = -1, None
        while True:
            item = self.raw_q.get()
            if item is None:
                break
            k, jpeg = item
            if last is not None:
                for _ in range(k - last_k - 1):          # empty slots: the last frame again
                    self.raw.write(last)
            self.raw.write(jpeg)
            last_k, last = k, jpeg

    def _decode_loop(self) -> None:
        """Every frame decoded as it comes; the newest decoded one waits for tracking (latest
        wins). Ends when the recording closes and the last frame is decoded."""
        while True:
            self.wake.wait(0.25)
            self.wake.clear()
            with self.lock:
                item, self.latest = self.latest, None
                if item is None:
                    if self.closing:
                        self.decoded_all = True
                        self.wake_tr.set()
                        return
                    continue
            k, t, rx, jpeg = item
            img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if img is None or (img.shape[1], img.shape[0]) != self.size:
                self.n_bad += 1
                continue
            with self.lock:
                self.ready = (k, t, rx, img)
            self.wake_tr.set()

    def take(self):
        """The newest decoded frame for tracking: (k, t, rx time, image), None, or "done" (closing
        and nothing left)."""
        with self.lock:
            item, self.ready = self.ready, None
            if item is None and self.decoded_all:
                return "done"
            return item

    # ---- stats.csv
    def stats(self, now: float | None = None) -> None:
        """A stats.csv row once STATS_S has passed since the last one (now=None: the last row,
        of whatever came since)."""
        t0, n_in0, n_tr0, skip0 = self._win
        t1 = time.monotonic() if now is None else now
        dt = t1 - t0
        if dt <= 0 or (now is not None and dt < STATS_S):
            return
        if now is None and (self.n_in, self.n_tracked, self.phone_skip) == (n_in0, n_tr0, skip0):
            return
        lags = self._win_lags
        row = {"t_s": round(t1 - self.t_start, 1), "frames_in": self.n_in,
               "frames_tracked": self.n_tracked, "fps_in": round((self.n_in - n_in0) / dt, 1),
               "fps_tracked": round((self.n_tracked - n_tr0) / dt, 1),
               "lag_ms": round(float(np.median(lags))) if lags else "",
               "lag_ms_max": round(max(lags)) if lags else "", "people": self.people,
               "tracks": len(self.index),
               "phone_skipped": self.phone_skip - (skip0 or 0) if self.phone_skip is not None else "",
               "lost": int(self.lost_at is not None)}
        self._win, self._win_lags = (t1, self.n_in, self.n_tracked, self.phone_skip), []
        self.stats_rows.append(row)
        try:
            self.stats_f.write(",".join(str(row[c]) for c in STATS_COLUMNS) + "\n")
            self.stats_f.flush()
        except (OSError, ValueError):
            pass

    def stats_summary(self) -> dict:
        """The medians of the full rows (the last one is partial; a short recording: its one row)."""
        full = self.stats_rows[:-1] or self.stats_rows

        def med(key):
            v = [r[key] for r in full if r[key] != ""]
            return round(float(np.median(v)), 1) if v else None

        return {"file": "stats.csv", "every_s": STATS_S, "rows": len(self.stats_rows),
                "fps_in": med("fps_in"), "fps_tracked": med("fps_tracked"), "lag_ms": med("lag_ms"),
                "phone_skipped": self.phone_skip}

    # ---- the calibration
    def calibrate(self) -> None:
        with self.lock:
            still = self.cal_ready
            if not still:                    # never still (or too short): the last second's poses
                self.cal_qs = [x.tolist() for _, x in self.win]
        self.h, self.K, self.R_cw, self.C = sensor_camera(self.cal_qs, self.size, self.height_m,
                                                          tilt_offset=self.tilt_offset)
        cp = self.h.camera_params
        self.h.save(self.root / "_working" / "homography_used.json")
        import yaml

        (self.root / "_working" / "config_used.yaml").write_text(
            yaml.safe_dump(json.loads(json.dumps(self.cfg.data, default=str)), sort_keys=False,
                           allow_unicode=True), encoding="utf-8")
        self.log(f"calibrated from {len(self.cal_qs)} poses ({'a still second' if still else 'not still'}"
                 f"): tilt {cp['tilt_deg']:.1f} deg down"
                 + (f" (nudged {self.tilt_offset:+g})" if self.tilt_offset else "")
                 + f", roll {cp['roll_deg']:.1f}, height {cp['height_m']:.2f} m, focal "
                 f"{cp['focal_px']:.0f} px ({cp['hfov_deg']:.0f} deg wide); fixed from now on")
        self.write_scene(growing=True)
        rows, self.pending = self.pending, []
        for fr in sorted({r[3] for r in rows}):
            self.place([r for r in rows if r[3] == fr])

    # ---- tracked rows -> points on the ground
    def place(self, rows: list[tuple]) -> None:
        """One frame's tracked rows (RAW_COLUMNS) as TRACK_DTYPE records, appended to tracks.bin."""
        if not rows:
            return
        if self.h is None:
            self.pending.extend(rows)
            return
        rows = sorted(rows, key=lambda r: r[0])
        foot = np.array([[(r[5] + r[7]) / 2, r[8]] for r in rows], float)
        xy = self.h.to_world(foot) - np.asarray(self.h.origin)
        out = []
        for r, (x, y) in zip(rows, xy):
            if not (np.isfinite(x) and np.isfinite(y)):
                continue                     # past the horizon: the machine draws nothing
            tid, name, frame, t = int(r[0]), str(r[1]), int(r[3]), float(r[4])
            prev = self.last.get(tid)
            if prev is None or t <= prev[0]:     # a track's first point: no speed yet
                speed = prev[3] if prev else None
            else:
                step = math.hypot(x - prev[1], y - prev[2]) / (t - prev[0])
                speed = step if prev[3] is None else SPEED_ALPHA * step + (1 - SPEED_ALPHA) * prev[3]
            self.last[tid] = (t, x, y, speed)
            out.append((tid, frame, t, x, y, 0.0, speed or 0.0, COCO_IDS.get(name, int(r[2])), 0))
            ix = self.index.get(str(tid))
            if ix is None:
                self.index[str(tid)] = {"class": name, "group": self.cfg.class_to_group.get(name, ""),
                                        "first_frame": frame, "last_frame": frame}
            else:
                ix["last_frame"] = frame
        if out:
            b = np.array(out, TRACK_DTYPE).tobytes()
            self.tracks_f.write(b)
            self.tracks_f.flush()
            with self.lock:
                self.records += b
            self.n_points += len(out)

    # ---- the people check
    def people_check(self) -> None:
        """selfcal.people_check on the people seen so far: self.check = {median_m, n, error_pct,
        ok, height_m (the height that would make them 1.7 m tall)}."""
        import pandas as pd

        from .detect import RAW_COLUMNS
        from .selfcal import PERSON_HEIGHT_M, people_check

        try:
            r = people_check(pd.DataFrame(list(self.person_rows), columns=RAW_COLUMNS), self.h)
        except Exception as e:  # noqa: BLE001  (a check only: logged once)
            if not self._check_failed:
                self._check_failed = True
                self.log(f"people check: {type(e).__name__}: {e}")
            return
        if r is None:
            return
        med = float(r["implied_person_height_m"])
        err = med / PERSON_HEIGHT_M - 1
        lo, hi = HEIGHT_RANGE
        was = self.check["ok"] if self.check else None
        self.check = {"median_m": round(med, 2), "n": int(r["n"]), "error_pct": round(100 * err, 1),
                      "ok": abs(err) <= CHECK_TOL,
                      "height_m": round(min(hi, max(lo, self.height_m / (1 + err))), 2)}
        if self.check["ok"] != was:                     # the first verdict, and each change
            self.log(f"people check: people {med:.2f} m tall ({100 * err:+.0f} %, {r['n']} seen)"
                     + ("" if self.check["ok"] else f": the camera nearer {self.check['height_m']:.2f} m?"))

    # ---- the blurred video
    def write_blurred(self, k: int, img: np.ndarray, boxes: np.ndarray) -> None:
        """Frame k blurred (these boxes and those of the last HOLD_S), after the empty slots
        since the last one (the last blurred frame again; black before the first)."""
        self.held.append((k, boxes))
        while self.held and self.held[0][0] < k - HOLD_S * FPS:
            self.held.popleft()
        from .debug_video import blur_boxes

        allb = np.concatenate([b for _, b in self.held]) if self.held else np.zeros((0, 4))
        if len(allb):
            pad = BLUR_PAD * np.column_stack([allb[:, 2] - allb[:, 0], allb[:, 3] - allb[:, 1]])
            blur_boxes(img, np.column_stack([allb[:, :2] - pad, allb[:, 2:] + pad]), fast=True)
        W, Hh = self.vsize
        img = np.ascontiguousarray(img[:Hh, :W])
        fill = self.last_blur if self.last_blur is not None else np.zeros_like(img)
        for _ in range(k - self.k_blur - 1):
            self.blur.write(fill.tobytes())
        self.blur.write(img.tobytes())
        self.k_blur, self.last_blur = k, img
        now = time.monotonic()
        if now - self._preview_t >= PREVIEW_S:                # the Live view's picture (blurred)
            self._preview_t = now
            sc = min(1.0, PREVIEW_W / max(W, Hh))
            small = cv2.resize(img, (max(2, round(W * sc)), max(2, round(Hh * sc))),
                               interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                self.preview = (k, buf.tobytes())

    # ---- the package
    def write_scene(self, growing: bool) -> dict:
        n = self.k_blur + 1
        reach = R_MAX_M                      # fixed from the start: the view draws as it grows
        half = reach + TERRAIN_MARGIN_M
        x0 = float(np.floor((self.C[0] - half) / CELL_M) * CELL_M)
        y1 = float(np.ceil((self.C[1] + half) / CELL_M) * CELL_M)
        nx = ny = round(2 * half / CELL_M) + 1
        np.zeros((ny, nx), "<f4").tofile(self.live / "terrain.f32")
        ext = [x0, x0 + nx * CELL_M, y1 - ny * CELL_M, y1]
        if not (self.live / "aerial.jpg").exists():
            cv2.imwrite(str(self.live / "aerial.jpg"), np.zeros((64, 64), np.uint8))
        cp = self.h.camera_params
        P = self.K @ self.R_cw @ np.column_stack([np.eye(3), -self.C])
        # every class it tracks, from the first write: the view makes its rules at open, a reopened
        # recording from the last write; the same names, the same drawing (exact replay)
        classes = sorted(self.cfg.classes, key=lambda c: COCO_IDS.get(c, 99))
        scene = {
            "version": 1, "clip": "realtime", "run": self.id, "crs": None,
            "origin": {"E": 0.0, "N": 0.0, "datum_z": None},
            "fps": FPS, "frames": n, "duration_s": n / FPS, "image_size": list(self.vsize),
            "camera": {"P": P.tolist(), "K": self.K.tolist(), "R_cw": self.R_cw.tolist(),
                       "C": self.C.tolist(), "yaw_deg": cp["yaw_deg"] % 360,
                       "tilt_deg": cp["tilt_deg"], "roll_deg": cp["roll_deg"],
                       "hfov_deg": cp["hfov_deg"], "height_m": cp["height_m"]},
            "terrain": {"file": "terrain.f32", "nx": nx, "ny": ny, "x0": x0, "y1": y1, "cell": CELL_M},
            "aerial": {"file": "aerial.jpg", "extent": ext, "blank": True},
            "tracks": {"file": "tracks.bin", "n_points": self.n_points, "n_tracks": len(self.index),
                       "layout": TRACK_LAYOUT, "classes": {str(COCO_IDS[c]): c for c in classes
                                                           if c in COCO_IDS}},
            "video": {"file": "video.mp4", "blurred": True, "stabilized": False},
            "R_max": reach, "speed_ramp": SPEED_RAMP,
            "realtime": True, "growing": bool(growing),
            "source_runs": {"homography": "../_working/homography_used.json",
                            "raw_video": "../raw.mkv"},
        }
        tmp = self.live / "scene.json.tmp"
        tmp.write_text(json.dumps(scene, indent=2), encoding="utf-8")
        tmp.replace(self.live / "scene.json")
        return scene

    def write_meta(self, complete: bool, notes: list[str]) -> None:
        cp = self.h.camera_params if self.h is not None else None
        meta = {"format": "motus-realtime-recording", "version": 1, "id": self.id,
                "started": self.started, "ended": self.ended, "size": list(self.size), "fps": FPS,
                "seconds": round((self.k_raw + 1) / FPS, 2) if self.k_raw >= 0 else 0.0,
                "frames_received": self.n_in, "frames_tracked": self.n_tracked,
                "frames_dropped": self.n_dropped, "frames_bad": self.n_bad,
                "raw_dropped": self.n_raw_dropped, "package_frames": self.k_blur + 1,
                "points": self.n_points, "tracks": len(self.index),
                "calibration": cp and {k: round(float(v), 4) if isinstance(v, float) else v
                                       for k, v in cp.items()},
                "height_m": self.height_m, "tilt_offset_deg": self.tilt_offset,
                "people_check": self.check,
                "stats": self.stats_summary(), "complete": complete, "notes": notes}
        (self.root / "recording.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def finish(self) -> dict:
        """After the last frame: the writers closed, the package made whole (or left out when
        nothing was tracked). Returns the recording's meta."""
        notes = []
        if self.h is None and self.n_tracked:          # shorter than the calibration's second
            try:
                self.calibrate()
            except Exception as e:  # noqa: BLE001  (the rest is still saved)
                notes.append(f"calibration: {type(e).__name__}: {e}")
        self.dec_thread.join(30)
        self.stats()
        try:
            self.stats_f.close()
        except OSError:
            pass
        self.raw_q.put(None)
        self.raw_thread.join(120)
        err = self.raw.close()
        if err:
            notes.append(f"raw.mkv: ffmpeg said {err}")
        err = self.blur.close()
        self.tracks_f.close()
        self.poses.close()
        self.ended = time.strftime("%Y-%m-%dT%H:%M:%S")
        complete = False
        if self.n_tracked and self.h is not None and not err:
            bad = _remux(self.live / "blurred.mkv", self.live / "video.mp4")
            if bad:
                notes.append(f"video.mp4: ffmpeg said {bad}")
            else:
                (self.live / "blurred.mkv").unlink(missing_ok=True)
                (self.live / "tracks_index.json").write_text(json.dumps(self.index, indent=1),
                                                             encoding="utf-8")
                self.write_scene(growing=False)
                complete = True
        elif err:
            notes.append(f"blurred video: ffmpeg said {err}")
        else:
            notes.append("nothing was tracked: no package (the raw video is kept)")
        if not complete:
            (self.live / "scene.json").unlink(missing_ok=True)
        self.write_meta(complete, notes)
        secs = (self.k_raw + 1) / FPS if self.k_raw >= 0 else 0.0
        self.log(f"recording {self.id} ended: {secs:.1f} s, {self.n_in} frames received "
                 f"({self.n_dropped} in a taken slot, {self.n_raw_dropped} not written raw), "
                 f"{self.n_tracked} tracked, {self.n_points} points of {len(self.index)} tracks"
                 + ("; package complete" if complete else "; " + "; ".join(notes)))
        try:
            self._logf.close()
        except OSError:
            pass
        return json.loads((self.root / "recording.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- the mode
class LiveTracker:
    """The realtime mode: the model (loaded when the mode starts, released when it stops) and
    at most one recording at a time. model: a stand-in for tests (anything with YOLO's predict).

    One thread runs the model for the whole mode: it loads it, warms it up, then tracks the
    recordings one after another. CUDA sets itself up once per thread (about 3 s on the first
    predict of each), so a thread per recording would start every recording with a stall."""

    def __init__(self, project: Path, cfg: Config | None = None, device: str | None = None,
                 log=print, model=None):
        self.project = Path(project)
        self.cfg = cfg or realtime_config()
        self._log = log
        self.mode = "ready" if model is not None else "off"
        self.error: str | None = None
        self.model = model
        self.device = device
        self.rec: Recording | None = None
        self.saving: list[str] = []
        self.saved: dict | None = None       # the last recording's meta
        self._lock = threading.Lock()
        self._jobs: queue.Queue = queue.Queue()      # ("load", done) / ("track", Recording) / None
        self._worker: threading.Thread | None = None
        self.long_side = LONG_SIDES[0]       # the recorder's frames (the hub tells the phone)

    def set_long_side(self, v) -> None:
        """The next recording's frame size: 1920 (1080p) or 1280 (720p, about half the data)."""
        if v not in LONG_SIDES or isinstance(v, bool):
            raise ValueError(f"frame size: long side {' or '.join(map(str, LONG_SIDES))}")
        self.long_side = int(v)

    def log(self, msg: str) -> None:
        self._log(f"{time.strftime('%H:%M:%S')} realtime: {msg}")

    def _submit(self, job) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._work, name="rt-model", daemon=True)
            self._worker.start()
        self._jobs.put(job)

    def _work(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                break
            if job[0] == "load":
                self._load()
                job[1].set()
            else:
                self._track_loop(job[1])

    # ---- the model
    def load(self, wait: bool = False) -> None:
        """Loads YOLO in the background (status says "loading", then "ready" or "error")."""
        if self.mode in ("loading", "ready"):
            return
        self.mode, self.error = "loading", None
        done = threading.Event()
        self._submit(("load", done))
        if wait:
            done.wait()

    def _load(self) -> None:
        try:
            from ultralytics import YOLO

            from .detect import PRIVACY_IDS, RIDEABLE, _precision_kwargs
            from .device import half_supported, resolve_device

            dev = resolve_device(self.device or self.cfg["device"])
            det = self.cfg["detection"]
            self.device = dev
            self.precision = _precision_kwargs(bool(det.get("half")) and half_supported(dev))
            model = YOLO(self.cfg.model_path())
            ids = sorted(set(self.cfg.class_ids) | RIDEABLE | PRIVACY_IDS)
            for _ in range(2):                                  # as _track calls it; the first is slow
                model.predict(np.zeros((1080, 1920, 3), np.uint8), imgsz=int(det["imgsz"]),
                              conf=min(float(det["conf"]), PRIVACY_CONF), iou=det["iou"],
                              classes=ids, device=dev, verbose=False, **self.precision)
            self.model = model
            self.mode = "ready"
            self.log(f"model {Path(self.cfg.model_path()).name} ready on {dev}")
        except Exception as e:  # noqa: BLE001  (shown in the status)
            self.mode, self.error = "error", f"{type(e).__name__}: {e}"
            self.log(f"the model did not load: {self.error}")

    def close(self) -> None:
        """The mode stops: the recording ends (and is saved), the model released."""
        self.end()
        th = self._worker
        if th is not None:
            self._jobs.put(None)
            th.join(300)
            self._worker = None
        self.model = None
        self.mode = "off"
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ---- the hub's calls
    def begin(self, w: int, h: int, resume: bool = False, height_m: float | None = None,
              tilt_offset: float | None = None) -> dict:
        """A recording starts (resume: the one the recorder dropped out of, when still open), the
        camera height_m above the ground (None: HEIGHT_M) and its tilt nudged by tilt_offset deg."""
        with self._lock:
            rec = self.rec
            if rec is not None and not rec.closing:
                if resume or rec.lost_at is not None:
                    rec.lost_at, rec.reanchor = None, True
                    rec.log("the recorder is back: the recording goes on")
                    return {"id": rec.id, "resumed": True}
                raise RuntimeError("a recording is running")
            if self.mode == "error":
                raise RuntimeError(f"the model did not load ({self.error})")
            if not (MIN_SIDE <= w <= MAX_SIDE and MIN_SIDE <= h <= MAX_SIDE):
                raise ValueError("frame size out of range")
            hm = HEIGHT_M if height_m is None else float(height_m)
            if not HEIGHT_RANGE[0] <= hm <= HEIGHT_RANGE[1]:
                raise ValueError(f"camera height: {HEIGHT_RANGE[0]:g} to {HEIGHT_RANGE[1]:g} m")
            to = 0.0 if tilt_offset is None else float(tilt_offset)
            if not abs(to) <= TILT_NUDGE_MAX:
                raise ValueError(f"tilt nudge: at most {TILT_NUDGE_MAX:g} deg")
            base = recordings_dir(self.project)
            base.mkdir(parents=True, exist_ok=True)
            rid = time.strftime("%Y%m%d-%H%M%S")
            n = 2
            while (base / rid).exists():
                rid, n = f"{time.strftime('%Y%m%d-%H%M%S')}-{n}", n + 1
            self.rec = Recording(base / rid, (w, h), self.cfg, self._log, hm, to)
            self.saved = None
            self._submit(("track", self.rec))        # after the loading (frames wait: latest wins)
            return {"id": rid, "resumed": False}

    def frame(self, head: dict, jpeg: bytes) -> bool:
        rec = self.rec
        return rec.receive(head, jpeg) if rec is not None else False

    def end(self) -> None:
        """Stop: the frames still waiting are tracked, then the package is made whole."""
        with self._lock:
            rec = self.rec
            if rec is None or rec.closing:
                return
            rec.closing = True
            self.saving.append(rec.id)
            rec.wake.set()

    def lost(self) -> None:
        """The recorder dropped out: the recording waits LOST_S for it, then ends."""
        rec = self.rec
        if rec is None or rec.closing:
            return
        rec.lost_at = time.monotonic()
        rec.log(f"the recorder dropped out: waiting {LOST_S:g} s for it")

        def check():
            if rec.lost_at is not None and self.rec is rec and not rec.closing:
                rec.log("the recorder did not come back")
                self.end()
        threading.Timer(LOST_S, check).start()

    def status(self) -> dict:
        rec = self.rec
        st = {"mode": self.mode, "error": self.error, "saving": list(self.saving),
              "long_side": self.long_side,
              "saved": self.saved and {**{k: self.saved.get(k) for k in
                                          ("id", "complete", "seconds", "tracks", "notes")},
                                       "run": str(recordings_dir(self.project) / self.saved["id"])}}
        if rec is None or rec.closing:
            return {**st, "recording": False}
        now = time.monotonic()

        def rate(ts):
            ts = [t for t in ts if now - t < 2.0]
            return round((len(ts) - 1) / (ts[-1] - ts[0]), 1) if len(ts) > 2 and ts[-1] > ts[0] else 0.0

        cp = rec.h.camera_params if rec.h is not None else None
        return {**st, "recording": True, "id": rec.id, "run": str(rec.root),
                "height_m": rec.height_m, "tilt_offset_deg": rec.tilt_offset,
                "still_s": round(min(rec.still_s, CAL_S), 2), "check": rec.check,
                "seconds": round(max(rec.k_raw + 1, 0) / FPS, 1), "frames": rec.n_in,
                "tracked": rec.n_tracked, "fps_in": rate(list(rec.rx_times)),
                "fps_tracked": rate(list(rec.tr_times)),
                "lag_ms": round(float(np.median(rec.lag_ms))) if rec.lag_ms else None,
                "people": rec.people, "tracks": len(rec.index), "lost": rec.lost_at is not None,
                "calibrated": cp is not None,
                "camera": cp and {"tilt_deg": round(cp["tilt_deg"], 1),
                                  "roll_deg": round(cp["roll_deg"], 1),
                                  "height_m": cp["height_m"], "focal_px": round(cp["focal_px"])}}

    # ---- the Live view (step 3)
    def tracks_since(self, rid: str, n0: int = 0) -> tuple[dict, bytes]:
        """Recording rid's points from point n0 on (TRACK_DTYPE records, whole frames only) and
        {"id", "n0", "n" (points so far), "frames" (frames whole so far: the view's clock may
        go up to there), "done" (the package is whole: these are all)}. While it records (or
        saves): from memory; after: from its package (FileNotFoundError: none was made)."""
        if not RECORDING_ID.fullmatch(str(rid)):
            raise ValueError("not a recording")
        size, rec = TRACK_DTYPE.itemsize, self.rec
        if rec is not None and rec.id == rid:
            with rec.lock:
                n = len(rec.records) // size
                n0 = max(0, min(int(n0), n))
                data, frames = bytes(rec.records[n0 * size:]), rec.upto
            return {"id": rid, "n0": n0, "n": n, "frames": frames, "done": False}, data
        return package_tracks(self.project, rid, n0)

    def preview(self, rid: str) -> bytes | None:
        """The newest blurred picture of recording rid while it records (JPEG, PREVIEW_W)."""
        rec = self.rec
        if rec is None or rec.id != rid or rec.preview is None:
            return None
        return rec.preview[1]

    # ---- tracking (the model's thread, one recording after another)
    def _track_loop(self, rec: Recording) -> None:
        from .detect import PRIVACY_IDS, RIDEABLE, FrameTracker, make_tracker

        det = self.cfg["detection"]
        ft = None
        try:
            while True:
                rec.stats(time.monotonic())
                rec.wake_tr.wait(0.25)
                rec.wake_tr.clear()
                item = rec.take()
                if item == "done":
                    break
                if item is None:
                    continue
                if self.model is None:                   # the model did not load: nothing tracked
                    if not rec.closing:
                        with rec.lock:
                            if rec.ready is None:
                                rec.ready = item
                        time.sleep(0.05)
                    continue
                if ft is None:
                    tracker, _ = make_tracker(det["tracker"], FPS, det.get("track_buffer_s"),
                                              self.device, rec.log)
                    wanted = set(self.cfg.class_ids)
                    ft = FrameTracker(det, tracker, wanted, None, record_predicted=False)
                    ids = sorted(wanted | (RIDEABLE if det.get("suppress_riders") else set())
                                 | PRIVACY_IDS)
                self._track(rec, ft, item, det, ids, PRIVACY_IDS)
        except Exception as e:  # noqa: BLE001  (the recording is still saved)
            rec.log(f"tracking stopped: {type(e).__name__}: {e}")
            rec.closing = True
        finally:
            meta = rec.finish()
            with self._lock:
                if self.rec is rec:
                    self.rec = None
                if rec.id in self.saving:
                    self.saving.remove(rec.id)
                self.saved = meta

    def _track(self, rec: Recording, ft, item, det, ids, privacy_ids) -> None:
        k, t, rx, img = item
        if rec.h is None and rec.cal_due(t):
            rec.calibrate()
        conf_min = min(float(det["conf"]), PRIVACY_CONF)
        r = self.model.predict(img, imgsz=int(det["imgsz"]), conf=conf_min, iou=det["iou"],
                               classes=ids, device=self.device, verbose=False,
                               **getattr(self, "precision", {}))[0]
        b = r.boxes.cpu().numpy()
        xyxy, conf, cls = b.xyxy, b.conf, b.cls.astype(int)
        sure = conf >= float(det["conf"])
        rows = ft.step(k, k / FPS, img, xyxy[sure], conf[sure], cls[sure])
        rec.place(rows)
        rec.write_blurred(k, img, xyxy[np.isin(cls, list(privacy_ids))])
        rec.n_tracked += 1
        rec.people = sum(1 for r_ in rows if r_[1] == "person")
        rec.person_rows.extend(r_ for r_ in rows if r_[1] == "person")
        now = time.monotonic()
        if rec.h is not None:
            with rec.lock:
                rec.upto = k + 1                     # the Live view may draw up to here
            if now - rec._check_t >= CHECK_EVERY_S and len(rec.person_rows) >= 30:
                rec._check_t = now
                rec.people_check()
        rec.tr_times.append(now)
        rec.lag_ms.append((now - rx) * 1000)
        rec._win_lags.append((now - rx) * 1000)


def package_tracks(project: Path, rid: str, n0: int = 0) -> tuple[dict, bytes]:
    """A finished recording's points from n0 on, from its package (as LiveTracker.tracks_since,
    "done": true). FileNotFoundError: no whole package."""
    if not RECORDING_ID.fullmatch(str(rid)):
        raise ValueError("not a recording")
    size, live = TRACK_DTYPE.itemsize, recordings_dir(project) / rid / "live"
    scene = json.loads((live / "scene.json").read_text(encoding="utf-8"))
    if scene.get("growing"):                         # left without being saved (a crash)
        raise FileNotFoundError(f"recording {rid}: no whole package")
    buf = (live / "tracks.bin").read_bytes()
    n = len(buf) // size
    n0 = max(0, min(int(n0), n))
    return ({"id": rid, "n0": n0, "n": n, "frames": int(scene["frames"]), "done": True},
            buf[n0 * size:n * size])


def recording_title(rid: str) -> str:
    """"Rec 30 Oct 11:01" for the recording 20261030-110119."""
    try:
        return "Rec " + time.strftime("%d %b %H:%M", time.strptime(rid[:15], "%Y%m%d-%H%M%S"))
    except ValueError:
        return f"Rec {rid}"


def list_recordings(project: Path) -> list[dict]:
    """The recordings under runs/live/, newest first: {id, run, title, seconds, tracks,
    fps_tracked (the median), complete, growing, notes}."""
    out = []
    base = recordings_dir(project)
    for d in sorted(base.iterdir(), reverse=True) if base.is_dir() else []:
        if not d.is_dir():
            continue
        try:
            m = json.loads((d / "recording.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            m = {}
        scene = d / "live" / "scene.json"
        try:
            growing = bool(json.loads(scene.read_text(encoding="utf-8")).get("growing"))
        except (OSError, ValueError):
            growing = False
        out.append({"id": d.name, "run": str(d), "title": recording_title(d.name),
                    "seconds": m.get("seconds"), "tracks": m.get("tracks"),
                    "fps_tracked": (m.get("stats") or {}).get("fps_tracked"),
                    "complete": bool(m.get("complete")) and scene.is_file() and not growing,
                    "growing": growing or not m, "notes": m.get("notes") or []})
    return out
