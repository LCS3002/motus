"""Synthetic demo site with known ground truth (used by the end-to-end test and `groundtrack demo`).

A virtual camera 10 m up looks at a paved plaza. Two people (sprites cut from Ultralytics'
sample photo) move at known speeds; one passes behind a pillar so the tracker has to predict
through an occlusion. A matching GeoTIFF (EPSG:27700) and calibration points are written too.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ORIGIN = np.array([530000.0, 180000.0])


class SyntheticCamera:
    """Pinhole camera. World: E, N on the ground (z = 0), z up. Yaw clockwise from north."""

    def __init__(self, width=1920, height=1080, f=1400.0, height_m=12.0, pitch_deg=30.0,
                 cam_local=(0.0, -20.0), yaw_deg=0.0, roll_deg=0.0, origin=ORIGIN):
        self.width, self.height = width, height
        self.yaw, self.pitch = yaw_deg, pitch_deg
        self.K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])
        p, yw, rl = np.radians(pitch_deg), np.radians(yaw_deg), np.radians(roll_deg)
        fwd = np.array([np.sin(yw) * np.cos(p), np.cos(yw) * np.cos(p), -np.sin(p)])
        right = np.array([np.cos(yw), -np.sin(yw), 0.0])
        down = np.cross(fwd, right)
        right, down = (np.cos(rl) * right + np.sin(rl) * down,
                       -np.sin(rl) * right + np.cos(rl) * down)
        self.R = np.vstack([right, down, fwd])  # world -> camera
        self.C = np.array([cam_local[0], cam_local[1], height_m]) + np.r_[origin, 0]

    def project(self, world_xyz) -> np.ndarray:
        X = np.asarray(world_xyz, float).reshape(-1, 3) - self.C
        uv = (X @ self.R.T) @ self.K.T
        return uv[:, :2] / uv[:, 2:3]

    def ground_to_pixel(self, en) -> np.ndarray:
        en = np.asarray(en, float).reshape(-1, 2)
        return self.project(np.column_stack([en, np.zeros(len(en))]))

    def ground_homography(self) -> np.ndarray:
        """3x3 world (E, N, 1) -> pixel."""
        T = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1.0]])  # (E,N,1) -> (E,N,0,1)
        P = self.K @ np.column_stack([self.R, -self.R @ self.C])
        return P @ T


def _paving(res: float, x0: float, y1: float, w: int, h: int) -> np.ndarray:
    rng = np.random.default_rng(7)
    img = np.full((h, w, 3), (178, 172, 160), np.uint8)
    img = (img.astype(int) + rng.integers(-12, 12, (h, w, 1))).clip(0, 255).astype(np.uint8)
    step = int(round(2.0 / res))
    img[::step, :] = (120, 116, 108)
    img[:, ::step] = (120, 116, 108)
    # a few distinctive features: benches, planters, a drain grid, a painted line
    for (cx, cy, sx, sy, col) in ((-12, 14, 3, 0.8, (90, 60, 40)), (10, 18, 0.8, 3, (90, 60, 40)),
                                  (-6, 30, 2.5, 2.5, (60, 110, 60)), (14, 34, 2.5, 2.5, (60, 110, 60)),
                                  (0, 0, 0.6, 0.6, (50, 50, 50)), (-10, 2, 0.6, 0.6, (50, 50, 50))):
        c0 = int((cx - sx / 2 - x0) / res)
        r0 = int((y1 - (cy + sy / 2)) / res)
        img[r0:r0 + int(sy / res), c0:c0 + int(sx / res)] = col
    r = int((y1 - 12) / res)
    img[r:r + int(0.15 / res), :] = (235, 235, 225)
    return img


def _person_sprite() -> np.ndarray:
    from ultralytics.utils import ASSETS

    img = cv2.imread(str(ASSETS / "bus.jpg"))
    # the woman in a dark coat on the left of Ultralytics' bus.jpg (fixed crop, no model needed)
    return img[398:903, 49:241].copy()


def _paste(frame, sprite, foot_uv, height_px):
    h = max(int(round(height_px)), 8)
    w = max(int(round(sprite.shape[1] * h / sprite.shape[0])), 4)
    s = cv2.resize(sprite, (w, h), interpolation=cv2.INTER_AREA)
    x0, y0 = int(round(foot_uv[0] - w / 2)), int(round(foot_uv[1] - h))
    H, W = frame.shape[:2]
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if xb > xa and yb > ya:
        frame[ya:yb, xa:xb] = s[ya - y0:yb - y0, xa - x0:xb - x0]


def make_demo_site(out_dir: str | Path, fps: float = 25.0, size=(1280, 720),
                   shake_deg: float = 0.0) -> dict:
    """Write video, GeoTIFF, calibration points and a site config. Returns paths + truth.

    shake_deg > 0 makes the camera sway like a handheld phone (smooth yaw/pitch/roll drift of
    up to about that many degrees; frame 0, the calibration frame, is unshaken).
    """
    from .calibrate import save_points_csv
    from .geo import write_geotiff

    out = Path(out_dir)
    for d in ("footage", "maps", "calibration", "sites"):
        (out / d).mkdir(parents=True, exist_ok=True)
    W, H = size
    cam_kw = dict(width=W, height=H, f=1100.0, height_m=10.0, cam_local=(0.0, -20.0))
    cam = SyntheticCamera(pitch_deg=20.0, yaw_deg=8.0, **cam_kw)
    sway_rng = np.random.default_rng(11)
    sway_f = sway_rng.uniform(0.05, 0.3, (3, 3))      # Hz
    sway_p = sway_rng.uniform(0, 2 * np.pi, (3, 3))

    def camera_at(t: float) -> SyntheticCamera:
        if shake_deg <= 0:
            return cam
        d = [shake_deg / 3 * sum(np.sin(2 * np.pi * sway_f[i, j] * t + sway_p[i, j])
                                 - np.sin(sway_p[i, j]) for j in range(3)) for i in range(3)]
        return SyntheticCamera(pitch_deg=20.0 + d[0], yaw_deg=8.0 + d[1], roll_deg=d[2], **cam_kw)

    # map: 60 m x 70 m at 5 cm/px, EPSG:27700
    res, x0, x1, y0, y1 = 0.05, -30.0, 30.0, -10.0, 60.0
    mw, mh = int((x1 - x0) / res), int((y1 - y0) / res)
    paving = _paving(res, x0, y1, mw, mh)
    write_geotiff(out / "maps/demo.tif", paving[:, :, ::-1].copy(), ORIGIN[0] + x0,
                  ORIGIN[1] + y1, res)

    # map pixel -> world -> video pixel
    A = np.array([[res, 0, ORIGIN[0] + x0], [0, -res, ORIGIN[1] + y1], [0, 0, 1.0]])
    M = cam.ground_homography() @ A
    background = cv2.warpPerspective(paving, M, (W, H), flags=cv2.INTER_AREA,
                                     borderValue=(200, 200, 200))

    # ground-truth movers (local metres); person A stands, walks east, stands (ground-truth walk)
    a_start, a_end, a_speed, stand = np.array([-4.5, 5.0]), np.array([4.5, 5.0]), 1.5, 1.0
    walk_t = np.linalg.norm(a_end - a_start) / a_speed
    b_start, b_vel = np.array([-7.0, 14.0]), np.array([0.0, -1.2])
    duration = walk_t + 2 * stand
    n = int(round(duration * fps))
    pillar = np.array([[0.2, 4.0], [1.8, 4.0]])  # pillar spans 1.6 m of person A's path

    def pos_a(t):
        if t < stand:
            return a_start
        if t > stand + walk_t:
            return a_end
        return a_start + (a_end - a_start) * (t - stand) / walk_t

    sprite = _person_sprite()
    writer = cv2.VideoWriter(str(out / "footage/demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (W, H))
    for k in range(n):
        t = k / fps
        c = camera_at(t)
        if c is cam:
            frame = background.copy()
        else:
            frame = cv2.warpPerspective(paving, c.ground_homography() @ A, (W, H),
                                        flags=cv2.INTER_AREA, borderValue=(200, 200, 200))
        movers = [pos_a(t), b_start + b_vel * t]
        # far to near (painter's algorithm)
        for p in sorted(movers, key=lambda p: -p[1]):
            en = ORIGIN + p
            foot = c.ground_to_pixel(en)[0]
            head = c.project(np.r_[en, 1.75])[0]
            _paste(frame, sprite, foot, foot[1] - head[1])
        # pillar: 1.6 m wide, 3 m tall, standing just in front of person A's path
        pb = c.ground_to_pixel(ORIGIN + pillar)
        top = c.project(np.r_[ORIGIN + pillar[0], 3.0])[0]
        cv2.rectangle(frame, (int(pb[0, 0]), int(top[1])), (int(pb[1, 0]), int(pb[0, 1])),
                      (70, 72, 78), -1)
        writer.write(frame)
    writer.release()

    # calibration point pairs: distinctive map features with 0.7 px click noise
    rng = np.random.default_rng(3)
    feats = ORIGIN + np.array([[-13.5, 13.6], [-10.5, 14.4], [10.4, 19.5], [-7.25, 31.25],
                               [15.25, 35.25], [0.3, 0.3], [-9.7, 2.3], [12.75, 32.75]])
    px = cam.ground_to_pixel(feats) + rng.normal(0, 0.7, (len(feats), 2))
    save_points_csv(out / "calibration/demo_homography_points.csv", px, feats)

    cfg_text = f"""\
site: demo
video: ../footage/demo.mp4
geotiff: ../maps/demo.tif
homography: ../calibration/demo_homography.json
output_dir: ../runs
models_dir: ../models
device: auto
detection:
  model: yolo26n.pt
  imgsz: 1280
  conf: 0.2
  tracker: bytetrack
  track_buffer_s: 1.5
groups:
  people:
    classes: [person]
cleaning:
  min_track_s: 0.0
grid:
  cell_size_m: 1.0
stats:
  count_lines:
    - {{name: crossing, a: [{ORIGIN[0] - 10}, {ORIGIN[1] + 9}], b: [{ORIGIN[0] - 4}, {ORIGIN[1] + 9}]}}
visuals:
  margin_m: 4.0
"""
    (out / "sites/demo.yaml").write_text(cfg_text, encoding="utf-8")
    return {
        "config": out / "sites/demo.yaml", "video": out / "footage/demo.mp4",
        "geotiff": out / "maps/demo.tif", "points_csv": out / "calibration/demo_homography_points.csv",
        "camera": cam, "fps": fps, "n_frames": n,
        "truth": {
            "a_start": (ORIGIN + a_start).tolist(), "a_end": (ORIGIN + a_end).tolist(),
            "a_speed": a_speed, "a_walk_s": walk_t, "b_speed": 1.2, "b_x": float(ORIGIN[0] - 7),
            "pillar_x": (float(ORIGIN[0] + pillar[0, 0]), float(ORIGIN[0] + pillar[1, 0])),
        },
    }
