"""motus Live: a point cloud of the surroundings (plans/live_controller/11, option A).

A separate function, OFF by default: nothing else in groundtrack imports this module (the CLI and
the app only call it when a cloud is asked for), and the depth model (transformers, the optional
extra `cloud`) is loaded only when a cloud is made. With the switch off, motus works as without it.

How: Depth Anything V2 (relative, Small) gives each frame an inverse depth up to an affine map;
that map is fitted per frame to the calibrated ground (the package's terrain: its true depth at
every ground pixel), so the depth is in metres where the camera's calibration says the ground is.
The calibrated camera then places every pixel. A still camera (a pre-tracked site) takes the median
over its frames, pixel by pixel: people who move drop out; tracked people are also masked frame by
frame, so nobody enters the cloud. Sky and the far field beyond the package's reach are dropped.
Points are merged on a voxel grid (mean position and colour, the count).

Realtime (step 2, RealtimeCloud / CloudBuilder): while a recording runs with the cloud on, its own
thread makes a depth frame now and then (the phone turned to new ground), placed from the recorder's
fixed position turned as the phone was, fitted to the flat ground: a rough cloud, shown in Live as it
grows and written into the package at Stop. The accurate one (the recorder's walk followed, for the
cloud only) is step 3.

Files, beside the package (<run>/live/):
    cloud.bin   packed little-endian records CLOUD_DTYPE: x, y, z f32 (the package's frame, as the
                camera's C and the tracks), r, g, b u8 (the blurred video's colour), n u8 (frames in
                the voxel, at most 255)
    cloud.json  {"format": "motus-cloud", "version": 1, "n", "voxel_m", "model", "frames", "made",
                 "source", "fit": {"inliers": share of fitted ground pixels, per frame}}
    cloud.ply   the same points for Rhino and Houdini (binary, x y z + colour)

Measured 1 Nov 2026 on Trafalgar 6376 against the LiDAR terrain: within ~5 % to 20 m, ~7 % to 35 m,
~20 % at 35-50 m; 20 ms a frame (fp16) on the RTX 5060.
"""
from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
import warnings
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
FORMAT, VERSION = "motus-cloud", 1
CLOUD_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1"),
                        ("n", "u1")])
FILES = ("cloud.bin", "cloud.json", "cloud.ply")
STEP = 2                     # every STEP-th pixel (1920 x 1080 -> 960 x 540)
FRAMES = 24                  # a still camera's frames, spread over its clip
VOXEL_M = 0.10               # the merged cloud's voxel
FIT_NEAR, FIT_FAR = 3.0, 25.0     # the ground fitted between these distances (m)
FIT_TOL = 0.08               # a ground pixel fits within this share of its inverse depth
MARGIN_M = 20.0              # points out to the package's reach (R_max) + this
EDGE = 0.06                  # flying pixels: where log depth jumps by more (between neighbours)
TRACK_BOX = (0.4, 1.9)       # a tracked person's half width, height (m): masked in each frame


# --------------------------------------------------------------------------- the depth model
class DepthModel:
    """Depth Anything V2 (relative): rgb (h x w x 3, uint8) -> its inverse depth at out_hw (bigger:
    nearer; an affine map from the truth). Loaded once, on the GPU when there is one (fp16)."""

    def __init__(self, name: str = MODEL, device: str | None = None):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.proc = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModelForDepthEstimation.from_pretrained(name, dtype=self.dtype).to(self.device).eval()
        self.name = name

    def __call__(self, rgb: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
        torch = self.torch
        with torch.inference_mode():
            x = self.proc(images=rgb, return_tensors="pt")["pixel_values"].to(self.device, self.dtype)
            d = self.model(pixel_values=x).predicted_depth
            d = torch.nn.functional.interpolate(d[:, None].float(), size=tuple(out_hw), mode="bilinear",
                                                align_corners=False)[0, 0]
        return d.cpu().numpy().astype(np.float32)


# --------------------------------------------------------------------------- the geometry
def grid_rays(K: np.ndarray, size: tuple[int, int], step: int = STEP) -> np.ndarray:
    """The camera rays (z = 1, OpenCV axes) of every step-th pixel's centre: h x w x 3."""
    W, H = int(size[0]), int(size[1])
    u = np.arange(W // step) * step + step / 2
    v = np.arange(H // step) * step + step / 2
    U, V = np.meshgrid(u, v)
    return np.stack([(U - K[0, 2]) / K[0, 0], (V - K[1, 2]) / K[1, 1], np.ones_like(U)], -1)


class Terrain:
    """The package's ground: heights (m, the package frame) on 1 m cells, row 0 north; bilinear
    between cell centres; the datum (0) past its edge."""

    def __init__(self, z: np.ndarray, x0: float, y1: float, cell: float = 1.0):
        self.z, self.x0, self.y1, self.cell = np.asarray(z, np.float64), float(x0), float(y1), float(cell)

    @classmethod
    def of_package(cls, live: Path, meta: dict) -> "Terrain":
        t = meta["terrain"]
        z = np.fromfile(Path(live) / t["file"], "<f4").reshape(int(t["ny"]), int(t["nx"]))
        return cls(z, t["x0"], t["y1"], t.get("cell", 1.0))

    def __call__(self, x, y):
        ny, nx = self.z.shape
        fx = (np.asarray(x) - self.x0) / self.cell - 0.5
        fy = (self.y1 - np.asarray(y)) / self.cell - 0.5
        j = np.clip(np.floor(fx).astype(int), 0, nx - 2)
        i = np.clip(np.floor(fy).astype(int), 0, ny - 2)
        a, b = np.clip(fx - j, 0, 1), np.clip(fy - i, 0, 1)
        h = (self.z[i, j] * (1 - a) * (1 - b) + self.z[i, j + 1] * a * (1 - b)
             + self.z[i + 1, j] * (1 - a) * b + self.z[i + 1, j + 1] * a * b)
        out = (fx < -0.5) | (fy < -0.5) | (fx > nx - 0.5) | (fy > ny - 0.5)
        return np.where(out, 0.0, h)


def ground_depth(dirs: np.ndarray, C: np.ndarray, ground) -> np.ndarray:
    """Each ray's depth (along the camera's z: dirs are world rays of camera z = 1) to the ground
    (ground(x, y) -> height); NaN for rays that never come down."""
    dz = dirs[..., 2]
    down = dz < -1e-4
    safe = np.where(down, dz, -1.0)
    t = np.where(down, (0.0 - C[2]) / safe, np.nan)
    for _ in range(6):                               # the terrain's height where the ray lands, again
        X = C + np.nan_to_num(t)[..., None] * dirs
        t = np.where(down, (ground(X[..., 0], X[..., 1]) - C[2]) / safe, np.nan)
    t[~(t > 0)] = np.nan
    return t


def fit_ground(x: np.ndarray, y: np.ndarray, tol: float = FIT_TOL, iters: int = 300,
               seed: int = 0) -> tuple[float, float, float]:
    """y = a x + b, robustly (RANSAC, then least squares on the inliers): x the model's inverse
    depth, y the ground's true inverse depth, at the same pixels. (a, b, share of inliers);
    (nan, nan, 0) when it cannot be fitted."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    ok = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x, y = x[ok], y[ok]
    if len(x) < 50:
        return math.nan, math.nan, 0.0
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        i, j = rng.integers(0, len(x), 2)
        if abs(x[i] - x[j]) < 1e-9:
            continue
        a = (y[i] - y[j]) / (x[i] - x[j])
        if a <= 0:                                   # nearer must stay nearer
            continue
        b = y[i] - a * x[i]
        inl = np.abs((a * x + b) / y - 1) < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < 20:
        return math.nan, math.nan, 0.0
    A = np.stack([x[best], np.ones(int(best.sum()))], 1)
    a, b = np.linalg.lstsq(A, y[best], rcond=None)[0]
    if not a > 0:
        return math.nan, math.nan, 0.0
    return float(a), float(b), float(best.mean())


def depth_edges(depth: np.ndarray, jump: float = EDGE) -> np.ndarray:
    """Pixels at a depth edge (flying pixels between a near and a far surface)."""
    ld = np.log(np.maximum(np.nan_to_num(depth, nan=1e6), 1e-3))
    g = np.zeros(ld.shape, bool)
    g[1:] |= np.abs(np.diff(ld, axis=0)) > jump
    g[:-1] |= np.abs(np.diff(ld, axis=0)) > jump
    g[:, 1:] |= np.abs(np.diff(ld, axis=1)) > jump
    g[:, :-1] |= np.abs(np.diff(ld, axis=1)) > jump
    return g


def track_mask(tracks: np.ndarray, frame: int, K: np.ndarray, R_cw: np.ndarray, C: np.ndarray,
               shape: tuple[int, int], step: int = STEP, near: int = 1) -> np.ndarray:
    """True where a tracked person or vehicle of frames frame +- near stands (a box TRACK_BOX m,
    upright on its ground point), on the step grid."""
    m = np.zeros(shape, bool)
    if tracks is None or not len(tracks):
        return m
    sel = tracks[np.abs(tracks["frame"].astype(np.int64) - int(frame)) <= near]
    hw, ht = TRACK_BOX
    for tr in sel:
        p = np.array([[tr["x"] + dx, tr["y"] + dy, tr["z"] + dz] for dx in (-hw, hw) for dy in (-hw, hw)
                      for dz in (0.0, ht)])
        q = (K @ (R_cw @ (p - C).T)).T
        if (q[:, 2] <= 0.3).any():
            continue
        u, v = q[:, 0] / q[:, 2] / step, q[:, 1] / q[:, 2] / step
        a, b = int(max(u.min() - 1, 0)), int(min(u.max() + 2, shape[1]))
        c, d = int(max(v.min() - 1, 0)), int(min(v.max() + 2, shape[0]))
        if a < b and c < d:
            m[c:d, a:b] = True
    return m


class VoxelCloud:
    """Points merged on a voxel grid: each voxel's mean position and colour, and its count."""

    def __init__(self, size: float = VOXEL_M):
        self.size = float(size)
        self.keys = np.zeros(0, np.int64)
        self.sum = np.zeros((0, 3))
        self.col = np.zeros((0, 3))
        self.n = np.zeros(0, np.int64)

    def add(self, P: np.ndarray, rgb: np.ndarray) -> None:
        P = np.asarray(P, np.float64).reshape(-1, 3)
        if not len(P):
            return
        rgb = np.asarray(rgb, np.float64).reshape(-1, 3)
        g = np.floor(P / self.size).astype(np.int64) + (1 << 20)        # +-1e6 voxels each way
        k = (g[:, 0] << 42) | (g[:, 1] << 21) | g[:, 2]
        uk, inv = np.unique(k, return_inverse=True)                     # this batch's voxels
        inv, m = inv.ravel(), len(uk)
        n = np.bincount(inv, minlength=m)
        s = np.stack([np.bincount(inv, P[:, i], m) for i in range(3)], 1)
        c = np.stack([np.bincount(inv, rgb[:, i], m) for i in range(3)], 1)
        pos = np.searchsorted(self.keys, uk)                            # merged into the kept ones
        hit = pos < len(self.keys)
        hit[hit] = self.keys[pos[hit]] == uk[hit]
        at = pos[hit]
        self.sum[at] += s[hit]
        self.col[at] += c[hit]
        self.n[at] += n[hit]
        new = ~hit
        if new.any():
            keys = np.concatenate([self.keys, uk[new]])
            order = np.argsort(keys, kind="stable")
            self.keys = keys[order]
            self.sum = np.concatenate([self.sum, s[new]])[order]
            self.col = np.concatenate([self.col, c[new]])[order]
            self.n = np.concatenate([self.n, n[new]])[order]

    def __len__(self) -> int:
        return len(self.keys)

    def points(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(positions n x 3, colours n x 3 uint8, counts n)."""
        n = np.maximum(self.n, 1)[:, None]
        return self.sum / n, np.clip(np.round(self.col / n), 0, 255).astype(np.uint8), self.n.copy()


# --------------------------------------------------------------------------- the files
def _replace(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def records(P: np.ndarray, rgb: np.ndarray, n: np.ndarray) -> np.ndarray:
    """The points as CLOUD_DTYPE records (cloud.bin's)."""
    rec = np.zeros(len(P), CLOUD_DTYPE)
    if len(P):
        rec["x"], rec["y"], rec["z"] = np.asarray(P, np.float64).T
        rec["r"], rec["g"], rec["b"] = np.asarray(rgb).T
        rec["n"] = np.clip(n, 0, 255)
    return rec


def write_cloud(live: Path, P: np.ndarray, rgb: np.ndarray, n: np.ndarray, info: dict) -> dict:
    """cloud.bin, cloud.ply, then cloud.json (last: a cloud is there once its header is)."""
    live = Path(live)
    rec = records(P, rgb, n)
    _replace(live / "cloud.bin", rec.tobytes())
    ply = np.zeros(len(P), [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    ply["x"], ply["y"], ply["z"] = rec["x"], rec["y"], rec["z"]
    ply["red"], ply["green"], ply["blue"] = rec["r"], rec["g"], rec["b"]
    head = ("ply\nformat binary_little_endian 1.0\ncomment motus point cloud (the package's frame: metres "
            "about its origin)\n" f"element vertex {len(P)}\nproperty float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    _replace(live / "cloud.ply", head.encode("ascii") + ply.tobytes())
    meta = {"format": FORMAT, "version": VERSION, "n": int(len(P)), "file": "cloud.bin", "record_bytes": CLOUD_DTYPE.itemsize,
            **info}
    _replace(live / "cloud.json", json.dumps(meta, indent=1).encode("utf-8"))
    return meta


def read_cloud(live: Path) -> tuple[dict, np.ndarray]:
    """(cloud.json, cloud.bin's records)."""
    live = Path(live)
    meta = json.loads((live / "cloud.json").read_text(encoding="utf-8"))
    if meta.get("format") != FORMAT:
        raise ValueError(f"{live / 'cloud.json'}: not a motus cloud")
    return meta, np.fromfile(live / "cloud.bin", CLOUD_DTYPE)


def remove_cloud(live: Path) -> None:
    for f in FILES:
        (Path(live) / f).unlink(missing_ok=True)


# --------------------------------------------------------------------------- a still camera (a site)
TRACK_DTYPE = np.dtype([("id", "<u4"), ("frame", "<u4"), ("t", "<f4"), ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                        ("speed", "<f4"), ("cls", "<u2")])


def _frames(video: Path, picks: list[int]):
    import cv2

    cap = cv2.VideoCapture(str(video))
    try:
        for f in picks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, bgr = cap.read()
            if ok:
                yield int(f), cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def make_site_cloud(run: Path, frames: int = FRAMES, voxel: float = VOXEL_M, depth=None, log=print) -> dict:
    """The cloud of a pre-tracked site camera's package (<run>/live/): still and calibrated, its video
    steadied to the calibrated view and blurred. depth: a DepthModel (or any rgb, (h, w) -> inverse
    depth); made here when None. Writes the files; returns cloud.json."""
    import cv2

    live = Path(run) / "live"
    meta = json.loads((live / "scene.json").read_text(encoding="utf-8"))
    if meta.get("realtime"):
        raise ValueError("a Realtime recording: its cloud is made while it records (plans/live_controller/11)")
    vid = meta.get("video") or {}
    if not vid.get("blurred"):
        raise ValueError(f"{live}: the package's video is not blurred: no cloud from it")
    cam = meta["camera"]
    K, R_cw, C = np.array(cam["K"], float), np.array(cam["R_cw"], float), np.array(cam["C"], float)
    R_wc = R_cw.T
    W, H = meta["image_size"]
    rays = grid_rays(K, (W, H))
    shape = rays.shape[:2]
    dirs = rays @ R_wc.T
    terrain = Terrain.of_package(live, meta)
    tg = ground_depth(dirs, C, terrain)                       # the ground's true depth per pixel
    dist = tg * np.linalg.norm(rays, axis=-1)
    reach = float(meta.get("R_max") or 60.0) + MARGIN_M
    fit_px = np.isfinite(tg) & (dist > FIT_NEAR) & (dist < min(FIT_FAR, reach))
    tracks = np.fromfile(live / meta["tracks"]["file"], TRACK_DTYPE) if (live / meta["tracks"]["file"]).is_file() else None
    n_frames = int(meta.get("frames") or 0)
    lo, hi = int(min(30, n_frames // 10)), max(int(n_frames - min(30, n_frames // 10)) - 1, 0)
    picks = sorted(set(np.linspace(lo, hi, max(int(frames), 1)).astype(int).tolist()))
    if depth is None:
        log(f"loading the depth model ({MODEL})")
        depth = DepthModel()
    inv_stack, col_stack, fits, t0 = [], [], [], time.perf_counter()
    for f, rgb in _frames(live / vid.get("file", "video.mp4"), picks):
        x = depth(rgb, shape)
        a, b, share = fit_ground(x[fit_px], 1.0 / tg[fit_px], seed=f)
        if not share:
            log(f"frame {f}: no ground fit, left out")
            continue
        inv = a * x + b
        inv[track_mask(tracks, f, K, R_cw, C, shape)] = np.nan
        inv_stack.append(inv.astype(np.float32))
        col_stack.append(cv2.resize(rgb, (shape[1], shape[0]), interpolation=cv2.INTER_AREA))
        fits.append(round(share, 3))
    if not inv_stack:
        raise RuntimeError("no frame could be fitted to the ground")
    with warnings.catch_warnings():                          # (a pixel masked in every frame: NaN)
        warnings.simplefilter("ignore", RuntimeWarning)
        inv = np.nanmedian(np.stack(inv_stack), axis=0)      # still camera: the passers-by drop out
    col = np.median(np.stack(col_stack), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        dep = np.where(inv > 1.0 / reach, 1.0 / inv, np.nan)  # sky and beyond the reach: none
    keep = np.isfinite(dep) & (dep > 0.5) & ~depth_edges(dep)
    P = C + (dep[..., None] * dirs)[keep]
    keep_rgb = col[keep]
    inside = np.linalg.norm(P[:, :2] - C[:2], axis=1) < reach
    cloud = VoxelCloud(voxel)
    cloud.add(P[inside], keep_rgb[inside])
    Pv, rgb, n = cloud.points()
    info = {"voxel_m": voxel, "model": getattr(depth, "name", "custom"), "frames": len(inv_stack),
            "made": datetime.now().isoformat(timespec="seconds"), "source": "site",
            "fit": {"inliers": fits}, "seconds": round(time.perf_counter() - t0, 1)}
    out = write_cloud(live, Pv, rgb, n, info)
    log(f"cloud: {out['n']} points from {len(inv_stack)} frames -> {live / 'cloud.bin'} (+ cloud.ply)")
    return out


# --------------------------------------------------------------------------- Realtime (step 2): a rough cloud while it records
RT_EVERY_S = 0.4             # a depth frame at most this often ...
RT_TURN_DEG = 4.0            # ... once the phone has turned this far from the last one's view
RT_AGAIN_S = 2.0             # (or this long after the last, turned or not)
RT_VOXEL_M = 0.15            # the rough cloud's voxel
RT_GRID = 480                # the depth grid: about this many points along the frame's long side
RT_BOX_PAD = 0.15            # whatever was detected (anyone, any vehicle) left out, its box grown by this
RT_MAX_VOXELS = 2_000_000    # it stops growing past this
RT_POSES = 1024              # the poses kept to look a frame's up (34 s)
RT_FINISH_S = 20.0           # at Stop: how long the frame being made may take (or the model to load)


def _nudge(deg: float) -> np.ndarray:
    """The tilt nudge's turn about the camera's own x axis (as live_track.sensor_camera)."""
    a = math.radians(deg)
    return np.array([[1.0, 0, 0], [0, math.cos(a), math.sin(a)], [0, -math.sin(a), math.cos(a)]])


def box_mask(boxes: np.ndarray, shape: tuple[int, int], step: int, pad: float = RT_BOX_PAD) -> np.ndarray:
    """True on the step grid inside the boxes (pixels x1 y1 x2 y2), each grown by pad of its size."""
    m = np.zeros(shape, bool)
    for x1, y1, x2, y2 in np.asarray(boxes, float).reshape(-1, 4):
        pw, ph = pad * (x2 - x1), pad * (y2 - y1)
        c0, c1 = int(max((x1 - pw) // step, 0)), int(min(-(-(x2 + pw) // step), shape[1]))
        r0, r1 = int(max((y1 - ph) // step, 0)), int(min(-(-(y2 + ph) // step), shape[0]))
        if c0 < c1 and r0 < r1:
            m[r0:r1, c0:c1] = True
    return m


class RealtimeCloud:
    """The point cloud's worker while it is on in Realtime (live_track.LiveTracker.set_cloud): its own
    thread, never the tracking's (tracking only offers a frame now and then: latest wins), loads the
    depth model and turns each offered frame into points of its recording's CloudBuilder. depth: a
    stand-in for tests. close(): the thread ends and the model is released."""

    def __init__(self, log=print, depth=None):
        self._log = log
        self.depth = depth
        self.state = "ready" if depth is not None else "loading"      # loading, ready, error
        self.error: str | None = None
        self._cv = threading.Condition()
        self._item = None                    # (builder, k, image, boxes, R_wc): the newest offered
        self._busy = None                    # the builder of the frame being made
        self._stop = False
        self.thread = threading.Thread(target=self._run, name="rt-cloud", daemon=True)
        self.thread.start()

    def builder(self, rec) -> "CloudBuilder":
        return CloudBuilder(self, rec)

    def offer(self, item) -> None:
        if self.state == "error":
            return
        with self._cv:
            self._item = item
            self._cv.notify_all()

    def idle(self, b, timeout: float) -> bool:
        """Waits until nothing of builder b waits or is being made (False: timed out)."""
        end = time.monotonic() + timeout
        with self._cv:
            while (self._item is not None and self._item[0] is b) or self._busy is b:
                left = end - time.monotonic()
                if left <= 0 or self._stop:
                    return False
                self._cv.wait(left)
        return True

    def _run(self) -> None:
        if self.depth is None:
            try:
                self.depth = DepthModel()
                self.state = "ready"
                self._log(f"point cloud: the depth model is ready ({MODEL})")
            except Exception as e:  # noqa: BLE001  (shown in the status; tracking goes on)
                self.state, self.error = "error", f"{type(e).__name__}: {e}"
                self._log(f"point cloud: the depth model did not load: {self.error}")
                with self._cv:
                    self._item = None
                    self._cv.notify_all()
                return
        while True:
            with self._cv:
                while self._item is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    break
                item, self._item = self._item, None
                self._busy = item[0]
            try:
                item[0].add(self.depth, *item[1:])
            except Exception as e:  # noqa: BLE001  (a frame left out; logged once)
                item[0].failed(e)
            finally:
                with self._cv:
                    self._busy = None
                    self._cv.notify_all()

    def close(self) -> None:
        with self._cv:
            self._stop, self._item = True, None
            self._cv.notify_all()
        self.thread.join(60)
        self.depth = None
        torch = sys.modules.get("torch")             # (only if it was loaded: off loads nothing)
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


class CloudBuilder:
    """One recording's rough cloud (plans/live_controller/11, step 2). Each depth frame is placed
    from the recorder's fixed position (the calibration's) turned as the phone was turned at that
    frame, its scale fitted to the flat ground; the boxes detected there are left out. Rough: a
    recorder who walks smears it (the accurate cloud, after Stop, is step 3); the recorder's move
    is never used for anything else. In memory while it records (snapshot: the Live view), written
    into the package at Stop (finish)."""

    def __init__(self, worker: RealtimeCloud, rec):
        self.worker, self.rec = worker, rec
        self.lock = threading.Lock()
        self.cloud = VoxelCloud(RT_VOXEL_M)
        self.frames = self.skipped = self.errors = 0
        self.fits: deque = deque(maxlen=200)
        self.version = 0
        self.model = None
        self.closed = False
        self._packed = (-1, b"")
        self._qk = np.full(RT_POSES, -1, np.int64)   # the poses by frame: a ring (the hub's thread writes)
        self._q = np.full((RT_POSES, 4), np.nan)
        self._last = None                    # (time, view direction) of the last frame offered
        self._turn = None                    # the phone's frame -> the package's (False: no poses)
        self._grid = None                    # (step, rays)

    # ---- the hub's thread and the tracking thread: quick, never raise
    def pose(self, k: int, q) -> None:
        try:
            i = int(k) % RT_POSES
            self._q[i] = q
            self._qk[i] = k
        except (TypeError, ValueError):
            pass

    def _pose(self, k: int):
        i = int(k) % RT_POSES
        q = self._q[i].copy()
        return q if self._qk[i] == k and np.all(np.isfinite(q)) else None

    def camera(self, q) -> np.ndarray:
        """world_from_camera at a frame: the phone's rotation then (q), turned as the calibration
        turned its still second (the start is north, the tilt nudge); without q: the calibrated."""
        from .live_track import mean_rotation, quat_matrix

        rec = self.rec
        R_cal = np.asarray(rec.R_cw, float).T
        if self._turn is None:
            qs = [x for x in (rec.cal_qs or []) if x is not None and np.all(np.isfinite(x))]
            self._turn = (R_cal @ (mean_rotation([quat_matrix(x) for x in qs]) @ _nudge(rec.tilt_offset)).T
                          if qs else False)
        if q is None or self._turn is False:
            return R_cal
        return self._turn @ quat_matrix(q) @ _nudge(rec.tilt_offset)

    def offer(self, k: int, img: np.ndarray, boxes: np.ndarray) -> None:
        """Frame k (BGR, already blurred) after tracking: handed to the worker when due (RT_EVERY_S
        since the last, and turned RT_TURN_DEG from it or RT_AGAIN_S later)."""
        try:
            if self.closed or self.rec.h is None or self.worker.state == "error" or len(self.cloud) >= RT_MAX_VOXELS:
                return
            now, last = time.monotonic(), self._last
            if last is not None and now - last[0] < RT_EVERY_S:
                return
            R = self.camera(self._pose(k))
            if (last is not None and now - last[0] < RT_AGAIN_S
                    and math.degrees(math.acos(min(1.0, float(R[:, 2] @ last[1])))) < RT_TURN_DEG):
                return
            self._last = (now, R[:, 2].copy())
            self.worker.offer((self, int(k), img, np.asarray(boxes, float).reshape(-1, 4), R))
        except Exception as e:  # noqa: BLE001  (never into tracking)
            self.failed(e)

    def failed(self, e: Exception) -> None:
        self.errors += 1
        if self.errors == 1:
            self.rec.log(f"point cloud: a frame left out: {type(e).__name__}: {e}")

    # ---- the worker's thread
    def add(self, depth, k: int, img: np.ndarray, boxes: np.ndarray, R_wc: np.ndarray) -> None:
        import cv2

        from .live_track import R_MAX_M

        rec = self.rec
        K, C = np.asarray(rec.K, float), np.asarray(rec.C, float)
        W, H = rec.size
        step = max(1, round(max(W, H) / RT_GRID))
        if self._grid is None or self._grid[0] != step:
            self._grid = (step, grid_rays(K, (W, H), step))
        rays = self._grid[1]
        shape = rays.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = depth(rgb, shape)
        dirs = rays @ R_wc.T
        dz = dirs[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            tg = np.where(dz < -1e-4, -C[2] / dz, np.nan)                # the flat ground's depth
        dist = tg * np.linalg.norm(rays, axis=-1)
        reach = R_MAX_M + MARGIN_M
        hide = box_mask(boxes, shape, step)
        fit_px = np.isfinite(tg) & (dist > FIT_NEAR) & (dist < min(FIT_FAR, reach)) & ~hide
        a, b, share = fit_ground(x[fit_px], 1.0 / tg[fit_px], seed=k)
        if not share:                                                     # no ground in view
            self.skipped += 1
            return
        inv = a * x + b
        inv[hide] = np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            dep = np.where(inv > 1.0 / reach, 1.0 / inv, np.nan)
        keep = np.isfinite(dep) & (dep > 0.5) & ~depth_edges(dep)
        P = C + (dep[..., None] * dirs)[keep]
        col = cv2.resize(rgb, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)[keep]
        inside = np.linalg.norm(P[:, :2] - C[:2], axis=1) < reach
        with self.lock:
            if self.closed:
                return
            self.cloud.add(P[inside], col[inside])
            self.frames += 1
            self.fits.append(round(share, 3))
            self.version += 1
            self.model = getattr(depth, "name", "custom")

    # ---- the Live view, Stop
    def snapshot(self, v0: int = -1) -> tuple[dict, bytes]:
        """{version, same (nothing new since version v0: no points sent), n, frames, state, error,
        voxel_m, rough} and the points (CLOUD_DTYPE records)."""
        with self.lock:
            v = self.version
            head = {"version": v, "same": v == v0, "n": len(self.cloud), "frames": self.frames,
                    "state": self.worker.state, "error": self.worker.error, "voxel_m": RT_VOXEL_M,
                    "rough": True}
            if v == v0:
                return head, b""
            if self._packed[0] != v:
                self._packed = (v, records(*self.cloud.points()).tobytes())
            return head, self._packed[1]

    def finish(self, live: Path) -> dict | None:
        """At Stop: the frame being made let finish, then the files (None: no frame made one)."""
        self.worker.idle(self, RT_FINISH_S)
        with self.lock:
            self.closed = True
            P, rgb, n = self.cloud.points()
            frames, fits, model = self.frames, list(self.fits), self.model
        if not frames:
            return None
        return write_cloud(live, P, rgb, n, {
            "voxel_m": RT_VOXEL_M, "model": model, "frames": frames, "skipped": self.skipped,
            "made": datetime.now().isoformat(timespec="seconds"), "source": "realtime", "rough": True,
            "fit": {"inliers": fits}})
