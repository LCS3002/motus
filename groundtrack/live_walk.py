"""motus Live: the walk cloud (plans/live_controller/11, step 3).

After Stop, a Realtime recording's cloud placed sharp: the recorder's walk worked out from the
recording's (blurred) video by VGGT (facebook/VGGT-1B: every picked frame's camera and depth at
once), for the cloud ONLY. The tracks, Live's cameras, the calibration and the exports' tracks never
use it (29 Oct: the recorder is calibrated once; its moves are not corrected). Part of the optional
point cloud: off unless asked (Live: the Point cloud layer's Sharpen; `motus live-cloud <run>` on a
Realtime recording); nothing imports this module at load, and VGGT (the extra `cloud`) only loads
here.

How:
  1. Frames: RATE_HZ a second over the recording (fewer past MAX_FRAMES), each the first fresh one
     at its time (the package repeats a frame for the slots none came for), small (SIDE px long).
  2. People and vehicles left out: YOLO on the same frames of raw.mkv (unblurred, in memory only:
     only the boxes are kept; without raw.mkv, the blurred frames).
  3. VGGT on CHUNK frames at once (8 GB: ~5.6 GB for 48), batches sharing OVERLAP frames; each
     batch joined to the one before by the shared frames' points (a similarity, robustly).
  4. Pinned to the calibration: the frame of the still second the calibration took is the
     calibrated camera (its position and turn); the ground (the biggest level plane below it) gives
     the scale (the calibrated height) and evens out a small tilt left in the calibration.
  5. Each frame's confident pixels (CONF_KEEP), off depth edges, within the reach of where it was
     filmed from, merged on a voxel grid: cloud.bin / .json / .ply, "source": "walk". The rough
     cloud made while it recorded is kept as cloud_rough.* (`motus live-cloud <run> --rough`: back).

Measured 1 Nov 2026, RTX 5060 Laptop (8 GB): VGGT's 1B backbone in bf16, heads in fp32; 48 frames
of 518 x 294 at once: 16 s, 5.6 GB; its focal on the 30 Oct recording 1609 px (the calibration's 1598).
"""
from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .live_cloud import (
    MARGIN_M,
    ROUGH_FILES,
    RT_BOX_PAD,
    VOXEL_M,
    VoxelCloud,
    box_mask,
    depth_edges,
    write_cloud,
)

MODEL = "facebook/VGGT-1B"
RATE_HZ = 2.0                # frames a second of the recording
MAX_FRAMES = 240             # at most (a longer recording: fewer a second)
CHUNK = 48                   # frames VGGT takes at once (halved when the GPU runs out)
OVERLAP = 8                  # frames two batches share (their join)
SIDE = 518                   # VGGT's input: the long side, px (the short one a multiple of 14)
CONF_KEEP = 0.5              # per frame, this share of its most confident pixels ...
CONF_MIN = 1.5               # ... and none below this (VGGT's confidence: 1 = none)
JOIN_MIN_PX = 200            # two batches join on at least this many confident shared pixels ...
JOIN_MAX_ERR = 0.1           # ... and fit within this (median residual / spread); else the walk breaks there
FRESH = 0.6                  # a frame differs from the one before by this (mean grey level): not a repeat
FRESH_WAIT_S = 0.5           # a picked time waits this long for a fresh frame
NEAR_M = 0.5                 # nothing nearer the camera
UP_DEG = 20.0                # the ground: a plane whose normal is within this of the vertical
GROUND_TOL = 0.012           # its points within this share of the frames' median distance (before the scale)
TILT_WARN_DEG = 8.0          # a bigger fix of the calibration's tilt / roll is said in the log
LOWER_SHARE = 0.3            # a plane below the biggest is the ground if it holds this share of its points
GROUND_WARN = 0.12           # less of the points on the ground: the scale is said to be unsure
DETECT_IDS = (0, 1, 2, 3, 5, 7)     # people and vehicles, as tracking blurs and tracks them (COCO)
DETECT_CONF = 0.2


# --------------------------------------------------------------------------- the recording
def read_poses(live: Path) -> np.ndarray:
    """recorder_pose.rec (live_track.POSE_DTYPE), by time."""
    from .live_track import POSE_DTYPE

    p = Path(live) / "recorder_pose.rec"
    a = np.fromfile(p, POSE_DTYPE) if p.is_file() else np.zeros(0, POSE_DTYPE)
    return a[np.argsort(a["t"], kind="stable")]


def still_window(poses: np.ndarray) -> list[int]:
    """The frames (k) of the second the calibration took (live_track.Recording._still / cal_due):
    the first still second (every pose within STILL_DEG of their mean), or, never still for
    CAL_TIMEOUT_S, that last second. [] without poses."""
    from .live_track import CAL_S, CAL_TIMEOUT_S, FPS, STILL_DEG

    ok = poses[np.all(np.isfinite(poses["q"]), axis=1)] if len(poses) else poses
    if not len(ok):
        return []
    win: list[int] = []
    t0 = float(ok["t"][0])
    for i in range(len(ok)):
        t = float(ok["t"][i])
        win.append(i)
        while len(win) > 1 and ok["t"][win[0]] < t - CAL_S:
            win.pop(0)
        Q = ok["q"][win].astype(float)
        Q *= np.where(Q @ Q[0] < 0, -1.0, 1.0)[:, None]
        m = Q.sum(0)
        m /= np.linalg.norm(m)
        if 2 * math.degrees(math.acos(min(1.0, float(np.abs(Q @ m).min())))) > STILL_DEG:
            win = [i]
        if t - float(ok["t"][win[0]]) >= CAL_S - 1.5 / FPS:
            return [int(ok["k"][j]) for j in win]
        if t - t0 >= CAL_TIMEOUT_S:
            return [int(ok["k"][j]) for j in range(len(ok)) if ok["t"][j] >= t - CAL_S]
    return [int(ok["k"][j]) for j in range(len(ok)) if ok["t"][j] >= float(ok["t"][-1]) - CAL_S]


def pick_times(n_frames: int, fps: float, anchor: int, rate: float = RATE_HZ, most: int = MAX_FRAMES) -> list[int]:
    """The frames to take (k): rate a second over the recording (at most most), and the anchor."""
    n = max(int(n_frames), 1)
    step = max(fps / max(rate, 1e-3), n / max(most - 1, 1))
    ks = set(np.round(np.arange(0, n, step)).astype(int).tolist()) | {int(min(max(anchor, 0), n - 1))}
    return sorted(k for k in ks if 0 <= k < n)


def small_size(W: int, H: int, side: int = SIDE) -> tuple[int, int]:
    """VGGT's input size for a W x H frame: the long side side, the other a multiple of 14."""
    if W >= H:
        return side, max(14, int(round(H * side / W / 14)) * 14)
    return max(14, int(round(W * side / H / 14)) * 14), side


def read_frames(video: Path, ks: list[int], size: tuple[int, int], fps: float, log=print) -> list[tuple[int, np.ndarray]]:
    """[(k, rgb at size)]: for each wanted k, the first fresh frame from it on (a repeat of the frame
    before is skipped, up to FRESH_WAIT_S; then it is taken as it is). One pass through the video."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"{video}: cannot be read")
    want, out, i, k, prev = sorted(set(ks)), [], 0, -1, None
    wait = max(1, int(round(FRESH_WAIT_S * fps)))
    try:
        while i < len(want):
            ok, bgr = cap.read()
            if not ok:
                break
            k += 1
            grey = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA).astype(np.int16)
            fresh = prev is None or float(np.abs(grey - prev).mean()) > FRESH
            prev = grey
            if k < want[i]:
                continue
            if fresh or k - want[i] >= wait:
                out.append((k, cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), size, interpolation=cv2.INTER_AREA)))
                while i < len(want) and want[i] <= k:
                    i += 1
    finally:
        cap.release()
    if len(out) < len(want):
        log(f"walk: {len(out)} of {len(want)} frames read")
    return out


def yolo_detector(log=print):
    """(rgb full frame, its k) -> boxes (x1 y1 x2 y2, px) of people and vehicles: the model tracking uses."""
    from ultralytics import YOLO

    from .device import resolve_device
    from .live_track import realtime_config

    cfg = realtime_config()
    dev = resolve_device(cfg["device"])
    model = YOLO(cfg.model_path())
    imgsz = int(cfg["detection"]["imgsz"])
    log(f"walk: people and vehicles found with {Path(cfg.model_path()).name} on {dev}")

    def detect(rgb: np.ndarray, k: int | None = None) -> np.ndarray:
        r = model.predict(np.ascontiguousarray(rgb[..., ::-1]), imgsz=imgsz, conf=DETECT_CONF, classes=list(DETECT_IDS), device=dev,
                          verbose=False)[0]
        return r.boxes.xyxy.cpu().numpy().reshape(-1, 4)

    return detect


def find_boxes(run: Path, live: Path, ks: list[int], size: tuple[int, int], full: tuple[int, int], detector,
               log=print) -> tuple[dict, str]:
    """{k: boxes at size}: the people and vehicles in frame k, from raw.mkv (unblurred, never kept:
    only the boxes) or, without it, the blurred video. (boxes, where from)."""
    import cv2

    raw = Path(run) / "raw.mkv"
    src, where = (raw, "raw") if raw.is_file() else (live / "video.mp4", "blurred")
    sx, sy = size[0] / full[0], size[1] / full[1]
    cap = cv2.VideoCapture(str(src))
    out = {}
    try:
        for k in ks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
            ok, bgr = cap.read()
            if not ok:
                out[k] = np.zeros((0, 4))
                continue
            if (bgr.shape[1], bgr.shape[0]) != tuple(full):
                bgr = cv2.resize(bgr, tuple(full), interpolation=cv2.INTER_AREA)
            b = np.asarray(detector(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), k), float).reshape(-1, 4)
            out[k] = b * [sx, sy, sx, sy]
            del bgr
    finally:
        cap.release()
    if where == "blurred":
        log("walk: no raw.mkv: people found in the blurred video (some may be missed)")
    return out, where


# --------------------------------------------------------------------------- VGGT
class VGGTModel:
    """facebook/VGGT-1B on the GPU: its backbone in bf16 (2.5 GB), the camera and depth heads in fp32
    (no point or track head). frames (S rgb, h x w, the same size) -> (R_cw S x 3 x 3, t S x 3:
    camera_from_world in the first frame's frame, OpenCV axes; K S x 3 x 3 at h x w; depth S x h x w;
    conf S x h x w)."""

    def __init__(self, name: str = MODEL):
        import torch
        try:
            from vggt.models.vggt import VGGT
        except ImportError as e:
            raise RuntimeError("VGGT is not installed: pip install -e .[cloud] (the walk cloud's model)") from e

        self.torch, self.name = torch, name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        m = VGGT.from_pretrained(name)
        m.point_head = None
        m.track_head = None
        m.eval()
        if self.device == "cuda":
            m.aggregator.to(torch.bfloat16)
        self.model = m.to(self.device)

    def __call__(self, frames: list[np.ndarray], keys=None):
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        torch = self.torch
        x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().div(255)[None].to(self.device)
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
                toks, psi = self.model.aggregator(x)
            toks = [None if a is None else a.float() for a in toks]
            pose = self.model.camera_head(toks)[-1]
            depth, conf = self.model.depth_head(toks, images=x, patch_start_idx=psi)
            E, K = pose_encoding_to_extri_intri(pose, x.shape[-2:])
        del toks, x
        E, K = E[0].float().cpu().numpy(), K[0].float().cpu().numpy()
        return (E[:, :, :3].astype(np.float64), E[:, :, 3].astype(np.float64), K.astype(np.float64),
                depth[0, ..., 0].float().cpu().numpy(), conf[0].float().cpu().numpy())

    def close(self) -> None:
        self.model = None
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def _out_of_memory(e: Exception) -> bool:
    return "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError"


# --------------------------------------------------------------------------- the geometry
def pixel_rays(K: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Camera rays (z = 1, OpenCV axes) of every pixel's centre: h x w x 3."""
    h, w = hw
    U, V = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    return np.stack([(U - K[0, 2]) / K[0, 0], (V - K[1, 2]) / K[1, 1], np.ones_like(U)], -1)


def points_of(R_cw: np.ndarray, t: np.ndarray, K: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """A frame's pixels in its world (h x w x 3) from its camera and depth."""
    cam = pixel_rays(K, depth.shape) * depth[..., None]
    return (cam - t) @ R_cw                                # R_cw.T @ (x - t), row-wise


def umeyama(A: np.ndarray, B: np.ndarray, w: np.ndarray | None = None) -> tuple[float, np.ndarray, np.ndarray]:
    """The similarity B ~ s R A + t (weighted least squares): (s, R, t)."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    w = np.ones(len(A)) if w is None else np.asarray(w, float)
    w = w / w.sum()
    ma, mb = w @ A, w @ B
    a, b = A - ma, B - mb
    S = (b * w[:, None]).T @ a
    U, D, Vt = np.linalg.svd(S)
    E = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt)) or 1.0])
    R = U @ E @ Vt
    va = float(w @ (a * a).sum(1))
    s = float(np.trace(np.diag(D) @ E) / va) if va > 0 else 1.0
    return s, R, mb - s * R @ ma


def robust_similarity(A: np.ndarray, B: np.ndarray, w: np.ndarray, iters: int = 6) -> tuple[float, np.ndarray, np.ndarray, float]:
    """umeyama, reweighted against outliers (Cauchy on the residual over its median): (s, R, t, the
    median residual over the median distance of B from its centre)."""
    w0 = np.asarray(w, float)
    s, R, t = umeyama(A, B, w0)
    for _ in range(iters):
        r = np.linalg.norm(B - (s * A @ R.T + t), axis=1)
        c = max(float(np.median(r)) * 2.0, 1e-9)
        s, R, t = umeyama(A, B, w0 / (1 + (r / c) ** 2))
    r = np.linalg.norm(B - (s * A @ R.T + t), axis=1)
    spread = float(np.median(np.linalg.norm(B - np.median(B, 0), axis=1))) or 1.0
    return s, R, t, float(np.median(r)) / spread


def _confident(conf: np.ndarray, keep: float = CONF_KEEP, least: float = CONF_MIN) -> np.ndarray:
    return conf >= max(float(np.quantile(conf, 1 - keep)), least)


def _join(b: dict, prev: dict, rng) -> tuple | None:
    """Batch b onto prev by the frames they share (their pixels' points: the same pixels in both):
    (s, R, t, err) or None (nothing confident to hold on to)."""
    A, B, W = [], [], []
    for k in [k for k in b["ks"] if k in prev["ks"]]:
        i, j = b["ks"].index(k), prev["ks"].index(k)
        m = _confident(b["conf"][i]) & _confident(prev["conf"][j])
        if not m.any():
            continue
        A.append(points_of(b["R_cw"][i], b["t"][i], b["K"][i], b["depth"][i])[m])
        B.append(points_of(prev["R_cw"][j], prev["t"][j], prev["K"][j], prev["depth"][j])[m])
        W.append(np.minimum(b["conf"][i][m], prev["conf"][j][m]))
    if not A or sum(len(a) for a in A) < JOIN_MIN_PX:
        return None
    A, B, W = np.concatenate(A), np.concatenate(B), np.concatenate(W)
    if len(A) > 60000:
        sel = rng.choice(len(A), 60000, replace=False)
        A, B, W = A[sel], B[sel], W[sel]
    return robust_similarity(A, B, W)


def join_batches(batches: list[dict], anchor: int | None = None, log=print, seed: int = 0):
    """The batches {ks, R_cw, t, K, depth, conf} joined one onto the next (by the frames each shares
    with the one before) into the first one's frame. Where two cannot be joined (nothing confident in
    the frames they share, or a join worse than JOIN_MAX_ERR) the walk breaks: the stretch holding the
    anchor frame is kept. ({k: (R_cw, t, K, depth, conf)} (a frame in two batches: the earlier's), each
    join's error (median residual / spread), the left-out stretches [(first k, last k)])."""
    rng = np.random.default_rng(seed)
    segs: list[dict] = []
    cur, T, prev = None, None, None
    for b in batches:
        res = _join(b, prev, rng) if prev is not None else None
        if res is not None and res[3] <= JOIN_MAX_ERR:
            s, R, t, err = res
            cur["errs"].append(round(err, 4))
            s0, R0, t0 = T                                   # the batch before -> the stretch's first; then this one
            T = (s0 * s, R0 @ R, s0 * R0 @ t + t0)
        else:
            if prev is not None:
                log(f"walk: frames {b['ks'][0]}-{b['ks'][-1]} do not join the ones before"
                    + (f" (off by {res[3]:.2f})" if res is not None else " (nothing to hold on to)"))
            cur = {"cams": {}, "errs": []}
            segs.append(cur)
            T = (1.0, np.eye(3), np.zeros(3))
        s, R, t = T
        for i, k in enumerate(b["ks"]):
            if k in cur["cams"]:
                continue
            Rc = b["R_cw"][i] @ R.T                          # x_cam (scaled by s) in the stretch's first frame
            cur["cams"][k] = (Rc, s * b["t"][i] - Rc @ t, b["K"][i], s * b["depth"][i], b["conf"][i])
        prev = b
    seg = next((g for g in segs if anchor in g["cams"]), segs[0])
    dropped = [(min(g["cams"]), max(g["cams"])) for g in segs if g is not seg]
    return seg["cams"], seg["errs"], dropped


def _plane(cand: np.ndarray, tol: float, cos_up: float, iters: int, rng):
    """The near-level plane below the origin with the most of cand within tol (RANSAC, refined on
    its inliers): (unit normal up, d, its count among cand) or None."""
    if len(cand) < 50:
        return None
    score = cand[rng.choice(len(cand), min(len(cand), 50000), replace=False)]
    best = None
    for _ in range(iters):
        p = cand[rng.choice(len(cand), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n = n / nn * (1 if n[2] >= 0 else -1)
        if n[2] < cos_up:
            continue
        d = -float(n @ p[0])
        if d <= 0:                                           # the plane must be below the camera
            continue
        c = int((np.abs(score @ n + d) < tol).sum())
        if best is None or c > best[2]:
            best = (n, d, c)
    if best is None or best[2] < 30:
        return None
    n, d = best[0], best[1]
    for _ in range(2):                                       # refined on its inliers
        P = cand[np.abs(cand @ n + d) < tol]
        if len(P) < 3:
            return None
        c = P.mean(0)
        n = np.linalg.svd(P - c, full_matrices=False)[2][2]
        n = n * (1 if n[2] >= 0 else -1)
        d = -float(n @ c)
    if n[2] < cos_up or d <= 0:
        return None
    return n, d, int((np.abs(cand @ n + d) < tol).sum())


def fit_ground(Y: np.ndarray, tol: float, up_deg: float = UP_DEG, iters: int = 500, seed: int = 0):
    """The ground among points Y (n x 3, z up, the camera at the origin): the lowest near-level plane
    below the camera holding at least LOWER_SHARE of the biggest one's points (outdoors the ground is
    the lowest and the biggest; a table, a bonnet or a step above it is not taken for it). (unit
    normal up, d: n . y + d = 0, so d is the camera's height above it; inlier mask over Y) or None."""
    rng = np.random.default_rng(seed)
    cos_up = math.cos(math.radians(up_deg))
    cand = Y[Y[:, 2] < 0]
    best = _plane(cand, tol, cos_up, iters, rng)
    if best is None:
        return None
    first = best[2]
    for _ in range(3):                                       # a substantial plane further down?
        n, d, _ = best
        lower = _plane(cand[cand @ n + d < -2 * tol], tol, cos_up, iters, rng)
        if lower is None or lower[2] < LOWER_SHARE * first or lower[1] <= d:
            break
        best = lower
    n, d, _ = best
    return n, d, np.abs(Y @ n + d) < tol


def level_turn(n: np.ndarray) -> np.ndarray:
    """The smallest turn taking unit vector n to the vertical (z)."""
    z = np.array([0.0, 0, 1])
    v, c = np.cross(n, z), float(n @ z)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def place_walk(cams: dict, anchor: int, R_wc_cal: np.ndarray, C: np.ndarray, height_m: float,
               seed: int = 0) -> dict:
    """The joined walk (join_batches) in the package's frame: the anchor frame is the calibrated
    camera (R_wc_cal at C), the ground below it the flat ground (z = 0, height_m below C: the
    scale), a tilt left in the calibration levelled out. {s, M (3 x 3: s x the turn), o (the anchor's
    centre in the joined frame), tilt_deg, ground share, ground rms (m)}; x_package = C + M (x - o)."""
    Rc, t = cams[anchor][0], cams[anchor][1]
    o = -Rc.T @ t                                           # the anchor camera's centre
    G = R_wc_cal @ Rc                                       # the joined frame -> the package's turn
    rng = np.random.default_rng(seed)
    Ys, dists = [], []
    for R_cw, tt, K, depth, conf in cams.values():
        P = points_of(R_cw, tt, K, depth)[_confident(conf)]
        if len(P) > 4000:
            P = P[rng.choice(len(P), 4000, replace=False)]
        Ys.append((P - o) @ G.T)
        dists.append(depth[_confident(conf)])
    Y = np.concatenate(Ys)
    scale0 = float(np.median(np.concatenate(dists)))
    g = fit_ground(Y, GROUND_TOL * scale0, seed=seed)
    if g is None:
        raise RuntimeError("no level ground found below the recorder: no scale for the cloud")
    n, d, inl = g
    s = float(height_m) / d
    L = level_turn(n)
    M = s * L @ G
    res = (Y[inl] @ n + d) * s
    return {"s": s, "M": M, "o": o, "C": np.asarray(C, float), "tilt_deg": math.degrees(math.acos(min(1.0, float(n[2])))),
            "ground_share": float(inl.mean()), "ground_rms_m": float(np.sqrt(np.mean(res ** 2))) if len(res) else math.nan}


# --------------------------------------------------------------------------- the cloud
def run_batches(frames: list[tuple[int, np.ndarray]], model, chunk: int = CHUNK, overlap: int = OVERLAP,
                log=print) -> list[dict]:
    """VGGT over the frames in batches of chunk sharing overlap; a batch the GPU cannot hold:
    smaller batches (halved, down to 2 x overlap)."""
    out, i, n = [], 0, len(frames)
    while i < n:
        c = min(chunk, n - i) if i == 0 else min(chunk, n - i + overlap)
        start = i if i == 0 else i - overlap
        part = frames[start:start + c]
        try:
            R_cw, t, K, depth, conf = model([f for _, f in part], [k for k, _ in part])
        except Exception as e:
            if _out_of_memory(e) and chunk > 2 * overlap + 2:
                chunk = max(2 * overlap + 2, chunk // 2)
                log(f"walk: the GPU is full: batches of {chunk}")
                import sys

                torch = sys.modules.get("torch")
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise
        out.append({"ks": [k for k, _ in part], "R_cw": R_cw, "t": t, "K": K, "depth": depth, "conf": conf})
        log(f"walk: frames {len(out[-1]['ks'])} ({start + 1}-{start + len(part)} of {n}) done")
        i = start + len(part)
    return out


def keep_rough(live: Path) -> bool:
    """The cloud made while it recorded (source realtime) kept as cloud_rough.* (once)."""
    live = Path(live)
    j = live / "cloud.json"
    if not j.is_file() or (live / "cloud_rough.json").is_file():
        return False
    try:
        if json.loads(j.read_text(encoding="utf-8")).get("source") != "realtime":
            return False
    except (OSError, ValueError):
        return False
    for f, r in zip(("cloud.bin", "cloud.json", "cloud.ply"), ROUGH_FILES):
        if (live / f).is_file():
            os.replace(live / f, live / r)
    return True


def restore_rough(live: Path) -> bool:
    """The rough cloud back as the cloud (the walk cloud's files go)."""
    live = Path(live)
    if not (live / "cloud_rough.json").is_file():
        return False
    for f, r in zip(("cloud.bin", "cloud.json", "cloud.ply"), ROUGH_FILES):
        if (live / r).is_file():
            os.replace(live / r, live / f)
        else:
            (live / f).unlink(missing_ok=True)
    return True


def make_walk_cloud(run: Path, rate: float = RATE_HZ, voxel: float = VOXEL_M, model=None, detector=None,
                    chunk: int = CHUNK, log=print) -> dict:
    """A Realtime recording's cloud from its walk (module docstring). model: a VGGTModel (or a stand-in:
    frames, keys -> R_cw, t, K, depth, conf); detector: (rgb, k) -> boxes (None: YOLO; False: nobody left
    out, tests only). Writes the files; returns cloud.json."""
    from .live_track import FPS, R_MAX_M

    run = Path(run)
    live = run / "live"
    meta = json.loads((live / "scene.json").read_text(encoding="utf-8"))
    if not meta.get("realtime"):
        raise ValueError("not a Realtime recording: a site camera's cloud is `live-cloud` without the walk")
    if meta.get("growing"):
        raise ValueError("the recording is still recording: sharpen its cloud after Stop")
    vid = meta.get("video") or {}
    if not vid.get("blurred"):
        raise ValueError(f"{live}: the package's video is not blurred: no cloud from it")
    rj = run / "recording.json"
    rmeta = json.loads(rj.read_text(encoding="utf-8")) if rj.is_file() else {}
    cam = meta["camera"]
    R_wc_cal, C = np.array(cam["R_cw"], float).T, np.array(cam["C"], float)
    height = float(cam.get("height_m") or C[2])
    W, H = (int(v) for v in meta["image_size"])
    fps = float(meta.get("fps") or FPS)
    n_frames = int(meta.get("frames") or 0)
    t0 = time.perf_counter()

    still = still_window(read_poses(live))
    anchor = still[len(still) // 2] if still else 0
    if not still:
        log("walk: no phone poses: the first frame taken as the calibrated camera")
    size = small_size(W, H)
    ks = pick_times(n_frames, fps, anchor, rate)
    frames = read_frames(live / vid.get("file", "video.mp4"), ks, size, fps, log)
    if len(frames) < 2:
        raise RuntimeError("too few frames in the recording's video")
    near = min(range(len(frames)), key=lambda i: abs(frames[i][0] - anchor))
    anchor = frames[near][0]
    log(f"walk: {len(frames)} frames ({rate:g} a second), the calibration's at frame {anchor}; {size[0]} x {size[1]}")

    if detector is None:
        detector = yolo_detector(log)
    if detector is False:
        boxes, where = {}, "none"
    else:
        boxes, where = find_boxes(run, live, [k for k, _ in frames], size, (W, H), detector, log)

    own = model is None
    if own:
        log(f"walk: loading {MODEL} (5 GB, downloaded once)")
        model = VGGTModel()
    try:
        batches = run_batches(frames, model, chunk, OVERLAP, log)
    finally:
        if own:
            model.close()
    cams, errs, dropped = join_batches(batches, anchor, log)
    del batches
    if dropped:
        log(f"walk: kept frames {min(cams)}-{max(cams)} (the calibration's stretch); left out "
            + ", ".join(f"{a}-{b}" for a, b in dropped))
    pl = place_walk(cams, anchor, R_wc_cal, C, height)
    if pl["tilt_deg"] > TILT_WARN_DEG:
        log(f"walk: the calibration's tilt / roll was off by {pl['tilt_deg']:.1f} deg (the tracks too): levelled for the cloud")
    if pl["ground_share"] < GROUND_WARN:
        log(f"walk: little ground in view ({pl['ground_share']:.0%} of the points; indoors?): the scale may be off")
    M, o = pl["M"], pl["o"]

    reach = R_MAX_M + MARGIN_M
    cloud, rgb_of = VoxelCloud(voxel), dict(frames)
    centres = []
    for k in sorted(cams):
        R_cw, t, K, depth, conf = cams[k]
        c_pkg = C + M @ (-R_cw.T @ t - o)
        centres.append(c_pkg)
        dep = depth * pl["s"]                                # metres
        keep = _confident(conf) & (dep > NEAR_M) & ~depth_edges(dep)
        if k in boxes and len(boxes[k]):
            keep &= ~box_mask(boxes[k], depth.shape, 1, RT_BOX_PAD)
        P = C + (points_of(R_cw, t, K, depth)[keep] - o) @ M.T
        inside = np.linalg.norm(P[:, :2] - c_pkg[:2], axis=1) < reach
        cloud.add(P[inside], rgb_of[k][keep][inside])
    if not len(cloud):
        raise RuntimeError("no point kept")
    path = np.array(centres)
    walked = float(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1).sum()) if len(path) > 1 else 0.0
    Pv, rgb, n = cloud.points()
    rough = keep_rough(live)
    info = {"voxel_m": voxel, "model": getattr(model, "name", MODEL), "frames": len(cams), "made":
            datetime.now().isoformat(timespec="seconds"), "source": "walk", "rough": False,
            "seconds": round(time.perf_counter() - t0, 1),
            "walk": {"rate_hz": rate, "batches": len(errs) + 1, "join_err": errs, "left_out": dropped, "anchor_frame": anchor,
                     "scale": round(pl["s"], 5), "height_m": height, "tilt_fix_deg": round(pl["tilt_deg"], 2),
                     "ground_share": round(pl["ground_share"], 3), "ground_rms_m": round(pl["ground_rms_m"], 3),
                     "walked_m": round(walked, 1), "people": where, "rough_kept": rough or (live / "cloud_rough.json").is_file(),
                     "recording": rmeta.get("id")}}
    out = write_cloud(live, Pv, rgb, n, info)
    log(f"walk cloud: {out['n']} points from {len(cams)} frames, walked {walked:.1f} m, ground +-{pl['ground_rms_m'] * 100:.0f} cm, "
        f"tilt fixed {pl['tilt_deg']:.1f} deg, {info['seconds']:.0f} s -> {live / 'cloud.bin'} (+ cloud.ply)")
    return out
