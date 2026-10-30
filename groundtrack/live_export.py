"""Scene package of a run for the Live view: everything one clip needs, nothing that
identifies people.

    <run>/live/
      scene.json         the header: clip, origin, camera (P, K, R_cw, C), terrain, aerial,
                         tracks, video, R_max, the speed ramp, the files it was made from
      terrain.f32        float32 heights above the datum, row-major, row 0 = north, 1 m cells
                         (x0 / y1 = the grid's west / north edge; a value is its cell's centre)
      aerial.jpg         the aerial in grey over the terrain's extent (brightness: the view's)
      tracks.bin         one packed record per point (TRACK_DTYPE), sorted by frame, then track
      tracks_index.json  per track: class, group, first and last frame
      video.mp4          the clip, every frame with people and number plates blurred; if the
                         camera moved, steadied to the calibration frame (black past its edge)
      check.png          the calibration frame (blurred) with the tracks drawn through the
                         exported camera and a grid on the terrain: proves the export
      export_log.txt     what the export did

Frames: local metres, x east, y north, z up; (0, 0) = the calibration's origin, z = 0 its
datum (ground_z_m), as Homography._rays() works. Cameras in OpenCV axes (x right, y down,
z forward): x = K R_cw (X - C). Frame numbers count the video's frames (0 = the run's first).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import COCO_IDS, Config
from .homography import Homography
from .layout import open_run

VERSION = 1
CELL_M = 1.0
TERRAIN_MARGIN_M = 20.0     # the terrain reaches this far past R_max
AERIAL_PX = 2048            # the aerial's long side
MAX_VIDEO_W = 1920
SPEED_RAMP = {"lo": 1.0, "hi": 40.0, "scale": "log"}     # m/s, as the Richmond sheets
SOURCES = {"detected": 0, "predicted": 1, "stitched": 2, "interpolated": 3}
TRACK_DTYPE = np.dtype([("track_id", "<u4"), ("frame", "<u4"), ("t", "<f4"), ("x", "<f4"),
                        ("y", "<f4"), ("z", "<f4"), ("speed", "<f4"), ("class", "u1"),
                        ("source", "u1")])
TRACK_LAYOUT = (f"packed little-endian records of {TRACK_DTYPE.itemsize} bytes: track_id u32, "
                "frame u32, t f32, x f32, y f32, z f32, speed f32, class u8, source u8 "
                "(0 seen, 1 predicted, 2 stitched, 3 interpolated); sorted by frame, then track")
GRID_M = 5.0                # check.png: ground grid spacing
STILL_CONF = 0.08           # the extra person detection on every frame (calfigure.still_people)


# --------------------------------------------------------------------------- camera
def camera_from_P(P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """K, R_cw, C of a 3x4 camera matrix (RQ decomposition of its left 3x3).

    K is upper triangular with a positive diagonal and K[2, 2] = 1, R_cw a rotation (det +1),
    C the centre; K @ R_cw @ (X - C) is P @ [X, 1] times a positive scale."""
    from scipy.linalg import rq

    P = np.asarray(P, float)
    K, R = rq(P[:, :3])
    D = np.diag(np.sign(np.diag(K)))
    K, R = K @ D, D @ R
    if np.linalg.det(R) < 0:
        raise ValueError("the camera matrix is mirrored (det R = -1 with points in front at "
                         "positive depth)")
    C = -np.linalg.solve(P[:, :3], P[:, 3])
    K = K / K[2, 2]
    Prec = K @ np.column_stack([R, -R @ C])
    s = float(np.sum(Prec * P) / np.sum(P * P))
    err = np.abs(Prec - s * P).max() / np.abs(s * P).max()
    assert s > 0 and err < 1e-6, f"K R_cw [I | -C] does not reproduce P ({err:.2e})"
    return K, R, C


def project(K: np.ndarray, R: np.ndarray, C: np.ndarray, X: np.ndarray):
    """Local points (N, 3) -> pixels (N, 2) and depth (N,): x = K R (X - C)."""
    q = (np.asarray(X, float).reshape(-1, 3) - C) @ R.T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = (q @ K.T)[:, :2] / q[:, 2:3]
    return uv, q[:, 2]


def _angles(R: np.ndarray) -> tuple[float, float, float]:
    """(yaw, tilt, roll) in degrees, as camera_params: yaw clockwise from north, tilt down."""
    right, fwd = R[0], R[2]
    yaw = np.arctan2(fwd[0], fwd[1])
    tilt = -np.arcsin(np.clip(fwd[2], -1, 1))
    r0 = np.array([np.cos(yaw), -np.sin(yaw), 0.0])
    d0 = np.cross(fwd, r0)
    roll = np.arctan2(right @ d0, right @ r0)
    return float(np.degrees(yaw) % 360), float(np.degrees(tilt)), float(np.degrees(roll))


def r_max(xy: np.ndarray, cam_xy) -> float:
    """How far out the view reaches: 1.25 x the 95th percentile of the tracks' distances
    from the camera, within 25-110 m (rp_drag_plans.Clip)."""
    d = np.hypot(xy[:, 0] - cam_xy[0], xy[:, 1] - cam_xy[1])
    d = d[np.isfinite(d)]
    far = float(np.percentile(d, 95)) if len(d) else 0.0
    return float(np.clip(1.25 * far, 25.0, 110.0))


# --------------------------------------------------------------------------- ground
def terrain_grid(h: Homography, centre, half: float, cell: float = CELL_M):
    """Heights above the datum (float32, row 0 = north) on cell x cell squares covering
    centre +- half (local metres), the grid snapped to whole cells. Each value is the ground
    at its cell's centre (Homography.ground_height: the datum past the terrain model).
    Returns (z, x0, y1): z[r, c] is at (x0 + (c + 0.5) cell, y1 - (r + 0.5) cell)."""
    x0 = float(np.floor((centre[0] - half) / cell) * cell)
    x1 = float(np.ceil((centre[0] + half) / cell) * cell)
    y0 = float(np.floor((centre[1] - half) / cell) * cell)
    y1 = float(np.ceil((centre[1] + half) / cell) * cell)
    nx, ny = int(round((x1 - x0) / cell)), int(round((y1 - y0) / cell))
    xs = x0 + (np.arange(nx) + 0.5) * cell
    ys = y1 - (np.arange(ny) + 0.5) * cell
    gx, gy = np.meshgrid(xs, ys)
    world = np.column_stack([gx.ravel(), gy.ravel()]) + np.asarray(h.origin)
    z = h.ground_height(world).reshape(ny, nx).astype(np.float32)
    return z, x0, y1


def aerial_grey(path: str | Path, bounds, size: tuple[int, int]) -> np.ndarray:
    """The GeoTIFF over world bounds (left, bottom, right, top) in grey (luminance, uint8),
    size (w, h); black past its edge."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds

    from .geo import _to_uint8

    w, hh = size
    with rasterio.open(path) as ds:
        epsg = ds.crs.to_epsg() if ds.crs else None
        if epsg not in (None, 27700):
            raise ValueError(f"{Path(path).name} is in EPSG:{epsg}, expected EPSG:27700")
        bands = [1, 2, 3] if ds.count >= 3 else [1]
        a = ds.read(bands, window=from_bounds(*bounds, transform=ds.transform),
                    out_shape=(len(bands), hh, w), boundless=True, fill_value=0,
                    resampling=Resampling.average)
        if len(bands) == 1 and ds.colorinterp and ds.colorinterp[0].name == "palette":
            lut = np.zeros((256, 3), np.uint8)
            for k, rgba in ds.colormap(1).items():
                if 0 <= k < 256:
                    lut[k] = rgba[:3]
            a = np.transpose(lut[a[0].astype(np.uint8)], (2, 0, 1))
    a = _to_uint8(a).astype(np.float32)
    g = a[0] if len(a) == 1 else a[0] * 0.299 + a[1] * 0.587 + a[2] * 0.114
    return np.clip(g + 0.5, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- tracks
def track_records(points: pd.DataFrame, h: Homography, start: int = 0, stride: int = 1,
                  fps: float = 30.0) -> tuple[np.ndarray, dict, dict]:
    """points.csv -> (records in TRACK_DTYPE, tracks_index, classes {"id": name}).

    Positions in local metres, z = the ground under each point above the datum (where the
    calibration's terrain put it). Frames count the package's video frames."""
    p = points.copy()
    p["fr"] = (p["frame"].to_numpy() - start) // max(stride, 1)
    p = p.sort_values(["fr", "track_id"], kind="stable").reset_index(drop=True)
    xy = p[["x", "y"]].to_numpy(float)
    if "source" in p:
        unknown = set(p["source"]) - set(SOURCES)
        if unknown:
            raise ValueError(f"points.csv: unknown source {sorted(unknown)}")
        src = p["source"].map(SOURCES).to_numpy()
    else:          # older runs: seen or predicted only
        src = p["predicted"].astype(str).str.lower().isin(["true", "1"]).to_numpy().astype(int)
    unknown = set(p["class"]) - set(COCO_IDS)
    if unknown:
        raise ValueError(f"points.csv: unknown class {sorted(unknown)}")
    rec = np.zeros(len(p), TRACK_DTYPE)
    rec["track_id"] = p["track_id"].to_numpy()
    rec["frame"] = p["fr"].to_numpy()
    rec["t"] = p["fr"].to_numpy() * max(stride, 1) / fps
    rec["x"] = xy[:, 0] - h.origin[0]
    rec["y"] = xy[:, 1] - h.origin[1]
    rec["z"] = h.ground_height(xy)
    rec["speed"] = p["speed"].to_numpy(float)
    rec["class"] = p["class"].map(COCO_IDS).to_numpy()
    rec["source"] = src
    index = {}
    for tid, g in p.groupby("track_id", sort=True):
        index[str(int(tid))] = {"class": str(g["class"].iloc[0]),
                                "group": str(g["group"].iloc[0]) if "group" in g else "",
                                "first_frame": int(g["fr"].min()),
                                "last_frame": int(g["fr"].max())}
    classes = {str(COCO_IDS[c]): c for c in sorted(set(p["class"]), key=COCO_IDS.get)}
    return rec, index, classes


def reprojection_errors(points: pd.DataFrame, run, cfg: Config, fps: float, stride: int,
                        h: Homography, K, R, C) -> pd.DataFrame:
    """Every seen point drawn through the exported camera against the detector's foot point
    in the calibration frame (camera motion taken out): track_id, frame, group, err_px."""
    from .registration import apply_registration, load_registration
    from .trajectories import foot_points, recover_feet

    cols = ["track_id", "frame", "group", "err_px"]
    if not run.tracks.exists() or not run.raw_tracks.exists():
        return pd.DataFrame(columns=cols)
    summ = pd.read_csv(run.tracks)
    if "tracker_ids" not in summ:
        return pd.DataFrame(columns=cols)
    pairs = summ[["track_id", "tracker_ids"]].assign(
        raw_id=summ["tracker_ids"].astype(str).str.split("+")).explode("raw_id")
    pairs["raw_id"] = pairs["raw_id"].astype(int)
    raw = pd.read_csv(run.raw_tracks)
    clean = cfg["cleaning"]
    if clean.get("recover_occluded_feet", True):
        uv, _ = recover_feet(raw, int(round(float(clean.get("feet_window_s", 2.0)) * fps
                                            / max(stride, 1))),
                             float(clean.get("squish_ratio", 0.8)))
    else:
        uv = foot_points(raw)
    if run.registration.exists():
        reg = load_registration(run.registration)[0]
        uv = apply_registration(raw["frame"].to_numpy(), uv, reg)
    raw = pd.DataFrame({"raw_id": raw["track_id"], "frame": raw["frame"], "u": uv[:, 0],
                        "v": uv[:, 1]})
    seen = points[points["source"] == "detected"] if "source" in points else \
        points[~points["predicted"].astype(str).str.lower().isin(["true", "1"])]
    m = (seen[["track_id", "frame", "group", "x", "y"]]
         .merge(pairs[["track_id", "raw_id"]], on="track_id")
         .merge(raw, on=["raw_id", "frame"])
         .drop_duplicates(["track_id", "frame"]))
    if not len(m):
        return pd.DataFrame(columns=cols)
    xy = m[["x", "y"]].to_numpy(float)
    X = np.column_stack([xy - np.asarray(h.origin), h.ground_height(xy)])
    px, depth = project(K, R, C, X)
    err = np.hypot(px[:, 0] - m["u"].to_numpy(), px[:, 1] - m["v"].to_numpy())
    err[~(depth > 0)] = np.nan
    return m.assign(err_px=err)[cols]


def _turn_note(run, K: np.ndarray) -> str | None:
    """How far the camera turned from the calibration frame (registration.npz)."""
    if not run.registration.exists():
        return None
    from .registration import load_registration

    reg, ref, _ = load_registration(run.registration)
    Kinv = np.linalg.inv(K)
    turn = 0.0
    for H in reg.values():
        d = np.linalg.det(H)
        if not np.isfinite(d) or d == 0:
            continue
        U, _, Vt = np.linalg.svd(Kinv @ (H / np.cbrt(d)) @ K)
        turn = max(turn, float(np.degrees(np.linalg.norm(cv2.Rodrigues(U @ Vt)[0]))))
    note = (f"camera motion: up to {turn:.1f} deg from the calibration frame (frame {ref}). "
            "registration.apply_registration mirrors rays turned past 90 deg (left as is); ")
    return note + ("this clip stays below it, so it does not bite." if turn < 90 else
                   "WARNING: this clip turns past it, positions there are mirrored.")


# --------------------------------------------------------------------------- video
def _still_model(cfg: Config) -> str:
    """The detector's weights, never downloaded here: the config's, else the repo's models/
    (set in memory only)."""
    p = Path(cfg.model_path())
    if p.is_file():
        return str(p)
    repo = Path(__file__).resolve().parents[1] / "models" / p.name
    if repo.is_file():
        cfg.data["detection"]["model"] = str(repo)
        return str(repo)
    raise FileNotFoundError(f"detection model {p.name} not found (looked in {p.parent} and "
                            f"{repo.parent}); live-export does not download it")


def write_video(cfg: Config, run, video: Path, out: Path, frames: range, fps: float,
                keep: int | None = None, detect: bool = True, device: str | None = None,
                steady: bool = True, log=print) -> tuple[int, np.ndarray | None, bool]:
    """The run's frames, people and number plates blurred on every one, as H.264 (+faststart):
    the run's boxes held as in the videos (debug_video.privacy_boxes + blur_boxes) and, with
    detect, anyone a full-resolution low-confidence detection finds on the frame (as
    calfigure.blurred_still). With steady and a camera-motion registration, each blurred frame
    is then warped into the calibration frame's pixels, as the overlay videos' stabilised
    output (black where the camera turned away): the picture holds the start view, the one
    the exported camera sees. No unblurred frame leaves memory. Returns (frames written, the
    frame `keep` as written, full size; steadied or not)."""
    import shutil

    from tqdm import tqdm

    from .debug_video import HOLD_S, blur_boxes, privacy_boxes, raw_all
    from .video import VideoSink

    if not shutil.which("ffmpeg"):
        raise RuntimeError("live-export needs ffmpeg (H.264) on the PATH")
    meta_fps = fps * max(frames.step, 1)
    hide = privacy_boxes(raw_all(run.root, pd.read_csv(run.raw_tracks)), True, True,
                         int(round(HOLD_S * meta_fps)))
    boxes = {f: g[["x1", "y1", "x2", "y2"]].to_numpy(float) for f, g in hide.groupby("frame")}
    reg = None
    if steady and run.registration.exists():
        from .registration import load_registration

        reg = load_registration(run.registration)[0]
    model = None
    if detect:
        from ultralytics import YOLO

        from .detect import _precision_kwargs
        from .device import half_supported, resolve_device

        dev = resolve_device(device or cfg["device"])
        precision = _precision_kwargs(bool(cfg["detection"].get("half")) and half_supported(dev))
        model = YOLO(_still_model(cfg))
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open the video {video}")
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, MAX_VIDEO_W / W)
    size = (int(round(W * scale)), int(round(H * scale)))
    sink = VideoSink(out, fps, size)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames.start)
    n, kept, extra_n, unreg = 0, None, 0, 0
    try:
        for f in tqdm(range(frames.start, frames.stop), desc="live video", unit="frame"):
            ok, img = cap.read()
            if not ok:
                break
            if (f - frames.start) % frames.step:
                continue
            blur_boxes(img, boxes.get(f, np.zeros((0, 4))))
            if model is not None:
                r = model.predict(img, imgsz=int(np.ceil(max(H, W) / 32) * 32), conf=STILL_CONF,
                                  classes=[0], device=dev, verbose=False, **precision)[0]
                b = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))
                pad = 0.15 * np.column_stack([b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]])
                blur_boxes(img, np.column_stack([b[:, :2] - pad, b[:, 2:] + pad]))
                extra_n += len(b)
            if reg is not None:          # blurred first, then steadied (debug_video's warp)
                H_f = reg.get(f)
                if H_f is None:
                    unreg += 1
                else:
                    img = cv2.warpPerspective(img, H_f, (W, H), flags=cv2.INTER_LINEAR,
                                              borderMode=cv2.BORDER_CONSTANT)
            if f == keep:
                kept = img.copy()
            sink.write(img if scale >= 1 else cv2.resize(img, size, interpolation=cv2.INTER_AREA))
            n += 1
    finally:
        cap.release()
        sink.release()
    log(f"video: {n} frames blurred ({len(hide)} run boxes held +-{HOLD_S:g} s"
        + (f", {extra_n} more people found by the per-frame detection" if detect else
           ", no per-frame detection") + ")"
        + (f", then steadied to the calibration frame ({len(reg)} registered frames"
           + (f"; {unreg} without a registration left as filmed" if unreg else "") + ")"
           if reg is not None else ", not steadied (no camera-motion registration)")
        + f", {size[0]}x{size[1]} @ {fps:.3f} fps -> {out}")
    return n, kept, reg is not None


# --------------------------------------------------------------------------- check image
def _ramp_bgr(speed: np.ndarray) -> np.ndarray:
    from .colors import cmap

    lo, hi = SPEED_RAMP["lo"], SPEED_RAMP["hi"]
    v = np.clip(np.log(np.maximum(np.nan_to_num(speed), 1e-6) / lo) / np.log(hi / lo), 0, 1)
    return (cmap()(v)[:, 2::-1] * 255).astype(int)


def check_image(img: np.ndarray, rec: np.ndarray, frame: int, h: Homography, K, R, C,
                reach: float) -> np.ndarray:
    """The frame (BGR, blurred) with every track drawn through the exported camera (speed
    ramp, guesses dotted white), the tracks' positions at `frame` as dots, and a GRID_M grid
    on the terrain out to `reach` m from the camera."""
    out = (img.astype(np.float32) * 0.8).astype(np.uint8)
    H, W = out.shape[:2]
    grid = out.copy()
    lo = np.floor((C[:2] - reach) / GRID_M) * GRID_M
    hi = np.ceil((C[:2] + reach) / GRID_M) * GRID_M
    t = np.arange(-reach, reach + 1e-6, 0.5)
    for axis in (0, 1):
        for c in np.arange(lo[axis], hi[axis] + 1e-6, GRID_M):
            xy = np.column_stack([np.full_like(t, c), C[1] + t] if axis == 0 else
                                 [C[0] + t, np.full_like(t, c)])
            z = h.ground_height(xy + np.asarray(h.origin))
            uv, depth = project(K, R, C, np.column_stack([xy, z]))
            ok = (depth > 0.5) & (np.hypot(*(xy - C[:2]).T) <= reach)
            edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
            for a, b in zip(edges[::2], edges[1::2]):
                if b - a >= 2:
                    cv2.polylines(grid, [np.round(uv[a:b] * 4).astype(np.int32)], False,
                                  (255, 255, 255), 1, cv2.LINE_AA, shift=2)
    out = cv2.addWeighted(grid, 0.45, out, 0.55, 0)
    X = np.column_stack([rec["x"], rec["y"], rec["z"]]).astype(float)
    uv, depth = project(K, R, C, X)
    col = _ramp_bgr(rec["speed"])
    order = np.lexsort((rec["frame"], rec["track_id"]))
    for i, j in zip(order[:-1], order[1:]):
        if rec["track_id"][i] != rec["track_id"][j] or not (depth[i] > 0 and depth[j] > 0):
            continue
        a, b = uv[i], uv[j]
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            continue
        if max(abs(a).max(), abs(b).max()) > 4 * max(W, H):
            continue
        pa, pb = tuple(np.round(a * 4).astype(int)), tuple(np.round(b * 4).astype(int))
        if rec["source"][j]:            # a guess: dotted white
            if (int(rec["frame"][j]) // 3) % 2 == 0:
                cv2.line(out, pa, pb, (235, 235, 235), 1, cv2.LINE_AA, shift=2)
        else:
            cv2.line(out, pa, pb, tuple(int(v) for v in col[j]), 2, cv2.LINE_AA, shift=2)
    now = np.flatnonzero((rec["frame"] == frame) & (depth > 0))
    for k in now:
        c = tuple(np.round(uv[k] * 4).astype(int))
        cv2.circle(out, c, 4 * 5, (20, 20, 22), -1, cv2.LINE_AA, shift=2)
        cv2.circle(out, c, 4 * 3, tuple(int(v) for v in col[k]), -1, cv2.LINE_AA, shift=2)
    return out


# --------------------------------------------------------------------------- the package
def find_site_config(run_dir: Path) -> Path | None:
    """sites/<site>.yaml of a run in runs/<site>/<run> (the project's layout)."""
    run_dir = Path(run_dir).resolve()
    p = run_dir.parent.parent.parent / "sites" / f"{run_dir.parent.name}.yaml"
    return p if p.exists() else None


def export_live(run_dir: str | Path, cfg: Config, detect: bool = True,
                device: str | None = None, log=None) -> dict:
    """Write <run>/live/ (see the module docstring) for a processed run. Returns the paths and
    the numbers (points, tracks, frames, R_max, reprojection errors, seconds)."""
    from .pipeline import RunLog

    t0 = time.time()
    run = open_run(run_dir)
    if not run.points.exists():
        raise FileNotFoundError(f"{run.points} is missing: process the run first "
                                "(groundtrack process)")
    if not run.raw_tracks.exists():
        raise FileNotFoundError(f"{run.raw_tracks} is missing: the boxes to blur come from it")
    live = run.root / "live"
    live.mkdir(parents=True, exist_ok=True)
    (live / "export_log.txt").unlink(missing_ok=True)
    log = log or RunLog(live / "export_log.txt")
    meta = json.loads(run.meta.read_text(encoding="utf-8"))
    hpath = run.homography_used if run.homography_used.exists() else cfg.path("homography")
    h = Homography.load(hpath)
    video = Path(meta["video"])
    stride = int(meta.get("vid_stride", 1))
    start, end = int(meta.get("start_frame", 0)), int(meta["end_frame"])
    fps = float(meta["fps"]) / stride
    log(f"live export of {run.root} (calibration {hpath})")
    if h.undistorted:
        log("WARNING: calibrated on undistorted frames; the video is not undistorted")

    # camera
    P, _, Ch = h._rays()
    K, R, C = camera_from_P(P)
    assert np.allclose(C, Ch, atol=1e-6)
    # the calibration's own numbers where it has them (camera_params), else from K and R_cw
    cp = getattr(h, "camera_params", None) or {}
    yaw, tilt, roll = _angles(R)
    W, Hh = h.image_size
    hfov = float(np.degrees(2 * np.arctan(W / (2 * K[0, 0]))))
    above = float(C[2] - h.ground_height(C[:2] + np.asarray(h.origin))[0])
    cam = {"P": P.tolist(), "K": K.tolist(), "R_cw": R.tolist(), "C": C.tolist(),
           "yaw_deg": float(cp.get("yaw_deg", yaw)) % 360,
           "tilt_deg": float(cp.get("tilt_deg", tilt)),
           "roll_deg": float(cp.get("roll_deg", roll)),
           "hfov_deg": float(cp.get("hfov_deg", hfov)),
           "height_m": float(cp.get("height_m") or above)}
    log(f"camera: C ({C[0]:.2f}, {C[1]:.2f}, {C[2]:.2f}) m, focal {K[0, 0]:.1f} px, yaw "
        f"{cam['yaw_deg']:.1f}, tilt {cam['tilt_deg']:.2f}, roll {cam['roll_deg']:.2f} deg "
        f"(from R_cw: {yaw:.1f}, {tilt:.2f}, {roll:.2f}); K R_cw (X - C) reproduces P")
    note = _turn_note(run, K)
    if note:
        log(note)

    # tracks
    points = pd.read_csv(run.points)
    rec, index, classes = track_records(points, h, start, stride, fps)
    reach = r_max(np.column_stack([rec["x"], rec["y"]]).astype(float), C[:2])
    (live / "tracks.bin").write_bytes(rec.tobytes())
    (live / "tracks_index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    log(f"tracks: {len(rec)} points, {len(index)} tracks ({run.points.name}: "
        f"{points['track_id'].nunique()} tracks), R_max {reach:.1f} m")

    # ground: terrain and aerial over R_max + the margin around the camera
    z, x0, y1 = terrain_grid(h, C[:2], reach + TERRAIN_MARGIN_M)
    ny, nx = z.shape
    z.astype("<f4").tofile(live / "terrain.f32")
    ext = [x0, x0 + nx * CELL_M, y1 - ny * CELL_M, y1]
    if cfg.path("geotiff") is None:
        raise ValueError("live-export needs the site's geotiff (the aerial)")
    k = AERIAL_PX / max(ext[1] - ext[0], ext[3] - ext[2])
    size = (int(round((ext[1] - ext[0]) * k)), int(round((ext[3] - ext[2]) * k)))
    E0, N0 = h.origin
    grey = aerial_grey(cfg.path("geotiff"), (ext[0] + E0, ext[2] + N0, ext[1] + E0, ext[3] + N0),
                       size)
    cv2.imwrite(str(live / "aerial.jpg"), grey, [cv2.IMWRITE_JPEG_QUALITY, 90])
    log(f"terrain: {nx} x {ny} cells of {CELL_M:g} m, heights {z.min():+.2f} to {z.max():+.2f} m "
        f"above the datum; aerial {size[0]} x {size[1]} px")

    # video, and the frame the calibration was made on for the check
    ref = int(h.reference_frame or 0)
    frames = range(start, end, stride)
    check_f = ref if ref in frames else start
    if check_f != ref:
        log(f"the calibration frame {ref} is not in the run: check.png shows frame {check_f}")
    n, still, steadied = write_video(cfg, run, video, live / "video.mp4", frames, fps, check_f,
                                     detect, device, log=log)
    if n != len(frames):
        log(f"the video has {n} of the run's {len(frames)} frames")

    # check: the tracks through the exported camera, against the detector's foot points
    rel = (check_f - start) // stride
    errs = reprojection_errors(points, run, cfg, float(meta["fps"]), stride, h, K, R, C)
    errs = errs[np.isfinite(errs["err_px"].astype(float))]
    people = errs[errs["group"] == "people"] if (errs["group"] == "people").any() else errs
    med = float(np.median(people["err_px"])) if len(people) else float("nan")
    at = people[people["frame"] == check_f]
    med_at = float(np.median(at["err_px"])) if len(at) else float("nan")
    by = ", ".join(f"{g} {np.median(e['err_px']):.1f}" for g, e in errs.groupby("group"))
    log(f"reprojection through the exported camera vs the detector's foot points: median "
        f"{med:.2f} px over {len(people)} seen points"
        + (f" ({people['group'].iloc[0]})" if len(people) else "")
        + f", {med_at:.2f} px on the check frame; per group: {by}")
    if still is not None:
        img = check_image(still, rec, rel, h, K, R, C, reach)
        if img.shape[1] > MAX_VIDEO_W:
            s = MAX_VIDEO_W / img.shape[1]
            img = cv2.resize(img, (MAX_VIDEO_W, int(round(img.shape[0] * s))),
                             interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(live / "check.png"), img)

    # the header, last: its presence means the package is complete
    scene = {
        "version": VERSION, "clip": run.root.parent.name, "run": run.root.name, "crs": h.crs,
        "origin": {"E": float(E0), "N": float(N0),
                   "datum_z": float(h.ground_z_m) if h.ground_z_m is not None else None},
        "fps": fps, "frames": n, "duration_s": n / fps, "image_size": [int(W), int(Hh)],
        "camera": cam,
        "terrain": {"file": "terrain.f32", "nx": nx, "ny": ny, "x0": x0, "y1": y1,
                    "cell": CELL_M},
        "aerial": {"file": "aerial.jpg", "extent": ext},
        "tracks": {"file": "tracks.bin", "n_points": int(len(rec)), "n_tracks": len(index),
                   "layout": TRACK_LAYOUT, "classes": classes},
        "video": {"file": "video.mp4", "blurred": True, "stabilized": steadied},
        "R_max": reach,
        "speed_ramp": SPEED_RAMP,
        "source_runs": {"points": "../data/points.csv",
                        "homography": Path(hpath).resolve().as_posix()
                        if hpath != run.homography_used else "../_working/homography_used.json"},
    }
    (live / "scene.json").write_text(json.dumps(scene, indent=2), encoding="utf-8")
    took = time.time() - t0
    files = {p.name: p.stat().st_size for p in sorted(live.iterdir()) if p.is_file()}
    log("files: " + ", ".join(f"{k} {v / 1e6:.2f} MB" for k, v in files.items()))
    log(f"live export took {took:.1f} s -> {live}")
    return {"dir": live, "scene": live / "scene.json", "check": live / "check.png",
            "n_points": int(len(rec)), "n_tracks": len(index), "frames": n, "R_max": reach,
            "reproj_median_px": med, "reproj_median_check_px": med_at, "errors": errs,
            "files": files, "seconds": took}
