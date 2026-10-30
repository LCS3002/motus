"""Optional debug.mp4: boxes, IDs and speed-coloured trails over the original video.

Off by default; people are blurred. Trails are drawn in the calibration frame's pixel space
and warped onto every video frame, so with `stabilize` they stay attached to the ground while
the camera moves. Colour = speed on the same blue -> red ramp as topdown.png:
  * calibrated runs (points.csv present): the smoothed ground path and its speed in m/s;
  * before calibration: smoothed image path; people's speed ~m/s from their box height,
    other classes relative on-screen speed.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .colors import group_rgb, ramp_rgb
from .config import Config
from .layout import open_run

GROUP_BGR = {"people": (255, 200, 0), "cycles": (0, 220, 120), "vehicles": (60, 80, 255)}


def _dashed_rect(img, p1, p2, color, thickness=1, dash=6):
    x1, y1 = p1
    x2, y2 = p2
    for a, b in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)),
                 ((x1, y2), (x1, y1))):
        length = int(np.hypot(b[0] - a[0], b[1] - a[1]))
        for s in range(0, length, dash * 2):
            e = min(s + dash, length)
            pa = (int(a[0] + (b[0] - a[0]) * s / max(length, 1)),
                  int(a[1] + (b[1] - a[1]) * s / max(length, 1)))
            pb = (int(a[0] + (b[0] - a[0]) * e / max(length, 1)),
                  int(a[1] + (b[1] - a[1]) * e / max(length, 1)))
            cv2.line(img, pa, pb, color, thickness, cv2.LINE_AA)


def blur_box(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    x1, y1 = max(int(x1), 0), max(int(y1), 0)
    x2, y2 = min(int(x2), w), min(int(y2), h)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    roi = img[y1:y2, x1:x2]
    k = max(15, (max(x2 - x1, y2 - y1) // 3) | 1)
    img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)


PERSON_HEIGHT_M = 1.7

VEHICLE_CLASSES = ("car", "truck", "bus", "motorcycle")
PLATE_BAND = 0.45         # lower part of a vehicle box: front and rear plates sit in it
PLATE_MIN_PX = 28         # smaller (further) vehicles: a plate is a few pixels, unreadable


HOLD_S = 0.25            # every blur box is kept this long before and after its detection:
                         # bridges the frames where the detector briefly loses someone


def privacy_boxes(r: pd.DataFrame, people: bool = True, plates: bool = True,
                  hold: int = 0) -> pd.DataFrame:
    """Boxes to blur (frame, x1, y1, x2, y2): every person, and the lower band of every
    vehicle close enough for its number plate to be readable. hold: also apply each box this
    many frames before and after (a person the detector misses for a frame stays blurred)."""
    cols = ["frame", "x1", "y1", "x2", "y2"]
    parts = []
    if people:
        parts.append(r.loc[r["class"] == "person", cols])
    if plates:
        v = r.loc[r["class"].isin(VEHICLE_CLASSES), cols]
        v = v[(v["y2"] - v["y1"]) >= PLATE_MIN_PX]
        parts.append(v.assign(y1=v["y2"] - PLATE_BAND * (v["y2"] - v["y1"])))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
    if hold > 0 and len(out):
        offs = np.arange(-int(hold), int(hold) + 1)
        out = out.loc[out.index.repeat(len(offs))].reset_index(drop=True)
        out["frame"] = out["frame"].to_numpy() + np.tile(offs, len(out) // len(offs))
    return out


def blur_boxes(img, boxes, fast: bool = False) -> None:
    """Blur every box; overlapping boxes (a held person, a crowd) are blurred as one area.
    fast (realtime, live_track.py, ~10x quicker): the areas found on a quarter-size mask and
    each blurred whole (its bounding rectangle: never less than the boxes), the large kernel's
    blur done on the area scaled down and back up."""
    b = np.asarray(boxes, float).reshape(-1, 4)
    if not len(b):
        return
    h, w = img.shape[:2]
    q = 4 if fast else 1
    m = np.zeros((-(-h // q), -(-w // q)), np.uint8)
    for x1, y1, x2, y2 in b:
        if fast:                                 # every quarter-pixel the box touches
            m[max(int(y1) // q, 0):max(-(-int(np.ceil(y2)) // q), 0),
              max(int(x1) // q, 0):max(-(-int(np.ceil(x2)) // q), 0)] = 1
        else:
            m[max(int(y1), 0):max(int(y2), 0), max(int(x1), 0):max(int(x2), 0)] = 1
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    for k in range(1, n):
        x, y, bw, bh, _ = stats[k] * q
        bw, bh = min(bw, w - x), min(bh, h - y)
        roi = img[y:y + bh, x:x + bw]
        kk = int(min(max(15, (min(bw, bh) // 3) | 1), 151))
        if fast:
            s = max(1, kk // 15)
            small = cv2.resize(roi, (max(1, bw // s), max(1, bh // s)), interpolation=cv2.INTER_AREA)
            small = cv2.GaussianBlur(small, ((kk // s) | 1, (kk // s) | 1), 0)
            roi[:] = cv2.resize(small, (bw, bh), interpolation=cv2.INTER_LINEAR)
            continue
        blurred = cv2.GaussianBlur(roi, (kk, kk), 0)
        sel = lab[y:y + bh, x:x + bw] == k
        roi[sel] = blurred[sel]


def _smooth_image_paths(raw: pd.DataFrame, uv: np.ndarray, fps: float, window_s: float = 0.4):
    """Per-track Savitzky-Golay smoothing of the foot points in image space (+ px/s speed)."""
    from scipy.signal import savgol_filter

    pos = uv.astype(float).copy()
    speed = np.full(len(raw), np.nan)
    tid_all = raw["track_id"].to_numpy()
    fr_all = raw["frame"].to_numpy()
    order = np.lexsort((fr_all, tid_all))
    for t in np.unique(tid_all):
        m = order[tid_all[order] == t]
        if len(m) < 5 or not np.isfinite(uv[m]).all():
            continue
        w = min(int(round(fps * window_s)) | 1, len(m) if len(m) % 2 else len(m) - 1)
        if w < 3:
            continue
        dt = max(float(np.median(np.diff(fr_all[m]))), 1.0) / fps
        for i in (0, 1):
            pos[m, i] = savgol_filter(uv[m, i], w, 2)
        speed[m] = np.hypot(savgol_filter(uv[m, 0], w, 2, deriv=1, delta=dt),
                            savgol_filter(uv[m, 1], w, 2, deriv=1, delta=dt))
    return pos, speed


def trail_samples(raw: pd.DataFrame, cfg: Config, run_dir: Path | None, fps: float,
                  uv: np.ndarray, window_s: float = 0.4):
    """Where to draw each raw row (reference-frame px) and its speed colour.

    Trails are always drawn where the video sees the object: its foot point, lightly
    smoothed in image space, so dots sit on the cars / people and trails do not lag.
    Only the colour depends on the mode:
      "calibrated": the smoothed metric speed (m/s) from points.csv; rows removed by
                    cleaning get NaN (hidden in the clean style, grey in the debug style);
      "approx":     not calibrated yet; people's on-screen speed converted to ~m/s using
                    their own box height (~1.7 m tall);
      "relative":   other classes before calibration: px/s scaled to the 95th percentile.
    Returns (pos, speed, lo, hi, mode).
    """
    n = len(raw)
    pos, px_speed = _smooth_image_paths(raw, uv, fps, window_s)
    rd = Path(run_dir) if run_dir else None
    run = open_run(rd) if rd is not None else None
    if run is not None and run.points.exists() and run.tracks.exists():
        pts = pd.read_csv(run.points)
        summ = pd.read_csv(run.tracks)
        lookup = {}
        for row in summ.itertuples():
            tp = pts[pts["track_id"] == row.track_id]
            sp = dict(zip(tp["frame"].tolist(), tp["speed"].tolist()))
            for orig in str(row.tracker_ids).split("+"):
                for f, v in sp.items():
                    lookup[(int(orig), int(f))] = v
        speed = np.array([lookup.get((int(t), int(f)), np.nan)
                          for t, f in zip(raw["track_id"].tolist(), raw["frame"].tolist())])
        groups = [cfg.group_of(c) for c in raw["class"]]
        lo = np.array([g.speed_range[0] if g else 0.0 for g in groups])
        hi = np.array([g.speed_range[1] if g else 1.0 for g in groups])
        return pos, speed, lo, hi, "calibrated"

    box_h = (raw["y2"] - raw["y1"]).to_numpy(float)
    is_person = (raw["class"] == "person").to_numpy()
    if is_person.all():
        # body heights per second -> ~m/s (median box height per track: robust to partial boxes)
        med_h = raw.assign(h=box_h).groupby("track_id")["h"].transform("median").to_numpy()
        speed = px_speed / np.maximum(med_h, 1.0) * PERSON_HEIGHT_M
        g = cfg.group_of("person")
        rng = g.speed_range if g else (0.0, 2.5)
        return pos, speed, np.full(n, rng[0]), np.full(n, rng[1]), "approx"
    ok = np.isfinite(px_speed) & ~raw["predicted"].to_numpy(bool)
    top = float(np.nanpercentile(px_speed[ok], 95)) if ok.any() else 1.0
    return pos, px_speed, np.zeros(n), np.full(n, max(top, 1e-6)), "relative"


def _bgr(speed, lo, hi):
    if not np.isfinite(speed):
        return (150, 150, 150)  # sample dropped by cleaning (short / parked / outside ROI)
    r, g, b = ramp_rgb(np.array([speed]), lo, hi)[0]
    return (int(b * 255), int(g * 255), int(r * 255))


def _legend(img, mode: str, cfg: Config, th: int, groups=()):
    """Speed bar (bottom left); with boxes, one small key per group above it."""
    h, w = img.shape[:2]
    bw, bh = int(w * 0.32), max(8, 6 * th)
    x0, y0 = 12 * th, h - 24 * th
    key_h = 14 * th if groups else 0
    cv2.rectangle(img, (x0 - 6 * th, y0 - 16 * th - key_h),
                  (x0 + bw + 6 * th, y0 + bh + 12 * th), (30, 30, 30), -1)
    x = x0
    for g in groups:
        r, gg, b = group_rgb(g)
        ky = y0 - 16 * th - key_h + 4 * th
        cv2.rectangle(img, (x, ky), (x + 7 * th, ky + 7 * th), (b, gg, r), max(1, th // 2 + 1),
                      cv2.LINE_AA)
        cv2.putText(img, g, (x + 10 * th, ky + 6 * th), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * th,
                    (255, 255, 255), max(1, th // 2), cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(g, cv2.FONT_HERSHEY_SIMPLEX, 0.4 * th, max(1, th // 2))
        x += 10 * th + tw + 12 * th
    bar = (ramp_rgb(np.linspace(0, 1, bw), 0, 1)[:, ::-1] * 255).astype(np.uint8)
    img[y0:y0 + bh, x0:x0 + bw] = bar[None, :, :]
    if mode == "relative":
        title, left, right = "speed (relative; calibrate for m/s)", "slow", "fast"
    else:
        g = cfg.group_of("person") if mode == "approx" else next(iter(cfg.groups.values()))
        title = "speed, m/s" + (" (approx. from body height)" if mode == "approx" else "")
        left, right = f"{g.speed_range[0]:g}", f"{g.speed_range[1]:g}"
    font, fs, tt = cv2.FONT_HERSHEY_SIMPLEX, 0.4 * th, max(1, th // 2)
    white = (255, 255, 255)
    cv2.putText(img, title, (x0, y0 - 5 * th), font, fs, white, tt, cv2.LINE_AA)
    cv2.putText(img, left, (x0, y0 + bh + 8 * th), font, fs, white, tt, cv2.LINE_AA)
    (tw, _), _ = cv2.getTextSize(right, font, fs, tt)
    cv2.putText(img, right, (x0 + bw - tw, y0 + bh + 8 * th), font, fs, white, tt, cv2.LINE_AA)


def _draw(canvas, alpha, a, b, col, t, opacity=255):
    pa = (int(round(a[0])), int(round(a[1])))
    pb = (int(round(b[0])), int(round(b[1])))
    cv2.line(canvas, pa, pb, col, t, cv2.LINE_AA)
    cv2.line(alpha, pa, pb, int(opacity), t, cv2.LINE_AA)


def _good_tracks(raw: pd.DataFrame, speed: np.ndarray, mode: str, fps: float,
                 min_track_s: float) -> set:
    """Tracks worth drawing in the clean style: survived cleaning (calibrated) or long enough."""
    if mode == "calibrated":
        return set(raw.loc[np.isfinite(speed), "track_id"].unique())
    span = raw.groupby("track_id")["frame"].agg(lambda f: (f.max() - f.min()) / fps)
    return set(span[span >= min_track_s].index)


def _box_in_ref(r, H):
    """Axis-aligned box of a frame-pixel bbox mapped into the reference frame."""
    c = np.array([[r.x1, r.y1], [r.x2, r.y1], [r.x2, r.y2], [r.x1, r.y2]], np.float64)
    if H is not None:
        c = cv2.perspectiveTransform(c.reshape(-1, 1, 2), H).reshape(-1, 2)
    return (int(c[:, 0].min()), int(c[:, 1].min())), (int(c[:, 0].max()), int(c[:, 1].max()))


def render_debug_video(video: Path, raw: pd.DataFrame, cfg: Config, out_path: Path,
                       fps: float, roi=None, log=print, max_width: int = 1920,
                       run_dir: Path | None = None, clean_path: Path | None = None) -> None:
    """Render the overlay video.

    style "clean" (default): only the tracks that survive cleaning, as speed-coloured trails
    fading out over the last `trail_s` seconds, with a dot and a thin box in the group's
    colour (people blue, cyclists green, vehicles red) at each current position.
    style "boxes": everything the tracker saw, with boxes, IDs and rejected tracks in grey.
    With a camera-motion registration the output itself is aligned to the calibration
    frame (`stabilize_output`), so the picture and the trails stay still.
    clean_path also writes the same video without the clock and legend.
    """
    from tqdm import tqdm

    from .registration import apply_registration, load_registration
    from .trajectories import foot_points, recover_feet

    dv = cfg["debug_video"]
    blur = bool(dv.get("blur_people", True))
    plates = bool(dv.get("blur_plates", True))
    trail_s = dv.get("trail_s")              # None = keep the whole path
    style = dv.get("style", "clean")
    show_boxes = bool(dv.get("boxes", True))   # clean style: thin box around each object
    raw = raw.copy()
    raw["predicted"] = raw["predicted"].astype(str).str.lower().isin(["true", "1"])
    if raw.empty:
        log("debug video: no tracks to draw")
        return
    raw = raw.sort_values(["frame", "track_id"]).reset_index(drop=True)

    reg = None
    if run_dir is not None and open_run(run_dir).registration.exists():
        reg, ref_frame, _ = load_registration(open_run(run_dir).registration)
    stable = reg is not None and bool(dv.get("stabilize_output", True))
    ref_img = None
    if stable:
        from .video import read_frame

        ref_img = read_frame(video, ref_frame)
        if blur or plates:  # the reference frame shows people and vehicles too
            for r in privacy_boxes(raw_all(run_dir, raw), blur, plates).query(
                    "frame == @ref_frame").itertuples():
                blur_box(ref_img, r.x1, r.y1, r.x2, r.y2)
    clean = cfg["cleaning"]
    if clean.get("recover_occluded_feet", True):
        foot, _ = recover_feet(raw, int(float(clean.get("feet_window_s", 2.0)) * fps),
                               float(clean.get("squish_ratio", 0.8)))
    else:
        foot = foot_points(raw)
    ref_uv = apply_registration(raw["frame"].to_numpy(), foot, reg) if reg else foot
    pos, speed, lo, hi, mode = trail_samples(raw, cfg, run_dir, fps, ref_uv)
    raw["ref_u"], raw["ref_v"], raw["spd"], raw["spd_lo"], raw["spd_hi"] = \
        pos[:, 0], pos[:, 1], speed, lo, hi
    if style == "clean":
        keep = _good_tracks(raw, speed, mode, fps, float(clean.get("min_track_s", 1.0)))
        raw = raw[raw["track_id"].isin(keep)]
        if mode == "calibrated":
            raw = raw[np.isfinite(raw["spd"])]
    draw_rows = {f: g for f, g in raw.groupby("frame")}

    cap = cv2.VideoCapture(str(video))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f0 = min(draw_rows) if draw_rows else 0
    f1 = max(draw_rows) if draw_rows else n_frames - 1
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, max_width / max(W, H))
    size = (int(W * scale), int(H * scale))
    from .video import VideoSink

    writer = VideoSink(out_path, fps, size)
    clean_writer = VideoSink(clean_path, fps, size) if clean_path else None
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    th = max(1, int(round(2 * max(W, H) / 1920)))
    # every blur needs the unfiltered detections, even in the clean style
    hold = int(round(HOLD_S * fps))
    blur_rows = {f: g[["x1", "y1", "x2", "y2"]].to_numpy()
                 for f, g in privacy_boxes(raw_all(run_dir, raw), blur, plates, hold)
                 .groupby("frame")} if (blur or plates) else {}

    canvas = np.zeros((H, W, 3), np.uint8)   # trail layer, calibration-frame pixels
    alpha = np.zeros((H, W), np.uint8)
    last: dict[int, tuple] = {}               # track -> (frame, point)
    segments: list[tuple] = []                # (frame, p0, p1, colour, thickness), windowed mode
    max_gap = max(1, int(fps * 0.5))
    dim = float(dv.get("dim", 1.0))
    key = ([g for g in cfg.groups if g in set(raw["class"].map(cfg.class_to_group))]
           if style == "clean" and show_boxes else [])

    for f in tqdm(range(f0, f1 + 1), desc="overlay video", unit="frame"):
        ok, img = cap.read()
        if not ok:
            break
        rows = draw_rows.get(f)
        new, heads = [], []
        if rows is not None:
            for r in rows.itertuples():
                pt = (float(r.ref_u), float(r.ref_v))
                if not (np.isfinite(pt[0]) and np.isfinite(pt[1])):
                    continue
                col = _bgr(r.spd, r.spd_lo, r.spd_hi)
                prev = last.get(r.track_id)
                if prev is not None and f - prev[0] <= max_gap:
                    if r.predicted:  # machine-made gap: dotted white
                        if f % 4 < 2:
                            new.append((f, prev[1], pt, (235, 235, 235), max(1, th - 1)))
                    else:
                        new.append((f, prev[1], pt, col, max(2, th)))
                last[r.track_id] = (f, pt)
                heads.append((pt, col, r))
        if trail_s is None:
            for _, a, b, c, t in new:
                _draw(canvas, alpha, a, b, c, t)
        else:
            span = max(float(trail_s) * fps, 1.0)
            segments.extend(new)
            segments = [sg for sg in segments if sg[0] >= f - span]
            canvas[:] = 0
            alpha[:] = 0
            for f_s, a, b, c, t in segments:      # oldest first: the newest on top
                _draw(canvas, alpha, a, b, c, t, 255 * (1 - (f - f_s) / span) ** 0.7)

        if f in blur_rows:
            blur_boxes(img, blur_rows[f])
        H_f = reg.get(f) if reg is not None else None
        if stable and H_f is not None:
            # output in calibration-frame pixels: the picture stops moving
            img = cv2.warpPerspective(img, H_f, (W, H), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT)
            # fill the edges the moved frame no longer covers with the (still) reference frame
            cover = cv2.warpPerspective(np.full((H, W), 255, np.uint8), H_f, (W, H),
                                        flags=cv2.INTER_NEAREST)
            if ref_img is not None and cover.min() == 0:
                hole = cover == 0
                img[hole] = ref_img[hole]
            lay, a = canvas, alpha
            to_out = None
        elif H_f is not None:
            Hinv = np.linalg.inv(H_f)
            lay = cv2.warpPerspective(canvas, Hinv, (W, H), flags=cv2.INTER_LINEAR)
            a = cv2.warpPerspective(alpha, Hinv, (W, H), flags=cv2.INTER_LINEAR)
            to_out = Hinv
        else:
            lay, a, to_out = canvas, alpha, None
        if dim < 1:
            img = (img * dim).astype(np.uint8)
        a = (a.astype(np.float32) * (0.92 / 255.0))[..., None]
        img = (img * (1 - a) + lay * a).astype(np.uint8)

        if roi is not None and style != "clean":
            poly = np.asarray(roi, np.float64).reshape(-1, 1, 2)
            if to_out is not None:
                poly = cv2.perspectiveTransform(poly, to_out)
            cv2.polylines(img, [poly.astype(np.int32)], True, (255, 255, 255), th, cv2.LINE_AA)
        for pt, col, r in heads:
            q = np.array([[pt]], np.float64)
            if to_out is not None:
                q = cv2.perspectiveTransform(q, to_out)
            c = (int(q[0, 0, 0]), int(q[0, 0, 1]))
            if style == "clean":
                if show_boxes and not r.predicted:      # a thin frame in the group colour
                    p1, p2 = _box_in_ref(r, H_f if stable else None)
                    if to_out is not None and not stable:
                        p1, p2 = _box_in_ref(r, None)
                    gr, gg, gb = group_rgb(cfg.class_to_group.get(r[2], ""))
                    cv2.rectangle(img, p1, p2, (gb, gg, gr), max(1, th - 1), cv2.LINE_AA)
                cv2.circle(img, c, 3 * th, (20, 20, 20), -1, cv2.LINE_AA)
                cv2.circle(img, c, 2 * th, col, -1, cv2.LINE_AA)
            else:
                box_col = GROUP_BGR.get(cfg.class_to_group.get(r[2], ""), (200, 200, 200))
                p1, p2 = _box_in_ref(r, H_f if stable else None)
                if r.predicted:
                    _dashed_rect(img, p1, p2, (180, 180, 180), max(1, th - 1))
                else:
                    cv2.rectangle(img, p1, p2, box_col, max(1, th - 1), cv2.LINE_AA)
                cv2.putText(img, f"{int(r.track_id)}" + (" pred" if r.predicted else ""),
                            (p1[0], max(p1[1] - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4 * th, box_col, max(1, th // 2), cv2.LINE_AA)
        if clean_writer is not None:
            clean_writer.write(img if scale >= 1 else
                               cv2.resize(img, size, interpolation=cv2.INTER_AREA))
        cv2.putText(img, f"t = {f / fps:6.2f} s", (10 * th, 18 * th), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6 * th, (255, 255, 255), max(1, th // 2), cv2.LINE_AA)
        _legend(img, mode, cfg, th, key)
        if scale < 1:
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        writer.write(img)
    writer.release()
    if clean_writer is not None:
        clean_writer.release()
    cap.release()
    log(f"overlay video -> {out_path} (style {style}"
        + (", people blurred" if blur else "") + (", number plates blurred" if plates else "") + (", stabilized" if stable else "")
        + f"; trail colour = speed, {mode})")


def raw_all(run_dir, fallback: pd.DataFrame) -> pd.DataFrame:
    """Every box to blur: the tracks (incl. those the clean style hides) and every untracked
    detection (privacy_boxes.csv: e.g. people cut off by the frame edge, not yet tracked)."""
    if run_dir is None:
        return fallback
    run = open_run(run_dir)
    r = pd.read_csv(run.raw_tracks) if run.raw_tracks.exists() else fallback
    if run.privacy.exists():
        p = pd.read_csv(run.privacy)
        r = pd.concat([r[["frame", "class", "x1", "y1", "x2", "y2"]], p], ignore_index=True)
    return r
