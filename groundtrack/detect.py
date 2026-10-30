"""Stage 1: detect + track with Ultralytics YOLO and ByteTrack / BoT-SORT.

We drive the tracker ourselves (rather than `model.track`) so that we can
  * drop detections outside the ROI and on-bike 'persons' *before* they reach the tracker,
  * read the tracker's lost tracks every frame and record their Kalman-predicted boxes
    while an object is occluded (`predicted=True`). Predicted rows are only kept if the
    track is re-acquired afterwards, i.e. they bridge a gap; trailing predictions after an
    object leaves the scene are discarded.

Nothing but box coordinates is stored: no crops, no appearance embeddings.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd

from .config import COCO_IDS, COCO_NAMES, Config
from .device import half_supported
from .video import point_in_polygon, video_info

RAW_COLUMNS = ["track_id", "class", "class_id", "frame", "time_s", "x1", "y1", "x2", "y2",
               "confidence", "predicted"]
RIDEABLE = {COCO_IDS["bicycle"], COCO_IDS["motorcycle"]}
# always detected and written to privacy_boxes.csv, whatever is tracked: everything that gets
# blurred (people, and vehicles for their number plates), before any filter drops a box -
# a person cut off by the bottom of the frame is no use for measuring but must be blurred
PRIVACY_IDS = {COCO_IDS[c] for c in ("person", "car", "motorcycle", "bus", "truck")}
PRIVACY_COLUMNS = ["frame", "class", "x1", "y1", "x2", "y2"]


def _load_tracker_cfg(name: str) -> tuple[str, SimpleNamespace]:
    from ultralytics.utils.checks import check_yaml

    try:
        from ultralytics.utils import YAML

        load = YAML.load
    except ImportError:  # older ultralytics
        from ultralytics.utils import yaml_load as load
    from ultralytics.utils import IterableSimpleNamespace

    fname = {"bytetrack": "bytetrack.yaml", "botsort": "botsort.yaml"}.get(name.lower(), name)
    cfg = load(check_yaml(fname))
    return cfg["tracker_type"], IterableSimpleNamespace(**cfg)


def make_tracker(name: str, updates_per_s: float, buffer_s: float | None = None,
                 device: str | None = None, log=print):
    """Build a tracker from an Ultralytics tracker yaml (bytetrack, botsort or a custom file).

    buffer_s: how long a lost track is kept alive (and predicted through occlusion), in
    seconds. Ultralytics counts this in tracker updates, so we convert using the effective
    update rate (fps / vid_stride).
    """
    from ultralytics.trackers.track import TRACKER_MAP

    ttype, args = _load_tracker_cfg(name)
    if ttype not in TRACKER_MAP:
        raise ValueError(f"Unsupported tracker_type {ttype!r}; available: {sorted(TRACKER_MAP)}")
    if buffer_s is not None:
        args.track_buffer = max(1, int(round(buffer_s * updates_per_s)))
    if getattr(args, "with_reid", False) and getattr(args, "model", "auto") == "auto":
        # 'auto' ReID reuses YOLO's internal features, only available inside model.track()
        log("NOTE: with_reid + model=auto is not supported here; ReID disabled. Point `model:` "
            "in a custom tracker yaml at a ReID/classification model (e.g. yolo26n-cls.pt).")
        args.with_reid = False
    args.device = device
    tracker = TRACKER_MAP[ttype](args=args)
    log(f"tracker {ttype}: lost tracks kept for {args.track_buffer} updates "
        f"({args.track_buffer / updates_per_s:.2f} s)")
    return tracker, ttype


def _suppress_riders(xyxy: np.ndarray, cls: np.ndarray, frac: float = 0.3) -> np.ndarray:
    """Mask out person boxes that substantially overlap a bicycle/motorcycle box (riders)."""
    keep = np.ones(len(cls), bool)
    persons = np.flatnonzero(cls == COCO_IDS["person"])
    bikes = np.flatnonzero(np.isin(cls, list(RIDEABLE)))
    if len(persons) == 0 or len(bikes) == 0:
        return keep
    for p in persons:
        px1, py1, px2, py2 = xyxy[p]
        parea = max((px2 - px1) * (py2 - py1), 1e-6)
        for b in bikes:
            bx1, by1, bx2, by2 = xyxy[b]
            iw = max(0.0, min(px2, bx2) - max(px1, bx1))
            ih = max(0.0, min(py2, by2) - max(py1, by1))
            # rider: overlaps the bike and the person's feet are inside the bike's height
            if iw * ih / parea > frac and by1 <= py2 <= by2 + 0.1 * (by2 - by1):
                keep[p] = False
                break
    return keep


def site_roi(cfg: Config, log=None):
    """ROI polygon from detection.roi, else `<site>_roi.json` next to the config (written
    by `groundtrack roi`), else None."""
    roi = cfg["detection"].get("roi")
    if roi is None:
        auto = cfg.base_dir / f"{cfg.site}_roi.json"
        if auto.exists():
            roi = str(auto)
            if log:
                log(f"using ROI {auto.name}")
    return _load_roi(roi, cfg.base_dir)


def _load_roi(roi, base_dir: Path):
    if roi is None:
        return None
    if isinstance(roi, str):
        p = Path(roi)
        p = p if p.is_absolute() else base_dir / p
        roi = json.loads(p.read_text(encoding="utf-8"))
    roi = np.asarray(roi, float)
    if roi.ndim != 2 or roi.shape[1] != 2 or len(roi) < 3:
        raise ValueError("detection.roi must be a list of >= 3 [x, y] pixel vertices")
    return roi


def reference_frame_for(cfg: Config) -> int:
    """The frame the homography was clicked on (stabilization aligns everything to it)."""
    hp = cfg.path("homography")
    if hp is not None and hp.exists():
        return int(json.loads(hp.read_text(encoding="utf-8")).get("reference_frame", 0))
    return int(cfg["detection"].get("reference_frame", 0))


def _precision_kwargs(half: bool) -> dict:
    """FP16 switch: `quantize=16` in current Ultralytics, `half=True` in older releases."""
    if not half:
        return {}
    from ultralytics.cfg import get_cfg

    return {"quantize": 16} if "quantize" in vars(get_cfg()) else {"half": True}


class _NumpyBoxes:
    """Minimal stand-in for ultralytics Boxes, which is what tracker.update() expects."""

    def __init__(self, xyxy, conf, cls):
        self.xyxy = xyxy.astype(np.float32)
        self.conf = conf.astype(np.float32)
        self.cls = cls.astype(np.float32)
        xywh = np.empty_like(self.xyxy)
        xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2
        xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2
        xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
        xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]
        self.xywh = xywh
        self.data = np.column_stack([self.xyxy, self.conf, self.cls])

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, idx):
        return _NumpyBoxes(self.xyxy[idx], self.conf[idx], self.cls[idx])


class FrameTracker:
    """One frame at a time, as run_tracking tracks a video: the filters (riders, classes, the
    bottom edge, the ROI), the tracker, and the boxes it predicts through an occlusion (kept once
    the track is found again). run_tracking and the realtime tracker (live_track.py) share it, so
    both find the same people in the same frames."""

    def __init__(self, det: dict, tracker, wanted, roi=None, record_predicted: bool = True):
        self.det, self.tracker, self.wanted, self.roi = det, tracker, set(wanted), roi
        self.record_pred = bool(record_predicted)
        self.pending: dict[int, list[tuple]] = {}
        self.n_pred_kept = 0

    @staticmethod
    def privacy_rows(fidx: int, xyxy: np.ndarray, cls: np.ndarray) -> list[tuple]:
        """PRIVACY_COLUMNS rows: every person / vehicle box of the frame, before any filter."""
        m = np.isin(cls, list(PRIVACY_IDS))
        return [(fidx, COCO_NAMES.get(int(c), str(c)), x1, y1, x2, y2)
                for (x1, y1, x2, y2), c in zip(xyxy[m], cls[m])]

    def step(self, fidx: int, t: float, frame: np.ndarray, xyxy: np.ndarray, conf: np.ndarray,
             cls: np.ndarray, H_ref: np.ndarray | None = None) -> list[tuple]:
        """The RAW_COLUMNS rows this frame adds: the tracks seen in it, and the predicted rows of
        tracks found again. H_ref: the frame's registration to the reference frame (the ROI was
        drawn there)."""
        det = self.det
        keep = np.ones(len(cls), bool)
        if det.get("suppress_riders"):
            keep &= _suppress_riders(xyxy, cls)
        keep &= np.isin(cls, list(self.wanted))
        margin = det.get("edge_margin_px")
        if margin is not None and margin >= 0:
            # a box cut off by the bottom of the frame has no real foot point
            keep &= xyxy[:, 3] < frame.shape[0] - margin
        if self.roi is not None and keep.any():
            foot = np.column_stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]])
            if H_ref is not None:  # the ROI was drawn on the reference frame
                foot = cv2.perspectiveTransform(foot.reshape(-1, 1, 2).astype(np.float64),
                                                H_ref).reshape(-1, 2)
            keep &= point_in_polygon(foot, self.roi)
        dets = _NumpyBoxes(xyxy[keep], conf[keep], cls[keep])
        out = self.tracker.update(dets, frame)

        rows: list[tuple] = []
        seen = set()
        for o in np.asarray(out).reshape(-1, out.shape[-1] if len(out) else 8):
            x1, y1, x2, y2, tid, score, c = o[:7]
            tid, c = int(tid), int(c)
            seen.add(tid)
            if tid in self.pending:
                self.n_pred_kept += len(self.pending[tid])
                rows.extend(self.pending.pop(tid))
            rows.append((tid, COCO_NAMES.get(c, str(c)), c, fidx, t, x1, y1, x2, y2,
                         float(score), False))
        if self.record_pred:
            lost = getattr(self.tracker, "lost_stracks", [])
            for s in lost:
                if s.track_id in seen or s.mean is None:
                    continue
                x1, y1, x2, y2 = s.xyxy
                c = int(s.cls)
                self.pending.setdefault(s.track_id, []).append(
                    (s.track_id, COCO_NAMES.get(c, str(c)), c, fidx, t, x1, y1, x2, y2,
                     np.nan, True))
            # forget pending predictions of tracks the tracker has removed for good
            alive = {s.track_id for s in lost}
            for tid in [k for k in self.pending if k not in alive]:
                del self.pending[tid]
        return rows


def detect_privacy_boxes(cfg: Config, video: Path, out_csv: Path, device: str, start: int,
                         end: int, stride: int, log=print) -> int:
    """Detection only: every person / vehicle box of every processed frame, for blurring
    (privacy_boxes.csv). For runs tracked before that file existed. Returns the box count."""
    from ultralytics import YOLO

    det = cfg["detection"]
    model = YOLO(cfg.model_path())
    half = bool(det.get("half")) and half_supported(device)
    rows: list[tuple] = []
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    batch: list[tuple[int, np.ndarray]] = []

    def flush():
        if not batch:
            return
        res = model.predict([f for _, f in batch], imgsz=det["imgsz"], conf=det["conf"],
                            iou=det["iou"], classes=sorted(PRIVACY_IDS), device=device,
                            verbose=False, **_precision_kwargs(half))
        for (fidx, _), r in zip(batch, res):
            b = r.boxes.cpu().numpy()
            for (x1, y1, x2, y2), c in zip(b.xyxy, b.cls.astype(int)):
                rows.append((fidx, COCO_NAMES.get(int(c), str(c)), x1, y1, x2, y2))
        batch.clear()

    for fidx in range(start, end):
        if (fidx - start) % stride:
            if not cap.grab():
                break
            continue
        ok, frame = cap.read()
        if not ok:
            break
        batch.append((fidx, frame))
        if len(batch) >= max(1, int(det.get("batch", 4))):
            flush()
    flush()
    cap.release()
    pd.DataFrame(rows, columns=PRIVACY_COLUMNS).round(1).to_csv(out_csv, index=False)
    log(f"privacy boxes: {len(rows)} person / vehicle boxes -> {out_csv}")
    return len(rows)


def run_tracking(cfg: Config, video: Path, out_csv: Path, device: str, log=print,
                 progress: bool = True, max_frames: int | None = None) -> pd.DataFrame:
    from ultralytics import YOLO

    det = cfg["detection"]
    info = video_info(video, cfg.get("fps"))
    stride = max(1, int(det["vid_stride"]))
    start = int(round(float(det.get("start_s") or 0) * info.fps))
    end = info.n_frames if det.get("end_s") is None else min(info.n_frames,
                                                             int(round(det["end_s"] * info.fps)))
    if max_frames:
        end = min(end, start + max_frames * stride)
    roi = site_roi(cfg, log)

    model = YOLO(cfg.model_path())
    tracker, ttype = make_tracker(det["tracker"], info.fps / stride, det.get("track_buffer_s"),
                                  device, log)
    wanted = set(cfg.class_ids)
    predict_ids = sorted(wanted | (RIDEABLE if det.get("suppress_riders") else set())
                         | PRIVACY_IDS)
    half = bool(det.get("half")) and half_supported(device)
    precision = _precision_kwargs(half)
    log(f"video {Path(video).name}: {info.width}x{info.height} @ {info.fps:.3f} fps, "
        f"{info.n_frames} frames ({info.duration_s:.1f} s); processing frames {start}-{end} "
        f"every {stride}")
    log(f"model {det['model']} imgsz={det['imgsz']} on {device}{' fp16' if half else ''}, "
        f"tracker {ttype}, classes {cfg.classes}")

    registrar, reg_frames, reg_H, reg_inl, ref_frame = None, [], [], [], 0
    stab = det.get("stabilize")
    if stab == "auto":
        from .registration import measure_camera_motion

        moved = measure_camera_motion(video, reference_frame_for(cfg))
        stab = moved > float(det.get("stabilize_threshold_px", 2.5))
        log("stabilize auto: " + (f"camera moved up to {moved:.1f} px" if np.isfinite(moved)
                                  else "the view moved too far to match the reference frame")
            + (" -> compensating camera motion" if stab else
               " -> static camera, no compensation needed"))
    if stab:
        from .registration import FrameRegistrar
        from .video import read_frame

        ref_frame = reference_frame_for(cfg)
        focal = None
        hp = cfg.path("homography")
        if hp is not None and hp.exists():     # calibrated: model the motion as a rotation
            focal = (json.loads(hp.read_text(encoding="utf-8")).get("camera_params")
                     or {}).get("focal_px")
        if focal:
            from .registration import RotationRegistrar

            registrar = RotationRegistrar(read_frame(video, ref_frame), float(focal))
        else:
            registrar = FrameRegistrar(read_frame(video, ref_frame))
        log(f"stabilize: registering every frame to frame {ref_frame} (the calibration frame)"
            + (", as a camera rotation (calibrated focal length)" if focal else ""))

    cap = cv2.VideoCapture(str(video))
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    rows: list[tuple] = []
    privacy_rows: list[tuple] = []
    ft = FrameTracker(det, tracker, wanted, roi, det.get("record_predicted", True))

    it = range(start, end)
    pbar = None
    if progress:
        from tqdm import tqdm

        pbar = tqdm(total=(end - start + stride - 1) // stride, unit="frame", desc="detect+track")
    batch_size = max(1, int(det.get("batch", 4)))

    def handle(fidx, frame, r):
        t = fidx / info.fps
        b = r.boxes.cpu().numpy()
        xyxy, conf, cls = b.xyxy, b.conf, b.cls.astype(int)
        H_ref = None
        if registrar is not None:
            H_ref, n_in = registrar.register(frame, xyxy)
            reg_frames.append(fidx)
            reg_H.append(H_ref)
            reg_inl.append(n_in)
        privacy_rows.extend(ft.privacy_rows(fidx, xyxy, cls))
        rows.extend(ft.step(fidx, t, frame, xyxy, conf, cls, H_ref))
        if pbar:
            pbar.update(1)

    batch: list[tuple[int, np.ndarray]] = []

    def flush():
        if not batch:
            return
        results = model.predict([f for _, f in batch], imgsz=det["imgsz"], conf=det["conf"],
                                iou=det["iou"], classes=predict_ids, device=device,
                                verbose=False, **precision)
        for (fidx, frame), r in zip(batch, results):   # tracking stays strictly in order
            handle(fidx, frame, r)
        batch.clear()

    for fidx in it:
        if (fidx - start) % stride:
            if not cap.grab():
                break
            continue
        ok, frame = cap.read()
        if not ok:
            break
        batch.append((fidx, frame))
        if len(batch) >= batch_size:
            flush()
    flush()
    cap.release()
    if pbar:
        pbar.close()
    n_pred_kept = ft.n_pred_kept

    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    df = df.sort_values(["track_id", "frame"]).reset_index(drop=True)
    for c in ("x1", "y1", "x2", "y2"):
        df[c] = df[c].astype(float).round(2)
    df["time_s"] = df["time_s"].round(4)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    pd.DataFrame(privacy_rows, columns=PRIVACY_COLUMNS).round(1).to_csv(
        out_csv.with_name("privacy_boxes.csv"), index=False)
    reg_info = {}
    if registrar is not None:
        from .registration import drift_px, save_registration

        reg_H = registrar.finalize(info.fps / stride)  # smooth: flow + reference fusion
        save_registration(out_csv.with_name("registration.npz"), reg_frames, reg_H, reg_inl,
                          ref_frame)
        drift = [drift_px(H, info.width, info.height) for H in reg_H]
        failed = int(sum(1 for k, ok in enumerate(registrar.abs_ok)
                         if k % registrar.abs_every == 0 and not ok))
        reg_info = {"reference_frame": ref_frame, "max_drift_px": round(max(drift), 1),
                    "registration_failures": failed}
        log(f"stabilize: camera moved up to {max(drift):.1f} px from the reference frame; "
            f"{failed} frames could not be registered (previous alignment reused)")
    meta = {
        "video": str(video), "fps": info.fps, "width": info.width, "height": info.height,
        "n_frames": info.n_frames, "start_frame": start, "end_frame": end, "vid_stride": stride,
        "model": det["model"], "tracker": ttype, "device": device,
        "n_tracks": int(df["track_id"].nunique()) if len(df) else 0,
        "n_rows": len(df), "n_predicted_rows": n_pred_kept, **reg_info,
    }
    out_csv.with_name("raw_tracks_meta.json").write_text(json.dumps(meta, indent=2),
                                                        encoding="utf-8")
    log(f"raw tracks: {meta['n_tracks']} tracks, {len(df)} rows "
        f"({n_pred_kept} predicted through occlusion) -> {out_csv}")
    return df
