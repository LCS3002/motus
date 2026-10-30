"""Per-site YAML config: defaults, merging, path resolution and class groups."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# COCO class ids used by YOLO models.
COCO_IDS = {
    "person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "bus": 5, "truck": 7,
    "train": 6, "boat": 8, "dog": 16, "horse": 17,
}
COCO_NAMES = {v: k for k, v in COCO_IDS.items()}

DEFAULTS: dict[str, Any] = {
    "site": "site",
    "video": None,
    "geotiff": None,
    "homography": None,           # JSON written by `groundtrack calibrate`
    "lens": None,                 # optional JSON written by `groundtrack lens-calibrate`
    "terrain": None,              # ground with level changes (terraces, steps): a DTM GeoTIFF,
                                  # or "auto" (England: Environment Agency LiDAR 1 m, fetched).
                                  # camera_position.height_m is then above the ground there
    "output_dir": "runs",
    "models_dir": None,           # where bare model names (yolo26m.pt) are downloaded/cached
    "device": "auto",             # auto | cuda | mps | cpu
    "fps": None,                  # override if the container reports a wrong frame rate
    "detection": {
        "model": "yolo26m.pt",
        "imgsz": 1280,
        "conf": 0.25,
        "iou": 0.7,
        "tracker": "bytetrack",   # bytetrack | botsort | path/to/custom_tracker.yaml
        "track_buffer_s": 1.0,    # keep (and predict) lost tracks this long before giving up
        "vid_stride": 1,          # process every Nth frame
        "half": True,             # FP16 on CUDA
        "start_s": 0.0,
        "end_s": None,
        "roi": None,              # [[x, y], ...] pixel polygon in the video frame; detections
                                  # whose foot point is outside are ignored
        "record_predicted": True, # keep tracker-predicted positions bridging occlusions
        "suppress_riders": True,  # drop 'person' boxes sitting on a bicycle/motorcycle
        "stabilize": False,       # True | False | "auto" (measure camera motion first)
        "batch": 4,               # frames per YOLO call (GPU throughput)
        "edge_margin_px": 3,      # drop boxes touching the bottom frame edge: feet not visible
    },
    # Every tracked class must belong to exactly one group. Groups carry the
    # physical parameters (speed limits, smoothing, ground offset, colour range).
    "groups": {
        "people": {
            "classes": ["person"],
            "speed_range": [0.0, 2.5],      # m/s, colour ramp blue -> red
            "max_speed": 6.0,               # m/s, faster jumps are treated as tracking errors
            "jump_tolerance_m": 1.0,        # extra allowance per step for detection jitter
            "stitch_radius_m": 1.5,         # re-link fragments if the new one starts this close
                                            # to where the old one was heading (grows with gap)
            "smooth_window_s": 1.0,
            "ground_offset_m": 0.0,
        },
        "cycles": {
            "classes": ["bicycle"],
            "speed_range": [0.0, 10.0],
            "max_speed": 20.0,
            "jump_tolerance_m": 1.5,
            "stitch_radius_m": 2.5,
            "smooth_window_s": 0.8,
            "ground_offset_m": 0.0,
        },
        "trains": {
            "classes": ["train"],
            "speed_range": [0.0, 25.0],     # light rail / metro: up to ~90 km/h
            "max_speed": 35.0,
            "jump_tolerance_m": 5.0,
            "stitch_radius_m": 8.0,
            "min_displacement_m": 10.0,     # drop trains standing in a station the whole clip
            "smooth_window_s": 1.0,
            "ground_offset_m": 0.0,         # the box spans the whole train: its middle is fine
            "plane_height_m": "auto",       # elevated tracks (DLR viaduct): fitted to OSM rails
        },
        "vehicles": {
            "classes": ["car", "motorcycle", "bus", "truck"],
            "speed_range": [0.0, 35.0],
            "max_speed": 70.0,
            "jump_tolerance_m": 3.0,
            "stitch_radius_m": 4.0,
            "min_displacement_m": 5.0,      # drop parked vehicles (never move this far)
            "smooth_window_s": 0.6,
            # number, or per-class mapping e.g. {default: 2.2, bus: 5.5, truck: 5.5}
            "ground_offset_m": 2.2,
            # travel: shift the foot point back along the direction of travel (assumes the
            #         bbox bottom is the vehicle FRONT, i.e. traffic approaching the camera)
            # view:   shift it away from the camera, scaled by how end-on/side-on the
            #         vehicle is seen (correct for both carriageways; see README)
            "offset_mode": "travel",
            "vehicle_width_m": 1.8,         # used by offset_mode: view
        },
    },
    "projection": {
        # drop ground points farther than this from the camera (m). At shallow angles one
        # pixel of foot jitter becomes metres far away. A number, None (keep everything) or
        # "auto": per group, where the depth error reaches 0.25 m/px (people), 0.4 (cycles),
        # 1.0 (vehicles); needs a calibration with a camera estimate.
        "max_range_m": None,
    },
    "cleaning": {
        "min_track_s": 0.0,       # 0 keeps short pieces, as Live does; > 0 drops them as flickers
        "max_gap_s": 2.0,         # gaps longer than this split a track instead of interpolating
        "stitch": True,           # re-link tracks broken by occlusion, in ground coordinates
        "stitch_max_gap_s": 2.0,
        "smooth_polyorder": 2,
        "radial_smoothing": True,  # smooth depth (along the view ray) more than sideways
        "drop_passengers": True,   # people inside a bus / car box most of the time: passengers
        "hidden_feet": "keep",   # "bridge": leave out detections whose feet are hidden behind
                                 # nearer people (crowds), interpolate across them
        "min_moving_speed": 0.3,  # m/s below which heading is held from the last moving sample
        "recover_occluded_feet": True,  # rebuild foot points of boxes squished by occlusion
        "squish_ratio": 0.8,      # box shorter than this x its normal height = squished
        "feet_window_s": 2.0,     # window for a track's normal box height
    },
    "grid": {
        "cell_size_m": 1.0,
        "include_predicted": False,
    },
    # smoothed vector field for particle sims (vector_field.csv, flow_field.png, houdini_field.py)
    "field": {
        "cell_size_m": 1.0,
        "smooth_m": 2.0,            # Gaussian kernel sigma: bigger = smoother, less detail
        "confidence_samples": 15,   # effective samples for confidence ~0.63
        "time_window_s": None,      # e.g. 10 -> also a field per 10 s window (animated sims)
        "include_predicted": False,
        "direction_bins": 8,        # each cell follows its dominant direction (0 = plain mean)
    },
    "stats": {
        "histogram_bins": 20,
        # optional count lines in world coords: [{name: northbound, a: [E, N], b: [E, N]}]
        "count_lines": [],
    },
    "visuals": {
        "dpi": 300,
        "extent": "data",          # data | geotiff
        "margin_m": 10.0,
        "basemap_brightness": 0.55,
        "line_width": 1.2,
        "show_predicted": True,    # draw predicted (occlusion) segments dashed
        "topdown_video": True,     # topdown.mp4: movement animated over the map
        "topdown_video_speedup": 1.0,
        "flowfield_video": True,   # flowfield.mp4: particles streaming through the vector field
    },
    # package/ per run: frameless images, separate labels + metrics, the complete plate
    "package": {
        "enabled": True,
        "plan_height_px": 2400,   # size of every plan_* image (they all share one extent)
        "title": None,            # plate title (default: the site name)
        "index": None,            # small number next to the title, e.g. "03"
        "subtitle": None,         # e.g. "Canary Wharf, London  ·  pedestrian movement"
        "date": None,             # default: today, DD.MM.YYYY
        "credit": None,           # imagery credit in the plate's footnote
        "plate": False,            # true: also the complete plate (<site>_plate.png)
        "label_variants": ["light", "dark"],  # ink for dark / light backgrounds
    },
    "debug_video": {
        "enabled": False,
        "blur_people": True,
        "blur_plates": True,      # the lower part of every near vehicle (number plates)
        "trail_s": None,          # null = trails stay for the whole clip; e.g. 3 = last 3 s
        "style": "clean",         # clean: trails + dots, cleaned tracks only | boxes: debug
        "boxes": True,            # clean style: a thin box per object (people blue,
                                  # cyclists green, vehicles red)
        "stabilize_output": True, # align the output video to the calibration frame
        "dim": 0.75,              # darken the video under the trails (as the first videos)
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if k == "groups" and isinstance(v, dict):
            # groups are replaced wholesale per group name (so a site can drop groups),
            # but missing keys inside a group fall back to that group's defaults
            out[k] = {}
            for gname, gcfg in v.items():
                out[k][gname] = deep_merge(base["groups"].get(gname, {}), gcfg or {})
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


@dataclass
class Group:
    name: str
    classes: list[str]
    speed_range: tuple[float, float]
    max_speed: float
    smooth_window_s: float
    ground_offset_m: float | dict
    offset_mode: str = "travel"
    vehicle_width_m: float = 1.8
    jump_tolerance_m: float = 1.0
    stitch_radius_m: float = 1.5
    min_displacement_m: float = 0.0
    # objects that move on a raised surface (an elevated railway, a bridge deck): project
    # them onto that plane instead of the ground. A number (m above the calibrated ground) or
    # "auto": fitted so the tracks land on the OpenStreetMap rail lines (trains)
    plane_height_m: float | str = 0.0

    def offset_for(self, cls: str) -> float:
        off = self.ground_offset_m
        if isinstance(off, dict):
            return float(off.get(cls, off.get("default", 0.0)))
        return float(off or 0.0)


class Config:
    def __init__(self, data: dict, base_dir: Path):
        self.data = data
        self.base_dir = base_dir
        self.groups: dict[str, Group] = {}
        for name, g in data["groups"].items():
            self.groups[name] = Group(
                name=name,
                classes=list(g["classes"]),
                speed_range=(float(g["speed_range"][0]), float(g["speed_range"][1])),
                max_speed=float(g["max_speed"]),
                smooth_window_s=float(g["smooth_window_s"]),
                ground_offset_m=g.get("ground_offset_m", 0.0),
                offset_mode=g.get("offset_mode", "travel"),
                vehicle_width_m=float(g.get("vehicle_width_m", 1.8)),
                jump_tolerance_m=float(g.get("jump_tolerance_m", 1.0)),
                stitch_radius_m=float(g.get("stitch_radius_m", 1.5)),
                min_displacement_m=float(g.get("min_displacement_m", 0.0)),
                plane_height_m=g.get("plane_height_m", 0.0) or 0.0,
            )
        self.class_to_group: dict[str, str] = {}
        for g in self.groups.values():
            for c in g.classes:
                if c not in COCO_IDS:
                    raise ValueError(f"Unknown class {c!r}. Known: {sorted(COCO_IDS)}")
                if c in self.class_to_group:
                    raise ValueError(f"Class {c!r} is in two groups")
                self.class_to_group[c] = g.name
            if g.offset_mode not in ("travel", "view"):
                raise ValueError(f"groups.{g.name}.offset_mode must be 'travel' or 'view'")

    # convenience accessors -------------------------------------------------
    def __getitem__(self, key):
        return self.data[key]

    def get(self, key, default=None):
        return self.data.get(key, default)

    @property
    def site(self) -> str:
        return str(self.data["site"])

    @property
    def classes(self) -> list[str]:
        return list(self.class_to_group)

    @property
    def class_ids(self) -> list[int]:
        return [COCO_IDS[c] for c in self.classes]

    def group_of(self, cls: str) -> Group | None:
        g = self.class_to_group.get(cls)
        return self.groups[g] if g else None

    def model_path(self) -> str:
        """YOLO weights: paths resolve relative to the config; bare names go to models_dir."""
        m = str(self.data["detection"]["model"])
        if "/" in m or "\\" in m:
            return str(resolve_path(m, self.base_dir))
        md = self.path("models_dir")
        if md is None:
            return m
        md.mkdir(parents=True, exist_ok=True)
        return str(md / m)

    def path(self, key: str) -> Path | None:
        """Resolve a path-valued top-level key relative to the YAML file."""
        v = self.data.get(key)
        return resolve_path(v, self.base_dir)


def resolve_path(v, base_dir: Path) -> Path | None:
    if v in (None, ""):
        return None
    p = Path(v).expanduser()
    return p if p.is_absolute() else (base_dir / p).resolve()


def _read_yaml_chain(path: Path, seen=()) -> dict:
    """Read a site YAML; `extends: other.yaml` inherits everything from that file first.

    Use it for several clips of one site: a shared base (classes, groups, cleaning, ...)
    and one tiny file per camera position (video, homography, roi).
    """
    if path in seen:
        raise ValueError(f"circular `extends` at {path}")
    with open(path, encoding="utf-8") as f:
        user = yaml.safe_load(f) or {}
    parent = user.pop("extends", None)
    if not parent:
        return user
    ppath = Path(parent)
    ppath = (ppath if ppath.is_absolute() else path.parent / ppath).resolve()
    base = _read_yaml_chain(ppath, (*seen, path))
    if ppath.parent != path.parent:
        raise ValueError("`extends` must point to a config in the same folder (relative paths)")
    return _merge_user(base, user)


def _merge_user(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_user(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path, overrides: dict | None = None) -> Config:
    path = Path(path).resolve()
    user = _read_yaml_chain(path)
    data = deep_merge(DEFAULTS, user)
    if overrides:
        data = deep_merge(data, overrides)
    return Config(data, path.parent)


def default_config() -> Config:
    return Config(copy.deepcopy(DEFAULTS), Path.cwd())
