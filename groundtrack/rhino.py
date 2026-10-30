"""Rhino 8 / Grasshopper: the tracks and vector fields as Rhino geometry.

    rhino/<site>.3dm              open in Rhino (or reference in Grasshopper): one layer tree
                                  "groundtrack" with, per group,
                                    tracks_<group>          one polyline per track, coloured by
                                                            its mean speed
                                    speed_segments_<group>  the same tracks cut wherever the
                                                            speed colour changes (hidden)
                                    samples_<group>         every sample as a point cloud in the
                                                            speed colours (hidden)
                                    field_<group>           vector-field arrows (well-supported
                                                            cells; length ~ speed)
                                  plus cameras (and, for composites, each clip's view outline).
                                  Every object carries its numbers as attribute user text.
    rhino/grasshopper_import.py   paste into a Grasshopper "Python 3 Script" component: reads
                                  the same CSVs as the Houdini scripts and outputs tracks,
                                  moving points with velocity vectors (time slider) and the
                                  vector field, live.

Coordinates: metres from ORIGIN (British National Grid E, N), Z up, heights relative to Z0
(the terrain datum where a terrain model is used, else 0). Add ORIGIN back for EPSG:27700.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .colors import ramp_stops
from .config import Config

SEGMENT_BINS = 12              # speed classes for the speed_segments layers
FIELD_MIN_CONFIDENCE = 0.5     # arrows only where the field is well supported ...
FIELD_MIN_DOMINANCE = 0.6      # ... and one direction dominates (as in the plan images)


def _ranges(cfg: Config) -> dict:
    r = {g.name: tuple(float(v) for v in g.speed_range) for g in cfg.groups.values()}
    r["all"] = max(r.values(), key=lambda x: x[1]) if r else (0.0, 1.0)
    return r


def _rgb(speed, lo, hi) -> tuple[int, int, int]:
    from .portfolio import speed_rgb

    c = speed_rgb(float(speed), lo, hi)
    return int(c[0]), int(c[1]), int(c[2])


def _cell_heights(points: pd.DataFrame, cell: float, z0: float) -> dict:
    """(i, j) -> mean ground height above Z0 of the samples in that field cell."""
    if "z" not in points or points["z"].isna().all():
        return {}
    p = points.dropna(subset=["z"])
    i = np.floor(p["x"].to_numpy() / cell).astype(int)
    j = np.floor(p["y"].to_numpy() / cell).astype(int)
    s = pd.Series(p["z"].to_numpy() - z0).groupby([i, j]).mean()
    return {k: float(v) for k, v in s.items()}


def _field_z(field: pd.DataFrame, heights: dict) -> np.ndarray:
    """Height of each field cell: the samples in it, else the nearest cell that has some."""
    if not heights:
        return np.zeros(len(field))
    from scipy.spatial import cKDTree

    keys = np.array(list(heights))
    vals = np.array(list(heights.values()))
    _, k = cKDTree(keys).query(field[["i", "j"]].to_numpy())
    return vals[k]


# --------------------------------------------------------------------------- .3dm
def write_3dm(path: Path, points: pd.DataFrame, fields: dict, origin, z0: float | None,
              cfg: Config, cell: float, cameras=(), outlines=(), title: str = "",
              by_camera: dict | None = None) -> Path:
    """points: points.csv rows (x, y[, z], speed, group, track_id, ...); fields: group ->
    vector-field DataFrame (smooth_field columns); cameras: [(name, E, N, height above Z0)];
    outlines: [(name, [(E, N), ...])] closed view outlines. by_camera: {points' camera: layer
    name}, the tracks layered per camera, its groups under it (a Live layers export only)."""
    import rhino3dm as r3

    ox, oy = float(origin[0]), float(origin[1])
    z0 = float(z0 or 0.0)
    ranges = _ranges(cfg)
    m = r3.File3dm()
    m.Settings.ModelUnitSystem = r3.UnitSystem.Meters

    def layer(name, colour=(30, 30, 30), parent=None, visible=True, **user):
        L = r3.Layer()
        L.Name = name
        L.Color = (*colour, 255)
        L.Visible = visible
        if parent is not None:
            L.ParentLayerId = parent
        for k, v in user.items():
            L.SetUserString(k, str(v))
        return m.Layers[m.Layers.Add(L)]

    root = layer("groundtrack", origin_E=ox, origin_N=oy, origin_Z=z0, epsg=27700,
                 units="metres", title=title)

    def attrs(lay, colour=None, name="", **user):
        a = r3.ObjectAttributes()
        a.LayerIndex = lay.Index
        if colour is not None:
            a.ColorSource = r3.ObjectColorSource.ColorFromObject
            a.ObjectColor = (*colour, 255)
        a.Name = name
        for k, v in user.items():
            a.SetUserString(k, str(v))
        return a

    has_z = "z" in points and points["z"].notna().any()
    p = points.copy()
    p["lx"] = p["x"] - ox
    p["ly"] = p["y"] - oy
    p["lz"] = (p["z"].fillna(z0) - z0) if has_z else 0.0
    p = p.sort_values(["track_id", "time_s"])

    def groups(p, parent):
        for gname in sorted(p["group"].unique()):
            lo, hi = ranges.get(gname, ranges["all"])
            gl = layer(gname, _rgb(0.5 * (lo + hi), lo, hi), parent.Id)
            lt = layer(f"tracks_{gname}", (60, 60, 60), gl.Id)
            ls = layer(f"speed_segments_{gname}", (60, 60, 60), gl.Id, visible=False)
            lp = layer(f"samples_{gname}", (60, 60, 60), gl.Id, visible=False)
            g = p[p["group"] == gname]
            cloud = r3.PointCloud()
            for tid, t in g.groupby("track_id"):
                xyz = t[["lx", "ly", "lz"]].to_numpy(float)
                sp = t["speed"].to_numpy(float)
                for q, s in zip(xyz, sp):
                    cloud.Add(r3.Point3d(*q), (*_rgb(s, lo, hi), 255))
                if len(t) < 2:
                    continue
                pl = r3.Polyline()
                for q in xyz:
                    pl.Add(*q)
                ms = float(np.mean(sp))
                info = dict(track_id=int(tid), group=gname, cls=t["class"].iloc[0],
                            mean_speed=round(ms, 3), duration_s=round(float(np.ptp(t["time_s"])), 2))
                if "clip" in t:
                    info["clip"] = t["clip"].iloc[0]
                m.Objects.AddCurve(r3.PolylineCurve(pl), attrs(lt, _rgb(ms, lo, hi),
                                                               f"track {int(tid)}", **info))
                # speed segments: cut where the speed class changes (shared end points)
                b = np.clip(((sp - lo) / max(hi - lo, 1e-9) * SEGMENT_BINS).astype(int), 0,
                            SEGMENT_BINS - 1)
                start = 0
                for k in range(1, len(b) + 1):
                    if k == len(b) or b[k] != b[start]:
                        seg = xyz[start:min(k + 1, len(xyz))]
                        if len(seg) >= 2:
                            sl = r3.Polyline()
                            for q in seg:
                                sl.Add(*q)
                            s_mid = lo + (b[start] + 0.5) / SEGMENT_BINS * (hi - lo)
                            m.Objects.AddCurve(r3.PolylineCurve(sl), attrs(
                                ls, _rgb(s_mid, lo, hi), "", track_id=int(tid),
                                speed=round(float(sp[start:k].mean()), 3)))
                        start = k
            m.Objects.AddPointCloud(cloud, attrs(lp, None, f"samples {gname}"))

    if by_camera is None:
        groups(p, root)
    else:                                # one parent layer per camera, its groups under it
        for cam, label in by_camera.items():
            if (p["camera"] == cam).any():
                groups(p[p["camera"] == cam], layer(str(label), (30, 30, 30), root.Id, camera=cam))

    # vector-field arrows, centred on their cell, as long as the cell at the top speed
    heights = _cell_heights(points, cell, z0) if has_z else {}
    for fname, fld in fields.items():
        if fld is None or fld.empty:
            continue
        lo, hi = ranges.get(fname, ranges["all"])
        lf = layer(f"field_{fname}", (60, 60, 60), root.Id)
        ok = fld["confidence"] >= FIELD_MIN_CONFIDENCE
        if "dominance" in fld:
            ok &= fld["dominance"] >= FIELD_MIN_DOMINANCE
        f = fld[ok]
        zc = _field_z(f, heights)
        for (_, rw), zz in zip(f.iterrows(), zc):
            sp = float(rw["speed"])
            if sp <= 1e-6:
                continue
            d = np.array([rw["vx"], rw["vy"]]) / sp
            half = 0.45 * cell * float(np.clip(sp / max(hi * 0.6, 1e-9), 0.25, 1.0))
            cx, cy = float(rw["cell_x"]) - ox, float(rw["cell_y"]) - oy
            line = r3.LineCurve(r3.Point3d(cx - d[0] * half, cy - d[1] * half, zz),
                                r3.Point3d(cx + d[0] * half, cy + d[1] * half, zz))
            a = attrs(lf, _rgb(sp, lo, hi), "", vx=round(float(rw["vx"]), 3),
                      vy=round(float(rw["vy"]), 3), speed=round(sp, 3),
                      confidence=round(float(rw["confidence"]), 3))
            a.ObjectDecoration = r3.ObjectDecoration.EndArrowhead
            m.Objects.AddCurve(line, a)

    if len(cameras) or len(outlines):
        lc = layer("cameras", (20, 20, 20), root.Id)
        for name, e, n, h in cameras:
            m.Objects.AddPoint(r3.Point3d(e - ox, n - oy, h), attrs(lc, None, str(name),
                                                                     clip=name))
        for name, ring in outlines:
            pl = r3.Polyline()
            for e, n in list(ring) + [ring[0]]:
                pl.Add(e - ox, n - oy, 0.0)
            m.Objects.AddCurve(r3.PolylineCurve(pl), attrs(lc, None, f"view {name}",
                                                           clip=name))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m.Write(str(path), 8)
    return path


# --------------------------------------------------------------------------- Grasshopper
GRASSHOPPER_TEMPLATE = r'''"""groundtrack -> Grasshopper (Rhino 8): tracks, moving points and the vector field.

1. Grasshopper: add a "Python 3 Script" component (Maths > Script). Double-click it, paste
   this whole file, close the editor.
2. Zoom in on the component and use (+) / right-click to add OUTPUTS named exactly like these
   (only the ones you need):
     tracks        one polyline per track (the whole track, or the last `tail` s up to t)
     track_colour  the colour of each track's mean speed (Custom Preview / Geometry Pipeline)
     track_info    "track_id group class mean_speed clip" per track
     points        every sample (or, with t, where each object is at time t)
     vectors       velocity at each point in m/s (Vector Display, Kangaroo, Anemone ...)
     speeds        speed at each point (m/s)
     colours       speed colour of each point
     field_points  vector-field cell centres
     field_vectors field velocity per cell (m/s): a flow field for particle sims
     field_speed   speed per cell (m/s); field_confidence: how well supported (0..1)
     field_colour  speed colour per cell
3. Optional INPUTS (add the same way, set Type hint to float / str):
     t       time in seconds (a Number Slider from 0 to DURATION), empty = all at once
     tail    seconds of path to keep behind t (empty = from the start)
     group   only this group ("people", "vehicles", ...), empty = all
     field   which vector field: "all" or a group name (see FIELD_FILES)
     min_confidence   hide field cells below this (default 0.5)
     step    use every n-th sample for `points` (default: thin to <= 60 000 points)

Units: metres, scaled to the document's units automatically. Coordinates are relative to
ORIGIN (British National Grid), Z up (height above Z0 on terrain, else 0). In a composite
every clip starts at t = 0: the slider plays all clips at once.
"""
import bisect
import csv
import math
import os

import Rhino
import Rhino.Geometry as rg
import scriptcontext as sc
import System.Drawing as sd

POINTS_CSV = __POINTS_CSV__
FIELD_FILES = __FIELD_FILES__
ORIGIN = __ORIGIN__
Z0 = __Z0__
CELL = __CELL__
SPEED_RANGES = __SPEED_RANGES__
RAMP = __RAMP__
DURATION = __DURATION__
MAX_POINTS = 60000


def _in(name, default):
    v = globals().get(name)
    return default if v is None or v == "" else v


try:
    UNIT = Rhino.RhinoMath.UnitScale(Rhino.UnitSystem.Meters,
                                     Rhino.RhinoDoc.ActiveDoc.ModelUnitSystem)
except Exception:
    UNIT = 1.0


def _colour(speed, group):
    lo, hi = SPEED_RANGES.get(group, SPEED_RANGES.get("all", (0.0, 1.0)))
    x = min(max((speed - lo) / max(hi - lo, 1e-9), 0.0), 1.0)
    for (p0, c0), (p1, c1) in zip(RAMP[:-1], RAMP[1:]):
        if x <= p1:
            f = (x - p0) / max(p1 - p0, 1e-9)
            c = [a + (b - a) * f for a, b in zip(c0, c1)]
            break
    else:
        c = list(RAMP[-1][1])
    return sd.Color.FromArgb(int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))


def _cached(path, parse):
    key = "groundtrack|" + path
    mt = os.path.getmtime(path)
    hit = sc.sticky.get(key)
    if hit is not None and hit[0] == mt:
        return hit[1]
    data = parse(path)
    sc.sticky[key] = (mt, data)
    return data


def _parse_points(path):
    ox, oy = ORIGIN
    tracks, zcell = {}, {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            tid = int(r["track_id"])
            x, y = float(r["x"]), float(r["y"])
            z = float(r["z"]) - Z0 if r.get("z") not in (None, "") else 0.0
            d = tracks.get(tid)
            if d is None:
                d = tracks[tid] = {"group": r.get("group", ""), "class": r.get("class", ""),
                                   "clip": r.get("clip", ""), "s": []}
            d["s"].append((float(r["time_s"]), x - ox, y - oy, z, float(r["vx"]),
                           float(r["vy"]), float(r["speed"])))
            if r.get("z") not in (None, ""):
                k = (int(math.floor(x / CELL)), int(math.floor(y / CELL)))
                a = zcell.get(k, (0.0, 0))
                zcell[k] = (a[0] + z, a[1] + 1)
    for d in tracks.values():
        d["s"].sort()
        d["t"] = [q[0] for q in d["s"]]
    return tracks, {k: v[0] / v[1] for k, v in zcell.items()}


def _parse_field(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _ground(i, j):
    """Height of a field cell: its samples, else the nearest cell with samples (6 cells)."""
    if not ZCELL:
        return 0.0
    for r in range(7):
        ring = [ZCELL[(i + a, j + b)] for a in range(-r, r + 1) for b in range(-r, r + 1)
                if max(abs(a), abs(b)) == r and (i + a, j + b) in ZCELL]
        if ring:
            return sum(ring) / len(ring)
    return ZMID


TRACKS, ZCELL = _cached(POINTS_CSV, _parse_points)
ZMID = sorted(ZCELL.values())[len(ZCELL) // 2] if ZCELL else 0.0
t = _in("t", None)
tail = _in("tail", None)
grp = str(_in("group", ""))
n_all = sum(len(d["s"]) for d in TRACKS.values())
step = int(_in("step", max(1, int(math.ceil(n_all / float(MAX_POINTS))))))

tracks, track_colour, track_info = [], [], []
points, vectors, speeds, colours = [], [], [], []
U = UNIT
for tid in sorted(TRACKS):
    d = TRACKS[tid]
    if grp and d["group"] != grp:
        continue
    s = d["s"]
    if t is None:
        sel = s
    else:
        a = bisect.bisect_left(d["t"], float(t) - float(tail)) if tail is not None else 0
        sel = s[a:bisect.bisect_right(d["t"], float(t))]
    if len(sel) >= 2:
        pl = rg.Polyline()        # (RhinoCommon takes no Python list here: add point by point)
        for q in sel:
            pl.Add(q[1] * U, q[2] * U, q[3] * U)
        tracks.append(rg.PolylineCurve(pl))
        ms = sum(q[6] for q in sel) / len(sel)
        track_colour.append(_colour(ms, d["group"]))
        track_info.append("%d %s %s %.2f %s" % (tid, d["group"], d["class"], ms, d["clip"]))
    if t is None:
        now = s[::step]
    elif s[0][0] <= float(t) <= s[-1][0]:
        k = min(bisect.bisect_left(d["t"], float(t)), len(s) - 1)
        now = [s[k]]
    else:
        now = []
    for q in now:
        points.append(rg.Point3d(q[1] * U, q[2] * U, q[3] * U))
        vectors.append(rg.Vector3d(q[4] * U, q[5] * U, 0.0))
        speeds.append(q[6])
        colours.append(_colour(q[6], d["group"]))

DEFAULT_FIELD = "all" if "all" in FIELD_FILES else next(iter(FIELD_FILES), "")
fname = str(_in("field", DEFAULT_FIELD))
if fname not in FIELD_FILES:      # not exported for this run: use the default field
    fname = DEFAULT_FIELD
field_points, field_vectors, field_speed, field_confidence, field_colour = [], [], [], [], []
if fname in FIELD_FILES and os.path.exists(FIELD_FILES[fname]):
    mc = float(_in("min_confidence", 0.5))
    ox, oy = ORIGIN
    for r in _cached(FIELD_FILES[fname], _parse_field):
        c = float(r["confidence"])
        if c < mc:
            continue
        cx, cy = float(r["cell_x"]), float(r["cell_y"])
        z = _ground(int(math.floor(cx / CELL)), int(math.floor(cy / CELL)))
        field_points.append(rg.Point3d((cx - ox) * U, (cy - oy) * U, z * U))
        field_vectors.append(rg.Vector3d(float(r["vx"]) * U, float(r["vy"]) * U, 0.0))
        field_speed.append(float(r["speed"]))
        field_confidence.append(c)
        field_colour.append(_colour(float(r["speed"]), fname))

a = ("%d tracks, %d points, %d field cells | origin E %.1f N %.1f, Z0 %.2f m | "
     "duration %.1f s | units x%g" % (len(tracks), len(points), len(field_points), ORIGIN[0],
                                       ORIGIN[1], Z0, DURATION, UNIT))
'''


def write_grasshopper_script(path: Path, points_csv: Path, field_files: dict, origin,
                             z0: float | None, cfg: Config, cell: float, duration_s: float) -> Path:
    stops = [(p, tuple(c)) for p, c in ramp_stops()]
    txt = (GRASSHOPPER_TEMPLATE
           .replace("__POINTS_CSV__", repr(Path(points_csv).resolve().as_posix()))
           .replace("__FIELD_FILES__", repr({k: Path(v).resolve().as_posix()
                                             for k, v in field_files.items()}))
           .replace("__ORIGIN__", repr((float(origin[0]), float(origin[1]))))
           .replace("__Z0__", repr(round(float(z0 or 0.0), 2)))
           .replace("__CELL__", repr(float(cell)))
           .replace("__SPEED_RANGES__", repr(_ranges(cfg)))
           .replace("__RAMP__", repr(stops))
           .replace("__DURATION__", repr(round(float(duration_s), 2))))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(txt, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- a run
def write_rhino(run_dir: Path, cfg: Config, origin, z0: float | None, log=print,
                title: str = "", cameras=(), outlines=(), by_camera: dict | None = None) -> dict:
    """grasshopper_import.py (+ <site>.3dm when rhino3dm is installed) into run/rhino/."""
    from .layout import open_run

    run = open_run(run_dir)
    run.rhino.mkdir(parents=True, exist_ok=True)
    points = pd.read_csv(run.points)
    cell = float(cfg["field"]["cell_size_m"])
    files = {("all" if f.stem == "vector_field" else f.stem.replace("vector_field_", "")): f
             for f in sorted(run.houdini.glob("vector_field*.csv"))
             if not f.stem.startswith("vector_field_slices")
             and f.stem != "vector_field_master"}       # a Live pass's copy of "all" (live_pass_export.py)
    dur = float(points["time_s"].max() - points["time_s"].min()) if len(points) else 0.0
    out = {"grasshopper": write_grasshopper_script(run.rhino / "grasshopper_import.py",
                                                   run.points, files, origin, z0, cfg, cell, dur)}
    try:
        import rhino3dm  # noqa: F401
    except ImportError:
        log("rhino: rhino3dm not installed (pip install rhino3dm), .3dm skipped")
        return out
    fields = {k: pd.read_csv(f) for k, f in files.items()}
    if len(fields) > 1:
        fields.pop("all", None)          # one field per group is enough in the .3dm
    out["3dm"] = write_3dm(run.rhino / f"{cfg.site}.3dm", points, fields, origin, z0, cfg,
                           cell, cameras, outlines, title or cfg.site, by_camera=by_camera)
    log(f"rhino: {out['3dm'].name} + grasshopper_import.py -> {run.rhino}")
    return out
