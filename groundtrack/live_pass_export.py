"""A saved pass as a run (plans/live_controller/05, "Export"): what the machine drew in one pass,
in the files a processed run has, so `groundtrack rhino` and `groundtrack houdini-pop` take it as
they take any run.

    <start run>/live/passes/<id>/pass_<id>/          (the pass's own folder; nothing else written)
      data/points.csv        every drawn point (dropped and not drawn left out), BNG metres:
                             the run columns (track_id: one per camera, footage track and piece;
                             x, y, z absolute; vx, vy, speed, heading as drawn along the piece;
                             source from the kind: detected / predicted / stitched / interpolated)
                             + camera, source_track, part, kind, cause
      data/track_metrics.csv one row per drawn piece
      data/tracks.geojson, data/predicted_gaps.geojson     EPSG:27700 (a realtime recording: local metres)
      data/camera_path.csv   every recorded camera row: where each camera stood and looked
      houdini/               points.csv, houdini_import.py (curves), vector_field*.csv +
                             houdini_field.py (the field of the drawn points), vector_field_master.csv
                             (the same "all" field, so the folder serves as --composite)
      rhino/                 <site>.3dm + grasshopper_import.py (--rhino)
      houdini_pop/           the POP kit (--houdini; built with hython unless --no-build)
      _working/              homography_used.json (the start camera's), config_used.yaml
      meta.json              the pass, its cameras, the counts

The collage of the Live view (08 sandbox: each camera a layer, built from one or more plays), all of
it or one camera's layer (05, 29 Oct (3)), the same way:

    <run of the first layer>/live/exports/<YYYYMMDD-HHMMSS>-<collage | camera id>/
      layers.json            the layers used, oldest first per camera: camera, its package, the pass
                             that drew it (run + id; none: not saved), key, kind, stacked, up_to, file;
                             the counts once exported
      layers/<k>_<camera id>.bin   each layer's drawing as the view showed it (a layer can be part of
                             a pass; live_passes LAYOUT)
      data/, houdini/, rhino/, houdini_pop/, _working/   as above; points.csv + layer, pass; a
                             track_id per layer, footage track and piece; camera_path.csv + layer
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .live_passes import (
    BYTES_PER_POINT,
    HELD,
    LAYOUT,
    LEVELLING,
    PERFORMED,
    SOURCES,
    _folder,
    _rel,
    read_pass,
    read_recording,
)

KINDS = ("detected", "predicted", "stitched", "interpolated")      # rules.js kinds 0-3 -> source
KIND_NAMES = ("solid", "predicted", "stitched", "interpolated", "dropped")
CAUSES = ("none", "footage", "live")
DROPPED = 4
SPEED_WIN = 7
MIN_HEADING_SPEED = 0.2      # m/s: the heading is held below it (trajectories.heading_deg)


def export_dir(run: Path, pass_id: str) -> Path:
    return _folder(Path(run), pass_id) / f"pass_{pass_id.replace('-', '_')}"


def _project_root(run: Path) -> Path:
    """runs/<site>/<run> -> the project (camera runs in pass.json are relative to it)."""
    run = Path(run).resolve()
    return run.parent.parent.parent


def read_drawing(path: Path, n: int) -> dict:
    """A pass .bin (live_passes LAYOUT) -> X (n x 3), part, kind, cause."""
    b = Path(path).read_bytes()
    if len(b) != BYTES_PER_POINT * n:
        raise ValueError(f"{Path(path).name}: {len(b)} bytes for {n} points")
    return {"X": np.frombuffer(b, "<f4", 3 * n, 0).reshape(n, 3).astype(float),
            "part": np.frombuffer(b, "<u2", n, 12 * n), "kind": np.frombuffer(b, "u1", n, 14 * n),
            "cause": np.frombuffer(b, "u1", n, 15 * n)}


def _package(run_dir: Path) -> tuple[dict, np.ndarray, dict]:
    from .live_export import TRACK_DTYPE

    live = Path(run_dir) / "live"
    scene = json.loads((live / "scene.json").read_text(encoding="utf-8"))
    rec = np.fromfile(live / "tracks.bin", TRACK_DTYPE)
    index = json.loads((live / "tracks_index.json").read_text(encoding="utf-8"))
    return scene, rec, index


def _velocity(df: pd.DataFrame) -> pd.DataFrame:
    """vx, vy, speed (m/s) and heading (deg, clockwise from grid north) along each piece, as drawn:
    positions smoothed (rolling mean over SPEED_WIN rows, centred), central differences."""
    vx = np.zeros(len(df))
    vy = np.zeros(len(df))
    for rows in df.groupby("track_id", sort=False).indices.values():
        if len(rows) < 2:
            continue
        xy = df[["x", "y"]].to_numpy(float)[rows]
        s = pd.DataFrame(xy).rolling(SPEED_WIN, center=True, min_periods=1).mean().to_numpy()
        dt = np.gradient(df["time_s"].to_numpy(float)[rows])
        with np.errstate(divide="ignore", invalid="ignore"):
            vx[rows] = np.nan_to_num(np.gradient(s[:, 0]) / np.where(dt > 0, dt, np.nan))
            vy[rows] = np.nan_to_num(np.gradient(s[:, 1]) / np.where(dt > 0, dt, np.nan))
    out = df.copy()
    out["vx"], out["vy"] = vx.round(3), vy.round(3)
    out["speed"] = np.hypot(vx, vy).round(3)
    hd = (np.degrees(np.arctan2(vx, vy)) + 360.0) % 360.0
    hd = np.where(np.hypot(vx, vy) >= MIN_HEADING_SPEED, hd, np.nan)
    out["heading"] = (pd.Series(hd, index=out.index).groupby(out["track_id"]).transform(
        lambda s: s.ffill().bfill()).fillna(0.0).round(1))
    return out


POINT_COLUMNS = ["track_id", "class", "frame", "time_s", "x", "y", "vx", "vy", "speed", "heading", "predicted",
                 "source", "group", "beyond_limit", "z", "camera", "source_track", "part", "kind", "cause"]


def _origin(scene: dict) -> tuple[float, float, float]:
    o = scene.get("origin") or {}
    return float(o.get("E", 0)), float(o.get("N", 0)), float(o.get("datum_z") or 0)


def _drawn(name: str, scene: dict, rec: np.ndarray, index: dict, d: dict, next_id: int) -> tuple[pd.DataFrame | None, int]:
    """One camera's drawing (read_drawing) -> its drawn points (dropped and not drawn left out) in BNG
    metres, one track_id per footage track and piece from next_id; and the next free id."""
    E0, N0, Z0 = _origin(scene)
    drawn = (d["kind"] < DROPPED) & np.isfinite(d["X"]).all(axis=1)
    if not drawn.any():
        return None, next_id
    r = rec[drawn]
    classes = {int(k): v for k, v in ((scene.get("tracks") or {}).get("classes") or {}).items()}
    df = pd.DataFrame({
        "source_track": r["track_id"].astype(int), "part": d["part"][drawn].astype(int),
        "frame": r["frame"].astype(int), "time_s": r["t"].astype(float).round(4),
        "x": (d["X"][drawn, 0] + E0).round(3), "y": (d["X"][drawn, 1] + N0).round(3),
        "z": (d["X"][drawn, 2] + Z0).round(2),
        "class": [classes.get(int(k), "person") for k in r["class"]],
        "kind": [KIND_NAMES[k] for k in d["kind"][drawn]],
        "cause": [CAUSES[k] if k < len(CAUSES) else str(k) for k in d["cause"][drawn]],
    })
    df["source"] = [KINDS[k] for k in d["kind"][drawn]]
    df["group"] = [str((index.get(str(t)) or {}).get("group") or "") for t in df["source_track"]]
    df["camera"] = name
    df = df.sort_values(["source_track", "part", "frame"], kind="stable").reset_index(drop=True)
    key = df["source_track"].astype(np.int64) * 65536 + df["part"]
    ids = {k: next_id + i for i, k in enumerate(pd.unique(key))}
    df["track_id"] = key.map(ids).astype(int)
    return _velocity(df), next_id + len(ids)


def _points(frames: list, cols: list) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=cols)
    pts = pd.concat(frames, ignore_index=True)
    pts["predicted"] = pts["source"] != "detected"
    pts["beyond_limit"] = False
    return pts[cols]


def pass_points(run: Path, pass_id: str, root: Path | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """The pass's drawn points in BNG metres (the run columns + camera, source_track, part, kind,
    cause), and per camera {name, run, origin, scene, points drawn}."""
    run = Path(run)
    m = read_pass(run, pass_id)
    root = Path(root) if root else _project_root(run)
    folder = _folder(run, pass_id)
    frames, cams = [], []
    next_id = 1
    for c in m.get("cameras") or []:
        crun = Path(c["run"]) if Path(c["run"]).is_absolute() else root / c["run"]
        scene, rec, index = _package(crun)
        n = int(c["points"])
        if len(rec) != n:
            raise ValueError(f"{c['name']}: its package has {len(rec)} points, the pass {n} (re-exported since)")
        df, next_id = _drawn(c["name"], scene, rec, index, read_drawing(folder / c["file"], n), next_id)
        cams.append({"name": c["name"], "run": crun, "origin": _origin(scene), "scene": scene,
                     "drawn": 0 if df is None else len(df), "points": n})
        if df is not None:
            frames.append(df)
    return _points(frames, POINT_COLUMNS), cams


def track_metrics(points: pd.DataFrame) -> pd.DataFrame:
    """One row per drawn piece: camera, footage track, part, class, group, length, time, speed."""
    rows = []
    for tid, g in points.groupby("track_id", sort=True):
        xy = g[["x", "y"]].to_numpy(float)
        rows.append({"track_id": int(tid), "camera": g["camera"].iloc[0],
                     "source_track": int(g["source_track"].iloc[0]), "part": int(g["part"].iloc[0]),
                     "class": g["class"].iloc[0], "group": g["group"].iloc[0], "n_points": len(g),
                     "start_s": float(g["time_s"].min()), "duration_s": float(np.ptp(g["time_s"])),
                     "length_m": round(float(np.hypot(*np.diff(xy, axis=0).T).sum()) if len(g) > 1 else 0.0, 2),
                     "mean_speed": round(float(g["speed"].mean()), 3),
                     "predicted_share": round(float(g["predicted"].mean()), 3)})
    return pd.DataFrame(rows, columns=["track_id", "camera", "source_track", "part", "class", "group", "n_points",
                                       "start_s", "duration_s", "length_m", "mean_speed", "predicted_share"])


PATH_COLUMNS = ["k", "t", "T", "camera", "player", "source", "frame", "E", "N", "Z", "heading", "tilt",
                "performed", "held", "levelling"]


def _path_rows(rec: dict, origin: dict, keep=None) -> list[pd.DataFrame]:
    """A recording's camera rows (kept to the cameras in keep, when given) in BNG metres."""
    clock, out = rec["clock"], []
    for name, rows in rec["cameras"].items():
        if not len(rows) or (keep is not None and name not in keep):
            continue
        E0, N0, Z0 = origin.get(name, (0.0, 0.0, 0.0))
        k = rows["k"].astype(int)
        f = rows["R_wc"][:, :, 2]                               # forward (OpenCV z)
        out.append(pd.DataFrame({
            "k": k, "t": clock["t"][np.minimum(k, len(clock) - 1)].round(4) if len(clock) else 0.0,
            "T": clock["T"][np.minimum(k, len(clock) - 1)].round(4) if len(clock) else 0.0,
            "camera": name, "player": rows["player"].astype(int),
            "source": [SOURCES[s] if s < len(SOURCES) else "other" for s in rows["src"]],
            "frame": rows["frame"].astype(int),
            "E": (rows["C"][:, 0] + E0).round(3), "N": (rows["C"][:, 1] + N0).round(3),
            "Z": (rows["C"][:, 2] + Z0).round(3),
            "heading": ((np.degrees(np.arctan2(f[:, 0], f[:, 1])) + 360) % 360).round(2),
            "tilt": np.degrees(np.arcsin(np.clip(-f[:, 2], -1, 1))).round(2),     # down: positive
            "performed": (rows["flags"] & PERFORMED) > 0, "held": (rows["flags"] & HELD) > 0,
            "levelling": (rows["flags"] & LEVELLING) > 0}))
    return out


def camera_path(run: Path, pass_id: str, cams: list[dict]) -> pd.DataFrame:
    """Every recorded camera row: k, t (the pass's clock), T (the footage clock), camera, player,
    source, frame, E, N, Z (BNG), heading / tilt (deg), performed, held, levelling."""
    rec = read_recording(run, pass_id)
    out = _path_rows(rec, {c["name"]: c["origin"] for c in cams}) if rec is not None else []
    return pd.concat(out, ignore_index=True).sort_values(["k", "camera"], kind="stable")[PATH_COLUMNS] if out         else pd.DataFrame(columns=PATH_COLUMNS)


def _config(run: Path):
    from .config import load_config
    from .live_export import find_site_config

    p = find_site_config(run)
    if p is None:
        used = Path(run) / "_working" / "config_used.yaml"
        if not used.exists():
            raise FileNotFoundError(f"no site config for {run} (sites/<site>.yaml)")
        p = used
    return load_config(p), p


def _cameras3d(cams: list[dict]) -> list:
    """(name, E, N, height) of each camera's own calibration, for the Rhino model."""
    from .homography import Homography

    out = []
    for c in {c["name"]: c for c in cams}.values():          # a camera with several layers: once
        hp = Homography.load(Path(c["run"]) / "_working" / "homography_used.json")
        cp = getattr(hp, "camera_params", None) or {}
        if "E" in cp:
            out.append((c["name"], cp["E"], cp["N"], cp.get("height_m", 0.0)))
    return out


def _open_out(out: Path, run: Path):
    """The export folder as a run: its folders, the start camera's homography, the site config."""
    from .homography import Homography, snapshot
    from .layout import open_run

    cfg, cfg_path = _config(run)
    pr = open_run(out).make()
    snapshot(run / "_working" / "homography_used.json", pr.homography_used)
    shutil.copy2(cfg_path, pr.config_used)
    return pr, cfg, cfg_path, Homography.load(pr.homography_used)


def _crs(h) -> str:
    """The exports' coordinates: the British National Grid, or "local metres" for a realtime
    recording (plans/live_controller/10: calibrated from the phone, flat, at (0, 0))."""
    return "local metres" if getattr(h, "crs", None) == "local" else "EPSG:27700"


def _write_out(pr, cfg, h, points: pd.DataFrame, cams: list[dict], path: pd.DataFrame) -> dict:
    """Points, metrics, GeoJSON, camera path, Houdini curves and field into the export folder."""
    from . import exports
    from .pipeline import _export_field

    exports.write_points_csv(points, pr.points)
    summary = track_metrics(points)
    summary.to_csv(pr.tracks, index=False)
    for gj, name in ((exports.tracks_geojson(points, summary), "tracks"),
                     (exports.predicted_segments_geojson(points), "predicted_gaps")):
        if _crs(h) != "EPSG:27700":
            gj.pop("crs", None)                  # a realtime recording's own metres: no CRS
        exports.write_geojson(gj, pr.data / f"{name}.geojson")
    path.to_csv(pr.data / "camera_path.csv", index=False)
    res = {"dir": str(pr.root), "points": len(points), "tracks": int(summary.shape[0]),
           "geojson": [str(pr.data / "tracks.geojson"), str(pr.data / "predicted_gaps.geojson")]}
    shutil.copy2(pr.points, pr.houdini / "points.csv")
    exports.write_houdini_script(pr.houdini / "houdini_import.py", pr.houdini / "points.csv", h.origin, cfg,
                                 z0=h.ground_z_m)
    res["houdini_curves"] = str(pr.houdini / "houdini_import.py")
    if len(points) >= 2:
        fps = float(cams[0]["scene"].get("fps") or 30.0) if cams else 30.0
        _export_field(cfg, pr, points, h, 1.0 / fps)
        shutil.copy2(pr.field_csv("all"), pr.houdini / "vector_field_master.csv")
        res["field"] = str(pr.field_csv("all"))
    return res


def _write_models(res: dict, pr, cfg, h, cams: list[dict], title: str, kit: str, rhino: bool, houdini: bool,
                  build: bool, log, by_camera: dict | None = None) -> dict:
    """The Rhino model (rhino) and the POP kit (houdini; its scene built with hython unless not build)."""
    if rhino:
        from .rhino import write_rhino

        r = write_rhino(pr.root, cfg, h.origin, h.ground_z_m, log, title=title, cameras=_cameras3d(cams),
                        by_camera=by_camera)
        res["rhino"] = {k: str(v) for k, v in r.items()}
    if houdini:
        if "field" not in res:
            raise ValueError("too few drawn points for a field (the POP kit needs one)")
        from .houdini_pop import build_scene, prepare_kit

        k = prepare_kit(kit, pr.root, [pr.root], pr.root / "houdini_pop", log=log)
        res["houdini_kit"] = str(k)
        if build:
            hip = build_scene(k, log=log)
            res["houdini_scene"] = str(hip) if hip else None
    return res


def export_pass(run: str | Path, pass_id: str, rhino: bool = False, houdini: bool = False,
                build: bool = True, log=print) -> dict:
    """Write the pass's folder (module docstring): points, metrics, GeoJSON, camera path, Houdini
    curves and field always; the Rhino model with rhino, the POP kit with houdini."""
    run = Path(run).resolve()
    m = read_pass(run, pass_id)
    out = export_dir(run, pass_id)
    pr, cfg, cfg_path, h = _open_out(out, run)
    points, cams = pass_points(run, pass_id)
    log(f"pass {pass_id}: {len(points)} drawn points, {points['track_id'].nunique() if len(points) else 0} "
        f"pieces, {len(cams)} cameras")
    res = _write_out(pr, cfg, h, points, cams, camera_path(run, pass_id, cams))
    meta = {"format": "motus-live-pass-export", "version": 1, "pass": pass_id, "run": str(run),
            "site": m.get("site"), "started": m.get("started"), "ended": m.get("ended"),
            "complete": bool(m.get("complete")), "config": str(cfg_path),
            "cameras": [{"name": c["name"], "run": str(c["run"]), "origin": list(c["origin"]),
                         "points": c["points"], "drawn": c["drawn"]} for c in cams],
            "points": len(points), "tracks": res["tracks"], "crs": _crs(h)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _write_models(res, pr, cfg, h, cams, f"{cfg.site} · pass {pass_id}",
                  f"{cfg.site}_pass_{pass_id.replace('-', '_')}", rhino, houdini, build, log)
    log(f"pass export -> {out}")
    return res


# --------------------------------------------------------------------------- the collage's layers
EXPORTS_DIR = "exports"          # <run of the first layer>/live/exports/<stamp>-<collage | camera id>/
LAYERS_FILE = "layers.json"
LAYERS_FORMAT = "motus-live-layers-export"
MAX_LAYERS = 512
LAYER_COLUMNS = [*POINT_COLUMNS, "layer", "pass"]


def safe_id(s) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(s or ""))[:40]


def camera_label(name: str, cid=None) -> str:
    """The Rhino layer of a camera: its id first ("North-east · 6376" -> "6376 · North-east")."""
    cid = str(cid or "")
    if not cid:
        return name
    spot = name.removesuffix(f" · {cid}")
    return f"{cid} · {spot}" if spot and spot != cid else cid


def new_exports_dir(run: Path, name: str) -> Path:
    """A new <run>/live/exports/<YYYYMMDD-HHMMSS>-<name>/ (not made yet)."""
    base = Path(run) / "live" / EXPORTS_DIR
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out, k = base / f"{stamp}-{name}", 2
    while out.exists():
        out, k = base / f"{stamp}-{k}-{name}", k + 1
    return out


def _abs(p: str | Path, root: Path) -> Path:
    return Path(p) if Path(p).is_absolute() else Path(root) / p


def prepare_layers(layers: list, files: dict, camera: str | None = None, site: str = "",
                   root: Path | None = None) -> Path:
    """The layers of the Live view's collage (08 sandbox; live.js collageLayers), each with its drawing
    as the view shows it (live_passes LAYOUT: a layer can be part of a pass), kept to `camera` when
    given: checked, then written into a new <run of the first layer>/live/exports/<stamp>-<collage |
    camera id>/ as layers/<k>_<camera id>.bin + layers.json. layers: [{camera, camera_id, package (the
    camera's package run), run (where its pass is saved), id (its pass; none: not saved), key, kind,
    stacked, up_to, file}], runs already resolved (desktop.py: inside runs/). Returns the folder."""
    from .live_export import TRACK_DTYPE

    if camera:
        layers = [ly for ly in layers if ly.get("camera") == camera]
    if not layers:
        raise ValueError("no layer to export" + (f" for {camera}" if camera else ""))
    if len(layers) > MAX_LAYERS:
        raise ValueError(f"more than {MAX_LAYERS} layers")
    run0 = Path(layers[0]["run"])
    root = (Path(root) if root else _project_root(run0)).resolve()
    rows, blobs = [], {}
    for i, ly in enumerate(layers, 1):
        name, pkg = ly["camera"], Path(ly["package"])
        b = files.get(ly.get("file"))
        n = (pkg / "live" / "tracks.bin").stat().st_size // TRACK_DTYPE.itemsize
        if b is None or len(b) != BYTES_PER_POINT * n:
            raise ValueError(f"{name}: its drawing ({0 if b is None else len(b)} bytes) is not one of its "
                             f"package ({n} points)")
        if ly.get("id"):
            m = read_pass(Path(ly["run"]), ly["id"])
            if name not in {c.get("name") for c in m.get("cameras") or []}:
                raise ValueError(f"{name}: not in pass {ly['id']}")
        file = f"{i:03d}_{safe_id(ly.get('camera_id') or name) or 'camera'}.bin"
        blobs[file] = b
        rows.append({"layer": i, "camera": name, "camera_id": ly.get("camera_id"), "package": _rel(pkg, root),
                     "run": _rel(ly["run"], root), "id": ly.get("id") or None, "key": ly.get("key"),
                     "kind": str(ly.get("kind") or "play"), "stacked": bool(ly.get("stacked")),
                     "up_to": ly.get("up_to"), "file": f"layers/{file}", "points": int(n)})
    cid = safe_id(layers[0].get("camera_id") or camera) or "camera" if camera else "collage"
    out = new_exports_dir(run0, cid)
    (out / "layers").mkdir(parents=True)
    for file, b in blobs.items():
        (out / "layers" / file).write_bytes(b)
    meta = {"format": LAYERS_FORMAT, "version": 1, "kind": "camera" if camera else "collage", "camera": camera,
            "site": site, "run": _rel(run0, root), "made": time.strftime("%Y-%m-%dT%H:%M:%S"), "layout": LAYOUT,
            "layers": rows}
    (out / LAYERS_FILE).write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return out


def layer_points(out: Path, meta: dict, root: Path) -> tuple[pd.DataFrame, list[dict]]:
    """The layers' drawn points in BNG metres (the run columns + camera, source_track, part, kind,
    cause, layer (1, 2, ... as listed), pass (its id; empty: not saved)), a track_id per layer, footage
    track and piece; and per layer {name, run (its package), origin, scene, points, drawn, layer}."""
    frames, cams, pk, next_id = [], [], {}, 1
    for ly in meta["layers"]:
        crun = _abs(ly["package"], root)
        if crun not in pk:
            pk[crun] = _package(crun)
        scene, rec, index = pk[crun]
        df, next_id = _drawn(ly["camera"], scene, rec, index, read_drawing(out / ly["file"], len(rec)), next_id)
        cams.append({"name": ly["camera"], "run": crun, "origin": _origin(scene), "scene": scene,
                     "points": len(rec), "drawn": 0 if df is None else len(df), "layer": ly["layer"]})
        if df is not None:
            df["layer"], df["pass"] = ly["layer"], ly.get("id") or ""
            frames.append(df)
    return _points(frames, LAYER_COLUMNS), cams


def layers_path(meta: dict, cams: list[dict], root: Path) -> pd.DataFrame:
    """camera_path for the layers: each layer's camera rows in the recording of the pass that drew it
    (none for a layer not saved, saved before recordings, or deleted since), with its layer number."""
    out, recs = [], {}
    for ly, c in zip(meta["layers"], cams, strict=True):
        if not ly.get("id"):
            continue
        run = _abs(ly["run"], root)
        k = (str(run), ly["id"])
        if k not in recs:
            try:
                recs[k] = read_recording(run, ly["id"])
            except FileNotFoundError:               # the pass deleted since: its drawing is in layers/
                recs[k] = None
        if recs[k] is None:
            continue
        for df in _path_rows(recs[k], {ly["camera"]: c["origin"]}, keep={ly["camera"]}):
            df.insert(0, "layer", ly["layer"])
            out.append(df)
    cols = ["layer", *PATH_COLUMNS]
    return pd.concat(out, ignore_index=True)[cols] if out else pd.DataFrame(columns=cols)


def export_layers(out_dir: str | Path, rhino: bool = False, houdini: bool = False, build: bool = True,
                  log=print) -> dict:
    """The layers prepared in out_dir (prepare_layers) as a run, in that folder: points, metrics,
    GeoJSON, camera path, Houdini curves and field always; the Rhino model with rhino, the POP kit
    with houdini. The site's config and the start camera are the first layer's run's."""
    out = Path(out_dir).resolve()
    meta = json.loads((out / LAYERS_FILE).read_text(encoding="utf-8"))
    if meta.get("format") != LAYERS_FORMAT:
        raise ValueError(f"{out / LAYERS_FILE}: not a layers export")
    run = out.parent.parent.parent                  # <run>/live/exports/<this>
    root = _project_root(run)
    pr, cfg, cfg_path, h = _open_out(out, run)
    points, cams = layer_points(out, meta, root)
    what = f"camera {meta['camera']}" if meta.get("camera") else "collage"
    log(f"{what}: {len(cams)} layers, {len(points)} drawn points, "
        f"{points['track_id'].nunique() if len(points) else 0} pieces")
    res = _write_out(pr, cfg, h, points, cams, layers_path(meta, cams, root))
    _write_models(res, pr, cfg, h, cams, f"{cfg.site} · {what} · {out.name}",
                  f"{cfg.site}_{safe_id(out.name.replace('-', '_'))}", rhino, houdini, build, log,
                  by_camera={ly["camera"]: camera_label(ly["camera"], ly.get("camera_id")) for ly in meta["layers"]})
    for ly, c in zip(meta["layers"], cams, strict=True):
        ly["drawn"] = c["drawn"]
    meta.update({"config": str(cfg_path), "crs": _crs(h), "points": len(points), "tracks": res["tracks"],
                 "exported": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "files": {k: v for k, v in res.items() if k not in ("dir", "points", "tracks")}})
    (out / LAYERS_FILE).write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log(f"layers export -> {out}")
    return res
