"""Orchestration: one output folder per run."""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import exports, visuals
from .config import Config
from .geo import load_geotiff
from .homography import Homography, snapshot
from .layout import Run, open_run
from .lens import Lens
from .trajectories import assign_track_classes, process_tracks
from .video import video_info


class RunLog:
    def __init__(self, path: Path | None = None):
        self.path = path

    def __call__(self, msg: str = "") -> None:
        print(msg, flush=True)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")


def new_run_dir(cfg: Config, name: str | None = None) -> Path:
    root = cfg.path("output_dir") or (cfg.base_dir / "runs")
    d = root / cfg.site / (name or datetime.now().strftime("%Y%m%d-%H%M%S"))
    Run(d).make()
    return d


def _require(path: Path | None, what: str) -> Path:
    if path is None:
        raise ValueError(f"`{what}` is not set in the site config")
    if not path.exists():
        raise FileNotFoundError(f"{what}: {path} does not exist")
    return path


def load_calibration(cfg: Config, video: Path | None = None) -> tuple[Homography, Lens | None]:
    h = Homography.load(_require(cfg.path("homography"), "homography"))
    lens = Lens.load(cfg.path("lens")) if cfg.get("lens") else None
    if video is not None and h.image_size:
        info = video_info(video, cfg.get("fps"))
        if (info.width, info.height) != tuple(h.image_size):
            raise ValueError(
                f"Video is {info.width}x{info.height} but the homography was calibrated on a "
                f"{h.image_size[0]}x{h.image_size[1]} frame. Re-calibrate with this footage "
                "(same camera position, resolution and lens).")
    return h, lens


def load_run_registration(run_dir: Path, h: Homography, log=print) -> dict | None:
    """Per-frame camera-motion registration of a run, if it was tracked with stabilize."""
    p = open_run(run_dir).registration
    if not p.exists():
        return None
    from .registration import load_registration

    reg, ref, _ = load_registration(p)
    if ref != h.reference_frame:
        raise ValueError(
            f"This run was stabilized to frame {ref} but the homography was clicked on frame "
            f"{h.reference_frame}. Re-run `track` (or `run`) after calibrating.")
    log(f"  camera-motion compensation: {len(reg)} frames aligned to frame {ref}")
    return reg


def stage_track(cfg: Config, run_dir: Path, device: str, log=print,
                video: Path | None = None, max_frames: int | None = None) -> pd.DataFrame:
    from .detect import run_tracking

    video = video or _require(cfg.path("video"), "video")
    return run_tracking(cfg, video, Run(run_dir).make().raw_tracks, device, log=log,
                        max_frames=max_frames)


def stage_process(cfg: Config, run_dir: Path, log=print, raw: pd.DataFrame | None = None,
                  video: Path | None = None) -> dict:
    t0 = time.time()
    run = open_run(run_dir).make()
    run_dir = run.root
    meta = json.loads(run.meta.read_text(encoding="utf-8"))
    if raw is None:
        raw = pd.read_csv(run.raw_tracks)
    video = video or Path(meta["video"])
    h, lens = load_calibration(cfg, video if video.exists() else None)
    # keep the run's calibration snapshot in sync with the one these results are made with
    # (the overlay video projects the results back into the footage with it)
    snapshot(cfg.path("homography"), run.homography_used)
    fps, stride = float(meta["fps"]), int(meta["vid_stride"])

    reg = load_run_registration(run_dir, h, log)
    log("projecting, cleaning and smoothing tracks ...")
    points, summary = process_tracks(raw, cfg, h, fps, stride, lens, log=log, registration=reg)
    log(f"  {summary.shape[0]} tracks, {len(points)} samples "
        f"({int(points['predicted'].sum()) if len(points) else 0} predicted/interpolated)")

    if h.terrain is not None and len(points):
        # the ground elevation under every sample (m above sea level): terraces, steps
        points["z"] = np.round(h.terrain.height(points[["x", "y"]].to_numpy(float)), 2)
    out = {}
    out["points"] = run.points
    exports.write_points_csv(points, out["points"])
    shutil.copy2(run.points, run.houdini / "points.csv")      # houdini/ is self-contained
    out["track_metrics"] = run.tracks
    summary.to_csv(out["track_metrics"], index=False)
    out["tracks_geojson"] = run.data / "tracks.geojson"
    exports.write_geojson(exports.tracks_geojson(points, summary), out["tracks_geojson"])
    out["gaps_geojson"] = run.data / "predicted_gaps.geojson"
    exports.write_geojson(exports.predicted_segments_geojson(points), out["gaps_geojson"])

    gcfg = cfg["grid"]
    cell = float(gcfg["cell_size_m"])
    out["field_grid"] = run.data / "field_grid.csv"
    exports.field_grid(points, cell, gcfg["include_predicted"]).to_csv(out["field_grid"],
                                                                       index=False)
    present = sorted(set(points["group"])) if len(points) else []
    if len(present) > 1:
        for gname in present:
            p = run.data / f"field_grid_{gname}.csv"
            exports.field_grid(points[points["group"] == gname], cell,
                               gcfg["include_predicted"]).to_csv(p, index=False)

    out.update(_export_field(cfg, run, points, h, stride / fps))
    out.update(_export_rhino(cfg, run, h, log))

    window = (meta["start_frame"] / fps, meta["end_frame"] / fps)
    stats = exports.compute_stats(points, summary, cfg, window)
    stats["ground"] = ({"model": "terrain", "datum_m": round(h.ground_z_m, 2),
                        "source": Path(h.terrain_path).name}
                       if h.terrain is not None else {"model": "plane"})
    hidden = _feet_hidden_check(summary, log)
    if hidden is not None:
        stats["people_feet_hidden_share"] = hidden
    check = _people_height_check(cfg, raw, h, reg, log)
    if check:
        stats["calibration_check"] = check
    roads = _roads_check(cfg, raw, h, reg, log)
    if roads:
        stats["calibration_roads_check"] = roads
    out["stats_json"] = run.stats_json
    out["stats_json"].write_text(json.dumps(stats, indent=2), encoding="utf-8")
    out["stats_csv"] = run.stats_csv
    exports.stats_table(stats).to_csv(out["stats_csv"], index=False)

    out["houdini"] = run.houdini / "houdini_import.py"
    exports.write_houdini_script(out["houdini"], run.houdini / "points.csv", h.origin, cfg,
                                 z0=h.ground_z_m)

    log("rendering visuals ...")
    raster = None
    if cfg.get("geotiff"):
        raster = load_geotiff(_require(cfg.path("geotiff"), "geotiff"), max_dim=8000, warn=log)
    title = f"{cfg.site} · {Path(meta['video']).name}"
    out["topdown"] = run.extras / "topdown.png"
    visuals.plot_topdown(points, raster, cfg, out["topdown"], title)
    out["density"] = run.extras / "density.png"
    visuals.plot_density(points, raster, cfg, out["density"], dt=stride / fps)
    out["speed_histogram"] = run.extras / "speed_histogram.png"
    visuals.plot_speed_histogram(summary, cfg, out["speed_histogram"])
    for gname, (fpath, label, rng) in _field_plots(cfg, run, points).items():
        field = pd.read_csv(fpath)
        sel = points if gname == "all" else points[points["group"] == gname]
        png = run.extras / ("flow_field.png" if gname == "all" else f"flow_field_{gname}.png")
        visuals.plot_flow_field(field, sel, raster, cfg, png, float(cfg["field"]["cell_size_m"]),
                                rng, label)
        out[png.stem] = png
    lanes = _lanes_check(points, raster, log)
    if lanes:
        stats["tracking_check"] = lanes
        out["stats_json"].write_text(json.dumps(stats, indent=2), encoding="utf-8")
    clean = _clean_videos(cfg)
    if cfg["visuals"].get("topdown_video", True) and len(points):
        from .animation import render_topdown_video

        out["topdown_video"] = run.videos / "topdown.mp4"
        render_topdown_video(points, raster, cfg, out["topdown_video"], fps / stride,
                             visuals._extent(points, raster, cfg),
                             speedup=float(cfg["visuals"].get("topdown_video_speedup", 1.0)),
                             log=log, clean_path=clean and run.videos / "topdown_clean.mp4")
    if cfg["visuals"].get("flowfield_video", True) and len(points):
        from .animation import render_flowfield_video

        present = sorted(set(points["group"]))
        gname = points.groupby("group")["track_id"].nunique().idxmax()
        fields = ([(pd.read_csv(run.field_csv(g)), cfg.groups[g].speed_range) for g in present]
                  if len(present) > 1 else pd.read_csv(run.field_csv()))
        out["flowfield_video"] = run.videos / "flowfield.mp4"
        render_flowfield_video(fields, points, raster, cfg,
                               out["flowfield_video"], visuals._extent(points, raster, cfg),
                               float(cfg["field"]["cell_size_m"]),
                               cfg.groups[gname].speed_range, log=log,
                               clean_path=clean and run.videos / "flowfield_clean.mp4")

    if cfg["debug_video"].get("enabled"):
        from .debug_video import render_debug_video
        from .detect import site_roi

        if video.exists():
            r = raw.copy()
            cls = assign_track_classes(r)
            r["class"] = r["track_id"].map(cls).fillna(r["class"])
            out["overlay_video"] = run.videos / "overlay.mp4"
            render_debug_video(video, r, cfg, out["overlay_video"], fps,
                               site_roi(cfg), log=log, run_dir=run_dir,
                               clean_path=clean and run.videos / "overlay_clean.mp4")
        else:
            log(f"debug video skipped: {video} not found")

    if clean and len(points):
        from .package import build_package

        try:
            build_package(cfg, run_dir, raster, video, log=log)
        except Exception as e:  # never lose a finished run over the presentation layer
            log(f"images / labels failed: {e!r} (re-try with `motus package`)")

    _log_stats(stats, log)
    log(f"processing took {time.time() - t0:.1f} s. Outputs in {run_dir}")
    return out


def _clean_videos(cfg: Config) -> bool:
    """Frameless outputs (images/, labels/, *_clean.mp4) are on unless package.enabled: false."""
    return bool((cfg.get("package") or {}).get("enabled", True))


def _lanes_check(points: pd.DataFrame, raster, log) -> dict | None:
    """Vehicle tracks against the painted lanes in the aerial (rotation + sideways scatter)."""
    if raster is None or not len(points) or "vehicles" not in set(points["group"]):
        return None
    try:
        from .autocal import lane_check

        chk = lane_check(points, raster)
    except Exception as e:  # a check must never break a run
        log(f"  lane check skipped: {e}")
        return None
    if chk:
        rot = chk["rotation_vs_painted_lanes_deg"]
        log(f"  tracking check: straight tracks run {rot:+.1f}° off the painted lanes "
            f"(median |{chk['abs_angle_vs_lanes_deg']:.1f}°|), sideways scatter "
            f"{chk['sideways_scatter_cm']:.0f} cm" + (" -> consistent" if abs(rot) <= 1.5 else
                                                      " -> CHECK the calibration direction"))
    return chk


def _roads_check(cfg: Config, raw: pd.DataFrame, h: Homography, reg, log) -> dict | None:
    """Share of vehicle positions on OpenStreetMap roads (needs camera_position; the road
    lines are downloaded once per camera spot and cached next to the calibration)."""
    prior = cfg.get("camera_position")
    if not prior or not cfg.path("homography") or \
            raw["class"].isin(["car", "bus", "truck"]).sum() < 100:
        return None
    try:
        from .autocal import _road_cache, osm_roads, quality_checks

        ways = osm_roads(prior["E"], prior["N"], cache=_road_cache(cfg.path("homography"), prior),
                         timeout=20)
        c = [x for x in quality_checks(h, raw, cfg, reg, ways, people=False)["checks"]
             if x["name"] == "vehicles on roads"]
    except Exception as e:  # offline, Overpass busy, ...: the check is optional
        log(f"  road check skipped: {e}")
        return None
    if not c:
        return None
    log(f"  calibration check: {c[0]['value']} of vehicle positions within 6 m of a mapped road"
        + (" -> consistent" if c[0]["ok"] else " -> CHECK the calibration"))
    return {"vehicles_on_roads": c[0]["value"], "ok": c[0]["ok"], "note": c[0]["note"]}


def _feet_hidden_check(summary: pd.DataFrame, log) -> float | None:
    """How often people's feet are hidden behind nearer people (detections, all tracks); warn
    when it is most of the time: a crowd seen from low down, positions and speeds doubtful."""
    from .trajectories import FEET_HIDDEN_WARN

    if not len(summary) or "feet_hidden_share" not in summary:
        return None
    ppl = summary[summary["class"] == "person"]
    if not len(ppl):
        return None
    share = float(np.average(ppl["feet_hidden_share"], weights=ppl["n_points"]))
    if share > FEET_HIDDEN_WARN:
        log(f"  WARNING: people's feet are hidden behind others in {share:.0%} of the frames "
            "(a crowd seen from low down): their ground positions come from where the boxes "
            "are cut off, so people's positions and speeds here are doubtful (often too fast). "
            "Per track: feet_hidden_share in track_metrics.csv. A higher camera avoids it.")
    return round(share, 3)


def _people_height_check(cfg: Config, raw: pd.DataFrame, h: Homography, reg, log) -> dict | None:
    """How tall people would be under this calibration (~1.7 m = consistent)."""
    if "person" not in cfg.classes:
        return None
    from .selfcal import PERSON_HEIGHT_M, people_check
    from .trajectories import max_range_for

    try:
        chk = people_check(raw, h, reg, max_range_for(cfg, h, "people"))
    except Exception as e:  # a check must never break a run
        log(f"  calibration check skipped: {e}")
        return None
    if chk:
        z, err = chk["implied_person_height_m"], chk["scale_error_pct"]
        share = chk.get("consistent_share")
        good = abs(err) <= 10 and (share is None or share >= 0.8)
        verdict = "consistent" if good else "CHECK the calibration"
        bands = "  ".join(f"{k}: {v:.2f}" for k, v in (chk.get("by_distance_m") or {}).items())
        log(f"  calibration check: people come out {z:.2f} m tall (expected ~{PERSON_HEIGHT_M} m,"
            f" {err:+.0f}%)" + (f"; by distance {bands}" if bands else "") + f" -> {verdict}")
        if not good:
            log("    on a right calibration people are ~1.7 m near and far; bands that drift "
                "apart point at the tilt / zoom / camera height: re-click well-spread points "
                "on flat ground (not on steps or walls), or limit projection.max_range_m to the "
                "distances that check out.")
    return chk


def _field_groups(points: pd.DataFrame) -> list[str]:
    present = sorted(set(points["group"])) if len(points) else []
    return ["all"] + (present if len(present) > 1 else [])


def _export_rhino(cfg: Config, run: Run, h: Homography, log) -> dict:
    """rhino/<site>.3dm and grasshopper_import.py (see rhino.py)."""
    from .rhino import write_rhino

    cp = getattr(h, "camera_params", None) or {}
    cams = [(cfg.site, cp["E"], cp["N"], cp.get("height_m", 0.0))] if "E" in cp else []
    try:
        return write_rhino(run.root, cfg, h.origin, h.ground_z_m, log, cameras=cams)
    except Exception as e:                       # never lose a run over the Rhino files
        log(f"rhino export failed: {e}")
        return {}


def _export_field(cfg: Config, run: Run, points: pd.DataFrame, h: Homography,
                  dt: float) -> dict:
    """vector_field*.csv (+ time slices) and houdini_field.py."""
    from .field import smooth_field, time_sliced_fields

    fc = cfg["field"]
    cell, smooth = float(fc["cell_size_m"]), float(fc["smooth_m"])
    kw = dict(include_predicted=bool(fc["include_predicted"]),
              confidence_samples=float(fc["confidence_samples"]),
              direction_bins=int(fc.get("direction_bins") or 0))
    out, files, slices = {}, {}, {}
    for gname in _field_groups(points):
        sel = points if gname == "all" else points[points["group"] == gname]
        suffix = "" if gname == "all" else f"_{gname}"
        p = run.houdini / f"vector_field{suffix}.csv"
        smooth_field(sel, cell, smooth, dt, **kw).to_csv(p, index=False)
        files[gname] = p
        out[p.stem] = p
        if fc.get("time_window_s"):
            ps = run.houdini / f"vector_field_slices{suffix}.csv"
            time_sliced_fields(sel, float(fc["time_window_s"]), cell, smooth, dt, **kw)                 .to_csv(ps, index=False)
            slices[gname] = ps
            out[ps.stem] = ps
    out["houdini_field"] = run.houdini / "houdini_field.py"
    exports.write_houdini_field_script(out["houdini_field"], files, slices,
                                       run.houdini / "points.csv", h.origin, cell, cfg)
    return out


def _field_plots(cfg: Config, run: Run, points: pd.DataFrame) -> dict:
    """group -> (field csv, label, speed range) for the flow-field PNGs."""
    from .visuals import GROUP_LABELS

    groups = _field_groups(points)
    res = {}
    for gname in groups:
        if gname == "all" and len(groups) > 1:
            continue  # mixed speeds (people + cycles) would share one colour scale: skip
        g = gname if gname != "all" else (sorted(set(points["group"]))[0] if len(points)
                                          else next(iter(cfg.groups)))
        res[gname] = (run.field_csv(gname), GROUP_LABELS.get(g, g),
                      cfg.groups[g].speed_range)
    return res


def _log_stats(stats: dict, log) -> None:
    for g, b in stats["per_group"].items():
        log(f"  {g:9s} {b['n_tracks']:5d} tracks  mean "
            f"{b['mean_speed'] or 0:6.2f} m/s  median {b['median_speed'] or 0:6.2f} m/s  "
            f"straightness {b['mean_straightness'] or 0:.2f}")
    for cl in stats["count_lines"]:
        log(f"  count line {cl['name']}: {cl['crossings_left_to_right']} L->R, "
            f"{cl['crossings_right_to_left']} R->L")


def snapshot_inputs(cfg: Config, run_dir: Path) -> None:
    run = Run(run_dir).make()
    run.config_used.write_text(
        yaml.safe_dump(cfg.data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    hp = cfg.path("homography")
    if hp and hp.exists():
        snapshot(hp, run.homography_used)


def run_all(cfg: Config, device: str, run_name: str | None = None,
            max_frames: int | None = None) -> Path:
    run_dir = new_run_dir(cfg, run_name)
    log = RunLog(Run(run_dir).log)
    log(f"run folder: {run_dir}")
    snapshot_inputs(cfg, run_dir)
    video = _require(cfg.path("video"), "video")
    load_calibration(cfg, video)  # fail fast before spending time on detection
    raw = stage_track(cfg, run_dir, device, log=log, video=video, max_frames=max_frames)
    stage_process(cfg, run_dir, log=log, raw=raw, video=video)
    return run_dir
