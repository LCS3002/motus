"""groundtrack command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cfg(args, overrides=None):
    from .config import load_config

    ov = dict(overrides or {})
    if getattr(args, "device", None):
        ov["device"] = args.device
    return load_config(args.config, ov)


def _device(cfg, log=print):
    from .device import describe_device, resolve_device

    d = resolve_device(cfg["device"])
    log(describe_device(d))
    return d


def _run_dir_arg(cfg, run: str | None) -> Path:
    """--run accepts a folder path, a run name, or 'latest'."""
    from .layout import Run, open_run

    root = (cfg.path("output_dir") or cfg.base_dir / "runs") / cfg.site
    if run in (None, "latest"):
        runs = sorted(p for p in root.glob("*") if Run(p).is_run())
        if not runs:
            raise SystemExit(f"No tracked runs under {root}")
        return open_run(runs[-1]).root
    p = Path(run)
    return open_run(p if p.exists() else root / run).root


# --------------------------------------------------------------------------- commands
def cmd_device(args):
    from .device import describe_device, resolve_device

    print(describe_device(resolve_device(args.device or "auto")))


def cmd_init(args):
    from .templates import TEMPLATES

    out = Path(args.out or f"sites/{args.site}.yaml")
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists (use --force to overwrite)")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(TEMPLATES[args.template].format(site=args.site, fname=out.as_posix()),
                   encoding="utf-8")
    for d in ("footage", "maps", "calibration", "runs"):
        (out.parent.parent / d).mkdir(exist_ok=True)
    print(f"wrote {out}. Put the video in footage/ and the GeoTIFF in maps/, then run:\n"
          f"  motus calibrate --config {out.as_posix()}")


def cmd_demo(args):
    from .calibrate import run_calibration
    from .config import load_config
    from .demo import make_demo_site
    from .pipeline import run_all

    out = Path(args.out)
    site = make_demo_site(out)
    print(f"demo site written to {out}")
    cfg = load_config(site["config"], {"device": args.device} if args.device else None)
    run_calibration(site["video"], site["geotiff"], cfg.path("homography"),
                    points_csv=site["points_csv"], interactive=False)
    run_all(cfg, _device(cfg), "demo")


def cmd_lens(args):
    from .lens import calibrate_lens

    cols, rows = (int(v) for v in args.pattern.lower().split("x"))
    lens = calibrate_lens(args.source, (cols, rows), args.square_mm, args.every_s)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    lens.save(args.out)
    print(f"saved {args.out}. Set `lens: {args.out}` in the site config, then re-calibrate.")


def cmd_calibrate(args):
    from .calibrate import run_calibration
    from .lens import Lens
    from .video import video_info

    cfg = _cfg(args)
    video = cfg.path("video")
    frame = args.frame
    if args.time is not None:
        frame = int(round(args.time * video_info(video, cfg.get("fps")).fps))
    lens = Lens.load(cfg.path("lens")) if cfg.get("lens") else None
    out = cfg.path("homography")
    if out is None:
        raise SystemExit("Set `homography:` (output path) in the site config")
    from .terrain import resolve_terrain

    h = run_calibration(video, cfg.path("geotiff"), out, frame=frame or 0, lens=lens,
                        points_csv=Path(args.points_csv) if args.points_csv else None,
                        interactive=not args.no_gui, ransac_thresh_m=args.ransac_m,
                        warn_m=args.warn_m, camera_prior=cfg.get("camera_position"),
                        terrain=resolve_terrain(cfg), hide=_privacy_boxes_of(cfg))
    if h is not None and getattr(h, "camera_prior_set", None):
        from .locate import write_camera_position

        write_camera_position(args.config, h.camera_prior_set, note="set on the map in the picker")
        print(f"camera position saved to {args.config}")


def _privacy_boxes_of(cfg):
    """People / number plates in the site's latest tracked run, to blur saved check images."""
    import pandas as pd

    from .autocal import _latest_run
    from .debug_video import privacy_boxes
    from .layout import open_run

    rd = _latest_run(cfg)
    if rd is None or not open_run(rd).raw_tracks.exists():
        return None
    dv = cfg.get("debug_video") or {}
    from .debug_video import raw_all

    return privacy_boxes(raw_all(rd, pd.read_csv(open_run(rd).raw_tracks)),
                         bool(dv.get("blur_people", True)),
                         bool(dv.get("blur_plates", True)))


def cmd_composite(args):
    """Several clips of one place on one plan (see composite.py)."""
    from .composite import build_composite, load_clip
    from .config import load_config
    from .geo import load_geotiff
    from .layout import Run

    clips = []
    for c in args.configs:
        cfg = load_config(c)
        root = (cfg.path("output_dir") or cfg.base_dir / "runs") / cfg.site
        if args.run:
            rd = root / args.run
        else:
            runs = sorted((p for p in root.glob("*") if Run(p).processed()),
                          key=lambda p: p.stat().st_mtime) if root.exists() else []
            rd = runs[-1] if runs else None
        clip = load_clip(cfg.site, cfg, rd) if rd is not None else None
        if clip is None:
            print(f"  {cfg.site}: no processed run, left out")
            continue
        if args.only_ok and clip.grade != "ok":
            print(f"  {cfg.site}: calibration graded {clip.grade}, left out (--only-ok)")
            continue
        clips.append(clip)
    if not clips:
        raise SystemExit("no processed runs to combine")
    cfg0 = clips[0].cfg
    pk = cfg0.get("package") or {}
    raster = load_geotiff(cfg0.path("geotiff"), max_dim=20000) if cfg0.get("geotiff") else None
    build_composite(clips, raster, Path(args.out), args.title or pk.get("title") or cfg0.site,
                    args.subtitle or "", pk.get("credit", ""), plan_height_px=args.height)


def cmd_privacy(args):
    """Every person / vehicle box of a run's frames, for blurring (runs tracked before
    privacy_boxes.csv was written); re-render the overlay video / package afterwards."""
    from .detect import detect_privacy_boxes
    from .layout import open_run

    cfg = _cfg(args)
    run = open_run(args.run)
    meta = json.loads(run.meta.read_text(encoding="utf-8"))
    detect_privacy_boxes(cfg, Path(meta["video"]), run.privacy, _device(cfg),
                         int(meta["start_frame"]), int(meta["end_frame"]),
                         int(meta["vid_stride"]))


def cmd_autocalibrate(args):
    from .autocal import run_autocalibration

    cfg = _cfg(args)
    run_dir = _run_dir_arg(cfg, args.run) if args.run else None
    rep = run_autocalibration(cfg, run_dir=run_dir, device=cfg["device"], replace=args.replace,
                              track_frames=args.frames)
    if rep.get("result") is None:
        raise SystemExit(1)


def cmd_roi(args):
    from .calibrate import pick_roi

    cfg = _cfg(args)
    from .detect import reference_frame_for

    out = Path(args.out) if args.out else cfg.base_dir / f"{cfg.site}_roi.json"
    # draw on the calibration frame: with `stabilize` the ROI is applied in that frame
    frame = args.frame if args.frame is not None else reference_frame_for(cfg)
    pick_roi(cfg.path("video"), frame, out)
    if not args.out:
        print(f"{out.name} is picked up automatically for site '{cfg.site}'.")


def cmd_track(args):
    from .layout import Run
    from .pipeline import RunLog, load_calibration, new_run_dir, snapshot_inputs, stage_track

    cfg = _cfg(args)
    run_dir = new_run_dir(cfg, args.run_name)
    log = RunLog(Run(run_dir).log)
    snapshot_inputs(cfg, run_dir)
    if cfg.get("homography") and cfg.path("homography").exists():
        load_calibration(cfg, cfg.path("video"))
    stage_track(cfg, run_dir, _device(cfg, log), log=log, max_frames=args.max_frames)
    log(f"next: motus process --config {args.config} --run {run_dir}")


def cmd_process(args):
    from .layout import Run
    from .pipeline import RunLog, stage_process

    ov = {"debug_video": {"enabled": True}} if args.debug_video else {}
    cfg = _cfg(args, ov)
    run_dir = _run_dir_arg(cfg, args.run)
    stage_process(cfg, run_dir, log=RunLog(Run(run_dir).log))


def cmd_stabilize(args):
    """Camera motion of an existing run (a handheld clip tracked without it), without
    detecting again: writes the registration, then process the run again."""
    import json

    import pandas as pd

    from .detect import reference_frame_for
    from .layout import open_run
    from .registration import register_run

    cfg = _cfg(args)
    run = open_run(_run_dir_arg(cfg, args.run))
    meta = json.loads(run.meta.read_text(encoding="utf-8"))
    raw = pd.read_csv(run.raw_tracks)
    extra = pd.read_csv(run.privacy) if run.privacy.exists() else None
    from .homography import Homography

    hp = cfg.path("homography")
    cam = (Homography.load(hp).camera_params or {}) if hp is not None and hp.exists() else {}
    info = register_run(meta["video"], raw, reference_frame_for(cfg), int(meta["start_frame"]),
                        int(meta["end_frame"]), int(meta["vid_stride"]), float(meta["fps"]),
                        run.registration, extra, focal_px=cam.get("focal_px"))
    meta.update(info)
    run.meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"registration -> {run.registration}; now: motus process -c {args.config} "
          f"--run {run.root}")


def cmd_rhino(args):
    """rhino/<site>.3dm + grasshopper_import.py for an existing run."""
    from .homography import Homography
    from .layout import open_run
    from .rhino import write_rhino

    cfg = _cfg(args)
    run = open_run(_run_dir_arg(cfg, args.run))
    h = Homography.load(run.homography_used)
    cp = getattr(h, "camera_params", None) or {}
    cams = [(cfg.site, cp["E"], cp["N"], cp.get("height_m", 0.0))] if "E" in cp else []
    write_rhino(run.root, cfg, h.origin, h.ground_z_m, cameras=cams)


def cmd_calibration_picture(args):
    """images/calibration*.png: the calibration of a run, to show the process."""
    from .calfigure import calibration_figure
    from .geo import load_geotiff
    from .homography import Homography
    from .layout import open_run

    cfg = _cfg(args)
    run_dir = _run_dir_arg(cfg, args.run)
    raster = load_geotiff(cfg.path("geotiff"), max_dim=8000)
    calibration_figure(cfg, run_dir, raster, Homography.load(open_run(run_dir).homography_used),
                       out_h=args.height)


def cmd_process_sequence(args):
    """images/process/: the whole process of a run at one moment, one image per step."""
    from .calfigure import process_sequence
    from .geo import load_geotiff
    from .homography import Homography
    from .layout import open_run

    cfg = _cfg(args)
    run_dir = _run_dir_arg(cfg, args.run)
    raster = load_geotiff(cfg.path("geotiff"), max_dim=8000)
    process_sequence(cfg, run_dir, raster, Homography.load(open_run(run_dir).homography_used),
                     at_s=args.at, out_h=args.height)


def cmd_houdini_pop(args):
    """A Houdini POP kit of a site: the combined field and every clip's field as modules."""
    from .houdini_pop import build_scene, prepare_kit

    kit = prepare_kit(args.site, Path(args.composite), [Path(c) for c in args.clips],
                      Path(args.out))
    if not args.no_build:
        build_scene(kit)


def cmd_live_pass_export(args):
    """<run>/live/passes/<id>/pass_<id>/: a saved Live pass as a run (live_pass_export.py)."""
    from .live_pass_export import export_pass

    export_pass(args.run, args.id, rhino=args.rhino, houdini=args.houdini, build=not args.no_build)


def cmd_live_layers_export(args):
    """<run>/live/exports/<stamp>-<collage | camera id>/: the Live collage's layers as a run."""
    from .live_pass_export import export_layers

    export_layers(args.dir, rhino=args.rhino, houdini=args.houdini, build=not args.no_build)


def cmd_live_sheet(args):
    """<run>/live/passes/<id>/sheet_<theme>.pdf / .png: the pass's performance sheet (live_sheet.py)."""
    from .live_sheet import make_sheet

    themes = ("black", "light") if args.theme == "both" else (args.theme,)
    make_sheet(args.run, args.id, layout=args.layout, themes=themes, view=not args.no_view)


def cmd_package(args):
    from .geo import load_geotiff
    from .package import build_package

    cfg = _cfg(args)
    run_dir = _run_dir_arg(cfg, args.run)
    raster = load_geotiff(cfg.path("geotiff"), max_dim=8000) if cfg.get("geotiff") else None
    build_package(cfg, run_dir, raster, plate=not args.no_plate)


def cmd_locate(args):
    from .locate import camera_from_place, search, write_camera_position

    hits = search(args.place, limit=args.limit)
    if not hits:
        raise SystemExit(f"OpenStreetMap found nothing for {args.place!r}")
    for k, h in enumerate(hits, 1):
        print(f"{k}. {h['name']}\n   E {h['E']:.1f}  N {h['N']:.1f}  (~{h['size_m']:.0f} m across)")
    pick = hits[args.pick - 1]
    cam = camera_from_place(pick, args.floor, args.floor_height, ground_offset_m=args.ground_m)
    if args.height is not None:
        cam["height_m"] = args.height
        cam["height_tol_m"] = round(max(1.0, 0.06 * args.height), 1)
    print("\ncamera_position:\n" + "".join(f"  {k}: {v}\n" for k, v in cam.items()))
    if args.config and args.write:
        write_camera_position(args.config, cam, note=f"OSM: {pick['name'][:60]}")
        print(f"written to {args.config}")
    elif args.config:
        print("add --write to put this into the config")


def cmd_ui(args):
    from .ui import serve

    serve(Path(args.root), port=args.port, open_browser=not args.no_browser)


def cmd_app(args):
    from .desktop import make_shortcut, serve

    root = Path(args.root).resolve()
    if args.shortcut:
        for p in make_shortcut(root):
            print(f"shortcut: {p}")
        return
    serve(root, port=args.port, window=not args.no_window, debug=args.debug,
          custom_frame=not args.native_frame)


def cmd_houdini_build(args):
    """Build (again) the scene of an existing Houdini kit, e.g. one from Grasshopper."""
    from .houdini_pop import build_scene

    kit = Path(args.kit)
    if not (kit / "modules.json").exists() or not (kit / "hou_pop_build.py").exists():
        raise SystemExit(f"{kit} is not a Houdini kit (modules.json + hou_pop_build.py)")
    if build_scene(kit) is None:
        raise SystemExit(1)


def cmd_run(args):
    from .pipeline import run_all

    ov = {"debug_video": {"enabled": True}} if args.debug_video else {}
    cfg = _cfg(args, ov)
    run_all(cfg, _device(cfg), args.run_name, max_frames=args.max_frames)


def cmd_debug_video(args):
    import json

    import pandas as pd

    from .debug_video import render_debug_video
    from .detect import site_roi
    from .layout import Run
    from .trajectories import assign_track_classes

    dv = {"blur_people": not args.no_blur}
    if args.boxes:
        dv["style"] = "boxes"
    if args.no_stabilize_output:
        dv["stabilize_output"] = False
    if args.trail_s is not None:
        dv["trail_s"] = args.trail_s
    cfg = _cfg(args, {"debug_video": dv})
    run = Run(_run_dir_arg(cfg, args.run)).make()
    meta = json.loads(run.meta.read_text(encoding="utf-8"))
    raw = pd.read_csv(run.raw_tracks)
    raw["class"] = raw["track_id"].map(assign_track_classes(raw)).fillna(raw["class"])
    clean = None
    if (cfg.get("package") or {}).get("enabled", True) and not args.boxes:
        clean = run.videos / "overlay_clean.mp4"                 # the same, without labels
    out = run.videos / ("overlay_boxes.mp4" if args.boxes else "overlay.mp4")
    render_debug_video(Path(meta["video"]), raw, cfg, out, meta["fps"], site_roi(cfg),
                       run_dir=run.root, clean_path=clean)


def cmd_live_export(args):
    """live/: the scene package of a processed run for the Live view (see live_export.py)."""
    from .config import load_config
    from .live_export import export_live, find_site_config

    run_dir = Path(args.run_dir)
    config = Path(args.config) if args.config else find_site_config(run_dir)
    if config is None:
        raise SystemExit(f"no sites/{run_dir.resolve().parent.name}.yaml for {run_dir}: "
                         "pass --config")
    res = export_live(run_dir, load_config(config), device=args.device)
    print(f"{res['n_points']} points, {res['n_tracks']} tracks, {res['frames']} frames; "
          f"reprojection median {res['reproj_median_px']:.2f} px -> {res['dir']}")


def cmd_groundtruth(args):
    import json

    from .config import load_config
    from .groundtruth import analyse_walk, pick_track, plot_walk, write_report
    from .layout import Run
    from .pipeline import RunLog, load_calibration, new_run_dir, stage_track
    from .trajectories import process_tracks

    base = load_config(args.config)
    people = base["groups"].get("people") or {"classes": ["person"]}
    ov = {"groups": {"people": people}, "detection": {"roi": None}, "site": base.site}
    if args.device:
        ov["device"] = args.device
    cfg = load_config(args.config, ov)
    video = Path(args.video)
    h, lens = load_calibration(cfg, video)
    run_dir = new_run_dir(cfg, args.run_name or "groundtruth-" + video.stem)
    run = Run(run_dir)
    log = RunLog(run.log)
    raw = stage_track(cfg, run_dir, _device(cfg, log), log=log, video=video)
    meta = json.loads(run.meta.read_text(encoding="utf-8"))
    from .pipeline import load_run_registration

    points, summary = process_tracks(raw, cfg, h, meta["fps"], meta["vid_stride"], lens, log=log,
                                     registration=load_run_registration(run_dir, h, log))
    points.to_csv(run.points, index=False)
    summary.to_csv(run.tracks, index=False)
    a = tuple(args.start) if args.start else None
    b = tuple(args.end) if args.end else None
    tr = pick_track(points, summary, args.track_id, a, b)
    rep = analyse_walk(tr, a, b, args.length, args.stopwatch)
    write_report(rep, run.extras, log)
    plot_walk(tr, rep, run.extras / "groundtruth.png", a, b)
    log(f"report: {run.extras / 'groundtruth_report.json'}\n"
        f"plot:   {run.extras / 'groundtruth.png'}")


def cmd_live_hub(args):
    """The phone controller's hub on its own (Live mode in the app starts its own)."""
    from .live_hub import LiveHub, qr_text

    hub = LiveHub(port=args.port, view_port=args.view_port, host=args.host)
    if args.new_code:
        hub.rotate_token()
    info = hub.start()
    for a in info["phone_urls"]:
        print(f"phone: {a['url']}" + (f"  ({a['name']})" if a["name"] else ""))
    if not info["phone_urls"]:
        print("no network address found: is Wi-Fi / the hotspot on?")
    print(f"view:  {info['view_url']}\ncertificate SHA-256: {info['fingerprint']}")
    if info["url"]:
        try:
            print(qr_text(info["url"]))
        except UnicodeEncodeError:
            pass
    print("Ctrl+C to stop")
    hub.wait()


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="motus",
                                description="Oblique video -> top-down movement data (EPSG:27700)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, config=True):
        if config:
            sp.add_argument("--config", "-c", required=True, help="site YAML")
        sp.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default=None,
                        help="override the config's device")
        return sp

    s = common(sub.add_parser("device", help="show which PyTorch device will be used"), False)
    s.set_defaults(func=cmd_device)

    s = sub.add_parser("init", help="write a new site config from a template")
    s.add_argument("site")
    s.add_argument("--template", choices=["plaza", "motorway"], default="plaza")
    s.add_argument("--out")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_init)

    s = common(sub.add_parser("demo", help="build a synthetic site and run the whole pipeline"),
               False)
    s.add_argument("--out", default="demo_site")
    s.set_defaults(func=cmd_demo)

    s = sub.add_parser("lens-calibrate", help="checkerboard lens calibration (optional)")
    s.add_argument("--source", required=True, help="folder, glob of images, or a video")
    s.add_argument("--pattern", default="9x6", help="inner corners COLSxROWS (default 9x6)")
    s.add_argument("--square-mm", type=float, default=25.0)
    s.add_argument("--every-s", type=float, default=1.0, help="video: sample a frame every N s")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_lens)

    s = common(sub.add_parser("calibrate", help="pick ground points, fit + save the homography"))
    s.add_argument("--frame", type=int, default=0)
    s.add_argument("--time", type=float, help="pick the frame at this many seconds")
    s.add_argument("--points-csv", help="start from / use these point pairs (u,v,E,N)")
    s.add_argument("--no-gui", action="store_true", help="fit from --points-csv without the picker")
    s.add_argument("--ransac-m", type=float, default=1.0,
                   help="RANSAC inlier threshold in m (far points are less precise; keep >= 1)")
    s.add_argument("--warn-m", type=float, default=0.5, help="warn if RMSE exceeds this (m)")
    s.set_defaults(func=cmd_calibrate)

    s = common(sub.add_parser("autocalibrate",
                              help="calibrate without clicking (same spot / vehicles on roads)"))
    s.add_argument("--run", help="use this tracked run (default: the latest, else track first)")
    s.add_argument("--frames", type=int, default=900, help="frames to track if there is no run")
    s.add_argument("--replace", action="store_true",
                   help="write over the existing calibration (a backup is kept)")
    s.set_defaults(func=cmd_autocalibrate)

    s = common(sub.add_parser("roi", help="draw a region-of-interest polygon on the video"))
    s.add_argument("--frame", type=int, default=None, help="default: the calibration frame")
    s.add_argument("--out")
    s.set_defaults(func=cmd_roi)

    s = common(sub.add_parser("track", help="stage 1 only: detect + track -> raw_tracks.csv"))
    s.add_argument("--run-name")
    s.add_argument("--max-frames", type=int, help="process only this many frames (testing)")
    s.set_defaults(func=cmd_track)

    s = sub.add_parser("composite", help="several clips of one place on one plan")
    s.add_argument("configs", nargs="+", help="site configs (one per clip)")
    s.add_argument("--out", "-o", required=True)
    s.add_argument("--run", help="run name in each site (default: the latest processed)")
    s.add_argument("--title")
    s.add_argument("--subtitle")
    s.add_argument("--only-ok", action="store_true", help="leave out calibrations graded rough")
    s.add_argument("--height", type=int, default=3000, help="plan height in px")
    s.set_defaults(func=cmd_composite)

    s = common(sub.add_parser("privacy", help="detect every person / vehicle of a run for "
                                              "blurring (older runs)"))
    s.add_argument("--run", required=True)
    s.set_defaults(func=cmd_privacy)

    s = common(sub.add_parser("process", help="stages 3-6 on an existing run (fast re-run)"))
    s.add_argument("--run", default="latest", help="run folder, run name, or 'latest'")
    s.add_argument("--debug-video", action="store_true")
    s.set_defaults(func=cmd_process)

    s = common(sub.add_parser("run", help="full pipeline: track + process + exports + visuals"))
    s.add_argument("--run-name")
    s.add_argument("--max-frames", type=int)
    s.add_argument("--debug-video", action="store_true", help="also write debug.mp4")
    s.set_defaults(func=cmd_run)

    s = common(sub.add_parser("stabilize", help="camera motion for an existing run "
                                                "(handheld, tracked without it)"))
    s.add_argument("--run", default="latest", help="run folder, run name, or 'latest'")
    s.set_defaults(func=cmd_stabilize)

    s = common(sub.add_parser("rhino", help="Rhino .3dm + Grasshopper script for a run"))
    s.add_argument("--run", default="latest", help="run folder, run name, or 'latest'")
    s.set_defaults(func=cmd_rhino)

    s = common(sub.add_parser("calibration-picture",
                              help="the calibration of a run as a picture (points, grid, aerial)"))
    s.add_argument("--run", default="latest", help="run folder, run name, or 'latest'")
    s.add_argument("--height", type=int, default=1080, help="panel height in pixels")
    s.set_defaults(func=cmd_calibration_picture)

    s = common(sub.add_parser("process-sequence",
                              help="the whole process at one moment: footage, calibration, "
                                   "tracked, projection, tracks, flow field, density"))
    s.add_argument("--run", default="latest", help="run folder, run name, or 'latest'")
    s.add_argument("--at", type=float, default=2.0, help="seconds into the run")
    s.add_argument("--height", type=int, default=1080, help="video panel height in pixels")
    s.set_defaults(func=cmd_process_sequence)

    s = sub.add_parser("houdini-pop", help="Houdini particle scene: a site's combined field "
                                           "and each clip's field as modules")
    s.add_argument("--site", required=True, help="name, e.g. plaza")
    s.add_argument("--composite", required=True, help="the site's composite folder")
    s.add_argument("--clips", nargs="*", default=[], help="clip folders (run or results)")
    s.add_argument("--out", required=True, help="kit folder to write")
    s.add_argument("--no-build", action="store_true", help="only the files, no .hip")
    s.set_defaults(func=cmd_houdini_pop)

    s = sub.add_parser("live-pass-export", help="a saved Live pass as a run: points, GeoJSON, "
                                                "camera path, Houdini curves + field; Rhino, POP kit")
    s.add_argument("run", help="the run the pass was saved under (runs/<site>/<run>)")
    s.add_argument("id", help="the pass id, e.g. 20261029-101500")
    s.add_argument("--rhino", action="store_true", help="also the Rhino model (.3dm + Grasshopper)")
    s.add_argument("--houdini", action="store_true", help="also the Houdini POP kit")
    s.add_argument("--no-build", action="store_true", help="the POP kit's files only, no .hip")
    s.set_defaults(func=cmd_live_pass_export)

    s = sub.add_parser("live-layers-export", help="the Live collage's layers (all, or one camera's) as a run: "
                                                  "points, GeoJSON, camera path, Houdini curves + field; "
                                                  "Rhino, POP kit")
    s.add_argument("dir", help="the export folder the app prepared (<run>/live/exports/<stamp>-<name>/, "
                               "with layers.json and layers/)")
    s.add_argument("--rhino", action="store_true", help="also the Rhino model (.3dm + Grasshopper)")
    s.add_argument("--houdini", action="store_true", help="also the Houdini POP kit")
    s.add_argument("--no-build", action="store_true", help="the POP kit's files only, no .hip")
    s.set_defaults(func=cmd_live_layers_export)

    s = sub.add_parser("live-sheet", help="a saved Live pass's performance sheet: the finished drawing "
                                          "(the Live view, plans, sections) over the score of how it was "
                                          "played; PDF + PNG, black and light, in the pass's folder")
    s.add_argument("run", help="the run the pass was saved under (runs/<site>/<run>)")
    s.add_argument("id", help="the pass id, e.g. 20261029-101500")
    s.add_argument("--layout", default="C", choices=["A", "B", "C"],
                   help="the top half: A view | plan | 2 sections, B view | the drawing at 1/3, 2/3 and the "
                        "end, C a wider view | plan over a section (default C)")
    s.add_argument("--theme", default="light", choices=["both", "black", "light"])
    s.add_argument("--no-view", action="store_true",
                   help="no browser: the view rendered before (sheet_view.png), if any")
    s.set_defaults(func=cmd_live_sheet)

    s = common(sub.add_parser("package", help="frameless images + labels/metrics for a run"))
    s.add_argument("--run", default="latest", help="run folder, run name, or 'latest'")
    s.add_argument("--no-plate", action="store_true", help="skip the complete plate")
    s.set_defaults(func=cmd_package)

    s = sub.add_parser("locate", help="camera position from a place name (OpenStreetMap)")
    s.add_argument("place", help='e.g. "Urbanest Canary Wharf"')
    s.add_argument("--floor", type=float, help="floor you filmed from (height = floor x 3.1 m)")
    s.add_argument("--floor-height", type=float, default=3.1, help="m per floor")
    s.add_argument("--height", type=float, help="camera height above the ground in m instead")
    s.add_argument("--ground-m", type=float, default=0.0,
                   help="building ground level above the mapped ground (m)")
    s.add_argument("--pick", type=int, default=1, help="use the Nth search result")
    s.add_argument("--limit", type=int, default=5)
    s.add_argument("--config", "-c", help="site YAML to put camera_position into")
    s.add_argument("--write", action="store_true", help="write it into --config")
    s.set_defaults(func=cmd_locate)

    s = sub.add_parser("ui", help="open the local web interface")
    s.add_argument("--root", default=".", help="project folder (with sites/, runs/)")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--no-browser", action="store_true")
    s.set_defaults(func=cmd_ui)

    s = sub.add_parser("app", help="open the desktop app (the whole workflow in one window)")
    s.add_argument("--root", default=".", help="project folder (with sites/, runs/)")
    s.add_argument("--port", type=int, default=0, help="local port (default: any free one)")
    s.add_argument("--no-window", action="store_true", help="only the server (for a browser)")
    s.add_argument("--shortcut", action="store_true",
                   help="make Desktop and Start-menu shortcuts to the app, then exit")
    s.add_argument("--debug", action="store_true", help="window with developer tools")
    s.add_argument("--native-frame", action="store_true",
                   help="keep the Windows title bar instead of the app's own")
    s.set_defaults(func=cmd_app)

    s = sub.add_parser("houdini-build", help="build the scene of an existing Houdini kit")
    s.add_argument("kit", help="kit folder (modules.json + hou_pop_build.py)")
    s.set_defaults(func=cmd_houdini_build)

    s = common(sub.add_parser("live-export", help="scene package of a processed run for the "
                                                  "Live view (<run>/live/)"), False)
    s.add_argument("run_dir", help="run folder (with data/points.csv)")
    s.add_argument("--config", "-c", help="site YAML (default: sites/<site>.yaml of the run)")
    s.set_defaults(func=cmd_live_export)

    s = common(sub.add_parser("debug-video", help="render debug.mp4 for an existing run"))
    s.add_argument("--run", default="latest")
    s.add_argument("--no-blur", action="store_true", help="do NOT blur people (careful)")
    s.add_argument("--boxes", action="store_true",
                   help="debug style: boxes, IDs and rejected tracks (default: clean trails)")
    s.add_argument("--no-stabilize-output", action="store_true",
                   help="keep the original (moving) camera framing")
    s.add_argument("--trail-s", type=float, help="only show the last N seconds of each trail")
    s.set_defaults(func=cmd_debug_video)

    s = common(sub.add_parser("groundtruth", help="check accuracy on a walk along a known line"))
    s.add_argument("--video", required=True)
    s.add_argument("--start", type=float, nargs=2, metavar=("E", "N"), help="surveyed point A")
    s.add_argument("--end", type=float, nargs=2, metavar=("E", "N"), help="surveyed point B")
    s.add_argument("--length", type=float, help="known A-B length in m (if no coordinates)")
    s.add_argument("--stopwatch", type=float, help="walking time A->B you timed, in s (optional)")
    s.add_argument("--track-id", type=int, help="default: the longest person track")
    s.add_argument("--run-name")
    s.set_defaults(func=cmd_groundtruth)

    s = sub.add_parser("live-hub", help="the phone controller's hub on its own "
                                        "(Live mode in the app starts it)")
    s.add_argument("--port", type=int, default=8766, help="phone page and socket (HTTPS, LAN)")
    s.add_argument("--view-port", type=int, default=8767,
                   help="the Live view's socket (this machine only)")
    s.add_argument("--host", default="0.0.0.0",
                   help="interface for the phone (default: all; 127.0.0.1 = this machine only)")
    s.add_argument("--new-code", action="store_true",
                   help="make a new pairing code (phones scan the QR code again)")
    s.set_defaults(func=cmd_live_hub)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
