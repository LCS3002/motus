"""Interactive calibration: pick matching ground points in a video frame and a GeoTIFF.

Controls (shown in the window title too):
  left click            add a point (video first, then the matching point on the map)
  right click / u       undo the last click
  scroll wheel          zoom around the cursor (toolbar zoom/pan also work)
  enter                 finish and save
  escape                quit without saving
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from .geo import GeoRaster, load_geotiff
from .homography import Homography, fit_homography, report
from .lens import Lens
from .video import read_frame


def load_points_csv(path) -> tuple[np.ndarray, np.ndarray]:
    px, world = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            px.append([float(row["u"]), float(row["v"])])
            world.append([float(row["E"]), float(row["N"])])
    return np.array(px), np.array(world)


def save_points_csv(path, px, world) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "u", "v", "E", "N"])
        for i, ((u, v), (e, n)) in enumerate(zip(px, world), 1):
            w.writerow([i, f"{u:.2f}", f"{v:.2f}", f"{e:.3f}", f"{n:.3f}"])


def _scroll_zoom(ax, event, base=1.4):
    if event.inaxes is not ax or event.xdata is None:
        return
    scale = 1 / base if event.button == "up" else base
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    cx, cy = event.xdata, event.ydata
    ax.set_xlim(cx - (cx - x0) * scale, cx + (x1 - cx) * scale)
    ax.set_ylim(cy - (cy - y0) * scale, cy + (y1 - cy) * scale)
    ax.figure.canvas.draw_idle()


def pick_points(frame_rgb: np.ndarray, raster: GeoRaster, init_px=None, init_world=None,
                warn_m: float = 0.5, fit_fn=None, min_live: int = 5, camera_xy=None,
                view_m: float = 90.0, on_camera=None):
    """Open the two-pane picker. Returns (px, world) arrays, or None if cancelled.

    camera_xy: the known camera position: the map opens around it (view_m each way), marked.
    on_camera(E, N): called when the user marks where they stood (key c, then a map click)."""
    import matplotlib.pyplot as plt

    for k in ("keymap.back", "keymap.forward"):        # c / v: our keys, not view history
        plt.rcParams[k] = [x for x in plt.rcParams[k] if x not in ("c", "v")]

    px: list[list[float]] = [list(p) for p in (init_px if init_px is not None else [])]
    world: list[list[float]] = [list(p) for p in (init_world if init_world is not None else [])]
    state = {"done": False, "cancel": False}

    fig, (axv, axm) = plt.subplots(1, 2, figsize=(17, 8.5))
    fig.subplots_adjust(left=0.03, right=0.99, top=0.9, bottom=0.04, wspace=0.08)
    axv.imshow(frame_rgb)
    axv.set_title("VIDEO FRAME")
    axm.imshow(raster.image, extent=raster.extent, interpolation="bilinear")
    axm.set_title(f"MAP  (EPSG:{raster.epsg})")
    axm.ticklabel_format(useOffset=False, style="plain")
    cam_art: list = []

    def draw_camera(ce, cn, label):
        for a in cam_art:
            a.remove()
        cam_art.clear()
        cam_art.append(axm.plot(ce, cn, "^", color="white", mec="black", ms=11, mew=1.2)[0])
        cam_art.append(axm.annotate(label, (ce, cn), xytext=(8, -14), textcoords="offset points",
                                    color="white", fontsize=9, weight="bold"))

    if camera_xy is not None:      # start where the camera was, not on the whole map
        ce, cn = (float(v) for v in camera_xy)
        draw_camera(ce, cn, "camera (approx.)  - press c to set")
        axm.set_xlim(max(ce - view_m, raster.left), min(ce + view_m, raster.right))
        axm.set_ylim(max(cn - view_m, raster.bottom), min(cn + view_m, raster.top))
    artists: list = []

    def redraw():
        state["confirm"] = False  # any edit re-arms the "fewer than 6 pairs" check
        for a in artists:
            a.remove()
        artists.clear()
        if len(px) > len(world) and state.get("guess"):
            ge, gn = state["guess"]  # where the fit expects this point: just refine the click
            artists.append(axm.plot(ge, gn, "o", mfc="none", mec="yellow", ms=24, mew=1.8)[0])
        else:
            state["guess"] = None
        for i, p in enumerate(px):
            artists.append(axv.plot(*p, "+", color="#ff2d55", ms=16, mew=1.6)[0])
            artists.append(axv.annotate(str(i + 1), p, xytext=(6, 6), textcoords="offset points",
                                        color="#ff2d55", fontsize=11, weight="bold"))
        for i, p in enumerate(world):
            artists.append(axm.plot(*p, "+", color="#00e5ff", ms=16, mew=1.6)[0])
            artists.append(axm.annotate(str(i + 1), p, xytext=(6, 6), textcoords="offset points",
                                        color="#00e5ff", fontsize=11, weight="bold"))
        n = min(len(px), len(world))
        status = f"{n} pairs"
        if len(px) > len(world):
            status += f" | now click point {len(px)} on the MAP"
        else:
            status += f" | now click point {len(px) + 1} on the VIDEO"
        if n >= min_live:
            try:
                h = fit_fn(np.array(px[:n]), np.array(world[:n]))
                proj = h.to_world(np.array(px[:n]))
                for (e, nn), (pe, pn) in zip(world[:n], proj):
                    if np.isfinite(pe):
                        artists.append(axm.plot([e, pe], [nn, pn], "-", color="yellow", lw=1.2)[0])
                        artists.append(axm.plot(pe, pn, "x", color="yellow", ms=8)[0])
                flag = "  <-- TOO HIGH" if h.rmse_m > warn_m else ""
                status += f" | RMSE {h.rmse_m:.2f} m{flag}"
                worst = max(h.points, key=lambda p: p["error_m"] if p["error_m"] is not None
                            else float("inf"))
                we = worst["error_m"]
                status += f" | worst #{worst['id']} " + (f"{we:.2f} m" if we is not None
                                                          else "behind camera")
                if h.spread < 0.3:
                    status += (" | POINTS ALONG ONE LINE: add some to the left / right and "
                               "nearer / further")
            except Exception as e:  # noqa: BLE001
                status += f" | fit failed: {e}"
        if state.get("set_cam"):
            status = "click on the MAP where you were standing (the camera)"
        fig.suptitle(status + "\nleft-click add · right-click a point: delete pair · u undo · "
                     "scroll zoom · " + ("c set where you stood · " if on_camera else "")
                     + "enter save · esc cancel", fontsize=11)
        fig.canvas.draw_idle()

    def on_click(ev):
        tb = getattr(fig.canvas.manager, "toolbar", None)
        if tb is not None and getattr(tb, "mode", ""):
            return  # zoom/pan tool active
        if ev.button == 3:
            # right-click on an existing point deletes that pair; elsewhere undoes the last click
            if ev.inaxes in (axv, axm) and ev.xdata is not None:
                pts = px if ev.inaxes is axv else world
                n = min(len(px), len(world))
                if n:
                    disp = ev.inaxes.transData.transform(np.asarray(pts[:n], float))
                    d = np.hypot(disp[:, 0] - ev.x, disp[:, 1] - ev.y)
                    k = int(np.argmin(d))
                    if d[k] < 15:
                        del px[k]
                        del world[k]
                        redraw()
                        return
            undo()
            return
        if ev.button != 1 or ev.xdata is None:
            return
        if state.get("set_cam"):
            if ev.inaxes is axm:
                state["set_cam"] = False
                on_camera(float(ev.xdata), float(ev.ydata))
                draw_camera(ev.xdata, ev.ydata, "camera (you)")
                redraw()
            return
        if ev.inaxes is axv and len(px) == len(world):
            fh, fw = frame_rgb.shape[:2]
            if not (0 <= ev.xdata < fw and 0 <= ev.ydata < fh):
                return                       # outside the picture
            px.append([ev.xdata, ev.ydata])
            _guide_map(ev.xdata, ev.ydata)
        elif ev.inaxes is axm and len(px) == len(world) + 1:
            world.append([ev.xdata, ev.ydata])
            if state.get("map_view"):  # back to the overview after a guided click
                axm.set_xlim(*state["map_view"][0])
                axm.set_ylim(*state["map_view"][1])
                state["map_view"] = None
        redraw()

    def _guide_map(u, v):
        """With >= 4 good pairs, zoom the map to where the video click should be."""
        n = min(len(px), len(world))
        if n < max(min_live - 1, 3):
            return
        try:
            h = fit_fn(np.array(px[:n]), np.array(world[:n]))
            if not np.isfinite(h.rmse_m) or h.rmse_m > 5:
                return
            e, nn = h.to_world(np.array([[u, v]]))[0]
        except Exception:  # noqa: BLE001
            return
        if not np.isfinite(e):
            return
        state["map_view"] = (axm.get_xlim(), axm.get_ylim())
        state["guess"] = (e, nn)
        r = max(8.0, 4 * h.rmse_m + 6)
        axm.set_xlim(e - r, e + r)
        axm.set_ylim(nn - r, nn + r)

    def undo():
        if len(px) > len(world):
            px.pop()
        elif world:
            world.pop()
        redraw()

    def on_key(ev):
        if ev.key in ("u", "backspace"):
            undo()
        elif ev.key == "c" and on_camera is not None:
            state["set_cam"] = not state.get("set_cam")
            redraw()
        elif ev.key == "enter":
            n = min(len(px), len(world))
            if n < 6 and not state.get("confirm"):
                state["confirm"] = True
                fig.suptitle(f"Only {n} pairs - at least 6 well-spread pairs are needed for a "
                             "trustworthy fit.\nAdd more points, or press Enter again to save "
                             "anyway.", fontsize=11, color="#d00")
                fig.canvas.draw_idle()
                return
            state["done"] = True
            plt.close(fig)
        elif ev.key == "escape":
            state["cancel"] = True
            plt.close(fig)

    def on_scroll(ev):
        _scroll_zoom(axv, ev)
        _scroll_zoom(axm, ev)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    redraw()
    plt.show()
    if state["cancel"]:
        return None
    n = min(len(px), len(world))
    return np.array(px[:n]), np.array(world[:n])


def warp_frame_to_map(frame_bgr: np.ndarray, h: Homography, raster: GeoRaster,
                      max_dim: int = 3000) -> np.ndarray:
    """Project the video frame onto the map raster grid (RGB, NaN-free, black where invalid)."""
    H_img, W_img = raster.image.shape[:2]
    s = min(1.0, max_dim / max(H_img, W_img))
    out_w, out_h = int(W_img * s), int(H_img * s)
    cols, rows = np.meshgrid(np.arange(out_w) + 0.5, np.arange(out_h) + 0.5)
    world = raster.pixel_to_world(np.column_stack([cols.ravel() / s, rows.ravel() / s]))
    if h.terrain is not None:              # the ground follows the terrain model
        uv = h.to_pixel(world)
        uv[~np.isfinite(uv).all(axis=1)] = -1e6
    else:
        local = world - np.asarray(h.origin)
        Hinv = np.linalg.inv(h.H)
        ph = np.column_stack([local, np.ones(len(local))]) @ Hinv.T
        valid = np.sign(ph[:, 2]) == np.sign(h.w_sign)
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = ph[:, :2] / ph[:, 2:3]
        uv[~valid] = -1e6
    mapx = uv[:, 0].reshape(out_h, out_w).astype(np.float32)
    mapy = uv[:, 1].reshape(out_h, out_w).astype(np.float32)
    warped = cv2.remap(frame_bgr, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)


def save_check_image(path, frame_bgr, h: Homography, raster: GeoRaster) -> None:
    """Side-by-side: map with the warped video blended on top, plus point errors."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    warped = warp_frame_to_map(frame_bgr, h, raster)
    base = cv2.resize(raster.image, (warped.shape[1], warped.shape[0]), interpolation=cv2.INTER_AREA)
    mask = warped.sum(axis=2) > 0
    blend = base.copy()
    blend[mask] = (0.5 * base[mask] + 0.5 * warped[mask]).astype(np.uint8)

    pts = np.array([[p["E"], p["N"]] for p in h.points])
    proj = h.to_world(np.array([[p["u"], p["v"]] for p in h.points]))
    pad = max(20.0, 0.3 * np.ptp(pts, axis=0).max())
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, img, title in ((axes[0], blend, "video warped onto map (50% blend)"),
                           (axes[1], base, "calibration points: + clicked, x projected")):
        ax.imshow(img, extent=raster.extent)
        ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
        ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)
        ax.set_title(title)
        ax.ticklabel_format(useOffset=False, style="plain")
    ax = axes[1]
    for p, (pe, pn) in zip(h.points, proj):
        c = "#00e5ff" if p["inlier"] else "#ff2d55"
        ax.plot(p["E"], p["N"], "+", color=c, ms=14, mew=2)
        if np.isfinite(pe):
            ax.plot([p["E"], pe], [p["N"], pn], "-", color="yellow", lw=1.5)
            ax.plot(pe, pn, "x", color="yellow", ms=9, mew=2)
        ax.annotate((f"{p['id']}: {p['error_m']:.2f} m" if p['error_m'] is not None else f"{p['id']}: behind camera"), (p["E"], p["N"]), xytext=(8, 8),
                    textcoords="offset points", color=c, fontsize=10, weight="bold")
    fig.suptitle(f"RMSE {h.rmse_m:.3f} m · max {h.max_error_m:.3f} m"
                 + (f" · leave-one-out RMSE {h.loo_rmse_m:.3f} m" if h.loo_rmse_m else ""))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_calibration(video: Path, geotiff: Path, out_json: Path, frame: int = 0,
                    lens: Lens | None = None, points_csv: Path | None = None,
                    interactive: bool = True, ransac_thresh_m: float = 1.0,
                    warn_m: float = 0.5, camera_prior: dict | None = None,
                    log=print, terrain=None, hide=None) -> Homography:
    frame_bgr = read_frame(video, frame)
    size = (frame_bgr.shape[1], frame_bgr.shape[0])
    camera_set: dict = {}
    camera_prior = dict(camera_prior) if camera_prior else {}   # empty until c sets one
    from .posefit import camera_calibration

    def fit_fn(a, b):
        if camera_prior:
            return camera_calibration(a, b, size, camera_prior, terrain=terrain)
        return fit_homography(a, b, ransac_thresh_m=ransac_thresh_m)
    if camera_prior:
        min_live = 3
        log(f"camera position known (E {camera_prior['E']}, N {camera_prior['N']}, "
            f"{camera_prior['height_m']} m up): fitting a physical camera, 3+ pairs needed")
        if terrain is not None:
            log("ground follows the terrain model: click on any level (terrace, steps, street)")
    else:
        min_live = 5
    if lens is not None:
        frame_bgr = lens.undistort_image(frame_bgr)
    raster = load_geotiff(geotiff, warn=log)
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    default_csv = out_json.with_name(out_json.stem + "_points.csv")

    init_px = init_world = None
    src_csv = points_csv or (default_csv if default_csv.exists() else None)
    if src_csv:
        init_px, init_world = load_points_csv(src_csv)
        log(f"loaded {len(init_px)} point pairs from {src_csv}")

    if interactive:
        import matplotlib.pyplot as plt  # noqa: F401  (ensure an interactive backend loads)

        cam_xy = (camera_prior["E"], camera_prior["N"]) if camera_prior else None

        def on_camera(e, n):     # the user marked where they stood: trust it to a few metres
            if not camera_prior:     # no position in the config: a handheld phone at eye level
                camera_prior.update(height_m=1.5, height_tol_m=1.0)
            camera_prior.update(E=round(e, 1), N=round(n, 1), position_tol_m=min(
                float(camera_prior.get("position_tol_m", 15.0)), 4.0))
            camera_set["by_user"] = True
            log(f"camera position set on the map: E {e:.1f}, N {n:.1f}")

        res = pick_points(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), raster, init_px,
                          init_world, warn_m, fit_fn=fit_fn, min_live=min_live,
                          camera_xy=cam_xy, on_camera=on_camera)
        if res is None:
            raise SystemExit("Calibration cancelled; nothing saved.")
        px, world = res
    else:
        if init_px is None:
            raise ValueError("Non-interactive calibration needs --points-csv")
        px, world = init_px, init_world

    if camera_prior and len(px) < 3:
        # too few for a camera on its own: the people / vehicles in the footage finish it
        save_points_csv(default_csv, px, world)
        log(f"{len(px)} point pair(s) saved to {default_csv}. With the camera position known, "
            "the walking people or the traffic can supply the rest:\n  groundtrack "
            "autocalibrate --config <this site>.yaml")
        return None

    if camera_prior:
        from .posefit import camera_calibration

        h = camera_calibration(px, world, size, camera_prior, terrain=terrain,
                               consensus=True)
        if getattr(h, "consensus_dropped", None):
            log(f"clicks {h.consensus_dropped} disagree with the rest (one camera explains the "
                f"other {len(px) - len(h.consensus_dropped)} to within 15 px): left out")
        h.undistorted, h.crs = lens is not None, f"EPSG:{raster.epsg}"
    else:
        h = fit_homography(px, world, ransac_thresh_m=ransac_thresh_m, image_size=size,
                           undistorted=lens is not None, crs=f"EPSG:{raster.epsg}")
    h.reference_frame = int(frame)
    if camera_set.get("by_user"):
        h.camera_prior_set = dict(camera_prior)        # cli writes it into the site config
    save_points_csv(default_csv, px, world)
    h.save(out_json)
    if hide is not None:       # the saved check images: people and number plates blurred
        from .debug_video import blur_box

        frame_bgr = frame_bgr.copy()
        for r in hide[hide["frame"] == int(frame)].itertuples():
            blur_box(frame_bgr, r.x1, r.y1, r.x2, r.y2)
    check_png = out_json.with_name(out_json.stem + "_check.png")
    save_check_image(check_png, frame_bgr, h, raster)
    from .overlay_check import map_in_video

    mp, ins = map_in_video(frame_bgr, h, raster)
    blend = frame_bgr.copy()
    blend[ins] = (0.5 * frame_bgr[ins] + 0.5 * mp[ins]).astype(np.uint8)
    cv2.imwrite(str(out_json.with_name(out_json.stem + "_check_video.jpg")), blend)
    log(report(h, warn_m))
    if getattr(h, "camera_moved_m", None):
        log(f"NOTE: the clicks put the camera {h.camera_moved_m:.0f} m from camera_position in "
            "the site config: update it there if this fit looks right")
    if getattr(h, "camera_params", None):
        cp = h.camera_params
        log(f"Fitted camera: direction {cp['yaw_deg']:.1f} deg, tilt {cp['tilt_deg']:.1f} deg, "
            f"roll {cp['roll_deg']:.1f} deg, horizontal FOV {cp['hfov_deg']:.1f} deg, "
            f"{cp['height_m']:.1f} m up at E {cp['E']:.1f}, N {cp['N']:.1f}")
    log(f"saved {out_json}\n      {default_csv}\n      {check_png}  <- open this and check "
        "that kerbs and markings line up")
    return h


def pick_roi(video: Path, frame: int = 0, out_json: Path | None = None, log=print):
    """Click a polygon on a video frame; enter to finish. Saves [[x, y], ...] JSON."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    img = cv2.cvtColor(read_frame(video, frame), cv2.COLOR_BGR2RGB)
    pts: list[list[float]] = []
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(img)
    patch = Polygon(np.zeros((1, 2)), closed=True, fill=True, alpha=0.25, color="#00e5ff")
    ax.add_patch(patch)
    (line,) = ax.plot([], [], "o-", color="#00e5ff")
    done = {"ok": False}

    def update():
        a = np.array(pts) if pts else np.zeros((0, 2))
        line.set_data(a[:, 0] if len(a) else [], a[:, 1] if len(a) else [])
        if len(a) >= 3:
            patch.set_xy(a)
        ax.set_title(f"ROI: {len(pts)} vertices. Left-click add, right-click undo, enter save.\n"
                     "Exclude sky, far/blurry areas and reflections.")
        fig.canvas.draw_idle()

    def on_click(ev):
        tb = getattr(fig.canvas.manager, "toolbar", None)
        if (tb is not None and getattr(tb, "mode", "")) or ev.inaxes is not ax:
            return
        if ev.button == 1:
            pts.append([round(ev.xdata, 1), round(ev.ydata, 1)])
        elif ev.button == 3 and pts:
            pts.pop()
        update()

    def on_key(ev):
        if ev.key == "enter":
            done["ok"] = True
            plt.close(fig)
        elif ev.key == "escape":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    update()
    plt.show()
    if not done["ok"] or len(pts) < 3:
        raise SystemExit("ROI cancelled (need >= 3 vertices).")
    if out_json:
        Path(out_json).write_text(json.dumps(pts), encoding="utf-8")
        log(f"saved {out_json}. In the site config set:\n  detection:\n    roi: {out_json}")
    log("or paste inline:\n  detection:\n    roi: " + json.dumps(pts))
    return pts
