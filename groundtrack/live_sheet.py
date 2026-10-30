"""Live: the performance sheet (plans/live_controller/09; decided 27-29 Oct 2026).

One 2:1 landscape sheet per pass, PDF (vector) and PNG (300 dpi), in black and in light, saved in
the pass's own folder (inside runs/: it shows tracks):

    <start run>/live/passes/<id>/sheet_view.png           the Live view at print size (fixed preset)
    <start run>/live/passes/<id>/sheet_<theme>.pdf / .png  the sheet
    <start run>/live/passes/<id>/sheet.json               what it drew: every mark with its time

Top half, the finished drawing: the large view (the Live view's look, rendered by the view itself
from a fixed preset: live.js sheetView) beside plans and sections of the pass's saved drawing as
vector line work (the speed ramp, log 1-40 m/s, as drawn; the machine's guesses dashed). LAYOUTS
holds the options for where they go.
Bottom half, the score: the whole performance on one time axis (performance time t, the
recording's clock), squeezed to the width. Shared, once: the clip clock T (play, pause, scrubs
show as its flats and jumps). Then one group per player, like instruments in an orchestra score:
turn, tilt and roll of the phone as three waves (the raw pose: turn from its first heading in the
pass, tilt and roll against gravity), SPACE held as a bar, level and zero as marks, gain as a
stepped line. Then the machine's answer per camera: placed, guessed (dashes), lost, and the new
identities as ticks. A pass without a recording (format v1) gets the top half and a note.

The view needs a browser: `groundtrack live-sheet` serves the app on this machine only and opens
/live?sheet=<id> in headless Chrome or Edge (render_view); the app's Sheet button renders it in a
hidden frame of the Live view. Both post it to /api/live/sheet (request). No video is drawn (no
people) and nothing leaves the machine.
"""

from __future__ import annotations

import colorsys
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from .live_passes import (
    CLOCK_PLAYING,
    SOURCES,
    _folder,
    parse_files,
    read_pass,
    read_recording,
)

SHEET_MM = (420.0, 210.0)                  # 2:1 landscape
DPI = 300
MARGIN = 7.0
VIEW_FILE = "sheet_view.png"
INFO_FILE = "sheet.json"
SPEED_LO, SPEED_HI = 1.0, 40.0             # the speed ramp, log (colors.cmap)
MOUSE = SOURCES.index("mouse")
PNG = b"\x89PNG\r\n\x1a\n"
MAX_VIEW_BYTES = 80 * 1024 * 1024
FONT_DIR = Path(__file__).parent / "desktop_web" / "fonts"
FONT = "Spline Sans Mono"
TINTS = ("#9b82d9", "#8b8fe6", "#b787dc", "#c98bd2")      # live/spots.js: the violet hue, per camera

THEMES = {
    "black": {"bg": "#000000", "fg": "#ffffff", "fg2": (1, 1, 1, 0.72), "muted": (1, 1, 1, 0.5),
              "faint": (1, 1, 1, 0.32), "rule": (1, 1, 1, 0.22), "staff": (1, 1, 1, 0.13),
              "guide": (1, 1, 1, 0.16), "guess": (1, 1, 1, 0.85), "lost": (1, 1, 1, 0.45),
              "contour": (1, 1, 1, 0.12), "ghost": (1, 1, 1, 0.5), "violet": "#9b82d9", "light": False},
    "light": {"bg": "#ffffff", "fg": "#000000", "fg2": (0, 0, 0, 0.72), "muted": (0, 0, 0, 0.5),
              "faint": (0, 0, 0, 0.32), "rule": (0, 0, 0, 0.22), "staff": (0, 0, 0, 0.12),
              "guide": (0, 0, 0, 0.16), "guess": (0, 0, 0, 0.5), "lost": (0, 0, 0, 0.4),
              "contour": (0, 0, 0, 0.11), "ghost": (0, 0, 0, 0.5), "violet": "#4b2e83", "light": True},
}

# where the view, the plan and the sections go in the top half (mm in the band, x from the left,
# y from the band's top; the band is SHEET_MM[0] - 2 MARGIN wide): the user picks one
LAYOUTS = {
    "A": "strip: view 2:1 | plan | two sections stacked",
    "B": "sequence: view 2:1 | three plans, the drawing as it stood at 1/3, 2/3 and the end of the "
         "performance, each over its section A-A (one scale)",
    "C": "view first: view 3:1 | plan over section A-A",
}
DEFAULT_LAYOUT = "C"                       # the user's pick (with the light theme)
DEFAULT_THEMES = ("light",)


# ----------------------------------------------------------------------------- poses
def quat_matrix(q: np.ndarray) -> np.ndarray:
    """[w, x, y, z] (n x 4) -> rotation matrices (n x 3 x 3); NaN rows stay NaN."""
    q = np.asarray(q, float).reshape(-1, 4)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)


def pose_angles(R: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """world_from_camera (OpenCV axes, z up) -> heading (deg, clockwise from north), tilt (+ up),
    roll (+ clockwise), as the view's HUD reads them (live/pose.js cameraDelta)."""
    R = np.asarray(R, float).reshape(-1, 3, 3)
    f, right = R[:, :, 2], R[:, :, 0]
    yaw = np.arctan2(f[:, 0], f[:, 1])
    tilt = np.arcsin(np.clip(f[:, 2], -1, 1))
    r0 = np.stack([np.cos(yaw), -np.sin(yaw), np.zeros_like(yaw)], -1)
    d0 = np.cross(f, r0)
    roll = np.arctan2((right * d0).sum(1), (right * r0).sum(1))
    return np.degrees(yaw), np.degrees(tilt), np.degrees(roll)


# ----------------------------------------------------------------------------- the score's data
def _events_of(events, types, player=None):
    out = [e for e in events if e.get("type") in types]
    if player is not None:
        out = [e for e in out if int(e.get("player") or 0) == player]
    return out


def _player_tint(rec_cams: list, name: str, player: int) -> str:
    for c in rec_cams:
        if c.get("name") == name and isinstance(c.get("tint"), str):
            return c["tint"]
    return TINTS[(max(player, 1) - 1) % len(TINTS)]


def score_data(rec: dict) -> dict:
    """A recording (live_passes.read_recording) -> what the score draws:
    {"t_end", "duration", "clock": {t, T, playing, seeks, plays, pauses, end},
     "players": [{player, label, cameras: [(name, t0, t1)], tint, t, turn, tilt, roll, src,
                  held: [(t0, t1)], level: [t], zero: [t], gain: (t, g), gain_events: [(t, g)]}],
     "machine": [{camera, t, placed, guessed, dropped, lost, new: (t, n)}]}"""
    clock, events = rec["clock"], rec["events"]
    meta = rec.get("meta") or {}
    t_clk = clock["t"].astype(float) if len(clock) else np.zeros(1)
    t_end = float(max(meta.get("seconds") or 0, t_clk[-1] if len(clock) else 0, 1e-3))
    duration = float(max((rec.get("manifest") or {}).get("duration_s") or 0, clock["T"].max() if len(clock) else 1))
    tk = lambda k: t_clk[np.minimum(np.asarray(k, int), len(t_clk) - 1)]
    rows, names = [], []
    for name, a in rec["cameras"].items():
        if len(a):
            rows.append(a)
            names.append(np.full(len(a), name, dtype=object))
    allr = np.concatenate(rows) if rows else None
    alln = np.concatenate(names) if rows else np.zeros(0, object)
    order = np.argsort(allr["k"], kind="stable") if rows else np.zeros(0, int)

    out_players = []
    pl_ids = sorted({int(p) for p in (allr["player"] if rows else [])} |
                    {int(e.get("player") or 0) for e in _events_of(events, ("space_press", "level", "zero", "gain"))})
    for p in pl_ids:
        sel = order[allr["player"][order] == p] if rows else order
        r, n = allr[sel] if rows else None, alln[sel]
        ok = np.zeros(len(sel), bool)
        if len(sel):
            src = r["src"]
            ok = np.isfinite(r["q"]).all(1) & (src != 0) & ((src == MOUSE) if p == 0 else (src != MOUSE))
        ev_p = _events_of(events, ("space_press", "space_release", "level", "zero", "gain"), p)
        if p == 0 and not ev_p:                       # the laptop: only when it played
            if ok.sum() < 2:
                continue
            a = np.stack(pose_angles(quat_matrix(r["q"][ok])))
            if np.ptp(a, axis=1).max() < 0.05:
                continue
        t = tk(r["k"][ok]) if len(sel) else np.zeros(0)
        if ok.any():
            hd, tl, rl = pose_angles(quat_matrix(r["q"][ok]))
            turn = np.degrees(np.unwrap(np.radians(hd)))
            turn = turn - turn[0]
        else:
            turn = tl = rl = np.zeros(0)
        cams = []
        for i in range(len(sel)):
            tt = float(tk(r["k"][i]))
            if cams and cams[-1][0] == n[i]:
                cams[-1][2] = tt
            else:
                cams.append([n[i], tt, tt])
        held, open_ = [], {}
        for e in sorted(_events_of(events, ("space_press", "space_release"), p), key=lambda e: e["t"]):
            key = e.get("camera")
            if e["type"] == "space_press":
                open_.setdefault(key, float(e["t"]))
            elif key in open_:
                held.append((open_.pop(key), float(e["t"])))
        held += [(t0, t_end) for t0 in open_.values()]
        if len(sel):
            g_t, g = tk(r["k"]), r["gain"].astype(float)
            keep = np.r_[True, np.abs(np.diff(g)) > 1e-6]
            gain = (np.round(g_t[keep], 4).tolist(), np.round(g[keep], 4).tolist())
        else:
            gain = ([], [])
        first = cams[0][0] if cams else None
        srcs = sorted({SOURCES[s] if s < len(SOURCES) else "other" for s in (r["src"][ok] if len(sel) else [])})
        out_players.append({
            "player": p, "label": f"P{p}" if p else "LAPTOP", "cameras": [tuple(c) for c in cams],
            "tint": _player_tint(meta.get("cameras") or [], first, p), "src": srcs,
            "t": t, "turn": turn, "tilt": tl, "roll": rl, "held": sorted(held),
            "level": [float(e["t"]) for e in ev_p if e["type"] == "level"],
            "zero": [float(e["t"]) for e in ev_p if e["type"] == "zero"],
            "gain": gain, "gain_events": [(float(e["t"]), e.get("value")) for e in ev_p if e["type"] == "gain"]})

    machine = []
    for name, a in rec["cameras"].items():
        if not len(a):
            continue
        a = a[np.argsort(a["k"], kind="stable")]
        t = tk(a["k"])
        born = np.r_[True, np.diff(a["frame"]) != 0] & (a["new"] > 0)
        machine.append({"camera": name, "t": t, "placed": a["placed"].astype(int), "guessed": a["guessed"].astype(int),
                        "dropped": a["dropped"].astype(int), "lost": a["lost"].astype(int),
                        "new": (t[born], a["new"][born].astype(int))})
    seeks = [(float(e["t"]), float(e.get("value") if e.get("value") is not None else e.get("T") or 0))
             for e in _events_of(events, ("seek",))]
    ends = _events_of(events, ("end",))
    return {"t_end": t_end, "duration": max(duration, 1e-3),
            "clock": {"t": t_clk, "T": clock["T"].astype(float) if len(clock) else np.zeros(1),
                      "playing": (clock["flags"] & CLOCK_PLAYING) > 0 if len(clock) else np.zeros(1, bool),
                      "seeks": seeks, "plays": [float(e["t"]) for e in _events_of(events, ("play",))],
                      "pauses": [float(e["t"]) for e in _events_of(events, ("pause",))],
                      "end": float(ends[0]["t"]) if ends else None},
            "players": out_players, "machine": machine}


def value_at(t: np.ndarray, v: np.ndarray, at: float) -> float:
    """A recorded series at time `at` (between two rows everything held)."""
    i = int(np.searchsorted(t, at, side="right")) - 1
    return float(v[max(i, 0)]) if len(v) else float("nan")


# ----------------------------------------------------------------------------- the drawing's data
def drawing_data(run: Path, pass_id: str, rec: dict | None, keep: set | None = None) -> dict:
    """The pass's saved drawing (live_pass_export.pass_points) in the pass's frame (metres from the
    start package's origin) as polylines, the cameras (start, and the virtual one at its last
    recorded row) and the terrain."""
    from .live_pass_export import _project_root, pass_points

    m = read_pass(run, pass_id)
    pts, cams = pass_points(run, pass_id, _project_root(run))
    if keep is not None and len(pts):                    # a collage: the cameras whose layer it made
        pts = pts[pts["camera"].isin(keep)].reset_index(drop=True)
    o = m.get("origin") or {}
    E0, N0, Z0 = float(o.get("E", 0)), float(o.get("N", 0)), float(o.get("datum_z") or 0)
    pieces = []
    if len(pts):
        xyz = np.c_[pts["x"].to_numpy(float) - E0, pts["y"].to_numpy(float) - N0, pts["z"].to_numpy(float) - Z0]
        sp = pts["speed"].to_numpy(float)
        guess = (pts["kind"] != "solid").to_numpy()
        tp = _drawn_at(pts, rec)
        for rows in pts.groupby("track_id", sort=False).indices.values():
            if len(rows) >= 2:
                pieces.append((xyz[rows], sp[rows], guess[rows], tp[rows]))
    cam_out = []
    path = {}                                         # each camera's virtual camera over t
    if rec:
        t_clk = rec["clock"]["t"].astype(float)
        for name, a in rec["cameras"].items():
            if len(a):
                a = a[np.argsort(a["k"], kind="stable")]
                hd, tl, _ = pose_angles(a["R_wc"])
                path[name] = (t_clk[np.minimum(a["k"].astype(int), len(t_clk) - 1)], hd, tl, np.asarray(a["C"], float))
    for c in cams:
        sc = c["scene"]
        cam = sc.get("camera") or {}
        d = np.array(c["origin"]) - (E0, N0, Z0)
        C = np.asarray(cam.get("C", [0, 0, 0]), float) + d
        e = {"name": c["name"], "C": C, "yaw": float(cam.get("yaw_deg", 0)), "hfov": float(cam.get("hfov_deg", 60)),
             "R_max": float(sc.get("R_max") or 50), "terrain": _terrain(c["run"], sc, d)}
        if c["name"] in path:
            t, hd, tl, Cv = path[c["name"]]
            e["path"] = (t, hd, tl, Cv + d)
            e["virtual"] = {"C": Cv[-1] + d, "yaw": float(hd[-1]), "tilt": float(tl[-1])}
        cam_out.append(e)
    return {"pieces": pieces, "cameras": cam_out, "origin": (E0, N0, Z0), "points": len(pts),
            "guessed": int((pts["kind"] != "solid").sum()) if len(pts) else 0}


def _drawn_at(pts, rec) -> np.ndarray:
    """When each point was drawn (performance time t): the last recorded row of its camera at its
    frame (a scrub back draws a frame again: the later time). NaN without a recording."""
    out = np.full(len(pts), np.nan)
    if not rec or not len(pts):
        return out
    t_clk = rec["clock"]["t"].astype(float)
    for name, a in rec["cameras"].items():
        sel = (pts["camera"] == name).to_numpy()
        if not len(a) or not sel.any():
            continue
        f = a["frame"].astype(int)
        t = t_clk[np.minimum(a["k"].astype(int), len(t_clk) - 1)]
        last = np.full(max(int(f.max()), int(pts["frame"][sel].max())) + 1, np.nan)
        ok = f >= 0
        order = np.argsort(t[ok], kind="stable")
        last[f[ok][order]] = t[ok][order]                  # later rows overwrite: the last time
        out[sel] = last[pts["frame"][sel].to_numpy(int)]
    return out


def _cameras_at(cams: list, t_cut: float) -> list:
    """The cameras with each virtual camera as it stood at t_cut (its last row by then)."""
    out = []
    for c in cams:
        c = dict(c)
        if "path" in c:
            t, hd, tl, C = c["path"]
            j = int(np.searchsorted(t, t_cut, side="right")) - 1
            if j < 0:
                c.pop("virtual", None)
            else:
                c["virtual"] = {"C": C[j], "yaw": float(hd[j]), "tilt": float(tl[j])}
        out.append(c)
    return out


def until(pieces: list, t_cut: float) -> list:
    """The drawing as it stood at t_cut: each piece's points drawn by then (split where a later
    redraw breaks it)."""
    out = []
    for xyz, sp, g, tp in pieces:
        keep = tp <= t_cut
        if not keep.any():
            continue
        cut = np.flatnonzero(np.diff(keep.astype(int))) + 1
        for a, b in zip(np.r_[0, cut], np.r_[cut, len(keep)]):
            if keep[a] and b - a >= 2:
                out.append((xyz[a:b], sp[a:b], g[a:b], tp[a:b]))
    return out


def _terrain(run: Path, scene: dict, d: np.ndarray):
    t = scene.get("terrain") or {}
    f = Path(run) / "live" / str(t.get("file", "terrain.f32"))
    try:
        nx, ny, cell = int(t["nx"]), int(t["ny"]), float(t.get("cell", 1))
        z = np.fromfile(f, "<f4").reshape(ny, nx).astype(float)
    except (KeyError, OSError, ValueError):
        return None
    x = float(t["x0"]) + (np.arange(nx) + 0.5) * cell + d[0]
    y = float(t["y1"]) - (np.arange(ny) + 0.5) * cell + d[1]
    return {"x": x, "y": y, "z": z + d[2]}


# ----------------------------------------------------------------------------- drawing helpers
def _fonts():
    from matplotlib import font_manager

    found = False
    for p in sorted(FONT_DIR.glob("SplineSansMono-*.ttf")):
        if p.name not in {Path(f.fname).name for f in font_manager.fontManager.ttflist}:
            font_manager.fontManager.addfont(str(p))
        found = True
    return FONT if found else "DejaVu Sans Mono"


def _dark(hex_: str, light: bool) -> str:
    """A player's tint: as it is on black; on light, the same hue dark (as #4b2e83)."""
    import matplotlib.colors as mc

    r, g, b = mc.to_rgb(hex_)
    if not light:
        return mc.to_hex((r, g, b))
    h, _, s = colorsys.rgb_to_hls(r, g, b)
    return mc.to_hex(colorsys.hls_to_rgb(h, 0.34, min(s, 0.5)))


LIGHT_GREY = 0.45                     # on the light view: what is white on black (guesses, the start camera)


def light_view(rgb: np.ndarray) -> np.ndarray:
    """The Live view's render (lines on black, uint8 h x w x 3) -> the same drawing on white, the
    light look: each pixel read as a colour over black (its coverage = its brightness against the
    colour's own), then laid on white - the speed ramp as drawn (its hue gives back its colour), the
    violet tints dark (as the score's players), grey and white (ground, contours, guesses, start
    camera) as light to mid grey."""
    import matplotlib.colors as mc

    P = np.asarray(rgb, np.float32)[..., :3] / 255.0
    hsv = mc.rgb_to_hsv(P)
    hue, sat, val = hsv[..., 0] * 360, hsv[..., 1], hsv[..., 2]
    C = np.full(P.shape, LIGHT_GREY, np.float32)
    top = np.full(val.shape, 1.0, np.float32)            # the colour's own brightness (max channel)
    cmap, _ = _ramp()
    rc = cmap(np.linspace(0, 1, 512))[:, :3].astype(np.float32)
    rh = mc.rgb_to_hsv(rc)[:, 0] * 360                   # 231 (slow) .. 6 (fast): falling
    ramp = (sat >= 0.55) & ((hue <= 240) | (hue >= 330))
    if ramp.any():
        h = np.where(hue[ramp] >= 330, 0.0, hue[ramp])
        j = np.interp(h, rh[::-1], np.arange(len(rh))[::-1]).round().astype(int)
        C[ramp] = rc[j]
        top[ramp] = rc[j].max(1)
    violet = ~ramp & (sat >= 0.22) & (hue >= 225) & (hue < 330)
    if violet.any():
        tints = np.array([mc.to_rgb(t) for t in TINTS], np.float32)
        th_ = mc.rgb_to_hsv(tints)[:, 0] * 360
        j = np.abs(hue[violet][:, None] - th_[None, :]).argmin(1)
        dark = np.array([mc.to_rgb(_dark(t, True)) for t in TINTS], np.float32)
        C[violet] = dark[j]
        top[violet] = tints[j].max(1)
    a = np.clip(val / top, 0, 1)[..., None]
    return np.round((1 - a * (1 - C)) * 255).astype(np.uint8)


class _Sheet:
    """A figure laid out in mm (y from the top)."""

    def __init__(self, theme: dict, font: str):
        import matplotlib.pyplot as plt

        self.th, self.font = theme, font
        w, h = SHEET_MM
        self.fig = plt.figure(figsize=(w / 25.4, h / 25.4), dpi=DPI, facecolor=theme["bg"])

    def ax(self, x, y, w, h, frame=True):
        W, H = SHEET_MM
        a = self.fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
        a.set_facecolor("none")
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(frame)
            s.set_color(self.th["rule"])
            s.set_linewidth(0.3)
        return a

    def text(self, x, y, s, size=5.5, color=None, ha="left", va="baseline", weight="regular", **kw):
        W, H = SHEET_MM
        self.fig.text(x / W, 1 - y / H, s, fontsize=size, color=color or self.th["fg"], ha=ha, va=va,
                      family=self.font, weight=weight, **kw)

    def line(self, x0, y0, x1, y1, color=None, lw=0.3, **kw):
        from matplotlib.lines import Line2D

        W, H = SHEET_MM
        self.fig.add_artist(Line2D([x0 / W, x1 / W], [1 - y0 / H, 1 - y1 / H], color=color or self.th["rule"],
                                   lw=lw, transform=self.fig.transFigure, **kw))

    def cross(self, x, y, r=1.6):
        self.line(x - r, y, x + r, y, color=self.th["faint"], lw=0.3)
        self.line(x, y - r, x, y + r, color=self.th["faint"], lw=0.3)


def _ramp():
    from matplotlib.colors import LogNorm

    from .colors import cmap

    return cmap(), LogNorm(SPEED_LO, SPEED_HI, clip=True)


def _runs(pieces, axes: tuple[int, int]):
    """Polylines in a projection: solid runs of one colour bin (the speed ramp) and guessed runs."""
    cm, norm = _ramp()
    solid, colors, guess = [], [], []
    for xyz, sp, g, *_ in pieces:
        p = xyz[:, axes]
        segg = g[:-1] | g[1:]
        b = np.floor(np.asarray(norm(np.maximum(0.5 * (sp[:-1] + sp[1:]), SPEED_LO * 1.0001))) * 47.999).astype(int)
        key = np.where(segg, -1, b)
        cut = np.flatnonzero(np.diff(key)) + 1
        for s0, s1 in zip(np.r_[0, cut], np.r_[cut, len(key)]):
            line = p[s0:s1 + 1]
            if key[s0] < 0:
                guess.append(line)
            else:
                solid.append(line)
                colors.append(cm((key[s0] + 0.5) / 48))
    return solid, colors, guess


def _draw_lines(ax, pieces, axes, th, lw=0.32):
    from matplotlib.collections import LineCollection

    solid, colors, guess = _runs(pieces, axes)
    if solid:
        ax.add_collection(LineCollection(solid, colors=colors, linewidths=lw, capstyle="round", joinstyle="round",
                                         zorder=3))
    if guess:
        ax.add_collection(LineCollection(guess, colors=[th["guess"]], linewidths=lw * 0.85,
                                         linestyles=[(0, (1.6, 1.6))], zorder=4))


def _extent(dr: dict) -> tuple[float, float, float, float, float, float]:
    """x0, x1, y0, y1, z0, z1 of the drawing (robust) and the cameras."""
    xs = [np.concatenate([p[0] for p in dr["pieces"]])] if dr["pieces"] else []
    pts = xs[0] if xs else np.zeros((0, 3))
    cams = np.array([c["C"] for c in dr["cameras"]]) if dr["cameras"] else np.zeros((0, 3))
    if len(pts):
        lo, hi = np.percentile(pts, 0.2, axis=0), np.percentile(pts, 99.8, axis=0)
    else:
        lo = hi = cams.mean(0) if len(cams) else np.zeros(3)
    if len(cams):
        lo, hi = np.minimum(lo, cams.min(0)), np.maximum(hi, cams.max(0))
    if not len(pts) and len(cams):
        r = max(c["R_max"] for c in dr["cameras"]) * 0.6
        lo, hi = lo - [r, r, 2], hi + [r, r, 2]
    return lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]


def _scale_for(dr, w_mm, h_mm, pad=1.08):
    x0, x1, y0, y1, *_ = _extent(dr)
    return min(w_mm / max((x1 - x0) * pad, 5), h_mm / max((y1 - y0) * pad, 5))      # mm per m


def _nice(v: float) -> float:
    e = 10 ** np.floor(np.log10(max(v, 1e-9)))
    for k in (1, 2, 2.5, 5, 10):
        if k * e >= v:
            return k * e
    return 10 * e


def _set_view(ax, cx, cy, w_mm, h_mm, s):
    ax.set_xlim(cx - w_mm / s / 2, cx + w_mm / s / 2)
    ax.set_ylim(cy - h_mm / s / 2, cy + h_mm / s / 2)
    ax.set_aspect("equal", adjustable="box")


def _plan(sh, ax, dr, w, h, s, cx, cy, th, cuts=None):
    """The plan: contours of the ground, the drawing, the cameras (start: grey; virtual: violet)."""
    _set_view(ax, cx, cy, w, h, s)
    seen = []
    for c in dr["cameras"]:
        t = c["terrain"]
        if t is None:
            continue
        z = t["z"].copy()
        X, Y = np.meshgrid(t["x"], t["y"])
        for (a0, a1, b0, b1) in seen:                       # one ground: the first package's where they overlap
            z[(X >= a0) & (X <= a1) & (Y >= b0) & (Y <= b1)] = np.nan
        seen.append((t["x"].min(), t["x"].max(), t["y"].min(), t["y"].max()))
        span = float(np.nanmax(z) - np.nanmin(z)) if np.isfinite(z).sum() >= 4 else 0.0
        if span < 0.05:
            continue
        step = _nice(max(span / 14, 0.25))
        lv = np.arange(np.floor(np.nanmin(z) / step) * step, np.nanmax(z) + step, step)
        ax.contour(X, Y, z, levels=lv, colors=[th["contour"]], linewidths=0.22, zorder=1)
    _draw_lines(ax, dr["pieces"], (0, 1), th)
    for c in dr["cameras"]:
        _camera_plan(ax, c, th)
    if cuts:
        for lab, (p0, p1) in cuts.items():
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=th["faint"], lw=0.3, ls=(0, (6, 2, 1, 2)), zorder=2)
            for p in (p0, p1):
                ax.text(p[0], p[1], lab, fontsize=4.5, color=th["muted"], family=sh.font, ha="center", va="center",
                        bbox={"boxstyle": "square,pad=0.15", "fc": th["bg"], "ec": "none"}, zorder=6)
    _north_and_scale(sh, ax, s, th)


def _camera_plan(ax, c, th):
    C, r = c["C"], c["R_max"]
    for yaw, col, ls, rr, lw in [(c["yaw"], th["ghost"], "-", r * 0.3, 0.3)] + \
            ([(c["virtual"]["yaw"], c.get("violet", th["violet"]), "-", r * 0.55, 0.45)] if "virtual" in c else []):
        a = np.radians(yaw)
        for side in (-1, 1):
            b = a + side * np.radians(c["hfov"] / 2)
            ax.plot([C[0], C[0] + rr * np.sin(b)], [C[1], C[1] + rr * np.cos(b)], color=col, lw=lw, ls=ls, zorder=5)
    ax.plot([C[0]], [C[1]], marker="o", ms=1.6, mfc="none", mec=th["fg2"], mew=0.35, zorder=6)


def _north_and_scale(sh, ax, s, th):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    L = _nice((x1 - x0) * 0.18)
    bx, by = x0 + (x1 - x0) * 0.05, y0 + (y1 - y0) * 0.06
    ax.plot([bx, bx + L], [by, by], color=th["fg2"], lw=0.4, zorder=7, solid_capstyle="butt")
    for x in (bx, bx + L / 2, bx + L):
        ax.plot([x, x], [by, by + (y1 - y0) * 0.012], color=th["fg2"], lw=0.3, zorder=7)
    ax.text(bx + L, by + (y1 - y0) * 0.022, f"{L:g} m", fontsize=4.2, color=th["muted"], family=sh.font, ha="right",
            va="bottom", zorder=7)
    nx, ny = x1 - (x1 - x0) * 0.05, y0 + (y1 - y0) * 0.06
    hgt = (y1 - y0) * 0.07
    ax.plot([nx, nx], [ny, ny + hgt], color=th["fg2"], lw=0.35, zorder=7)
    ax.plot([nx - hgt * 0.18, nx, nx + hgt * 0.18], [ny + hgt * 0.7, ny + hgt, ny + hgt * 0.7], color=th["fg2"],
            lw=0.35, zorder=7)
    ax.text(nx, ny + hgt * 1.12, "N", fontsize=4.2, color=th["muted"], family=sh.font, ha="center", va="bottom")


def _section(sh, ax, dr, w, h, s, along: str, cut: float, centre_h: float, th):
    """A section: the drawing projected onto a vertical plane (A-A: looking north, x across;
    B-B: looking west... x = north), the ground cut at the plane, at the plan's scale."""
    axes = (0, 2) if along == "x" else (1, 2)
    *_, z0, z1 = _extent(dr)
    zc = 0.5 * (z0 + z1)
    _set_view(ax, centre_h, zc, w, h, s)
    for c in dr["cameras"][:1] + dr["cameras"][1:]:
        t = c["terrain"]
        if t is None:
            continue
        if along == "x":
            j = int(np.argmin(np.abs(t["y"] - cut)))
            if abs(t["y"][j] - cut) <= 1.5:
                ax.plot(t["x"], t["z"][j], color=th["faint"], lw=0.3, zorder=2)
        else:
            i = int(np.argmin(np.abs(t["x"] - cut)))
            if abs(t["x"][i] - cut) <= 1.5:
                ax.plot(t["y"], t["z"][:, i], color=th["faint"], lw=0.3, zorder=2)
    _draw_lines(ax, dr["pieces"], axes, th, lw=0.3)
    for c in dr["cameras"]:
        h_ = c["C"][axes[0]]
        ax.plot([h_], [c["C"][2]], marker="o", ms=1.4, mfc="none", mec=th["fg2"], mew=0.35, zorder=6)
        if "virtual" in c:
            v = c["virtual"]
            a = np.radians(v["yaw"])
            d = np.array([np.sin(a), np.cos(a)])[axes[0]] * np.cos(np.radians(v["tilt"]))
            L = c["R_max"] * 0.25
            ax.plot([h_, h_ + L * d], [c["C"][2], c["C"][2] + L * np.sin(np.radians(v["tilt"]))],
                    color=c.get("violet", th["violet"]), lw=0.45, zorder=6)


# ----------------------------------------------------------------------------- the score
class _Staves:
    """Staves top-down in staff units (y grows downward)."""

    def __init__(self):
        self.items, self.y = [], 0.0

    def add(self, key, h, gap=0.28, **kw):
        self.y += gap if self.items else 0
        it = {"key": key, "y0": self.y, "h": h, **kw}
        self.items.append(it)
        self.y += h
        return it

    def space(self, g):
        self.y += g


def _map(st, v, lo, hi, pad=0.12):
    v = np.asarray(v, float)
    return st["y0"] + st["h"] * (1 - pad - (1 - 2 * pad) * (v - lo) / max(hi - lo, 1e-9))


def _score(sh, x, y, w, h, sc: dict, th: dict, marks: dict, players_colors: dict):
    """The score in the box (mm): label column + the time area."""
    import matplotlib.transforms as mt

    lab_w = 44.0
    ax = sh.ax(x + lab_w, y, w - lab_w, h - 6.0, frame=False)
    t_end = sc["t_end"]
    ax.set_xlim(0, t_end)
    S = _Staves()
    clock = S.add("clock", 2.2, label="CLIP CLOCK", sub=f"T 0–{sc['duration']:.0f} s")
    S.space(0.55)
    groups = []
    for p in sc["players"]:
        g0 = len(S.items)
        for key, hh, lab in (("turn", 1.0, "TURN"), ("tilt", 0.8, "TILT"), ("roll", 0.8, "ROLL"),
                             ("space", 0.55, "SPACE"), ("gain", 0.6, "GAIN")):
            S.add(key, hh, gap=0.18, label=lab, player=p)
        groups.append((p, g0, len(S.items)))
        S.space(0.5)
    mgroups = []
    for mc in sc["machine"]:
        g0 = len(S.items)
        S.add("counts", 1.05, gap=0.18, label="COUNTS", cam=mc)
        S.add("new", 0.42, gap=0.12, label="NEW IDS", cam=mc)
        mgroups.append((mc, g0, len(S.items)))
        S.space(0.5)
    total = S.y - 0.5
    ax.set_ylim(total + 0.05, -0.05)
    lab = mt.blended_transform_factory(ax.transAxes, ax.transData)
    fx = lambda mm: -mm / (w - lab_w)                          # mm left of the time area, in axes fraction
    rng = {}
    for k in ("turn", "tilt", "roll"):
        v = np.concatenate([np.abs(p[k]) for p in sc["players"]] or [np.zeros(1)])
        rng[k] = _nice(max(float(np.nanmax(v)) if len(v) and np.isfinite(v).any() else 1, 1))

    def staff_lines(st, n=5, strong=None):
        for i in range(n):
            yy = st["y0"] + st["h"] * (0.12 + 0.76 * i / (n - 1))
            ax.plot([0, t_end], [yy, yy], color=th["rule"] if i == strong else th["staff"], lw=0.22, zorder=1)

    def label(st, text, sub=None, color=None):
        ax.text(fx(9.5), st["y0"] + st["h"] / 2, text, transform=lab, fontsize=4.6, color=color or th["fg2"],
                ha="right", va="center", family=sh.font)
        if sub:
            ax.text(fx(1.2), st["y0"] + st["h"] / 2, sub, transform=lab, fontsize=3.6, color=th["faint"],
                    ha="right", va="center", family=sh.font)

    spans = [(a, b) for a, b, _ in sc.get("plays", [])] or [(0.0, t_end)]    # a collage: the plays

    def guide(t, color=None):
        ax.plot([t, t], [0, total], color=color or th["guide"], lw=0.25, ls=(0, (1, 2)), zorder=0)

    # time guides every 10 s
    step = _nice(t_end / 14)
    for tt in np.arange(0, t_end + 1e-9, step):
        ax.plot([tt, tt], [0, total], color=th["staff"], lw=0.18, ls=(0, (0.5, 3)), zorder=0)
    # the clip clock
    c = sc["clock"]
    staff_lines(clock, 5, strong=None)
    label(clock, "CLIP CLOCK", clock["sub"])
    t, T = c["t"], c["T"]
    starts = set(c.get("starts", []))                           # a collage: where each play begins (a row)
    if len(t) > 1:
        jump = np.abs(np.diff(T) - np.where(c["playing"][:-1], np.diff(t), 0)) > 0.25
        cuts = np.union1d(np.flatnonzero(jump) + 1, sorted(starts)).astype(int)
        for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(t)]):
            ax.plot(t[a:b], _map(clock, T[a:b], 0, sc["duration"]), color=th["fg"], lw=0.45, zorder=3,
                    drawstyle="steps-post")
        for i in (i for i in cuts if i not in starts):
            ax.plot([t[i], t[i]], _map(clock, [T[i - 1], T[i]], 0, sc["duration"]), color=th["fg"], lw=0.35,
                    ls=(0, (1, 1.2)), zorder=3)
    for t0, _, text in sc.get("plays", []):                     # a collage: each play labelled where it begins
        ax.plot([t0, t0], [0, total], color=th["rule"], lw=0.3, zorder=0)
        ax.text(t0 + t_end * 0.003, 0.08, text, fontsize=3.9, color=th["fg2"], family=sh.font, ha="left", va="top",
                zorder=5)
    for ts, Tv in c["seeks"]:
        guide(ts)
        marks["seek"].append(round(ts, 3))
    for tp in c["plays"]:
        ax.plot([tp], [clock["y0"] + clock["h"] * 0.97], marker=">", ms=1.7, color=th["fg2"], mew=0, zorder=4)
        marks["play"].append(round(tp, 3))
    for tp in c["pauses"]:
        yb = clock["y0"] + clock["h"] * 0.97
        for dx in (-0.12, 0.12):
            ax.plot([tp + dx * step / 10], [yb], marker="|", ms=1.8, color=th["fg2"], mew=0.4, zorder=4)
        marks["pause"].append(round(tp, 3))
    # the players
    for p, g0, g1 in groups:
        col = players_colors[p["player"]]
        items = S.items[g0:g1]
        top, bot = items[0]["y0"], items[-1]["y0"] + items[-1]["h"]
        ax.plot([fx(41.5), fx(41.5)], [top, bot], transform=lab, color=col, lw=0.6, clip_on=False, zorder=2,
                solid_capstyle="butt")
        cams = " ".join(sorted({str(cn).split(" · ")[-1] for cn, *_ in p["cameras"]}))
        ax.text(fx(40), top + 0.05, p["label"], transform=lab, fontsize=6.4, color=col, ha="left", va="top",
                family=sh.font, weight="medium")
        ax.text(fx(40), top + 0.95, cams, transform=lab, fontsize=4.2, color=th["muted"], ha="left", va="top",
                family=sh.font)
        ax.text(fx(40), top + 1.55, " ".join(p["src"]) or "", transform=lab, fontsize=3.6, color=th["faint"],
                ha="left", va="top", family=sh.font)
        pm = marks["players"].setdefault(p["label"], {"space": [], "level": [], "zero": [], "gain": []})
        for st in items:
            k = st["key"]
            if k in ("turn", "tilt", "roll"):
                staff_lines(st, 5, strong=2)
                label(st, st["label"], f"±{rng[k]:g}°")
                if len(p["t"]):
                    tt, vv = p["t"], p[k]
                    brk = np.flatnonzero(np.diff(tt) > 1.0) + 1
                    for a, b in zip(np.r_[0, brk], np.r_[brk, len(tt)]):
                        ax.plot(tt[a:b], _map(st, np.clip(vv[a:b], -rng[k], rng[k]), -rng[k], rng[k]), color=col,
                                lw=0.42, zorder=3, drawstyle="steps-post")
            elif k == "space":
                staff_lines(st, 3, strong=None)
                label(st, "BUTTONS")
                ym = st["y0"] + st["h"] * 0.5
                for a, b in p["held"]:
                    ax.add_patch(_bar(a, ym - st["h"] * 0.17, max(b - a, t_end * 1e-4), st["h"] * 0.34, col))
                    pm["space"].append([round(a, 3), round(b, 3)])
                for tl in p["level"]:
                    ax.plot([tl], [st["y0"] + st["h"] * 0.2], marker="v", ms=2.2, mfc="none", mec=col, mew=0.45,
                            zorder=5)
                    pm["level"].append(round(tl, 3))
                for tz in p["zero"]:
                    ax.plot([tz], [st["y0"] + st["h"] * 0.82], marker="o", ms=2.0, mfc="none", mec=col, mew=0.45,
                            zorder=5)
                    pm["zero"].append(round(tz, 3))
            elif k == "gain":
                staff_lines(st, 4, strong=1)
                label(st, "GAIN", "0–3×")
                gt, gv = (np.asarray(v, float) for v in p["gain"])
                if len(gt):
                    for a, b in spans:                  # within each play the player played, to its end
                        pt = p["t"][(p["t"] >= a - 1e-6) & (p["t"] <= b + 1e-6)]
                        if len(spans) > 1 and not len(pt):
                            continue
                        s0 = pt[0] if len(spans) > 1 else gt[0]
                        inside = (gt > s0) & (gt <= b + 1e-6)
                        v0 = value_at(gt, gv, s0)
                        xs, ys = np.r_[s0, gt[inside], b], np.r_[v0, gv[inside], gv[inside][-1] if inside.any() else v0]
                        ax.plot(xs, _map(st, ys, 0, 3), color=col, lw=0.45, drawstyle="steps-post", zorder=3)
                    for a, b in zip(gt, gv):
                        ax.text(a + t_end * 0.002, _map(st, b, 0, 3) - 0.04, f"{b:g}×", fontsize=3.4, color=col,
                                family=sh.font, ha="left", va="bottom", zorder=4)
                        pm["gain"].append([round(float(a), 3), round(float(b), 4)])
    # the machine
    for mc, g0, g1 in mgroups:
        items = S.items[g0:g1]
        top, bot = items[0]["y0"], items[-1]["y0"] + items[-1]["h"]
        ax.plot([fx(41.5), fx(41.5)], [top, bot], transform=lab, color=th["fg2"], lw=0.6, clip_on=False,
                solid_capstyle="butt")
        ax.text(fx(40), top + 0.05, "MACHINE", transform=lab, fontsize=4.6, color=th["fg"], ha="left", va="top",
                family=sh.font, weight="medium")
        ax.text(fx(40), top + 0.7, str(mc["camera"]).split(" · ")[-1], transform=lab, fontsize=4.2,
                color=th["muted"], ha="left", va="top", family=sh.font)
        mm = marks["machine"].setdefault(str(mc["camera"]), {})
        for st in items:
            if st["key"] == "counts":
                hi = _nice(max(int(max(mc["placed"].max(), mc["guessed"].max(), mc["lost"].max())), 4))
                staff_lines(st, 5, strong=None)
                label(st, "COUNTS", f"0–{hi:g}")
                for a, b in spans:                      # within each play its camera played, to its end
                    sel = (mc["t"] >= a - 1e-6) & (mc["t"] <= b + 1e-6)
                    if not sel.any():
                        continue
                    tt = np.r_[mc["t"][sel], b]
                    for key, color, lw, ls in (("lost", th["lost"], 0.3, "-"), ("placed", th["fg"], 0.42, "-"),
                                               ("guessed", th["guess"], 0.42, (0, (2.2, 1.6)))):
                        v = np.r_[mc[key][sel], mc[key][sel][-1]]
                        ax.plot(tt, _map(st, v, 0, hi), color=color, lw=lw, ls=ls, drawstyle="steps-post", zorder=3)
                for key in ("lost", "placed", "guessed"):
                    mm[key] = int(mc[key].max())
                # legend
                for i, (key, color, ls) in enumerate((("placed", th["fg"], "-"), ("guessed", th["guess"], (0, (2.2, 1.6))),
                                                     ("lost", th["lost"], "-"))):
                    yy = st["y0"] + st["h"] * (0.2 + 0.3 * i)
                    ax.plot([fx(29), fx(25.5)], [yy, yy], transform=lab, color=color, lw=0.42, ls=ls, clip_on=False)
                    ax.text(fx(25), yy, key, transform=lab, fontsize=3.5, color=th["muted"], ha="left", va="center",
                            family=sh.font)
            else:
                staff_lines(st, 2, strong=None)
                label(st, "NEW IDS")
                nt, nv = mc["new"]
                if len(nt):
                    hmax = max(int(nv.max()), 1)
                    yb = st["y0"] + st["h"] * 0.88
                    hh = st["h"] * 0.76 * (0.3 + 0.7 * np.minimum(nv, hmax) / hmax)
                    ax.vlines(nt, yb - hh, yb, colors=[th["fg2"]], linewidths=0.25, zorder=3)
                mm["new"] = int(nv.sum()) if len(nt) else 0
    # the time axis
    ay = y + h - 4.0
    for tt in np.arange(0, t_end + 1e-9, step):
        px = x + lab_w + (w - lab_w) * tt / t_end
        sh.line(px, ay - 1.2, px, ay - 0.4, color=th["muted"], lw=0.3)
        lab_t = f"{int(tt // 60)}:{int(tt % 60):02d}" if step >= 1 else f"{int(tt // 60)}:{tt % 60:04.1f}"
        sh.text(px, ay + 1.6, lab_t, size=4.2, color=th["muted"], ha="center")
    sh.line(x + lab_w, ay - 0.4, x + w, ay - 0.4, color=th["rule"], lw=0.3)
    sh.text(x + lab_w - 4.0, ay + 1.6, "PERFORMANCE t", size=4.2, color=th["muted"], ha="right")
    return ax


def _bar(x, y, w, h, color, transform=None):
    from matplotlib.patches import Rectangle

    kw = {"transform": transform, "clip_on": False} if transform is not None else {}
    return Rectangle((x, y), w, h, facecolor=color, edgecolor="none", alpha=0.9, zorder=4, **kw)


# ----------------------------------------------------------------------------- the sheet
def _top(sh, layout, x, y, w, h, dr, view, th, gap=6.0, seq=None):
    """The finished drawing in the band: the view, the plan, the sections. seq (layout B): the
    times of its three plans (without a recording, B falls back to A)."""
    boxes = {}
    if layout == "C":
        vw = h * 3
        boxes["view"] = (x, y, vw, h)
        cw = w - vw - gap
        boxes["plan"] = (x + vw + gap, y, cw, h * 0.66)
        boxes["A"] = (x + vw + gap, y + h * 0.66 + 3, cw, h * 0.34 - 3)
    elif layout == "B" and seq:
        return _top_sequence(sh, x, y, w, h, dr, view, th, seq, gap)
    else:
        vw = h * 2
        boxes["view"] = (x, y, vw, h)
        boxes["plan"] = (x + vw + gap, y, h, h)
        sx = x + vw + gap + h + gap
        boxes["A"] = (sx, y, x + w - sx, (h - 4) / 2)
        boxes["B"] = (sx, y + (h - 4) / 2 + 4, x + w - sx, (h - 4) / 2)
    _view(sh, boxes["view"], view, th)
    pb = boxes["plan"]
    s = _scale_for(dr, pb[2], pb[3])
    x0, x1, y0, y1, *_ = _extent(dr)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    hw, hh = pb[2] / s / 2, pb[3] / s / 2
    cuts = {"A": ((cx - hw * 0.96, cy), (cx + hw * 0.96, cy))}
    if "B" in boxes:
        cuts["B"] = ((cx, cy - hh * 0.96), (cx, cy + hh * 0.96))
    ax = sh.ax(*pb)
    _plan(sh, ax, dr, pb[2], pb[3], s, cx, cy, th, cuts)
    den = 1000 / s
    scale_txt = f"1:{round(den, -int(np.floor(np.log10(den))) + 1):,.0f}".replace(",", " ")
    sh.text(pb[0] + 1.5, pb[1] + 3.2, "PLAN", size=4.8, color=th["fg2"])
    sh.text(pb[0] + 1.5, pb[1] + 5.6, f"{scale_txt} at {SHEET_MM[0]:.0f} × {SHEET_MM[1]:.0f} mm", size=3.6,
            color=th["faint"])
    for key, along, cut, ch in (("A", "x", cy, cx), ("B", "y", cx, cy)):
        if key not in boxes:
            continue
        b = boxes[key]
        a2 = sh.ax(*b)
        _section(sh, a2, dr, b[2], b[3], s, along, cut, ch, th)
        look = "looking north" if along == "x" else "looking west"
        sh.text(b[0] + 1.5, b[1] + 3.2, f"SECTION {key}–{key}", size=4.8, color=th["fg2"])
        sh.text(b[0] + 1.5, b[1] + 5.6, f"{look} · the drawing projected, the ground cut", size=3.6, color=th["faint"])
    return boxes


def _top_sequence(sh, x, y, w, h, dr, view, th, times, gap):
    """Layout B: the view, then the drawing at each of `times` (cumulative): a plan over its
    section A-A, one scale and one frame for all."""
    vw = h * 2
    _view(sh, (x, y, vw, h), view, th)
    rx, rw = x + vw + gap, w - vw - gap
    cw = (rw - 2 * 4.0) / 3
    sec_h = 15.0
    ph = h - sec_h - 3.0
    s = _scale_for(dr, cw, ph)
    x0, x1, y0, y1, *_ = _extent(dr)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    hw = cw / s / 2
    den = 1000 / s
    scale_txt = f"1:{round(den, -int(np.floor(np.log10(den))) + 1):,.0f}".replace(",", " ")
    for i, t in enumerate(times):
        bx = rx + i * (cw + 4.0)
        part = dr if i == len(times) - 1 else dict(dr, pieces=until(dr["pieces"], t), cameras=_cameras_at(dr["cameras"], t))
        ax = sh.ax(bx, y, cw, ph)
        _plan(sh, ax, part, cw, ph, s, cx, cy, th, {"A": ((cx - hw * 0.96, cy), (cx + hw * 0.96, cy))})
        last = i == len(times) - 1
        sh.text(bx + 1.5, y + 3.2, f"PLAN AT {_fmt_dur(t)}" + (" · THE END" if last else ""), size=4.8, color=th["fg2"])
        sh.text(bx + 1.5, y + 5.6, f"{scale_txt} · drawn by then", size=3.6, color=th["faint"])
        a2 = sh.ax(bx, y + ph + 3.0, cw, sec_h)
        _section(sh, a2, part, cw, sec_h, s, "x", cy, cx, th)
        sh.text(bx + 1.5, y + ph + 3.0 + 3.0, "A–A", size=4.2, color=th["fg2"])
    return {}


def _view(sh, box, view, th):
    x, y, w, h = box
    ax = sh.ax(x, y, w, h)
    ax.set_facecolor(th["bg"])
    img = None
    if view is not None:
        from PIL import Image

        try:
            im = Image.open(view).convert("RGB")
            iw, ih = im.size
            want = w / h
            if abs(iw / ih - want) > 0.01:                        # cropped to the frame (centred)
                if iw / ih > want:
                    nw = round(ih * want)
                    im = im.crop(((iw - nw) // 2, 0, (iw - nw) // 2 + nw, ih))
                else:
                    nh = round(iw / want)
                    im = im.crop((0, (ih - nh) // 2, iw, (ih - nh) // 2 + nh))
            img = np.asarray(im)
            if th["light"]:
                img = light_view(img)
        except OSError:
            img = None
    if img is not None:
        ax.imshow(img, extent=(0, 1, 0, 1), aspect="auto", interpolation="lanczos", zorder=0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        ax.text(0.5, 0.5, "the Live view's render is missing\n(groundtrack live-sheet renders it)", fontsize=5,
                color=th["muted"], ha="center", va="center", family=sh.font, transform=ax.transAxes)
    ax.text(0.012, 0.03, "VIEW · THE LIVE VIEW, FIXED PRESET", fontsize=4.8, color=th["fg2"],
            family=sh.font, transform=ax.transAxes)


def _fmt_dur(s: float) -> str:
    s = round(s)
    return f"{s // 60}:{s % 60:02d}"


def _draw(out_dir: Path, name: str, themes, layout: str, view, dr: dict, sc: dict | None, title: str,
          header: str, note=("NO SCORE", "")) -> tuple[dict, dict]:
    """The sheet in each theme: <out_dir>/<name>_<theme>.pdf and .png. Returns (files, the marks
    the score drew)."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    if layout not in LAYOUTS:
        raise ValueError(f"layout: one of {', '.join(LAYOUTS)}")
    for theme in themes:
        if theme not in THEMES:
            raise ValueError(f"themes: of {', '.join(THEMES)}")
    font = _fonts()
    W, H = SHEET_MM
    files, info = {}, None
    for theme in themes:
        th = THEMES[theme]
        colors = {p["player"]: _dark(p["tint"], th["light"]) for p in (sc["players"] if sc else [])}
        for c in dr["cameras"]:
            for p in (sc["players"] if sc else []):
                if any(cn == c["name"] for cn, *_ in p["cameras"]):
                    c["violet"] = colors[p["player"]]
                    break
            else:
                c.pop("violet", None)
        marks = {"seek": [], "play": [], "pause": [], "players": {}, "machine": {}}
        with plt.rc_context({"pdf.fonttype": 42, "font.family": font, "lines.solid_capstyle": "round",
                             "path.simplify": False}):
            sh = _Sheet(th, font)
            for cx, cy in ((MARGIN, MARGIN), (W - MARGIN, MARGIN), (MARGIN, H - MARGIN), (W - MARGIN, H - MARGIN)):
                sh.cross(cx, cy)
            # the header
            sh.text(MARGIN + 3, MARGIN + 5.2, title, size=6.2, weight="medium")
            sh.text(MARGIN + 62, MARGIN + 5.2, header, size=4.6, color=th["fg2"])
            _legend(sh, W - MARGIN - 3, MARGIN + 5.2, th)
            band_y, band_h = MARGIN + 9.5, 86.0
            seq = [sc["t_end"] * k / 3 for k in (1, 2, 3)] if sc and layout == "B" else None
            _top(sh, layout, MARGIN + 3, band_y, W - 2 * MARGIN - 6, band_h, dr, view, th, seq=seq)
            sy = band_y + band_h + 5.0
            sh.line(MARGIN + 3, sy - 2.5, W - MARGIN - 3, sy - 2.5, color=th["rule"], lw=0.3)
            if sc:
                _score(sh, MARGIN + 3, sy, W - 2 * MARGIN - 6, H - MARGIN - sy - 1.0, sc, th, marks, colors)
            else:
                sh.text(W / 2, sy + 40, note[0], size=5.5, color=th["muted"], ha="center")
                sh.text(W / 2, sy + 46, note[1], size=4.6, color=th["faint"], ha="center")
            stem = out_dir / f"{name}_{theme}"
            sh.fig.savefig(stem.with_suffix(".pdf.part"), format="pdf", facecolor=th["bg"])
            sh.fig.savefig(stem.with_suffix(".png.part"), format="png", dpi=DPI, facecolor=th["bg"])
            plt.close(sh.fig)
        for ext in ("pdf", "png"):
            os.replace(stem.with_suffix(f".{ext}.part"), stem.with_suffix(f".{ext}"))
        files[theme] = {"pdf": str(stem.with_suffix(".pdf")), "png": str(stem.with_suffix(".png"))}
        info = marks
    return files, info


def _counted(n: int, what: str) -> str:
    return f"{n} {what}{'S' if n != 1 else ''}"


def build_sheet(run: str | Path, pass_id: str, themes=DEFAULT_THEMES, layout: str = DEFAULT_LAYOUT,
                view: str | Path | None = None, name: str = "sheet", out_dir: Path | None = None) -> dict:
    """The sheet of a saved pass, per theme: <pass folder>/<name>_<theme>.pdf and .png (and
    sheet.json: what the score drew). view: the Live view's render (default: sheet_view.png in the
    pass folder, when there). Returns {"files": {theme: {"pdf", "png"}}, "info"}."""
    run = Path(run)
    if layout not in LAYOUTS:
        raise ValueError(f"layout: one of {', '.join(LAYOUTS)}")
    m = read_pass(run, pass_id)
    folder = _folder(run, pass_id)
    out_dir = Path(out_dir) if out_dir else folder
    rec = read_recording(run, pass_id)
    sc = score_data(rec) if rec else None
    dr = drawing_data(run, pass_id, rec)
    if view is None and (folder / VIEW_FILE).is_file():
        view = folder / VIEW_FILE
    started = str(m.get("started") or "")[:16].replace("T", " ")
    secs = sc["t_end"] if sc else float(m.get("clock_s") or 0)
    header = (f"{str(m.get('site') or '').upper()} · PASS {pass_id} · {started} · "
              + (f"{_fmt_dur(secs)} PLAYED · {_counted(len(sc['players']), 'PLAYER')} · " if sc else
                 f"THE CLIP TO {_fmt_dur(secs)} · NO RECORDING · ") + _counted(len(m.get("cameras") or []), "CAMERA"))
    files, info = _draw(out_dir, name, themes, layout, view, dr, sc, "MOTUS · PERFORMANCE SHEET", header,
                        ("NO SCORE: THIS PASS WAS SAVED BEFORE RECORDINGS (FORMAT V1, BEFORE 29 OCT)",
                         "its drawing is above; its moves, buttons and the machine's counts were not kept"))
    meta = {"pass": pass_id, "layout": layout, "layouts": LAYOUTS, "themes": list(themes), "view": str(view) if view else None,
            "recorded": bool(rec), "seconds": sc["t_end"] if sc else None, "points": dr["points"],
            "guessed": dr["guessed"], "players": [p["label"] for p in sc["players"]] if sc else [],
            "marks": info, "files": files, "made": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out_dir / (INFO_FILE if name == "sheet" else f"{name}.json")).write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return {"files": files, "info": meta}


# ----------------------------------------------------------------------------- the collage (08 sandbox)
SHEETS_DIR = "sheets"                       # <run of the first layer>/live/sheets/<stamp>-<collage | camera id>/
LAYERS_FILE = "layers.json"


def _plays(layers: list) -> list:
    """The plays (saved passes) that made the layers, oldest first: [(run, id, {cameras})]."""
    out = {}
    for ly in sorted(layers, key=lambda ly: ly.get("key") or 0):
        k = (str(Path(ly["run"]).resolve()), ly["id"])
        out.setdefault(k, (Path(ly["run"]), ly["id"], set()))[2].add(ly["camera"])
    return list(out.values())


def collage_recording(plays: list) -> tuple[dict | None, list]:
    """The plays' recordings one after another on one time axis (each kept to the cameras whose
    layer it made; a short gap between plays): a recording as score_data reads it, and the plays as
    [(t0, t1, label)] (a play saved before recordings: none)."""
    clocks, cams, events, spans, metas = [], {}, [], [], []
    t0, k0, dur = 0.0, 0, 0.0
    recs = [(run, pid, keep, read_recording(run, pid)) for run, pid, keep in plays]
    total = sum(float(r["clock"]["t"][-1]) for *_, r in recs if r and len(r["clock"]))
    gap = max(2.0, 0.02 * total)
    for i, (_run, pid, keep, r) in enumerate(recs):
        if not r or not len(r["clock"]):
            continue
        c = r["clock"].copy()
        c["t"] = c["t"] + t0
        clocks.append(c)
        for name, a in r["cameras"].items():
            if name in keep and len(a):
                a = a.copy()
                a["k"] = a["k"] + k0
                cams[name] = np.concatenate([cams[name], a]) if name in cams else a
        for e in r["events"]:
            if e.get("camera") in (None, *keep):
                events.append({**e, "t": float(e.get("t") or 0) + t0, "k": int(e.get("k") or 0) + k0})
        t1 = float(c["t"][-1])
        held = {}                                        # SPACE still held at the play's end: released there
        for e in sorted(r["events"], key=lambda e: float(e.get("t") or 0)):
            if e.get("camera") in keep and e.get("type") in ("space_press", "space_release"):
                if e["type"] == "space_press":
                    held[e.get("camera")] = e
                else:
                    held.pop(e.get("camera"), None)
        events += [{**e, "type": "space_release", "t": t1, "k": k0 + len(c) - 1} for e in held.values()]
        spans.append((t0, t1, f"PLAY {i + 1} · {pid} · {' '.join(sorted(str(n).split(' · ')[-1] for n in keep))}"))
        metas.append(r["meta"] or {})
        dur = max(dur, float((r.get("manifest") or {}).get("duration_s") or 0))
        k0 += len(c)
        t0 = t1 + gap
    if not clocks:
        return None, []
    meta = {"seconds": spans[-1][1], "cameras": [c for m in metas for c in (m.get("cameras") or [])]}
    rec = {"meta": meta, "manifest": {"duration_s": dur or None}, "clock": np.concatenate(clocks),
           "cameras": cams, "events": events}
    rec["starts"] = list(np.cumsum([0] + [len(c) for c in clocks[:-1]]).astype(int))
    return rec, spans


def collage_drawing(plays: list) -> dict:
    """The layers' drawings in one frame (the first play's): each play's drawing kept to the
    cameras whose layer it made, its times one after another as in collage_recording; each camera
    as it stood at the end of its last play."""
    pieces, cams, origin, pts, guessed = [], {}, None, 0, 0
    recs = [(run, pid, keep, read_recording(run, pid)) for run, pid, keep in plays]
    total = sum(float(r["clock"]["t"][-1]) for *_, r in recs if r and len(r["clock"]))
    gap, t0 = max(2.0, 0.02 * total), 0.0
    for run, pid, keep, r in recs:
        d = drawing_data(run, pid, r, keep=keep)
        if origin is None:
            origin = d["origin"]
        sh_ = np.subtract(d["origin"], origin)
        pieces += [(xyz + sh_, sp, g, tp + t0) for xyz, sp, g, tp in d["pieces"]]
        pts, guessed = pts + d["points"], guessed + d["guessed"]
        for c in d["cameras"]:
            if c["name"] in keep:
                c = dict(c, C=c["C"] + sh_, terrain=_shift_terrain(c["terrain"], sh_))
                if "path" in c:
                    tt, hd, tl, C = c["path"]
                    c["path"] = (tt + t0, hd, tl, C + sh_)
                    c["virtual"] = dict(c["virtual"], C=c["virtual"]["C"] + sh_)
                cams[c["name"]] = c                      # a later play: the camera as it stood then
        if r and len(r["clock"]):
            t0 += float(r["clock"]["t"][-1]) + gap
    return {"pieces": pieces, "cameras": list(cams.values()), "origin": origin or (0.0, 0.0, 0.0), "points": pts,
            "guessed": guessed}


def _shift_terrain(t, d):
    if not t:
        return t
    return {"x": t["x"] + d[0], "y": t["y"] + d[1], "z": t["z"] + d[2]}


def build_collage(layers: list, camera: str | None = None, themes=DEFAULT_THEMES, layout: str = DEFAULT_LAYOUT,
                  view: str | Path | None = None, out_dir: Path | None = None, site: str = "") -> dict:
    """The sheet of the collage (08 sandbox) or of one camera's layer: the layers' drawings on top
    (view: the Live view's render of them), the score of the plays that made them below, one after
    another. layers: [{run, id, camera, key, kind, stacked}] (saved passes; live.js collageLayers),
    kept to `camera` when given. Written to <run of the first layer>/live/sheets/<stamp>-<collage |
    camera id>/ (sheet_<theme>.pdf / .png, layers.json). Returns {"files", "info", "dir"}."""
    if camera:
        layers = [ly for ly in layers if ly.get("camera") == camera]
    if not layers:
        raise ValueError("no layer to draw" + (f" for {camera}" if camera else ""))
    for ly in layers:
        if not ly.get("id"):
            raise ValueError(f"{ly.get('camera')}: its play is not saved yet")
        m = read_pass(Path(ly["run"]), ly["id"])
        if ly["camera"] not in {c.get("name") for c in m.get("cameras") or []}:
            raise ValueError(f"{ly['camera']}: not in pass {ly['id']}")
    plays = _plays(layers)
    rec, spans = collage_recording(plays)
    sc = score_data(rec) if rec else None
    if sc:
        sc["plays"] = spans
        sc["clock"]["starts"] = rec["starts"][1:]
    dr = collage_drawing(plays)
    if out_dir is None:
        cid = "collage"
        if camera:
            m = read_pass(plays[0][0], plays[0][1])
            cid = next((str(c.get("id") or "") for c in m.get("cameras") or [] if c.get("name") == camera), "") or camera
            cid = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in cid)[:40]
        base = Path(layers[0]["run"]) / "live" / SHEETS_DIR
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_dir = base / f"{stamp}-{cid}"
        for k in range(2, 100):
            if not out_dir.exists():
                break
            out_dir = base / f"{stamp}-{k}-{cid}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if view is not None and Path(view).parent != out_dir:
        shutil.copyfile(view, out_dir / VIEW_FILE)
        view = out_dir / VIEW_FILE
    cams = sorted({ly["camera"] for ly in layers})
    secs = sc["t_end"] if sc else 0.0
    header = (f"{site.upper() + ' · ' if site else ''}{_counted(len(layers), 'LAYER')} FROM {_counted(len(plays), 'PLAY')} · "
              + (f"{_fmt_dur(secs)} PLAYED · {_counted(len(sc['players']), 'PLAYER')} · " if sc else "NO RECORDING · ")
              + _counted(len(cams), "CAMERA"))
    title = f"MOTUS · CAMERA SHEET · {camera}" if camera else "MOTUS · COLLAGE SHEET"
    files, info = _draw(out_dir, "sheet", themes, layout, view, dr, sc, title, header,
                        ("NO SCORE: NONE OF THESE PLAYS WAS RECORDED (SAVED BEFORE 29 OCT)",
                         "their drawing is above; their moves, buttons and the machine's counts were not kept"))
    meta = {"kind": "camera" if camera else "collage", "camera": camera, "layout": layout, "themes": list(themes),
            "view": str(view) if view else None, "layers": [
                {**{k: ly.get(k) for k in ("camera", "id", "key", "kind", "stacked")}, "run": str(ly["run"])} for ly in layers],
            "plays": [{"run": str(r), "id": pid, "cameras": sorted(keep),
                       "t0": next((round(a, 3) for a, _, lab in spans if pid in lab), None)} for r, pid, keep in plays],
            "seconds": secs, "points": dr["points"], "guessed": dr["guessed"],
            "players": [p["label"] for p in sc["players"]] if sc else [], "marks": info, "files": files,
            "made": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out_dir / LAYERS_FILE).write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return {"files": files, "info": meta, "dir": str(out_dir)}


def _legend(sh, x_right, y, th):
    """Top right: the lexicon (the speed ramp, guesses, the players)."""
    from matplotlib.colors import LogNorm

    from .colors import cmap

    cm = cmap()
    norm = LogNorm(SPEED_LO, SPEED_HI)
    bw = 34.0
    x0 = x_right - bw
    for i in range(48):
        v = SPEED_LO * (SPEED_HI / SPEED_LO) ** ((i + 0.5) / 48)
        sh.line(x0 + bw * i / 48, y - 1.4, x0 + bw * (i + 1) / 48, y - 1.4, color=cm(norm(v)), lw=1.6,
                solid_capstyle="butt")
    sh.text(x0, y + 1.6, "1", size=3.6, color=th["muted"])
    sh.text(x0 + bw, y + 1.6, "40 m/s", size=3.6, color=th["muted"], ha="right")
    sh.text(x0 - 2, y - 0.6, "AS DRAWN", size=4.2, color=th["muted"], ha="right")
    gx = x0 - 30
    sh.line(gx, y - 1.4, gx + 6, y - 1.4, color=th["guess"], lw=0.5, ls=(0, (2, 1.6)))
    sh.text(gx + 7, y - 0.6, "GUESSED", size=4.2, color=th["muted"])
    vx = gx - 34
    sh.line(vx, y - 1.4, vx + 6, y - 1.4, color=th["violet"], lw=0.8, solid_capstyle="butt")
    sh.text(vx + 7, y - 0.6, "THE PLAYERS", size=4.2, color=th["muted"])
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    W, H = SHEET_MM
    bx = vx - 62
    sh.fig.add_artist(Rectangle((bx / W, 1 - (y - 0.8) / H), 5 / W, 1.1 / H, transform=sh.fig.transFigure,
                                facecolor=th["violet"], edgecolor="none"))
    sh.text(bx + 6, y - 0.6, "SPACE HELD", size=4.2, color=th["muted"])
    for i, (mk, txt) in enumerate((("v", "LEVEL"), ("o", "ZERO"))):
        mx = bx + 25 + 17 * i
        sh.fig.add_artist(Line2D([mx / W], [1 - (y - 1.3) / H], marker=mk, ms=2.2, mfc="none", mec=th["violet"],
                                 mew=0.45, transform=sh.fig.transFigure))
        sh.text(mx + 2.2, y - 0.6, txt, size=4.2, color=th["muted"])


# ----------------------------------------------------------------------------- the view: saved, rendered
def save_view(run: Path, pass_id: str, png: bytes) -> Path:
    """The Live view's render for the sheet, into the pass folder (one PNG)."""
    read_pass(run, pass_id)
    if not isinstance(png, (bytes, bytearray)) or not bytes(png[:8]) == PNG or len(png) > MAX_VIEW_BYTES:
        raise ValueError("the view: one PNG")
    p = _folder(Path(run), pass_id) / VIEW_FILE
    tmp = p.with_name(p.name + ".part")
    tmp.write_bytes(bytes(png))
    os.replace(tmp, p)
    return p


ERRORS: dict = {}                                   # (run, id) -> the view's last error (render_view)


def request(resolve_run, body: bytes) -> dict:
    """/api/live/sheet (desktop.py): [u32 header length][JSON header][files] (live_passes.parse_files).
    {"action": "view", "run", "id", "build" (default true), "layout"} + view.png: saved, and the sheet
    built (light; "themes": ["black", "light"] for others); {"action": "error", "run", "id", "message"}: the view could not render."""
    head, files = parse_files(body)
    if head.get("action") == "collage":
        return _collage_request(resolve_run, head, files)
    run, pid = resolve_run(head.get("run")), str(head.get("id", ""))
    read_pass(run, pid)
    if head.get("action") == "error":
        ERRORS[(str(Path(run).resolve()), pid)] = str(head.get("message") or "the view failed")[:400]
        return {"ok": True}
    if head.get("action") != "view":
        raise ValueError("action: view, collage or error")
    view = save_view(run, pid, files.get("view.png") or b"")
    if head.get("build") is False:
        return {"ok": True, "view": str(view)}
    themes = head.get("themes") or DEFAULT_THEMES
    if not isinstance(themes, list | tuple) or not themes or any(t not in THEMES for t in themes):
        raise ValueError(f"themes: of {', '.join(THEMES)}")
    out = build_sheet(run, pid, themes=tuple(themes), layout=str(head.get("layout") or DEFAULT_LAYOUT), view=view)
    return {"ok": True, "view": str(view), "files": out["files"], "dir": str(_folder(Path(run), pid))}


def _collage_request(resolve_run, head: dict, files: dict) -> dict:
    """{"action": "collage", "camera" (a camera sheet; none: the collage), "layers": [{run, id,
    camera, key, kind, stacked}], "site", "layout", "themes"} + view.png (the view's render of the
    layers): the sheet in <run of the first layer>/live/sheets/."""
    raw = head.get("layers")
    if not isinstance(raw, list) or not raw or len(raw) > 512:
        raise ValueError("layers: a list")
    layers = []
    for ly in raw:
        ok = isinstance(ly, dict) and isinstance(ly.get("camera"), str)
        if not ok:
            raise ValueError("a layer: {run, id, camera}")
        if not ly.get("id"):
            raise ValueError(f"{ly['camera']}: its play is not saved yet")
        layers.append({"run": resolve_run(ly.get("run")), "id": str(ly["id"]), "camera": ly["camera"],
                       "key": ly.get("key"), "kind": str(ly.get("kind") or "play"), "stacked": bool(ly.get("stacked"))})
    themes = head.get("themes") or DEFAULT_THEMES
    if not isinstance(themes, list | tuple) or not themes or any(t not in THEMES for t in themes):
        raise ValueError(f"themes: of {', '.join(THEMES)}")
    png = files.get("view.png")
    tmp = None
    if png is not None:
        if not bytes(png[:8]) == PNG or len(png) > MAX_VIEW_BYTES:
            raise ValueError("the view: one PNG")
        fd, tmp = tempfile.mkstemp(suffix=".png")
        with os.fdopen(fd, "wb") as f:
            f.write(png)
    try:
        cam = head.get("camera")
        out = build_collage(layers, camera=cam if isinstance(cam, str) and cam else None, themes=tuple(themes),
                            layout=str(head.get("layout") or DEFAULT_LAYOUT), view=tmp, site=str(head.get("site") or ""))
    finally:
        if tmp:
            os.unlink(tmp)
    return {"ok": True, "files": out["files"], "dir": out["dir"]}


def find_browser() -> str | None:
    """Chrome or Edge on this machine (headless, for the view)."""
    env = os.environ.get("MOTUS_BROWSER")
    cands = [env] if env else []
    for base in (os.environ.get("PROGRAMFILES"), os.environ.get("PROGRAMFILES(X86)"), os.environ.get("LOCALAPPDATA")):
        if base:
            cands += [os.path.join(base, "Google", "Chrome", "Application", "chrome.exe"),
                      os.path.join(base, "Microsoft", "Edge", "Application", "msedge.exe")]
    cands += [shutil.which(n) for n in ("chrome", "google-chrome", "chromium", "msedge")]
    return next((c for c in cands if c and os.path.isfile(c)), None)


def view_size(layout: str = DEFAULT_LAYOUT) -> tuple[int, int]:
    """The view's frame at print size (px at DPI): 2:1 (A, B) or 3:1 (C), 86 mm high."""
    h_mm = 86.0
    w_mm = h_mm * (3 if layout == "C" else 2)
    px = lambda mm: round(mm / 25.4 * DPI / 2) * 2
    return px(w_mm), px(h_mm)


def render_view(run: str | Path, pass_id: str, root: Path | None = None, layout: str = DEFAULT_LAYOUT,
                timeout: float = 180.0, browser: str | None = None, log=print) -> Path:
    """The Live view at print size from the fixed preset: the app's server on this machine only
    (127.0.0.1, a free port, no phone hub), its /live page in headless Chrome or Edge with
    sheet=<id>; the page posts its render back (sheet_view.png). Returns its path."""
    from http.server import ThreadingHTTPServer
    from urllib.parse import quote

    from .desktop import DesktopApp, make_desktop_handler
    from .live_pass_export import _project_root

    run = Path(run).resolve()
    read_pass(run, pass_id)
    root = Path(root).resolve() if root else _project_root(run)
    exe = browser or find_browser()
    if not exe:
        raise RuntimeError("no Chrome or Edge found for the Live view's render (set MOTUS_BROWSER, or --no-view)")
    target = _folder(run, pass_id) / VIEW_FILE
    before = target.stat().st_mtime if target.is_file() else 0.0
    app = DesktopApp(root)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_desktop_handler(app, 0))
    port = srv.server_address[1]
    srv.RequestHandlerClass = make_desktop_handler(app, port)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    w, h = view_size(layout)
    url = (f"http://127.0.0.1:{port}/live?run={quote(str(run))}&sheet={quote(pass_id)}&sw={w}&sh={h}"
           f"&build=0&hub=0&save=0&video=0&panel=0&hud=0")
    prof = Path(tempfile.mkdtemp(prefix="motus-sheet-"))
    args = [exe, "--headless=new", "--enable-gpu", "--ignore-gpu-blocklist", "--no-first-run",
            "--no-default-browser-check", "--disable-extensions", "--mute-audio", f"--user-data-dir={prof}",
            "--window-size=1600,900", url]
    if os.name == "nt":
        args.insert(2, "--use-angle=d3d11")
    key = (str(run), pass_id)
    ERRORS.pop(key, None)
    log(f"the view: {Path(exe).name} headless, {w} x {h} px")
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if key in ERRORS:
                raise RuntimeError(f"the view: {ERRORS.pop(key)}")
            if target.is_file() and target.stat().st_mtime > before:
                return target
            if proc.poll() is not None:
                raise RuntimeError(f"the browser stopped (exit {proc.returncode}) before the view was rendered")
            time.sleep(0.25)
        raise TimeoutError(f"the view was not rendered in {timeout:.0f} s")
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
        srv.shutdown()
        srv.server_close()
        shutil.rmtree(prof, ignore_errors=True)


def make_sheet(run: str | Path, pass_id: str, layout: str = DEFAULT_LAYOUT, themes=DEFAULT_THEMES,
               view: bool = True, root: Path | None = None, log=print) -> dict:
    """`groundtrack live-sheet`: the view rendered (unless view=False: the one saved before, if
    any), then the sheet in each theme."""
    run = Path(run)
    if view:
        log(f"the view: {render_view(run, pass_id, root=root, layout=layout, log=log)}")
    out = build_sheet(run, pass_id, themes=themes, layout=layout)
    for theme, f in out["files"].items():
        log(f"{theme}: {f['pdf']}")
        log(f"{theme}: {f['png']}")
    return out
