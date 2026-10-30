"""Live controller: the machine's own rules on the drawn positions (plans/live_controller/08 (c)).

The Python reference of ``desktop_web/live/rules.js``: the same rules on the same arrays. The view
runs the JS; this one is for tests, sessions and exports.

White dashes mean one thing: the machine did not see, so it guessed. In Live the tracks are drawn
where the moved camera (and the turned space) puts them, and the machine applies the rules it applies
to the footage to those drawn positions, in the world (after the space tilt, so in 3D; on flat, level
ground the same as groundtrack's 2D rules). The detector is not re-run: the footage's points are the
detections, and their footage guesses (predicted, stitched, interpolated) stay guesses.

Per track (one identity of the footage), over its points in frame order:
- **visibility:** a lost point (behind, out of view, past the horizon) ends a sighting. After a
  track's last segment so far the machine predicts along that segment's end velocity for
  ``track_buffer_s``, then drops the track. A return within the stitch window is bridged;
  otherwise it is a new identity.
- **cleaning** (``trajectories.py``, per class ``max_speed`` + ``jump_tolerance_m``), per sighting:
  ``remove_spikes`` drops a single impossible jump, which is then interpolated; ``split_segments``
  splits at a sustained one (or a gap over ``cleaning.max_gap_s``). Distances are 3D
  (``trajectories`` measures x, y).
- **stitching** (``trajectories.stitch_segments`` on the track's segments): pieces that end and
  start up to ``cleaning.stitch_max_gap_s`` apart and pass its test (the jump, the stitch radius
  around the predicted start) are joined, greedy by the smallest error; the gap is bridged.
  One slot per frame: the chain with the most kept points takes its span first, so a short piece
  the stitch jumped over (a two-frame excursion) is bridged by it, as the pipeline's output is once
  ``len < 2`` / ``min_track_s`` drop that piece.

Every point gets a kind (what the rules made of it), a cause and a part (the identity within its
footage track: 0, then 1 after the first new identity, ...). The cause is measured against the
machine at rest (08 (c), decided 27 Oct): Live's own result with the phone untouched (the start
pose, the space level), over the same points of the clip (the same prefix when performed):
- footage: an original guess of the footage (whatever Live does), or a point the machine at rest
  also guesses or drops (Live's own setup: R_max, the camera's wobble while filming, the steps);
- live: a guess or a drop on a point the machine at rest places, i.e. caused by the interference;
- none: placed (solid).
An untouched phone gives no live cause. Without a rest baseline the machine at rest is taken to
place every point.

Kinds follow the source codes of tracks.bin: SOLID 0 (placed), PREDICTED 1, STITCHED 2, INTERPOLATED 3,
then DROPPED 4 (nothing drawn) and NOT_YET 255 (not performed yet: the prefix ends there).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SOLID, PREDICTED, STITCHED, INTERPOLATED, DROPPED, NOT_YET = 0, 1, 2, 3, 4, 255
KINDS = ("solid", "predicted", "stitched", "interpolated", "dropped")
NONE, FOOTAGE, LIVE = 0, 1, 2
CAUSES = ("none", "footage", "live")
DRAWN = 0                 # live_projection.DRAWN: the state of a point the virtual camera draws
FIT_S = 0.6               # trajectories.Segment.end_state: the line fit at a segment's ends
SPIKE_PASSES = 3          # trajectories.remove_spikes
EPS_S = 1e-6              # s: the prediction's time limit is inclusive (float times)


# --------------------------------------------------------------------------- the config
def rules_config(cfg) -> dict:
    """The rules' parameters (JSON-able, what rules.js takes) from a groundtrack Config, a config
    yaml, or a run folder / its live/ package (then ``_working/config_used.yaml``)."""
    from .config import Config, load_config

    if not isinstance(cfg, Config):
        p = Path(cfg)
        if p.is_dir():
            run = p.parent if (p / "scene.json").exists() else p
            p = run / "_working" / "config_used.yaml"
        cfg = load_config(p)
    det, clean = cfg["detection"], cfg["cleaning"]
    buf = det.get("track_buffer_s")
    return {
        "version": 1,
        "track_buffer_s": float(buf) if buf is not None else 1.0,
        "max_gap_s": float(clean["max_gap_s"]),
        "stitch": bool(clean.get("stitch", True)),
        "stitch_max_gap_s": float(clean["stitch_max_gap_s"]),
        "groups": {g.name: {"classes": list(g.classes), "max_speed": float(g.max_speed),
                            "jump_tolerance_m": float(g.jump_tolerance_m),
                            "stitch_radius_m": float(g.stitch_radius_m)}
                   for g in cfg.groups.values()},
    }


def class_params(rules: dict, name: str):
    """(max_speed, jump_tolerance_m, stitch_radius_m) of a class name; a class in no group: no
    cleaning (inf, 0, inf), the visibility rule only."""
    for g in rules["groups"].values():
        if name in g["classes"]:
            return float(g["max_speed"]), float(g["jump_tolerance_m"]), float(g["stitch_radius_m"])
    return np.inf, 0.0, np.inf


# --------------------------------------------------------------------------- trajectories.py in 3D
def _dist(P, i, j) -> float:
    d = P[j] - P[i]
    return float(np.hypot(np.hypot(d[0], d[1]), d[2]))      # z = 0: trajectories' hypot(dx, dy)


def step_ok(t, P, i, j, vmax: float, tol: float) -> bool:
    """trajectories._step_ok, in 3D."""
    return _dist(P, i, j) <= vmax * abs(t[j] - t[i]) + tol


def remove_spikes(t, P, vmax: float, tol: float, passes: int = SPIKE_PASSES) -> np.ndarray:
    """trajectories.remove_spikes in 3D: the mask of points to keep after removing isolated jumps
    (out and back in one sample). t (n,), P (n, 3)."""
    keep = np.ones(len(t), bool)
    for _ in range(passes):
        idx = np.flatnonzero(keep)
        if len(idx) < 3:
            break
        bad = []
        for k in range(len(idx)):
            i = idx[k]
            prev_i = idx[k - 1] if k > 0 else None
            next_i = idx[k + 1] if k + 1 < len(idx) else None
            if prev_i is not None and next_i is not None:
                if (not step_ok(t, P, prev_i, i, vmax, tol)
                        and not step_ok(t, P, i, next_i, vmax, tol)
                        and step_ok(t, P, prev_i, next_i, vmax, tol)):
                    bad.append(i)
            elif next_i is not None and k + 2 < len(idx):      # first point
                if not step_ok(t, P, i, next_i, vmax, tol) and step_ok(t, P, next_i, idx[k + 2], vmax, tol):
                    bad.append(i)
            elif prev_i is not None and k >= 2:                # last point
                if not step_ok(t, P, prev_i, i, vmax, tol) and step_ok(t, P, idx[k - 2], prev_i, vmax, tol):
                    bad.append(i)
        if not bad:
            break
        keep[bad] = False
    return keep


def split_segments(t, P, vmax: float, tol: float, max_gap_s: float) -> list[np.ndarray]:
    """trajectories.split_segments in 3D: split where a step is impossible or the time gap too long."""
    if len(t) == 0:
        return []
    cuts = [0]
    for i in range(1, len(t)):
        if (t[i] - t[i - 1] > max_gap_s) or not step_ok(t, P, i - 1, i, vmax, tol):
            cuts.append(i)
    cuts.append(len(t))
    return [np.arange(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


def end_state(t, P, at_end: bool, fit_s: float = FIT_S):
    """trajectories.Segment.end_state in 3D: position and velocity at the end (or start), from a
    line fit over fit_s seconds; fewer than 3 points: the end point, no velocity."""
    t, P = np.asarray(t, float), np.asarray(P, float)
    sel = (t >= t[-1] - fit_s) if at_end else (t <= t[0] + fit_s)
    tt, pp = t[sel], P[sel]
    p = pp[-1].copy() if at_end else pp[0].copy()
    if len(tt) < 3 or np.ptp(tt) <= 0:
        return p, np.zeros(3)
    fits = [np.polyfit(tt, pp[:, c], 1) for c in range(3)]
    tref = tt[-1] if at_end else tt[0]
    return np.array([np.polyval(f, tref) for f in fits]), np.array([f[0] for f in fits])


def _norm(v) -> float:
    return float(np.sqrt(np.sum(np.asarray(v, float) ** 2)))    # z = 0: np.linalg.norm of (x, y)


def stitch_test(end_a, start_b, gap: float, vmax: float, tol: float, radius: float,
                max_gap_s: float) -> tuple[bool, float]:
    """The pair test of trajectories.stitch_segments (end of a, start of b, gap seconds): a gap in
    (0, max_gap_s], a plausible jump, and b's start where the mean of the two velocities puts it
    (within the stitch radius, growing with the gap). Returns (ok, prediction error)."""
    (pa, va), (pb, vb) = end_a, start_b
    if not (gap > 0 and gap <= max_gap_s):
        return False, np.inf
    jump = _norm(pb - pa)
    err = _norm(pb - (pa + 0.5 * (va + vb) * gap))
    return bool(jump <= vmax * gap + tol and err <= radius * (1 + gap)), err


# --------------------------------------------------------------------------- one track
def judge_track(t, P, state, src, vmax: float, tol: float, radius: float, rules: dict, rest=None):
    """The rules on one track: t (L,) seconds, P (L, 3) drawn positions (world), state (L,) the
    projection's states (DRAWN seen, else lost; NOT_YET ends the prefix), src (L,) tracks.bin
    sources (0 seen in the footage); rest: (P, state) of the same track at rest (the start pose,
    rest_baseline), judged over the same prefix for the cause (None: at rest every point placed).
    Returns X (L, 3) where the machine draws each point (NaN: nothing), kind, cause, part."""
    t = np.asarray(t, float)
    state, src = np.asarray(state), np.asarray(src)
    nyet = np.flatnonzero(state == NOT_YET)
    Lp = int(nyet[0]) if len(nyet) else len(t)
    X, kind, part = _judge(t, P, state, Lp, vmax, tol, radius, rules)
    rest_solid = np.ones(len(t), bool)
    if rest is not None:
        rest_solid = _judge(t, rest[0], np.asarray(rest[1]), Lp, vmax, tol, radius, rules)[1] == SOLID
    cause = np.where(src != 0, FOOTAGE, np.where(kind == SOLID, NONE,
                                                 np.where(rest_solid, LIVE, FOOTAGE))).astype(np.uint8)
    cause[Lp:] = NONE
    return X, kind, cause, part


def _judge(t, P, state, Lp, vmax, tol, radius, rules):
    """judge_track without the cause, over the prefix 0..Lp-1: X, kind, part."""
    P = np.asarray(P, float).reshape(-1, 3)
    L = len(t)
    X = np.full((L, 3), np.nan)
    kind = np.full(L, NOT_YET, np.uint8)
    part = np.zeros(L, np.uint16)
    kind[:Lp] = DROPPED
    seen = (state[:Lp] == DRAWN) & np.isfinite(P[:Lp]).all(axis=1)

    # sightings -> spikes -> splits: segments of kept points (track positions)
    segs = []
    r = 0
    while r < Lp:
        if not seen[r]:
            r += 1
            continue
        r1 = r
        while r1 < Lp and seen[r1]:
            r1 += 1
        run = np.arange(r, r1)
        k = run[remove_spikes(t[run], P[run], vmax, tol)]
        segs += [k[s] for s in split_segments(t[k], P[k], vmax, tol, float(rules["max_gap_s"]))]
        r = r1
    chains = stitch_chains(t, P, segs, vmax, tol, radius, rules)

    # one slot per frame: the chain with the most kept points takes its span first (its segments,
    # the dropped spikes inside them, its stitched gaps), then the next (a short piece the stitch
    # jumped over is bridged by the larger chain); parts in the order the chains first appear
    owner = np.full(Lp, -1)
    for c in sorted(range(len(chains)), key=lambda c: (-sum(len(segs[g]) for g in chains[c]), segs[chains[c][0]][0])):
        gs = chains[c]
        for a, g in enumerate(gs):
            s = segs[g]
            for i, j in zip(s[:-1], s[1:]):
                _claim(owner, c, kind, X, s[0], s[-1], i, j, INTERPOLATED, t, P)
            if a + 1 < len(gs):
                _claim(owner, c, kind, X, s[-1] + 1, segs[gs[a + 1]][0] - 1, s[-1], segs[gs[a + 1]][0], STITCHED, t, P)
            free = owner[s] < 0
            owner[s[free]], kind[s[free]], X[s[free]] = c, SOLID, P[s[free]]
    firsts = {}
    for r in np.flatnonzero(owner >= 0):
        firsts.setdefault(int(owner[r]), r)
    pnum = {c: k for k, c in enumerate(sorted(firsts, key=firsts.get))}
    part[:Lp] = [pnum[o] if o >= 0 else 0 for o in owner]

    # the rest: after the latest chain end, predicted along its last velocity for track_buffer_s,
    # then dropped; nothing before the first sighting
    buf = float(rules["track_buffer_s"])
    end_of = {segs[gs[-1]][-1]: c for c, gs in enumerate(chains)}
    last = e = v = None
    for r in range(Lp):
        if owner[r] >= 0:
            if r in end_of and owner[r] == end_of[r]:        # a chain ends here (and owns its end)
                last, e = end_of[r], r
                s = segs[chains[last][-1]]
                v = end_state(t[s], P[s], True)[1]
            continue
        if last is None:
            continue
        part[r] = pnum[last]
        dt = t[r] - t[e]
        if dt <= buf + EPS_S:
            X[r], kind[r] = P[e] + dt * v, PREDICTED
    return X, kind, part


def stitch_chains(t, P, segs, vmax, tol, radius, rules) -> list[list[int]]:
    """trajectories.stitch_segments on one track's segments (in time order): every pair that ends /
    starts up to stitch_max_gap_s apart and passes the stitch test is a candidate; greedy by the
    smallest prediction error, one link per end and per start. -> chains of segment numbers."""
    n = len(segs)
    if not rules.get("stitch", True) or n < 2:
        return [[g] for g in range(n)]
    smax = float(rules["stitch_max_gap_s"])
    ends = [end_state(t[s], P[s], True) for s in segs]
    starts = [end_state(t[s], P[s], False) for s in segs]
    cands = []
    for i in range(n):
        for j in range(i + 1, n):
            gap = float(t[segs[j][0]] - t[segs[i][-1]])
            if gap > smax:
                break
            ok, err = stitch_test(ends[i], starts[j], gap, vmax, tol, radius, smax)
            if ok:
                cands.append((err, i, j))
    nxt, has_prev, used_end = {}, set(), set()
    for _err, i, j in sorted(cands):
        if i in used_end or j in has_prev:
            continue
        nxt[i] = j
        used_end.add(i)
        has_prev.add(j)
    chains = []
    for i in range(n):
        if i in has_prev:
            continue
        chain = [i]
        while chain[-1] in nxt:
            chain.append(nxt[chain[-1]])
        chains.append(chain)
    return chains


def _claim(owner, c, kind, X, r0, r1, i, j, k, t, P):
    """Slots r0..r1 not yet taken (and not i, j) to chain c as kind k, on the line from point i to j."""
    q = np.arange(max(r0, i + 1), min(r1, j - 1) + 1)
    q = q[owner[q] < 0]
    if len(q):
        owner[q], kind[q], X[q] = c, k, _lerp(t, P, i, j, q)


def _lerp(t, P, i, j, q):
    w = ((t[q] - t[i]) / (t[j] - t[i]))[:, None]
    return P[i] + w * (P[j] - P[i])


# --------------------------------------------------------------------------- a clip's tracks
def apply_rules(rules: dict, classes: dict, tid, t, cls, src, P, state, rest=None):
    """The rules on every track of a clip. tid, t, cls, src per point (tracks.bin, any order: the
    points of a track are taken in their order); classes: code -> name (scene.json tracks.classes);
    P (n, 3) drawn positions in the world, state (n,) as judge_track; rest: (P, state) of every
    point at rest (rest_baseline), for the cause. Returns X, kind, cause, part."""
    tid, t, cls, src = (np.asarray(a) for a in (tid, t, cls, src))
    P = np.asarray(P, float).reshape(-1, 3)
    state = np.asarray(state)
    if rest is not None:
        rest = (np.asarray(rest[0], float).reshape(-1, 3), np.asarray(rest[1]))
    n = len(tid)
    X = np.full((n, 3), np.nan)
    kind = np.full(n, NOT_YET, np.uint8)
    cause = np.zeros(n, np.uint8)
    part = np.zeros(n, np.uint16)
    order = np.argsort(tid, kind="stable")
    for rows in np.split(order, np.flatnonzero(np.diff(tid[order])) + 1):
        if not len(rows):
            continue
        vmax, tol, radius = class_params(rules, classes.get(str(int(cls[rows[0]])), ""))
        X[rows], kind[rows], cause[rows], part[rows] = judge_track(
            t[rows].astype(float), P[rows], state[rows], src[rows], vmax, tol, radius, rules,
            None if rest is None else (rest[0][rows], rest[1][rows]))
    return X, kind, cause, part


def rest_baseline(scene, X, mode: str = "ground"):
    """The machine at rest: every true point X (n, 3) drawn with the phone untouched (the start
    camera, the space level; the drawing's mode) -> (X', state), live_projection.project's."""
    from .live_projection import project, start_camera

    return project(scene, start_camera(scene), X, mode)


def load_tracks(live_dir) -> dict:
    """tracks.bin of a scene package as arrays (track_id, frame, t, X (n, 3), speed, class, source)."""
    from .live_export import TRACK_DTYPE

    a = np.fromfile(Path(live_dir) / "tracks.bin", TRACK_DTYPE)
    return {"tid": a["track_id"], "frame": a["frame"], "t": a["t"],
            "X": np.column_stack([a["x"], a["y"], a["z"]]).astype(float),
            "speed": a["speed"], "cls": a["class"], "src": a["source"]}
