"""Live controller: the machine's own rules on the drawn positions (plans/live_controller/08 (c)).

The cleaning is trajectories.py's (same input, same flags), the visibility rule predicts, drops,
stitches; distances are 3D; the cause (against the machine at rest) and the identity of every
point; JS = Python, incremental = batch; timings on Trafalgar 6376."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation as Rot

from groundtrack import live_projection as lp
from groundtrack import live_rules as lr
from groundtrack import trajectories as tj
from groundtrack.config import default_config

REPO = Path(__file__).resolve().parents[1]
LIVE = REPO / "groundtrack" / "desktop_web" / "live"
TRAFALGAR = REPO / "runs" / "trafalgar6376" / "v1" / "live"
FPS = 30.0
VMAX, TOL, RADIUS = 6.0, 1.0, 1.5                 # people (config.py, Trafalgar)
RULES = {"version": 1, "track_buffer_s": 1.5, "max_gap_s": 2.0, "stitch": True, "stitch_max_gap_s": 2.0,
         "groups": {"people": {"classes": ["person"], "max_speed": VMAX, "jump_tolerance_m": TOL,
                               "stitch_radius_m": RADIUS},
                    "vehicles": {"classes": ["car", "bus"], "max_speed": 30.0, "jump_tolerance_m": 3.0,
                                 "stitch_radius_m": 4.0}}}
CLASSES = {"0": "person", "2": "car", "5": "bus", "9": "boat"}


# --------------------------------------------------------------------------- tracks
def walk(rng, n, speed=1.3, z=0.0):
    """A walker at 30 fps: frames, t, P (n, 3) on the plane z, with a little detection jitter."""
    f = np.arange(n)
    head = rng.uniform(0, 2 * np.pi)
    v = speed * np.array([np.sin(head), np.cos(head)])
    turn = np.cumsum(rng.normal(0, 0.02, n))
    step = np.column_stack([np.cos(turn) * v[0] - np.sin(turn) * v[1], np.sin(turn) * v[0] + np.cos(turn) * v[1]]) / FPS
    xy = rng.uniform(-30, 30, 2) + np.cumsum(step, axis=0) + rng.normal(0, 0.04, (n, 2))
    return f, f / FPS, np.column_stack([xy, np.full(n, z)])


def mess(rng, P, spikes=4, jumps=1, zigzag=True):
    """Detection errors: single spikes (out and back), sustained jumps (an id switch), a zigzag."""
    P = P.copy()
    n = len(P)
    for _ in range(spikes):
        k = rng.integers(0, n)
        P[k, :2] += rng.normal(0, 4, 2)
    for _ in range(jumps if n > 7 else 0):
        k = rng.integers(3, n - 3)
        P[k:, :2] += rng.normal(0, 5, 2)
    if zigzag and n > 40:
        k = rng.integers(10, n - 10)
        P[k:k + 6:2, :2] += rng.normal(0, 3, 2)
    return P


def lost_runs(rng, n, runs=2, longest=90):
    """state: DRAWN, with a few lost runs (out of view, behind, past the horizon)."""
    st = np.zeros(n, np.uint8)
    for _ in range(runs):
        k, w = rng.integers(0, n), rng.integers(1, longest)
        st[k:k + w] = rng.choice([lp.BEHIND, lp.OUT_OF_VIEW, lp.PAST_HORIZON])
    return st


def judge(t, P, state=None, src=None, rules=RULES, rest=None):
    n = len(t)
    return lr.judge_track(t, P, np.zeros(n, np.uint8) if state is None else state,
                          np.zeros(n, np.uint8) if src is None else src, VMAX, TOL, RADIUS, rules, rest)


# --------------------------------------------------------------------------- = trajectories.py
@pytest.mark.parametrize("z", [0.0, 3.7])
def test_spikes_and_splits_are_trajectories_py(z):
    """On level ground (z constant) the 3D rules give exactly trajectories.py's flags."""
    rng = np.random.default_rng(1)
    spikes = splits = 0
    for _ in range(300):
        n = int(rng.integers(2, 200))
        f, t, P = walk(rng, n, speed=rng.uniform(0.5, 2.5), z=z)
        P = mess(rng, P, spikes=int(rng.integers(0, 8)), jumps=int(rng.integers(0, 3)))
        keep = rng.random(n) > 0.05                                 # missing detections
        t, P = t[keep], P[keep]
        k_ref = tj.remove_spikes(t, P[:, 0], P[:, 1], VMAX, TOL)
        k = lr.remove_spikes(t, P, VMAX, TOL)
        assert (k == k_ref).all()
        s_ref = tj.split_segments(t[k], P[k, 0], P[k, 1], VMAX, TOL, 2.0)
        s = lr.split_segments(t[k], P[k], VMAX, TOL, 2.0)
        assert len(s) == len(s_ref) and all((a == b).all() for a, b in zip(s, s_ref))
        spikes += int((~k).sum())
        splits += len(s) - 1
    print(f"\n= trajectories.py (z = {z}): 300 tracks, {spikes} spikes, {splits} splits, the same flags")
    assert spikes > 300 and splits > 100


def segment(tid, f, t, P):
    df = pd.DataFrame({"frame": f, "time_s": t, "gx": P[:, 0], "gy": P[:, 1], "u": 0.0, "v": 0.0,
                       "predicted": False, "class": "person"})
    return tj.Segment(int(tid), [int(tid)], "person", "people", df)


def test_end_state_and_stitch_test_are_trajectories_py():
    rng = np.random.default_rng(2)
    for _ in range(200):
        n = int(rng.integers(1, 40))
        f, t, P = walk(rng, n, z=2.5)
        seg = segment(1, f, t, P)
        for at_end in (True, False):
            p, v = lr.end_state(t, P, at_end)
            p_ref, v_ref = seg.end_state(at_end)
            assert np.abs(p[:2] - p_ref).max() < 1e-9 and np.abs(v[:2] - v_ref).max() < 1e-9
            assert np.isclose(p[2], 2.5) and abs(v[2]) < 1e-9
    # the stitch test: pairs of segments, stitch_segments decides; consecutive pairs only
    cfg = default_config()
    links = n_ok = 0
    for _ in range(400):
        f, t, P = walk(rng, 120)
        cut, gap = int(rng.integers(5, 100)), int(rng.integers(1, 75))
        P = P.copy()
        P[cut + gap:, :2] += rng.normal(0, rng.choice([0.3, 1.5, 4.0]), 2)
        a, b = np.arange(cut), np.arange(cut + gap, 120)
        if len(b) == 0:
            continue
        out = tj.stitch_segments([segment(1, f[a], t[a], P[a]), segment(2, f[b], t[b], P[b])], cfg,
                                 float(cfg["cleaning"]["stitch_max_gap_s"]))
        ok, _ = lr.stitch_test(lr.end_state(t[a], P[a], True), lr.end_state(t[b], P[b], False),
                               float(t[b[0]] - t[a[-1]]), VMAX, TOL, RADIUS, 2.0)
        assert ok == (len(out) == 1)
        links += 1
        n_ok += ok
    print(f"\nstitch test = stitch_segments: {links} pairs, {n_ok} stitched")
    assert 50 < n_ok < links - 50


def pipeline_sources(f, t, P, seen):
    """What trajectories.py makes of a track whose detections are only where seen: each sighting a
    tracker id, then remove_spikes, split_segments, stitch_segments and _process_segment's sources
    (min_track_s 0). -> [(frames, sources)] per output track."""
    cfg = default_config()
    cfg.data["cleaning"]["min_track_s"] = 0.0
    group = cfg.groups["people"]
    segs, tid = [], 1
    runs = np.split(np.flatnonzero(seen), np.flatnonzero(np.diff(np.flatnonzero(seen)) > 1) + 1)
    for run in runs:
        if not len(run):
            continue
        k = run[tj.remove_spikes(t[run], P[run, 0], P[run, 1], VMAX, TOL)]
        for s in tj.split_segments(t[k], P[k, 0], P[k, 1], VMAX, TOL, float(cfg["cleaning"]["max_gap_s"])):
            segs.append(segment(tid, f[k[s]], t[k[s]], P[k[s]]))
            tid += 1
    segs = tj.stitch_segments(segs, cfg, float(cfg["cleaning"]["stitch_max_gap_s"]))
    out = []
    for seg in sorted(segs, key=lambda s: s.t0):
        res = tj._process_segment(seg, group, cfg, None, FPS, 1, 1 / FPS, 2)
        if res is not None:
            out.append((res[0]["frame"].to_numpy(), res[0]["source"].to_numpy(), len(seg.df)))
    return out


def test_the_rules_are_the_pipeline_on_the_same_track():
    """Sightings, spikes, splits and stitches: the pipeline's own functions make its output tracks
    (sources detected / interpolated / stitched). One slot per frame, the larger track first: on
    the frames each one takes, the kinds are its sources and it is one identity; the rest was
    predicted, then dropped."""
    rng = np.random.default_rng(3)
    names = {"detected": lr.SOLID, "interpolated": lr.INTERPOLATED, "stitched": lr.STITCHED}
    seen_kinds = np.zeros(5, int)
    tracks = jumped = 0
    for _ in range(250):
        n = int(rng.integers(30, 300))
        f, t, P = walk(rng, n, speed=rng.uniform(0.3, 2.0))
        P = mess(rng, P, spikes=int(rng.integers(0, 5)), jumps=int(rng.integers(0, 2)))
        state = lost_runs(rng, n, runs=int(rng.integers(0, 4)), longest=80)
        seen = state == lp.DRAWN
        X, kind, cause, part = judge(t, P, state)
        ref = pipeline_sources(f, t, P, seen)
        taken = np.zeros(n, bool)
        parts = []
        for fr, src, size in sorted(ref, key=lambda r: (-r[2], r[0][0])):
            mine = ~taken[fr]
            jumped += int((~mine).any() or not mine.any())
            if not mine.any():
                continue
            fr, src = fr[mine], src[mine]
            assert (kind[fr] == [names[s] for s in src]).all()
            assert len(set(part[fr])) == 1
            parts.append(part[fr[0]])
            taken[fr] = True
        assert len(parts) == len(set(parts))                       # one identity per output track
        rest = ~taken & (np.arange(n) >= (np.flatnonzero(seen)[0] if seen.any() else n))
        assert np.isin(kind[rest & ~seen], [lr.PREDICTED, lr.DROPPED]).all()
        assert np.isin(kind[rest & seen], [lr.SOLID, lr.PREDICTED, lr.DROPPED]).all()   # 1-point pieces, spikes
        seen_kinds += np.bincount(kind, minlength=5)[:5]
        tracks += len(ref)
    print(f"\n= the pipeline: {tracks} output tracks ({jumped} overlapped by a larger one's stitch); kinds "
          f"solid / predicted / stitched / interpolated / dropped: {seen_kinds.tolist()}")
    assert (seen_kinds > 100).all() and jumped > 0


# --------------------------------------------------------------------------- in 3D, in the world
def test_jumps_are_measured_in_3d():
    rng = np.random.default_rng(4)
    f, t, P = walk(rng, 90)
    up = P.copy()
    up[40, 2] += 3.0                                               # thrown up 3 m, back down
    up[70:, 2] += 4.0                                              # and stays 4 m higher
    X, kind, cause, part = judge(t, up)
    assert kind[40] == lr.INTERPOLATED and cause[40] == lr.LIVE
    assert part[69] == 0 and part[70] == 1
    assert tj.remove_spikes(t, up[:, 0], up[:, 1], VMAX, TOL).all()     # in plan: nothing to see
    # a fixed tilt of the whole space changes nothing (distances kept); level z = the 2D rules
    P = mess(rng, P, spikes=3, jumps=1)
    S = Rot.from_rotvec([0.3, -0.2, 0.5]).as_matrix()
    a, b = judge(t, P), judge(t, lp.apply_space(S, [3.0, -4.0, 1.9], P))
    for u, v in zip(a[1:], b[1:]):
        assert (u == v).all()
    assert np.nanmax(np.abs(lp.apply_space(S, [3.0, -4.0, 1.9], a[0]) - b[0])) < 1e-9


def tilting(rng, rate_deg_s, d=40.0, n=150, t_on=(1.5, 3.0)):
    """A walker d m in front of C0, drawn while the space turns at rate (deg/s) about a horizontal
    axis between t_on: X' = C0 + S(t) (X - C0), S(t) frozen outside the turn."""
    C0 = np.array([0.0, 0.0, 1.9])
    f, t, P = walk(rng, n, speed=1.2)
    P[:, :2] += np.array([0.0, d]) - P[0, :2]
    ang = np.radians(rate_deg_s) * np.clip(t - t_on[0], 0, t_on[1] - t_on[0])
    Xw = np.array([lp.apply_space(Rot.from_rotvec([a, 0, 0]).as_matrix(), C0, p[None])[0] for a, p in zip(ang, P)])
    return t, Xw


def test_space_tilts_slow_fool_the_machine_fast_get_caught():
    rng = np.random.default_rng(5)
    t, Xw = tilting(rng, 10.0)                                     # 10 deg/s at 40 m: 0.23 m per frame
    X, kind, cause, part = judge(t, Xw)
    assert (kind == lr.SOLID).all() and (part == 0).all() and (cause == lr.NONE).all()
    t, Xw = tilting(rng, 120.0)                                    # 120 deg/s: 2.8 m per frame
    X, kind, cause, part = judge(t, Xw)
    assert part.max() > 30 and (cause[kind != lr.SOLID] == lr.LIVE).all()
    near = tilting(rng, 120.0, d=5.0)                              # near the camera the same turn is slow
    assert (judge(*near)[3] == 0).all()


def test_the_level_glide_is_judged_like_any_fast_tilt():
    """Double-tap: 25 deg back to level over 0.3 s. Far points are caught, near ones are not."""
    C0 = np.array([0.0, 0.0, 1.9])
    for d, caught in ((45.0, True), (4.0, False)):
        rng = np.random.default_rng(6)
        tilt = Rot.from_rotvec([np.radians(25.0), 0, 0]).as_matrix()
        cl = lp.Clutch().press(np.eye(3)).move(tilt).release(tilt)
        assert np.isclose(np.degrees(cl.angle), 25.0)
        f, t, P = walk(rng, 120, speed=1.2)
        P[:, :2] += np.array([0.0, d]) - P[0, :2]
        Xw = np.empty_like(P)
        for k in range(len(t)):
            if k == 60:
                cl.level(t[k])
            cl.move(np.eye(3), t[k])
            Xw[k] = lp.apply_space(cl.S, C0, P[k:k + 1])[0]
        assert not cl.tilted
        _, kind, cause, part = judge(t, Xw)
        assert (part.max() > 0) == caught, d
        if caught:
            changed = np.flatnonzero(np.diff(part.astype(int)))
            assert t[changed].min() >= t[60] - 1e-9 and t[changed].max() <= t[60] + 0.3 + 1e-9


# --------------------------------------------------------------------------- visibility
def test_lost_points_are_predicted_then_dropped_and_a_return_is_stitched_or_new():
    rng = np.random.default_rng(7)
    f, t, P = walk(rng, 240, speed=1.4)
    # lost for 1 s, back where it was heading: stitched, the same person
    st = np.zeros(240, np.uint8)
    st[100:130] = lp.OUT_OF_VIEW
    X, kind, cause, part = judge(t, P, st)
    assert (kind[100:130] == lr.STITCHED).all() and (part == 0).all() and (cause[100:130] == lr.LIVE).all()
    w = (t[100:130] - t[99]) / (t[130] - t[99])
    assert np.abs(X[100:130] - (P[99] + w[:, None] * (P[130] - P[99]))).max() < 1e-9
    # while still lost (performed, the prefix ends there): predicted along the last velocity, then dropped
    st2 = np.zeros(240, np.uint8)
    st2[100:] = lp.BEHIND
    st2[200:] = lr.NOT_YET
    X, kind, cause, part = judge(t, P, st2)
    v = lr.end_state(t[:100], P[:100], True)[1]
    buf = t[100:200] - t[99] <= 1.5 + 1e-6
    assert (kind[100:200][buf] == lr.PREDICTED).all() and (kind[100:200][~buf] == lr.DROPPED).all()
    assert buf.sum() == 45 and np.isnan(X[100:200][~buf]).all() and (kind[200:] == lr.NOT_YET).all()
    q = np.arange(100, 145)
    assert np.abs(X[q] - (P[99] + np.outer(t[q] - t[99], v))).max() < 1e-9
    assert np.hypot(*v[:2]) > 1.0
    # back somewhere else (a jump the stitch radius does not allow): a new person
    P3 = P.copy()
    P3[130:, :2] += [4.0, -3.0]
    X, kind, cause, part = judge(t, P3, st)
    assert (kind[100:130] == lr.PREDICTED).all() and part[99] == 0 and part[130] == 1
    # lost for longer than the stitch window: predicted for track_buffer_s, dropped, a new person
    st4 = np.zeros(240, np.uint8)
    st4[60:150] = lp.PAST_HORIZON
    X, kind, cause, part = judge(t, P, st4)
    assert (kind[60:105] == lr.PREDICTED).all() and (kind[105:150] == lr.DROPPED).all()
    assert part[59] == 0 and part[150] == 1
    # not seen yet: nothing to draw
    st5 = np.zeros(240, np.uint8)
    st5[:20] = lp.OUT_OF_VIEW
    X, kind, cause, part = judge(t, P, st5)
    assert (kind[:20] == lr.DROPPED).all() and (part == 0).all() and np.isnan(X[:20]).all()
    # stitch off: every return is a new identity
    X, kind, cause, part = judge(t, P, st, rules={**RULES, "stitch": False})
    assert part[130] == 1 and (kind[100:130] == lr.PREDICTED).all()


def test_cause_footage_stays_footage():
    rng = np.random.default_rng(8)
    f, t, P = walk(rng, 200)
    src = np.zeros(200, np.uint8)
    src[50:70] = 2                                                  # stitched in the footage
    src[120:125] = 1                                                # predicted in the footage
    st = np.zeros(200, np.uint8)
    st[60:100] = lp.OUT_OF_VIEW                                     # Live loses half the stitched gap too
    P2 = P.copy()
    P2[150] += [5.0, 0.0, 0.0]                                      # a Live spike on a seen point
    P2[122] += [5.0, 0.0, 0.0]                                      # and on a footage guess
    X, kind, cause, part = judge(t, P2, st, src)
    assert (cause[src != 0] == lr.FOOTAGE).all()
    assert (kind[50:60] == lr.SOLID).all() and (kind[60:100] == lr.STITCHED).all()
    assert kind[150] == lr.INTERPOLATED and cause[150] == lr.LIVE
    assert kind[122] == lr.INTERPOLATED and cause[122] == lr.FOOTAGE
    assert (cause[70:100] == lr.LIVE).all()
    plain = (src == 0) & (kind == lr.SOLID)
    assert (cause[plain] == lr.NONE).all() and plain.sum() > 100


def test_the_cause_is_measured_at_rest():
    """08 (c): live = what differs from the machine at rest (the start pose, the phone untouched),
    not from the footage. The setup's own losses and errors are the footage's; an untouched phone
    gives no live cause; performed, the rest is judged over the same prefix."""
    rng = np.random.default_rng(11)
    f, t, P = walk(rng, 240, speed=1.4)
    P[150, :2] += [4.0, 3.0]                       # a detection spike: the machine at rest drops it too
    P[175:, :2] += [6.0, -4.0]                     # an id switch: a new identity at rest too
    rest_st = np.zeros(240, np.uint8)
    rest_st[60:80] = lp.PAST_HORIZON               # beyond R_max at rest (the setup)
    rest = (np.where(rest_st[:, None] == lp.DRAWN, P, np.nan), rest_st)
    # untouched: the same drawing, no live cause; the setup's guesses are the footage's
    X, kind, cause, part = judge(t, *rest, rest=rest)
    assert not (cause == lr.LIVE).any()
    assert (kind[60:80] == lr.STITCHED).all() and (cause[60:80] == lr.FOOTAGE).all()
    assert kind[150] == lr.INTERPOLATED and cause[150] == lr.FOOTAGE and part[175] == 1
    # the phone turned: more of the clip out of view, another spike
    st = rest_st.copy()
    st[100:130] = lp.OUT_OF_VIEW
    P2 = P.copy()
    P2[200, :2] += [0.0, 5.0]
    P2[st != lp.DRAWN] = np.nan
    X, kind, cause, part = judge(t, P2, st, rest=rest)
    assert (kind[100:130] == lr.STITCHED).all() and kind[200] == lr.INTERPOLATED
    assert ((cause == lr.LIVE) == np.isin(np.arange(240), np.r_[100:130, 200])).all()
    assert (cause[60:80] == lr.FOOTAGE).all() and cause[150] == lr.FOOTAGE
    # performed up to one point after the switch: that point looks like a spike, to the machine at
    # rest too (over the same prefix; over the whole clip it places it)
    pre = np.where(np.arange(240) <= 175, st, lr.NOT_YET).astype(np.uint8)
    X, kind, cause, part = judge(t, P2, pre, rest=rest)
    assert kind[175] == lr.PREDICTED and cause[175] == lr.FOOTAGE
    assert lr._judge(t, rest[0], rest[1], 240, VMAX, TOL, RADIUS, RULES)[1][175] == lr.SOLID
    # without a rest baseline the machine at rest places every point: the setup's guesses are live
    X, kind, cause, part = judge(t, P2, st)
    assert (cause[60:80] == lr.LIVE).all() and cause[150] == lr.LIVE


def test_rules_config_from_the_run():
    cfg = default_config()
    r = lr.rules_config(cfg)
    assert r["track_buffer_s"] == cfg["detection"]["track_buffer_s"]
    assert r["groups"]["people"] == {"classes": ["person"], "max_speed": 6.0, "jump_tolerance_m": 1.0,
                                     "stitch_radius_m": 1.5}
    assert lr.class_params(r, "car") == (70.0, 3.0, 4.0) and lr.class_params(r, "boat")[0] == np.inf
    json.dumps(r)
    if not (TRAFALGAR.parent / "_working" / "config_used.yaml").exists():
        pytest.skip("Trafalgar 6376 run is missing (gitignored)")
    r = lr.rules_config(TRAFALGAR)
    assert r["track_buffer_s"] == 1.5 and r["stitch_max_gap_s"] == 2.0
    assert r["groups"]["vehicles"]["max_speed"] == 30.0 and r["groups"]["buses"]["classes"] == ["bus"]


# --------------------------------------------------------------------------- JS = Python
def random_clip(rng, n_tracks=60, frames=600):
    """Tracks as tracks.bin holds them (sorted by frame, then track), footage sources, a few
    classes; the machine at rest (Pr, sr: drawn positions with errors and lost runs, the setup's)
    and the drawing (P, state): the rest, plus the interference on most tracks (more lost runs,
    spikes, a slow tilt)."""
    rows = []
    for k in range(n_tracks):
        n = int(rng.integers(2, 400))
        f0 = int(rng.integers(0, frames - 2))
        n = min(n, frames - f0)
        _, _, P = walk(rng, n, speed=rng.uniform(0.3, 2.5), z=0.0)
        P[:, 2] = rng.normal(0, 0.3) + np.cumsum(rng.normal(0, 0.01, n))   # sloping, tilted ground
        Pr = mess(rng, P, spikes=int(rng.integers(0, 6)), jumps=int(rng.integers(0, 3)))
        sr = lost_runs(rng, n, runs=int(rng.integers(0, 3)), longest=100)
        P, st = Pr.copy(), sr.copy()
        if rng.random() < 0.75:                                            # moved
            st = np.where(st == lp.DRAWN, lost_runs(rng, n, runs=int(rng.integers(0, 3)), longest=100), st)
            P = mess(rng, P, spikes=int(rng.integers(0, 3)), jumps=0, zigzag=False)
            P[:, 2] += np.linspace(0, rng.normal(0, 0.5), n)
        src = np.where(rng.random(n) < 0.15, rng.integers(1, 4, n), 0)
        cls = rng.choice([0, 0, 0, 2, 5, 9])
        for i in range(n):
            rows.append((k + 1, f0 + i, P[i], st[i], src[i], cls, Pr[i], sr[i]))
    rows.sort(key=lambda r: (r[1], r[0]))
    tid = np.array([r[0] for r in rows], np.uint32)
    frame = np.array([r[1] for r in rows], np.uint32)
    t = (frame / FPS).astype(np.float32)
    clip = {"tid": tid, "frame": frame, "t": t, "src": np.array([r[4] for r in rows], np.uint8),
            "cls": np.array([r[5] for r in rows], np.uint8)}
    for kP, kS, a, b in (("P", "state", 2, 3), ("Pr", "sr", 6, 7)):
        P = np.array([r[a] for r in rows]).astype(np.float32).astype(float)    # as the view holds them
        clip[kS] = np.array([r[b] for r in rows], np.uint8)
        P[clip[kS] != lp.DRAWN] = np.nan
        clip[kP] = P
    return clip


RUNNER = r"""
import { readFileSync, writeFileSync } from "node:fs";
import { trackIndex } from "./projection.js";
import * as R from "./rules.js";
const dir = process.argv[2];
const rd = (f, T) => { const b = readFileSync(dir + "/" + f); return new T(Uint8Array.from(b).buffer); };
const wr = (f, a) => writeFileSync(dir + "/" + f, Buffer.from(a.buffer, a.byteOffset, a.byteLength));
const job = JSON.parse(readFileSync(dir + "/job.json", "utf8"));
const tid = rd("tid.u32", Uint32Array), frame = rd("frame.u32", Uint32Array), t = rd("t.f32", Float32Array);
const P = rd("P.f32", Float32Array), state = rd("state.u8", Uint8Array), src = rd("src.u8", Uint8Array);
const cls = rd("cls.u8", Uint8Array), n = tid.length, F = frame[n - 1] + 1;
const rest = { X: rd("Pr.f32", Float32Array), state: rd("sr.u8", Uint8Array) };
const frameStart = new Uint32Array(F + 1);
for (let i = 0, f = 0; f <= F; f++) { while (i < n && frame[i] < f) i++; frameStart[f] = i; }
const tracks = { t, cls, src, frame, frameStart, index: trackIndex(tid) };
const dump = (pre, o) => { wr(pre + "X.f64", Float64Array.from(o.X)); wr(pre + "kind.u8", o.kind); wr(pre + "cause.u8", o.cause); wr(pre + "part.u16", o.part); };
let threw = 0;
try { new R.Rules(tracks, job.rules, job.classes, { out64: true }); } catch (e) { threw = 1; }
wr("threw.u8", Uint8Array.of(threw));
// batch; and with no rest (every point placed at rest); untouched (drawn as at rest)
const rb = new R.Rules(tracks, job.rules, job.classes, { out64: true, rest });
dump("b_", rb.all(P, state));
dump("n_", new R.Rules(tracks, job.rules, job.classes, { out64: true, rest: null }).all(P, state));
dump("r_", rb.all(rest.X, rest.state));
// incremental, frame by frame (performed): not yet, then each frame as it comes; checkpoints
const ri = new R.Rules(tracks, job.rules, job.classes, { out64: true, rest });
const ps = new Uint8Array(n).fill(R.NOT_YET), dirty = [];
for (let f = 0; f < F; f++) {
  for (let i = frameStart[f]; i < frameStart[f + 1]; i++) ps[i] = state[i];
  ri.frames(P, ps, f);
  dirty.push(ri.dirty[0], ri.dirty[1]);
  if (job.checks.includes(f)) dump("i" + f + "_", ri.out);
}
dump("i_", ri.out);
// scrub back to frame 100 and play again
for (let i = frameStart[101]; i < n; i++) ps[i] = R.NOT_YET;
ri.frames(P, ps, 101, F - 1);
dump("s_", ri.out);
wr("dirty.f64", Float64Array.from(dirty));
// single-track helpers
const one = job.one;
const T = Float64Array.from(one.t), H = Float64Array.from(one.H), Q = Float64Array.from(one.P);
wr("keep.u8", R.removeSpikes(T, H, 6, 1));
wr("cuts.i32", R.splitSegments(T, H, 6, 1, 2.0));
wr("end.f64", Float64Array.from([...R.segmentEndState(T, H, true), ...R.segmentEndState(T, H, false)]));
const oneRest = { X: Float64Array.from(one.rP), state: Uint8Array.from(one.rs) }, oneSt = Uint8Array.from(one.state);
const oneSrc = Uint8Array.from(one.src);
dump("j_", R.judgeTrack(T, Q, oneSt, oneSrc, 6, 1, 1.5, job.rules, oneRest));
// the same track performed point by point (Rules.frames) and judged on every prefix (judgeTrack)
const m = T.length, ids = (k) => Uint32Array.from({ length: k }, (_, i) => i);
const oneTr = { t: Float32Array.from(one.t), cls: new Uint8Array(m), src: oneSrc, frame: ids(m), frameStart: ids(m + 1),
                index: trackIndex(new Uint32Array(m).fill(1)) };
const r1 = new R.Rules(oneTr, job.rules, job.classes, { out64: true, rest: oneRest });
const ps1 = new Uint8Array(m).fill(R.NOT_YET), pk = new Uint8Array(2 * m * m), pc = new Uint8Array(2 * m * m);
for (let f = 0; f < m; f++) {
  ps1[f] = oneSt[f];
  r1.frames(Q, ps1, f);
  const jp = R.judgeTrack(T, Q, ps1, oneSrc, 6, 1, 1.5, job.rules, oneRest);
  pk.set(r1.out.kind, 2 * f * m); pk.set(jp.kind, (2 * f + 1) * m);
  pc.set(r1.out.cause, 2 * f * m); pc.set(jp.cause, (2 * f + 1) * m);
}
wr("pk.u8", pk); wr("pc.u8", pc);
"""


def run_js(tmp_path, clip, checks, one):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    for f in ("projection.js", "rules.js"):
        (tmp_path / f).write_bytes((LIVE / f).read_bytes())
    (tmp_path / "runner.mjs").write_text(RUNNER, encoding="utf-8")
    (tmp_path / "job.json").write_text(json.dumps({"rules": RULES, "classes": CLASSES, "checks": checks, "one": one}),
                                       encoding="utf-8")
    for k, dt in (("tid", "<u4"), ("frame", "<u4"), ("t", "<f4"), ("state", "u1"), ("src", "u1"), ("cls", "u1"),
                  ("sr", "u1"), ("P", "<f4"), ("Pr", "<f4")):
        clip[k].astype(dt).tofile(tmp_path / f"{k}.{ {'<u4': 'u32', '<f4': 'f32', 'u1': 'u8'}[dt] }")
    r = subprocess.run([node, "runner.mjs", str(tmp_path)], cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr

    def load(pre):
        return (np.fromfile(tmp_path / f"{pre}X.f64").reshape(-1, 3), np.fromfile(tmp_path / f"{pre}kind.u8", np.uint8),
                np.fromfile(tmp_path / f"{pre}cause.u8", np.uint8), np.fromfile(tmp_path / f"{pre}part.u16", np.uint16))
    return load


def same(js, py, what):
    X, kind, cause, part = js
    Xp, kp, cp, pp = py
    assert (kind == kp).all() and (cause == cp).all() and (part == pp).all(), what
    both = np.isfinite(Xp)
    assert (np.isfinite(X) == both).all(), what
    return np.abs(X[both] - Xp[both]).max() if both.any() else 0.0


def test_js_equals_python_batch_and_incremental(tmp_path):
    rng = np.random.default_rng(9)
    clip = random_clip(rng)
    F = int(clip["frame"].max()) + 1
    checks = [40, 150, 333, 501]
    _, t1, P1 = walk(rng, 150)
    t1 = t1.astype(np.float32).astype(float)
    rP1 = mess(rng, P1, spikes=5, jumps=2)
    H1 = rP1.copy()
    rs1 = lost_runs(rng, 150, runs=1, longest=40)
    st1 = np.where(rs1 == lp.DRAWN, lost_runs(rng, 150, runs=2, longest=40), rs1)
    P1 = mess(rng, rP1, spikes=2, jumps=0, zigzag=False)
    rP1[rs1 != lp.DRAWN] = np.nan
    P1[st1 != lp.DRAWN] = np.nan
    flat = lambda a: np.nan_to_num(a, nan=1e300).ravel().tolist()      # JSON: no NaN (not seen anyway)
    one = {"t": t1.tolist(), "H": flat(H1), "P": flat(P1), "state": st1.tolist(),
           "src": np.where(rng.random(150) < 0.2, 1, 0).tolist(), "rP": flat(rP1), "rs": rs1.tolist()}
    load = run_js(tmp_path, clip, checks, one)
    args = (RULES, CLASSES, clip["tid"], clip["t"], clip["cls"], clip["src"], clip["P"])
    rest = (clip["Pr"], clip["sr"])
    batch = lr.apply_rules(*args, clip["state"], rest=rest)
    worst = same(load("b_"), batch, "batch")
    worst = max(worst, same(load("n_"), lr.apply_rules(*args, clip["state"]), "no rest"))
    at_rest = lr.apply_rules(*args[:-1], clip["Pr"], clip["sr"], rest=rest)
    worst = max(worst, same(load("r_"), at_rest, "drawn as at rest"))
    assert np.fromfile(tmp_path / "threw.u8", np.uint8)[0] == 1          # no rest given: an error
    # incremental = batch on the clip so far, at every checkpoint, at the end, after a scrub back
    for f in checks:
        st = np.where(clip["frame"] <= f, clip["state"], lr.NOT_YET).astype(np.uint8)
        worst = max(worst, same(load(f"i{f}_"), lr.apply_rules(*args, st, rest=rest), f"incremental at frame {f}"))
    worst = max(worst, same(load("i_"), batch, "incremental at the end"))
    st = np.where(clip["frame"] <= 100, clip["state"], lr.NOT_YET).astype(np.uint8)
    worst = max(worst, same(load("s_"), lr.apply_rules(*args, st, rest=rest), "after a scrub back"))
    kinds, causes = np.bincount(batch[1], minlength=5), np.bincount(batch[2], minlength=3)
    print(f"\nJS = Python: {len(clip['tid'])} points in {len(np.unique(clip['tid']))} tracks over {F} frames; "
          f"kinds {kinds[:5].tolist()}, causes {causes.tolist()}, parts up to {batch[3].max()}, "
          f"max |X| difference {worst:.1e} m")
    assert worst < 1e-6 and (kinds[:5] > 50).all() and (causes > 50).all()
    assert not (at_rest[2] == lr.LIVE).any()                                # untouched: no live cause
    dirty = np.fromfile(tmp_path / "dirty.f64").reshape(-1, 2)
    assert (dirty[:, 1] <= np.arange(F)).all()                       # nothing after the frame performed
    # the single-track helpers
    t1f = np.asarray(one["t"])
    k = lr.remove_spikes(t1f, H1, VMAX, TOL)
    assert (np.fromfile(tmp_path / "keep.u8", np.uint8).astype(bool) == tj.remove_spikes(t1f, H1[:, 0], H1[:, 1], VMAX, TOL)).all()
    cuts = [s[0] for s in lr.split_segments(t1f, H1, VMAX, TOL, 2.0)] + [150]
    assert np.fromfile(tmp_path / "cuts.i32", np.int32).tolist() == cuts
    e = np.fromfile(tmp_path / "end.f64").reshape(2, 2, 3)
    for j, at_end in enumerate((True, False)):
        p, v = lr.end_state(t1f, H1, at_end)
        assert np.abs(e[j, 0] - p).max() < 1e-9 and np.abs(e[j, 1] - v).max() < 1e-9
    src1 = np.asarray(one["src"])
    py = lr.judge_track(t1f, P1, st1, src1, VMAX, TOL, RADIUS, RULES, (rP1, rs1))
    same(load("j_"), py, "judgeTrack")
    assert not k.all()
    # performed point by point (JS Rules.frames and judgeTrack) = Python on every prefix; on some
    # prefixes the rest over the prefix gives another cause than the rest over the whole track would
    pk = np.fromfile(tmp_path / "pk.u8", np.uint8).reshape(150, 2, 150)
    pc = np.fromfile(tmp_path / "pc.u8", np.uint8).reshape(150, 2, 150)
    full = lr._judge(t1f, rP1, rs1, 150, VMAX, TOL, RADIUS, RULES)[1] != lr.SOLID
    differ = 0
    for f in range(150):
        pre = np.where(np.arange(150) <= f, st1, lr.NOT_YET).astype(np.uint8)
        _, kd, c, _ = lr.judge_track(t1f, P1, pre, src1, VMAX, TOL, RADIUS, RULES, (rP1, rs1))
        assert (pk[f] == kd).all() and (pc[f] == c).all(), f"performed, prefix {f}"
        cf = np.where(src1 != 0, lr.FOOTAGE, np.where(kd == lr.SOLID, lr.NONE, np.where(full, lr.FOOTAGE, lr.LIVE)))
        differ += bool((c[:f + 1] != cf[:f + 1]).any())
    print(f"one track: {differ} of 150 prefixes where the rest over the prefix changes a cause")
    assert differ > 0


# --------------------------------------------------------------------------- Trafalgar 6376
BENCH = r"""
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
const [dir, live] = process.argv.slice(2);
const L = await import(pathToFileURL(live + "/projection.js").href);
const R = await import(pathToFileURL(live + "/rules.js").href);
const meta = JSON.parse(readFileSync(dir + "/scene.json", "utf8"));
const rules = JSON.parse(readFileSync(process.argv[4], "utf8"));
const buf = readFileSync(dir + "/tracks.bin"), dv = new DataView(Uint8Array.from(buf).buffer), n = buf.length / 30;
const tid = new Uint32Array(n), frame = new Uint32Array(n), t = new Float32Array(n), X = new Float32Array(3 * n);
const cls = new Uint8Array(n), src = new Uint8Array(n);
for (let i = 0; i < n; i++) {
  const o = 30 * i;
  tid[i] = dv.getUint32(o, true); frame[i] = dv.getUint32(o + 4, true); t[i] = dv.getFloat32(o + 8, true);
  for (let c = 0; c < 3; c++) X[3 * i + c] = dv.getFloat32(o + 12 + 4 * c, true);
  cls[i] = dv.getUint8(o + 28); src[i] = dv.getUint8(o + 29);
}
const F = frame[n - 1] + 1, frameStart = new Uint32Array(F + 1);
for (let i = 0, f = 0; f <= F; f++) { while (i < n && frame[i] < f) i++; frameStart[f] = i; }
const scene = L.prepareScene(meta, new Float32Array(Uint8Array.from(readFileSync(dir + "/terrain.f32")).buffer));
const tracks = { t, X, cls, src, frame, frameStart, index: L.trackIndex(tid) };
const res = {};
let t0 = performance.now();
const rl = new R.Rules(tracks, rules, meta.tracks.classes, { scene });       // + the machine at rest
res.build_ms = performance.now() - t0;
const out = { X: new Float32Array(3 * n), state: new Uint8Array(n) };
const count = (o) => { const k = [0, 0, 0, 0, 0], c = [0, 0, 0]; for (let i = 0; i < n; i++) { k[o.kind[i]]++; c[o.cause[i]]++; } return { kind: k, cause: c }; };
const nLive = (o) => { let c = 0; for (let i = 0; i < n; i++) c += o.cause[i] === R.LIVE; return c; };
const zero = { q: [1, 0, 0, 0] }, D = new Float64Array(9), raw = L.startCamera(scene), cam = L.startCamera(scene);
const q = (ax, deg) => { const a = deg * Math.PI / 360, s = Math.sin(a) / Math.hypot(...ax); return [Math.cos(a), ax[0] * s, ax[1] * s, ax[2] * s]; };
// the untouched phone (at its zero), the view's path: pose -> clutch -> camera -> drawing -> space
const cl0 = new L.Clutch();
L.poseToCamera(scene, zero, zero, 1, 1, { out: raw, delta: D });
cl0.move(D, 0);
L.turnCamera(scene, cl0.Dc, cam);
L.project(scene, cam, X, out);
L.applySpace(cl0.S, scene.C, out.X, out.X);
let k0 = 0; for (let i = 0; i < n; i++) k0 += out.state[i] !== 0;
res.identity = { lost: k0, ...count(rl.all(out.X, out.state)) };
// whole clip at a pose, the space tilted: the batch, timed
const cl = new L.Clutch();
L.poseToCamera(scene, { q: q([0, 0, 1], 12) }, zero, 1, 1, { delta: D });
cl.press(new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]));
cl.move(L.rotvecToMatrix([0.15, 0.05, 0]));
cl.release(new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]));
cl.move(D);
L.turnCamera(scene, cl.Dc, cam);
L.project(scene, cam, X, out);
L.applySpace(cl.S, scene.C, out.X, out.X);
const tb = [];
for (let r = 0; r < 30; r++) { const t0 = performance.now(); rl.all(out.X, out.state); tb.push(performance.now() - t0); }
tb.sort((a, b) => a - b);
res.whole = { ms_median: tb[15], ms_min: tb[0], ...count(rl.out) };
// performed, frame by frame: pose(f, s) the phone; a clutch press / release; LIVE counted per frame
const play = (ri, pose, press = -1, release = -1, live = false) => {
  const perf = { X: new Float32Array(3 * n).fill(NaN), state: new Uint8Array(n).fill(R.NOT_YET) };
  const c = new L.Clutch(), tf = [];
  let liveMax = 0;
  ri.reset();
  for (let f = 0; f < F; f++) {
    const s = f / 30;
    L.poseToCamera(scene, pose(f, s), zero, 1, 1, { out: raw, delta: D });
    if (f === press) c.press(D);
    if (f === release) c.release(D);
    c.move(D, s);
    L.turnCamera(scene, c.Dc, cam);
    const a = frameStart[f], b = frameStart[f + 1];
    L.project(scene, cam, X, perf, a, b);
    L.applySpace(c.S, scene.C, perf.X, perf.X, a, b);
    const t0 = performance.now();
    ri.frames(perf.X, perf.state, f);
    tf.push(performance.now() - t0);
    if (live) liveMax = Math.max(liveMax, nLive(ri.out));
  }
  const ts = tf.slice().sort((a, b) => a - b);
  return { frames: F, ms_mean: tf.reduce((a, b) => a + b, 0) / F, ms_p95: ts[Math.floor(0.95 * F)], ms_max: ts[F - 1],
           live_max: live ? liveMax : null, ...count(ri.out) };
};
// a phone turning back and forth with a tilt held midway (a fresh Rules: its rest cache cold)
res.performed = play(new R.Rules(tracks, rules, meta.tracks.classes, { scene }),
                     (f, s) => ({ q: q([0, 0.2, 1], 25 * Math.sin(s / 2)) }), 400, 460);
// untouched (at its zero) all the clip; and lying still (0.02 deg of sensor noise per frame)
res.rest = play(rl, () => zero, -1, -1, true);
let seed = 1;
const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648 - 0.5;
res.still = play(rl, () => ({ q: q([rnd(), rnd(), rnd()], 0.02) }), -1, -1, true);
console.log(JSON.stringify(res));
"""


def test_trafalgar_6376_flags_and_timings(tmp_path):
    """Trafalgar 6376 (281 tracks, 69 217 points, 1228 frames): what the rules flag with the
    phone untouched (no live cause, whole and performed), lying still (sensor noise: reported),
    and the time per frame (performed) and per whole clip (node)."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    if not (TRAFALGAR / "tracks.bin").exists() or not (TRAFALGAR.parent / "_working" / "config_used.yaml").exists():
        pytest.skip("Trafalgar 6376 package is missing (gitignored)")
    rules = lr.rules_config(TRAFALGAR)
    (tmp_path / "rules.json").write_text(json.dumps(rules), encoding="utf-8")
    (tmp_path / "bench.mjs").write_text(BENCH, encoding="utf-8")
    r = subprocess.run([node, "bench.mjs", str(TRAFALGAR), str(LIVE), str(tmp_path / "rules.json")], cwd=tmp_path,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    res = json.loads(r.stdout.strip().splitlines()[-1])
    print(f"\nTrafalgar 6376: {json.dumps(res)}")
    # the Python reference agrees on the untouched phone (the pose path), with its rest baseline
    scene = lp.load_scene(TRAFALGAR)
    tr = lr.load_tracks(TRAFALGAR)
    zero = {"q": [1.0, 0.0, 0.0, 0.0]}
    cam, D = lp.pose_to_camera(scene, zero, zero, with_delta=True)
    cl = lp.Clutch()
    cl.move(D)
    Xp, st = lp.project(scene, lp.turn_camera(scene, cl.Dc, cam), tr["X"])
    Xp = lp.apply_space(cl.S, scene.C, Xp)
    meta = json.loads((TRAFALGAR / "scene.json").read_text(encoding="utf-8"))
    X, kind, cause, part = lr.apply_rules(rules, meta["tracks"]["classes"], tr["tid"], tr["t"], tr["cls"], tr["src"], Xp, st,
                                          rest=lr.rest_baseline(scene, tr["X"]))
    assert np.bincount(kind, minlength=5)[:5].tolist() == res["identity"]["kind"]
    assert np.bincount(cause, minlength=3).tolist() == res["identity"]["cause"]
    # 08 (c): the untouched phone gives no live cause (whole, and performed at every frame)
    assert res["identity"]["cause"][lr.LIVE] == 0 and res["rest"]["live_max"] == 0
    assert res["identity"]["cause"][lr.FOOTAGE] > 1000 and res["whole"]["cause"][lr.LIVE] > 1000
    assert res["performed"]["ms_p95"] < 5 and res["whole"]["ms_median"] < 50
