"""Live controller: the projection core (pose -> virtual camera -> drawn positions).

The Python reference of ``desktop_web/live/projection.js``: the same functions on the same scene
structure (``scene.json`` of a scene package + the heights of ``terrain.f32``), in numpy. The view
runs the JS; this one is for tests, sessions and exports. The two agree to < 1 cm.

The machine keeps its start calibration while the camera (the phone) turns and moves. A true ground
point X is seen by the moved, virtual camera at some pixel; the start calibration puts that pixel back
on the ground, at X'. What the virtual view does not show, or what is thrown past the horizon, is lost.

Frames (plans/live_controller/00_README.md): the scene frame is groundtrack's local frame (metres,
x east, y north, z up, z = 0 the datum ``ground_z_m``). Cameras use the OpenCV axes (x right, y down,
z forward); ``R_cw`` is camera_from_world, x = K R_cw (X - C).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DRAWN, BEHIND, OUT_OF_VIEW, PAST_HORIZON = 0, 1, 2, 3
STATES = ("drawn", "behind", "out_of_view", "past_horizon")
STEP_M = 0.5          # the raycast's march, as terrain.raycast
BISECT = 8
SPEED_WIN = 7         # frames, the speed as drawn is smoothed over (rp_drag_plans.speed_segments)
TILE = 8              # cells per side of the skip's max-height tiles
SKIP_TOL = 1e-6       # m: the ray must clear a tile's highest ground by this to skip a sample


# --------------------------------------------------------------------------- the scene
@dataclass
class Grid:
    """terrain.f32: heights relative to the datum, row 0 = north, cells of ``cell`` metres.
    ``x0`` / ``y1`` are the west / north edges of the grid (local metres); heights sit at the cell
    centres and are bilinear between them, as ``Terrain.height``."""
    z: np.ndarray         # (ny, nx)
    x0: float
    y1: float
    cell: float
    zlo: float            # min(lowest height, 0): the raycast's bracket
    zhi: float            # max(highest height, 0)
    tile: int = TILE      # the skip: highest ground per tile of tile x tile cells
    tmax: np.ndarray | None = None

    def build_tiles(self, T: int = TILE) -> None:
        """Bilinear heights never exceed the samples they come from; a tile takes the samples of
        its cells and one more cell around (a point a hair outside its tile, by rounding, is still
        bounded), and the datum (0) where that reaches past the grid."""
        ny, nx = self.z.shape
        nty, ntx = -(-ny // T), -(-nx // T)
        tmax = np.empty((nty, ntx))
        for i in range(nty):
            r0, r1 = i * T - 2, (i + 1) * T + 1
            for j in range(ntx):
                c0, c1 = j * T - 2, (j + 1) * T + 1
                h = float(self.z[max(r0, 0):min(r1, ny - 1) + 1, max(c0, 0):min(c1, nx - 1) + 1].max())
                if r0 < 0 or c0 < 0 or r1 > ny - 1 or c1 > nx - 1:
                    h = max(h, 0.0)
                tmax[i, j] = h
        self.tile, self.tmax = T, tmax

    def span(self, s, cx, cy, dx, dy):
        """For the points C + s (dx, dy): the highest ground of the tile under each, and the s where
        the ray leaves that tile. Beyond the tiles the ground is the datum, up to where the ray
        enters them (if ahead)."""
        T = self.tile * self.cell
        nty, ntx = self.tmax.shape
        x, y = cx + dx * s, cy + dy * s
        j, i = np.floor((x - self.x0) / T), np.floor((self.y1 - y) / T)
        inside = (j >= 0) & (j < ntx) & (i >= 0) & (i < nty)
        hm = np.zeros(len(s))
        se = np.full(len(s), np.inf)
        ji, ii = j[inside].astype(int), i[inside].astype(int)
        hm[inside] = self.tmax[ii, ji]
        xa, yt = self.x0 + ji * T, self.y1 - ii * T
        a, b = dx[inside], dy[inside]
        with np.errstate(divide="ignore", invalid="ignore"):
            sx = np.where(a > 0, (xa + T - cx) / a, np.where(a < 0, (xa - cx) / a, np.inf))
            sy = np.where(b > 0, (yt - cy) / b, np.where(b < 0, (yt - T - cy) / b, np.inf))
        se[inside] = np.minimum(sx, sy)
        out = ~inside
        if out.any():
            X0, X1, Y0, Y1 = self.x0, self.x0 + ntx * T, self.y1 - nty * T, self.y1
            so = s[out]
            t0 = np.full(len(so), -np.inf)
            t1 = np.full(len(so), np.inf)
            for d, c, lo, hi in ((dx[out], cx, X0, X1), (dy[out], cy, Y0, Y1)):
                with np.errstate(divide="ignore", invalid="ignore"):
                    p, q = (lo - c) / d, (hi - c) / d
                moving = d != 0
                t0 = np.where(moving, np.maximum(t0, np.minimum(p, q)), np.where((c < lo) | (c > hi), np.inf, t0))
                t1 = np.where(moving, np.minimum(t1, np.maximum(p, q)), np.where((c < lo) | (c > hi), -np.inf, t1))
            se[out] = np.where((t0 > t1) | (t1 <= so), np.inf, np.where(t0 > so, t0, so))
        return hm, se

    def height(self, x, y) -> np.ndarray:
        """Ground height at local (x, y); the datum plane (0) beyond the grid."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        ny, nx = self.z.shape
        c = (x - self.x0) / self.cell - 0.5
        r = (self.y1 - y) / self.cell - 0.5
        ok = (c >= 0) & (r >= 0) & (c <= nx - 1) & (r <= ny - 1)
        out = np.zeros(np.shape(x))
        if not ok.any():
            return out
        c, r = c[ok], r[ok]
        c0 = np.minimum(np.floor(c).astype(int), nx - 2)
        r0 = np.minimum(np.floor(r).astype(int), ny - 2)
        fc, fr = c - c0, r - r0
        z = self.z
        out[ok] = ((z[r0, c0] * (1 - fc) + z[r0, c0 + 1] * fc) * (1 - fr)
                   + (z[r0 + 1, c0] * (1 - fc) + z[r0 + 1, c0 + 1] * fc) * fr)
        return out


@dataclass
class Scene:
    meta: dict            # scene.json as read
    K: np.ndarray
    R_cw: np.ndarray      # the start camera
    R_wc: np.ndarray
    C: np.ndarray
    Minv: np.ndarray      # the start calibration's pixel -> ray: R_wc K^-1
    W: float
    H: float
    R_max: float
    grid: Grid | None     # None: flat ground, the datum plane
    skip: bool = True     # the raycast jumps tiles whose ground is certainly below the ray


@dataclass
class Camera:
    """A virtual camera: the start camera's K with a pose."""
    R_cw: np.ndarray
    C: np.ndarray


def prepare_scene(meta: dict, heights=None, skip: bool = True, tile: int = TILE) -> Scene:
    """The scene from scene.json (a dict) and the heights of terrain.f32 (None: flat ground).
    skip: the raycast jumps over tiles of tile x tile cells whose ground is certainly below the ray
    (the same results as the plain march, faster)."""
    cam = meta["camera"]
    K = np.asarray(cam["K"], float)
    R_cw = np.asarray(cam["R_cw"], float)
    C = np.asarray(cam["C"], float)
    if cam.get("P") is not None:          # K R_cw (X - C) must be P [X, 1] with a positive scale
        P = np.asarray(cam["P"], float)
        Q = K @ np.column_stack([R_cw, -R_cw @ C])
        a, b = P / np.linalg.norm(P), Q / np.linalg.norm(Q)
        if np.abs(a - b).max() > 1e-6:
            raise ValueError("scene.json: camera K R_cw C does not reproduce P (up to a positive scale)")
    grid = None
    t = meta.get("terrain")
    if t and heights is not None:
        z = np.asarray(heights, np.float32).astype(float).reshape(int(t["ny"]), int(t["nx"]))
        grid = Grid(z=z, x0=float(t["x0"]), y1=float(t["y1"]), cell=float(t.get("cell", 1.0)),
                    zlo=min(float(z.min()), 0.0), zhi=max(float(z.max()), 0.0))
        grid.build_tiles(tile)
    W, H = meta["image_size"]
    R_wc = R_cw.T
    return Scene(meta=meta, K=K, R_cw=R_cw, R_wc=R_wc, C=C, Minv=R_wc @ np.linalg.inv(K),
                 W=float(W), H=float(H), R_max=float(meta["R_max"]), grid=grid, skip=skip)


def load_scene(live_dir: str | Path) -> Scene:
    """A scene package (runs/<clip>/<run>/live/): scene.json + terrain.f32."""
    live_dir = Path(live_dir)
    meta = json.loads((live_dir / "scene.json").read_text(encoding="utf-8"))
    t = meta.get("terrain")
    heights = None
    if t and t.get("file") and (live_dir / t["file"]).exists():
        heights = np.fromfile(live_dir / t["file"], dtype="<f4")
    return prepare_scene(meta, heights)


def start_camera(scene: Scene) -> Camera:
    return Camera(R_cw=scene.R_cw.copy(), C=scene.C.copy())


def ground_height(scene: Scene, x, y) -> np.ndarray:
    if scene.grid is None:
        return np.zeros(np.shape(np.asarray(x, float)))
    return scene.grid.height(x, y)


# --------------------------------------------------------------------------- rotations
def quat_to_matrix(q) -> np.ndarray:
    """[w, x, y, z] (normalised here) -> 3x3 rotation."""
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> axis * angle (rad), through the quaternion (stable near 0 and 180 deg)."""
    R = np.asarray(R, float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2 * np.sqrt(tr + 1)
        w, x, y, z = s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, s / 4, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2 * np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, s / 4, (R[1, 2] + R[2, 1]) / s
    else:
        s = 2 * np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, s / 4
    if w < 0:
        w, x, y, z = -w, -x, -y, -z
    n = np.sqrt(x * x + y * y + z * z)
    if n < 1e-12:
        return np.array([2 * x, 2 * y, 2 * z])
    return np.array([x, y, z]) / n * (2 * np.arctan2(n, w))


def rotvec_to_matrix(rv) -> np.ndarray:
    rv = np.asarray(rv, float)
    a = np.linalg.norm(rv)
    if a < 1e-12:
        return np.eye(3)
    k = rv / a
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * Kx + (1 - np.cos(a)) * (Kx @ Kx)


def heading(R_wc: np.ndarray) -> float:
    """Heading (rad, clockwise from north = +y) of a camera's forward (its OpenCV z axis):
    atan2(x, y). Looking straight up or down, the image's up (-y axis) stands in."""
    f = R_wc[:, 2]
    if np.hypot(f[0], f[1]) < 1e-6:
        f = -R_wc[:, 1]
    return float(np.arctan2(f[0], f[1]))


def heading_turn(psi: float) -> np.ndarray:
    """A turn about z that adds psi to every heading (clockwise seen from above)."""
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


# --------------------------------------------------------------------------- pose -> camera
def pose_to_camera(scene: Scene, src: dict | None, zero: dict | None, gain: float = 1.0,
                   scale: float = 1.0, smoother: "PoseSmoother | None" = None,
                   with_delta: bool = False):
    """The virtual camera for a source pose (a pose message: q [w, x, y, z] world_from_camera in
    the source's gravity frame, p metres or None), relative to the pose at zero.

    The turn since zero, re-headed onto the scene (the source's forward at zero onto the start
    camera's), times the gain, turns the start camera; the move since zero, re-headed the same
    way, times the scale, moves it. No zero yet: the start camera.
    with_delta: returns (camera, Δ), Δ that turn with the gain applied (R_wc = Δ R_wc0; the
    identity without zero), for the Clutch."""
    if src is None or zero is None:
        return (start_camera(scene), np.eye(3)) if with_delta else start_camera(scene)
    Q, Q0 = quat_to_matrix(src["q"]), quat_to_matrix(zero["q"])
    A = heading_turn(heading(scene.R_wc) - heading(Q0))
    D = A @ (Q @ Q0.T) @ A.T
    rv = matrix_to_rotvec(D)
    dp = None
    if src.get("p") is not None and zero.get("p") is not None:
        dp = A @ (np.asarray(src["p"], float) - np.asarray(zero["p"], float))
    if smoother is not None:
        rv, dp = smoother(float(src.get("t") or 0.0), rv, dp)
    Dg = rotvec_to_matrix(rv * gain)
    R_wc = Dg @ scene.R_wc
    C = scene.C + (scale * dp if dp is not None else 0.0)
    cam = Camera(R_cw=R_wc.T, C=np.asarray(C, float))
    return (cam, Dg) if with_delta else cam


class OneEuro:
    """One-euro filter on a vector (Casiez et al. 2012): the cutoff rises with the speed, so slow
    moves are smoothed and fast ones follow. One cutoff for all components (from |dx/dt|)."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.t = self.x = None
        self.dx = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, t: float, x) -> np.ndarray:
        x = np.asarray(x, float)
        if self.x is None or not (t > self.t):
            if self.x is None:
                self.t, self.x, self.dx = t, x.copy(), np.zeros_like(x)
            return self.x.copy()
        dt = t - self.t
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * (x - self.x) / dt + (1 - a_d) * self.dx
        a = self._alpha(self.min_cutoff + self.beta * float(np.linalg.norm(self.dx)), dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x.copy()


class PoseSmoother:
    """Optional smoothing of the turn (as axis * angle, before the gain) and of the move (before
    the scale). Off unless passed to pose_to_camera."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.rot = OneEuro(min_cutoff, beta, d_cutoff)
        self.pos = OneEuro(min_cutoff, beta, d_cutoff)

    def __call__(self, t, rv, dp):
        rv = self.rot(t, rv)
        if dp is None:
            self.pos.reset()
        else:
            dp = self.pos(t, dp)
        return rv, dp


# --------------------------------------------------------------------------- the space clutch
LEVEL_S = 0.3         # s: the double-tap's glide back to level


class Clutch:
    """The phone's hold button (plans/live_controller/08_space_and_multiplayer.md), one per space
    (a clip's scene). Δ: the phone's turn since zero, gain applied (pose_to_camera(...,
    with_delta=True)).

    - S: the space's turn about the start camera's centre C0. All of the clip is shown at
      p -> C0 + S (p - C0) (space_matrix), and a drawn point is X' = C0 + S (G(X) - C0), G the
      ground-mode projection with this clutch's camera (turn_camera, then project, then apply_space).
    - Dc = Δ K: the camera's turn, R_wc_v = Dc R_wc0 (turn_camera). K is the catch-up that keeps it
      continuous at a release.

    press(Δ): Dc freezes; the snapshot Δp, Sp. move(Δ, t): while held S = δ Sp, δ = Δ Δpᵀ the turn
    since the press (with the hand: tilt the phone down and the far side goes down). release(Δ): S
    stays, K = Δᵀ Dc(press), so Dc goes on from where it froze. zero(): K = I (the caller's Δ is I
    from there), S kept. level(t, duration): S glides back to I (a slerp, eased), driven by move's
    t (the same clock, seconds); during the glide the hand does not turn the space, and a hold goes
    on from where the glide is. No jumps: press, release and zero continue from the current state.
    The JS twin is projection.js's Clutch (the same steps, in place)."""

    def __init__(self):
        I = np.eye(3)
        self.S, self.K, self.Dc = I.copy(), I.copy(), I.copy()
        self.held = False
        self.Dp, self.Sp, self.Dcp = I.copy(), I.copy(), I.copy()     # the press snapshot
        self.D = I.copy()                                              # the last Δ seen
        self.glide = None                                              # (t0, duration, rv of S at its start)

    @property
    def angle(self) -> float:
        """S's angle (rad)."""
        return float(np.linalg.norm(matrix_to_rotvec(self.S)))

    @property
    def tilted(self) -> bool:
        """Whether the space is turned (the phone's "space turned")."""
        return self.angle > 1e-6

    def press(self, D):
        self.D = np.array(D, float)
        if self.held:
            return self
        self.Dc = self.D @ self.K
        self.Dcp, self.Dp, self.Sp = self.Dc.copy(), self.D.copy(), self.S.copy()
        self.held = True
        return self

    def move(self, D, t: float | None = None):
        """The current Δ (and the clock, for a glide)."""
        self.D = np.array(D, float)
        gliding = self.glide is not None
        if gliding:
            self._glide(t)
        if not self.held:
            self.Dc = self.D @ self.K
        elif gliding:                                     # the hold goes on from the glide
            self.Dp, self.Sp = self.D.copy(), self.S.copy()
        else:
            self.S = self.D @ self.Dp.T @ self.Sp
        return self

    def release(self, D):
        self.D = np.array(D, float)
        if not self.held:
            return self
        if self.glide is None:
            self.S = self.D @ self.Dp.T @ self.Sp
        self.K = self.D.T @ self.Dcp
        self.Dc = self.D @ self.K
        self.held = False
        return self

    def level(self, t: float, duration: float = LEVEL_S):
        rv = matrix_to_rotvec(self.S)
        if np.linalg.norm(rv) < 1e-12 or not duration > 0:
            self.S = np.eye(3)
            self.glide = None
            if self.held:
                self.Dp, self.Sp = self.D.copy(), np.eye(3)
            return self
        self.glide = (float(t), float(duration), rv)
        return self

    def zero(self):
        self.K, self.Dc, self.D = np.eye(3), np.eye(3), np.eye(3)
        if self.held:
            self.Dcp, self.Dp, self.Sp = np.eye(3), np.eye(3), self.S.copy()
        return self

    def _glide(self, t):
        if t is None:
            return
        t0, dur, rv = self.glide
        u = min(max((t - t0) / dur, 0.0), 1.0)
        if u >= 1:
            self.S, self.glide = np.eye(3), None
            return
        k = 1 - u * u * (3 - 2 * u)                       # smoothstep: no kick at either end
        self.S = rotvec_to_matrix(rv * k)


def turn_camera(scene: Scene, Dc, cam: Camera | None = None) -> Camera:
    """A clutch's camera: the start camera turned by Dc (R_wc = Dc R_wc0), at cam's position (the
    start camera's when None)."""
    R_wc = np.asarray(Dc, float) @ scene.R_wc
    return Camera(R_cw=R_wc.T, C=(cam.C if cam is not None else scene.C).copy())


def apply_space(S, C0, X) -> np.ndarray:
    """Points (n, 3) into the turned space: X' = C0 + S (X - C0). NaN stays NaN."""
    X = np.asarray(X, float).reshape(-1, 3)
    C0 = np.asarray(C0, float)
    return C0 + (X - C0) @ np.asarray(S, float).T


def space_matrix(S, C0) -> np.ndarray:
    """The same as a 4x4 for whole objects: [S | C0 - S C0; 0 0 0 1]."""
    S, C0 = np.asarray(S, float), np.asarray(C0, float)
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = S, C0 - S @ C0
    return M


# --------------------------------------------------------------------------- camera -> drawn
def raycast(scene: Scene, C, D, max_range_m: float | None = None, skip: bool | None = None) -> np.ndarray:
    """First ground hit of rays C + t D (local metres) within max_range_m (horizontal, default
    R_max): (n, 3), NaN where a ray never meets the ground (sky, past the horizon, too far).

    Mirrors terrain.raycast: the bracket between where the ray passes the highest and the lowest
    ground, a 0.5 m march, 8 bisections; beyond the grid the datum plane. Flat ground: the plane.
    skip (default scene.skip): march samples in a tile whose highest ground is below the ray there
    cannot be the hit, so they are jumped (same samples, same hit; tried once per tile)."""
    R = scene.R_max if max_range_m is None else float(max_range_m)
    skip = scene.skip if skip is None else skip
    D = np.asarray(D, float).reshape(-1, 3)
    n = len(D)
    horiz = np.maximum(np.hypot(D[:, 0], D[:, 1]), 1e-9)
    dx, dy, m = D[:, 0] / horiz, D[:, 1] / horiz, D[:, 2] / horiz
    cx, cy, cz = (float(v) for v in C)
    hit = np.full(n, np.nan)
    g = scene.grid
    if g is None:
        with np.errstate(divide="ignore", invalid="ignore"):
            s = cz / -m
        ok = (m < 0) & (cz > 0) & (s <= R)
        hit[ok] = s[ok]
    else:
        def ground(s, i):
            return g.height(cx + dx[i] * s, cy + dy[i] * s)

        with np.errstate(divide="ignore", invalid="ignore"):
            s_lo = np.where(m < 0, (cz - g.zhi) / -m - STEP_M, 0.0)
            s_hi = np.where(m < 0, (cz - g.zlo) / -m + STEP_M, R if cz < g.zhi else 0.0)
        s_lo = np.clip(np.nan_to_num(s_lo), 0.0, R)
        s_hi = np.clip(np.nan_to_num(s_hi, nan=R), 0.0, R)
        act = np.flatnonzero(s_hi > s_lo)
        s = s_lo[act]
        above = cz + m[act] * s >= ground(s, act) - 1e-6
        wait = np.full(act.size, -np.inf)
        tiles = skip and g.tmax is not None
        while act.size:
            done = np.zeros(act.size, bool)
            step = np.ones(act.size, bool)
            if tiles:
                # the ray clears the tile's highest ground up to sc (descending), or all along / not at all
                sn, sh, ma = s + STEP_M, s_hi[act], m[act]
                hm, se = g.span(sn, cx, cy, dx[act], dy[act])
                with np.errstate(divide="ignore", invalid="ignore"):
                    sc = np.where(ma < 0, (cz - hm - 2 * SKIP_TOL) / -ma,
                                  np.where(cz + ma * sn > hm + SKIP_TOL, np.inf, -np.inf))
                lim = np.minimum(se, sc)
                tried = (sn < sh) & (sn > wait)
                clear = tried & (sn <= lim)
                done = clear & (lim >= sh)                      # clear of the ground to the bracket's end
                jump = clear & ~done                            # to the last sample that clears
                s[jump] = s[jump] + (1 + np.floor((lim[jump] - sn[jump]) / STEP_M)) * STEP_M
                above[jump] = True
                # a tile that does not clear here (or no further): march through it
                wait = np.where((tried & ~clear) | (jump & (sc < se)), se, wait)
                step = ~clear
            if step.any():
                ia = act[step]
                s_new = np.minimum(s[step] + STEP_M, s_hi[ia])
                below = cz + m[ia] * s_new <= ground(s_new, ia)
                found = above[step] & below
                if found.any():
                    i, a, b = ia[found], s[step][found], s_new[found]
                    for _ in range(BISECT):
                        mid = (a + b) / 2
                        up = cz + m[i] * mid > ground(mid, i)
                        a, b = np.where(up, mid, a), np.where(up, b, mid)
                    hit[i] = (a + b) / 2
                done[step] = found | (s_new >= s_hi[ia])
                s[step] = s_new
                above[step] = ~below
            keep = ~done
            act, s, above, wait = act[keep], s[keep], above[keep], wait[keep]
    x, y = cx + dx * hit, cy + dy * hit
    out = np.column_stack([x, y, np.full(n, np.nan)])
    ok = np.isfinite(hit)
    out[ok, 2] = ground_height(scene, x[ok], y[ok])
    return out


MODES = ("ground", "hinge")


def project(scene: Scene, cam: Camera, X, mode: str = "ground") -> tuple[np.ndarray, np.ndarray]:
    """True ground points X (n, 3, local) -> where the machine draws them with the virtual camera
    cam: X' (n, 3; NaN when lost) and the state per point (DRAWN, BEHIND, OUT_OF_VIEW,
    PAST_HORIZON). A point with a NaN coordinate counts as behind.

    mode "ground": the start calibration's ray through the virtual pixel, cast onto the terrain.
    mode "hinge": the point keeps its place relative to the camera, so the scene turns rigidly about
    the start camera, the other way: X' = C0 + R_wc0 R_cw (X - C). Lost only when the virtual camera
    does not see it (behind, out of view); never past the horizon."""
    if mode not in MODES:
        raise ValueError(f"project: unknown mode {mode!r}")
    X = np.asarray(X, float).reshape(-1, 3)
    x = (X - cam.C) @ (scene.K @ cam.R_cw).T
    state = np.full(len(X), BEHIND, np.uint8)
    front = x[:, 2] > 0
    u = np.full((len(X), 2), np.nan)
    u[front] = x[front, :2] / x[front, 2:3]
    seen = front & (u[:, 0] >= 0) & (u[:, 0] <= scene.W) & (u[:, 1] >= 0) & (u[:, 1] <= scene.H)
    state[front & ~seen] = OUT_OF_VIEW
    out = np.full((len(X), 3), np.nan)
    if seen.any() and mode == "hinge":
        out[seen] = scene.C + (X[seen] - cam.C) @ (scene.R_wc @ cam.R_cw).T
        state[seen] = DRAWN
    elif seen.any():
        D = np.column_stack([u[seen], np.ones(seen.sum())]) @ scene.Minv.T
        hit = raycast(scene, scene.C, D)
        ok = np.isfinite(hit[:, 0])
        idx = np.flatnonzero(seen)
        out[idx[ok]] = hit[ok]
        state[idx[ok]] = DRAWN
        state[idx[~ok]] = PAST_HORIZON
    return out, state


# --------------------------------------------------------------------------- speed and colour
def speed_as_drawn(Xp, t, track_id, win: int = SPEED_WIN, dims: int = 2) -> np.ndarray:
    """Speed (m/s) along each drawn track, as rp_drag_plans.speed_segments: the drawn positions
    smoothed (rolling mean over win rows, centred, >= 2 valid), then central differences over
    the track's rows. Rows in their order within a track (frame order); lost rows (NaN) skipped
    by the smoothing; 0 where no speed. dims 2: along the plan (as the Richmond sheets); dims 3:
    in space (the hinge mode lifts the plan)."""
    import pandas as pd

    Xp = np.asarray(Xp, float).reshape(len(t), -1)
    t = np.asarray(t, float)
    track_id = np.asarray(track_id)
    out = np.zeros(len(t))
    order = np.argsort(track_id, kind="stable")
    tid = track_id[order]
    cuts = np.flatnonzero(np.diff(tid)) + 1
    for rows in np.split(order, cuts):
        if len(rows) < 2:
            continue
        s = pd.DataFrame(Xp[rows, :dims]).rolling(win, center=True, min_periods=2).mean().to_numpy()
        tt = t[rows]
        dt = np.gradient(tt)
        g = [np.gradient(s[:, k]) for k in range(dims)]
        g = np.hypot(g[0], g[1]) if dims == 2 else np.sqrt(sum(c * c for c in g))
        with np.errstate(divide="ignore", invalid="ignore"):
            v = g / np.where(dt > 0, dt, np.nan)
        out[rows] = np.nan_to_num(v)
    return out


def ramp_color(v, lo: float = 1.0, hi: float = 40.0) -> np.ndarray:
    """Speed (m/s) -> RGB (n, 3, 0..1): groundtrack's speed ramp, log from lo to hi."""
    from .colors import cmap

    x = np.clip(np.log(np.maximum(np.asarray(v, float), 1e-6) / lo) / np.log(hi / lo), 0, 1)
    return cmap()(x)[..., :3]


def ramp_table() -> np.ndarray:
    """The ramp's 256 colours (what projection.js carries)."""
    from .colors import cmap

    return cmap()(np.arange(256))[:, :3]


# --------------------------------------------------------------------------- view on the ground
def footprint(scene: Scene, cam: Camera | None = None, n: int = 50) -> np.ndarray:
    """The ground a camera shows (its frame's border cast onto the ground), out to R_max:
    (4n + 1, 3), closed. Rays past the horizon end on the R_max circle (z = 0)."""
    cam = cam or start_camera(scene)
    W, H = scene.W, scene.H
    t = np.arange(n) / n
    b = np.vstack([np.c_[t * W, 0 * t], np.c_[W + 0 * t, t * H], np.c_[(1 - t) * W, H + 0 * t],
                   np.c_[0 * t, (1 - t) * H]])
    D = np.column_stack([b, np.ones(len(b))]) @ (cam.R_cw.T @ np.linalg.inv(scene.K)).T
    hit = raycast(scene, cam.C, D)
    dirn = D[:, :2] / np.maximum(np.hypot(D[:, 0], D[:, 1]), 1e-9)[:, None]
    bad = ~np.isfinite(hit[:, 0])
    hit[bad, :2] = cam.C[:2] + scene.R_max * dirn[bad]
    hit[bad, 2] = 0.0
    return np.vstack([hit, hit[:1]])


def inside_polygon(pts, poly) -> np.ndarray:
    """Even-odd test of points (n, 2) against a closed polygon (m, 2)."""
    pts, poly = np.asarray(pts, float), np.asarray(poly, float)
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    xa, ya, xb, yb = poly[:-1, 0], poly[:-1, 1], poly[1:, 0], poly[1:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        cross = ((ya > y) != (yb > y)) & (x < (xb - xa) * (y - ya) / (yb - ya) + xa)
    return (cross.sum(axis=1) % 2) == 1


def drag_grid(scene: Scene, nb: int = 5, rings=(0.3, 0.55, 0.8), cam: Camera | None = None) -> np.ndarray:
    """A calm polar grid of ground points to show the drag (rp_drag_3d.polar_drag): nb bearings
    across the start view, a few rings between its near edge and R_max. (n, 3)."""
    cam = cam or start_camera(scene)
    fp = footprint(scene, cam)
    yaw = heading(scene.R_wc)
    r_ = np.array([np.cos(yaw), -np.sin(yaw)])
    f_ = np.array([np.sin(yaw), np.cos(yaw)])
    rel = fp[:, :2] - cam.C[:2]
    a, b = rel @ r_, rel @ f_
    bear = np.arctan2(a, b)
    lo, hi = np.percentile(bear, 2), np.percentile(bear, 98)
    bs = np.linspace(lo, hi, nb + 2)[1:-1]
    dmin = max(float(np.percentile(np.hypot(a, b), 2)), 0.5)
    G = np.array([[rr * np.sin(bb), rr * np.cos(bb)] for r in rings
                  for rr in [dmin + r * (scene.R_max - dmin)] for bb in bs])
    Gw = cam.C[:2] + G[:, :1] * r_ + G[:, 1:] * f_
    Gw = Gw[inside_polygon(Gw, fp[:, :2])]
    return np.column_stack([Gw, ground_height(scene, Gw[:, 0], Gw[:, 1])])
