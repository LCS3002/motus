/* Live controller: the projection core (pose -> virtual camera -> drawn positions).
   An ES module with no dependencies, used by the Live view. Its Python reference is
   groundtrack/live_projection.py: the same functions on the same scene, agreeing to < 1 cm.

   The machine keeps its start calibration while the camera (the phone) turns and moves. A true
   ground point X is seen by the moved, virtual camera at some pixel; the start calibration puts
   that pixel back on the ground, at X'. What the virtual view does not show, or what is thrown
   past the horizon, is lost.

   Frames (plans/live_controller/00_README.md): the scene frame is groundtrack's local frame
   (metres, x east, y north, z up, z = 0 the datum). Cameras use the OpenCV axes (x right, y down,
   z forward); R_cw is camera_from_world, x = K R_cw (X - C). Matrices here are row-major
   Float64Array(9); point lists are flat [x, y, z, x, y, z, ...] typed arrays. Per point nothing is
   allocated. */

export const DRAWN = 0, BEHIND = 1, OUT_OF_VIEW = 2, PAST_HORIZON = 3;
export const STATES = ["drawn", "behind", "out_of_view", "past_horizon"];
export const STEP_M = 0.5;      // the raycast's march, as groundtrack/terrain.py
export const BISECT = 8;
export const SPEED_WIN = 7;     // frames the speed as drawn is smoothed over (odd)
export const TILE = 8;          // cells per side of the skip's max-height tiles
const SKIP_TOL = 1e-6;          // m: the ray must clear a tile's highest ground by this to skip a sample

// ---------------------------------------------------------------------------- 3x3 helpers
const m3 = () => new Float64Array(9);
const flat3 = (rows) => Float64Array.from(rows.flat());

function mul(a, b, o) {            // o = a b
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++)
    o[3 * i + j] = a[3 * i] * b[j] + a[3 * i + 1] * b[3 + j] + a[3 * i + 2] * b[6 + j];
  return o;
}

function mulT(a, b, o) {           // o = a bᵀ
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++)
    o[3 * i + j] = a[3 * i] * b[3 * j] + a[3 * i + 1] * b[3 * j + 1] + a[3 * i + 2] * b[3 * j + 2];
  return o;
}

function transpose(a, o) {
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) o[3 * j + i] = a[3 * i + j];
  return o;
}

function eye3(o = m3()) {
  o.fill(0); o[0] = o[4] = o[8] = 1;
  return o;
}

function inv3(a, o) {
  const [a0, a1, a2, a3, a4, a5, a6, a7, a8] = a;
  const c0 = a4 * a8 - a5 * a7, c1 = a5 * a6 - a3 * a8, c2 = a3 * a7 - a4 * a6;
  const d = a0 * c0 + a1 * c1 + a2 * c2;
  o[0] = c0 / d; o[1] = (a2 * a7 - a1 * a8) / d; o[2] = (a1 * a5 - a2 * a4) / d;
  o[3] = c1 / d; o[4] = (a0 * a8 - a2 * a6) / d; o[5] = (a2 * a3 - a0 * a5) / d;
  o[6] = c2 / d; o[7] = (a1 * a6 - a0 * a7) / d; o[8] = (a0 * a4 - a1 * a3) / d;
  return o;
}

// ---------------------------------------------------------------------------- the scene
/* scene.json (parsed) + the heights of terrain.f32 (Float32Array or its ArrayBuffer; null: flat
   ground, the datum plane). terrain: x0 / y1 are the grid's west / north edges (local metres), row
   0 = north; heights sit at the cell centres, bilinear between them; beyond the grid the datum.
   opts.skip (default true): the raycast jumps over tiles whose ground is certainly below the ray
   (same results as the plain march); opts.tile: cells per tile side. */
export function prepareScene(meta, heights = null, opts = {}) {
  const cam = meta.camera;
  const K = flat3(cam.K), R_cw = flat3(cam.R_cw), C = Float64Array.from(cam.C);
  const R_wc = transpose(R_cw, m3()), Kinv = inv3(K, m3());
  let grid = null;
  const t = meta.terrain;
  if (t && heights) {
    const z = heights instanceof Float32Array ? heights : new Float32Array(heights);
    const nx = t.nx | 0, ny = t.ny | 0;
    if (z.length !== nx * ny) throw new Error(`terrain: ${z.length} heights for ${nx} x ${ny} cells`);
    let lo = 0, hi = 0;
    for (let i = 0; i < z.length; i++) { if (z[i] < lo) lo = z[i]; if (z[i] > hi) hi = z[i]; }
    grid = { z, nx, ny, x0: +t.x0, y1: +t.y1, cell: +(t.cell ?? 1), zlo: lo, zhi: hi };
    buildTiles(grid, opts.tile ?? TILE);
  }
  return {
    meta, K, Kinv, R_cw, R_wc, C, Minv: mul(R_wc, Kinv, m3()),
    W: +meta.image_size[0], H: +meta.image_size[1], R_max: +meta.R_max, grid, yaw: heading(R_wc),
    skip: opts.skip ?? true,
  };
}

/* The skip's coarse max grid: per tile of T x T cells the highest ground anywhere in it. Bilinear
   heights never exceed the samples they come from; a tile takes the samples of its cells and one
   more cell around (a point a hair outside its tile, by rounding, is still bounded), and the datum
   (0) where that reaches past the grid. */
function buildTiles(g, T) {
  const ntx = Math.ceil(g.nx / T), nty = Math.ceil(g.ny / T), tmax = new Float32Array(ntx * nty);
  for (let i = 0; i < nty; i++) for (let j = 0; j < ntx; j++) {
    const r0 = i * T - 2, r1 = (i + 1) * T + 1, c0 = j * T - 2, c1 = (j + 1) * T + 1;
    let h = r0 < 0 || c0 < 0 || r1 > g.ny - 1 || c1 > g.nx - 1 ? 0 : -Infinity;
    for (let r = Math.max(r0, 0); r <= Math.min(r1, g.ny - 1); r++)
      for (let c = Math.max(c0, 0); c <= Math.min(c1, g.nx - 1); c++) {
        const v = g.z[r * g.nx + c];
        if (v > h) h = v;
      }
    tmax[i * ntx + j] = h;
  }
  g.tile = T; g.tileM = T * g.cell; g.ntx = ntx; g.nty = nty; g.tmax = tmax;
}

const _hmax = new Float64Array(1);    // tileSpan's second result (a typed slot: no boxing)

/* The tile under C + s (dx, dy): sets _hmax[0] to its highest ground and returns the s where the ray
   leaves it. Beyond the tiles the ground is the datum, up to where the ray enters them (if ahead). */
function tileSpan(g, cx, cy, dx, dy, s) {
  const x = cx + dx * s, y = cy + dy * s, T = g.tileM;
  const j = Math.floor((x - g.x0) / T), i = Math.floor((g.y1 - y) / T);
  if (j >= 0 && j < g.ntx && i >= 0 && i < g.nty) {
    _hmax[0] = g.tmax[i * g.ntx + j];
    const xa = g.x0 + j * T, yt = g.y1 - i * T;
    const sx = dx > 0 ? (xa + T - cx) / dx : dx < 0 ? (xa - cx) / dx : Infinity;
    const sy = dy > 0 ? (yt - cy) / dy : dy < 0 ? (yt - T - cy) / dy : Infinity;
    return sx < sy ? sx : sy;
  }
  _hmax[0] = 0;
  const X0 = g.x0, X1 = g.x0 + g.ntx * T, Y0 = g.y1 - g.nty * T, Y1 = g.y1;
  let t0 = -Infinity, t1 = Infinity;
  if (dx !== 0) {
    const a = (X0 - cx) / dx, b = (X1 - cx) / dx;
    t0 = Math.max(t0, Math.min(a, b)); t1 = Math.min(t1, Math.max(a, b));
  } else if (cx < X0 || cx > X1) return Infinity;
  if (dy !== 0) {
    const a = (Y0 - cy) / dy, b = (Y1 - cy) / dy;
    t0 = Math.max(t0, Math.min(a, b)); t1 = Math.min(t1, Math.max(a, b));
  } else if (cy < Y0 || cy > Y1) return Infinity;
  if (t0 > t1 || t1 <= s) return Infinity;
  return t0 > s ? t0 : s;
}

export function startCamera(scene, out = null) {
  out = out || { R_cw: m3(), R_wc: m3(), C: new Float64Array(3) };
  out.R_cw.set(scene.R_cw); out.R_wc.set(scene.R_wc); out.C.set(scene.C);
  return out;
}

function gridHeight(g, x, y) {
  const c = (x - g.x0) / g.cell - 0.5, r = (g.y1 - y) / g.cell - 0.5;
  if (!(c >= 0 && r >= 0 && c <= g.nx - 1 && r <= g.ny - 1)) return 0;
  let c0 = Math.floor(c), r0 = Math.floor(r);
  if (c0 > g.nx - 2) c0 = g.nx - 2;
  if (r0 > g.ny - 2) r0 = g.ny - 2;
  const fc = c - c0, fr = r - r0, z = g.z, i = r0 * g.nx + c0, j = i + g.nx;
  return (z[i] * (1 - fc) + z[i + 1] * fc) * (1 - fr) + (z[j] * (1 - fc) + z[j + 1] * fc) * fr;
}

/** Ground height (m above the datum) at local (x, y). */
export function groundHeight(scene, x, y) {
  return scene.grid ? gridHeight(scene.grid, x, y) : 0;
}

// ---------------------------------------------------------------------------- rotations
/** [w, x, y, z] (normalised here) -> rotation. */
export function quatToMatrix(q, o = m3()) {
  const n = Math.hypot(q[0], q[1], q[2], q[3]);
  const w = q[0] / n, x = q[1] / n, y = q[2] / n, z = q[3] / n;
  o[0] = 1 - 2 * (y * y + z * z); o[1] = 2 * (x * y - w * z); o[2] = 2 * (x * z + w * y);
  o[3] = 2 * (x * y + w * z); o[4] = 1 - 2 * (x * x + z * z); o[5] = 2 * (y * z - w * x);
  o[6] = 2 * (x * z - w * y); o[7] = 2 * (y * z + w * x); o[8] = 1 - 2 * (x * x + y * y);
  return o;
}

/** Rotation -> axis * angle (rad), through the quaternion (stable near 0 and 180 deg). */
export function matrixToRotvec(R, o = new Float64Array(3)) {
  const tr = R[0] + R[4] + R[8];
  let w, x, y, z, s;
  if (tr > 0) {
    s = 2 * Math.sqrt(tr + 1);
    w = s / 4; x = (R[7] - R[5]) / s; y = (R[2] - R[6]) / s; z = (R[3] - R[1]) / s;
  } else if (R[0] > R[4] && R[0] > R[8]) {
    s = 2 * Math.sqrt(1 + R[0] - R[4] - R[8]);
    w = (R[7] - R[5]) / s; x = s / 4; y = (R[1] + R[3]) / s; z = (R[2] + R[6]) / s;
  } else if (R[4] > R[8]) {
    s = 2 * Math.sqrt(1 + R[4] - R[0] - R[8]);
    w = (R[2] - R[6]) / s; x = (R[1] + R[3]) / s; y = s / 4; z = (R[5] + R[7]) / s;
  } else {
    s = 2 * Math.sqrt(1 + R[8] - R[0] - R[4]);
    w = (R[3] - R[1]) / s; x = (R[2] + R[6]) / s; y = (R[5] + R[7]) / s; z = s / 4;
  }
  if (w < 0) { w = -w; x = -x; y = -y; z = -z; }
  const n = Math.sqrt(x * x + y * y + z * z);
  if (n < 1e-12) { o[0] = 2 * x; o[1] = 2 * y; o[2] = 2 * z; return o; }
  const k = 2 * Math.atan2(n, w) / n;
  o[0] = x * k; o[1] = y * k; o[2] = z * k;
  return o;
}

export function rotvecToMatrix(rv, o = m3()) {
  const a = Math.hypot(rv[0], rv[1], rv[2]);
  if (a < 1e-12) { o.fill(0); o[0] = o[4] = o[8] = 1; return o; }
  const x = rv[0] / a, y = rv[1] / a, z = rv[2] / a, s = Math.sin(a), c = 1 - Math.cos(a);
  // I + sin(a) [k]x + (1 - cos(a)) [k]x²
  o[0] = 1 + c * (-y * y - z * z); o[1] = -s * z + c * x * y; o[2] = s * y + c * x * z;
  o[3] = s * z + c * x * y; o[4] = 1 + c * (-x * x - z * z); o[5] = -s * x + c * y * z;
  o[6] = -s * y + c * x * z; o[7] = s * x + c * y * z; o[8] = 1 + c * (-x * x - y * y);
  return o;
}

/** Heading (rad, clockwise from north = +y) of a camera's forward (its OpenCV z axis) from its
    world_from_camera rotation: atan2(x, y). Looking straight up or down, the image's up stands in. */
export function heading(R_wc) {
  let fx = R_wc[2], fy = R_wc[5];
  if (Math.hypot(fx, fy) < 1e-6) { fx = -R_wc[1]; fy = -R_wc[4]; }
  return Math.atan2(fx, fy);
}

/** A turn about z that adds psi to every heading (clockwise seen from above). */
export function headingTurn(psi, o = m3()) {
  const c = Math.cos(psi), s = Math.sin(psi);
  o[0] = c; o[1] = s; o[2] = 0; o[3] = -s; o[4] = c; o[5] = 0; o[6] = 0; o[7] = 0; o[8] = 1;
  return o;
}

// ---------------------------------------------------------------------------- pose -> camera
const _Q = m3(), _Q0 = m3(), _A = m3(), _T1 = m3(), _T2 = m3(), _D = m3(), _Dg = m3();
const _rv = new Float64Array(3), _dp = new Float64Array(3);

/* The virtual camera for a source pose (a pose message: q [w, x, y, z] world_from_camera in the
   source's gravity frame, p metres or null), relative to the pose at zero (a pose message too).
   The turn since zero, re-headed onto the scene (the source's forward at zero onto the start
   camera's), times the gain, turns the start camera; the move since zero, re-headed the same way,
   times the scale, moves it. No zero yet: the start camera.
   opts.out: a camera to fill ({R_cw, R_wc, C}); opts.smoother: a PoseSmoother (off by default);
   opts.delta: a Float64Array(9) to fill with that turn, gain applied (Δ: R_wc = Δ R_wc0; the
   identity without zero), for the Clutch. */
export function poseToCamera(scene, src, zero, gain = 1, scale = 1, opts = {}) {
  const out = startCamera(scene, opts.out || null);
  if (!src || !zero) {
    if (opts.delta) eye3(opts.delta);
    return out;
  }
  quatToMatrix(src.q, _Q); quatToMatrix(zero.q, _Q0);
  headingTurn(scene.yaw - heading(_Q0), _A);
  mulT(_Q, _Q0, _T1);                       // the turn since zero, in the source's frame
  mulT(mul(_A, _T1, _T2), _A, _D);          // re-headed onto the scene
  matrixToRotvec(_D, _rv);
  let dp = null;
  if (src.p && zero.p) {
    const ex = src.p[0] - zero.p[0], ey = src.p[1] - zero.p[1], ez = src.p[2] - zero.p[2];
    _dp[0] = _A[0] * ex + _A[1] * ey + _A[2] * ez;
    _dp[1] = _A[3] * ex + _A[4] * ey + _A[5] * ez;
    _dp[2] = _A[6] * ex + _A[7] * ey + _A[8] * ez;
    dp = _dp;
  }
  if (opts.smoother) dp = opts.smoother.filter(+(src.t || 0), _rv, dp);
  _rv[0] *= gain; _rv[1] *= gain; _rv[2] *= gain;
  mul(rotvecToMatrix(_rv, _Dg), scene.R_wc, out.R_wc);
  transpose(out.R_wc, out.R_cw);
  if (dp) for (let i = 0; i < 3; i++) out.C[i] = scene.C[i] + scale * dp[i];
  if (opts.delta) opts.delta.set(_Dg);
  return out;
}

/* One-euro filter on a vector (Casiez et al. 2012): the cutoff rises with the speed, so slow moves
   are smoothed and fast ones follow. One cutoff for all components (from |dx/dt|). In place. */
export class OneEuro {
  constructor(minCutoff = 1, beta = 0, dCutoff = 1) {
    this.minCutoff = minCutoff; this.beta = beta; this.dCutoff = dCutoff;
    this.reset();
  }

  reset() { this.t = null; this.x = null; this.dx = null; }

  static alpha(cutoff, dt) { return 1 / (1 + 1 / (2 * Math.PI * cutoff) / dt); }

  filter(t, x) {
    if (this.x === null) {
      this.t = t; this.x = Float64Array.from(x); this.dx = new Float64Array(x.length);
      return x;
    }
    if (!(t > this.t)) { x.set(this.x); return x; }
    const dt = t - this.t, ad = OneEuro.alpha(this.dCutoff, dt);
    let sp = 0;
    for (let i = 0; i < x.length; i++) {
      this.dx[i] = ad * (x[i] - this.x[i]) / dt + (1 - ad) * this.dx[i];
      sp += this.dx[i] * this.dx[i];
    }
    const a = OneEuro.alpha(this.minCutoff + this.beta * Math.sqrt(sp), dt);
    for (let i = 0; i < x.length; i++) x[i] = this.x[i] = a * x[i] + (1 - a) * this.x[i];
    this.t = t;
    return x;
  }
}

/* Optional smoothing of the turn (axis * angle, before the gain) and of the move (before the
   scale). Pass it as poseToCamera's opts.smoother; reset() on zero. */
export class PoseSmoother {
  constructor(minCutoff = 1, beta = 0, dCutoff = 1) {
    this.rot = new OneEuro(minCutoff, beta, dCutoff);
    this.pos = new OneEuro(minCutoff, beta, dCutoff);
  }

  reset() { this.rot.reset(); this.pos.reset(); }

  filter(t, rv, dp) {
    this.rot.filter(t, rv);
    if (!dp) { this.pos.reset(); return null; }
    return this.pos.filter(t, dp);
  }
}

// ---------------------------------------------------------------------------- the space clutch
export const LEVEL_S = 0.3;       // s: the double-tap's glide back to level
const _c1 = m3(), _c2 = m3(), _crv = new Float64Array(3);

/* The phone's hold button (plans/live_controller/08_space_and_multiplayer.md), one per space (a
   clip's scene). Δ: the phone's turn since zero, gain applied (poseToCamera's opts.delta).
   - S: the space's turn about the start camera's centre C0. All of the clip is shown at
     p -> C0 + S (p - C0) (spaceMatrix), and a drawn point is X' = C0 + S (G(X) - C0), G the
     ground-mode projection with this clutch's camera (turnCamera, then project, then applySpace).
   - Dc = Δ K: the camera's turn, R_wc_v = Dc R_wc0 (turnCamera). K is the catch-up that keeps it
     continuous at a release.
   press(Δ): Dc freezes; the snapshot Δp, Sp. move(Δ, t): while held, S = δ Sp with δ = Δ Δpᵀ, the
   turn since the press (with the hand: tilt the phone down and the far side goes down). release(Δ):
   S stays, K = Δᵀ Dc(press), so Dc goes on from where it froze. zero(): K = I (the caller's Δ is I
   from there), S kept. level(t, duration): S glides back to I (a slerp, eased), driven by move's t
   (the same clock, seconds); during the glide the hand does not turn the space, and a hold goes on
   from where the glide is. No jumps: press, release and zero continue from the current state.
   S, K, Dc are row-major Float64Array(9) owned by the clutch and updated in place (copy to keep). */
export class Clutch {
  constructor() {
    this.S = eye3(); this.K = eye3(); this.Dc = eye3();
    this.held = false;
    this.Dp = eye3(); this.Sp = eye3(); this.Dcp = eye3();     // the press snapshot
    this.D = eye3();                                            // the last Δ seen
    this.glide = null;                                          // {t0, duration, rv: S's turn at its start}
  }

  /** S's angle (rad). */
  get angle() {
    matrixToRotvec(this.S, _crv);
    return Math.hypot(_crv[0], _crv[1], _crv[2]);
  }

  /** Whether the space is turned (the phone's "space turned"). */
  get tilted() { return this.angle > 1e-6; }

  press(D) {
    this.D.set(D);
    if (this.held) return this;
    mul(D, this.K, this.Dc);
    this.Dcp.set(this.Dc); this.Dp.set(D); this.Sp.set(this.S);
    this.held = true;
    return this;
  }

  /** The current Δ (and the clock, for a glide) -> this, with this.Dc and this.S. */
  move(D, t = null) {
    this.D.set(D);
    const gliding = this.glide !== null;
    if (gliding) this._glide(t);
    if (!this.held) mul(D, this.K, this.Dc);
    else if (gliding) { this.Dp.set(D); this.Sp.set(this.S); }       // the hold goes on from the glide
    else mul(mulT(D, this.Dp, _c1), this.Sp, this.S);
    return this;
  }

  release(D) {
    this.D.set(D);
    if (!this.held) return this;
    if (this.glide === null) mul(mulT(D, this.Dp, _c1), this.Sp, this.S);
    mul(transpose(D, _c2), this.Dcp, this.K);
    mul(D, this.K, this.Dc);
    this.held = false;
    return this;
  }

  level(t, duration = LEVEL_S) {
    matrixToRotvec(this.S, _crv);
    if (Math.hypot(_crv[0], _crv[1], _crv[2]) < 1e-12 || !(duration > 0)) {
      eye3(this.S);
      this.glide = null;
      if (this.held) { this.Dp.set(this.D); eye3(this.Sp); }
      return this;
    }
    this.glide = { t0: +t, duration: +duration, rv: Float64Array.from(_crv) };
    return this;
  }

  zero() {
    eye3(this.K); eye3(this.Dc); eye3(this.D);
    if (this.held) { eye3(this.Dcp); eye3(this.Dp); this.Sp.set(this.S); }
    return this;
  }

  _glide(t) {
    if (t === null || t === undefined) return;
    const g = this.glide, u = Math.min(Math.max((t - g.t0) / g.duration, 0), 1);
    if (u >= 1) { eye3(this.S); this.glide = null; return; }
    const k = 1 - u * u * (3 - 2 * u);                       // smoothstep: no kick at either end
    _crv[0] = g.rv[0] * k; _crv[1] = g.rv[1] * k; _crv[2] = g.rv[2] * k;
    rotvecToMatrix(_crv, this.S);
  }
}

/** A clutch's camera: the start camera turned by Dc (R_wc = Dc R_wc0) at cam's position, in place. */
export function turnCamera(scene, Dc, cam) {
  mul(Dc, scene.R_wc, cam.R_wc);
  transpose(cam.R_wc, cam.R_cw);
  return cam;
}

/* Points (flat xyz) into the turned space: X' = C0 + S (X - C0), points i0..i1-1 into out (made
   like X when null; out may be X). NaN stays NaN. */
export function applySpace(S, C0, X, out = null, i0 = 0, i1 = -1) {
  const n = (X.length / 3) | 0;
  if (i1 < 0) i1 = n;
  out = out || new X.constructor(X.length);
  const cx = C0[0], cy = C0[1], cz = C0[2];
  for (let i = i0; i < i1; i++) {
    const j = 3 * i, ax = X[j] - cx, ay = X[j + 1] - cy, az = X[j + 2] - cz;
    out[j] = cx + S[0] * ax + S[1] * ay + S[2] * az;
    out[j + 1] = cy + S[3] * ax + S[4] * ay + S[5] * az;
    out[j + 2] = cz + S[6] * ax + S[7] * ay + S[8] * az;
  }
  return out;
}

/* The same as a 4x4 for whole objects, row-major: [S | C0 - S C0; 0 0 0 1] (three.js:
   new THREE.Matrix4().set(...m)). */
export function spaceMatrix(S, C0, out = new Float64Array(16)) {
  for (let i = 0; i < 3; i++) {
    out[4 * i] = S[3 * i]; out[4 * i + 1] = S[3 * i + 1]; out[4 * i + 2] = S[3 * i + 2];
    out[4 * i + 3] = C0[i] - (S[3 * i] * C0[0] + S[3 * i + 1] * C0[1] + S[3 * i + 2] * C0[2]);
  }
  out[12] = 0; out[13] = 0; out[14] = 0; out[15] = 1;
  return out;
}

// ---------------------------------------------------------------------------- camera -> drawn
/* First ground hit of the ray C + s D within maxRange (horizontal metres, default R_max), into
   out[o..o+2] (NaN when none: sky, past the horizon, too far). Returns whether it hit.
   As groundtrack/terrain.py::raycast: the bracket between where the ray passes the highest and
   the lowest ground, a 0.5 m march, 8 bisections; beyond the grid the datum plane. Flat ground
   (no grid): the plane z = 0. skip: march samples in a tile whose highest ground is below the ray
   there cannot be the hit, so they are jumped (same samples, same hit; tried once per tile). */
export function raycast(scene, cx, cy, cz, Dx, Dy, Dz, out, o = 0, maxRange = scene.R_max,
                        skip = scene.skip) {
  let horiz = Math.hypot(Dx, Dy);
  if (!(horiz > 1e-9)) horiz = 1e-9;
  const dx = Dx / horiz, dy = Dy / horiz, m = Dz / horiz, R = maxRange, g = scene.grid;
  let hit = NaN;
  if (!g) {
    const s = cz / -m;
    if (m < 0 && cz > 0 && s <= R) hit = s;
  } else {
    let lo = m < 0 ? (cz - g.zhi) / -m - STEP_M : 0;
    let hi = m < 0 ? (cz - g.zlo) / -m + STEP_M : (cz < g.zhi ? R : 0);
    if (lo !== lo) lo = 0;
    if (hi !== hi) hi = R;
    lo = lo < 0 ? 0 : lo > R ? R : lo;
    hi = hi < 0 ? 0 : hi > R ? R : hi;
    if (hi > lo) {
      let s = lo, wait = -Infinity;
      let above = cz + m * s >= gridHeight(g, cx + dx * s, cy + dy * s) - 1e-6;
      const tiles = skip && g.tmax;
      for (;;) {
        if (tiles) {
          const sn = s + STEP_M;
          if (sn < hi && sn > wait) {
            // the ray clears the tile's highest ground up to sc (descending), or all along / not at all
            const se = tileSpan(g, cx, cy, dx, dy, sn);
            const hm = _hmax[0];
            const sc = m < 0 ? (cz - hm - 2 * SKIP_TOL) / -m : cz + m * sn > hm + SKIP_TOL ? Infinity : -Infinity;
            const lim = se < sc ? se : sc;
            if (sn <= lim) {
              if (lim >= hi) break;                 // clear of the ground to the bracket's end
              s += (1 + Math.floor((lim - sn) / STEP_M)) * STEP_M;   // the last sample that clears
              above = true;
              if (sc < se) wait = se;               // the rest of this tile: march through it
              continue;
            }
            wait = se;                              // this tile does not clear here: march through it
          }
        }
        const sn = Math.min(s + STEP_M, hi);
        const below = cz + m * sn <= gridHeight(g, cx + dx * sn, cy + dy * sn);
        if (above && below) {                     // refine the crossing by bisection
          let a = s, b = sn;
          for (let k = 0; k < BISECT; k++) {
            const mid = (a + b) / 2;
            if (cz + m * mid > gridHeight(g, cx + dx * mid, cy + dy * mid)) a = mid; else b = mid;
          }
          hit = (a + b) / 2;
          break;
        }
        if (sn >= hi) break;
        above = !below;                           // a ray starting underground must surface first
        s = sn;
      }
    }
  }
  if (hit !== hit) { out[o] = out[o + 1] = out[o + 2] = NaN; return false; }
  const x = cx + dx * hit, y = cy + dy * hit;
  out[o] = x; out[o + 1] = y; out[o + 2] = g ? gridHeight(g, x, y) : 0;
  return true;
}

const _KR = m3(), _HG = m3();
export const MODES = ["ground", "hinge"];

/* True ground points X (flat xyz, local) -> where the machine draws them with the virtual camera
   cam: out.X (flat xyz, NaN when lost) and out.state (DRAWN, BEHIND, OUT_OF_VIEW, PAST_HORIZON).
   Only points i0..i1-1 are done (the rest of out is left as it is): a frame's slice of the tracks.
   out = {X: Float32Array | Float64Array (3n), state: Uint8Array (n)}; made when null. A point with
   a NaN coordinate counts as behind.
   mode "ground": the start calibration's ray through the virtual pixel, cast onto the terrain.
   mode "hinge": the point keeps its place relative to the camera, so the scene turns rigidly about
   the start camera, the other way: X' = C0 + R_wc0 R_cw (X - C). Lost only when the virtual camera
   does not see it (behind, out of view); never past the horizon. */
export function project(scene, cam, X, out = null, i0 = 0, i1 = -1, mode = "ground") {
  const n = (X.length / 3) | 0;
  if (i1 < 0) i1 = n;
  if (mode !== "ground" && mode !== "hinge") throw new Error(`project: unknown mode ${mode}`);
  out = out || { X: new Float64Array(3 * n), state: new Uint8Array(n) };
  const KR = mul(scene.K, cam.R_cw, _KR), M = scene.Minv, C = cam.C, C0 = scene.C;
  const W = scene.W, H = scene.H, P = out.X, st = out.state;
  const hinge = mode === "hinge", G = hinge ? mul(scene.R_wc, cam.R_cw, _HG) : null;
  for (let i = i0; i < i1; i++) {
    const j = 3 * i;
    const ax = X[j] - C[0], ay = X[j + 1] - C[1], az = X[j + 2] - C[2];
    const xz = KR[6] * ax + KR[7] * ay + KR[8] * az;
    if (!(xz > 0)) { st[i] = BEHIND; P[j] = P[j + 1] = P[j + 2] = NaN; continue; }
    const u = (KR[0] * ax + KR[1] * ay + KR[2] * az) / xz;
    const v = (KR[3] * ax + KR[4] * ay + KR[5] * az) / xz;
    if (!(u >= 0 && u <= W && v >= 0 && v <= H)) { st[i] = OUT_OF_VIEW; P[j] = P[j + 1] = P[j + 2] = NaN; continue; }
    if (hinge) {
      P[j] = C0[0] + (G[0] * ax + G[1] * ay + G[2] * az);
      P[j + 1] = C0[1] + (G[3] * ax + G[4] * ay + G[5] * az);
      P[j + 2] = C0[2] + (G[6] * ax + G[7] * ay + G[8] * az);
      st[i] = DRAWN;
      continue;
    }
    st[i] = raycast(scene, C0[0], C0[1], C0[2], M[0] * u + M[1] * v + M[2], M[3] * u + M[4] * v + M[5],
                    M[6] * u + M[7] * v + M[8], P, j) ? DRAWN : PAST_HORIZON;
  }
  return out;
}

// ---------------------------------------------------------------------------- speed and colour
/** Points grouped by track, in their order within each track (frame order): {order, starts}. */
export function trackIndex(trackId) {
  // by id, then row: the ids sorted (few), the rows counted into their tracks (a growing recording
  // indexes its points again with each piece: no sort of every row)
  const n = trackId.length, order = new Uint32Array(n), rank = new Map();
  for (let i = 0; i < n; i++) if (!rank.has(trackId[i])) rank.set(trackId[i], 0);
  const ids = [...rank.keys()].sort((a, b) => a - b);
  ids.forEach((id, k) => rank.set(id, k));
  const starts = new Uint32Array(ids.length + 1), fill = new Uint32Array(ids.length);
  for (let i = 0; i < n; i++) starts[rank.get(trackId[i]) + 1]++;
  for (let k = 0; k < ids.length; k++) { starts[k + 1] += starts[k]; fill[k] = starts[k]; }
  for (let i = 0; i < n; i++) order[fill[rank.get(trackId[i])]++] = i;
  return { order, starts };
}

let _sx = new Float64Array(0), _sy = new Float64Array(0), _sz = new Float64Array(0);

/* Speed (m/s) along each drawn track, as rp_drag_plans.speed_segments: the drawn positions
   smoothed (rolling mean over win rows, centred, >= 2 valid), then central differences over the
   track's rows. Lost rows (NaN) are skipped by the smoothing; 0 where there is no speed.
   Xp flat xyz, t seconds per point, index from trackIndex(). dims 2: along the plan (as the
   Richmond sheets); dims 3: in space (the hinge mode lifts the plan). */
export function speedAsDrawn(Xp, t, index, out = null, win = SPEED_WIN, dims = 2) {
  const { order, starts } = index, half = (win - 1) >> 1, d3 = dims === 3;
  out = out || new Float64Array(t.length);
  for (let k = 0; k + 1 < starts.length; k++) {
    const a = starts[k], L = starts[k + 1] - a;
    if (L < 2) { if (L === 1) out[order[a]] = 0; continue; }
    if (_sx.length < L) { _sx = new Float64Array(L); _sy = new Float64Array(L); _sz = new Float64Array(L); }
    for (let r = 0; r < L; r++) {
      let sx = 0, sy = 0, sz = 0, nx = 0, ny = 0, nz = 0;
      const e = Math.min(L - 1, r + half);
      for (let q = Math.max(0, r - half); q <= e; q++) {
        const j = 3 * order[a + q], x = Xp[j], y = Xp[j + 1], z = Xp[j + 2];
        if (x === x) { sx += x; nx++; }
        if (y === y) { sy += y; ny++; }
        if (z === z) { sz += z; nz++; }
      }
      _sx[r] = nx >= 2 ? sx / nx : NaN;
      _sy[r] = ny >= 2 ? sy / ny : NaN;
      _sz[r] = nz >= 2 ? sz / nz : NaN;
    }
    for (let r = 0; r < L; r++) {
      const p = r === 0 ? 0 : r - 1, q = r === L - 1 ? L - 1 : r + 1, d = q - p;
      const gx = (_sx[q] - _sx[p]) / d, gy = (_sy[q] - _sy[p]) / d;
      const dt = (t[order[a + q]] - t[order[a + p]]) / d;
      const g = d3 ? Math.sqrt(gx * gx + gy * gy + ((_sz[q] - _sz[p]) / d) ** 2) : Math.hypot(gx, gy);
      const v = g / (dt > 0 ? dt : NaN);
      out[order[a + r]] = v !== v ? 0 : v === Infinity ? Number.MAX_VALUE : v;
    }
  }
  return out;
}

// groundtrack's speed ramp (colors.cmap: turbo, trimmed), its 256 colours
const RAMP_HEX =
  "4559cb455ccf455ed34661d64664da4664da4666dd4669e0466be3476ee64771e94771e94773eb4776ee4778f0477bf2" +
  "467df44680f64680f64682f84685fa4687fb458afc458cfd448ffe448ffe4391fe4294ff4196ff4099ff3e9bfe3d9efe" +
  "3d9efe3ba0fd3aa3fc38a5fb37a8fa35abf833adf733adf731aff52fb2f42eb4f22cb7f02ab9ee28bceb28bceb27bee9" +
  "25c0e723c3e422c5e220c7df20c7df1fc9dd1ecbda1ccdd81bd0d51ad2d21ad4d01ad4d019d5cd18d7ca18d9c818dbc5" +
  "18ddc218dec018dec018e0bd19e2bb19e3b91ae4b61ce6b41de7b21de7b21fe9af20eaac22ebaa25eca727eea42aefa1" +
  "2aefa12cf09e2ff19b32f29835f39438f49138f4913cf58e3ff68a43f78746f8844af8804ef97d4ef97d52fa7a55fa76" +
  "59fb735dfc6f61fc6c65fd6965fd6969fd666dfe6271fe5f75fe5c79fe597dff567dff5680ff5384ff5188ff4e8bff4b" +
  "8fff4992ff4792ff4796fe4499fe429cfe409ffd3fa1fd3da4fc3ca4fc3ca7fc3aa9fb39acfb38affa37b1f936b1f936" +
  "b4f836b7f735b9f635bcf534bef434c1f334c1f334c3f134c6f034c8ef34cbed34cdec34d0ea34d0ea34d2e935d4e735" +
  "d7e535d9e436dbe236dde037dde037dfdf37e1dd37e3db38e5d938e7d739e9d539e9d539ebd339ecd13aeecf3aefcd3a" +
  "f1cb3af1cb3af2c93af4c73af5c53af6c33af7c13af8be39f8be39f9bc39faba39fbb838fbb637fcb336fcb136fcb136" +
  "fdae35fdac34fea933fea732fea431fea130fea130fe9e2ffe9b2dfe992cfe962bfe932afe9029fe9029fd8d27fd8a26" +
  "fc8725fc8423fb8122fb8122fb7e21fa7b1ff9781ef9751df8721cf76f1af76f1af66c19f56918f46617f36315f26014" +
  "f15d13f15d13f05b12ef5811ed5510ec530feb500eea4e0dea4e0de84b0ce7490ce5470be4450ae2430ae14109e14109" +
  "df3f08dd3d08dc3b07da3907d83706d63506d63506d43305d23105d02f05ce2d04cc2b04cc2b04ca2a04c82803c52603" +
  "c32503c12302be2102be2102bc2002b91e02b71d02b41b01b21a01af1801af1801ac1701a91601a71401a41301a11201";
export const RAMP = new Float32Array(768);
for (let i = 0; i < 768; i++) RAMP[i] = parseInt(RAMP_HEX.substr(2 * i, 2), 16) / 255;

/** Speed (m/s) -> RGB (0..1) into out[o..o+2]: the speed ramp, log from lo to hi. */
export function rampColor(v, out = new Float32Array(3), o = 0, lo = 1, hi = 40) {
  let x = Math.log(Math.max(v, 1e-6) / lo) / Math.log(hi / lo);
  x = x > 0 ? (x < 1 ? x : 1) : 0;
  const i = 3 * Math.min(255, Math.floor(x * 256));
  out[o] = RAMP[i]; out[o + 1] = RAMP[i + 1]; out[o + 2] = RAMP[i + 2];
  return out;
}

// ---------------------------------------------------------------------------- view on the ground
/** The ground a camera shows (its frame's border cast onto the ground), out to R_max: flat xyz of
    4n + 1 points, closed. Rays past the horizon end on the R_max circle (z = 0). */
export function footprint(scene, cam = null, n = 50) {
  cam = cam || startCamera(scene);
  const M = mul(cam.R_wc || transpose(cam.R_cw, m3()), scene.Kinv, m3()), W = scene.W, H = scene.H;
  const out = new Float64Array(3 * (4 * n + 1));
  for (let k = 0; k < 4 * n; k++) {
    const side = Math.floor(k / n), t = (k % n) / n;
    const u = side === 0 ? t * W : side === 1 ? W : side === 2 ? (1 - t) * W : 0;
    const v = side === 0 ? 0 : side === 1 ? t * H : side === 2 ? H : (1 - t) * H;
    const Dx = M[0] * u + M[1] * v + M[2], Dy = M[3] * u + M[4] * v + M[5], Dz = M[6] * u + M[7] * v + M[8];
    if (!raycast(scene, cam.C[0], cam.C[1], cam.C[2], Dx, Dy, Dz, out, 3 * k)) {
      const h = Math.max(Math.hypot(Dx, Dy), 1e-9);
      out[3 * k] = cam.C[0] + scene.R_max * Dx / h; out[3 * k + 1] = cam.C[1] + scene.R_max * Dy / h;
      out[3 * k + 2] = 0;
    }
  }
  out.copyWithin(12 * n, 0, 3);
  return out;
}

/** Even-odd test of the point (x, y) against a closed polygon (flat xyz, as footprint()). */
export function insidePolygon(x, y, poly) {
  let inside = false;
  for (let i = 0; i + 3 < poly.length; i += 3) {
    const xa = poly[i], ya = poly[i + 1], xb = poly[i + 3], yb = poly[i + 4];
    if ((ya > y) !== (yb > y) && x < (xb - xa) * (y - ya) / (yb - ya) + xa) inside = !inside;
  }
  return inside;
}

function percentile(a, q) {              // numpy's default (linear)
  const s = Float64Array.from(a).sort(), p = (q / 100) * (s.length - 1), i = Math.floor(p);
  return i + 1 < s.length ? s[i] + (s[i + 1] - s[i]) * (p - i) : s[i];
}

/** A calm polar grid of ground points to show the drag (rp_drag_3d.polar_drag): nb bearings
    across the start view, a few rings between its near edge and R_max. Flat xyz. */
export function dragGrid(scene, nb = 5, rings = [0.3, 0.55, 0.8], cam = null) {
  cam = cam || startCamera(scene);
  const fp = footprint(scene, cam), m = fp.length / 3, yaw = scene.yaw;
  const rx = Math.cos(yaw), ry = -Math.sin(yaw), fx = Math.sin(yaw), fy = Math.cos(yaw);
  const bear = new Float64Array(m), dist = new Float64Array(m);
  for (let k = 0; k < m; k++) {
    const ex = fp[3 * k] - cam.C[0], ey = fp[3 * k + 1] - cam.C[1];
    const a = ex * rx + ey * ry, b = ex * fx + ey * fy;
    bear[k] = Math.atan2(a, b); dist[k] = Math.hypot(a, b);
  }
  const lo = percentile(bear, 2), hi = percentile(bear, 98);
  const dmin = Math.max(percentile(dist, 2), 0.5), pts = [];
  for (const r of rings) {
    const rr = dmin + r * (scene.R_max - dmin);
    for (let j = 1; j <= nb; j++) {
      const bb = lo + (hi - lo) * j / (nb + 1), gx = rr * Math.sin(bb), gy = rr * Math.cos(bb);
      const x = cam.C[0] + gx * rx + gy * fx, y = cam.C[1] + gx * ry + gy * fy;
      if (insidePolygon(x, y, fp)) pts.push(x, y, groundHeight(scene, x, y));
    }
  }
  return Float64Array.from(pts);
}
