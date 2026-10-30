/* Live view: the tracks as the machine draws them (plans/live_controller/03, 08 (a) and (c)).

   Where: the virtual camera (the clutch's, in the space's own frame) casts each point onto the
   ground (projection.project, ground mode), the space's turn puts it in the world (applySpace S),
   and the machine's own rules (rules.js: visibility, cleaning, stitching, as on the footage) judge
   those drawn positions in the world and say what it draws.
   - Performed (default): each frame's points are projected with the pose of the moment and kept,
     in the world where they were drawn (a later turn of the space does not move them); the rules
     re-judge the tracks with new points (the newest are provisional: a spike is known a frame
     later, a stitch within 0.6 s). Scrubbing back clears what lies after; going forward fills the
     skipped frames with the current pose.
   - Whole clip: every point with the current pose and space (02's worker pool), judged as a whole.
   The lexicon: coloured solid = what the machine places (the speed ramp as drawn, in 3D, log
   1-40 m/s), even where it is wrong; white dashes = the machine's guesses (predicted, stitched,
   interpolated: the footage's own and the ones Live causes); a new identity (part) breaks the line;
   dropped: nothing. Cause mode: the guesses Live caused violet, the footage's white, the solid
   lines dimmed. Heads as small dots; thin grey = the reference, the machine at rest (Live with the
   phone untouched: the start pose, a level space), which is not "truth".
   Only the drawn crowd (crowd: the lines, dashes and heads) is turned by the space (08 a, 28 Oct);
   the reference stays level with the site. The ground never hides the crowd; what lies below the
   ground is drawn by setUnder: "same", "dim" (darker), or "veil" (the ground opaque, the crowd
   seen through it faintly: below reads as under). */

import * as THREE from "three";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { project, applySpace, startCamera, speedAsDrawn, trackIndex, rampColor, groundHeight, SPEED_WIN } from "./projection.js";
import { Rules, SOLID, DROPPED, NOT_YET, FOOTAGE, LIVE } from "./rules.js";
import { Segs, lineMaterial, dotTexture } from "./lines.js";
import { snapshot } from "./passes.js";

const LIFT = 0.05;           // tracks sit this far above the ground
const MAX_GAP = 2;           // frames: rows further apart are not joined (rp_drag_plans)
const MARGIN = 12;           // frames rewritten before the first changed one (the speed's smoothing)
const DIM = 0.3;             // cause mode: the solid lines' opacity
const VIOLET_RGB = [0x9b / 255, 0x82 / 255, 0xd9 / 255];      // scene.js VIOLET: the interference
export const UNDER = ["same", "dim", "veil"];                    // the crowd below the ground
const DIM_BELOW = 0.32;      // "dim": below the ground at this brightness
const BELOW_M = 0.02;        // m: below the ground by more than this
const GHOST = 0.3;           // "veil": the crowd through the ground at this opacity

// ---------------------------------------------------------------------------- the package
/** tracks.bin (30-byte records, see live_export.TRACK_DTYPE) -> typed arrays + indexes. */
export function parseTracks(buf, meta) {
  const dv = new DataView(buf), n = (buf.byteLength / 30) | 0, frames = meta.frames | 0;
  const tid = new Uint32Array(n), frame = new Uint32Array(n), t = new Float32Array(n), X = new Float32Array(3 * n);
  const speed = new Float32Array(n), cls = new Uint8Array(n), src = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const o = 30 * i;
    tid[i] = dv.getUint32(o, true); frame[i] = dv.getUint32(o + 4, true); t[i] = dv.getFloat32(o + 8, true);
    X[3 * i] = dv.getFloat32(o + 12, true); X[3 * i + 1] = dv.getFloat32(o + 16, true); X[3 * i + 2] = dv.getFloat32(o + 20, true);
    speed[i] = dv.getFloat32(o + 24, true); cls[i] = dv.getUint8(o + 28); src[i] = dv.getUint8(o + 29);
  }
  const F = Math.max(frames, n ? frame[n - 1] + 1 : 0);
  const frameStart = new Uint32Array(F + 1);
  for (let i = 0, f = 0; f <= F; f++) { while (i < n && frame[i] < f) i++; frameStart[f] = i; }
  const index = trackIndex(tid);
  // consecutive rows of each track (a -> b), sorted by b's frame: segments draw in frame order
  const pa = [], pb = [];
  const { order, starts } = index;
  for (let k = 0; k + 1 < starts.length; k++)
    for (let r = starts[k]; r + 1 < starts[k + 1]; r++) { pa.push(order[r]); pb.push(order[r + 1]); }
  const P = pa.length, pairStart = new Uint32Array(F + 2), cnt = new Uint32Array(F + 1);
  for (let s = 0; s < P; s++) cnt[frame[pb[s]]]++;
  for (let f = 0; f < F; f++) pairStart[f + 1] = pairStart[f] + cnt[f];
  pairStart[F + 1] = P;
  const A = new Uint32Array(P), B = new Uint32Array(P), fill = pairStart.slice(0, F + 1);
  for (let s = 0; s < P; s++) { const f = frame[pb[s]], q = fill[f]++; A[q] = pa[s]; B[q] = pb[s]; }
  const near = new Uint8Array(P), guess = new Uint8Array(P);
  for (let s = 0; s < P; s++) {
    near[s] = frame[B[s]] - frame[A[s]] <= MAX_GAP ? 1 : 0;
    guess[s] = src[A[s]] !== 0 || src[B[s]] !== 0 ? 1 : 0;   // predicted, stitched, interpolated
  }
  return { n, frames: F, tid, frame, t, X, speed, cls, src, frameStart, index,
           pairs: { n: P, A, B, near, guess, start: pairStart }, nTracks: starts.length - 1 };
}

// ---------------------------------------------------------------------------- the view
const _c = new Float32Array(3);
const guessed = (kind, src, i) => kind[i] !== SOLID || src[i] !== 0;      // the machine guessed here

export class TrackView {
  /* sc: prepareScene(); data: parseTracks(); rules: the run's rules config (live_rules.rules_config,
     /api/live/rules); classes: scene.json tracks.classes (code -> name); tint: the violet of Live's
     own guesses in cause mode (the spot's tint with several spots, 08 b). */
  constructor(sc, data, rules, classes = {}, { tint = VIOLET_RGB } = {}) {
    this.sc = sc;
    this.tint = Float32Array.from(tint);
    this.d = data;
    this.cfg = rules;
    this.classes = classes;
    const n = data.n, P = data.pairs.n;
    this.mode = "performed";
    // the machine at rest: the start camera's drawing of every point (the rules measure the cause
    // against it, and the reference is its verdict)
    this.restDraw = project(sc, startCamera(sc), data.X, { X: new Float32Array(3 * n), state: new Uint8Array(n) });
    // performed: per point the drawn position (world) and the projection's state (NOT_YET until
    // its frame is performed); the rules' verdict on them in rPerf.out
    this.perf = { X: new Float32Array(3 * n).fill(NaN), state: new Uint8Array(n).fill(NOT_YET) };
    this.perfFrame = -1;
    this.rPerf = this._rules();
    this.rWhole = null;                // whole clip: its own rules, made on first use
    this.wholeX = null;                // the whole clip's drawn positions in the world
    this.whole = null;                 // rWhole.out once judged
    this.rest = null;                  // the reference: the machine at rest (computeReference)
    this.cause = false;
    this.speed = new Float64Array(n);
    this.arc = new Float32Array(n);    // metres along each drawn line (dash phase)
    this.atSolid = new Uint32Array(data.frames + 2);
    this.atGuess = new Uint32Array(data.frames + 2);
    this.width = 1.15;
    this.solid = new Segs(P, lineMaterial({ width: this.width, vertexColors: true }), { order: 10 });
    this.guess = new Segs(P, lineMaterial({ width: this.width, vertexColors: true, dashed: true, dash: 0.12, gap: 0.3 }),
                          { distances: true, order: 11 });
    this.ref = new Segs(P, lineMaterial({ width: 0.8, opacity: 0.5, vertexColors: true }), { order: 1 });
    this.refAt = new Uint32Array(data.frames + 2);
    this.refCount = 0;
    this.headGeo = new THREE.BufferGeometry();
    this.headPos = new Float32Array(3 * data.nTracks + 3);
    this.headCol = new Float32Array(3 * data.nTracks + 3);
    this.headGeo.setAttribute("position", new THREE.BufferAttribute(this.headPos, 3).setUsage(THREE.DynamicDrawUsage));
    this.headGeo.setAttribute("color", new THREE.BufferAttribute(this.headCol, 3).setUsage(THREE.DynamicDrawUsage));
    this.headMat = new THREE.PointsMaterial({ size: 4, sizeAttenuation: false, vertexColors: true, map: dotTexture(),
                                              alphaTest: 0.5, transparent: true, depthWrite: false });
    this.headMat.baseSize = 4;
    this.heads = new THREE.Points(this.headGeo, this.headMat);
    this.heads.frustumCulled = false;
    this.heads.renderOrder = 12;
    // the drawn crowd (what the space turns) and, for "veil", the same once more through the ground
    this.crowd = new THREE.Group();
    this.crowd.add(this.solid.obj, this.guess.obj, this.heads);
    const ghostLines = (obj, order) => {
      const o = new LineSegments2(obj.geometry, obj.material.clone());
      o.material.depthTest = false;
      o.frustumCulled = false;
      o.renderOrder = order;
      return o;
    };
    const gh = new THREE.Points(this.headGeo, this.headMat.clone());
    gh.material.depthTest = false;
    gh.material.alphaTest = 0.5 * GHOST;
    gh.frustumCulled = false;
    gh.renderOrder = 15;
    this.ghost = [ghostLines(this.solid.obj, 13), ghostLines(this.guess.obj, 14), gh];
    for (const g of this.ghost) { g.visible = false; this.crowd.add(g); }
    this.under = "same";
    this.group = new THREE.Group();
    this.group.add(this.ref.obj, this.crowd);
    this.ref.obj.visible = false;
    this.dirtyFrom = 0;
    this.stats = { perform: 0, rules: 0, speed: 0, build: 0 };
  }

  _rules() { return new Rules(this.d, this.cfg, this.classes, { rest: this.restDraw }); }

  materials() {
    return [this.solid.material, this.guess.material, this.ref.material, this.headMat, ...this.ghost.map((g) => g.material)];
  }

  /** Dots about 1.6 px long, 4 px apart at the orbit target (ppm: pixels per metre there). */
  setDashScale(ppm) {
    const m = this.guess.material;
    m.dashSize = 1.6 * m.linewidth / ppm;
    m.gapSize = 3.2 / ppm + m.dashSize;
    this._ghosts();
  }

  /* The crowd below the ground: "same" (as above), "dim" (darker), "veil" (the ground hides it and
     it shows through faintly: the view's ground is opaque then, scene.js setUnder). */
  setUnder(v) {
    if (!UNDER.includes(v) || v === this.under) return;
    this.under = v;
    for (const g of this.ghost) g.visible = v === "veil";
    this._ghosts();
    this.dirtyFrom = 0;
  }

  _ghosts() {                          // the veil's materials follow the crowd's
    const [s, g, h] = this.ghost.map((o) => o.material);
    for (const [a, m] of [[s, this.solid.material], [g, this.guess.material]]) {
      a.linewidth = m.linewidth; a.dashSize = m.dashSize; a.gapSize = m.gapSize; a.opacity = GHOST * m.opacity;
    }
    h.size = this.headMat.size;
    h.opacity = GHOST;
  }

  /* The reference: the machine at rest, i.e. Live's own result with the phone untouched (the start
     camera, a level space): the rules on the start camera's drawing, once. Thin grey lines. */
  computeReference() {
    const d = this.d, g = this.restDraw;
    this.rest = this._rules().all(g.X, g.state);
    const { A, B, near, start, n: P } = d.pairs, X = this.rest.X, kind = this.rest.kind, part = this.rest.part;
    let k = 0;
    for (let f = 0; f <= d.frames; f++) {
      this.refAt[f] = k;
      for (let s = start[f]; s < (f < d.frames ? start[f + 1] : P); s++) {
        const a = A[s], b = B[s];
        if (!near[s] || kind[a] >= DROPPED || kind[b] >= DROPPED || part[a] !== part[b]) continue;
        const ja = 3 * a, jb = 3 * b;
        this.ref.set(k++, X[ja], X[ja + 1], X[ja + 2] + LIFT, X[jb], X[jb + 1], X[jb + 2] + LIFT, 0.45, 0.45, 0.45);
      }
    }
    this.refAt[d.frames + 1] = k;
    this.refCount = k;
    this.ref.commit(k);
    return this.rest;
  }

  setMode(m) {
    if (m === this.mode) return;
    this.mode = m;
    this.dirtyFrom = 0;
  }

  /** Cause mode: the guesses Live caused violet, the footage's white, the solid lines dimmed. */
  setCause(on) {
    this.cause = !!on;
    this.solid.material.opacity = this.cause ? DIM : 1;
    this._ghosts();
    this.dirtyFrom = 0;
  }

  /** The performed drawing so far, a copy (a pass, 08 b), or null when nothing is drawn yet. */
  snapshot() { return this.perfFrame >= 0 ? snapshot(this.rPerf.out, this.perfFrame) : null; }

  /** Forget the performed trails (a new pass, a new clip). */
  resetPerformed() {
    this.perf.X.fill(NaN);
    this.perf.state.fill(NOT_YET);
    this.perfFrame = -1;
    this.rPerf.reset();
    this.dirtyFrom = 0;
  }

  _judge(a, b) {                       // performed: the rules on frames a..b; widens dirtyFrom
    const t0 = performance.now(), R = this.rPerf;
    R.frames(this.perf.X, this.perf.state, a, b);
    if (R.dirty[0] < this.dirtyFrom) this.dirtyFrom = R.dirty[0];
    this.stats.rules += performance.now() - t0;
  }

  /* Performed mode: bring the trails to frame f with camera cam (in the space's own frame) and the
     space's turn S. poseChanged: the current frame is projected again (live feedback while
     paused). Returns whether anything changed. */
  perform(f, cam, S, poseChanged) {
    const d = this.d, fs = d.frameStart, P = this.perf, C0 = this.sc.C, t0 = performance.now();
    f = Math.max(0, Math.min(f, d.frames - 1));
    this.stats.rules = 0;
    let changed = false;
    if (f < this.perfFrame) {                        // scrubbed back: clear what lies after f
      P.X.fill(NaN, 3 * fs[f + 1], 3 * fs[this.perfFrame + 1]);
      P.state.fill(NOT_YET, fs[f + 1], fs[this.perfFrame + 1]);
      this._judge(f + 1, this.perfFrame);
      this.dirtyFrom = Math.min(this.dirtyFrom, f);
      this.perfFrame = f;
      poseChanged = true;
      changed = true;
    }
    let a = -1;
    if (f > this.perfFrame) a = this.perfFrame + 1;  // forward: every frame up to f, this pose
    else if (poseChanged) a = f;
    if (a >= 0) {
      project(this.sc, cam, d.X, P, fs[a], fs[f + 1]);
      applySpace(S, C0, P.X, P.X, fs[a], fs[f + 1]);
      this.perfFrame = f;
      this.dirtyFrom = Math.min(this.dirtyFrom, a);
      this._judge(a, f);
      changed = true;
    }
    this.stats.perform = performance.now() - t0;
    return changed;
  }

  /* Whole clip: the pool's ground projection of every point (local: {X, state}, the space's own
     frame), judged as a whole, then turned by S into the world (turnWhole). One turn for every
     point is rigid, and the rules only measure distances, times and velocities, so judging and
     then turning is the same as turning and then judging: a new S alone needs neither a new
     projection nor the rules again. */
  judgeWhole(local) {
    const d = this.d, t0 = performance.now();
    if (!this.rWhole) { this.rWhole = this._rules(); this.wholeX = new Float32Array(3 * d.n); }
    this.rWhole.all(local.X, local.state);
    this.stats.rules = performance.now() - t0;
  }

  /** Whole clip: the judged drawing into the world, turned by S (after judgeWhole). */
  turnWhole(S) {
    if (!this.rWhole) return;
    const o = this.rWhole.out;
    applySpace(S, this.sc.C, o.X, this.wholeX);
    this.whole = { X: this.wholeX, kind: o.kind, cause: o.cause, part: o.part };
    this.dirtyFrom = 0;
  }

  /** The rules' output shown now (performed or whole), or null. */
  get out() { return this.mode === "performed" ? this.rPerf.out : this.whole; }

  /* Rebuild the lines for frame f. In performed mode only the frames from dirtyFrom on are
     written again (a margin for the speed's smoothing). */
  build(f) {
    const d = this.d, t0 = performance.now();
    const performed = this.mode === "performed", o = this.out;
    if (!o) return;
    const X = o.X, kind = o.kind, cause = o.cause, part = o.part, src = d.src;
    speedAsDrawn(X, d.t, d.index, this.speed, SPEED_WIN, 3);         // in space: tilts move people up and down
    const t1 = performance.now();
    const fEnd = performed ? Math.min(f, this.perfFrame) : d.frames - 1;
    const f0 = performed ? Math.max(0, Math.min(this.dirtyFrom, fEnd + 1) - MARGIN) : 0;
    const { A, B, near, start } = d.pairs;
    const solid = this.solid, gs = this.guess, arc = this.arc, v = this.speed, causeOn = this.cause;
    const [vr, vg, vb] = this.tint, sc = this.sc, dim = this.under === "dim";
    let si = f0 === 0 ? 0 : this.atSolid[f0], gi = f0 === 0 ? 0 : this.atGuess[f0];
    const s0 = si, g0 = gi;
    for (let ff = f0; ff <= fEnd; ff++) {
      this.atSolid[ff] = si; this.atGuess[ff] = gi;
      for (let s = start[ff]; s < start[ff + 1]; s++) {
        const a = A[s], b = B[s];
        // a line joins two drawn points of the same identity (dropped / not yet: nothing)
        if (!near[s] || kind[a] >= DROPPED || kind[b] >= DROPPED || part[a] !== part[b]) { arc[b] = 0; continue; }
        const ja = 3 * a, jb = 3 * b;
        const ax = X[ja], ay = X[ja + 1], az = X[ja + 2] + LIFT, bx = X[jb], by = X[jb + 1], bz = X[jb + 2] + LIFT;
        if (!(ax === ax && bx === bx)) { arc[b] = 0; continue; }
        const L = Math.hypot(bx - ax, by - ay, bz - az);
        arc[b] = arc[a] + L;
        // "dim": each end below the ground darker
        const ka = dim && X[ja + 2] < groundHeight(sc, ax, ay) - BELOW_M ? DIM_BELOW : 1;
        const kb = dim && X[jb + 2] < groundHeight(sc, bx, by) - BELOW_M ? DIM_BELOW : 1;
        if (guessed(kind, src, a) || guessed(kind, src, b)) {
          // cause mode: violet when Live caused it (an original guess stays white)
          const live = causeOn && (cause[a] === LIVE || cause[b] === LIVE) && cause[a] !== FOOTAGE && cause[b] !== FOOTAGE;
          const r = live ? vr : 1, g = live ? vg : 1, bb = live ? vb : 1;
          gs.set(gi, ax, ay, az, bx, by, bz, r * ka, g * ka, bb * ka, r * kb, g * kb, bb * kb);
          gs.setDist(gi, arc[a], arc[b]);
          gi++;
        } else {
          rampColor(v[a], _c);
          solid.set(si++, ax, ay, az, bx, by, bz, _c[0] * ka, _c[1] * ka, _c[2] * ka, _c[0] * kb, _c[1] * kb, _c[2] * kb);
        }
      }
    }
    this.atSolid[fEnd + 1] = si; this.atGuess[fEnd + 1] = gi;
    solid.commit(si, s0);
    gs.commit(gi, g0);
    this.dirtyFrom = performed ? fEnd + 1 : 0;
    this._heads(f, o);
    if (this.rest) this.ref.commit(performed ? this.refAt[Math.min(f, d.frames - 1) + 1] : this.refCount);
    this.stats.speed = t1 - t0;
    this.stats.build = performance.now() - t1;
  }

  /** Whole clip, the frame moved but not the pose: only the heads change. */
  frameOnly(f) {
    if (this.mode === "whole" && this.whole) this._heads(f, this.whole);
  }

  _heads(f, o) {
    const d = this.d, fs = d.frameStart, X = o.X, kind = o.kind, cause = o.cause;
    let k = 0;
    if (f >= 0 && f < d.frames)
      for (let i = fs[f]; i < fs[f + 1] && k < d.nTracks; i++) {
        if (kind[i] >= DROPPED) continue;
        const j = 3 * i;
        if (!(X[j] === X[j])) continue;
        this.headPos[3 * k] = X[j]; this.headPos[3 * k + 1] = X[j + 1]; this.headPos[3 * k + 2] = X[j + 2] + LIFT;
        if (!guessed(kind, d.src, i)) rampColor(this.speed[i], this.headCol, 3 * k);
        else if (this.cause && cause[i] === LIVE) this.headCol.set(this.tint, 3 * k);
        else this.headCol.fill(1, 3 * k, 3 * k + 3);
        if (this.under === "dim" && X[j + 2] < groundHeight(this.sc, X[j], X[j + 1]) - BELOW_M)
          for (let c = 3 * k; c < 3 * k + 3; c++) this.headCol[c] *= DIM_BELOW;
        k++;
      }
    this.headGeo.setDrawRange(0, k);
    this.headGeo.attributes.position.needsUpdate = true;
    this.headGeo.attributes.color.needsUpdate = true;
    this.nHeads = k;
  }

  /** Counts for the HUD at frame f: placed (solid), guessed (white dashes), dropped. */
  counts(f) {
    const o = this.out;
    if (!o) return null;
    const fs = this.d.frameStart, kind = o.kind, src = this.d.src;
    let placed = 0, guess = 0, dropped = 0;
    for (let i = fs[f]; i < fs[f + 1]; i++) {
      const k = kind[i];
      if (k === NOT_YET) continue;
      if (k === DROPPED) dropped++; else if (k !== SOLID || src[i]) guess++; else placed++;
    }
    return { placed, guessed: guess, dropped };
  }

  setRef(on) { this.ref.obj.visible = !!on; }

  setWidth(scale) {
    for (const m of [this.solid.material, this.guess.material]) m.linewidth = this.width * scale;
    this.headMat.size = this.headMat.baseSize * Math.sqrt(scale);
    this._ghosts();
  }

  dispose() {
    for (const s of [this.solid, this.guess, this.ref]) s.dispose();
    for (const g of this.ghost) g.material.dispose();
    this.headGeo.dispose(); this.headMat.map.dispose(); this.headMat.dispose();
  }
}

// ---------------------------------------------------------------------------- past passes
/* The drawings of the passes before the current one (08 b, "Passes"): each camera's drawing of
   each pass, frozen (no heads), drawn as the performed trails are (the same lexicon, cause mode,
   veil), in the world where it was drawn. One set of materials for all of them (the section cuts
   them, the look widens them). Kept apart from the cameras: a camera that leaves the scene leaves
   its drawings here. */
export class Past {
  constructor() {
    this.group = new THREE.Group();
    this.width = 1.15;
    this.solidMat = lineMaterial({ width: this.width, vertexColors: true });
    this.guessMat = lineMaterial({ width: this.width, vertexColors: true, dashed: true, dash: 0.12, gap: 0.3 });
    this.ghostSolid = this.solidMat.clone(); this.ghostSolid.depthTest = false;
    this.ghostGuess = this.guessMat.clone(); this.ghostGuess.depthTest = false;
    this.items = [];                   // {key (the pass), cam (its name), data, snap, tint, obj, solid, guess, ghosts}
    this.cause = false;
    this.under = "same";
    this._ghosts();
  }

  materials() { return [this.solidMat, this.guessMat, this.ghostSolid, this.ghostGuess]; }

  get segments() { return this.items.reduce((a, it) => a + (it.solid ? it.solid.n : 0) + (it.guess ? it.guess.n : 0), 0); }

  /** A camera's drawing of pass `key` (data: parseTracks of its package, offset: its place). */
  add(key, cam, data, snap, offset, tint = VIOLET_RGB) {
    const it = { key, cam, data, snap, tint: Float32Array.from(tint), obj: new THREE.Group(), solid: null, guess: null, ghosts: [] };
    it.obj.position.set(...offset);
    this.group.add(it.obj);
    this._build(it);
    this.items.push(it);
    return it;
  }

  has(key) { return this.items.some((it) => it.key === key); }

  /** Pass `key`'s drawings go. */
  remove(key) {
    for (const it of this.items.filter((x) => x.key === key)) this._drop(it);
    this.items = this.items.filter((x) => x.key !== key);
  }

  /** One drawing goes (a camera's layer replaced or cleared, 08 b "Sandbox / collage"). */
  drop(it) {
    const i = this.items.indexOf(it);
    if (i < 0) return;
    this._drop(it);
    this.items.splice(i, 1);
  }

  clear() { for (const it of this.items) this._drop(it); this.items = []; }

  _drop(it) {
    this._free(it);
    this.group.remove(it.obj);
  }

  _free(it) {
    for (const s of [it.solid, it.guess]) if (s) { it.obj.remove(s.obj); s.geo.dispose(); }
    for (const g of it.ghosts) it.obj.remove(g);
    it.solid = it.guess = null;
    it.ghosts = [];
  }

  /* The lines of one drawing: counted, then written into buffers of that size. */
  _build(it) {
    this._free(it);
    const d = it.data, o = it.snap, n = d.n;
    const speed = new Float64Array(n), arc = new Float32Array(n);
    speedAsDrawn(o.X, d.t, d.index, speed, SPEED_WIN, 3);
    const fEnd = Math.min(o.upTo, d.frames - 1);
    const nS = this._fill(d, o, speed, arc, it.tint, fEnd, null, null);
    const solid = new Segs(nS[0], this.solidMat, { order: 10 });
    const guess = new Segs(nS[1], this.guessMat, { distances: true, order: 11 });
    const k = this._fill(d, o, speed, arc, it.tint, fEnd, solid, guess);
    solid.commit(k[0]); guess.commit(k[1]);
    it.solid = solid; it.guess = guess;
    it.obj.add(solid.obj, guess.obj);
    it.ghosts = [[solid, this.ghostSolid, 13], [guess, this.ghostGuess, 14]].map(([s, m, order]) => {
      const g = new LineSegments2(s.geo, m);
      g.frustumCulled = false; g.renderOrder = order; g.visible = this.under === "veil";
      it.obj.add(g);
      return g;
    });
  }

  /* As TrackView.build for the frames 0..fEnd ("dim" is drawn as "same"); counts only when the
     segments are null. Returns [solid, guess] counts. */
  _fill(d, o, v, arc, tint, fEnd, solid, guess) {
    const X = o.X, kind = o.kind, cause = o.cause, part = o.part, src = d.src;
    const { A, B, near, start } = d.pairs, causeOn = this.cause, [vr, vg, vb] = tint;
    let si = 0, gi = 0;
    for (let ff = 0; ff <= fEnd; ff++) {
      for (let s = start[ff]; s < start[ff + 1]; s++) {
        const a = A[s], b = B[s];
        if (!near[s] || kind[a] >= DROPPED || kind[b] >= DROPPED || part[a] !== part[b]) { arc[b] = 0; continue; }
        const ja = 3 * a, jb = 3 * b;
        const ax = X[ja], ay = X[ja + 1], az = X[ja + 2] + LIFT, bx = X[jb], by = X[jb + 1], bz = X[jb + 2] + LIFT;
        if (!(ax === ax && bx === bx)) { arc[b] = 0; continue; }
        arc[b] = arc[a] + Math.hypot(bx - ax, by - ay, bz - az);
        if (guessed(kind, src, a) || guessed(kind, src, b)) {
          if (guess) {
            const live = causeOn && (cause[a] === LIVE || cause[b] === LIVE) && cause[a] !== FOOTAGE && cause[b] !== FOOTAGE;
            const r = live ? vr : 1, g = live ? vg : 1, bb = live ? vb : 1;
            guess.set(gi, ax, ay, az, bx, by, bz, r, g, bb);
            guess.setDist(gi, arc[a], arc[b]);
          }
          gi++;
        } else {
          if (solid) { rampColor(v[a], _c); solid.set(si, ax, ay, az, bx, by, bz, _c[0], _c[1], _c[2]); }
          si++;
        }
      }
    }
    return [si, gi];
  }

  /** Cause mode (as TrackView.setCause): the drawings again in its colours. */
  setCause(on) {
    if (!!on === this.cause) return;
    this.cause = !!on;
    this.solidMat.opacity = this.cause ? DIM : 1;
    this._ghosts();
    for (const it of this.items) this._build(it);
  }

  setUnder(v) {
    if (!UNDER.includes(v) || v === this.under) return;
    this.under = v;
    for (const it of this.items) for (const g of it.ghosts) g.visible = v === "veil";
    this._ghosts();
  }

  setWidth(scale) {
    for (const m of [this.solidMat, this.guessMat]) m.linewidth = this.width * scale;
    this._ghosts();
  }

  setDashScale(ppm) {
    const m = this.guessMat;
    m.dashSize = 1.6 * m.linewidth / ppm;
    m.gapSize = 3.2 / ppm + m.dashSize;
    this._ghosts();
  }

  _ghosts() {
    for (const [a, m] of [[this.ghostSolid, this.solidMat], [this.ghostGuess, this.guessMat]]) {
      a.linewidth = m.linewidth; a.dashSize = m.dashSize; a.gapSize = m.gapSize; a.opacity = GHOST * m.opacity;
    }
  }

  dispose() {
    this.clear();
    for (const m of this.materials()) m.dispose();
  }
}
