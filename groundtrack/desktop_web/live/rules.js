/* Live controller: the machine's own rules on the drawn positions (plans/live_controller/08 (c)).
   An ES module with no view dependency. Its Python reference is groundtrack/live_rules.py: the same
   rules on the same arrays, the same kinds, causes and parts.

   White dashes mean one thing: the machine did not see, so it guessed. The tracks are drawn where the
   moved camera (and the turned space) puts them, and the machine applies the rules it applies to the
   footage to those drawn positions in the world: after the space tilt, so in 3D (on flat, level
   ground the same as groundtrack's 2D rules). The detector is not re-run: the footage's points are
   the detections, and its guesses (predicted, stitched, interpolated) stay guesses.

   Per track (one identity of the footage), over its points in frame order:
   - visibility: a lost point (behind, out of view, past the horizon) ends a sighting. After the
     track's latest segment end the machine predicts along that segment's end velocity for
     track_buffer_s, then drops the track. A return within the stitch window is bridged;
     otherwise it is a new identity.
   - cleaning (groundtrack/trajectories.py, per class max_speed + jump_tolerance_m), per sighting:
     remove_spikes drops a single impossible jump, then interpolated; split_segments splits at a
     sustained one (or a gap over cleaning.max_gap_s). Distances in 3D.
   - stitching (trajectories.stitch_segments on the track's segments): pieces that end and start up
     to cleaning.stitch_max_gap_s apart and pass its test (the jump, the stitch radius around the
     predicted start) are joined, greedy by the smallest error; the gap is bridged. One slot per
     frame: the chain with the most kept points takes its span first, so a short piece the stitch
     jumped over (a two-frame excursion) is bridged by it, as the pipeline's output is once
     len < 2 / min_track_s drop that piece.

   Out, per point: X (where the machine draws it, NaN: nothing), kind (SOLID placed, PREDICTED,
   STITCHED, INTERPOLATED, DROPPED nothing drawn, NOT_YET not performed yet), cause and part (the
   identity within its footage track: 0, then 1 after the first new identity, ...).
   The cause is measured against the machine at rest (08 (c)): Live's own result with the phone
   untouched (the start camera, the space level), over the same points (the same prefix when
   performed). FOOTAGE: an original guess (whatever Live does), or a guess / drop the machine at
   rest makes too (Live's setup: R_max, the camera's wobble while filming, the steps); LIVE: a guess
   or drop on a point the machine at rest places, caused by the interference; NONE: placed. An
   untouched phone gives no LIVE.

   Performed trails run incrementally: frames(P, state, fa, fb) re-judges the tracks with points in
   those frames, each over its prefix (up to its first NOT_YET point), so the result is the batch
   result of the clip so far. The newest points are provisional: a spike is known one point later,
   a stitch over the first 0.6 s after the return (the start-velocity fit). */

import { DRAWN, project, startCamera } from "./projection.js";

export const SOLID = 0, PREDICTED = 1, STITCHED = 2, INTERPOLATED = 3, DROPPED = 4, NOT_YET = 255;
export const KINDS = ["solid", "predicted", "stitched", "interpolated", "dropped"];
export const NONE = 0, FOOTAGE = 1, LIVE = 2;
export const CAUSES = ["none", "footage", "live"];
export const FIT_S = 0.6;          // trajectories.Segment.end_state: the line fit at a segment's ends
export const SPIKE_PASSES = 3;     // trajectories.remove_spikes
const EPS_S = 1e-6;                // s: the prediction's time limit is inclusive (float times)

/** [max_speed, jump_tolerance_m, stitch_radius_m] of a class name; in no group: no cleaning. */
export function classParams(rules, name) {
  for (const g of Object.values(rules.groups || {}))
    if ((g.classes || []).includes(name)) return [+g.max_speed, +g.jump_tolerance_m, +g.stitch_radius_m];
  return [Infinity, 0, Infinity];
}

// ---------------------------------------------------------------------------- one track (scratch)
/* A track's working arrays, by position r along the track (frame order). */
class Scratch {
  constructor(L) {
    L = Math.max(1, L | 0);
    this.L = L;
    this.tt = new Float64Array(L); this.px = new Float64Array(L); this.py = new Float64Array(L);
    this.pz = new Float64Array(L);
    this.keep = new Uint8Array(L); this.sid = new Int32Array(L);
    this.idx = new Int32Array(L); this.bad = new Int32Array(L); this.K = new Int32Array(L);
    this.q0 = new Int32Array(L); this.q1 = new Int32Array(L);
    this.ends = new Float64Array(6 * L); this.starts = new Float64Array(6 * L);
    // stitching: links, chains (by segment), the slots' owners and what they hold
    this.nxt = new Int32Array(L); this.hasPrev = new Uint8Array(L); this.usedEnd = new Uint8Array(L);
    this.chHead = new Int32Array(L); this.chSize = new Int32Array(L); this.chOrd = new Int32Array(L);
    this.pnum = new Int32Array(L); this.endChain = new Int32Array(L);
    this.owner = new Int32Array(L); this.kd = new Uint8Array(L);
    this.ox = new Float64Array(L); this.oy = new Float64Array(L); this.oz = new Float64Array(L);
    // a judged track (Rules), before it is compared and written out
    this.wk = new Uint8Array(L); this.wp = new Uint16Array(L);
    this.wx = new Float64Array(L); this.wy = new Float64Array(L); this.wz = new Float64Array(L);
    this.growCands(L);
  }

  growCands(m) {
    this.cErr = new Float64Array(m); this.cI = new Int32Array(m); this.cJ = new Int32Array(m);
    this.cOrd = new Int32Array(m);
  }
}

function stepOk(s, i, j, vmax, tol) {         // trajectories._step_ok, in 3D
  const dx = s.px[j] - s.px[i], dy = s.py[j] - s.py[i], dz = s.pz[j] - s.pz[i];
  return Math.sqrt(dx * dx + dy * dy + dz * dz) <= vmax * Math.abs(s.tt[j] - s.tt[i]) + tol;
}

/* trajectories.remove_spikes on positions r0..r1-1 (keep set there): clears keep of the spikes. */
function spikes(s, r0, r1, vmax, tol, passes = SPIKE_PASSES) {
  const keep = s.keep, idx = s.idx, bad = s.bad;
  for (let p = 0; p < passes; p++) {
    let m = 0;
    for (let r = r0; r < r1; r++) if (keep[r]) idx[m++] = r;
    if (m < 3) break;
    let nb = 0;
    for (let q = 0; q < m; q++) {
      const i = idx[q];
      if (q > 0 && q + 1 < m) {
        const a = idx[q - 1], b = idx[q + 1];
        if (!stepOk(s, a, i, vmax, tol) && !stepOk(s, i, b, vmax, tol) && stepOk(s, a, b, vmax, tol)) bad[nb++] = i;
      } else if (q === 0) {                        // first point
        if (!stepOk(s, i, idx[1], vmax, tol) && stepOk(s, idx[1], idx[2], vmax, tol)) bad[nb++] = i;
      } else if (!stepOk(s, idx[q - 1], i, vmax, tol) && stepOk(s, idx[q - 2], idx[q - 1], vmax, tol)) {
        bad[nb++] = i;                             // last point
      }
    }
    if (!nb) break;
    for (let b = 0; b < nb; b++) keep[bad[b]] = 0;
  }
}

/* trajectories.Segment.end_state in 3D, of the kept points K[q0..q1]: position and velocity at the
   end (or start) from a line fit over FIT_S seconds; fewer than 3 points: the end point, no
   velocity. Into out[o..o+5] (p, v). */
function endState(s, K, q0, q1, atEnd, out, o) {
  const tt = s.tt;
  let qa = q0, qb = q1;
  if (atEnd) { const lim = tt[K[q1]] - FIT_S; qa = q1; while (qa > q0 && tt[K[qa - 1]] >= lim) qa--; }
  else { const lim = tt[K[q0]] + FIT_S; qb = q0; while (qb < q1 && tt[K[qb + 1]] <= lim) qb++; }
  const n = qb - qa + 1, re = atEnd ? K[qb] : K[qa];
  if (n < 3 || !(tt[K[qb]] - tt[K[qa]] > 0)) {
    out[o] = s.px[re]; out[o + 1] = s.py[re]; out[o + 2] = s.pz[re];
    out[o + 3] = out[o + 4] = out[o + 5] = 0;
    return;
  }
  let mt = 0, mx = 0, my = 0, mz = 0;
  for (let q = qa; q <= qb; q++) { const r = K[q]; mt += tt[r]; mx += s.px[r]; my += s.py[r]; mz += s.pz[r]; }
  mt /= n; mx /= n; my /= n; mz /= n;
  let stt = 0, stx = 0, sty = 0, stz = 0;
  for (let q = qa; q <= qb; q++) {
    const r = K[q], d = tt[r] - mt;
    stt += d * d; stx += d * (s.px[r] - mx); sty += d * (s.py[r] - my); stz += d * (s.pz[r] - mz);
  }
  const vx = stx / stt, vy = sty / stt, vz = stz / stt, dr = tt[re] - mt;
  out[o] = mx + vx * dr; out[o + 1] = my + vy * dr; out[o + 2] = mz + vz * dr;
  out[o + 3] = vx; out[o + 4] = vy; out[o + 5] = vz;
}

/* The rules on positions 0..L-1 of the scratch (tt, px/py/pz, keep = seen, Lp the prefix). Calls
   put(r, kind, part, x, y, z) for every r < Lp. */
function judge(s, Lp, vmax, tol, radius, rules, put) {
  const keep = s.keep, sid = s.sid, K = s.K, tt = s.tt, q0 = s.q0, q1 = s.q1;
  // sightings -> spikes
  let nsid = 0;
  for (let r = 0; r < Lp;) {
    if (!keep[r]) { sid[r] = -1; r++; continue; }
    let r1 = r;
    while (r1 < Lp && keep[r1]) sid[r1++] = nsid;
    spikes(s, r, r1, vmax, tol);
    nsid++;
    r = r1;
  }
  // kept points -> segments (split at a new sighting, a long gap, an impossible step)
  const maxGap = +rules.max_gap_s;
  let m = 0, nseg = 0;
  for (let r = 0; r < Lp; r++) if (keep[r]) K[m++] = r;
  for (let q = 0; q < m; q++) {
    if (q === 0 || sid[K[q]] !== sid[K[q - 1]] || tt[K[q]] - tt[K[q - 1]] > maxGap ||
        !stepOk(s, K[q - 1], K[q], vmax, tol)) q0[nseg++] = q;
    q1[nseg - 1] = q;
  }
  // stitching (trajectories.stitch_segments on this track's segments): candidates up to
  // stitch_max_gap_s apart that pass the test, greedy by the smallest error, one link per end / start
  const smax = +rules.stitch_max_gap_s, ends = s.ends, starts = s.starts, nxt = s.nxt;
  const hasPrev = s.hasPrev, usedEnd = s.usedEnd;
  for (let g = 0; g < nseg; g++) {
    endState(s, K, q0[g], q1[g], true, ends, 6 * g);
    nxt[g] = -1; hasPrev[g] = 0; usedEnd[g] = 0;
  }
  if (rules.stitch !== false && nseg > 1) {
    for (let g = 0; g < nseg; g++) endState(s, K, q0[g], q1[g], false, starts, 6 * g);
    let nc = 0;
    for (let i = 0; i < nseg; i++) {
      const o = 6 * i, te = tt[K[q1[i]]];
      for (let j = i + 1; j < nseg; j++) {
        const gap = tt[K[q0[j]]] - te;
        if (gap > smax) break;
        if (!(gap > 0)) continue;
        const b = 6 * j, jx = starts[b] - ends[o], jy = starts[b + 1] - ends[o + 1], jz = starts[b + 2] - ends[o + 2];
        const ex = jx - 0.5 * (ends[o + 3] + starts[b + 3]) * gap, ey = jy - 0.5 * (ends[o + 4] + starts[b + 4]) * gap,
              ez = jz - 0.5 * (ends[o + 5] + starts[b + 5]) * gap;
        const err = Math.sqrt(ex * ex + ey * ey + ez * ez);
        if (!(Math.sqrt(jx * jx + jy * jy + jz * jz) <= vmax * gap + tol && err <= radius * (1 + gap))) continue;
        if (nc === s.cErr.length) {
          const e0 = s.cErr, i0 = s.cI, j0 = s.cJ;
          s.growCands(2 * nc);
          s.cErr.set(e0); s.cI.set(i0); s.cJ.set(j0);
        }
        s.cErr[nc] = err; s.cI[nc] = i; s.cJ[nc] = j; nc++;
      }
    }
    if (nc) {
      const ce = s.cErr, ci = s.cI, cj = s.cJ, ord = s.cOrd.subarray(0, nc);
      for (let c = 0; c < nc; c++) ord[c] = c;
      ord.sort((a, b) => (ce[a] - ce[b]) || (ci[a] - ci[b]) || (cj[a] - cj[b]));
      for (let c = 0; c < nc; c++) {
        const i = ci[ord[c]], j = cj[ord[c]];
        if (usedEnd[i] || hasPrev[j]) continue;
        nxt[i] = j; usedEnd[i] = 1; hasPrev[j] = 1;
      }
    }
  }
  // chains, the larger first (kept points, then the earlier); each takes the slots of its span not
  // taken yet: its kept points (solid), the dropped spikes inside its segments (interpolated), its
  // stitched gaps (stitched)
  const chHead = s.chHead, chSize = s.chSize, chOrd = s.chOrd, owner = s.owner, kd = s.kd;
  const ox = s.ox, oy = s.oy, oz = s.oz, px = s.px, py = s.py, pz = s.pz, endChain = s.endChain;
  let nch = 0;
  for (let g = 0; g < nseg; g++) {
    if (hasPrev[g]) continue;
    let size = 0, h = g, last = g;
    for (; h >= 0; h = nxt[h]) { size += q1[h] - q0[h] + 1; last = h; }
    chHead[nch] = g; chSize[nch] = size; chOrd[nch] = nch;
    s.pnum[nch] = -1;
    endChain[nch] = last;
    nch++;
  }
  chOrd.subarray(0, nch).sort((a, b) => (chSize[b] - chSize[a]) || (K[q0[chHead[a]]] - K[q0[chHead[b]]]));
  for (let r = 0; r < Lp; r++) owner[r] = -1;
  const claim = (c, r0, r1, i, j, k) => {
    const a = Math.max(r0, i + 1), b = Math.min(r1, j - 1);
    for (let r = a; r <= b; r++) {
      if (owner[r] >= 0) continue;
      const w = (tt[r] - tt[i]) / (tt[j] - tt[i]);
      owner[r] = c; kd[r] = k;
      ox[r] = px[i] + w * (px[j] - px[i]); oy[r] = py[i] + w * (py[j] - py[i]); oz[r] = pz[i] + w * (pz[j] - pz[i]);
    }
  };
  for (let n = 0; n < nch; n++) {
    const c = chOrd[n];
    for (let g = chHead[c]; g >= 0; g = nxt[g]) {
      const a = K[q0[g]], e = K[q1[g]];
      for (let q = q0[g]; q < q1[g]; q++) if (K[q + 1] > K[q] + 1) claim(c, a, e, K[q], K[q + 1], INTERPOLATED);
      if (nxt[g] >= 0) claim(c, e + 1, K[q0[nxt[g]]] - 1, e, K[q0[nxt[g]]], STITCHED);
      for (let q = q0[g]; q <= q1[g]; q++) {
        const r = K[q];
        if (owner[r] < 0) { owner[r] = c; kd[r] = SOLID; ox[r] = px[r]; oy[r] = py[r]; oz[r] = pz[r]; }
      }
    }
  }
  // parts in the order the chains first appear; the rest: after the latest chain end (a chain that
  // owns its end), predicted along its last velocity for track_buffer_s, then dropped
  const pnum = s.pnum, buf = +rules.track_buffer_s;
  let np = 0, last = -1, e = -1, o = 0;
  for (let r = 0; r < Lp; r++) {
    const c = owner[r];
    if (c >= 0) {
      if (pnum[c] < 0) pnum[c] = np++;
      const lg = endChain[c];
      if (K[q1[lg]] === r) { last = c; e = r; o = 6 * lg; }
      put(r, kd[r], pnum[c], ox[r], oy[r], oz[r]);
      continue;
    }
    if (last < 0) { put(r, DROPPED, 0, NaN, NaN, NaN); continue; }
    const dt = tt[r] - tt[e];
    if (dt <= buf + EPS_S) put(r, PREDICTED, pnum[last], px[e] + ends[o + 3] * dt, py[e] + ends[o + 4] * dt, pz[e] + ends[o + 5] * dt);
    else put(r, DROPPED, pnum[last], NaN, NaN, NaN);
  }
}

function load(s, L, t, P, state, upTo = -1) {   // plain arrays of one track -> scratch; returns Lp
  for (let r = 0; r < L; r++) {
    if (upTo < 0 ? state[r] === NOT_YET : r >= upTo) return r;
    s.tt[r] = t[r];
    const x = P[3 * r], y = P[3 * r + 1], z = P[3 * r + 2];
    s.px[r] = x; s.py[r] = y; s.pz[r] = z;
    s.keep[r] = state[r] === DRAWN && x === x && y === y && z === z ? 1 : 0;
  }
  return L;
}

/* The cause of a point (08 (c)): an original guess, or a guess / drop where the machine at rest
   (rk: its kind there, over the same prefix) also guesses or drops, is the footage's; a guess or
   drop where the machine at rest places the point is the interference's. */
function causeOf(src, kd, rk) {
  return kd === NOT_YET ? NONE : src ? FOOTAGE : kd === SOLID ? NONE : rk !== SOLID ? FOOTAGE : LIVE;
}

/* The rules on one track (plain arrays, for tests and tools): t seconds, P flat xyz drawn positions
   (world), state the projection's states (DRAWN seen, else lost; NOT_YET ends the prefix), src
   tracks.bin sources; rest: {X, state} of the same track at rest (the start pose: project's out),
   judged over the same prefix for the cause (null: at rest every point placed). -> {X
   (Float64Array 3L), kind, cause, part}. */
export function judgeTrack(t, P, state, src, vmax, tol, radius, rules, rest = null) {
  const L = t.length, s = new Scratch(L);
  const X = new Float64Array(3 * L).fill(NaN), kind = new Uint8Array(L).fill(NOT_YET);
  const cause = new Uint8Array(L), part = new Uint16Array(L), rk = new Uint8Array(L);
  const Lp = load(s, L, t, P, state);
  judge(s, Lp, vmax, tol, radius, rules, (r, k, p, x, y, z) => {
    kind[r] = k; part[r] = p;
    X[3 * r] = x; X[3 * r + 1] = y; X[3 * r + 2] = z;
  });
  if (rest) {
    load(s, L, t, rest.X, rest.state, Lp);
    judge(s, Lp, vmax, tol, radius, rules, (r, k) => { rk[r] = k; });
  }
  for (let r = 0; r < Lp; r++) cause[r] = causeOf(src[r], kind[r], rk[r]);
  return { X, kind, cause, part };
}

/** trajectories.remove_spikes in 3D (t seconds, P flat xyz) -> keep (Uint8Array). */
export function removeSpikes(t, P, vmax, tol, passes = SPIKE_PASSES) {
  const n = t.length, s = new Scratch(n);
  for (let r = 0; r < n; r++) { s.tt[r] = t[r]; s.px[r] = P[3 * r]; s.py[r] = P[3 * r + 1]; s.pz[r] = P[3 * r + 2]; }
  s.keep.fill(1, 0, n);
  spikes(s, 0, n, vmax, tol, passes);
  return s.keep.slice(0, n);
}

/** trajectories.split_segments in 3D -> the segments' first indices, then n (Int32Array). */
export function splitSegments(t, P, vmax, tol, maxGap) {
  const n = t.length, s = new Scratch(n), cuts = [];
  for (let r = 0; r < n; r++) { s.tt[r] = t[r]; s.px[r] = P[3 * r]; s.py[r] = P[3 * r + 1]; s.pz[r] = P[3 * r + 2]; }
  for (let i = 0; i < n; i++) if (i === 0 || t[i] - t[i - 1] > maxGap || !stepOk(s, i - 1, i, vmax, tol)) cuts.push(i);
  cuts.push(n);
  return Int32Array.from(n ? cuts : [0]);
}

/** trajectories.Segment.end_state in 3D of a whole series -> Float64Array(6): p, v. */
export function segmentEndState(t, P, atEnd) {
  const n = t.length, s = new Scratch(n), out = new Float64Array(6);
  for (let r = 0; r < n; r++) {
    s.tt[r] = t[r]; s.px[r] = P[3 * r]; s.py[r] = P[3 * r + 1]; s.pz[r] = P[3 * r + 2]; s.K[r] = r;
  }
  endState(s, s.K, 0, n - 1, atEnd, out, 0);
  return out;
}

// ---------------------------------------------------------------------------- a clip's tracks
/* The rules on a clip's tracks, kept between calls (one per space / spot).
   tracks: tracks.js parseTracks() (t, X (the true points: opts.scene), cls, src, frame, frameStart,
   index = trackIndex(tid));
   rules: the run's config (live_rules.rules_config: track_buffer_s, max_gap_s, stitch,
   stitch_max_gap_s, groups {name: {classes, max_speed, jump_tolerance_m, stitch_radius_m}});
   classes: class code -> name (scene.json tracks.classes).
   The machine at rest (the cause), one of: opts.scene, the prepared scene: the start camera's
   drawing of tracks.X (opts.mode "ground" / "hinge", as the tracks are drawn), made here once; or
   opts.rest = {X, state} (project's out for every point; null: at rest every point placed).
   opts.out64: X (and the rest drawing) as Float64Array.
   this.out = {X (3n, NaN: nothing drawn), kind, cause, part}, updated in place; after each call
   this.dirty = [first, last] frame whose output changed ([Infinity, -1]: none). */
export class Rules {
  constructor(tracks, rules, classes = {}, opts = {}) {
    if (!opts.scene && opts.rest === undefined)
      throw new Error("Rules: opts.scene (or opts.rest) is needed: the cause is measured at rest");
    const n = tracks.t.length, { order, starts } = tracks.index, nT = starts.length - 1;
    this.tr = tracks; this.rules = rules; this.n = n; this.nTracks = nT;
    this.order = order; this.starts = starts;
    this.params = new Float64Array(3 * Math.max(nT, 1));
    this.trackOf = new Uint32Array(n);
    this.unknown = new Set();
    let maxL = 1;
    for (let k = 0; k < nT; k++) {
      const a = starts[k], b = starts[k + 1];
      maxL = Math.max(maxL, b - a);
      for (let r = a; r < b; r++) this.trackOf[order[r]] = k;
      const name = classes[String(tracks.cls[order[a]])] ?? "";
      const pr = classParams(rules, name);
      if (pr[0] === Infinity) this.unknown.add(name);
      this.params.set(pr, 3 * k);
    }
    this.s = new Scratch(maxL);
    this.frameStart = tracks.frameStart || null;
    this.touched = new Uint8Array(Math.max(nT, 1));
    const XT = opts.out64 ? Float64Array : Float32Array;
    this.out = { X: new XT(3 * n).fill(NaN), kind: new Uint8Array(n).fill(NOT_YET), cause: new Uint8Array(n),
                 part: new Uint16Array(n) };
    this.dirty = [Infinity, -1];
    const w = this.s;
    this._put = (r, kd, p, x, y, z) => { w.wk[r] = kd; w.wp[r] = p; w.wx[r] = x; w.wy[r] = y; w.wz[r] = z; };
    // the machine at rest: its kinds over the whole clip, and over a track's prefix restLp[k]
    this.rest = opts.scene ? project(opts.scene, startCamera(opts.scene), tracks.X,
                                     { X: new XT(3 * n), state: new Uint8Array(n) }, 0, -1, opts.mode || "ground")
                           : opts.rest;
    this.restKind = new Uint8Array(n);                 // SOLID: no rest given
    this.restPre = new Uint8Array(n);
    this.restLp = new Int32Array(Math.max(nT, 1)).fill(-1);
    if (this.rest) for (let k = 0; k < nT; k++) this._rest(k, starts[k + 1] - starts[k], this.restKind);
  }

  /** Every track (the whole clip, or after a reset): -> this.out. */
  all(P, state) {
    this.dirty = [Infinity, -1];
    for (let k = 0; k < this.nTracks; k++) this._track(k, P, state);
    return this.out;
  }

  /* Performed: the points of frames fa..fb were (re)projected or cleared (NOT_YET): the tracks with
     points there are judged again over their prefix. -> this.out. */
  frames(P, state, fa, fb = fa) {
    this.dirty = [Infinity, -1];
    const fs = this.frameStart, tk = this.touched;
    if (!fs) return this.all(P, state);
    fa = Math.max(0, fa | 0); fb = Math.min(fs.length - 2, fb | 0);
    if (fb < fa) return this.out;
    const i0 = fs[fa], i1 = fs[fb + 1];
    for (let i = i0; i < i1; i++) tk[this.trackOf[i]] = 1;
    for (let i = i0; i < i1; i++) {
      const k = this.trackOf[i];
      if (tk[k]) { tk[k] = 0; this._track(k, P, state); }
    }
    return this.out;
  }

  /** The given tracks (numbers in index order). */
  tracks(P, state, list) {
    this.dirty = [Infinity, -1];
    for (const k of list) this._track(k, P, state);
    return this.out;
  }

  /** Everything back to NOT_YET (a new clip pass). */
  reset() {
    this.out.X.fill(NaN); this.out.kind.fill(NOT_YET); this.out.cause.fill(0); this.out.part.fill(0);
    this.dirty = [0, this.tr.frameStart ? this.tr.frameStart.length - 2 : Infinity];
  }

  /* Track k's points into the scratch: up to its first NOT_YET (upTo < 0) or its first upTo.
     -> the prefix's length. */
  _load(k, P, state, upTo) {
    const s = this.s, a = this.starts[k], L = this.starts[k + 1] - a, order = this.order, t = this.tr.t;
    const lim = upTo < 0 ? L : upTo;
    for (let r = 0; r < lim; r++) {
      const i = order[a + r], st = state[i];
      if (upTo < 0 && st === NOT_YET) return r;
      s.tt[r] = t[i];
      const x = P[3 * i], y = P[3 * i + 1], z = P[3 * i + 2];
      s.px[r] = x; s.py[r] = y; s.pz[r] = z;
      s.keep[r] = st === DRAWN && x === x && y === y && z === z ? 1 : 0;
    }
    return lim;
  }

  _rest(k, Lp, dst) {               // the rest's kinds of track k over its prefix Lp -> dst (by point)
    const a = this.starts[k], order = this.order, pr = this.params;
    this._load(k, this.rest.X, this.rest.state, Lp);
    judge(this.s, Lp, pr[3 * k], pr[3 * k + 1], pr[3 * k + 2], this.rules, (r, kd) => { dst[order[a + r]] = kd; });
  }

  _atRest(k, P, state, Lp) {        // track k's prefix drawn as at rest (same input: the same kinds)
    const a = this.starts[k], order = this.order, RX = this.rest.X, RS = this.rest.state;
    for (let r = 0; r < Lp; r++) {
      const i = order[a + r], j = 3 * i;
      if (state[i] !== RS[i] || !eq(P[j], RX[j]) || !eq(P[j + 1], RX[j + 1]) || !eq(P[j + 2], RX[j + 2])) return false;
    }
    return true;
  }

  _track(k, P, state) {
    const s = this.s, a = this.starts[k], L = this.starts[k + 1] - a, order = this.order, pr = this.params;
    const Lp = this._load(k, P, state, -1);
    judge(s, Lp, pr[3 * k], pr[3 * k + 1], pr[3 * k + 2], this.rules, this._put);
    const wk = s.wk, wp = s.wp, wx = s.wx, wy = s.wy, wz = s.wz, src = this.tr.src;
    // the rest's kinds, when a guess or drop needs them: none given (all placed), the same input
    // (the same kinds), the whole clip's, or judged over this prefix (kept until the prefix changes)
    let rk = null, asRest = false;
    if (this.rest) {
      let need = false;
      for (let r = 0; r < Lp && !need; r++) need = wk[r] !== SOLID && !src[order[a + r]];
      if (need) {
        if (this._atRest(k, P, state, Lp)) asRest = true;
        else if (Lp === L) rk = this.restKind;
        else {
          if (this.restLp[k] !== Lp) { this._rest(k, Lp, this.restPre); this.restLp[k] = Lp; }
          rk = this.restPre;
        }
      }
    }
    const out = this.out, X = out.X, kind = out.kind, cause = out.cause, part = out.part;
    const frame = this.tr.frame, dirty = this.dirty;
    for (let r = 0; r < L; r++) {
      const i = order[a + r], j = 3 * i;
      let kd = NOT_YET, p = 0, x = NaN, y = NaN, z = NaN;
      if (r < Lp) { kd = wk[r]; p = wp[r]; x = wx[r]; y = wy[r]; z = wz[r]; }
      const c = causeOf(src[i], kd, asRest ? kd : rk ? rk[i] : SOLID);
      if (kind[i] === kd && part[i] === p && cause[i] === c && same(X[j], x) && same(X[j + 1], y) && same(X[j + 2], z)) continue;
      kind[i] = kd; part[i] = p; cause[i] = c; X[j] = x; X[j + 1] = y; X[j + 2] = z;
      const f = frame ? frame[i] : 0;
      if (f < dirty[0]) dirty[0] = f;
      if (f > dirty[1]) dirty[1] = f;
    }
  }
}

function eq(a, b) { return a === b || (a !== a && b !== b); }

const _f = new Float32Array(1);
function same(a, x) {             // a stored (Float32 or 64), x the new value: equal as stored, NaN = NaN
  if (a !== a) return x !== x;
  _f[0] = x;
  return a === x || a === _f[0];
}
