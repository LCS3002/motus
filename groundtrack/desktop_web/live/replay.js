/* Live view: a saved pass played back (plans/live_controller/05, "Replay"): its recording
   (record.js, live_passes.py) drives the cameras as a pose source (src "replay"), row by row in the
   order they were recorded, so the machine redraws exactly what the pass drew. Pure (no DOM, no
   three.js): the caller gives each camera's TrackView (tracks.js) and shows the result.
   - A camera starts at its camera_join (its drawing reset there, from its join frame); a
     camera_leave keeps what it drew, merged with a later stint as the live view merges them.
   - A row with "performed" performs its frame with the row's camera and space, "changed" as then.
   - The threshold the pass drew with (pass_start's, 1 in passes before it, 5 Nov) and its changes
     ("threshold" events) are the cameras' (TrackView.setThreshold) where they came. */

import { PERFORMED, CHANGED, readRecording } from "./record.js";
import { mergeSnaps } from "./passes.js";

export { readRecording };

/** A camera {R_wc, R_cw, C} from row r (R_cw = R_wcᵀ, exactly). */
export function rowCamera(rows, r, cam = { R_wc: new Float64Array(9), R_cw: new Float64Array(9), C: new Float64Array(3) }) {
  const R = rows.R(r);
  cam.R_wc.set(R);
  const o = cam.R_cw;
  o[0] = R[0]; o[1] = R[3]; o[2] = R[6]; o[3] = R[1]; o[4] = R[4]; o[5] = R[7]; o[6] = R[2]; o[7] = R[5]; o[8] = R[8];
  cam.C.set(rows.C(r));
  return cam;
}

export class Replay {
  /* rec: readRecording(); tracks: name -> TrackView (fresh or reset) for every recorded camera. */
  constructor(rec, tracks) {
    this.rec = rec;
    this.tracks = tracks;
    this.k = 0;                                        // the next clock row
    this.ev = 0;                                       // the next event
    this.cur = new Map([...rec.cams.keys()].map((n) => [n, 0]));   // the next row per camera
    this.cams = new Map([...rec.cams.keys()].map((n) => [n, rowCamera(rec.cams.get(n), 0, undefined)]));
    this.S = new Map([...rec.cams.keys()].map((n) => [n, new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1])]));
    this.frame = new Map([...rec.cams.keys()].map((n) => [n, 0]));
    this.join = new Map();                             // name -> its stint's join frame
    this.left = new Map();                             // name -> what it drew before it left (a snapshot)
    this.changed = new Set();                          // cameras whose drawing or camera moved since take()
    this.onEvent = null;
    for (const [n, rows] of rec.cams) if (rows.n === 0) this.cur.set(n, 0);
  }

  get clock() { return this.rec.clock; }
  get done() { return this.k >= this.rec.clock.n; }
  /** The recording's length (s, the view's clock). */
  get seconds() { const c = this.rec.clock; return c.n ? c.t[c.n - 1] : 0; }
  /** The shared clock at the last row applied. */
  get T() { return this.k ? this.rec.clock.T[this.k - 1] : 0; }
  get playing() { return this.k ? !!(this.rec.clock.flags[this.k - 1] & 1) : false; }

  _events(upTo) {
    const ev = this.rec.events;
    while (this.ev < ev.length && ev[this.ev].k <= upTo) {
      const e = ev[this.ev++];
      if (e.type === "camera_join" && this.tracks.has(e.camera)) this._join(e.camera, (e.value && e.value.join_frame) | 0);
      else if (e.type === "camera_leave" && this.tracks.has(e.camera)) this._leave(e.camera);
      else if (e.type === "pass_start") this._threshold(e.value && e.value.threshold);
      else if (e.type === "threshold") this._threshold(e.value);
      if (this.onEvent) this.onEvent(e);
    }
  }

  _threshold(k) {
    for (const [name, tv] of this.tracks) if (tv.setThreshold && tv.setThreshold(+k > 0 ? +k : 1)) this.changed.add(name);
  }

  _drew(name) { const tv = this.tracks.get(name); return tv.perfFrame > (this.join.get(name) ?? 0); }

  _join(name, jf) {
    const tv = this.tracks.get(name);
    if (this.join.has(name)) this._leave(name);        // joined again without a leave: as a leave
    tv.resetPerformed();
    tv.perfFrame = jf - 1;
    this.join.set(name, jf);
    this.changed.add(name);
  }

  _leave(name) {
    if (!this.join.has(name)) return;
    const tv = this.tracks.get(name), snap = this._drew(name) ? tv.snapshot() : null;
    if (snap) this.left.set(name, this.left.has(name) ? mergeSnaps(this.left.get(name), snap) : snap);
    this.join.delete(name);
  }

  /** One clock row: its events, then every camera's row at it. */
  step() {
    const k = this.k;
    this._events(k);
    for (const [name, rows] of this.rec.cams) {
      let r = this.cur.get(name);
      while (r < rows.n && rows.k(r) === k) {
        const cam = rowCamera(rows, r, this.cams.get(name)), S = this.S.get(name), f = rows.frame(r), fl = rows.flags(r);
        S.set(rows.S(r));
        this.frame.set(name, f);
        const tv = this.tracks.get(name);
        if (fl & PERFORMED && this.join.has(name)) {
          tv.perform(f, cam, S, !!(fl & CHANGED));
          if (this.onRow) this.onRow(name, rows, r, tv);
        }
        this.changed.add(name);
        r++;
      }
      this.cur.set(name, r);
    }
    this.k++;
    if (this.done) this._events(Infinity);             // what came after the last frame (the pass's end)
  }

  /** Every clock row up to the view's time t (s). Returns whether any was applied. */
  advance(t) {
    const c = this.rec.clock, k0 = this.k;
    while (this.k < c.n && c.t[this.k] <= t + 1e-9) this.step();
    return this.k > k0;
  }

  /** All of it (a check: the drawing as the pass drew it). */
  run() { while (!this.done) this.step(); return this; }

  /** The cameras that changed since the last take(). */
  take() { const s = [...this.changed]; this.changed.clear(); return s; }

  /** Each camera's drawing so far, as the live view saves it (stints merged): name -> snapshot. */
  drawings() {
    const out = new Map();
    for (const [name, snap] of this.left) out.set(name, { ...snap, X: snap.X.slice(), kind: snap.kind.slice(), cause: snap.cause.slice(), part: snap.part.slice() });
    for (const name of this.join.keys()) {
      if (!this._drew(name)) continue;
      const snap = this.tracks.get(name).snapshot();
      out.set(name, out.has(name) ? mergeSnaps(out.get(name), snap) : snap);
    }
    return out;
  }
}

/* A replayed drawing against the saved one (.bin unpacked): {points (both drawn), max_m (largest
   distance), kind (points whose kind differs), nan (drawn in one, not the other)}. */
export function compareDrawings(a, b) {
  let points = 0, max = 0, kind = 0, nan = 0;
  for (let i = 0; i < a.kind.length; i++) {
    if (a.kind[i] !== b.kind[i]) kind++;
    const x = a.X[3 * i], y = b.X[3 * i];
    if ((x === x) !== (y === y)) { nan++; continue; }
    if (x !== x) continue;
    points++;
    const d = Math.hypot(a.X[3 * i] - b.X[3 * i], a.X[3 * i + 1] - b.X[3 * i + 1], a.X[3 * i + 2] - b.X[3 * i + 2]);
    if (d > max) max = d;
  }
  return { points, max_m: max, kind, nan };
}
