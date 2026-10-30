/* Live view: the pass records itself (plans/live_controller/05, 29 Oct: a session is a pass). Pure
   helpers (no DOM, no three.js). The format is groundtrack/live_passes.py's (recording v1):
   - clock.rec, 24 bytes a row, one per recorded frame k: t f64 (s since the pass began), T f64
     (the shared clock), rate f32, flags u8 (1 playing, 2 at the end);
   - <camera>.rec, 232 bytes a row, one per recorded frame the camera changed in: R_wc f64 x 9, C
     f64 x 3, S f64 x 9, k u32, frame i32, q f32 x 4, p f32 x 3, gain f32, scale f32, placed,
     guessed, dropped, lost, new u16, (u16), player u8, src u8, status u8, flags u8, seq u32;
   - events.jsonl: {k, t, T, type, player, camera, value} a line.
   A frame is recorded when something changed (a pose, the clock, a camera's frame or counts, an
   event); between two rows everything held. */

import { DRAWN } from "./projection.js";
import { SOLID, DROPPED, NOT_YET } from "./rules.js";

export const CLOCK_BYTES = 24, CAM_BYTES = 232;
export const SOURCES = ["none", "web", "mouse", "replay", "bench", "webcam", "freed", "arkit", "url"];
export const STATUSES = ["ok", "limited", "lost"];
export const PLAYING = 1, AT_END = 2;
export const CHANGED = 1, HELD = 2, LEVELLING = 4, TILTED = 8, PERFORMED = 16, JOINED = 32;
export const SMOOTH = 64;             // the camera's Smooth on (its R_wc, C: the eased camera)
const W64 = CAM_BYTES / 8, W32 = CAM_BYTES / 4, W16 = CAM_BYTES / 2;   // a row in 8 / 4 / 2-byte words

/** A growing buffer of fixed-size rows. */
class Rows {
  constructor(bytes, cap = 256) { this.bytes = bytes; this.n = 0; this._alloc(cap); }
  _alloc(cap) {
    const old = this.u8;
    this.buf = new ArrayBuffer(cap * this.bytes);
    this.u8 = new Uint8Array(this.buf);
    if (old) this.u8.set(old.subarray(0, this.n * this.bytes));
    this.f64 = new Float64Array(this.buf); this.f32 = new Float32Array(this.buf);
    this.u32 = new Uint32Array(this.buf); this.i32 = new Int32Array(this.buf); this.u16 = new Uint16Array(this.buf);
    this.cap = cap;
  }
  /** The next row's index (room made). */
  push() { if (this.n === this.cap) this._alloc(this.cap * 2); return this.n++; }
  /** The rows so far, a copy. */
  bytesCopy() { return this.buf.slice(0, this.n * this.bytes); }
}

const srcCode = (s) => { const i = SOURCES.indexOf(s || "none"); return i < 0 ? 255 : i; };
const statusCode = (s) => Math.max(0, STATUSES.indexOf(s || "ok"));

/** The previous point of each point's track (-1: its first), cached on the tracks. */
export function prevIndex(d) {
  if (d._prev) return d._prev;
  const { order, starts } = d.index, prev = new Int32Array(d.n).fill(-1);
  for (let k = 0; k + 1 < starts.length; k++)
    for (let r = starts[k] + 1; r < starts[k + 1]; r++) prev[order[r]] = order[r - 1];
  d._prev = prev;
  return prev;
}

/* The machine's counts at frame f (out: the rules' output shown, state: the projection's states;
   into c, Uint16Array(5) or any array): placed (solid), guessed (white dashes), dropped (nothing
   drawn), lost (points the projection lost: behind, out of view, past the horizon), new (identity
   pieces beginning here: a track's first drawn point, or a new identity). */
export function machineCounts(d, out, state, f, c = new Uint16Array(5)) {
  c[0] = c[1] = c[2] = c[3] = c[4] = 0;
  if (!out || f < 0 || f >= d.frames) return c;
  const fs = d.frameStart, kind = out.kind, part = out.part, src = d.src, prev = prevIndex(d);
  for (let i = fs[f]; i < fs[f + 1]; i++) {
    const k = kind[i];
    if (state && state[i] !== DRAWN && state[i] !== NOT_YET) c[3]++;
    if (k === NOT_YET) continue;
    if (k === DROPPED) { c[2]++; continue; }
    if (k !== SOLID || src[i]) c[1]++; else c[0]++;
    const p = prev[i];
    if (p < 0 || kind[p] >= DROPPED || part[p] !== part[i]) c[4]++;
  }
  return c;
}

export class Recorder {
  constructor() { this.begin(0); }

  /** A new pass: nothing recorded yet; t0 the view's clock (ms) at its start. */
  begin(t0, started = "") {
    this.t0 = t0;
    this.started = started;
    this.clock = new Rows(CLOCK_BYTES, 1024);
    this.cams = new Map();             // name -> {rows: Rows, last: Uint8Array (the last row), info, players: Set}
    this.events = [];
    this.pending = false;              // an event since the last clock row: this frame is recorded
    this.lastClock = null;             // [T, rate, flags]
    this.k = 0;                        // the clock row of this frame (written at commit when needed)
    this.dirty = false;
    this.now = t0; this.T = 0;
    this.scratch = new Rows(CAM_BYTES, 1);
    this.scratch.push();
  }

  get frames() { return this.clock.n; }
  get seconds() { return this.clock.n ? this.clock.f64[(this.clock.n - 1) * 3] : 0; }

  /** A camera takes part (its package and place, kept for the manifest). */
  camera(name, info) {
    let c = this.cams.get(name);
    if (!c) { c = { rows: new Rows(CAM_BYTES, 1024), last: null, info, players: new Set() }; this.cams.set(name, c); }
    else if (info) c.info = info;
    return c;
  }

  /* Something happened (before this frame's rows): {type, player, camera, value}; at now (ms, the
     view's clock) and T (the shared clock), by default this.clockFn() ([now, T]) or the frame's. */
  event(type, { player = null, camera = null, value = null } = {}, now = null, T = null) {
    if (now === null) [now, T] = this.clockFn ? this.clockFn() : [this.now, this.T];
    this.events.push({ k: this.clock.n, t: Math.round((now - this.t0) * 1e3) / 1e6, T: Math.round(T * 1e4) / 1e4, type,
                       player, camera, value });
    this.pending = true;
  }

  /** The frame starts: the view's clock (ms) and the shared clock. */
  frame(now, T) { this.now = now; this.T = T; this.k = this.clock.n; this.dirty = false; }

  /* A camera's state this frame: written when it differs from its last row. r: {cam (R_wc, C), S,
     frame, pose (a pose message or null), gain, scale, counts (5), player, src, status, flags}. */
  row(name, r) {
    const c = this.camera(name), s = this.scratch, o = 0;
    s.f64.set(r.cam.R_wc, o); s.f64.set(r.cam.C, o + 9); s.f64.set(r.S, o + 12);
    s.u32[42] = this.k; s.i32[43] = r.frame | 0;
    const m = r.pose;
    if (m && Array.isArray(m.q)) { s.f32[44] = m.q[0]; s.f32[45] = m.q[1]; s.f32[46] = m.q[2]; s.f32[47] = m.q[3]; }
    else s.f32.fill(NaN, 44, 48);
    if (m && Array.isArray(m.p)) { s.f32[48] = m.p[0]; s.f32[49] = m.p[1]; s.f32[50] = m.p[2]; }
    else s.f32.fill(NaN, 48, 51);
    s.f32[51] = r.gain; s.f32[52] = r.scale;
    s.u16.set(r.counts, 106); s.u16[111] = 0;
    s.u8[224] = r.player || 0; s.u8[225] = srcCode(r.src); s.u8[226] = statusCode(r.status); s.u8[227] = r.flags;
    s.u32[57] = m && m.seq >>> 0 || 0;
    if (c.last && (r.flags & JOINED) === 0) {        // the same as its last row (but k): nothing to write
      let same = true;
      for (let i = 0; i < W32 && same; i++) if (i !== 42 && s.u32[i] !== c.last[i]) same = false;
      if (same) return false;
    }
    const i = c.rows.push();
    c.rows.u8.set(s.u8, i * CAM_BYTES);
    if (!c.last) c.last = new Uint32Array(W32);
    c.last.set(s.u32);
    if (r.player) c.players.add(r.player);
    this.dirty = true;
    return true;
  }

  /** The frame ends: its clock row is written when anything changed. */
  commit(rate, flags) {
    const lc = this.lastClock;
    if (!this.dirty && !this.pending && lc && lc[0] === this.T && lc[1] === rate && lc[2] === flags) return false;
    const i = this.clock.push();
    this.clock.f64[3 * i] = (this.now - this.t0) / 1000;
    this.clock.f64[3 * i + 1] = this.T;
    this.clock.f32[6 * i + 4] = rate;
    this.clock.u8[24 * i + 20] = flags;
    this.lastClock = [this.T, rate, flags];
    this.pending = false;
    return true;
  }

  /* The recording so far, copied (the pass is saved while the next one records): {meta (pass.json's
     "recording", files named by fileOf(name)), files: {name: ArrayBuffer | string}}. */
  snapshot(fileOf) {
    const files = {}, cameras = [];
    files["clock.rec"] = this.clock.bytesCopy();
    const used = new Set(["clock.rec"]);
    for (const [name, c] of this.cams) {
      if (!c.rows.n) continue;
      let file = fileOf(name, c.info), k = 2;
      while (used.has(file)) file = file.replace(/(_\d+)?\.rec$/, `_${k++}.rec`);
      used.add(file);
      files[file] = c.rows.bytesCopy();
      cameras.push({ ...(c.info || {}), name, file, rows: c.rows.n, players: [...c.players] });
    }
    const text = this.events.map((e) => JSON.stringify(e)).join("\n") + (this.events.length ? "\n" : "");
    files["events.jsonl"] = text;
    return { files, meta: { started: this.started, seconds: Math.round(this.seconds * 1000) / 1000,
                            clock: { file: "clock.rec", rows: this.clock.n }, events: { file: "events.jsonl", n: this.events.length },
                            cameras } };
  }
}

// ---------------------------------------------------------------------------- reading it back
/* A saved recording (the files' bytes) -> {clock: {t, T, rate, flags, n}, cams: Map(name -> rows),
   events}; rows: {n, R (r -> Float64Array(9) R_wc), C, S, k, frame, flags, view(r) (all fields)}. */
export function readRecording(meta, clockBuf, camBufs, eventsText) {
  const cn = meta.clock.rows;
  if (clockBuf.byteLength !== cn * CLOCK_BYTES) throw new Error("clock.rec: its size");
  const f64 = new Float64Array(clockBuf), f32 = new Float32Array(clockBuf), u8 = new Uint8Array(clockBuf);
  const clock = { n: cn, t: new Float64Array(cn), T: new Float64Array(cn), rate: new Float32Array(cn), flags: new Uint8Array(cn) };
  for (let i = 0; i < cn; i++) { clock.t[i] = f64[3 * i]; clock.T[i] = f64[3 * i + 1]; clock.rate[i] = f32[6 * i + 4]; clock.flags[i] = u8[24 * i + 20]; }
  const cams = new Map();
  for (const c of meta.cameras || []) {
    const buf = camBufs[c.file];
    if (!buf || buf.byteLength !== c.rows * CAM_BYTES) throw new Error(`${c.file}: its size`);
    cams.set(c.name, camRows(buf, c));
  }
  const events = (eventsText || "").split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l));
  return { clock, cams, events, meta };
}

export function camRows(buf, info = {}) {
  const n = buf.byteLength / CAM_BYTES, f64 = new Float64Array(buf), f32 = new Float32Array(buf);
  const u32 = new Uint32Array(buf), i32 = new Int32Array(buf), u16 = new Uint16Array(buf), u8 = new Uint8Array(buf);
  return {
    n, info,
    R: (r) => f64.subarray(W64 * r, W64 * r + 9),
    C: (r) => f64.subarray(W64 * r + 9, W64 * r + 12),
    S: (r) => f64.subarray(W64 * r + 12, W64 * r + 21),
    k: (r) => u32[W32 * r + 42],
    frame: (r) => i32[W32 * r + 43],
    flags: (r) => u8[CAM_BYTES * r + 227],
    view: (r) => ({ k: u32[W32 * r + 42], frame: i32[W32 * r + 43], q: Array.from(f32.subarray(W32 * r + 44, W32 * r + 48)),
                    p: Array.from(f32.subarray(W32 * r + 48, W32 * r + 51)), gain: f32[W32 * r + 51], scale: f32[W32 * r + 52],
                    counts: Array.from(u16.subarray(W16 * r + 106, W16 * r + 111)), player: u8[CAM_BYTES * r + 224],
                    src: SOURCES[u8[CAM_BYTES * r + 225]] || "other", status: STATUSES[u8[CAM_BYTES * r + 226]] || "ok",
                    flags: u8[CAM_BYTES * r + 227], seq: u32[W32 * r + 57] }),
  };
}
