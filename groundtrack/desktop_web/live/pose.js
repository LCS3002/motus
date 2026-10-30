/* Live view: where poses come from, and the link to the phone.

   - PoseLink: the view's client of the live hub, ws://127.0.0.1:8767/ws/view (live_hub.py). It
     receives pose (v1), zero, control (play, pause, seek, clip, record, section, gain, scale,
     space: the clutch held / let go, level), each with the phone's "player" and "spot" (null with
     one clip), and {"type": "hub", "phones", "views", "players"}; it sends the view's state
     whenever something in it changes (the time at most 4 times a second); it reconnects by
     itself. Only poses and UI state cross it: never video, never tracks.
   - MouseController: the dev pose source (src "mouse"). Right-drag pans / tilts, Alt-drag rolls,
     WASD / QE move while the right button is held. It emits pose v1 messages into the same
     handler as the phone's.
   - PoseInput: the latest pose per source, the zero per source, gain and scale -> the virtual
     camera (projection.poseToCamera), through the space clutch (projection.Clutch, plans/
     live_controller/08 a): held, the phone turns the whole space about the start camera's
     centre; let go, it steers the camera inside the turned space. Smooth (per camera, off by
     default): the virtual camera eases after the mapped one (SMOOTH_S). */

import { poseToCamera, startCamera, Clutch, turnCamera, quatToMatrix } from "./projection.js";

/** The gain's range (the phone's slider, the panel's, the G keys) and the G keys' steps. */
export const GAIN_MIN = 0.1, GAIN_MAX = 3;
export const GAINS = [0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3];

/* Smooth (08, 29 Oct (4), the user): one fixed strength. Exponential smoothing of the virtual camera
   towards the mapped one, time constant SMOOTH_S: rotation by slerp, position by lerp, each step
   1 - exp(-dt / SMOOTH_S) (frame-rate independent). A steady turn trails by SMOOTH_S (0.1 s); a
   sudden move is 63 % there after 0.1 s, 95 % after 0.3 s; a 10 Hz tremor is cut about 6x. */
export const SMOOTH_S = 0.1;
const SMOOTH_EPS = 1e-6;          // rad, m: this close, the camera is the mapped one again (exactly)

/** The angle (rad) between two unit quaternions [w, x, y, z] (either sign). */
export function quatAngle(a, b) {
  const s = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3] < 0 ? -1 : 1;
  const d = Math.hypot(a[0] - s * b[0], a[1] - s * b[1], a[2] - s * b[2], a[3] - s * b[3]);
  return 4 * Math.asin(Math.min(1, d / 2));
}

/** Slerp from unit quaternion a towards b by u (0..1, the short way) into o (may be a). */
export function slerp(a, b, u, o = new Array(4)) {
  let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3], s = 1;
  if (d < 0) { d = -d; s = -1; }
  let wa = 1 - u, wb = u;
  if (u >= 1) { wa = 0; wb = 1; }
  else if (d < 1 - 1e-10) {
    const th = Math.acos(Math.min(1, d)), sn = Math.sin(th);
    wa = Math.sin((1 - u) * th) / sn; wb = Math.sin(u * th) / sn;
  }
  wb *= s;
  const w = wa * a[0] + wb * b[0], x = wa * a[1] + wb * b[1], y = wa * a[2] + wb * b[2], z = wa * a[3] + wb * b[3];
  const n = Math.hypot(w, x, y, z);
  o[0] = w / n; o[1] = x / n; o[2] = y / n; o[3] = z / n;
  return o;
}

const STATE_KEYS = ["clip", "clips", "playing", "time", "duration", "recording", "gain", "scale", "src", "status", "section",
                    "space_held", "space_tilted", "spots", "groups", "start_spot", "site"];
const TIME_EVERY_MS = 250;

// ---------------------------------------------------------------------------- the hub link
export class PoseLink {
  constructor({ url, onPose, onZero, onControl, onHub, onStatus, WebSocketImpl } = {}) {
    this.url = url;
    this.handlers = { onPose, onZero, onControl, onHub, onStatus };
    this.WS = WebSocketImpl || globalThis.WebSocket;
    this.state = {};
    this.status = "off";
    this.retry = 500;
    this.lastSent = 0;
    this.timer = 0;
    this.closed = false;
    this.counts = { pose: 0, zero: 0, control: 0, sent: 0 };
    if (url) this.connect();
  }

  _status(s) { this.status = s; this.handlers.onStatus && this.handlers.onStatus(s); }

  connect() {
    if (this.closed || !this.url) return;
    let ws;
    try { ws = new this.WS(this.url); } catch { this._again(); return; }
    this.ws = ws;
    this._status("connecting");
    ws.onopen = () => { this.retry = 500; this._status("connected"); this._send(); };
    ws.onmessage = (e) => this._receive(e.data);
    ws.onclose = () => { if (this.ws === ws) { this.ws = null; this._status("off"); this._again(); } };
    ws.onerror = () => {};
  }

  _again() {
    if (this.closed) return;
    clearTimeout(this.reconnect);
    this.reconnect = setTimeout(() => this.connect(), this.retry);
    this.retry = Math.min(5000, this.retry * 2);
  }

  setUrl(url) {
    if (url === this.url && this.ws) return;
    this.url = url;
    if (this.ws) { const w = this.ws; this.ws = null; w.close(); }
    this.connect();
  }

  close() {
    this.closed = true;
    clearTimeout(this.reconnect); clearTimeout(this.timer);
    if (this.ws) { const w = this.ws; this.ws = null; w.close(); }
  }

  _receive(raw) {
    let m;
    try { m = JSON.parse(raw); } catch { return; }
    if (!m || typeof m !== "object") return;
    const h = this.handlers;
    if (m.type === "pose") { this.counts.pose++; h.onPose && h.onPose(m); }
    else if (m.type === "zero") { this.counts.zero++; h.onZero && h.onZero(m); }
    else if (m.type === "control") {
      this.counts.control++;
      h.onControl && h.onControl(m.action, m.value, m.player ?? null, m.spot ?? null);
    }
    else if (m.type === "hub") h.onHub && h.onHub(m);
  }

  /* Merge st into the state and send it when something changed: at once, or, when only the time
     moved, at most every 250 ms. */
  sendState(st) {
    let changed = false, timeOnly = true;
    for (const k of STATE_KEYS) {
      if (!(k in st)) continue;
      const a = this.state[k], b = st[k];
      const same = b !== null && typeof b === "object" ? JSON.stringify(a) === JSON.stringify(b) : a === b;
      if (!same) {
        changed = true;
        if (k !== "time") timeOnly = false;
        this.state[k] = b !== null && typeof b === "object" ? JSON.parse(JSON.stringify(b)) : b;
      }
    }
    if (!changed) return false;
    const wait = TIME_EVERY_MS - (performance.now() - this.lastSent);
    if (timeOnly && wait > 0) {
      if (!this.timer) this.timer = setTimeout(() => { this.timer = 0; this._send(); }, wait);
      return false;
    }
    this._send();
    return true;
  }

  /** Another message for the hub (not the state), e.g. {type: "release", spot} (Remove camera, 08 b).
      Whether it went (connected). */
  sendMessage(m) {
    if (!this.ws || this.ws.readyState !== 1) return false;
    this.ws.send(JSON.stringify(m));
    return true;
  }

  _send() {
    clearTimeout(this.timer); this.timer = 0;
    if (!this.ws || this.ws.readyState !== 1) return;
    this.lastSent = performance.now();
    this.counts.sent++;
    this.ws.send(JSON.stringify({ type: "state", ...this.state }));
  }
}

// ---------------------------------------------------------------------------- the mouse source
/** Rotation matrix (row-major, 9) -> unit quaternion [w, x, y, z]. */
export function matToQuat(R) {
  const tr = R[0] + R[4] + R[8];
  let w, x, y, z;
  if (tr > 0) {
    const s = 2 * Math.sqrt(tr + 1);
    w = s / 4; x = (R[7] - R[5]) / s; y = (R[2] - R[6]) / s; z = (R[3] - R[1]) / s;
  } else if (R[0] > R[4] && R[0] > R[8]) {
    const s = 2 * Math.sqrt(1 + R[0] - R[4] - R[8]);
    w = (R[7] - R[5]) / s; x = s / 4; y = (R[1] + R[3]) / s; z = (R[2] + R[6]) / s;
  } else if (R[4] > R[8]) {
    const s = 2 * Math.sqrt(1 + R[4] - R[0] - R[8]);
    w = (R[2] - R[6]) / s; x = (R[1] + R[3]) / s; y = s / 4; z = (R[5] + R[7]) / s;
  } else {
    const s = 2 * Math.sqrt(1 + R[8] - R[0] - R[4]);
    w = (R[3] - R[1]) / s; x = (R[2] + R[6]) / s; y = (R[5] + R[7]) / s; z = s / 4;
  }
  const n = Math.hypot(w, x, y, z);
  return [w / n, x / n, y / n, z / n];
}

function mul3(a, b) {
  const o = new Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++)
    o[3 * i + j] = a[3 * i] * b[j] + a[3 * i + 1] * b[3 + j] + a[3 * i + 2] * b[6 + j];
  return o;
}

/* A camera head in a gravity frame (z up, heading arbitrary): pan about z, then tilt about the
   camera's right, roll about its forward. world_from_camera, OpenCV axes:
   R = Rz(-pan) Rx(tilt) Ry(roll) B, B = a level camera looking along +y. Angles in radians;
   pan > 0 turns right, tilt > 0 looks up, roll > 0 is clockwise seen from behind. */
export function headMatrix(pan, tilt, roll) {
  const B = [1, 0, 0, 0, 0, 1, 0, -1, 0];
  const cp = Math.cos(pan), sp = Math.sin(pan), ct = Math.cos(tilt), st = Math.sin(tilt);
  const cr = Math.cos(roll), sr = Math.sin(roll);
  const Rz = [cp, sp, 0, -sp, cp, 0, 0, 0, 1];
  const Rx = [1, 0, 0, 0, ct, -st, 0, st, ct];
  const Ry = [cr, 0, sr, 0, 1, 0, -sr, 0, cr];
  return mul3(Rz, mul3(Rx, mul3(Ry, B)));
}

export class MouseController {
  constructor(dom, emit, { degPerPx = 0.15, speed = 1.5, onDrag } = {}) {
    this.dom = dom;
    this.emit = emit;
    this.k = degPerPx * Math.PI / 180;
    this.speed = speed;               // m/s for WASD / QE
    this.onDrag = onDrag;
    this.pan = 0; this.tilt = 0; this.roll = 0;
    this.p = [0, 0, 0];
    this.seq = 0;
    this.drag = null;
    this.keys = new Set();
    this.lastEmit = -1e9;
    const down = (e) => {
      const roll = e.button === 0 && e.altKey, look = e.button === 2;
      if (!roll && !look) return;
      e.preventDefault(); e.stopPropagation();
      this.drag = { kind: roll ? "roll" : "look", x: e.clientX, y: e.clientY, id: e.pointerId };
      try { dom.setPointerCapture(e.pointerId); } catch {}
      this.onDrag && this.onDrag(true);
    };
    const move = (e) => {
      const d = this.drag;
      if (!d || e.pointerId !== d.id) return;
      const dx = e.clientX - d.x, dy = e.clientY - d.y;
      d.x = e.clientX; d.y = e.clientY;
      if (d.kind === "roll") this.roll += dx * this.k;
      else { this.pan += dx * this.k; this.tilt -= dy * this.k; }
      this.tilt = Math.max(-1.5, Math.min(1.5, this.tilt));
      this.send();
    };
    const up = (e) => {
      if (!this.drag || e.pointerId !== this.drag.id) return;
      this.drag = null;
      this.keys.clear();
      try { dom.releasePointerCapture(e.pointerId); } catch {}
      this.onDrag && this.onDrag(false);
    };
    dom.addEventListener("pointerdown", down, true);       // capture: before the orbit controls
    window.addEventListener("pointermove", move, true);
    window.addEventListener("pointerup", up, true);
    window.addEventListener("pointercancel", up, true);
    dom.addEventListener("contextmenu", (e) => e.preventDefault());
    // WASD / QE move only while the right button is held (S alone is the section)
    window.addEventListener("keydown", (e) => {
      if (!this.drag || this.drag.kind !== "look") return;
      const k = e.key.toLowerCase();
      if ("wasdqe".includes(k) && k.length === 1) { this.keys.add(k); e.preventDefault(); e.stopPropagation(); }
    }, true);
    window.addEventListener("keyup", (e) => this.keys.delete(e.key.toLowerCase()), true);
  }

  get active() { return !!this.drag; }

  /** Move with the held keys (call each frame). */
  tick(dt) {
    if (!this.keys.size || !this.drag) return;
    const s = this.speed * Math.min(dt, 0.1), c = Math.cos(this.pan), n = Math.sin(this.pan);
    let f = 0, r = 0, u = 0;
    for (const k of this.keys) {
      if (k === "w") f += s; else if (k === "s") f -= s; else if (k === "d") r += s; else if (k === "a") r -= s;
      else if (k === "e") u += s; else if (k === "q") u -= s;
    }
    // forward along the heading (pan), right square to it, z up
    this.p[0] += n * f + c * r; this.p[1] += c * f - n * r; this.p[2] += u;
    this.send();
  }

  /** Set the head's angles (degrees) and position (m), e.g. from the URL or a test. */
  set(panDeg = 0, tiltDeg = 0, rollDeg = 0, p = null) {
    const r = Math.PI / 180;
    this.pan = panDeg * r; this.tilt = tiltDeg * r; this.roll = rollDeg * r;
    if (p) this.p = p.slice(0, 3);
    this.send();
  }

  reset() { this.set(0, 0, 0, [0, 0, 0]); }

  pose() {
    return { type: "pose", v: 1, src: "mouse", seq: ++this.seq, t: performance.now() / 1000,
             q: matToQuat(headMatrix(this.pan, this.tilt, this.roll)), p: this.p.slice(), status: "ok" };
  }

  send() {
    this.lastEmit = performance.now();
    this.emit(this.pose());
  }
}

// ---------------------------------------------------------------------------- pose -> camera
export class PoseInput {
  constructor(sc) {
    this.sc = sc;
    this.latest = {};            // src -> last pose message
    this.zeros = {};             // src -> the pose at zero
    this.src = null;             // the source driving the view
    this.gain = 1;
    this.scale = 1;
    this.version = 0;            // bumps on every change of the camera's inputs (or the space's)
    this.raw = startCamera(sc);  // the source's camera (poseToCamera): its position is the camera's
    this.cam = startCamera(sc);  // the virtual camera, in the space's own frame (the clutch's turn)
    this.D = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);   // Δ: the source's turn since zero, gain applied
    this.clutch = new Clutch();
    this.holders = new Set();    // who holds the clutch ("key", "phone 2", ...): held while any does
    this.rates = {};             // src -> {n, t0, hz}
    this.mouseHold = 0;          // while the mouse is used, the phone waits (ms timestamp)
    this.smooth = false;         // Smooth: this.cam eases after the mapped camera (this.target)
    this.target = startCamera(sc);
    this.sm = { q: [1, 0, 0, 0], C: new Float64Array(3), t: null, tick: null, moving: false };
  }

  /* A pose v1 message from any source. A source's first pose is its zero (the view starts at
     the start camera). The mouse takes over while it is dragged; the phone otherwise. */
  pose(m) {
    if (!m || m.type !== "pose" || !Array.isArray(m.q)) return;
    const s = m.src || "web";
    this.latest[s] = m;
    if (!this.zeros[s]) this.zeros[s] = m;
    const r = this.rates[s] || (this.rates[s] = { n: 0, t0: performance.now(), hz: 0 });
    r.n++;
    r.last = performance.now();
    const dt = performance.now() - r.t0;
    if (dt > 1000) { r.hz = r.n * 1000 / dt; r.n = 0; r.t0 = performance.now(); }
    if (s === "mouse") this.mouseHold = performance.now() + 600;
    else {
      this.lastPhone = s;
      if (this.src === "mouse" && performance.now() < this.mouseHold) return;
    }
    this.src = s;
    this.version++;
  }

  /* Zero: the source's current pose becomes the start camera (src: that source, else the one
     driving). Only the camera: a turned space stays turned. */
  zero(src = null) {
    const s = src || this.src;
    if (s && this.latest[s]) this.zeros[s] = this.latest[s];
    this.clutch.zero();
    this.version++;
  }

  // ---- the space clutch (08 a): press / let go per holder, level (a double-tap)
  /** who presses the clutch (the first holder presses it; t: the clock, seconds). */
  press(who, t = performance.now() / 1000) {
    const was = this.holders.size;
    this.holders.add(who);
    if (was || this.clutch.held) return false;
    this._delta();
    this.clutch.move(this.D, t);              // a glide in progress goes on from where it is
    this.clutch.press(this.D);
    this.version++;
    return true;
  }

  /** who lets go (null: everyone, e.g. the hub link dropped); the last holder releases the clutch. */
  release(who = null, t = performance.now() / 1000) {
    if (who === null) this.holders.clear(); else this.holders.delete(who);
    if (this.holders.size || !this.clutch.held) return false;
    this._delta();
    this.clutch.move(this.D, t);
    this.clutch.release(this.D);
    this.version++;
    return true;
  }

  /** Every holder whose name passes test lets go (the phones, when the hub link drops). */
  releaseWhere(test, t = performance.now() / 1000) {
    for (const w of [...this.holders]) if (test(w)) this.holders.delete(w);
    return this.holders.size ? false : this.release(null, t);
  }

  /** The space glides back to level (about 0.3 s, driven by camera(t) / tick(t)). */
  level(t = performance.now() / 1000) {
    this._delta();
    this.clutch.move(this.D, t);
    this.clutch.level(t);
    this.version++;
  }

  /* Each frame (t: the clock, seconds): a glide in progress moves the space (bumps the version);
     with Smooth, an easing camera too (at rest, the next move counts from the frame before). */
  tick(t) {
    if (this.clutch.glide) this.version++;
    if (!this.smooth) return;
    const m = this.sm;
    if (m.moving) this.version++;
    else if (t !== undefined) m.t = m.tick ?? t;
    m.tick = t;
  }

  /** Smooth on / off (t: the clock, seconds). On: from where the camera is (no jump); off: the mapped
      camera again. Whether it changed. */
  setSmooth(on, t = performance.now() / 1000) {
    on = !!on;
    if (on === this.smooth) return false;
    this.smooth = on;
    if (on) {
      const m = this.sm;
      m.q = matToQuat(this.cam.R_wc); m.C.set(this.cam.C);
      m.t = t; m.tick = null; m.moving = true;
    }
    this.version++;
    return true;
  }

  get held() { return this.clutch.held; }
  get tilted() { return this.clutch.tilted; }
  get space() { return this.clutch.S; }
  get levelling() { return this.clutch.glide !== null; }

  /** The newest non-mouse source (a phone's zero applies to it). */
  get phoneSrc() { return this.lastPhone || null; }

  setGain(g) { this.gain = Math.max(GAIN_MIN, Math.min(GAIN_MAX, +g || 1)); this.version++; }
  setScale(s) { this.scale = Math.max(0.01, Math.min(100, +s || 1)); this.version++; }

  get status() { const m = this.src && this.latest[this.src]; return m ? m.status || "ok" : "ok"; }

  _delta() {                    // the driving source's camera and its turn since zero (this.raw, this.D)
    const m = this.src && this.latest[this.src];
    poseToCamera(this.sc, m || null, m ? this.zeros[this.src] : null, this.gain, this.scale, { out: this.raw, delta: this.D });
  }

  /* The virtual camera in the space's own frame: the start camera turned by the clutch's Dc (Δ while
     let go, frozen while held), at the source's position. Moves the clutch on (t: the clock,
     seconds, for a glide). The space's turn is this.space (S): shown at p -> C0 + S (p - C0).
     With Smooth, that is this.target and the camera eases after it (_ease). */
  camera(t = performance.now() / 1000) {
    this._delta();
    this.clutch.move(this.D, t);
    if (!this.smooth) {
      this.cam.C.set(this.raw.C);
      return turnCamera(this.sc, this.clutch.Dc, this.cam);
    }
    this.target.C.set(this.raw.C);
    turnCamera(this.sc, this.clutch.Dc, this.target);
    return this._ease(t);
  }

  /* One smoothing step to t: rotation slerped, position lerped by 1 - exp(-dt / SMOOTH_S); this
     close (SMOOTH_EPS), the mapped camera exactly. R_cw is R_wcᵀ exactly (as a replay rebuilds it). */
  _ease(t) {
    const m = this.sm, T = this.target, cam = this.cam;
    const dt = m.t === null ? Infinity : Math.max(0, t - m.t);
    m.t = t;
    const a = 1 - Math.exp(-dt / SMOOTH_S), qt = matToQuat(T.R_wc);
    slerp(m.q, qt, a, m.q);
    for (let i = 0; i < 3; i++) m.C[i] += a * (T.C[i] - m.C[i]);
    if (quatAngle(m.q, qt) < SMOOTH_EPS && Math.hypot(m.C[0] - T.C[0], m.C[1] - T.C[1], m.C[2] - T.C[2]) < SMOOTH_EPS) {
      cam.R_wc.set(T.R_wc); cam.R_cw.set(T.R_cw); cam.C.set(T.C);
      m.q = qt; m.C.set(T.C); m.moving = false;
      return cam;
    }
    const R = quatToMatrix(m.q, cam.R_wc), o = cam.R_cw;
    o[0] = R[0]; o[1] = R[3]; o[2] = R[6]; o[3] = R[1]; o[4] = R[4]; o[5] = R[7]; o[6] = R[2]; o[7] = R[5]; o[8] = R[8];
    cam.C.set(m.C);
    m.moving = true;
    return cam;
  }
}

/* The turn of the virtual camera against the start camera, as the HUD shows it (degrees):
   pan (heading change, + right), tilt (+ up), roll (+ clockwise); and the move (m). */
export function cameraDelta(sc, cam) {
  const ang = (R) => {
    const fx = R[2], fy = R[5], fz = R[8];               // forward = R_wc column 3
    const yaw = Math.atan2(fx, fy), tilt = Math.asin(Math.max(-1, Math.min(1, fz)));
    const r0 = [Math.cos(yaw), -Math.sin(yaw), 0];       // level right at this heading
    const d0 = [fy * r0[2] - fz * r0[1], fz * r0[0] - fx * r0[2], fx * r0[1] - fy * r0[0]];   // fwd x r0
    const right = [R[0], R[3], R[6]];
    const roll = Math.atan2(right[0] * d0[0] + right[1] * d0[1] + right[2] * d0[2],
                            right[0] * r0[0] + right[1] * r0[1] + right[2] * r0[2]);
    return [yaw, tilt, roll];
  };
  const a = ang(sc.R_wc), b = ang(cam.R_wc), D = 180 / Math.PI;
  let pan = (b[0] - a[0]) * D;
  pan = ((pan + 540) % 360) - 180;
  return { pan, tilt: (b[1] - a[1]) * D, roll: (b[2] - a[2]) * D,
           move: [cam.C[0] - sc.C[0], cam.C[1] - sc.C[1], cam.C[2] - sc.C[2]] };
}
