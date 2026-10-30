/* Live controller: the projection core spread over CPU cores (module Web Workers).
   One file, two roles:
   - started as a worker (new Worker(url, {type: "module"})), it wraps projection.js: it takes the
     scene once, then the points, then projects index ranges on request;
   - imported on the page, it is the pool that drives those workers:

       const pool = createProjectionPool(scene, 6);       // scene from prepareScene()
       const out = await pool.project(cam, X);            // cam from poseToCamera(); X flat xyz
       const lifted = await pool.project(cam, X, null, "hinge");    // or opts.mode for every call
       ... out.X (Float32Array, xyz, NaN when lost), out.state (Uint8Array) ...
       pool.close();

   Shared memory (SharedArrayBuffer) when the page is cross-origin isolated: the points and the
   results live in shared buffers, a frame posts only the camera. Without it the points are copied
   to each worker once and each worker's results come back as transferred buffers, copied into
   out. X is sent to the workers when a new array is passed (pool.setPoints(X) after changing one
   in place). Calls are queued: one projection at a time. */

import { prepareScene, project } from "./projection.js";

// ---------------------------------------------------------------------------- the worker
let _scene = null, _X = null, _out = null;

function onJob(d) {
  try {
    if (d.type === "scene") {
      _scene = prepareScene(d.meta, d.heights, d.opts);
    } else if (d.type === "points") {
      _X = d.X;
      _out = d.outX ? { X: d.outX, state: d.outS } : null;
    } else if (d.type === "project") {
      const cam = { R_cw: d.R_cw, C: d.C };
      if (d.outX) {                                   // transferred: this worker's range only
        project(_scene, cam, _X.subarray(3 * d.i0, 3 * d.i1), { X: d.outX, state: d.outS }, 0, d.i1 - d.i0,
                d.mode);
        self.postMessage({ type: "done", id: d.id, w: d.w, outX: d.outX, outS: d.outS },
                         [d.outX.buffer, d.outS.buffer]);
      } else {                                        // shared: straight into the pool's buffers
        project(_scene, cam, _X, _out, d.i0, d.i1, d.mode);
        self.postMessage({ type: "done", id: d.id });
      }
    }
  } catch (e) {
    self.postMessage({ type: "error", id: d.id, message: String((e && e.stack) || e) });
  }
}

if (typeof window === "undefined" && typeof self !== "undefined" && typeof self.postMessage === "function") {
  self.onmessage = (e) => onJob(e.data);
}

// ---------------------------------------------------------------------------- the pool (page side)
/** Whether results can be shared with the workers: needs a cross-origin isolated page. */
export function canShare() {
  return typeof SharedArrayBuffer !== "undefined" && globalThis.crossOriginIsolated !== false;
}

export function defaultWorkers() {
  const n = (globalThis.navigator && navigator.hardwareConcurrency) || 4;
  return Math.max(1, Math.min(8, n - 1));
}

/* scene: from prepareScene(); n workers. opts.shared: force shared memory on / off (default: when
   possible); opts.out64: results as Float64Array (default Float32Array); opts.mode: "ground"
   (default) or "hinge", as projection.js's project(); opts.spawn(i): makes a
   worker ({postMessage, onmessage, terminate}; default a module Worker of this file). */
export function createProjectionPool(scene, n = defaultWorkers(), opts = {}) {
  return new ProjectionPool(scene, n, opts);
}

export class ProjectionPool {
  constructor(scene, n, opts = {}) {
    this.n = Math.max(1, n | 0);
    this.shared = opts.shared ?? canShare();
    this.Out = opts.out64 ? Float64Array : Float32Array;
    this.mode = opts.mode || "ground";
    const spawn = opts.spawn || (() => new Worker(new URL(import.meta.url), { type: "module" }));
    const heights = scene.grid ? scene.grid.z : null;
    this.workers = [];
    this._jobs = new Map();
    this._seq = 0;
    this._chain = Promise.resolve();
    this._X = null;
    this.out = null;
    for (let i = 0; i < this.n; i++) {
      const w = spawn(i);
      w.onmessage = (e) => this._reply(e.data);
      w.postMessage({ type: "scene", meta: scene.meta, heights,
                      opts: { skip: scene.skip, tile: scene.grid ? scene.grid.tile : undefined } });
      this.workers.push(w);
    }
  }

  /** Send the points (flat xyz) to the workers; project() does this itself for a new array. */
  setPoints(X) {
    const n = (X.length / 3) | 0, k = this.n;
    this.ranges = Array.from({ length: k }, (_, i) => [Math.floor((n * i) / k), Math.floor((n * (i + 1)) / k)]);
    if (this.shared) {
      const sab = (len, T) => new T(new SharedArrayBuffer(len * T.BYTES_PER_ELEMENT));
      const Xs = X.buffer instanceof SharedArrayBuffer ? X : sab(X.length, X.constructor);
      if (Xs !== X) Xs.set(X);
      this.out = { X: sab(3 * n, this.Out), state: sab(n, Uint8Array) };
      for (const w of this.workers) w.postMessage({ type: "points", X: Xs, outX: this.out.X, outS: this.out.state });
    } else {
      this.out = { X: new this.Out(3 * n), state: new Uint8Array(n) };
      this._chunks = this.ranges.map(([a, b]) => ({ X: new this.Out(3 * (b - a)), state: new Uint8Array(b - a) }));
      for (const w of this.workers) w.postMessage({ type: "points", X: X.slice() });
    }
    this._X = X;
    this._n = n;
  }

  /* Project every point of X with cam ({R_cw, C}, copied now) in mode ("ground" | "hinge", default
     the pool's): resolves to out (when given, the results are copied into it) or to the pool's own
     result arrays (pool.out, reused each call). */
  project(cam, X, out = null, mode = this.mode) {
    const R_cw = Float64Array.from(cam.R_cw), C = Float64Array.from(cam.C);
    const p = this._chain.then(() => this._run(R_cw, C, X, out, mode));
    this._chain = p.catch(() => {});
    return p;
  }

  _run(R_cw, C, X, out, mode) {
    if (X !== this._X) this.setPoints(X);
    const id = ++this._seq;
    return new Promise((resolve, reject) => {
      this._jobs.set(id, { left: this.n, resolve, reject, out });
      this.workers.forEach((w, i) => {
        const [i0, i1] = this.ranges[i];
        if (this.shared) {
          w.postMessage({ type: "project", id, R_cw, C, i0, i1, mode });
        } else {
          const ch = this._chunks[i];
          w.postMessage({ type: "project", id, R_cw, C, i0, i1, mode, outX: ch.X, outS: ch.state, w: i },
                        [ch.X.buffer, ch.state.buffer]);
        }
      });
    });
  }

  _reply(d) {
    const job = this._jobs.get(d.id);
    if (!job) {
      if (d.type === "error") console.error("projection worker:", d.message);
      return;
    }
    if (d.type === "error") {
      this._jobs.delete(d.id);
      job.reject(new Error(d.message));
      return;
    }
    if (!this.shared) {                       // a worker's range is back: keep its buffers, copy in
      this._chunks[d.w] = { X: d.outX, state: d.outS };
      const i0 = this.ranges[d.w][0];
      this.out.X.set(d.outX, 3 * i0);
      this.out.state.set(d.outS, i0);
    }
    if (--job.left > 0) return;
    this._jobs.delete(d.id);
    const res = this.out;
    if (job.out && job.out !== res) { job.out.X.set(res.X); job.out.state.set(res.state); job.resolve(job.out); }
    else job.resolve(res);
  }

  close() {
    for (const w of this.workers) w.terminate();
    this.workers = [];
  }
}
