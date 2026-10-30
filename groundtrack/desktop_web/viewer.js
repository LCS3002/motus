"use strict";
/* The drawing surfaces of the desktop app: a canvas image viewer (pan, zoom, loupe,
   overlays), and on it the calibration picker (video frame + aerial), the area (ROI)
   editor and the camera map. Coordinates are image pixels; the aerial converts to
   British National Grid metres with its extent. */

const COL = { good: "#fff", warn: "#e5cf6a", bad: "#f0606f", fg: "#fff", bg: "#000",
              under: "rgba(0,0,0,.6)", rule: "rgba(255,255,255,.22)", grid: "rgba(255,255,255,.32)",
              guide: "rgba(255,255,255,.75)", proj: "#e5cf6a" };
const MONO = "500 10.5px 'Spline Sans Mono', ui-monospace, monospace";
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

class Viewer {
  constructor(host, opts = {}) {
    this.host = host; this.opts = opts;
    this.canvas = document.createElement("canvas");
    host.prepend(this.canvas);
    this.ctx = this.canvas.getContext("2d");
    this.img = null; this.iw = 0; this.ih = 0;
    this.scale = 1; this.tx = 0; this.ty = 0; this.minScale = 0.01;
    this.mouse = null; this.loupe = opts.loupe !== false;
    this.w = 0; this.h = 0; this.dpr = window.devicePixelRatio || 1;
    this._raf = 0; this._fitted = false;
    this.ro = new ResizeObserver(() => this._resize());
    this.ro.observe(host);
    this._bind();
  }

  destroy() { this.ro.disconnect(); cancelAnimationFrame(this._raf); this.canvas.remove(); }

  _resize() {
    const r = this.host.getBoundingClientRect();
    if (!r.width || !r.height) return;
    const cx = this.w ? this.toImage(this.w / 2, this.h / 2) : null;
    this.w = r.width; this.h = r.height; this.dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(this.w * this.dpr); this.canvas.height = Math.round(this.h * this.dpr);
    if (this.img && !this._fitted) this.fit(false);
    else if (cx && this.img) { this.tx = this.w / 2 - cx[0] * this.scale; this.ty = this.h / 2 - cx[1] * this.scale; }
    this._draw();
  }

  load(url) {
    return new Promise((resolve, reject) => {
      const im = new Image();
      im.decoding = "async";
      im.onload = () => {
        const same = this.iw === im.naturalWidth && this.ih === im.naturalHeight;
        this.img = im; this.iw = im.naturalWidth; this.ih = im.naturalHeight;
        if (!same || !this._fitted) this.fit(false); else this.redraw();
        resolve(im);
      };
      im.onerror = () => reject(new Error("could not load the image"));
      im.src = url;
    });
  }

  fitScale(pad = 10) { return Math.min((this.w - 2 * pad) / this.iw, (this.h - 2 * pad) / this.ih); }

  fit(animate = true) {
    if (!this.iw || !this.w) return;
    const s = this.fitScale();
    this.minScale = s * 0.5;
    this._fitted = true;
    this.setView({ scale: s, tx: (this.w - this.iw * s) / 2, ty: (this.h - this.ih * s) / 2 }, animate);
  }

  /* show the image rectangle x0..x1, y0..y1 */
  frame(x0, y0, x1, y1, animate = true) {
    if (!this.w) return;
    const s = clamp(Math.min(this.w / Math.max(x1 - x0, 1e-6), this.h / Math.max(y1 - y0, 1e-6)), this.minScale, 80);
    const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    this.setView({ scale: s, tx: this.w / 2 - cx * s, ty: this.h / 2 - cy * s }, animate);
  }

  getView() { return { scale: this.scale, tx: this.tx, ty: this.ty }; }

  /* wheel zoom eases toward its target around the cursor instead of jumping per notch */
  zoomSmooth(sx, sy, f) {
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) { this.zoomAt(sx, sy, f); return; }
    const from = this._zt ? this._zt.scale : this.scale;
    this._zt = { scale: clamp(from * f, this.minScale, 80), sx, sy };
    if (this._zRaf) return;
    const step = () => {
      const z = this._zt;
      if (!z) { this._zRaf = 0; return; }
      const next = this.scale + (z.scale - this.scale) * 0.32;
      const done = Math.abs(z.scale - next) < z.scale * 0.002;
      const s = done ? z.scale : next, k = s / this.scale;
      this.tx = z.sx - (z.sx - this.tx) * k; this.ty = z.sy - (z.sy - this.ty) * k; this.scale = s;
      this._draw();
      if (done) { this._zt = null; this._zRaf = 0; } else this._zRaf = requestAnimationFrame(step);
    };
    this._zRaf = requestAnimationFrame(step);
  }

  setView(v, animate = true) {
    cancelAnimationFrame(this._animRaf);
    cancelAnimationFrame(this._zRaf); this._zRaf = 0; this._zt = null;
    if (!animate || matchMedia("(prefers-reduced-motion: reduce)").matches) {
      Object.assign(this, v); this.redraw(); return;
    }
    const from = this.getView(), t0 = performance.now(), dur = 280;
    // zoom in log space so it feels even
    const step = (t) => {
      const k = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - k, 3);
      const s = Math.exp(Math.log(from.scale) + (Math.log(v.scale) - Math.log(from.scale)) * e);
      // keep the target centre moving on a straight line
      const fc = [(this.w / 2 - from.tx) / from.scale, (this.h / 2 - from.ty) / from.scale];
      const tc = [(this.w / 2 - v.tx) / v.scale, (this.h / 2 - v.ty) / v.scale];
      const c = [fc[0] + (tc[0] - fc[0]) * e, fc[1] + (tc[1] - fc[1]) * e];
      this.scale = s; this.tx = this.w / 2 - c[0] * s; this.ty = this.h / 2 - c[1] * s;
      this._draw();
      if (k < 1) this._animRaf = requestAnimationFrame(step);
    };
    this._animRaf = requestAnimationFrame(step);
  }

  toScreen(x, y) { return [x * this.scale + this.tx, y * this.scale + this.ty]; }
  toImage(sx, sy) { return [(sx - this.tx) / this.scale, (sy - this.ty) / this.scale]; }
  inside(x, y) { return x >= 0 && y >= 0 && x <= this.iw && y <= this.ih; }

  zoomAt(sx, sy, f) {
    const s = clamp(this.scale * f, this.minScale, 80), k = s / this.scale;
    this.tx = sx - (sx - this.tx) * k; this.ty = sy - (sy - this.ty) * k; this.scale = s;
    this.redraw();
  }

  redraw() { if (!this._raf) this._raf = requestAnimationFrame(() => { this._raf = 0; this._draw(); }); }

  _draw() {
    const c = this.ctx, d = this.dpr;
    if (!this.w) return;
    c.setTransform(d, 0, 0, d, 0, 0);
    c.fillStyle = COL.bg; c.fillRect(0, 0, this.w, this.h);
    if (this.img) {
      c.imageSmoothingEnabled = this.scale < 3; c.imageSmoothingQuality = "high";
      c.drawImage(this.img, this.tx, this.ty, this.iw * this.scale, this.ih * this.scale);
      c.lineWidth = 1; c.strokeStyle = COL.rule;
      c.strokeRect(Math.round(this.tx) - 0.5, Math.round(this.ty) - 0.5, Math.round(this.iw * this.scale) + 1, Math.round(this.ih * this.scale) + 1);
    }
    if (this.opts.draw) { c.save(); this.opts.draw(c, this); c.restore(); }
    if (this.loupe && this.mouse && this.img && !this._panning) this._drawLoupe(c);
  }

  _drawLoupe(c) {
    const [mx, my] = this.mouse, R = 62, zoom = Math.max(3, 6 / Math.max(this.scale, 0.3));
    const [ix, iy] = this.toImage(mx, my);
    if (!this.inside(ix, iy)) return;
    let cx = mx + 86, cy = my - 86;
    if (cx + R > this.w - 6) cx = mx - 86;
    if (cy - R < 6) cy = my + 86;
    const s = this.scale * zoom, half = R / s;
    c.save();
    c.beginPath(); c.rect(cx - R, cy - R, 2 * R, 2 * R);
    c.fillStyle = COL.bg; c.fill(); c.clip();
    c.imageSmoothingEnabled = s < 4;
    c.drawImage(this.img, ix - half, iy - half, 2 * half, 2 * half, cx - R, cy - R, 2 * R, 2 * R);
    if (this.opts.drawLoupe) this.opts.drawLoupe(c, (x, y) => [cx + (x - ix) * s, cy + (y - iy) * s], s);
    c.strokeStyle = "rgba(255,255,255,.9)"; c.lineWidth = 1;
    c.beginPath(); c.moveTo(cx - R, cy); c.lineTo(cx - 5, cy); c.moveTo(cx + 5, cy); c.lineTo(cx + R, cy);
    c.moveTo(cx, cy - R); c.lineTo(cx, cy - 5); c.moveTo(cx, cy + 5); c.lineTo(cx, cy + R); c.stroke();
    c.restore();
    c.beginPath(); c.rect(cx - R - 0.5, cy - R - 0.5, 2 * R + 1, 2 * R + 1);
    c.lineWidth = 3; c.strokeStyle = COL.under; c.stroke();
    c.lineWidth = 1; c.strokeStyle = COL.fg; c.stroke();
  }

  _bind() {
    const cv = this.canvas;
    const pos = (e) => { const r = cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    cv.addEventListener("pointerdown", (e) => {
      const [sx, sy] = pos(e);
      const handle = e.button === 0 && this.opts.pick ? this.opts.pick(sx, sy) : null;
      this._down = { sx, sy, lx: sx, ly: sy, button: e.button, moved: false, handle };
      cv.setPointerCapture(e.pointerId);
      if (e.button === 1) e.preventDefault();
    });
    cv.addEventListener("pointermove", (e) => {
      const [sx, sy] = pos(e);
      this.mouse = [sx, sy];
      const d = this._down;
      if (d) {
        if (Math.hypot(sx - d.sx, sy - d.sy) > 4) d.moved = true;
        if (d.moved && d.handle != null && d.button === 0) {
          const [ix, iy] = this.toImage(sx, sy);
          this.opts.onDrag && this.opts.onDrag(d.handle, ix, iy);
          cv.style.cursor = "grabbing";
        } else if (d.moved) {
          this._panning = true; cv.style.cursor = "grabbing";
          this.tx += sx - d.lx; this.ty += sy - d.ly;
        }
        d.lx = sx; d.ly = sy;
      } else {
        const h = this.opts.pick ? this.opts.pick(sx, sy) : null;
        cv.style.cursor = h != null ? "move" : (this.opts.cursor || "crosshair");
        this.opts.onHover && this.opts.onHover(h, sx, sy);
      }
      this.redraw();
    });
    const up = (e) => {
      const d = this._down;
      this._down = null; this._panning = false;
      cv.style.cursor = this.opts.cursor || "crosshair";
      if (!d) return;
      const [ix, iy] = this.toImage(d.sx, d.sy);
      if (!d.moved && d.button === 0) this.opts.onClick && this.opts.onClick(ix, iy, e);
      if (d.moved && d.handle != null && d.button === 0) this.opts.onDragEnd && this.opts.onDragEnd(d.handle);
      this.redraw();
    };
    cv.addEventListener("pointerup", up);
    cv.addEventListener("pointercancel", up);
    cv.addEventListener("pointerleave", () => { this.mouse = null; this.opts.onHover && this.opts.onHover(null); this.redraw(); });
    cv.addEventListener("contextmenu", (e) => {
      e.preventDefault();
      const [sx, sy] = pos(e), [ix, iy] = this.toImage(sx, sy);
      this.opts.onRightClick && this.opts.onRightClick(ix, iy, sx, sy);
    });
    cv.addEventListener("wheel", (e) => {
      e.preventDefault();
      const [sx, sy] = pos(e);
      this.zoomSmooth(sx, sy, Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0018)));
    }, { passive: false });
  }
}

/* ---------- drawing helpers ---------- */
function drawMarker(c, x, y, label, ring = "#fff", active = false) {
  const h = active ? 4.5 : 3.5, a = h + 2, b = h + 6;
  c.beginPath();
  c.rect(x - h, y - h, 2 * h, 2 * h);
  c.moveTo(x - b, y); c.lineTo(x - a, y); c.moveTo(x + a, y); c.lineTo(x + b, y);
  c.moveTo(x, y - b); c.lineTo(x, y - a); c.moveTo(x, y + a); c.lineTo(x, y + b);
  c.lineWidth = 3; c.strokeStyle = COL.under; c.stroke();
  c.lineWidth = 1.2; c.strokeStyle = ring; c.stroke();
  if (active) {
    const p = b + 4;
    c.lineWidth = 1; c.strokeStyle = COL.fg; c.strokeRect(x - p, y - p, 2 * p, 2 * p);
  }
  if (label != null) {
    c.font = MONO; c.lineJoin = "round";
    const lx = x + h + 5, ly = y - h - 5;
    c.lineWidth = 3; c.strokeStyle = COL.under; c.strokeText(String(label), lx, ly);
    c.fillStyle = ring; c.fillText(String(label), lx, ly);
  }
}

function drawCross(c, x, y, col, s = 5) {
  c.beginPath(); c.moveTo(x - s, y - s); c.lineTo(x + s, y + s); c.moveTo(x + s, y - s); c.lineTo(x - s, y + s);
  c.lineWidth = 3; c.strokeStyle = COL.under; c.stroke();
  c.lineWidth = 1.2; c.strokeStyle = col; c.stroke();
}

/* the real camera: a target at its position, its view as a thin fan */
function drawCamera(c, x, y, yawDeg, hfov, col, len = 46) {
  if (yawDeg != null) {
    const a = (yawDeg - 90) * Math.PI / 180, h = ((hfov || 60) / 2) * Math.PI / 180;
    c.fillStyle = "rgba(255,255,255,.06)"; c.strokeStyle = COL.guide; c.lineWidth = 1;
    c.beginPath(); c.moveTo(x, y); c.arc(x, y, len, a - h, a + h); c.closePath(); c.fill(); c.stroke();
  }
  c.beginPath(); c.arc(x, y, 5.5, 0, Math.PI * 2);
  c.moveTo(x - 10, y); c.lineTo(x - 3, y); c.moveTo(x + 3, y); c.lineTo(x + 10, y);
  c.moveTo(x, y - 10); c.lineTo(x, y - 3); c.moveTo(x, y + 3); c.lineTo(x, y + 10);
  c.lineWidth = 3; c.strokeStyle = COL.under; c.stroke();
  c.lineWidth = 1.2; c.strokeStyle = col; c.stroke();
}

function errClass(e, warn) {
  if (e == null) return "bad";
  return e <= 0.6 * warn ? "good" : e <= warn ? "warn" : "bad";
}

/* aerial pixel <-> British National Grid */
class GeoImage {
  constructor(meta, w, h) { Object.assign(this, meta); this.W = w || meta.width; this.H = h || meta.height; }
  toImg(E, N) { return [(E - this.left) / (this.right - this.left) * this.W, (this.top - N) / (this.top - this.bottom) * this.H]; }
  toWorld(x, y) { return [this.left + x / this.W * (this.right - this.left), this.top - y / this.H * (this.top - this.bottom)]; }
  get pxPerM() { return this.W / (this.right - this.left); }
}

/* ======================================================================== calibrator */
function mountCalibrator(root, config, ui) {
  const S = {
    st: null, pairs: [], pending: null, fit: null, frame: 0, hist: [], hover: null,
    grid: true, foot: true, markCam: false, seq: 0, fitting: false, mapView: null, geo: null, busy: false,
  };
  root.innerHTML = `
    <div class="panel-head">
      <h2>Calibrate</h2><span class="sub" id="c-sub">Click the same ground point in the video, then on the aerial. 6–8 well-spread pairs.</span>
      <div class="grow"></div>
      <button class="btn sm white" id="c-auto" title="Fit the camera from the people / traffic in a run (motus autocalibrate)">Auto from movement</button>
    </div>
    <div class="scrub">
      <span>Frame</span><input type="range" id="c-frame" min="0" max="0" step="1" value="0" aria-label="Video frame">
      <span class="t num" id="c-time"></span>
      <button class="btn xs white" id="c-ref" title="Back to the frame of the saved calibration">Saved frame</button>
    </div>
    <div class="cal">
      <div class="views">
        <div class="view" id="c-video"><span class="tag"><i></i>Video</span>
          <div class="tools"><button id="c-grid" class="on" title="Ground grid (G)">${ICON.grid}</button><button data-fit="v" title="Fit (F)">${ICON.fit}</button></div>
          <div class="loading">Loading frame…</div><div class="ask" id="c-ask-v" hidden></div></div>
        <div class="view" id="c-map"><span class="tag"><i class="round"></i><span id="c-map-tag">Aerial</span></span>
          <div class="tools"><button id="c-foot" class="on" title="What the camera sees">${ICON.view}</button><button id="c-cam" title="Mark where you stood (C)">${ICON.camera}</button><button data-fit="m" title="Fit (F)">${ICON.fit}</button></div>
          <div class="loading">Loading aerial…</div><div class="ask" id="c-ask-m" hidden></div></div>
      </div>
      <div class="calside" id="c-side"></div>
    </div>`;
  const $ = (s) => root.querySelector(s);
  const complete = () => S.pairs.filter((p) => p.E != null);

  // -------------------------------------------------------------- viewers
  const vv = new Viewer($("#c-video"), {
    draw: (c, v) => drawVideo(c, v),
    pick: (sx, sy) => pickPair(vv, sx, sy, (p) => [p.u, p.v]),
    onClick: (x, y) => clickVideo(x, y),
    onRightClick: (x, y, sx, sy) => rightClick(vv, sx, sy, (p) => [p.u, p.v]),
    onDrag: (i, x, y) => { S.pairs[i].u = clamp(x, 0, vv.iw); S.pairs[i].v = clamp(y, 0, vv.ih); },
    onDragEnd: () => { refit(); renderSide(); },
    onHover: (h) => setHover(h),
    drawLoupe: (c, T) => { for (const [i, p] of S.pairs.entries()) { const [x, y] = T(p.u, p.v); drawCross(c, x, y, COL.fg, i === S.hover ? 6 : 4); } },
  });
  const mv = new Viewer($("#c-map"), {
    draw: (c, v) => drawMap(c, v),
    pick: (sx, sy) => S.geo ? pickPair(mv, sx, sy, (p) => p.E != null ? S.geo.toImg(p.E, p.N) : null) : null,
    onClick: (x, y) => clickMap(x, y),
    onRightClick: (x, y, sx, sy) => S.geo && rightClick(mv, sx, sy, (p) => p.E != null ? S.geo.toImg(p.E, p.N) : null),
    onDrag: (i, x, y) => { const [E, N] = S.geo.toWorld(x, y); S.pairs[i].E = E; S.pairs[i].N = N; },
    onDragEnd: () => { refit(); renderSide(); },
    onHover: (h) => setHover(h),
    drawLoupe: (c, T) => { for (const [i, p] of S.pairs.entries()) { if (p.E == null) continue; const [x, y] = T(...S.geo.toImg(p.E, p.N)); drawCross(c, x, y, COL.fg, i === S.hover ? 6 : 4); } },
  });

  function setHover(h) {
    const i = h == null ? null : h;
    if (i === S.hover) return;
    S.hover = i;
    root.querySelectorAll(".pair").forEach((el) => el.classList.toggle("hover", +el.dataset.i === i));
    vv.redraw(); mv.redraw();
  }

  function pickPair(v, sx, sy, xy) {
    let best = null, bd = 12;
    S.pairs.forEach((p, i) => {
      const q = xy(p); if (!q) return;
      const [x, y] = v.toScreen(q[0], q[1]); const d = Math.hypot(x - sx, y - sy);
      if (d < bd) { bd = d; best = i; }
    });
    return best;
  }

  // -------------------------------------------------------------- drawing
  function pointColour(i) {
    const f = S.fit && S.fit.ok && S.fit.points && S.fit.points[i];
    if (!f) return "#fff";
    if (!f.inlier) return COL.bad;
    return COL[errClass(f.error_m, S.st.warn_m)];
  }

  function drawVideo(c, v) {
    if (S.grid && S.fit && S.fit.ok && S.fit.grid) {
      c.lineWidth = 1; c.strokeStyle = COL.grid;
      for (const ln of S.fit.grid.lines) {
        c.beginPath();
        ln.forEach(([u, w], k) => { const [x, y] = v.toScreen(u, w); k ? c.lineTo(x, y) : c.moveTo(x, y); });
        c.stroke();
      }
    }
    S.pairs.forEach((p, i) => { const [x, y] = v.toScreen(p.u, p.v); drawMarker(c, x, y, i + 1, pointColour(i), i === S.hover); });
    if (S.pending) {
      const [x, y] = v.toScreen(S.pending.u, S.pending.v);
      drawMarker(c, x, y, S.pairs.length + 1, COL.fg, true);
    }
  }

  function drawMap(c, v) {
    const g = S.geo; if (!g) return;
    const fit = S.fit && S.fit.ok ? S.fit : null;
    if (S.foot && fit && fit.footprint && fit.footprint.length > 2) {
      c.beginPath();
      fit.footprint.forEach(([E, N], k) => { const [x, y] = v.toScreen(...g.toImg(E, N)); k ? c.lineTo(x, y) : c.moveTo(x, y); });
      c.closePath(); c.fillStyle = "rgba(255,255,255,.04)"; c.fill();
      c.setLineDash([4, 4]); c.lineWidth = 1; c.strokeStyle = COL.guide; c.stroke(); c.setLineDash([]);
    }
    const cp = S.st.camera_position;
    if (cp) {
      const [x, y] = v.toScreen(...g.toImg(cp.E, cp.N));
      const r = (cp.position_tol_m || 0) * g.pxPerM * v.scale;
      if (r > 4) { c.setLineDash([2, 3]); c.lineWidth = 1; c.strokeStyle = COL.guide; c.beginPath(); c.arc(x, y, r, 0, Math.PI * 2); c.stroke(); c.setLineDash([]); }
    }
    if (fit && fit.camera && fit.camera.E != null) {
      const [x, y] = v.toScreen(...g.toImg(fit.camera.E, fit.camera.N));
      drawCamera(c, x, y, fit.camera.yaw_deg, fit.camera.hfov_deg, COL.fg);
    } else if (cp) {
      const [x, y] = v.toScreen(...g.toImg(cp.E, cp.N));
      drawCamera(c, x, y, null, null, COL.guide);
    }
    if (fit && fit.points) {
      S.pairs.forEach((p, i) => {
        const q = fit.points[i]; if (!q || !q.proj || p.E == null) return;
        const [x0, y0] = v.toScreen(...g.toImg(p.E, p.N)), [x1, y1] = v.toScreen(...g.toImg(q.proj[0], q.proj[1]));
        c.lineWidth = 1; c.strokeStyle = COL.proj; c.beginPath(); c.moveTo(x0, y0); c.lineTo(x1, y1); c.stroke();
        drawCross(c, x1, y1, COL.proj, 3.5);
      });
    }
    if (S.pending && S.guess) {
      const [x, y] = v.toScreen(...g.toImg(S.guess.E, S.guess.N));
      const r = Math.max(14, 0.25 * S.guess.r_m * g.pxPerM * v.scale);
      c.setLineDash([2, 4]); c.lineWidth = 1; c.strokeStyle = COL.guide;
      c.beginPath(); c.arc(x, y, r, 0, Math.PI * 2); c.stroke(); c.setLineDash([]);
    }
    S.pairs.forEach((p, i) => { if (p.E == null) return; const [x, y] = v.toScreen(...g.toImg(p.E, p.N)); drawMarker(c, x, y, i + 1, pointColour(i), i === S.hover); });
  }

  // -------------------------------------------------------------- editing
  const snapshot = () => S.hist.push(JSON.stringify({ pairs: S.pairs, pending: S.pending }));

  function clickVideo(x, y) {
    if (S.markCam) { ui.toast("Click where you stood on the aerial", "info"); return; }
    if (!vv.inside(x, y)) return;
    if (S.pending) { S.pending = { u: x, v: y }; askState(); vv.redraw(); return; }   // re-place it
    snapshot();
    S.pending = { u: x, v: y }; S.guess = null;
    askState(); vv.redraw();
    const n = complete().length;
    if (n >= Math.max((S.fit && S.fit.need || 3) - 1, 3) && S.fit && S.fit.ok) refit([x, y]);
  }

  function clickMap(x, y) {
    if (!S.geo || !mv.inside(x, y)) return;
    const [E, N] = S.geo.toWorld(x, y);
    if (S.markCam) { setCamera(E, N); return; }
    if (!S.pending) { ui.toast("First click the point in the video", "info"); flash($("#c-video")); return; }
    snapshot();
    S.pairs.push({ u: S.pending.u, v: S.pending.v, E, N });
    S.pending = null; S.guess = null;
    if (S.mapView) { mv.setView(S.mapView); S.mapView = null; }
    askState(); renderSide(); refit();
  }

  function rightClick(v, sx, sy, xy) {
    const i = pickPair(v, sx, sy, xy);
    if (i != null) { snapshot(); S.pairs.splice(i, 1); S.hover = null; renderSide(); refit(); vv.redraw(); mv.redraw(); return; }
    undo();
  }

  function undo() {
    if (S.markCam) { S.markCam = false; askState(); return; }
    const last = S.hist.pop();
    if (!last) return;
    const o = JSON.parse(last);
    S.pairs = o.pairs; S.pending = o.pending; S.guess = null;
    if (!S.pending && S.mapView) { mv.setView(S.mapView); S.mapView = null; }
    askState(); renderSide(); refit(); vv.redraw(); mv.redraw();
  }

  function flash(el) { el.classList.add("active"); setTimeout(() => askState(), 600); }

  function askState() {
    const av = $("#c-ask-v"), am = $("#c-ask-m");
    const n = S.pairs.length + 1;
    av.hidden = am.hidden = true;
    $("#c-video").classList.remove("active"); $("#c-map").classList.remove("active");
    $("#c-cam").classList.toggle("on", S.markCam);
    if (S.markCam) { am.hidden = false; am.innerHTML = "Click where you <b>stood</b> (the camera)"; $("#c-map").classList.add("active"); return; }
    if (S.pending) { am.hidden = false; am.innerHTML = `Now point <b>${n}</b> on the aerial`; $("#c-map").classList.add("active"); }
    else { av.hidden = false; av.innerHTML = `Click point <b>${n}</b> on the ground`; $("#c-video").classList.add("active"); }
  }

  // -------------------------------------------------------------- fitting
  let fitTimer = 0;
  function refit(query) {
    clearTimeout(fitTimer);
    fitTimer = setTimeout(() => doFit(query), query ? 0 : 90);
  }

  async function doFit(query) {
    const seq = ++S.seq;
    S.fitting = true; renderStats();
    try {
      const r = await ui.post("/api/cal/fit", { config, pairs: complete(), query: query || null });
      if (seq !== S.seq) return;
      S.fit = r;
      if (query && r.guess && S.pending && S.geo) {
        S.guess = r.guess;
        if (!S.mapView) S.mapView = mv.getView();
        const [x, y] = S.geo.toImg(r.guess.E, r.guess.N), rp = r.guess.r_m * S.geo.pxPerM;
        mv.frame(x - rp, y - rp, x + rp, y + rp);
      }
    } catch (e) {
      if (seq === S.seq) S.fit = { ok: false, error: e.message };
    } finally {
      if (seq === S.seq) { S.fitting = false; renderStats(); renderPairs(); vv.redraw(); mv.redraw(); }
    }
  }

  // -------------------------------------------------------------- camera
  async function setCamera(E, N) {
    const cp = S.st.camera_position || {};
    const h = parseFloat(($("#c-cam-h") || {}).value || cp.height_m);
    if (!(h > 0)) { ui.toast("Enter the camera height first (metres above the ground)", "error"); return; }
    S.markCam = false; askState();
    try {
      const body = { config, E: E.toFixed(1), N: N.toFixed(1), height_m: h, position_tol_m: 4,
                     note: "set on the aerial in the app" };
      if (cp.height_tol_m && cp.height_m === h) body.height_tol_m = cp.height_tol_m;
      if (cp.hfov_deg) body.hfov_deg = cp.hfov_deg;
      await ui.post("/api/camera", body);
      S.st = await ui.get(`/api/cal/state?config=${encodeURIComponent(config)}`);
      ui.toast("Camera position saved in the site config", "ok");
      ui.siteChanged && ui.siteChanged();
      renderSide(); refit(); mv.redraw();
    } catch (e) { ui.toast(e.message, "error"); }
  }

  // -------------------------------------------------------------- side panel
  function renderSide() {
    const st = S.st; if (!st) return;
    const cp = st.camera_position;
    const saved = st.saved;
    const site = ui.site() || {};
    const calib = site.calibration || {};
    $("#c-side").innerHTML = `
      <div class="card" id="c-stats"></div>
      <div class="card"><h3>Pairs <span class="chip" id="c-npairs">${S.pairs.length}</span><div class="grow"></div>
        <button class="btn xs" id="c-undo" title="Undo (U)">Undo</button><button class="btn xs danger" id="c-clear">Clear</button></h3>
        <div class="pairs" id="c-pairs"></div>
        <div class="keys"><span><kbd>Click</kbd> add</span><span><kbd>Drag</kbd> pan / move a point</span><span><kbd>Wheel</kbd> zoom</span><span><kbd>Right-click</kbd> delete</span><span><kbd>U</kbd> undo</span></div>
      </div>
      <div class="card">
        <button class="btn neon" id="c-save" style="justify-content:center">${ICON.check} Save calibration</button>
        <div class="btns"><button class="btn sm" id="c-revert" ${saved || st.pairs.length ? "" : "disabled"}>Back to saved pairs</button></div>
        <div class="hint">Saving runs <span class="mono">motus calibrate</span> with these pairs; the old calibration is kept in <span class="mono">calibration/backup</span>.</div>
      </div>
      <div class="card"><h3>Camera position ${cp ? '<span class="chip ok">set</span>' : '<span class="chip warn">not set</span>'}</h3>
        ${cp ? `<dl class="kv"><dt>E, N</dt><dd class="num">${cp.E}, ${cp.N}</dd><dt>Height</dt><dd class="num">${cp.height_m} m ± ${cp.height_tol_m ?? "–"}</dd><dt>Within</dt><dd class="num">${cp.position_tol_m ?? "–"} m</dd></dl>` : '<p class="hint">With the camera position known, 3 pairs are enough and the fit is a physical camera (much more reliable). Mark it on the aerial.</p>'}
        <div class="fields"><label class="field">Height above ground (m)<input id="c-cam-h" type="number" step="0.1" min="0.3" value="${cp ? cp.height_m : ""}"></label></div>
        <button class="btn sm white" id="c-mark">${ICON.camera} Mark where you stood</button>
      </div>
      <div class="card"><h3>Saved calibration ${saved ? `<span class="chip ${errClass(saved.rmse_m, st.warn_m) === "good" ? "ok" : errClass(saved.rmse_m, st.warn_m) === "warn" ? "warn" : "bad"} num">${fmtM(saved.rmse_m)}</span>` : '<span class="chip">none</span>'}</h3>
        ${saved ? `<dl class="kv"><dt>Pairs</dt><dd>${saved.n_points} on frame ${saved.frame}</dd><dt>Method</dt><dd>${saved.method === "points" ? (saved.camera ? "camera fit" : "homography") : saved.method}</dd><dt>Saved</dt><dd>${ui.ago(saved.mtime)}</dd></dl>` : ""}
        ${(calib.checks || []).length ? `<div class="gallery" style="grid-template-columns:1fr 1fr">${calib.checks.map((p, k) => `<button class="shot" data-check="${k}"><img src="${ui.thumb(p, 360)}" alt=""><span>${ui.base(p)}</span></button>`).join("")}</div>` : ""}
        ${calib.auto_candidate ? `<div class="note">An automatic calibration is waiting${calib.auto_scores != null ? ` (leave-one-out ${fmtM(calib.auto_scores)})` : ""}.</div>
           <div class="btns">${calib.auto_check ? `<button class="btn sm" data-autocheck>See its check</button>` : ""}<button class="btn sm primary" id="c-adopt">Use it</button></div>` : ""}
      </div>`;
    renderStats(); renderPairs();
    $("#c-undo").onclick = undo;
    $("#c-clear").onclick = async () => { if (!S.pairs.length || await ui.ask("Clear all pairs?", "The saved file is not touched until you save.", "Clear")) { snapshot(); S.pairs = []; S.pending = null; S.fit = null; renderSide(); vv.redraw(); mv.redraw(); askState(); } };
    $("#c-save").onclick = save;
    $("#c-revert").onclick = () => { snapshot(); S.pairs = st.pairs.map((p) => ({ ...p })); S.pending = null; renderSide(); refit(); vv.redraw(); mv.redraw(); askState(); };
    $("#c-mark").onclick = () => { S.markCam = !S.markCam; askState(); };
    root.querySelectorAll("[data-check]").forEach((b) => b.onclick = () => ui.lightbox(calib.checks.map((p) => ({ path: p })), +b.dataset.check));
    const ac = root.querySelector("[data-autocheck]"); if (ac) ac.onclick = () => ui.lightbox([{ path: calib.auto_check }], 0);
    const ad = $("#c-adopt");
    if (ad) ad.onclick = async () => {
      if (!await ui.ask("Use the automatic calibration?", "The current one is kept as <name>.before_auto.json.", "Use it")) return;
      try { await ui.post("/api/adopt-auto", { config }); ui.toast("Automatic calibration in use", "ok"); await reload(); ui.siteChanged && ui.siteChanged(); }
      catch (e) { ui.toast(e.message, "error"); }
    };
  }

  function renderStats() {
    const el = $("#c-stats"); if (!el || !S.st) return;
    const f = S.fit, n = complete().length, need = (f && f.need) || (S.st.camera_position ? 3 : 5);
    let html;
    if (f && f.ok) {
      const cls = errClass(f.rmse_m, S.st.warn_m);
      const cam = f.camera;
      html = `<div style="display:flex;align-items:end;gap:10px"><div class="big-rmse ${cls}"><span class="v">${(f.rmse_m ?? 0).toFixed(2)}</span><small>m RMSE</small></div><div class="grow"></div>${S.fitting ? '<span class="chip">fitting…</span>' : `<span class="chip">${f.method === "camera" ? "camera fit" : "homography"}</span>`}</div>
        <div class="mini-stats"><div><b>${fmtM(f.max_error_m)}</b><span>worst</span></div><div><b>${f.loo_rmse_m != null ? fmtM(f.loo_rmse_m) : "–"}</b><span>leave-one-out</span></div><div><b>${n}</b><span>pairs</span></div></div>
        ${f.spread != null && f.spread < 0.3 ? '<div class="note warn">The points lie along one line: add some left / right and near / far.</div>' : ""}
        ${n < 6 ? `<div class="note">${6 - n} more well-spread pair${6 - n === 1 ? "" : "s"} recommended.</div>` : ""}
        ${(f.dropped || []).length ? `<div class="note warn">Pairs ${f.dropped.join(", ")} disagree with the others.</div>` : ""}
        ${cam ? `<div class="hint num">Camera ${cam.height_m != null ? `${(+cam.height_m).toFixed(1)} m up` : ""}${cam.yaw_deg != null ? ` · facing ${(+cam.yaw_deg).toFixed(0)}°` : ""}${cam.tilt_deg != null ? ` · tilt ${(+cam.tilt_deg).toFixed(1)}°` : ""}${cam.hfov_deg != null ? ` · ${(+cam.hfov_deg).toFixed(0)}° wide` : ""}</div>` : ""}`;
    } else {
      html = `<div class="big-rmse" style="color:var(--muted)">–<small>m RMSE</small></div>
        <p class="hint">${f && f.error ? `Fit failed: ${ui.esc(f.error)}` : `The fit starts at ${need} pairs${S.st.camera_position ? " (camera position known)" : ""}. ${n} so far.`}</p>${S.fitting ? '<span class="chip">fitting…</span>' : ""}`;
    }
    el.innerHTML = html;
    const now = f && f.ok ? f.rmse_m : null;
    if (now != null && S.shownRmse != null && typeof tween === "function") tween(el.querySelector(".big-rmse .v"), S.shownRmse, now, (v) => v.toFixed(2));
    if (!S.fitting) S.shownRmse = now;
  }

  function renderPairs() {
    const el = $("#c-pairs"); if (!el) return;
    $("#c-npairs").textContent = S.pairs.length;
    if (!S.pairs.length) { el.innerHTML = '<div class="empty" style="padding:8px">No pairs yet</div>'; return; }
    const f = S.fit && S.fit.ok ? S.fit : null;
    el.innerHTML = S.pairs.map((p, i) => {
      const q = f && f.points && f.points[i];
      const e = q ? (q.inlier ? q.error_m : null) : undefined;
      const cls = q ? (q.inlier ? errClass(q.error_m, S.st.warn_m) : "bad") : "";
      return `<div class="pair ${i === S.hover ? "hover" : ""}" data-i="${i}"><span class="k">${String(i + 1).padStart(2, "0")}</span>
        <span class="xy">${p.E.toFixed(1)}, ${p.N.toFixed(1)}</span>
        <span class="e ${cls}">${e === undefined ? "" : q.inlier ? fmtM(e) : "outlier"}</span>
        <button title="Delete this pair" data-del="${i}">${ICON.x}</button></div>`;
    }).join("");
    el.querySelectorAll(".pair").forEach((row) => {
      row.onmouseenter = () => setHover(+row.dataset.i);
      row.onmouseleave = () => setHover(null);
      row.ondblclick = () => focusPair(+row.dataset.i);
    });
    el.querySelectorAll("[data-del]").forEach((b) => b.onclick = (ev) => { ev.stopPropagation(); snapshot(); S.pairs.splice(+b.dataset.del, 1); S.hover = null; renderSide(); refit(); vv.redraw(); mv.redraw(); });
  }

  function focusPair(i) {
    const p = S.pairs[i]; if (!p) return;
    const r = 60; vv.frame(p.u - r * 2, p.v - r, p.u + r * 2, p.v + r);
    const [x, y] = S.geo.toImg(p.E, p.N), rm = 12 * S.geo.pxPerM; mv.frame(x - rm, y - rm, x + rm, y + rm);
  }

  async function save() {
    const n = complete().length;
    if (S.pending) { ui.toast("Finish the open pair on the aerial first (or press U)", "error"); return; }
    if (n < 3) { ui.toast("At least 3 pairs are needed", "error"); return; }
    if (n < 6 && !await ui.ask(`Only ${n} pairs`, "At least 6 well-spread pairs are needed for a trustworthy fit. Save anyway?", "Save anyway")) return;
    try {
      $("#c-save").disabled = true;
      const job = await ui.post("/api/cal/save", { config, frame: S.frame, pairs: complete() });
      ui.toast("Fitting and saving the calibration…");
      ui.watch(job, async (j) => {
        if (!$("#c-save")) return;
        $("#c-save").disabled = false;
        if (j.status === "done") { await reload(); ui.toast(`Calibration saved: RMSE ${fmtM(S.st.saved && S.st.saved.rmse_m)}`, "ok"); }
      });
    } catch (e) { $("#c-save").disabled = false; ui.toast(e.message, "error"); }
  }

  // -------------------------------------------------------------- frames
  let frameTimer = 0;
  function setFrame(n, now = false) {
    S.frame = n;
    const fps = S.st.video.fps;
    $("#c-frame").value = n;
    $("#c-time").textContent = `${(n / fps).toFixed(2)} s · frame ${n}`;
    clearTimeout(frameTimer);
    frameTimer = setTimeout(() => vv.load(`/api/cal/frame?config=${encodeURIComponent(config)}&frame=${n}`).catch((e) => ui.toast(e.message, "error")), now ? 0 : 140);
  }

  async function reload() {
    S.st = await ui.get(`/api/cal/state?config=${encodeURIComponent(config)}`);
    await ui.refreshSite();
    renderSide();
  }

  // -------------------------------------------------------------- start
  (async () => {
    try {
      S.st = await ui.get(`/api/cal/state?config=${encodeURIComponent(config)}`);
    } catch (e) {
      root.querySelector(".cal").innerHTML = `<div class="card"><h3>Can't calibrate yet</h3><p>${ui.esc(e.message)}</p><div class="btns"><button class="btn primary" data-goto="1">Set footage and aerial</button></div></div>`;
      root.querySelector("[data-goto]").onclick = () => ui.goto(1);
      return;
    }
    const st = S.st;
    S.pairs = st.pairs.map((p) => ({ ...p }));
    const f0 = st.saved ? st.saved.frame : st.reference_frame || 0;
    $("#c-frame").max = Math.max(0, st.video.frames - 1);
    $("#c-map-tag").textContent = `Aerial · EPSG:${st.map.epsg || "?"}`;
    $("#c-ref").onclick = () => setFrame(f0, true);
    $("#c-frame").oninput = (e) => setFrame(+e.target.value);
    $("#c-grid").onclick = () => { S.grid = !S.grid; $("#c-grid").classList.toggle("on", S.grid); vv.redraw(); };
    $("#c-foot").onclick = () => { S.foot = !S.foot; $("#c-foot").classList.toggle("on", S.foot); mv.redraw(); };
    $("#c-cam").onclick = () => { S.markCam = !S.markCam; askState(); };
    root.querySelectorAll("[data-fit]").forEach((b) => b.onclick = () => (b.dataset.fit === "v" ? vv : mv).fit());
    $("#c-auto").onclick = async () => {
      if (!await ui.ask("Calibrate automatically?", "Fits the camera from the people / vehicles in the latest run (motus autocalibrate). The result waits as a candidate: your calibration is only replaced when you choose 'Use it'.", "Start")) return;
      try { const j = await ui.post("/api/job", { kind: "autocalibrate", config }); ui.watch(j, async () => { await reload(); }); ui.toast("Automatic calibration started"); }
      catch (e) { ui.toast(e.message, "error"); }
    };
    renderSide(); askState();
    setFrame(f0, true);
    await vv.load(`/api/cal/frame?config=${encodeURIComponent(config)}&frame=${f0}`).catch((e) => ui.toast(e.message, "error"));
    $("#c-video .loading").remove();
    try {
      await mv.load(`/api/cal/map?config=${encodeURIComponent(config)}`);
      S.geo = new GeoImage(st.map, mv.iw, mv.ih);
      $("#c-map .loading").remove();
      // open the aerial around the clicked points, else around the camera
      const pts = S.pairs.length ? S.pairs : st.camera_position ? [st.camera_position] : [];
      if (pts.length) {
        const xy = pts.map((p) => S.geo.toImg(p.E, p.N));
        const xs = xy.map((q) => q[0]), ys = xy.map((q) => q[1]), pad = 40 * S.geo.pxPerM;
        mv.frame(Math.min(...xs) - pad, Math.min(...ys) - pad, Math.max(...xs) + pad, Math.max(...ys) + pad, false);
      }
    } catch (e) { ui.toast(e.message, "error"); }
    refit();
  })();

  return {
    destroy() { vv.destroy(); mv.destroy(); clearTimeout(fitTimer); clearTimeout(frameTimer); },
    key(e) {
      const k = e.key.toLowerCase();
      if ((k === "z" && (e.ctrlKey || e.metaKey)) || k === "u" || k === "backspace") { undo(); return true; }
      if (k === "escape") { if (S.markCam) { S.markCam = false; askState(); return true; } if (S.pending) { S.pending = null; S.guess = null; if (S.mapView) { mv.setView(S.mapView); S.mapView = null; } askState(); vv.redraw(); mv.redraw(); return true; } return false; }
      if (k === "g") { $("#c-grid").click(); return true; }
      if (k === "f") { vv.fit(); mv.fit(); return true; }
      if (k === "c") { $("#c-cam").click(); return true; }
      if (k === "enter") { save(); return true; }
      return false;
    },
  };
}

/* ======================================================================== area (ROI) */
function mountArea(root, config, ui) {
  const S = { st: null, pts: [], hist: [], hover: null, saved: null };
  root.innerHTML = `
    <div class="panel-head"><h2>Area to track</h2><span class="sub">Outline the ground that matters. Sky, far blurry ground and reflections stay outside.</span></div>
    <div class="cal area">
      <div class="views"><div class="view" id="a-view"><span class="tag"><i></i><span id="a-tag">Calibration frame</span></span>
        <div class="tools"><button data-fit title="Fit (F)">${ICON.fit}</button></div><div class="loading">Loading frame…</div><div class="ask" id="a-ask"></div></div></div>
      <div class="calside" id="a-side"></div>
    </div>`;
  const $ = (s) => root.querySelector(s);

  const v = new Viewer($("#a-view"), {
    draw: (c, vw) => draw(c, vw),
    pick: (sx, sy) => pick(sx, sy),
    onClick: (x, y) => add(x, y),
    onRightClick: (x, y, sx, sy) => { const i = pick(sx, sy); if (i != null) { snap(); S.pts.splice(i, 1); render(); } else undo(); },
    onDrag: (i, x, y) => { S.pts[i] = [clamp(x, 0, v.iw), clamp(y, 0, v.ih)]; },
    onDragEnd: () => render(),
    onHover: (h) => { if (h !== S.hover) { S.hover = h; v.redraw(); } },
  });

  const snap = () => S.hist.push(JSON.stringify(S.pts));
  function undo() { const o = S.hist.pop(); if (o) { S.pts = JSON.parse(o); render(); } }

  function pick(sx, sy) {
    let best = null, bd = 11;
    S.pts.forEach(([x, y], i) => { const [a, b] = v.toScreen(x, y); const d = Math.hypot(a - sx, b - sy); if (d < bd) { bd = d; best = i; } });
    return best;
  }

  function add(x, y) {
    if (!v.inside(x, y)) return;
    snap();
    // a click on an edge puts the corner into that edge, else it extends the outline
    const n = S.pts.length;
    if (n >= 3) {
      const [sx, sy] = v.toScreen(x, y);
      let best = -1, bd = 8;
      for (let i = 0; i < n; i++) {
        const a = v.toScreen(...S.pts[i]), b = v.toScreen(...S.pts[(i + 1) % n]);
        const d = segDist(sx, sy, a, b);
        if (d < bd) { bd = d; best = i; }
      }
      if (best >= 0) { S.pts.splice(best + 1, 0, [x, y]); render(); return; }
    }
    S.pts.push([x, y]); render();
  }

  function segDist(px, py, a, b) {
    const dx = b[0] - a[0], dy = b[1] - a[1], L = dx * dx + dy * dy || 1;
    const t = clamp(((px - a[0]) * dx + (py - a[1]) * dy) / L, 0, 1);
    return Math.hypot(px - (a[0] + t * dx), py - (a[1] + t * dy));
  }

  function draw(c, vw) {
    const P = S.pts.map(([x, y]) => vw.toScreen(x, y));
    if (P.length >= 3) {
      c.beginPath(); c.rect(0, 0, vw.w, vw.h);
      P.forEach(([x, y], k) => k ? c.lineTo(x, y) : c.moveTo(x, y)); c.closePath();
      c.fillStyle = "rgba(0,0,0,.55)"; c.fill("evenodd");
    }
    if (P.length >= 2) {
      c.beginPath(); P.forEach(([x, y], k) => k ? c.lineTo(x, y) : c.moveTo(x, y));
      if (P.length >= 3) c.closePath();
      c.lineWidth = 3; c.strokeStyle = COL.under; c.stroke();
      c.lineWidth = 1.2; c.strokeStyle = COL.fg; c.stroke();
    }
    P.forEach(([x, y], i) => {
      const r = 3.5;
      c.fillStyle = COL.fg; c.fillRect(x - r, y - r, 2 * r, 2 * r);
      c.lineWidth = 1; c.strokeStyle = COL.bg; c.strokeRect(x - r, y - r, 2 * r, 2 * r);
      if (i === S.hover) { c.strokeStyle = COL.fg; c.strokeRect(x - 9, y - 9, 18, 18); }
    });
  }

  function render() {
    v.redraw();
    const n = S.pts.length, st = S.st;
    $("#a-ask").innerHTML = n < 3 ? `Click corner <b>${n + 1}</b> of the area` : "Drag a corner to move it · click an edge to add one";
    const changed = JSON.stringify(S.pts) !== JSON.stringify(S.saved || []);
    $("#a-side").innerHTML = `
      <div class="card"><h3>Outline ${st.roi ? '<span class="chip ok">saved</span>' : '<span class="chip">whole frame</span>'}</h3>
        <div class="big-rmse">${n}<small>corner${n === 1 ? "" : "s"}</small></div>
        <p class="hint">Only people and vehicles inside the outline are tracked. Without one, the whole frame is used.</p>
        <div class="keys"><span><kbd>Click</kbd> add</span><span><kbd>Drag</kbd> move</span><span><kbd>Right-click</kbd> delete</span><span><kbd>U</kbd> undo</span></div>
      </div>
      <div class="card">
        <button class="btn neon" id="a-save" style="justify-content:center" ${n >= 3 && changed ? "" : "disabled"}>${ICON.check} Save area</button>
        <div class="btns"><button class="btn sm" id="a-undo">Undo</button><button class="btn sm" id="a-clear" ${n ? "" : "disabled"}>Clear</button>
          <button class="btn sm" id="a-revert" ${changed && S.saved ? "" : "disabled"}>Back to saved</button></div>
        ${st.roi ? '<button class="btn sm danger" id="a-remove">Remove the area (track the whole frame)</button>' : ""}
        ${st.roi_inline ? '<div class="note warn">This config has the area written inline (detection.roi: [...]). Replace it with a file name to edit it here.</div>' : `<div class="path">${ui.esc(st.roi_file || "")}</div>`}
      </div>`;
    $("#a-save").onclick = save;
    $("#a-undo").onclick = undo;
    $("#a-clear").onclick = () => { snap(); S.pts = []; render(); };
    $("#a-revert").onclick = () => { snap(); S.pts = JSON.parse(JSON.stringify(S.saved)); render(); };
    const rm = $("#a-remove");
    if (rm) rm.onclick = async () => {
      if (!await ui.ask("Remove the area?", "The whole frame is tracked again. The file is kept in backup/.", "Remove")) return;
      try { await ui.post("/api/roi/clear", { config }); ui.toast("Area removed", "ok"); await load(); ui.refreshSite(); } catch (e) { ui.toast(e.message, "error"); }
    };
  }

  async function save() {
    try {
      const r = await ui.post("/api/roi/save", { config, points: S.pts });
      ui.toast("Area saved", "ok");
      await load(); ui.refreshSite();
      return r;
    } catch (e) { ui.toast(e.message, "error"); }
  }

  async function load() {
    S.st = await ui.get(`/api/cal/state?config=${encodeURIComponent(config)}`);
    S.saved = S.st.roi; S.pts = S.saved ? JSON.parse(JSON.stringify(S.saved)) : [];
    render();
  }

  (async () => {
    try { await load(); } catch (e) {
      root.querySelector(".cal").innerHTML = `<div class="card"><h3>No video yet</h3><p>${ui.esc(e.message)}</p></div>`;
      return;
    }
    const f = S.st.reference_frame || 0;
    $("#a-tag").textContent = `Calibration frame ${f}`;
    root.querySelector("[data-fit]").onclick = () => v.fit();
    await v.load(`/api/cal/frame?config=${encodeURIComponent(config)}&frame=${f}`).catch((e) => ui.toast(e.message, "error"));
    const l = root.querySelector(".loading"); if (l) l.remove();
  })();

  return {
    destroy() { v.destroy(); },
    key(e) {
      const k = e.key.toLowerCase();
      if ((k === "z" && (e.ctrlKey || e.metaKey)) || k === "u" || k === "backspace") { undo(); return true; }
      if (k === "f") { v.fit(); return true; }
      if (k === "enter" && S.pts.length >= 3) { save(); return true; }
      return false;
    },
  };
}

/* ======================================================================== camera map */
function mountCameraMap(host, config, ui, onPick) {
  const S = { geo: null, cam: null };
  host.innerHTML = `<span class="tag"><i class="round"></i>Click where you stood</span>
    <div class="tools"><button data-fit title="Fit">${ICON.fit}</button></div><div class="loading">Loading aerial…</div>`;
  const v = new Viewer(host, {
    draw: (c, vw) => {
      if (!S.geo || !S.cam || S.cam.E == null) return;
      const [x, y] = vw.toScreen(...S.geo.toImg(S.cam.E, S.cam.N));
      const r = (S.cam.tol || 0) * S.geo.pxPerM * vw.scale;
      if (r > 4) { c.fillStyle = "rgba(255,255,255,.05)"; c.strokeStyle = COL.guide; c.setLineDash([2, 3]); c.lineWidth = 1; c.beginPath(); c.arc(x, y, r, 0, Math.PI * 2); c.fill(); c.stroke(); c.setLineDash([]); }
      drawCamera(c, x, y, null, null, COL.fg);
    },
    onClick: (x, y) => { if (!S.geo || !v.inside(x, y)) return; const [E, N] = S.geo.toWorld(x, y); onPick(E, N); },
  });
  host.querySelector("[data-fit]").onclick = () => v.fit();
  (async () => {
    try {
      const st = await ui.get(`/api/cal/state?config=${encodeURIComponent(config)}`);
      await v.load(`/api/cal/map?config=${encodeURIComponent(config)}`);
      S.geo = new GeoImage(st.map, v.iw, v.ih);
      host.querySelector(".loading").remove();
      if (S.cam && S.cam.E != null) { const [x, y] = S.geo.toImg(S.cam.E, S.cam.N), p = 70 * S.geo.pxPerM; v.frame(x - p, y - p, x + p, y + p, false); }
    } catch (e) { host.querySelector(".loading").textContent = e.message; }
  })();
  return {
    set(E, N, tol) {
      S.cam = { E: +E, N: +N, tol: +tol || 0 };
      v.redraw();
    },
    centre() { if (S.geo && S.cam && S.cam.E != null) { const [x, y] = S.geo.toImg(S.cam.E, S.cam.N), p = 70 * S.geo.pxPerM; v.frame(x - p, y - p, x + p, y + p); } },
    destroy() { v.destroy(); },
  };
}

function fmtM(v) { return v == null || !isFinite(v) ? "–" : `${(+v).toFixed(2)} m`; }

const ICON = {
  grid: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><path d="M3 9h18M3 15h18M9 3v18M15 3v18"/></svg>',
  fit: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/></svg>',
  view: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"><path d="M12 20 4 6h16z"/></svg>',
  camera: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="10" r="3"/><path d="M12 21s-7-6.2-7-11a7 7 0 0 1 14 0c0 4.8-7 11-7 11z"/></svg>',
  check: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="m5 12 5 5 9-10"/></svg>',
  x: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M6 6l12 12M18 6 6 18"/></svg>',
};
