/* Live view: thin screen-space lines (three.js Line2 family) written in place.

   Segs is a LineSegments2 with a fixed capacity: segments are written into its buffers (no
   allocation per frame) and only the first n are drawn. Widths are CSS pixels; colours are
   plain sRGB values (the view runs without colour management: what is written is what shows). */

import * as THREE from "three";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";

export function lineMaterial(opts = {}) {
  const m = new LineMaterial({
    color: opts.color ?? 0xffffff, linewidth: opts.width ?? 1, vertexColors: !!opts.vertexColors,
    transparent: true, opacity: opts.opacity ?? 1, dashed: !!opts.dashed, dashSize: opts.dash ?? 0.3,
    gapSize: opts.gap ?? 0.6, alphaToCoverage: false, depthWrite: false,
    depthTest: opts.depthTest ?? true, blending: opts.additive ? THREE.AdditiveBlending : THREE.NormalBlending,
  });
  m.baseWidth = m.linewidth;          // the look presets scale from this
  m.baseOpacity = m.opacity;
  return m;
}

export class Segs {
  constructor(cap, material, { distances = false, order = 0 } = {}) {
    this.cap = Math.max(1, cap | 0);
    this.pos = new Float32Array(6 * this.cap);
    this.col = new Float32Array(6 * this.cap).fill(1);
    const g = new LineSegmentsGeometry();
    const buf = (arr, stride) => {
      const b = new THREE.InstancedInterleavedBuffer(arr, stride, 1);
      b.setUsage(THREE.DynamicDrawUsage);
      return b;
    };
    this.pb = buf(this.pos, 6);
    this.cb = buf(this.col, 6);
    g.setAttribute("instanceStart", new THREE.InterleavedBufferAttribute(this.pb, 3, 0));
    g.setAttribute("instanceEnd", new THREE.InterleavedBufferAttribute(this.pb, 3, 3));
    g.setAttribute("instanceColorStart", new THREE.InterleavedBufferAttribute(this.cb, 3, 0));
    g.setAttribute("instanceColorEnd", new THREE.InterleavedBufferAttribute(this.cb, 3, 3));
    if (distances) {
      this.dist = new Float32Array(2 * this.cap);
      this.db = buf(this.dist, 2);
      g.setAttribute("instanceDistanceStart", new THREE.InterleavedBufferAttribute(this.db, 1, 0));
      g.setAttribute("instanceDistanceEnd", new THREE.InterleavedBufferAttribute(this.db, 1, 1));
    }
    g.instanceCount = 0;
    this.geo = g;
    this.obj = new LineSegments2(g, material);
    this.obj.frustumCulled = false;
    this.obj.renderOrder = order;
    this.n = 0;
  }

  get material() { return this.obj.material; }

  /** Segment k from a to b, colour (r, g, b) at both ends (or c2 at the end). */
  set(k, ax, ay, az, bx, by, bz, r = 1, g = 1, b = 1, r2 = r, g2 = g, b2 = b) {
    const p = this.pos, c = this.col, o = 6 * k;
    p[o] = ax; p[o + 1] = ay; p[o + 2] = az; p[o + 3] = bx; p[o + 4] = by; p[o + 5] = bz;
    c[o] = r; c[o + 1] = g; c[o + 2] = b; c[o + 3] = r2; c[o + 4] = g2; c[o + 5] = b2;
  }

  /** Dash distances of segment k (metres along its line). */
  setDist(k, d0, d1) { this.dist[2 * k] = d0; this.dist[2 * k + 1] = d1; }

  /** Draw the first n; segments from `from` on were (re)written. */
  commit(n, from = 0) {
    n = Math.min(n | 0, this.cap);
    from = Math.max(0, Math.min(from | 0, n));
    this.n = n;
    this.geo.instanceCount = n;
    if (n > from) {
      for (const [b, w] of [[this.pb, 6], [this.cb, 6], [this.db, 2]]) {
        if (!b) continue;
        b.clearUpdateRanges();
        b.addUpdateRange(from * w, (n - from) * w);
        b.needsUpdate = true;
      }
    }
  }

  /** A closed or open polyline (flat xyz) as segments, one colour; returns the count written. */
  polyline(xyz, k0 = 0, r = 1, g = 1, b = 1, lift = 0, dist = false) {
    let k = k0, d = 0;
    for (let i = 0; i + 5 < xyz.length; i += 3) {
      const ax = xyz[i], ay = xyz[i + 1], az = xyz[i + 2] + lift;
      const bx = xyz[i + 3], by = xyz[i + 4], bz = xyz[i + 5] + lift;
      if (!(ax === ax && bx === bx)) continue;
      this.set(k, ax, ay, az, bx, by, bz, r, g, b);
      if (dist && this.dist) {
        const L = Math.hypot(bx - ax, by - ay, bz - az);
        this.setDist(k, d, d + L);
        d += L;
      }
      k++;
    }
    return k - k0;
  }

  dispose() { this.geo.dispose(); this.obj.material.dispose(); }
}

/** A round dot texture for heads (soft 1 px edge, no glow). */
export function dotTexture(px = 32) {
  const c = document.createElement("canvas");
  c.width = c.height = px;
  const g = c.getContext("2d");
  g.fillStyle = "#fff";
  g.beginPath();
  g.arc(px / 2, px / 2, px / 2 - 1, 0, 2 * Math.PI);
  g.fill();
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.NoColorSpace;
  return t;
}
