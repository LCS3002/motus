/* Live view: the ground, the two cameras (the start camera as a white ghost, the virtual
   camera in violet with the blurred video in its image plane, re-framed) and the drag grid.

   Scene frame = groundtrack's local frame (metres, x east, y north, z up). Cameras in OpenCV
   axes (x right, y down, z forward), R_wc = world_from_camera. All of it stays put when the space
   clutch turns (08 a, 28 Oct: only the drawn crowd turns, tracks.js), and the ground never hides
   the crowd (setUnder): the orbit can go below it. */

import * as THREE from "three";
import { footprint, project, dragGrid, groundHeight, DRAWN } from "./projection.js";
import { Segs, lineMaterial } from "./lines.js";

export const VIOLET = 0x9b82d9;           // tokens.css --violet-line: #4b2e83's hue, light enough on black
const VIOLET_RGB = [0x9b / 255, 0x82 / 255, 0xd9 / 255];
const LIFT = 0.04;                        // lines sit this far above the ground (no z-fighting)

// ---------------------------------------------------------------------------- the ground
const GROUND_VS = /* glsl */`
  #include <common>
  #include <clipping_planes_pars_vertex>
  varying vec3 vW;             // the scene frame
  varying vec2 vUv;
  void main() {
    vUv = uv;
    vW = position;
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mvPosition;
    #include <clipping_planes_vertex>
  }`;

const GROUND_FS = /* glsl */`
  #include <common>
  #include <clipping_planes_pars_fragment>
  uniform sampler2D map;
  uniform int mode;            // 0 A: the grey aerial, 1 B: contours, 2 C: a grid
  uniform float aerialGain;    // A: brightness of the aerial (0.3 = 30 %)
  uniform vec3 lineCol;        // B, C: colour of the lines
  uniform vec3 majorCol;       // B: every 5th contour; C: every 5th grid line
  uniform float contourStep;   // B: metres between contours
  uniform float gridStep;      // C: metres between grid lines
  uniform vec2 centre;         // the camera (xy): the ground fades out past fade.x .. fade.y
  uniform vec2 fade;
  varying vec3 vW;
  varying vec2 vUv;
  float lineAt(float v) {      // 1 on a whole value of v, ~1 px wide, antialiased
    float w = fwidth(v);
    float d = abs(fract(v - 0.5) - 0.5);
    return 1.0 - smoothstep(0.3 * w, 0.9 * w, d);
  }
  void main() {
    #include <clipping_planes_fragment>
    vec3 col = vec3(0.0);
    if (mode == 0) {
      col = texture2D(map, vUv).rgb * aerialGain;
    } else if (mode == 1) {
      float z = vW.z / contourStep;
      float major = lineAt(z / 5.0);
      col = max(lineCol * lineAt(z), majorCol * major);
    } else {
      vec2 g = vW.xy / gridStep;
      float a = max(lineAt(g.x), lineAt(g.y));
      vec2 g5 = g / 5.0;
      float b = max(lineAt(g5.x), lineAt(g5.y));
      col = max(lineCol * a, majorCol * b);
    }
    float d = length(vW.xy - centre);
    col *= 1.0 - smoothstep(fade.x, fade.y, d);
    gl_FragColor = vec4(col, 1.0);
  }`;

/* The ground options (the user picks a default): A aerial ~30 %, B dark 1 m contours, C faint grid.
   Each has a laptop and a projector look (the projector lifts the ground). */
export const GROUND_LOOK = {
  laptop: { aerial: 0.3, B: [0.17, 0.26], C: [0.1, 0.17] },
  projector: { aerial: 0.42, B: [0.26, 0.38], C: [0.16, 0.26] },
};

export function makeGround(sc, heights, aerialTex) {
  const t = sc.meta.terrain, a = sc.meta.aerial.extent;
  const nx = t.nx | 0, ny = t.ny | 0, cell = +(t.cell ?? 1);
  const pos = new Float32Array(3 * nx * ny), uv = new Float32Array(2 * nx * ny);
  for (let r = 0; r < ny; r++) for (let c = 0; c < nx; c++) {
    const i = r * nx + c, x = t.x0 + (c + 0.5) * cell, y = t.y1 - (r + 0.5) * cell;
    pos[3 * i] = x; pos[3 * i + 1] = y; pos[3 * i + 2] = heights ? heights[i] : 0;
    uv[2 * i] = (x - a[0]) / (a[1] - a[0]); uv[2 * i + 1] = (y - a[2]) / (a[3] - a[2]);
  }
  const idx = new Uint32Array(6 * (nx - 1) * (ny - 1));
  let k = 0;
  for (let r = 0; r + 1 < ny; r++) for (let c = 0; c + 1 < nx; c++) {
    const i = r * nx + c, j = i + nx;       // row r is north of row r + 1: counter-clockwise from above
    idx[k++] = i; idx[k++] = j; idx[k++] = i + 1;
    idx[k++] = i + 1; idx[k++] = j; idx[k++] = j + 1;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  aerialTex.colorSpace = THREE.NoColorSpace;
  aerialTex.anisotropy = 8;
  const half = Math.min(nx, ny) * cell / 2;
  const m = new THREE.ShaderMaterial({
    vertexShader: GROUND_VS, fragmentShader: GROUND_FS, clipping: true, side: THREE.DoubleSide,
    polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
    uniforms: {
      map: { value: aerialTex }, mode: { value: 0 }, aerialGain: { value: 0.3 },
      lineCol: { value: new THREE.Color(0.16, 0.16, 0.16) }, majorCol: { value: new THREE.Color(0.3, 0.3, 0.3) },
      contourStep: { value: 1.0 }, gridStep: { value: 5.0 },
      centre: { value: new THREE.Vector2(sc.C[0], sc.C[1]) },
      fade: { value: new THREE.Vector2(Math.min(sc.R_max, half - 8), half - 1) },
    },
  });
  const mesh = new THREE.Mesh(g, m);
  mesh.renderOrder = -10;                  // drawn first
  m.depthWrite = false;                    // see-through: it never hides what is drawn after it
  // several spots' grounds overlap (08 b): each adds its lines (the brighter wins), so one's dark
  // margin never paints over another's aerial or contours; alone on black it looks the same
  m.blending = THREE.CustomBlending;
  m.blendEquation = THREE.MaxEquation;
  let mode = "B", look = "laptop";
  const apply = () => {
    const L = GROUND_LOOK[look], u = m.uniforms;
    u.mode.value = { A: 0, B: 1, C: 2 }[mode];
    u.aerialGain.value = L.aerial;
    const [lo, hi] = mode === "B" ? L.B : L.C;
    u.lineCol.value.setRGB(lo, lo, lo);
    u.majorCol.value.setRGB(hi, hi, hi);
  };
  apply();
  return {
    mesh, material: m,
    get mode() { return mode; },
    setMode(v) { mode = v; apply(); },
    setLook(v) { look = v; apply(); },
    /* The crowd below the ground (tracks.js setUnder): "same" / "dim" see through the ground; "veil"
       keeps it opaque (the crowd shows through it faintly, a second pass). */
    setUnder(v) { m.depthWrite = v === "veil"; m.polygonOffset = v !== "veil"; },   // (the offset lets a crowd just below through)
    dispose() { g.dispose(); m.dispose(); aerialTex.dispose(); },
  };
}

// ---------------------------------------------------------------------------- the cameras
/* Image corners (OpenCV camera coordinates at depth d): top-left, top-right, bottom-right,
   bottom-left, as footprint() walks the frame. */
function imageCorners(sc, d) {
  const Ki = sc.Kinv, out = [];
  for (const [u, v] of [[0, 0], [sc.W, 0], [sc.W, sc.H], [0, sc.H]]) {
    const x = Ki[0] * u + Ki[1] * v + Ki[2], y = Ki[3] * u + Ki[4] * v + Ki[5], z = Ki[6] * u + Ki[7] * v + Ki[8];
    out.push([x / z * d, y / z * d, d]);
  }
  return out;
}

function toWorld(cam, p) {       // camera coords -> world: R_wc p + C
  const R = cam.R_wc;
  return [R[0] * p[0] + R[1] * p[1] + R[2] * p[2] + cam.C[0],
          R[3] * p[0] + R[4] * p[1] + R[5] * p[2] + cam.C[1],
          R[6] * p[0] + R[7] * p[1] + R[8] * p[2] + cam.C[2]];
}

/** The camera's pose as a three.js matrix (OpenCV axes kept: x right, y down, z forward). */
export function cameraMatrix(cam, m = new THREE.Matrix4()) {
  const R = cam.R_wc, C = cam.C;
  return m.set(R[0], R[1], R[2], C[0], R[3], R[4], R[5], C[1], R[6], R[7], R[8], C[2], 0, 0, 0, 1);
}

const FP_N = 50;                  // footprint samples per side

/* A camera drawn as lines: rays from its centre to the ground corners of its view (or out to
   R_max), the image rectangle at depth d, and its footprint on the ground. */
class CameraLines {
  constructor(sc, { color, width, opacity, dashedFootprint, axes, depth }) {
    this.sc = sc;
    this.depth = depth;
    this.rgb = color;
    this.frame = new Segs(16, lineMaterial({ width, opacity, vertexColors: true }), { order: 5 });
    this.foot = new Segs(4 * FP_N + 4, lineMaterial({ width, opacity, vertexColors: true, dashed: dashedFootprint,
                                                      dash: 0.6, gap: 0.5 }), { distances: dashedFootprint, order: 5 });
    this.axes = axes ? new Segs(3, lineMaterial({ width: width * 1.2, opacity: 1, vertexColors: true }), { order: 6 }) : null;
    this.group = new THREE.Group();
    this.group.add(this.frame.obj, this.foot.obj);
    if (this.axes) this.group.add(this.axes.obj);
    this.axisTips = [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()];
  }

  update(cam) {
    const sc = this.sc, [r, g, b] = this.rgb;
    const fp = footprint(sc, cam, FP_N);
    // draped: where the view reaches past the horizon footprint() ends on the R_max circle at
    // the datum, which may lie under the terrain
    for (let i = 0; i < fp.length; i += 3) fp[i + 2] = groundHeight(sc, fp[i], fp[i + 1]);
    const corners = imageCorners(sc, this.depth).map((p) => toWorld(cam, p));
    const C = cam.C;
    this.C = [C[0], C[1], C[2]];                     // (with corners: Live's Shift+click picks the camera)
    let k = 0;
    for (let i = 0; i < 4; i++) {
      const gi = 3 * i * FP_N, q = corners[i], q2 = corners[(i + 1) % 4];
      this.frame.set(k++, C[0], C[1], C[2], q[0], q[1], q[2], r, g, b);                  // centre -> image
      this.frame.set(k++, q[0], q[1], q[2], fp[gi], fp[gi + 1], fp[gi + 2] + LIFT, r * 0.6, g * 0.6, b * 0.6,
                     r * 0.35, g * 0.35, b * 0.35);                                         // image -> ground, fading
      this.frame.set(k++, q[0], q[1], q[2], q2[0], q2[1], q2[2], r, g, b);                // the image's edge
    }
    this.frame.commit(k);
    const n = this.foot.polyline(fp, 0, r, g, b, LIFT, true);
    this.foot.commit(n);
    if (this.axes) {
      const L = Math.max(0.8, this.depth * 0.22);
      // as sheet 16: x right, y ahead, z up (the camera's own right, forward and -down)
      const ends = [[L, 0, 0], [0, 0, L], [0, -L, 0]].map((p) => toWorld(cam, p));
      ends.forEach((e, i) => {
        this.axes.set(i, C[0], C[1], C[2], e[0], e[1], e[2], r, g, b);
        this.axisTips[i].set(e[0], e[1], e[2]);
      });
      this.axes.commit(3);
    }
    this.corners = corners;
  }

  materials() { return [this.frame.material, this.foot.material, ...(this.axes ? [this.axes.material] : [])]; }
  dispose() { this.frame.dispose(); this.foot.dispose(); if (this.axes) this.axes.dispose(); }
}

// ---------------------------------------------------------------------------- the re-framed video
/* The recording stays fixed in the world; the virtual camera's image plane is a window over it.
   Pixel xv of the virtual camera looks along R_wc_v K^-1 xv; the start camera (the one that
   filmed: the video is steadied to its view, so K and R_cw_0 hold for every frame) saw that ray
   at x0 = K R_cw_0 R_wc_v K^-1 xv. Turn down -> the people slide up in the frame, turn right ->
   they slide left, as the drawn tracks do. Exact for a pure turn; a pose with a move uses the same
   turn-only mapping (one picture has no depth to move with). Row-major 3 x 3 (pixels -> pixels). */
export function reframeMatrix(sc, cam, out = new Float64Array(9)) {
  const A = mul3(sc.K, sc.R_cw), B = mul3(cam.R_wc, sc.Kinv);
  return mul3(A, B, out);
}

function mul3(a, b, o = new Float64Array(9)) {
  const r = new Float64Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++)
    r[3 * i + j] = a[3 * i] * b[j] + a[3 * i + 1] * b[3 + j] + a[3 * i + 2] * b[6 + j];
  o.set(r);
  return o;
}

/* The plane's video lookup: the plane's own pixel (uv -> xv) through uH to the recording's pixel
   x0; outside the recorded frame (or behind) nothing is drawn: the camera never filmed there. */
const REFRAME_GLSL = /* glsl */`
#ifdef USE_MAP
  vec3 hx = uH * vec3(vMapUv.x * uWH.x, (1.0 - vMapUv.y) * uWH.y, 1.0);
  if (hx.z <= 0.0) discard;
  vec2 x0 = hx.xy / hx.z;
  if (x0.x < 0.0 || x0.y < 0.0 || x0.x > uWH.x || x0.y > uWH.y) discard;
  vec4 sampledDiffuseColor = texture2D(map, vec2(x0.x / uWH.x, 1.0 - x0.y / uWH.y));
  #ifdef DECODE_VIDEO_TEXTURE
    sampledDiffuseColor = sRGBTransferEOTF(sampledDiffuseColor);
  #endif
  diffuseColor *= sampledDiffuseColor;
#endif`;

export const VIDEO_MODES = ["reframed", "glued", "off"];

/* tint: the virtual camera's violet ([r, g, b]; with several spots each has its own tint, 08 b).
   video: the blurred clip, or a canvas (a recording while it records: its newest blurred picture,
   10 step 3; setSource puts its video in once there is one). */
function sourceTexture(src) {
  const isCanvas = typeof HTMLCanvasElement !== "undefined" && src instanceof HTMLCanvasElement;
  const tex = isCanvas ? new THREE.CanvasTexture(src) : new THREE.VideoTexture(src);
  tex.colorSpace = THREE.NoColorSpace;
  tex.minFilter = THREE.LinearFilter;
  tex.generateMipmaps = false;
  return tex;
}

export function makeCameras(sc, video, { tint = VIOLET_RGB } = {}) {
  const depth = Math.min(10, Math.max(3, 0.12 * sc.R_max));
  const ghost = new CameraLines(sc, { color: [1, 1, 1], width: 1, opacity: 0.55, dashedFootprint: false, depth });
  const virt = new CameraLines(sc, { color: tint, width: 1, opacity: 0.95, dashedFootprint: true, axes: true, depth });
  // the blurred video in the virtual camera's image plane: re-framed (the recording stays put in
  // the world, the plane is a window over it) or glued (the old way: the picture turns with the hand)
  let tex = sourceTexture(video);
  const c = imageCorners(sc, depth * 1.0005);
  const pg = new THREE.BufferGeometry();
  pg.setAttribute("position", new THREE.BufferAttribute(Float32Array.from([...c[0], ...c[1], ...c[2], ...c[3]]), 3));
  pg.setAttribute("uv", new THREE.BufferAttribute(Float32Array.from([0, 1, 1, 1, 1, 0, 0, 0]), 2));
  pg.setIndex([0, 3, 1, 1, 3, 2]);
  const pm = new THREE.MeshBasicMaterial({ map: tex, transparent: true, opacity: 0.85, side: THREE.DoubleSide,
                                           depthWrite: false, toneMapped: false });
  const uH = { value: new THREE.Matrix3() }, uWH = { value: new THREE.Vector2(sc.W, sc.H) };
  pm.onBeforeCompile = (sh) => {
    sh.uniforms.uH = uH;
    sh.uniforms.uWH = uWH;
    sh.fragmentShader = "uniform mat3 uH;\nuniform vec2 uWH;\n" +
      sh.fragmentShader.replace("#include <map_fragment>", REFRAME_GLSL);
  };
  pm.customProgramCacheKey = () => "motus-reframed-video";
  const Hm = new Float64Array(9);
  let mode = "reframed", last = null;
  const setH = () => {
    if (mode === "reframed" && last) { reframeMatrix(sc, last, Hm); uH.value.set(...Hm); }
    else uH.value.identity();                      // glued: the plane's pixel is the recording's
  };
  const plane = new THREE.Mesh(pg, pm);
  plane.matrixAutoUpdate = false;
  plane.renderOrder = 4;
  ghost.update({ R_wc: sc.R_wc, R_cw: sc.R_cw, C: sc.C });
  return {
    ghost, virt, plane, depth, uH,
    get texture() { return tex; },
    group: (() => { const g = new THREE.Group(); g.add(ghost.group, virt.group, plane); return g; })(),
    update(cam) {
      virt.update(cam);
      cameraMatrix(cam, plane.matrix);
      plane.matrixWorldNeedsUpdate = true;
      last = cam;
      setH();
    },
    /** "reframed" (default), "glued" or "off". */
    setVideoMode(m) {
      mode = VIDEO_MODES.includes(m) ? m : "reframed";
      plane.visible = mode !== "off";
      setH();
    },
    get videoMode() { return mode; },
    /** Another picture in the plane (a canvas or a video). */
    setSource(src) {
      const old = tex;
      tex = sourceTexture(src);
      pm.map = tex;
      old.dispose();
    },
    materials() { return [...ghost.materials(), ...virt.materials(), pm]; },
    dispose() { ghost.dispose(); virt.dispose(); pg.dispose(); pm.dispose(); tex.dispose(); },
  };
}

// ---------------------------------------------------------------------------- the drag grid
/* Faint white lines from ground points to where the machine draws them now (02, §2): how the
   ground is dragged by the move. */
export function makeDragGrid(sc) {
  const G = dragGrid(sc, 9, [0.15, 0.3, 0.45, 0.6, 0.75, 0.9]);
  const n = G.length / 3, out = { X: new Float64Array(3 * n), state: new Uint8Array(n) };
  const lines = new Segs(n, lineMaterial({ width: 0.8, opacity: 0.38, vertexColors: true }), { order: 2 });
  const geo = new THREE.BufferGeometry();
  const dots = new Float32Array(3 * n);
  geo.setAttribute("position", new THREE.BufferAttribute(dots, 3).setUsage(THREE.DynamicDrawUsage));
  const pmat = new THREE.PointsMaterial({ color: 0xffffff, size: 2.2, sizeAttenuation: false, transparent: true,
                                          opacity: 0.5, depthWrite: false });
  const pts = new THREE.Points(geo, pmat);
  pts.frustumCulled = false;
  const group = new THREE.Group();
  group.add(lines.obj, pts);
  return {
    group, points: G,
    update(cam) {
      project(sc, cam, G, out);
      let k = 0;
      for (let i = 0; i < n; i++) {
        const j = 3 * i;
        dots[j] = G[j]; dots[j + 1] = G[j + 1]; dots[j + 2] = G[j + 2] + LIFT;
        if (out.state[i] !== DRAWN) continue;
        lines.set(k++, G[j], G[j + 1], G[j + 2] + LIFT, out.X[j], out.X[j + 1], out.X[j + 2] + LIFT,
                  0.5, 0.5, 0.5, 1, 1, 1);
      }
      lines.commit(k);
      geo.attributes.position.needsUpdate = true;
    },
    materials() { return [lines.material, pmat]; },
    dispose() { lines.dispose(); geo.dispose(); pmat.dispose(); },
  };
}
