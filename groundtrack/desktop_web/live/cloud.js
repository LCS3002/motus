/* Live view: a camera's point cloud (plans/live_controller/11; groundtrack/live_cloud.py). Only
   made and loaded while Show › Point cloud is on (off by default): with it off, nothing here runs.
   cloud.bin: records of 16 bytes, x y z f32 (the package's frame, as the camera and the tracks),
   r g b u8 (the blurred video's colour), n u8 (frames in the voxel). Drawn as points in the
   camera's group, cut by the section like the rest. */
import * as THREE from "three";

export const CLOUD_RECORD = 16;
const LOOK = {
  laptop: { size: 0.09, opacity: 0.55 },
  projector: { size: 0.13, opacity: 0.8 },
};

/** The records of cloud.bin -> {positions Float32Array, colors Uint8Array (rgb), n}. */
export function parseCloud(buf) {
  if (buf.byteLength % CLOUD_RECORD) throw new Error(`cloud.bin: ${buf.byteLength} bytes is not whole records`);
  const n = buf.byteLength / CLOUD_RECORD, dv = new DataView(buf);
  const positions = new Float32Array(3 * n), colors = new Uint8Array(3 * n);
  for (let i = 0, o = 0; i < n; i++, o += CLOUD_RECORD) {
    positions[3 * i] = dv.getFloat32(o, true);
    positions[3 * i + 1] = dv.getFloat32(o + 4, true);
    positions[3 * i + 2] = dv.getFloat32(o + 8, true);
    colors[3 * i] = dv.getUint8(o + 12); colors[3 * i + 1] = dv.getUint8(o + 13); colors[3 * i + 2] = dv.getUint8(o + 14);
  }
  return { positions, colors, n };
}

export class CloudView {
  /** parsed: parseCloud's; mode "grey" (the drafting look: light points) or "colour" (the video's). */
  constructor(parsed, mode = "grey") {
    this.mode = mode;
    this.material = new THREE.PointsMaterial({ size: LOOK.laptop.size, sizeAttenuation: true, vertexColors: true,
                                               transparent: true, opacity: LOOK.laptop.opacity, depthWrite: false });
    this.points = new THREE.Points(this._geometry(parsed), this.material);
    this.points.renderOrder = -1;                 // under the tracks
    this.points.name = "cloud";
  }

  _geometry(parsed) {
    this.n = parsed.n;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(parsed.positions, 3));
    this.rgb = new THREE.BufferAttribute(parsed.colors, 3, true);
    const grey = new Uint8Array(parsed.colors.length);
    for (let i = 0; i < parsed.n; i++) {          // its brightness, lifted: light on the black ground
      const c = parsed.colors, l = 0.3 * c[3 * i] + 0.59 * c[3 * i + 1] + 0.11 * c[3 * i + 2];
      grey[3 * i] = grey[3 * i + 1] = grey[3 * i + 2] = Math.round(120 + 0.5 * l);
    }
    this.grey = new THREE.BufferAttribute(grey, 3, true);
    g.setAttribute("color", this.mode === "colour" ? this.rgb : this.grey);
    g.computeBoundingSphere();
    return g;
  }

  /** The same layer, its points now these (Realtime's cloud as it grows). */
  update(parsed) {
    const old = this.points.geometry;
    this.points.geometry = this._geometry(parsed);
    old.dispose();
  }

  setMode(mode) {
    if (mode === this.mode) return;
    this.mode = mode;
    this.points.geometry.setAttribute("color", mode === "colour" ? this.rgb : this.grey);
  }

  setLook(look) {
    const k = LOOK[look] || LOOK.laptop;
    this.material.size = k.size;
    this.material.opacity = k.opacity;
  }

  dispose() {
    this.points.removeFromParent();
    this.points.geometry.dispose();
    this.material.dispose();
  }
}
