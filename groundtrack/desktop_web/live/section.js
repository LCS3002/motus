/* Live view: the section, a movable cutting plane (S). It cuts the terrain, the tracks and the
   frustums (clipping planes on their materials) and is moved / turned with TransformControls.
   Presets: vertical along the start view, vertical across it, horizontal at height h. Slab mode:
   two parallel planes, what lies between them is kept. Drawn as a faint rectangle with a crisp
   outline. The kept side is the one the plane's normal (the pivot's local +z) points to. */

import * as THREE from "three";
import { TransformControls } from "three/addons/controls/TransformControls.js";
import { Segs, lineMaterial } from "./lines.js";

const Z = new THREE.Vector3(0, 0, 1);

export class Section {
  constructor({ scene, camera, dom, orbit, sc, onChange }) {
    this.sc = sc;
    this.onChange = onChange;
    this.enabled = false;
    this.slab = false;
    this.thickness = Math.max(2, Math.round(sc.R_max * 0.15));
    this.height = 1.0;
    this.preset = "along";
    this.materials = new Set();
    this.planes = [new THREE.Plane(), new THREE.Plane()];
    this.one = [this.planes[0]];
    this.pivot = new THREE.Object3D();
    scene.add(this.pivot);
    // the visual: a faint fill and a crisp outline (two of each in slab mode)
    this.fillMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.035,
                                                 side: THREE.DoubleSide, depthWrite: false });
    this.fills = [0, 1].map(() => {
      const m = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), this.fillMat);
      m.renderOrder = 20;
      this.pivot.add(m);
      return m;
    });
    this.outline = new Segs(8, lineMaterial({ width: 1, opacity: 0.85, vertexColors: true }), { order: 21 });
    this.pivot.add(this.outline.obj);
    this.tc = new TransformControls(camera, dom);
    this.tc.setSize(0.75);
    this.tc.setSpace("local");
    // the calm look: grey handles (the normal, z, brightest), white while held; no red / green / blue
    this.tc.setColors(0x8a8a8a, 0x8a8a8a, 0xd0d0d0, 0xffffff);
    this.tc.attach(this.pivot);
    this.helper = this.tc.getHelper();
    this.helper.traverse((o) => {
      const c = o.material && o.material.color;
      if (c && c.r === 1 && c.g === 1 && c.b === 0) c.setHex(0xb0b0b0);      // the yellow rings
    });
    scene.add(this.helper);
    this.tc.addEventListener("dragging-changed", (e) => { orbit.enabled = !e.value && !orbit.locked; this.dragging = e.value; });
    this.tc.addEventListener("objectChange", () => this.update());
    this.setPreset("along", false);
    this.setEnabled(false);
  }

  /** A material to cut (its clippingPlanes follow the section). */
  register(...mats) {
    for (const m of mats.flat()) if (m) { this.materials.add(m); m.clippingPlanes = this.enabled ? this.active : null; }
  }

  /** Materials no longer cut (a camera that left the scene). */
  unregister(...mats) {
    for (const m of mats.flat()) if (m && this.materials.delete(m)) m.clippingPlanes = null;
  }

  get active() { return this.slab ? this.planes : this.one; }

  setEnabled(on) {
    this.enabled = !!on;
    this.pivot.visible = this.enabled;
    this.helper.visible = this.enabled;
    this.tc.enabled = this.enabled;
    this._apply();
    this.onChange && this.onChange();
  }

  toggle() { this.setEnabled(!this.enabled); }

  setSlab(on) { this.slab = !!on; this.update(); }
  setThickness(t) { this.thickness = Math.max(0.2, +t); this.update(); }
  setMode(m) { this.tc.setMode(m === "rotate" ? "rotate" : "translate"); }

  /* along: vertical, through the start camera, containing its view axis (keeps the right side);
     across: vertical, square to the view, 40 % of R_max ahead (keeps the far side);
     horizontal: at height h above the datum (keeps what is below). */
  setPreset(p, notify = true) {
    const sc = this.sc, C = sc.C, yaw = sc.yaw, R = sc.R_max;
    const fwd = new THREE.Vector3(Math.sin(yaw), Math.cos(yaw), 0), right = new THREE.Vector3(Math.cos(yaw), -Math.sin(yaw), 0);
    this.preset = p;
    const q = new THREE.Quaternion();
    if (p === "horizontal") {
      this.pivot.position.set(C[0] + fwd.x * R * 0.45, C[1] + fwd.y * R * 0.45, this.height);
      q.setFromUnitVectors(Z, new THREE.Vector3(0, 0, -1));
      this.size = [2 * R, 2 * R];
    } else if (p === "across") {
      this.pivot.position.set(C[0] + fwd.x * R * 0.4, C[1] + fwd.y * R * 0.4, 2);
      q.setFromUnitVectors(Z, fwd);
      this.size = [2 * R, 14];
    } else {
      this.pivot.position.set(C[0] + fwd.x * R * 0.45, C[1] + fwd.y * R * 0.45, 2);
      q.setFromUnitVectors(Z, right);
      this.size = [1.2 * R, 14];
    }
    // keep the rectangle's long side level for the vertical planes
    if (p !== "horizontal") {
      const n = new THREE.Vector3(0, 0, 1).applyQuaternion(q);
      const m = new THREE.Matrix4().lookAt(new THREE.Vector3(), n.clone().negate(), Z);
      q.setFromRotationMatrix(m);
    }
    this.pivot.quaternion.copy(q);
    this.update();
    if (notify && this.onChange) this.onChange();
  }

  setHeight(h) {
    this.height = +h;
    if (this.preset === "horizontal") { this.pivot.position.z = this.height; this.update(); }
  }

  flip() {
    this.pivot.quaternion.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI));
    this.update();
  }

  update() {
    this.pivot.updateMatrixWorld(true);
    const n = Z.clone().applyQuaternion(this.pivot.quaternion).normalize();
    const p = this.pivot.position.clone(), h = this.slab ? this.thickness / 2 : 0;
    if (this.preset === "horizontal") this.height = p.z;
    // three.js keeps what lies on the positive side of each plane
    this.planes[0].setFromNormalAndCoplanarPoint(n, p.clone().addScaledVector(n, -h));
    this.planes[1].setFromNormalAndCoplanarPoint(n.clone().negate(), p.clone().addScaledVector(n, h));
    const [w, hh] = this.size;
    const zs = this.slab ? [-h, h] : [0];
    let k = 0;
    this.fills.forEach((m, i) => {
      m.visible = i < zs.length;
      m.scale.set(w, hh, 1);
      m.position.set(0, 0, zs[i] ?? 0);
    });
    for (const z of zs) {
      const c = [[-w / 2, -hh / 2], [w / 2, -hh / 2], [w / 2, hh / 2], [-w / 2, hh / 2]];
      for (let i = 0; i < 4; i++) {
        const a = c[i], b = c[(i + 1) % 4];
        this.outline.set(k++, a[0], a[1], z, b[0], b[1], z, 1, 1, 1);
      }
    }
    this.outline.commit(k);
    this._apply();
  }

  _apply() {
    const planes = this.enabled ? this.active : null;
    for (const m of this.materials) {
      if (m.clippingPlanes !== planes) { m.clippingPlanes = planes; m.needsUpdate = true; }
    }
  }

  /** For the phone's state. */
  get state() { return this.enabled; }

  dispose() { this.tc.detach(); this.tc.dispose(); this.fillMat.dispose(); this.outline.dispose(); }
}
