/* Live view: the players' panels (plans/live_controller/08 b). One per active player (a phone that
   holds a spot): the scene through that spot's virtual camera (its re-framed video and its crowd).
   Docked in a column over the stage (drawn into the same canvas, a scissored viewport each), or
   popped out to a window of its own (another screen, a projector) with its own renderer of the
   same scene. A popped panel comes back to the column when its window closes. */

import * as THREE from "three";
import { panelRects } from "./spots.js";

export class Panels {
  /* renderer: the stage's; host: the element the panels' frames go in; view(entry, camera, size)
     points camera through the entry's spot and returns what to hide while it draws (objects);
     onFocus(spot, event): a click on a panel (Shift+click: the view frames that camera). */
  constructor({ renderer, scene, host, view, onFocus }) {
    this.renderer = renderer;
    this.scene = scene;
    this.host = host;
    this.view = view;
    this.onFocus = onFocus;
    this.entries = [];            // [{key, player, spot, cam, el, popup}]
    this.rects = [];
  }

  /** The active players: [{player, spot}] (spot: the view's spot object). */
  set(list) {
    const keep = new Set(list.map((p) => `${p.player}:${p.spot.name}`));
    for (const e of this.entries.slice()) if (!keep.has(e.key)) this._remove(e);
    for (const p of list) {
      const key = `${p.player}:${p.spot.name}`;
      if (this.entries.some((e) => e.key === key)) continue;
      const e = { key, player: p.player, spot: p.spot, cam: new THREE.PerspectiveCamera(40, 16 / 9, 0.05, 4000), popup: null };
      e.el = document.createElement("div");
      e.el.className = "pp";
      e.el.title = "This player's view · click: mark its camera · Shift+click: focus";
      e.el.style.setProperty("--tint", p.spot.tint);
      e.el.innerHTML = `<span class="pp-name label"></span>
        <button class="pp-out" title="Pop out to its own window (another screen, a projector)" aria-label="Pop out">
          <svg viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1"><path d="M5 2.5H2.5v7h7V7M7 2.5h2.5V5M9.5 2.5 5.5 6.5"/></svg></button>`;
      e.el.querySelector(".pp-name").textContent = `${typeof p.player === "number" ? "P" + p.player : p.player} · ${p.spot.name}`;
      e.el.querySelector(".pp-out").onclick = (ev) => { ev.stopPropagation(); this.popOut(e); };
      e.el.onclick = (ev) => this.onFocus && this.onFocus(e.spot, ev);
      this.host.appendChild(e.el);
      this.entries.push(e);
    }
    this.entries.sort((a, b) => String(a.player).localeCompare(String(b.player), undefined, { numeric: true }));
  }

  clear() { for (const e of this.entries.slice()) this._remove(e); }

  _remove(e) {
    this.popIn(e, false);
    e.el.remove();
    this.entries.splice(this.entries.indexOf(e), 1);
  }

  get docked() { return this.entries.filter((e) => !e.popup); }

  /** The column's frames (CSS px) for a stage of W x H; layout: panelRects options. */
  layout(W, H, opts) {
    const d = this.docked, aspect = d.length ? d[0].spot.sc.W / d[0].spot.sc.H : 16 / 9;
    this.rects = panelRects(d.length, W, H, { ...opts, aspect });
    for (const e of this.entries) {
      const i = d.indexOf(e), r = this.rects[i];
      const key = e.popup || !r ? "out" : `${r.x},${r.y},${r.w},${r.h}`;
      if (key === e.at) continue;                  // (each frame: only what moved)
      e.at = key;
      if (key === "out") e.el.style.display = "none";
      else Object.assign(e.el.style, { display: "", left: `${r.x}px`, top: `${r.y}px`, width: `${r.w}px`, height: `${r.h}px` });
    }
  }

  /** Draw every panel (after the stage's own render). H: the stage's height in CSS px. */
  render(H) {
    const r0 = this.renderer;
    const d = this.docked;
    if (d.length) {
      r0.setScissorTest(true);
      d.forEach((e, i) => {
        const r = this.rects[i];
        if (!r) return;
        const y = H - r.y - r.h;
        r0.setViewport(r.x, y, r.w, r.h);
        r0.setScissor(r.x, y, r.w, r.h);
        this._draw(r0, e, r.w, r.h);
      });
      r0.setScissorTest(false);
      const size = r0.getSize(new THREE.Vector2());
      r0.setViewport(0, 0, size.x, size.y);
    }
    for (const e of this.entries) {
      const p = e.popup;
      if (!p) continue;
      if (p.win.closed) { this.popIn(e); continue; }
      const w = p.win.innerWidth, h = p.win.innerHeight;
      if (w !== p.w || h !== p.h) { p.renderer.setSize(w, h, false); p.w = w; p.h = h; }
      this._draw(p.renderer, e, w, h);
    }
  }

  _draw(renderer, e, w, h) {
    const hide = this.view(e, e.cam, w / h) || [];
    const was = hide.map((o) => o.visible);
    for (const o of hide) o.visible = false;
    renderer.render(this.scene, e.cam);
    hide.forEach((o, i) => { o.visible = was[i]; });
  }

  /* Its own window: a canvas there with a renderer of its own (the same scene; three.js keeps each
     renderer's GPU copies apart). Returns false when the window could not be opened. */
  popOut(e) {
    if (e.popup) { e.popup.win.focus(); return true; }
    const win = window.open("", `motus-panel-${e.player}-${e.spot.name}`, "width=960,height=540");
    if (!win) return false;
    const doc = win.document;
    doc.title = `motus · ${typeof e.player === "number" ? "P" + e.player : e.player} · ${e.spot.name}`;
    doc.body.style.cssText = "margin:0;background:#000;overflow:hidden";
    const canvas = doc.createElement("canvas");
    canvas.style.cssText = "display:block;width:100vw;height:100vh";
    doc.body.appendChild(canvas);
    const r = new THREE.WebGLRenderer({ canvas, antialias: true });
    r.setPixelRatio(Math.min(win.devicePixelRatio || 1, 2));
    r.outputColorSpace = THREE.LinearSRGBColorSpace;
    r.localClippingEnabled = true;
    r.setClearColor(0x000000, 1);
    e.popup = { win, renderer: r, w: 0, h: 0 };
    win.addEventListener("beforeunload", () => this.popIn(e, false));
    return true;
  }

  popIn(e, close = true) {
    const p = e.popup;
    if (!p) return;
    e.popup = null;
    p.renderer.dispose();
    if (close && !p.win.closed) p.win.close();
  }
}
