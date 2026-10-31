/* motus, the start menu (30 Oct, the user: "a motus live button right when i start, a cinematic
   main menu"). Shown when the app opens and on the way back from Live or Realtime (not when
   Live goes back to a run: sessionStorage gt.run), over the workspace; the logo shows it again.
   Behind it a site's own tracks draw themselves in, frame by frame, on a plan seen low: the start
   camera's package of the site Live opens (the one used last), its points in the speed ramp as
   Live draws them, the machine's guesses faint white; at the end it holds, fades and starts
   again. Nothing invented: no site with a package, no lines (a faint grid only).
   Live: /live?site=<the site>; Realtime: /live?realtime=1; Recordings: /live/realtime;
   Workspace: the app under it (the address keeps #workspace, so a reload stays there). */

import { rampColor } from "/static/live/projection.js";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const enc = encodeURIComponent;
const root = document.documentElement;
const intro = $("#intro");
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
const winApi = () => window.pywebview && window.pywebview.api;

const LOOP_S = 48;           // a clip longer than this is played faster (at most MAX_RATE x)
const MAX_RATE = 12;
const HOLD_S = 3, FADE_S = 1.8;
const MAX_GAP = 2;           // frames: rows of a track further apart are not joined (as Live)
const BINS = 32;             // colours of the ramp, batched per frame
const HEAD_S = 0.6;          // a track's head shows this long after its last point

let sites = [], si = 0, D = null, raf = 0, shown = false, loadTok = 0;

// ---------------------------------------------------------------------------- the site's tracks
async function loadSite(e) {
  const run = enc(e.start_run), get = async (f, how) => {
    const r = await fetch(`/live/pkg/${f}?run=${run}`);
    if (!r.ok) throw new Error(`${f}: ${r.status}`);
    return r[how]();
  };
  const meta = await get("scene.json", "json");
  const buf = await get("tracks.bin", "arrayBuffer");
  let img = null;
  if (meta.aerial && !meta.aerial.blank) {
    img = new Image();
    img.src = `/live/pkg/aerial.jpg?run=${run}`;
    try { await img.decode(); } catch { img = null; }
  }
  return build(meta, buf, img, e);
}

const pct = (a, q) => { const s = Float32Array.from(a).sort(); return s.length ? s[Math.min(s.length - 1, Math.floor(q * (s.length - 1)))] : 0; };

function build(meta, buf, img, site) {
  const dv = new DataView(buf), n = (buf.byteLength / 30) | 0;
  const tid = new Uint32Array(n), frame = new Uint32Array(n), x = new Float32Array(n), y = new Float32Array(n);
  const bin = new Uint8Array(n), prev = new Int32Array(n).fill(-1);
  const ramp = meta.speed_ramp || {}, lo = ramp.lo || 1, hi = ramp.hi || 40, last = new Map(), c = new Float32Array(3);
  for (let i = 0; i < n; i++) {
    const o = 30 * i;
    tid[i] = dv.getUint32(o, true); frame[i] = dv.getUint32(o + 4, true);
    x[i] = dv.getFloat32(o + 12, true); y[i] = dv.getFloat32(o + 16, true);
    const speed = dv.getFloat32(o + 24, true), src = dv.getUint8(o + 29);
    // the bin: the speed ramp (log lo..hi) for a seen point, BINS for one of the machine's guesses
    const v = Math.log(Math.max(speed, 1e-6) / lo) / Math.log(hi / lo);
    bin[i] = src ? BINS : Math.min(BINS - 1, Math.max(0, Math.floor(v * BINS)));
    const p = last.get(tid[i]);
    if (p != null && frame[i] - frame[p] <= MAX_GAP && frame[i] > frame[p]) prev[i] = p;
    last.set(tid[i], i);
  }
  const colours = [], glows = [];                  // each line twice: a soft glow, then its core
  for (let b = 0; b < BINS; b++) {
    rampColor(lo * Math.pow(hi / lo, (b + 0.5) / BINS), c, 0, lo, hi);
    const rgb = `${Math.round(c[0] * 255)} ${Math.round(c[1] * 255)} ${Math.round(c[2] * 255)}`;
    colours.push(`rgb(${rgb} / 0.92)`);
    glows.push(`rgb(${rgb} / 0.1)`);
  }
  colours.push("rgb(255 255 255 / 0.2)");
  glows.push("rgb(255 255 255 / 0.025)");
  // the frame of the plan: the middle 96 % of the points, a margin, at least 30 m
  let x0 = pct(x, 0.02), x1 = pct(x, 0.98), y0 = pct(y, 0.02), y1 = pct(y, 0.98);
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, half = Math.max(15, 0.62 * Math.max(x1 - x0, y1 - y0));
  x0 = cx - half; x1 = cx + half; y0 = cy - half; y1 = cy + half;
  const TW = 2048, trail = document.createElement("canvas");
  trail.width = trail.height = TW;
  const tctx = trail.getContext("2d");
  tctx.lineCap = "round";
  tctx.lineJoin = "round";
  const frames = Math.max(meta.frames || 0, n ? frame[n - 1] + 1 : 0), fps = meta.fps || 30;
  const dur = frames / fps, rate = Math.min(MAX_RATE, Math.max(1, dur / LOOP_S));
  let ground = null;
  if (img && meta.aerial.extent) {                 // the aerial, grey and low, in the same frame
    const [ax0, ax1, ay0, ay1] = meta.aerial.extent;
    ground = document.createElement("canvas");
    ground.width = ground.height = 1024;
    const g = ground.getContext("2d"), k = 1024 / (x1 - x0);
    g.filter = "grayscale(1) brightness(0.5) contrast(1.1)";
    g.globalAlpha = 0.34;
    g.drawImage(img, (ax0 - x0) * k, (y1 - ay1) * k, (ax1 - ax0) * k, (ay1 - ay0) * k);
  }
  const names = (meta.tracks && meta.tracks.classes) || {};
  return { site, meta, n, tid, frame, x, y, bin, prev, colours, glows, x0, y0, size: x1 - x0, TW, trail, tctx, ground,
           frames, fps, dur, rate, at: 0, clock: 0, phase: "draw", phaseT: 0, heads: new Map(),
           nTracks: (meta.tracks && meta.tracks.n_tracks) || new Set(tid).size, kinds: Object.keys(names).length };
}

/** The points up to frame f: their segments onto the trail, batched by colour. */
function drawTo(f) {
  const d = D, k = d.TW / d.size, paths = [];
  let i = d.at;
  for (; i < d.n && d.frame[i] <= f; i++) {
    d.heads.set(d.tid[i], i);
    const p = d.prev[i];
    if (p < 0) continue;
    (paths[d.bin[i]] ||= []).push(p, i);
  }
  d.at = i;
  const ctx = d.tctx, w = d.TW / 1150;
  ctx.globalCompositeOperation = "lighter";
  paths.forEach((ps, b) => {
    if (!ps) return;
    ctx.beginPath();
    for (let j = 0; j < ps.length; j += 2) {
      const a = ps[j], c = ps[j + 1];
      ctx.moveTo((d.x[a] - d.x0) * k, d.TW - (d.y[a] - d.y0) * k);
      ctx.lineTo((d.x[c] - d.x0) * k, d.TW - (d.y[c] - d.y0) * k);
    }
    ctx.strokeStyle = d.glows[b];
    ctx.lineWidth = 3 * w;
    ctx.stroke();
    ctx.strokeStyle = d.colours[b];
    ctx.lineWidth = w;
    ctx.stroke();
  });
}

function restart() {
  const d = D;
  d.tctx.globalCompositeOperation = "source-over";
  d.tctx.clearRect(0, 0, d.TW, d.TW);
  d.at = 0; d.clock = 0; d.heads.clear(); d.phase = "draw"; d.phaseT = 0;
}

// ---------------------------------------------------------------------------- the frame
const gl = $("#intro-gl"), g2 = gl.getContext("2d");
let W = 0, H = 0, t0 = 0, tPrev = 0, readT = 0;

function resize() {                               // its layout size (the tilt is only shown)
  W = gl.width = gl.offsetWidth;
  H = gl.height = gl.offsetHeight;
}

function grid(t) {                                 // no site: a faint drafting grid, drifting
  g2.strokeStyle = "rgb(255 255 255 / 0.07)";
  g2.lineWidth = 1;
  const step = Math.max(W, H) / 24, off = (t * 6) % step;
  g2.beginPath();
  for (let x = -step + off; x < W + step; x += step) { g2.moveTo(x, 0); g2.lineTo(x, H); }
  for (let y = -step + off; y < H + step; y += step) { g2.moveTo(0, y); g2.lineTo(W, y); }
  g2.stroke();
}

function frame(now) {
  raf = requestAnimationFrame(frame);
  if (!t0) t0 = tPrev = now;
  const t = (now - t0) / 1000, dt = Math.min(0.1, (now - tPrev) / 1000);
  tPrev = now;
  g2.setTransform(1, 0, 0, 1, 0, 0);
  g2.globalAlpha = 1;
  g2.clearRect(0, 0, W, H);
  if (!D) { grid(t); return; }
  const d = D;
  // the clock: drawing, holding at the end, fading out, again
  if (d.phase === "draw") {
    d.clock += dt * d.rate;
    drawTo(Math.floor(d.clock * d.fps));
    if (d.clock >= d.dur) { d.phase = "hold"; d.phaseT = 0; }
  } else {
    d.phaseT += dt;
    if (d.phase === "hold" && d.phaseT > HOLD_S) { d.phase = "fade"; d.phaseT = 0; }
    else if (d.phase === "fade" && d.phaseT > FADE_S) restart();
  }
  const fade = d.phase === "fade" ? 1 - d.phaseT / FADE_S : 1;
  // the camera: the plan's middle a little right of the frame's, breathing, turning slowly
  const s = (Math.min(W, H) / d.size) * (1.12 + 0.06 * Math.sin(t * 2 * Math.PI / 70));
  const rot = -0.22 + 0.07 * Math.sin(t * 2 * Math.PI / 110);
  g2.translate(W * 0.56, H * 0.54);
  g2.rotate(rot);
  g2.scale(s, s);
  g2.translate(-d.size / 2, -d.size / 2);
  if (d.ground) g2.drawImage(d.ground, 0, 0, d.size, d.size);
  else {
    g2.strokeStyle = "rgb(255 255 255 / 0.06)";
    g2.lineWidth = 1 / s;
    g2.beginPath();
    for (let m = 0; m <= d.size; m += 10) { g2.moveTo(m, 0); g2.lineTo(m, d.size); g2.moveTo(0, m); g2.lineTo(d.size, m); }
    g2.stroke();
  }
  g2.globalAlpha = fade;
  g2.drawImage(d.trail, 0, 0, d.size, d.size);
  // the heads: the tracks drawn in the last moments
  const f = d.clock * d.fps, keep = HEAD_S * d.fps, r = 2.2 / s;
  g2.fillStyle = "rgb(255 255 255 / 0.9)";
  g2.beginPath();
  for (const [id, i] of d.heads) {
    if (f - d.frame[i] > keep) { d.heads.delete(id); continue; }
    const px = d.x[i] - d.x0, py = d.size - (d.y[i] - d.y0);
    g2.moveTo(px + r, py);
    g2.arc(px, py, r, 0, 2 * Math.PI);
  }
  g2.fill();
  if (now - readT > 120) { readT = now; readout(); }
}

const fmt = (s) => { s = Math.max(0, s); const m = Math.floor(s / 60); return `${String(m).padStart(2, "0")}:${(s - 60 * m).toFixed(1).padStart(4, "0")}`; };

function readout() {
  const el = $("#intro-read");
  if (!D) { el.innerHTML = ""; return; }
  const d = D, cam = d.site.start_id || "";
  el.innerHTML = `<span>Site <b></b></span><span>Camera <b></b></span><span>T <b></b></span><span><b></b> tracks</span><span><b></b> points</span><span>&times;<b></b></span>`;
  const b = $$("b", el);
  b[0].textContent = d.site.title;
  b[1].textContent = cam;
  b[2].textContent = `${fmt(Math.min(d.clock, d.dur))} / ${fmt(d.dur)}`;
  b[3].textContent = d.nTracks.toLocaleString("en-GB");
  b[4].textContent = d.n.toLocaleString("en-GB");
  b[5].textContent = d.rate.toFixed(d.rate < 10 ? 1 : 0);
}

// ---------------------------------------------------------------------------- the menu
const rows = () => $$(".intro-row").filter((r) => !r.hidden);
let selI = 0;

function select(i) {
  const rs = rows();
  selI = (i + rs.length) % rs.length;
  rs.forEach((r, k) => r.classList.toggle("sel", k === selI));
}

function siteLine() {
  const e = sites[si], item = $('[data-go="live"]');
  $("#intro-site").textContent = e ? `${e.title} · ${e.cameras} camera${e.cameras === 1 ? "" : "s"}` : "No site with a Live package yet: track a clip in the workspace";
  item.disabled = !e;
  $(".intro-cycle").hidden = sites.length < 2;
}

async function setSite(i) {
  if (!sites.length) return;
  si = (i + sites.length) % sites.length;
  siteLine();
  const tok = ++loadTok;
  try {
    const d = await loadSite(sites[si]);
    if (tok !== loadTok) return;
    D = d;
    if (reduced) { drawTo(Infinity); D.clock = D.dur; D.phase = "hold"; D.phaseT = -1e9; }
  } catch { if (tok === loadTok) D = null; }
  readout();
}

/** Away to another page: the letterbox shuts first. */
function leave(url) {
  try { sessionStorage.removeItem("gt.run"); } catch {}
  intro.classList.add("leaving");
  setTimeout(() => { location.href = url; }, reduced ? 0 : 560);
}

function go(what) {
  if (what === "live") { const e = sites[si]; if (e) leave(`/live?site=${enc(e.site)}`); }
  else if (what === "realtime") leave("/live?realtime=1");
  else if (what === "recordings") leave("/live/realtime");
  else if (what === "workspace") hide();
  else if (what === "quit") { const api = winApi(); if (api) api.close(); }
}

function onKey(e) {
  if (!shown) return;
  const k = e.key;
  if (k === "ArrowDown") select(selI + 1);
  else if (k === "ArrowUp") select(selI - 1);
  else if (k === "ArrowLeft") setSite(si - 1);
  else if (k === "ArrowRight") setSite(si + 1);
  else if (k === "Enter" || k === " ") { const b = rows()[selI].querySelector(".intro-item"); if (!b.disabled) go(b.dataset.go); }
  else if (k === "Escape") go("workspace");
  else if (/^[1-5]$/.test(k)) { const r = rows()[+k - 1]; if (r) { select(+k - 1); const b = r.querySelector(".intro-item"); if (!b.disabled) go(b.dataset.go); } }
  else return;
  e.preventDefault();
  e.stopImmediatePropagation();                    // not the workspace's keys too
}

// ---------------------------------------------------------------------------- show / hide
function show() {
  shown = true;
  root.classList.add("intro-on");
  intro.classList.remove("out", "leaving", "lit");
  history.replaceState(null, "", location.pathname + location.search);
  resize();
  t0 = 0;
  cancelAnimationFrame(raf);
  raf = requestAnimationFrame(frame);
  requestAnimationFrame(() => requestAnimationFrame(() => intro.classList.add("lit")));
  select(0);
  $('[data-go="live"]').focus({ preventScroll: true });
}

function hide() {
  shown = false;
  history.replaceState(null, "", "#workspace");
  intro.classList.add("out");
  setTimeout(() => {
    if (shown) return;
    cancelAnimationFrame(raf);
    root.classList.remove("intro-on");
    intro.classList.remove("out", "lit");
  }, reduced ? 0 : 680);
}

function bind() {
  rows().forEach((r, i) => {
    r.style.transitionDelay = `${900 + i * 90}ms`;
    r.addEventListener("pointerenter", () => select(rows().indexOf(r)));
  });
  $$(".intro-item").forEach((b) => b.onclick = () => go(b.dataset.go));
  $$("[data-site]").forEach((b) => b.onclick = (e) => { e.stopPropagation(); setSite(si + +b.dataset.site); });
  window.addEventListener("keydown", onKey, true);
  window.addEventListener("resize", () => { if (shown) resize(); });
  const quit = () => { $('[data-go="quit"]').closest(".intro-row").hidden = !winApi(); };
  quit();
  window.addEventListener("pywebviewready", quit);
  $$("[data-intro]").forEach((b) => b.addEventListener("click", (e) => { e.preventDefault(); show(); }));
}

(async function start() {
  bind();
  if (root.classList.contains("intro-on")) show();
  try { sites = (await (await fetch("/api/live/sites")).json()).sites || []; } catch { sites = []; }
  let saved = null;
  try { saved = localStorage.getItem("motus-live-site"); } catch {}
  // the site Live opened last; else the one with the most cameras
  let i = sites.findIndex((e) => e.site === saved);
  if (i < 0) sites.forEach((e, k) => { if (i < 0 || e.cameras > sites[i].cameras) i = k; });
  siteLine();
  await setSite(Math.max(0, i));
})();

window.gtIntro = { show, hide };
