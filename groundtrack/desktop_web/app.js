"use strict";
/* groundtrack desktop app: the page. Everything goes through the local server
   (groundtrack/desktop.py on top of groundtrack/ui.py); nothing leaves the computer except
   a place search, which goes to OpenStreetMap. */

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const base = (p) => String(p || "").split(/[\\/]/).pop();
const dir = (p) => String(p || "").replace(/[\\/][^\\/]*$/, "");
const thumb = (p, w = 480) => `/file?path=${encodeURIComponent(p)}${w ? `&w=${w}` : ""}`;
const enc = encodeURIComponent;

/* ------------------------------------------------------------------ motion */
const reduced = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
// page changes morph (panels glide and resize, contents crossfade) where the browser can
function transition(fn) {
  if (!document.startViewTransition || reduced() || document.hidden) { fn(); return; }
  try { document.startViewTransition(fn); } catch { fn(); }
}
// only things that are new animate in: polling re-renders must not replay entrances
const seenKeys = new Set();
const fresh = (key) => (seenKeys.has(key) ? false : (seenKeys.add(key), true));
// a closing element plays its .out animation, then hides
function leave(el, done, ms = 200) {
  if (reduced()) { done(); return; }
  el.classList.add("out");
  const tok = (el._leaveTok = (el._leaveTok || 0) + 1);
  setTimeout(() => { if (el._leaveTok === tok) { el.classList.remove("out"); done(); } }, ms);
}
// the tab switcher's sliding pill
function segThumb(seg) {
  if (!seg) return;
  let t = seg.querySelector(".seg-thumb");
  if (!t) {
    t = document.createElement("span"); t.className = "seg-thumb"; seg.prepend(t); seg.classList.add("has-thumb");
    new ResizeObserver(() => segThumb(seg)).observe(seg);
  }
  const b = seg.querySelector('[aria-selected="true"]');
  if (!b || !b.offsetWidth) return;
  t.style.width = `${b.offsetWidth}px`; t.style.height = `${b.offsetHeight}px`;
  t.style.transform = `translateX(${b.offsetLeft}px)`;
}
// numbers count to their new value
function tween(el, from, to, fmt, ms = 380) {
  if (!el || from == null || to == null || !isFinite(from) || !isFinite(to) || from === to || reduced()) return;
  const t0 = performance.now();
  const step = (t) => {
    const k = Math.min(1, (t - t0) / ms), e = 1 - Math.pow(1 - k, 3);
    el.textContent = fmt(from + (to - from) * e);
    if (k < 1) requestAnimationFrame(step);
  };
  el.textContent = fmt(from);
  requestAnimationFrame(step);
}
// images fade in when they arrive (CSS hides them until .loaded)
for (const ev of ["load", "error"]) {
  document.addEventListener(ev, (e) => { if (e.target && e.target.tagName === "IMG") e.target.classList.add("loaded"); }, true);
}

Object.assign(ICON, {
  play: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5.5v13l10.5-6.5z"/></svg>',
  target: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3"/><path d="M12 1v4M12 19v4M1 12h4M19 12h4"/></svg>',
  area: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"><path d="M4 18 7 5l11 3 2 10z"/></svg>',
  layers: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"><path d="m12 3 9 5-9 5-9-5z"/><path d="m3 13 9 5 9-5"/></svg>',
  stack: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"><rect x="3" y="7" width="14" height="14"/><path d="M7 3h14v14"/></svg>',
  swirl: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><path d="M12 12a2 2 0 1 1 2-2c0 3-2.7 5-5.5 5S3.5 12.5 4 9.5C4.6 5.6 8 3 12 3c5 0 9 4 9 9s-4 9-9 9"/></svg>',
  cube: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"><path d="m12 2 9 5v10l-9 5-9-5V7z"/><path d="m3 7 9 5 9-5M12 12v10"/></svg>',
  folder: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"><path d="M3 5h6l2 2h10v13H3z"/></svg>',
  copy: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"><rect x="8" y="8" width="13" height="13"/><path d="M16 8V3H3v13h5"/></svg>',
  open: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M14 4h6v6M20 4l-9 9M18 14v6H4V6h6"/></svg>',
  stop: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="6.5" y="6.5" width="11" height="11"/></svg>',
  plus: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><path d="M12 5v14M5 12h14"/></svg>',
  image: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"><rect x="3" y="4" width="18" height="16"/><circle cx="9" cy="10" r="2"/><path d="m21 16-5-5-11 9"/></svg>',
  hammer: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M14 6 18 10M3 21l9-9M13 4l7 7-3 3-7-7z"/></svg>',
  file: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"><path d="M14 3H5v18h14V8z"/><path d="M14 3v5h5"/></svg>',
});

const STEPS = [
  { id: 1, label: "Footage & aerial" }, { id: 2, label: "Camera" },
  { id: 3, label: "Calibrate", focus: true }, { id: 4, label: "Area", focus: true },
  { id: 5, label: "Track" }, { id: 6, label: "Results" },
  { id: 7, label: "Combine" }, { id: 8, label: "Houdini & Grasshopper" },
];

const A = {
  root: "", sites: [], jobs: [], device: null, running: {},
  config: null, site: null, siteError: null, step: 1, prevStep: 5,
  links: null, sources: null, composites: [], run: null, runDetail: null,
  tab: "jobs", openJob: null, logs: {}, watchers: {}, seen: {},
  mounted: null, search: "", camMap: null, preset: {},
};

/* ------------------------------------------------------------------ api */
async function req(path, opts) {
  const r = await fetch(path, opts);
  const d = await r.json().catch(() => ({}));
  if (!r.ok || d.error) throw Object.assign(new Error(d.error || `${r.status} ${r.statusText}`), { status: r.status, data: d });
  return d;
}
const get = (p) => req(p);
const post = (p, b) => req(p, { method: "POST", headers: { "Content-Type": "application/json", "X-Groundtrack": "1" }, body: JSON.stringify(b || {}) });

async function startJob(body, confirmText) {
  try {
    const j = await post("/api/job", body);
    jobStarted(j); return j;
  } catch (e) {
    if (e.data && e.data.confirm && await ask("Are you sure?", e.message, confirmText || "Go ahead")) {
      const j = await post("/api/job", { ...body, options: { ...(body.options || {}), force: true } });
      jobStarted(j); return j;
    }
    if (!(e.data && e.data.confirm)) toast(e.message, "error");
    return null;
  }
}

function jobStarted(j) {
  if (!j) return;
  A.jobs = [j, ...A.jobs.filter((x) => x.id !== j.id)];
  A.openJob = j.id;
  if (!isFocus()) setTab("jobs");
  renderJobs(); renderBar();
  toast(`Started: ${j.label}`);
}

function watch(job, cb) { if (job) A.watchers[job.id] = cb; }

async function pickPath(kind, what) {
  const api = window.pywebview && window.pywebview.api;
  if (!api) { toast("File dialogs need the desktop window: type or paste the path", "info"); return null; }
  try { return await api.pick(kind, what || ""); } catch (e) { toast(e.message || String(e), "error"); return null; }
}

async function copyText(text, what = "Copied") {
  try { await navigator.clipboard.writeText(text); }
  catch {
    const t = document.createElement("textarea"); t.value = text; t.style.position = "fixed"; t.style.opacity = "0";
    document.body.appendChild(t); t.select(); document.execCommand("copy"); t.remove();
  }
  toast(what, "ok");
}

const openFolder = (p) => post("/api/open", { path: p }).catch((e) => toast(e.message, "error"));
const openFile = (p) => post("/api/app/open-file", { path: p }).then(() => toast(`Opening ${base(p)}…`)).catch((e) => toast(e.message, "error"));

/* ------------------------------------------------------------------ small ui */
function toast(msg, kind = "") {
  const t = document.createElement("div");
  t.className = `toast ${kind}`; t.textContent = msg;
  $("#toasts").appendChild(t);
  setTimeout(() => leave(t, () => t.remove(), 220), kind === "error" ? 6500 : 3600);
}

function ask(title, text, ok = "OK") {
  return new Promise((resolve) => {
    const m = $("#modal");
    $("#modal-title").textContent = title; $("#modal-text").textContent = text || "";
    $("#modal-ok").textContent = ok;
    m._leaveTok = (m._leaveTok || 0) + 1; m.classList.remove("out");
    m.hidden = false; $("#modal-ok").focus();
    const done = (v) => { $("#modal-ok").onclick = $("#modal-cancel").onclick = null; A.modalDone = null; leave(m, () => { m.hidden = true; }); resolve(v); };
    A.modalDone = done;
    $("#modal-ok").onclick = () => done(true);
    $("#modal-cancel").onclick = () => done(false);
  });
}

function ago(ts) {
  if (!ts) return "–";
  const s = Date.now() / 1000 - ts;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  if (s < 86400 * 30) return `${Math.round(s / 86400)} d ago`;
  return new Date(ts * 1000).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
}
function dur(s) { if (s == null || !isFinite(s)) return "–"; s = Math.round(s); const m = Math.floor(s / 60); return m >= 60 ? `${Math.floor(m / 60)}:${String(m % 60).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}` : `${m}:${String(s % 60).padStart(2, "0")}`; }
function elapsed(j) { return dur((j.ended || Date.now() / 1000) - j.started); }
const family = (n) => (String(n || "").replace(/[\d_.-]+$/, "") || String(n || "site"));
const famLabel = (f) => f.charAt(0).toUpperCase() + f.slice(1);
function hashIdx(s, n = 5) { let h = 0; for (const c of String(s)) h = (h * 31 + c.charCodeAt(0)) | 0; return Math.abs(h) % n; }
function initials(n) { const f = family(n); const d = String(n).slice(f.length).replace(/\D/g, ""); return (f.slice(0, 2) + (d ? d.slice(-2) : "")).slice(0, 4).toUpperCase(); }
function rmseClass(v) { return v == null ? "" : v <= 0.3 ? "good" : v <= 0.6 ? "warn" : "bad"; }
function checkChip(c) {
  if (!c || c.implied_person_height_m == null) return "";
  const e = Math.abs(c.scale_error_pct ?? 0), k = e <= 10 ? "ok" : e <= 20 ? "warn" : "bad";
  return `<span class="chip ${k}" title="People would be ${c.implied_person_height_m} m tall under this calibration (${c.scale_error_pct > 0 ? "+" : ""}${c.scale_error_pct}%)">people ${(+c.implied_person_height_m).toFixed(2)} m</span>`;
}
const percent = (s) => { const m = /(\d{1,3}(?:\.\d+)?)\s*%/.exec(s || ""); return m ? Math.min(100, +m[1]) : null; };
const isFocus = () => !!(STEPS.find((s) => s.id === A.step) || {}).focus && !!A.config;

/* ------------------------------------------------------------------ lightbox */
const LB = { items: [], i: 0 };
function lightbox(items, i = 0) {
  LB.items = items; LB.i = i;
  const lb = $("#lightbox");
  lb._leaveTok = (lb._leaveTok || 0) + 1; lb.classList.remove("out");
  lb.hidden = false; showLB();
}
async function showLB() {
  const it = LB.items[LB.i]; if (!it) return;
  const ext = base(it.path).split(".").pop().toLowerCase();
  const kind = it.kind || (["mp4", "mov", "webm"].includes(ext) ? "video" : ["png", "jpg", "jpeg"].includes(ext) ? "image" : "text");
  $("#lb-title").textContent = `${it.title || base(it.path)}${LB.items.length > 1 ? `  ·  ${LB.i + 1} / ${LB.items.length}` : ""}`;
  const st = $("#lb-stage");
  if (kind === "image") st.innerHTML = `<img src="${thumb(it.path, 0)}" alt="">`;
  else if (kind === "video") st.innerHTML = `<video src="${thumb(it.path, 0)}" controls autoplay></video>`;
  else {
    st.innerHTML = '<pre>Loading…</pre>';
    try { const t = await get(`/api/app/text?path=${enc(it.path)}`); $("pre", st).textContent = t.text; }
    catch (e) { $("pre", st).textContent = e.message; }
  }
  $("#lb-prev").hidden = $("#lb-next").hidden = LB.items.length < 2;
}
function closeLB() {
  const lb = $("#lightbox"); if (lb.hidden) return;
  leave(lb, () => { lb.hidden = true; $("#lb-stage").innerHTML = ""; });
}
$("#lb-close").onclick = closeLB;
$("#lb-prev").onclick = () => { LB.i = (LB.i - 1 + LB.items.length) % LB.items.length; showLB(); };
$("#lb-next").onclick = () => { LB.i = (LB.i + 1) % LB.items.length; showLB(); };
$("#lb-folder").onclick = () => { const it = LB.items[LB.i]; if (it) openFolder(it.path); };

/* ------------------------------------------------------------------ drawer */
function drawer(title, html, after) {
  const d = $("#drawer"), s = $("#scrim");
  d._leaveTok = s._leaveTok = (d._leaveTok || 0) + 1;           // cancels a closing in progress
  d.classList.remove("out"); s.classList.remove("out");
  $("#drawer-title").textContent = title;
  $("#drawer-body").innerHTML = html;
  d.hidden = s.hidden = false;
  if (after) after($("#drawer-body"));
  segThumb($(".seg", d));
}
function closeDrawer() {
  const d = $("#drawer"), s = $("#scrim");
  if (d.hidden) return;
  leave(s, () => { s.hidden = true; });
  leave(d, () => { d.hidden = true; $("#drawer-body").innerHTML = ""; });
}
$("#drawer-close").onclick = closeDrawer;
$("#scrim").onclick = closeDrawer;

/* ------------------------------------------------------------------ data */
async function loadOverview() {
  const d = await get("/api/app/overview");
  A.root = d.root; A.device = d.device; A.running = d.running || {};
  const before = JSON.stringify(A.sites);
  A.sites = d.sites;
  const prev = new Map(A.jobs.map((j) => [j.id, j]));
  A.jobs = d.jobs.slice().sort((a, b) => b.started - a.started);
  for (const j of A.jobs) {
    const p = prev.get(j.id);
    if (p && p.status === "running" && j.status !== "running") jobFinished(j);
  }
  if (before !== JSON.stringify(A.sites)) renderSites();
  renderBar(); renderJobs();
  if (A.step === 5 && !isFocus()) renderBoard();
}

async function jobFinished(j) {
  const ok = j.status === "done";
  toast(`${ok ? "Finished" : j.status === "stopped" ? "Stopped" : "Failed"}: ${j.label}`, ok ? "ok" : j.status === "stopped" ? "" : "error");
  const cb = A.watchers[j.id]; delete A.watchers[j.id];
  if (A.config) await refreshSite().catch(() => {});
  if (["houdini-pop", "houdini-build", "composite"].includes(j.kind)) { A.links = null; A.sources = null; }
  if (cb) { try { await cb(j); } catch (e) { console.error(e); } }
  if (!A.mounted) renderWork();
}

async function refreshSite(render = true) {
  if (!A.config) { A.site = null; return; }
  try { A.site = await get(`/api/site?config=${enc(A.config)}`); A.siteError = null; }
  catch (e) { A.site = null; A.siteError = e.message; }
  if (render) { renderHero(); renderSteps(); renderSitePane(); }
}

async function loadLinks(refresh = false) {
  if (!A.links || refresh) A.links = await get(`/api/app/links${refresh ? "?refresh=1" : ""}`);
  return A.links;
}

/* ------------------------------------------------------------------ bar */
function renderBar() {
  $("#project").textContent = A.root;
  const dp = $("#device-pill"), d = A.device || {};
  dp.className = "pill" + (d.device && d.device.startsWith("cuda") ? " gpu" : d.device ? " cpu" : "");
  $("span", dp).textContent = d.state === "checking" ? "Checking GPU…" : d.gpu ? d.gpu.replace(/^NVIDIA (GeForce )?/, "") : d.device === "mps" ? "Apple GPU" : d.device ? "CPU only" : "No device";
  dp.title = d.text || "";
  const running = A.jobs.filter((j) => j.status === "running");
  const jp = $("#jobs-pill");
  jp.hidden = !running.length;
  jp.innerHTML = `<i class="dot"></i>${running.length} running`;
  $("#n-jobs").textContent = A.jobs.length ? ` ${A.jobs.length}` : "";
  $("#menu-status").innerHTML = `<span>Project <b>${esc(A.root)}</b></span><span>${esc((d.text || "").split("\n").slice(-1)[0] || "")}</span>`;
}
$("#jobs-pill").onclick = () => { if (isFocus()) goto(A.prevStep || 5); setTab("jobs"); };

/* ------------------------------------------------------------------ hero */
function renderHero() {
  const el = $("#hero");
  const s = A.site;
  if (!s) {
    const cal = A.sites.filter((x) => x.calibrated).length, runs = A.sites.reduce((n, x) => n + (x.runs || 0), 0);
    el.innerHTML = `<div style="min-width:0"><div class="hero-label">Project</div><div class="hero-value">${esc(base(A.root) || "motus")}</div></div>
      <div class="stats">${stat("Sites", A.sites.length)}${stat("Calibrated", cal)}${stat("Runs", runs)}</div>
      <div class="actions">${act("new-site", ICON.plus, "New site", true)}</div>`;
    bindActs(el); return;
  }
  const v = s.video || {}, c = s.calibration || {}, last = (s.runs || [])[0];
  el.innerHTML = `<div style="min-width:0"><div class="hero-label"><span class="chip">${esc(famLabel(family(s.name)))}</span><span class="mono">${esc(s.config)}</span></div><div class="hero-value" title="${esc(s.name)}">${esc(s.name)}</div></div>
    <div class="stats">
      ${stat("Clip", v.ok ? dur(v.duration_s) : "missing", v.ok ? `${v.width}×${v.height} · ${(+v.fps).toFixed(0)} fps` : "", v.ok ? "" : "bad")}
      ${stat("Calibration", c.ok ? (c.rmse_m != null ? (+c.rmse_m).toFixed(2) : "–") : "not yet", c.ok ? "RMSE" : "", c.ok ? rmseClass(c.rmse_m) : "warn", c.ok ? "m" : "")}
      ${stat("Tracks", last && last.n_tracks != null ? last.n_tracks.toLocaleString("en-GB") : "–")}
      ${stat("Runs", (s.runs || []).length)}
    </div>
    <div class="actions">
      ${act("run", ICON.play, "Run", true)}${act("calibrate", ICON.target, "Calibrate")}${act("area", ICON.area, "Area")}
      ${act("results", ICON.image, "Results")}${act("houdini", ICON.swirl, "Houdini")}
    </div>`;
  bindActs(el);
}
function stat(label, value, title = "", cls = "", unit = "") {
  return `<div class="stat" title="${esc(title)}"><div class="stat-label">${esc(label)}</div><div class="stat-value ${cls}">${esc(value)}${unit ? `<small>${esc(unit)}</small>` : ""}</div></div>`;
}
function act(id, icon, label, primary) {
  return `<button class="act${primary ? " primary" : ""}" data-act="${id}"><span class="act-ic">${icon}</span>${esc(label)}</button>`;
}
function bindActs(el) {
  $$("[data-act]", el).forEach((b) => b.onclick = () => {
    const a = b.dataset.act;
    if (a === "new-site") newSiteDrawer();
    else goto({ run: 5, calibrate: 3, area: 4, results: 6, houdini: 8 }[a]);
  });
}

/* ------------------------------------------------------------------ steps */
function stepDone(id) {
  const s = A.site; if (!s) return false;
  const runs = s.runs || [];
  switch (id) {
    case 1: return !!(s.video && s.video.ok && s.map && s.map.ok);
    case 2: return !!s.camera_position;
    case 3: return !!(s.calibration && s.calibration.ok);
    case 4: return !!s.roi;
    case 5: return runs.length > 0;
    case 6: return runs.some((r) => r.processed);
    case 7: return A.composites.some((c) => c.name.startsWith(family(s.name)));
    case 8: return !!(A.links && A.links.kits.some((k) => String(k.site).startsWith(family(s.name))));
  }
  return false;
}
function defaultStep() {
  for (const id of [1, 3, 5, 6]) if (!stepDone(id)) return id;
  return 6;
}
function renderSteps() {
  const el = $("#steps");
  if (!A.config) { el.innerHTML = ""; return; }
  const back = isFocus() ? `<span class="lead"><button class="btn sm white back" id="back">← Back</button><span class="chip ink">${esc(A.site ? A.site.name : "")}</span></span>` : `<span class="steps-label">Workflow</span>`;
  el.innerHTML = back + `<div class="chain">${STEPS.map((s) => `<button class="step ${s.id === A.step ? "current" : stepDone(s.id) ? "done" : ""}" data-step="${s.id}"><span class="step-n">${String(s.id).padStart(2, "0")}</span>${stepDone(s.id) && s.id !== A.step ? ICON.check : ""}${esc(s.label)}</button>`).join("")}</div>`;
  $$("[data-step]", el).forEach((b) => b.onclick = () => goto(+b.dataset.step));
  const bk = $("#back", el); if (bk) bk.onclick = () => goto(A.prevStep || 5);
}

function goto(step) {
  if (!step) return;
  const cur = STEPS.find((s) => s.id === A.step);
  if (cur && !cur.focus) A.prevStep = A.step;
  try { localStorage.setItem(`gt.step.${A.config}`, step); } catch {}
  transition(() => {
    A.step = step;
    $("#app").classList.toggle("focus", isFocus());
    renderSteps(); renderWork();
  });
}

/* ------------------------------------------------------------------ sites */
function renderSites() {
  const q = A.search.trim().toLowerCase();
  const list = A.sites.filter((s) => !q || s.name.toLowerCase().includes(q) || s.config.toLowerCase().includes(q));
  $("#sites-n").textContent = A.sites.length ? ` ${A.sites.length}` : "";
  const fams = new Map();
  let k = 0;
  for (const s of list) { const f = family(s.name); if (!fams.has(f)) fams.set(f, []); fams.get(f).push(s); }
  const html = [...fams.entries()].sort((a, b) => a[0].localeCompare(b[0])).map(([f, ss]) =>
    `<li class="group">${esc(famLabel(f))} <span class="muted">${ss.length}</span></li>` + ss.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true })).map((s) => {
      const [txt, cls] = s.error ? ["Config error", "bad"] : !s.video_ok ? ["No video", "bad"] : !s.map_ok ? ["No aerial", "warn"] : !s.calibrated ? ["Calibrate", "warn"] : !s.runs ? ["Ready", "info"] : [`${s.runs} run${s.runs === 1 ? "" : "s"}`, "ok"];
      const sub = s.error ? s.error : [s.camera ? "camera set" : "no camera", s.calibrated ? "calibrated" : "not calibrated"].join(" · ");
      return `<li class="${fresh(`site:${s.config}`) ? "rise" : ""}" style="--i:${k++}"><button class="row ${s.config === A.config ? "sel" : ""}" data-config="${esc(s.config)}" title="${esc(sub)}">
        <span class="avatar c${hashIdx(f)}">${esc(initials(s.name))}</span>
        <span style="min-width:0"><span class="row-title">${esc(s.name)}</span><span class="row-sub">${esc(sub)}</span></span>
        <span class="row-end"><span class="st ${cls}">${txt}</span></span></button></li>`;
    }).join("")).join("");
  $("#sites").innerHTML = html || `<li class="empty">${A.sites.length ? "No site matches" : "No sites yet"}</li>`;
  $$("#sites [data-config]").forEach((b) => b.onclick = () => selectSite(b.dataset.config));
}
$("#site-search").oninput = (e) => { A.search = e.target.value; renderSites(); };

async function selectSite(config, step) {
  A.config = config; A.run = null; A.runDetail = null;
  try { localStorage.setItem("gt.config", config); } catch {}
  await refreshSite(false);                       // load first, then change the page in one go
  if (A.config !== config) return;                // another site was picked meanwhile
  let saved = null;
  try { saved = +localStorage.getItem(`gt.step.${config}`) || null; } catch {}
  transition(() => {
    A.step = step || saved || defaultStep();
    $("#app").classList.toggle("focus", isFocus());
    renderSites(); renderHero(); renderSteps(); renderSitePane(); renderWork();
  });
}

/* ------------------------------------------------------------------ work */
function renderWork() {
  if (A.mounted) { A.mounted.destroy(); A.mounted = null; }
  if (A.camMap) { A.camMap.destroy(); A.camMap = null; }
  const el = $("#work");
  if (!A.config) {
    el.innerHTML = `<div class="panel-head"><h2>Welcome</h2></div><div class="body"><div class="card"><h3>Start with a site</h3><p>A site is one clip of one place: a video, an aerial GeoTIFF (EPSG:27700) and a calibration.</p><div class="btns"><button class="btn neon" id="w-new">${ICON.plus} New site</button></div></div></div>`;
    $("#w-new").onclick = newSiteDrawer; return;
  }
  if (A.siteError) {
    el.innerHTML = `<div class="panel-head"><h2>${esc(A.config)}</h2></div><div class="body"><div class="note bad">${esc(A.siteError)}</div></div>`;
    return;
  }
  const views = { 1: viewSetup, 2: viewCamera, 3: viewCalibrate, 4: viewArea, 5: viewTrack, 6: viewResults, 7: viewCombine, 8: viewExport };
  (views[A.step] || viewSetup)(el);
}

const uiHooks = {
  get, post, toast, ask, esc, ago, thumb, base, lightbox, watch,
  site: () => A.site, refreshSite: () => refreshSite(), siteChanged: () => loadOverview(), goto,
};

function viewCalibrate(el) { A.mounted = mountCalibrator(el, A.config, uiHooks); }
function viewArea(el) { A.mounted = mountArea(el, A.config, uiHooks); }

function head(title, sub, right = "") {
  return `<div class="panel-head"><h2>${esc(title)}</h2>${sub ? `<span class="sub">${sub}</span>` : ""}<div class="grow"></div>${right}</div>`;
}

/* ---------- 1 footage & aerial */
function viewSetup(el) {
  const s = A.site, v = s.video, m = s.map;
  const ref = 0;
  el.innerHTML = head("Footage & aerial", esc(s.config_path)) + `<div class="body">
    <div class="grid2">
      <div class="card"><h3>Video ${v.ok ? '<span class="chip ok">found</span>' : '<span class="chip bad">missing</span>'}</h3>
        <div class="preview">${v.ok ? `<img src="/api/cal/frame?config=${enc(A.config)}&frame=${ref}&w=720" alt="" onerror="this.replaceWith('Can\\'t read the video')">` : "No video yet"}</div>
        ${v.ok ? `<dl class="kv"><dt>Size</dt><dd class="num">${v.width} × ${v.height}</dd><dt>Frame rate</dt><dd class="num">${(+v.fps).toFixed(2)} fps</dd><dt>Length</dt><dd class="num">${dur(v.duration_s)}</dd></dl>` : v.error ? `<div class="note bad">${esc(v.error)}</div>` : ""}
        <label class="field">File<div class="with"><input id="p-video" value="${esc(v.path || "")}" spellcheck="false"><button class="btn sm" data-pick="video">Browse…</button></div></label>
      </div>
      <div class="card"><h3>Aerial ${m.ok ? '<span class="chip ok">found</span>' : '<span class="chip bad">missing</span>'}</h3>
        <div class="preview contain">${m.ok ? `<img src="/api/cal/map?config=${enc(A.config)}&w=720" alt="" onerror="this.replaceWith('Can\\'t read the GeoTIFF')">` : "No aerial yet"}</div>
        <p class="hint">A north-up GeoTIFF in British National Grid (EPSG:27700), 5–25 cm per pixel, cropped to the site (Digimap, or QGIS → Export → Save As GeoTIFF).</p>
        <label class="field">File<div class="with"><input id="p-map" value="${esc(m.path || "")}" spellcheck="false"><button class="btn sm" data-pick="geotiff">Browse…</button></div></label>
      </div>
    </div>
    <div class="btns"><button class="btn primary" id="p-save">Save paths</button><span class="hint">Writes <span class="mono">video:</span> and <span class="mono">geotiff:</span> into ${esc(base(s.config))}.</span><div class="grow"></div><button class="btn sm white" id="p-next">Next: camera →</button></div>
    <div class="card"><h3>Filming checklist</h3>
      <div class="grid2 checklist">
        <span>• Tripod, locked off. A knock means re-calibrating.</span><span>• Stabilisation off (it shifts the frame).</span>
        <span>• Main 1× lens, no zoom, focus and exposure locked.</span><span>• Fixed frame rate (Auto FPS off).</span>
        <span>• Higher is better: a first-floor window beats the street.</span><span>• 6–8 sharp ground features visible on the aerial.</span>
      </div></div>
  </div>`;
  $$("[data-pick]", el).forEach((b) => b.onclick = async () => {
    const p = await pickPath("file", b.dataset.pick);
    if (p) $(b.dataset.pick === "video" ? "#p-video" : "#p-map").value = p;
  });
  $("#p-save").onclick = async () => {
    try {
      await post("/api/paths", { config: A.config, video: $("#p-video").value.trim(), geotiff: $("#p-map").value.trim() });
      toast("Paths saved", "ok"); await refreshSite(); loadOverview(); renderWork();
    } catch (e) { toast(e.message, "error"); }
  };
  $("#p-next").onclick = () => goto(2);
}

/* ---------- 2 camera */
function viewCamera(el) {
  const cp = A.site.camera_position || {};
  el.innerHTML = head("Where the camera stood", "Known, it turns the calibration into a physical camera: 3 pairs are enough and far points stay right.") + `<div class="body">
    <div style="display:grid;grid-template-columns:330px minmax(0,1fr);gap:12px">
      <div style="display:grid;gap:12px;align-content:start">
        <div class="card"><h3>Find the place</h3>
          <label class="field">Building, street or postcode<div class="with"><input id="cam-q" placeholder="e.g. National Gallery, London"><button class="btn sm primary" id="cam-go">Search</button></div></label>
          <label class="field">Floor you filmed from (0 = street)<input id="cam-floor" type="number" min="0" step="1" value="0"></label>
          <div class="hits" id="cam-hits"></div>
          <div class="hint">Only the search text is sent (to OpenStreetMap).</div>
        </div>
        <div class="card"><h3>Position ${A.site.camera_position ? '<span class="chip ok">saved</span>' : '<span class="chip warn">not set</span>'}</h3>
          <div class="fields" style="grid-template-columns:1fr 1fr">
            <label class="field">Easting (m)<input id="cam-E" type="number" step="0.1" value="${esc(cp.E ?? "")}"></label>
            <label class="field">Northing (m)<input id="cam-N" type="number" step="0.1" value="${esc(cp.N ?? "")}"></label>
            <label class="field">Height above ground (m)<input id="cam-h" type="number" step="0.1" value="${esc(cp.height_m ?? "")}"></label>
            <label class="field">Height ± (m)<input id="cam-ht" type="number" step="0.1" value="${esc(cp.height_tol_m ?? "")}" placeholder="auto"></label>
            <label class="field">Within (m)<input id="cam-tol" type="number" step="0.5" value="${esc(cp.position_tol_m ?? "")}" placeholder="15"></label>
            <label class="field">Field of view (°)<input id="cam-fov" type="number" step="0.5" value="${esc(cp.hfov_deg ?? "")}" placeholder="optional"></label>
          </div>
          <div class="btns"><button class="btn primary" id="cam-save">Save camera position</button><div class="grow"></div><button class="btn sm white" id="cam-next">Next: calibrate →</button></div>
        </div>
      </div>
      <div class="view" id="cam-map" style="min-height:470px;height:calc(100vh - 330px)"></div>
    </div></div>`;
  const setMarker = () => A.camMap && A.camMap.set($("#cam-E").value, $("#cam-N").value, $("#cam-tol").value || 15);
  if (A.site.map && A.site.map.ok) {
    A.camMap = mountCameraMap($("#cam-map"), A.config, uiHooks, (E, N) => {
      $("#cam-E").value = E.toFixed(1); $("#cam-N").value = N.toFixed(1);
      if (!$("#cam-tol").value || +$("#cam-tol").value > 6) $("#cam-tol").value = 4;
      setMarker();
    });
    setMarker();
  } else $("#cam-map").innerHTML = '<div class="loading">Add the aerial first (step 1)</div>';
  ["#cam-E", "#cam-N", "#cam-tol"].forEach((s) => $(s).oninput = setMarker);
  const search = async () => {
    const q = $("#cam-q").value.trim(); if (!q) return;
    $("#cam-hits").innerHTML = '<div class="hint">Searching…</div>';
    try {
      const r = await post("/api/locate", { query: q });
      $("#cam-hits").innerHTML = r.hits.length ? r.hits.map((h, i) => `<button class="hit" data-hit="${i}"><span>${esc(h.name.split(",").slice(0, 3).join(","))}</span><small class="num">E ${h.E} · N ${h.N} · ${h.size_m} m across</small></button>`).join("") : '<div class="hint">Nothing found in Great Britain.</div>';
      $$("[data-hit]").forEach((b) => b.onclick = () => {
        const h = r.hits[+b.dataset.hit], floor = +$("#cam-floor").value || 0;
        $("#cam-E").value = h.E; $("#cam-N").value = h.N;
        $("#cam-h").value = (floor * 3.1 + 1.5).toFixed(1);
        $("#cam-tol").value = Math.max(5, (h.size_m || 20) / 2).toFixed(1);
        setMarker(); A.camMap && A.camMap.centre();
      });
    } catch (e) { $("#cam-hits").innerHTML = `<div class="note bad">${esc(e.message)}</div>`; }
  };
  $("#cam-go").onclick = search;
  $("#cam-q").onkeydown = (e) => { if (e.key === "Enter") search(); };
  $("#cam-save").onclick = async () => {
    try {
      const body = { config: A.config, E: $("#cam-E").value, N: $("#cam-N").value, height_m: $("#cam-h").value,
                     position_tol_m: $("#cam-tol").value, height_tol_m: $("#cam-ht").value, hfov_deg: $("#cam-fov").value, note: "set in the app" };
      await post("/api/camera", body);
      toast("Camera position saved", "ok"); await refreshSite(); loadOverview(); renderWork();
    } catch (e) { toast(e.message, "error"); }
  };
  $("#cam-next").onclick = () => goto(3);
}

/* ---------- 5 track */
function viewTrack(el) {
  const s = A.site, c = s.calibration || {};
  el.innerHTML = head("Track", "Detect and track people and vehicles, project them onto the plan, clean, export, draw.") + `<div class="body">
    <div class="card"><h3>New run</h3>
      ${!c.ok ? '<div class="note warn">Not calibrated yet: the run tracks, but nothing lands on the plan until you calibrate (then "Process again" in Results).</div>' : ""}
      ${!s.video.ok ? '<div class="note bad">The video is missing (step 1).</div>' : ""}
      <div class="fields">
        <label class="field">Run name (optional)<input id="r-name" placeholder="date and time" spellcheck="false"></label>
        <label class="field">Only the first … frames (quick test)<input id="r-max" type="number" min="1" step="1" placeholder="all"></label>
      </div>
      <label class="check"><input type="checkbox" id="r-debug"> Also render the overlay video (boxes, ids, trails)</label>
      <div class="btns"><button class="btn neon" id="r-start" ${s.video.ok ? "" : "disabled"}>${ICON.play} Start run</button><span class="hint">${esc((A.device && A.device.gpu) || (A.device && A.device.device) || "")}</span></div>
    </div>
    <div class="board" id="board"></div>
  </div>`;
  $("#r-start").onclick = async () => {
    const opts = { run_name: $("#r-name").value.trim() || undefined, max_frames: $("#r-max").value || undefined, debug_video: $("#r-debug").checked };
    const j = await startJob({ kind: "run", config: A.config, options: opts });
    if (j) watch(j, () => { A.run = null; });
  };
  renderBoard();
}

function renderBoard() {
  const el = $("#board"); if (!el || !A.site) return;
  const name = A.site.name;
  const running = A.jobs.filter((j) => j.status === "running" && j.label.includes(`· ${name}`));
  const runs = A.site.runs || [];
  const lanes = [
    ["Running", running.map((j) => `<button class="tile running${fresh(`jt:${j.id}`) ? " rise" : ""}" data-job="${j.id}"><div class="tile-top"><span class="avatar sm c0">${jobIcon(j.kind)}</span><span class="tile-name">${esc(j.kind)}${j.label.split("·")[2] ? ` · ${esc(j.label.split("·")[2].trim())}` : ""}</span></div><div class="tile-sub mono">${esc(j.last || "starting…")}</div><div class="bar-progress"><i></i></div><div class="tile-sub">${elapsed(j)}</div></button>`)],
    ["Tracked", runs.filter((r) => !r.processed).map(runTile)],
    ["Finished", runs.filter((r) => r.processed).map(runTile)],
  ];
  el.innerHTML = lanes.map(([t, tiles]) => `<div class="lane"><div class="lane-head"><span>${t}</span><span class="lane-n">${tiles.length}</span></div><div class="lane-body">${tiles.join("") || `<div class="lane-empty">${t === "Running" ? "Nothing running" : "None"}</div>`}</div></div>`).join("");
  $$("[data-run]", el).forEach((b) => b.onclick = () => { A.run = b.dataset.run; goto(6); });
  $$("[data-job]", el).forEach((b) => b.onclick = () => { A.openJob = b.dataset.job; setTab("jobs"); renderJobs(); });
}
function runTile(r, i = 0) {
  return `<button class="tile${fresh(`run:${A.config}:${r.name}`) ? " rise" : ""}" style="--i:${i}" data-run="${esc(r.name)}"><div class="tile-top"><span class="avatar sm c${r.processed ? 0 : 1}">${ICON.layers}</span><span class="tile-name">${esc(r.name)}</span></div>
    <div class="tile-big">${r.n_tracks != null ? r.n_tracks.toLocaleString("en-GB") : "–"}<small>tracks</small></div>
    <div class="tile-sub">${ago(r.mtime)}</div>${checkChip(r.check) ? `<div>${checkChip(r.check)}</div>` : ""}</button>`;
}

/* ---------- 6 results */
async function viewResults(el) {
  const runs = A.site.runs || [];
  if (!runs.length) {
    el.innerHTML = head("Results") + `<div class="body"><div class="card"><h3>No runs yet</h3><p>Track the clip first.</p><div class="btns"><button class="btn neon" id="go-track">${ICON.play} Go to Track</button></div></div></div>`;
    $("#go-track").onclick = () => goto(5); return;
  }
  try {                                          // back from the Live view: the run it was opened from
    const back = JSON.parse(sessionStorage.getItem("gt.run") || "null");
    sessionStorage.removeItem("gt.run");
    if (back && back.config === A.config && runs.some((r) => r.name === back.run)) A.run = back.run;
  } catch {}
  if (!A.run || !runs.some((r) => r.name === A.run)) A.run = runs[0].name;
  const seg = `<select id="run-pick" class="btn sm white" style="padding-right:10px">${runs.map((r) => `<option value="${esc(r.name)}" ${r.name === A.run ? "selected" : ""}>${esc(r.name)} · ${r.n_tracks ?? "–"} tracks · ${ago(r.mtime)}</option>`).join("")}</select>`;
  el.innerHTML = head("Results", "", seg) + `<div class="body" id="res"><div class="empty">Loading…</div></div>`;
  $("#run-pick").onchange = (e) => { A.run = e.target.value; viewResults(el); };
  let d;
  try { d = await get(`/api/run?config=${enc(A.config)}&run=${enc(A.run)}`); }
  catch (e) { $("#res").innerHTML = `<div class="note bad">${esc(e.message)}</div>`; return; }
  if (A.step !== 6) return;
  A.runDetail = d;
  const r = runs.find((x) => x.name === A.run) || {};
  const plans = [...(d.plate ? [d.plate] : []), ...d.images];
  const shots = (list, cls = "") => `<div class="gallery">${list.map((p, i) => `<button class="shot rise ${cls}" style="--i:${i}" data-g="${cls || "img"}" data-i="${i}">${cls === "video" ? "" : `<img loading="lazy" src="${thumb(p, 420)}" alt="">`}<span>${esc(base(p))}</span></button>`).join("")}</div>`;
  const fileRows = (list) => `<div class="files">${list.map((p) => {
    const ext = base(p).split(".").pop().toLowerCase();
    const extra = ext === "3dm" ? `<button class="btn xs primary" data-open="${esc(p)}">Open in Rhino</button>` :
      base(p) === "grasshopper_import.py" ? `<button class="btn xs" data-copytext="${esc(p)}">${ICON.copy} Copy for Grasshopper</button>` :
      ["py", "txt", "json", "csv", "yaml", "md"].includes(ext) ? `<button class="btn xs" data-view="${esc(p)}">View</button>` : "";
    return `<div class="file"><span class="avatar sm c5">${ICON.file}</span><span class="name" title="${esc(p)}">${esc(base(p))}</span>${extra}<button class="btn xs white" data-folder="${esc(p)}">${ICON.folder}</button></div>`;
  }).join("")}</div>`;
  const metrics = d.metrics && d.metrics.length ? (() => {
    const cols = Object.keys(d.metrics[0]).slice(0, 9);
    return `<div class="card"><h3>Metrics</h3><div class="table-wrap"><table class="metrics"><tr>${cols.map((c) => `<th>${esc(c)}</th>`).join("")}</tr>${d.metrics.slice(0, 40).map((row) => `<tr>${cols.map((c) => `<td>${esc(row[c])}</td>`).join("")}</tr>`).join("")}</table></div></div>`;
  })() : "";
  $("#res").classList.add("fade-in");
  $("#res").innerHTML = `
    <div class="card summary"><div style="display:flex;gap:28px;align-items:flex-end;flex-wrap:wrap">
      <div><div class="stat-label">Tracks</div><div class="big-rmse">${r.n_tracks != null ? r.n_tracks.toLocaleString("en-GB") : "–"}</div></div>
      <div style="display:grid;gap:10px;min-width:0;flex:1"><span class="path">${esc(d.path)}</span><div class="btns">${checkChip(d.check)}${r.processed ? '<span class="chip ok">processed</span>' : '<span class="chip warn">tracked only</span>'}</div></div>
      <button class="btn sm white" data-folder="${esc(d.path)}">${ICON.folder} Open folder</button></div>
      <div class="btns">
        <span id="live-slot" hidden></span>
        <button class="btn sm" data-job="process" title="Stages 3–6 again (after a new calibration)">Process again</button>
        <button class="btn sm" data-job="package">Package images</button>
        <button class="btn sm" data-job="debug-video">Overlay video</button>
        <button class="btn sm" data-job="stabilize" title="For a handheld / moved camera">Stabilise</button>
        <button class="btn sm" data-job="rhino">${ICON.cube} Rhino model</button>
        <button class="btn sm" data-job="calibration-picture">Calibration picture</button>
        <span class="btns"><button class="btn sm" data-job="process-sequence">Process sequence at</button><input id="seq-at" class="btn sm white" style="width:64px" type="number" value="2" step="0.5" min="0"><span class="hint">s</span></span>
      </div></div>
    ${plans.length ? `<div class="section-title">Plans <span class="chip">${plans.length}</span></div>${shots(plans)}` : ""}
    ${d.videos.length ? `<div class="section-title">Videos <span class="chip">${d.videos.length}</span></div>${shots(d.videos, "video")}` : ""}
    ${d.process.length ? `<div class="section-title">Process sequence</div>${shots(d.process, "proc")}` : ""}
    ${d.extras.length ? `<div class="section-title">Extras</div>${shots(d.extras, "extra")}` : ""}
    ${metrics}
    <div class="grid3">
      ${d.data.length ? `<div class="card"><h3>Data <span class="chip">${d.data.length}</span></h3>${fileRows(d.data)}</div>` : ""}
      ${d.houdini.length ? `<div class="card"><h3>Houdini <span class="chip">${d.houdini.length}</span></h3>${fileRows(d.houdini)}</div>` : ""}
      ${d.rhino.length ? `<div class="card"><h3>Rhino / Grasshopper <span class="chip">${d.rhino.length}</span></h3>${fileRows(d.rhino)}</div>` : ""}
    </div>
    ${d.log_tail ? `<details class="card"><summary>Run log (last lines)</summary><pre class="log" style="margin-top:10px">${esc(d.log_tail.join("\n"))}</pre></details>` : ""}`;
  const res = $("#res");
  const groups = { img: plans, video: d.videos, proc: d.process, extra: d.extras };
  $$("[data-g]", res).forEach((b) => b.onclick = () => lightbox(groups[b.dataset.g].map((p) => ({ path: p })), +b.dataset.i));
  bindFiles(res);
  $$("[data-job]", res).forEach((b) => b.onclick = () => {
    const kind = b.dataset.job, options = kind === "process-sequence" ? { at: +$("#seq-at").value || 2 } : {};
    startJob({ kind, config: A.config, run: A.run, options }, "Write anyway");
  });
  // Live view (plans/live_controller/03): only for a run with a scene package (live/)
  get(`/api/live/info?run=${enc(d.path)}`).then((li) => {
    const slot = $("#live-slot");
    if (!li.live || !slot || A.runDetail !== d) return;
    slot.outerHTML = `<button class="btn sm primary" id="go-live" title="The clip in 3D, redrawn live by the phone or the mouse (F: full screen)">${ICON.play} Live</button>`;
    $("#go-live").onclick = () => {
      try { sessionStorage.setItem("gt.run", JSON.stringify({ config: A.config, run: A.run })); } catch {}
      location.href = `/live?run=${enc(d.path)}`;
    };
  }).catch(() => {});
}

function bindFiles(el) {
  $$("[data-folder]", el).forEach((b) => b.onclick = () => openFolder(b.dataset.folder));
  $$("[data-open]", el).forEach((b) => b.onclick = () => openFile(b.dataset.open));
  $$("[data-view]", el).forEach((b) => b.onclick = () => lightbox([{ path: b.dataset.view, kind: "text" }]));
  $$("[data-copytext]", el).forEach((b) => b.onclick = async () => {
    try { const t = await get(`/api/app/text?path=${enc(b.dataset.copytext)}`); copyText(t.text, "Copied: paste it into a Grasshopper Python 3 Script component"); }
    catch (e) { toast(e.message, "error"); }
  });
}

/* ---------- 7 combine */
async function viewCombine(el) {
  const f = family(A.site.name);
  try { A.composites = (await get("/api/composites")).composites; } catch { A.composites = []; }
  const fams = new Map();
  for (const s of A.sites) { const k = family(s.name); if (!fams.has(k)) fams.set(k, []); fams.get(k).push(s); }
  el.innerHTML = head("Combine clips", "Several clips of one place on one plan: one flow field, density and comparison where views overlap.") + `<div class="body">
    <div style="display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:12px">
      <div class="card"><h3>New combined plan</h3>
        <div class="fields"><label class="field">Name<input id="cb-name" value="${esc(f)}" spellcheck="false"></label><label class="field">Title (optional)<input id="cb-title" placeholder="${esc(famLabel(f))}"></label></div>
        <label class="check"><input type="checkbox" id="cb-ok"> Only clips whose calibration check is ok</label>
        <div class="section-title">Clips</div>
        <div style="display:grid;gap:4px;max-height:44vh;overflow:auto">
          ${[...fams.entries()].sort((a, b) => (b[0] === f) - (a[0] === f) || a[0].localeCompare(b[0])).map(([k, ss]) => `<div class="group" style="padding:6px 2px 2px">${esc(famLabel(k))}</div>` + ss.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true })).map((s) => `<label class="check file"><input type="checkbox" data-clip="${esc(s.config)}" ${k === f && s.calibrated && s.runs ? "checked" : ""} ${s.runs ? "" : "disabled"}> <span class="name">${esc(s.name)}</span><span class="st ${s.runs ? "ok" : "none"}">${s.runs ? (s.calibrated ? "calibrated" : "not calibrated") : "no run"}</span></label>`).join("")).join("")}
        </div>
        <div class="btns"><button class="btn neon" id="cb-go">${ICON.stack} Combine</button></div>
      </div>
      <div style="display:grid;gap:8px;align-content:start"><div class="section-title">Combined plans <span class="chip">${A.composites.length}</span></div><div class="gallery" id="cb-list" style="grid-template-columns:repeat(auto-fill,minmax(200px,1fr))"></div></div>
    </div></div>`;
  $("#cb-go").onclick = async () => {
    const configs = $$("[data-clip]:checked", el).map((x) => x.dataset.clip);
    if (!configs.length) { toast("Pick at least one clip", "error"); return; }
    const j = await startJob({ kind: "composite", configs, options: { name: $("#cb-name").value.trim(), title: $("#cb-title").value.trim(), only_ok: $("#cb-ok").checked } });
    if (j) watch(j, () => { if (A.step === 7) viewCombine($("#work")); });
  };
  const list = $("#cb-list");
  if (!A.composites.length) { list.innerHTML = '<div class="empty">None yet</div>'; return; }
  list.innerHTML = A.composites.map((c, i) => `<button class="tile thumb rise" style="--i:${i}" data-comp="${esc(c.name)}"><img alt="" data-src=""><span class="tile-cap"><span class="tile-name">${esc(c.name)}</span><span class="tile-sub">${c.images} images</span></span></button>`).join("");
  for (const c of A.composites) {
    get(`/api/composite?name=${enc(c.name)}`).then((d) => {
      const im = $(`[data-comp="${CSS.escape(c.name)}"] img`, list);
      if (im && d.images[0]) im.src = thumb(d.images[0], 420);
    }).catch(() => {});
  }
  $$("[data-comp]", list).forEach((b) => b.onclick = () => compositeDrawer(b.dataset.comp));
}

async function compositeDrawer(name) {
  let d;
  try { d = await get(`/api/composite?name=${enc(name)}`); } catch (e) { toast(e.message, "error"); return; }
  drawer("Combined plan", `<div class="d-title">${esc((d.title || [])[0] || name)}</div><div class="d-sub">${esc(d.path)}</div>
    <div class="btns"><button class="btn sm primary" id="cd-kit">${ICON.swirl} Make a Houdini kit from it</button><button class="btn sm white" data-folder="${esc(d.path)}">${ICON.folder} Folder</button></div>
    <div class="gallery">${d.images.map((p, i) => `<button class="shot" data-i="${i}"><img loading="lazy" src="${thumb(p, 360)}" alt=""><span>${esc(base(p))}</span></button>`).join("")}</div>
    ${d.rhino.length ? `<div class="card"><h3>Rhino</h3><div class="files">${d.rhino.map((p) => `<div class="file"><span class="name">${esc(base(p))}</span>${p.endsWith(".3dm") ? `<button class="btn xs primary" data-open="${esc(p)}">Open in Rhino</button>` : ""}<button class="btn xs white" data-folder="${esc(p)}">${ICON.folder}</button></div>`).join("")}</div></div>` : ""}`,
  (b) => {
    $$("[data-i]", b).forEach((x) => x.onclick = () => lightbox(d.images.map((p) => ({ path: p })), +x.dataset.i));
    bindFiles(b);
    $("#cd-kit", b).onclick = () => { A.preset = { composite: d.path, site: name }; closeDrawer(); goto(8); };
  });
}

/* ---------- 8 Houdini & Grasshopper */
async function viewExport(el) {
  el.innerHTML = head("Houdini & Grasshopper", "Particle kits for Houdini, models and definitions for Rhino / Grasshopper.",
    `<button class="btn sm white" id="x-refresh">Look again</button>`) + `<div class="body" id="xb"><div class="empty">Looking for Houdini, Rhino and kits…</div></div>`;
  $("#x-refresh").onclick = async () => { A.links = null; A.sources = null; viewExport(el); };
  let L, src;
  try { [L, src] = await Promise.all([loadLinks(), A.sources ? Promise.resolve(A.sources) : get("/api/app/kit-sources")]); A.sources = src; }
  catch (e) { $("#xb").innerHTML = `<div class="note bad">${esc(e.message)}</div>`; return; }
  if (A.step !== 8) return;
  const f = family(A.site.name), H = L.houdini, R = L.rhino, G = L.grasshopper, run = L.running || {};
  const preset = A.preset || {}; A.preset = {};
  let runRhino = [];
  const last = (A.site.runs || [])[0];
  if (last) { try { runRhino = (await get(`/api/run?config=${enc(A.config)}&run=${enc(last.name)}`)).rhino; } catch {} }
  const kits = L.kits.slice().sort((a, b) => (String(b.site).startsWith(f)) - (String(a.site).startsWith(f)) || b.mtime - a.mtime);
  const comps = src.composites;
  const defComp = preset.composite || (comps.find((c) => c.name.toLowerCase().includes(f) && !c.name.includes("ok_only")) || comps[0] || {}).path;
  const siteName = preset.site || f;
  $("#xb").classList.add("fade-in");
  $("#xb").innerHTML = `
    <div class="grid2">
      <div class="card"><h3>${ICON.swirl} Houdini ${H ? `<span class="chip ok">${esc(H.version)}</span>` : '<span class="chip bad">not found</span>'}</h3>
        <div class="status-line"><span class="status-dot ${run.houdini ? "on" : "off"}"></span>${run.houdini ? "Open now" : "Not running"}</div>
        <p class="hint">Kits open as a double-click would. <b>Add to open scene</b> copies one line for Windows ▸ Python Shell: it adds a subnet <span class="mono">/obj/groundtrack_&lt;site&gt;</span> and touches nothing else in your scene.</p>
      </div>
      <div class="card"><h3>${ICON.cube} Rhino ${R ? `<span class="chip ok">${esc(R.version)}</span>` : '<span class="chip bad">not found</span>'}
          <span class="chip ${G.installed ? "ok" : "warn"}">Grasshopper plugin ${G.installed ? esc(G.version || (G.yak || []).join(", ") || "") : "not installed"}</span></h3>
        <div class="status-line"><span class="status-dot ${run.rhino ? "on" : "off"}"></span>${run.rhino ? "Open now" : "Not running"}</div>
        ${G.conflict ? '<div class="note warn">The plugin is installed twice (Libraries folder and Yak package): Grasshopper loads only one. Remove one of them.</div>' : ""}
        ${G.path ? `<div class="path">${esc(G.path)}</div>` : `<p class="hint">groundtrack-grasshopper reads these results in Grasshopper (flow fields, streamlines, particles, Houdini Export).</p>`}
        <div class="btns"><button class="btn xs" id="x-rel">${ICON.copy} Copy releases link</button></div>
      </div>
    </div>

    <div class="section-title">Houdini kits <span class="chip">${kits.length}</span></div>
    <div class="grid2">${kits.map(kitCard).join("") || '<div class="empty">No kits yet: make one below, or with Houdini Export in Grasshopper.</div>'}</div>

    <div class="card"><h3>${ICON.plus} New kit from a combined plan</h3>
      ${comps.length ? `
      <div class="fields">
        <label class="field">Site name<input id="k-site" value="${esc(siteName)}" spellcheck="false"></label>
        <label class="field" style="grid-column:span 2">Combined plan<select id="k-comp">${comps.map((c) => `<option value="${esc(c.path)}" ${c.path === defComp ? "selected" : ""}>${esc(c.name)}</option>`).join("")}</select></label>
      </div>
      <div class="section-title">Clips as extra modules</div>
      <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:4px;max-height:200px;overflow:auto">${src.clips.map((c) => `<label class="check file"><input type="checkbox" data-kclip="${esc(c.path)}" ${family(c.site) === family(siteName) ? "checked" : ""}><span class="name">${esc(c.site)}</span><span class="st none">${esc(c.run)}</span></label>`).join("") || '<div class="hint">No clips with a vector field yet.</div>'}</div>
      <div class="fields" style="grid-template-columns:1fr auto"><label class="field">Kit folder<input id="k-out" value="${esc(`${A.root}\\runs\\houdini\\${siteName}`)}" spellcheck="false"></label><label class="field">&nbsp;<button class="btn sm" id="k-browse">Browse…</button></label></div>
      <label class="check"><input type="checkbox" id="k-nobuild"> Only the files (build the scene later, or add it to an open scene)</label>
      <div class="btns"><button class="btn neon" id="k-go">${ICON.hammer} Build kit</button><span class="hint">Runs <span class="mono">motus houdini-pop</span>; the scene is built with hython${H ? "" : " (Houdini not found: files only)"}.</span></div>` : '<p class="hint">Combine clips first (step 7): a kit is built from a combined plan. Combined plans in library folders are found too.</p>'}
    </div>

    <div class="section-title">Rhino & Grasshopper</div>
    <div class="grid2">
      <div class="card"><h3>This site's latest run</h3>
        ${last ? (runRhino.length ? `<div class="files">${runRhino.map((p) => `<div class="file"><span class="avatar sm c1">${p.endsWith(".3dm") ? ICON.cube : ICON.file}</span><span class="name">${esc(base(p))}</span>${p.endsWith(".3dm") ? `<button class="btn xs primary" data-open="${esc(p)}">Open in Rhino</button>` : base(p) === "grasshopper_import.py" ? `<button class="btn xs" data-copytext="${esc(p)}">${ICON.copy} Copy</button>` : ""}<button class="btn xs white" data-folder="${esc(p)}">${ICON.folder}</button></div>`).join("")}</div>` : '<p class="hint">No Rhino model in this run yet.</p>') + `<div class="btns"><button class="btn sm" id="x-rhino">${ICON.cube} ${runRhino.length ? "Write the model again" : "Write the Rhino model"}</button></div>` : '<p class="hint">No run yet.</p>'}
      </div>
      <div class="card"><h3>Definitions <span class="chip">${L.definitions.length}</span></h3>
        ${fileList(L.definitions.slice(0, 8), "Open")}
      </div>
    </div>
    <div class="card"><h3>Rhino models <span class="chip">${L.models.length}</span></h3>${fileList(L.models.slice(0, 12), "Open in Rhino")}</div>
    ${L.scenes.length ? `<div class="card"><h3>Other Houdini scenes <span class="chip">${L.scenes.length}</span></h3>${fileList(L.scenes.slice(0, 10), "Open in Houdini")}</div>` : ""}

    <div class="card"><h3>${ICON.folder} Library folders</h3>
      <p class="hint">Where kits, models and definitions are looked for. The project's runs and Grasshopper's default kit folder are always included.</p>
      <div class="files">${L.libraries.map((l) => `<div class="file"><span class="name" title="${esc(l.path)}">${esc(l.path)}</span>${l.added ? `<button class="btn xs danger" data-unlib="${esc(l.path)}">Remove</button>` : '<span class="chip">built in</span>'}</div>`).join("")}</div>
      <div class="btns"><button class="btn sm" id="x-addlib">${ICON.plus} Add a folder…</button></div>
    </div>`;
  const xb = $("#xb");
  bindFiles(xb);
  $("#x-rel").onclick = () => copyText(G.releases, "Copied the releases link");
  $$("[data-shell]", xb).forEach((b) => b.onclick = () => copyText(b.dataset.shell, "Copied: in Houdini, Windows ▸ Python Shell, paste, Enter"));
  $$("[data-rebuild]", xb).forEach((b) => b.onclick = async () => {
    try { const j = await post("/api/houdini/rebuild", { kit: b.dataset.rebuild }); jobStarted(j); watch(j, () => { A.links = null; if (A.step === 8) viewExport($("#work")); }); }
    catch (e) { toast(e.message, "error"); }
  });
  const xr = $("#x-rhino", xb);
  if (xr) xr.onclick = async () => { const j = await startJob({ kind: "rhino", config: A.config, run: last.name }, "Write anyway"); if (j) watch(j, () => { if (A.step === 8) viewExport($("#work")); }); };
  $("#x-addlib").onclick = addLibrary;
  $$("[data-unlib]", xb).forEach((b) => b.onclick = async () => { try { A.links = await post("/api/app/library", { action: "remove", path: b.dataset.unlib }); viewExport($("#work")); } catch (e) { toast(e.message, "error"); } });
  const kgo = $("#k-go", xb);
  if (kgo) {
    $("#k-site").oninput = (e) => { $("#k-out").value = `${A.root}\\runs\\houdini\\${e.target.value.trim()}`; };
    $("#k-browse").onclick = async () => { const p = await pickPath("folder"); if (p) $("#k-out").value = p; };
    kgo.onclick = async () => {
      const body = { site: $("#k-site").value.trim(), composite: $("#k-comp").value, out: $("#k-out").value.trim(),
                     clips: $$("[data-kclip]:checked", xb).map((x) => x.dataset.kclip), no_build: $("#k-nobuild").checked };
      try { const j = await post("/api/houdini/build", body); jobStarted(j); watch(j, () => { A.links = null; if (A.step === 8) viewExport($("#work")); }); }
      catch (e) { toast(e.message, "error"); }
    };
  }
}

function kitCard(k) {
  const mods = k.modules || [];
  return `<div class="card kit">
    <div class="kit-top"><span class="avatar c${k.source === "Grasshopper" ? 4 : 0}">${ICON.swirl}</span>
      <div style="min-width:0;flex:1"><div class="kit-name" title="${esc(k.path)}">${esc(k.site)}${k.name !== k.site ? ` <span class="muted" style="font-weight:450">· ${esc(k.name)}</span>` : ""}</div>
      <div class="tile-sub">${k.scene ? `${esc(base(k.scene))} · built ${ago(k.scene_mtime)}` : "scene not built yet"}</div></div>
      <span class="chip ${k.source === "Grasshopper" ? "violet" : "ok"}">${esc(k.source)}</span></div>
    ${mods.length ? `<div class="mods">${mods.slice(0, 6).map((m) => `<span class="chip">${esc(m)}</span>`).join("")}${mods.length > 6 ? `<span class="chip">+${mods.length - 6}</span>` : ""}</div>` : ""}
    <div class="btns">
      ${k.scene ? `<button class="btn sm primary" data-open="${esc(k.scene)}">${ICON.open} Open in Houdini</button>` : ""}
      <button class="btn sm" data-shell="${esc(k.shell_line)}" title="${esc(k.shell_line)}">${ICON.copy} Add to open scene</button>
      <button class="btn sm white" data-rebuild="${esc(k.path)}">${ICON.hammer} ${k.scene ? "Rebuild" : "Build scene"}</button>
      <button class="btn sm white" data-folder="${esc(k.path)}" title="Open the kit folder">${ICON.folder}</button>
    </div></div>`;
}

function fileList(list, openLabel) {
  if (!list.length) return '<p class="hint">None found in the library folders.</p>';
  return `<div class="files">${list.map((p) => `<div class="file"><span class="name" title="${esc(p.path)}">${esc(p.name)}</span><span class="tile-sub">${ago(p.mtime)}</span><button class="btn xs primary" data-open="${esc(p.path)}">${openLabel}</button><button class="btn xs white" data-folder="${esc(p.path)}">${ICON.folder}</button></div>`).join("")}</div>`;
}

async function addLibrary() {
  let p = await pickPath("folder");
  if (!p) return;
  try { A.links = await post("/api/app/library", { action: "add", path: p }); toast("Folder added", "ok"); if (A.step === 8) viewExport($("#work")); renderLinksPane(); }
  catch (e) { toast(e.message, "error"); }
}

/* ------------------------------------------------------------------ side panel */
function setTab(t) {
  A.tab = t;
  $$("#tabs [data-tab]").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === t)));
  $$(".side [data-pane]").forEach((p) => p.hidden = p.dataset.pane !== t);
  segThumb($("#tabs"));
  const pane = $(`.side [data-pane="${t}"] .scroll`);
  if (pane && !reduced()) { pane.classList.remove("fade-in"); void pane.offsetWidth; pane.classList.add("fade-in"); }
  if (t === "links") renderLinksPane();
  if (t === "site") renderSitePane();
}
$$("#tabs [data-tab]").forEach((b) => b.onclick = () => setTab(b.dataset.tab));

function jobIcon(kind) {
  return { run: ICON.play, track: ICON.play, calibrate: ICON.target, autocalibrate: ICON.target, composite: ICON.stack,
           "houdini-pop": ICON.swirl, "houdini-build": ICON.swirl, rhino: ICON.cube }[kind] || ICON.layers;
}

function renderJobs() {
  const el = $("#jobs"); if (!el) return;
  if (!A.jobs.length) { el.innerHTML = '<div class="empty">No jobs yet. Runs, calibrations and kits show up here with their live log.</div>'; return; }
  const scroll = el.querySelector(".log");
  const atBottom = scroll ? scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight < 30 : true;
  el.innerHTML = A.jobs.map((j) => {
    const cls = j.status === "running" ? "running" : j.status === "done" ? "ok" : j.status === "failed" ? "bad" : "none";
    const pct = j.status === "running" ? percent(j.last) : null;
    const open = j.id === A.openJob;
    const L = A.logs[j.id];
    return `<div class="job${fresh(`job:${j.id}`) ? " rise" : ""}"><div class="job-top" data-jid="${j.id}"><span class="avatar sm ${j.status === "failed" ? "c3" : j.status === "running" ? "c0" : "c5"}">${jobIcon(j.kind)}</span>
      <span style="min-width:0"><span class="row-title">${esc(j.label)}</span><span class="row-sub">${ago(j.started)} · ${elapsed(j)}</span></span>
      <span class="st ${cls}">${j.status === "running" ? '<i class="sq"></i>running' : j.status}</span></div>
      ${j.status === "running" ? `<div class="ruler${pct == null ? " indet" : ""}"><i${pct == null ? "" : ` style="width:${pct}%"`}></i></div><div class="progress">${esc(j.last || "")}</div>` : ""}
      ${open ? `<pre class="log">${L ? L.lines.slice(-500).map((x) => `<span class="${/error|Traceback|failed/i.test(x) ? "err" : /^saved |✓|done/i.test(x) ? "ok" : ""}">${esc(x)}</span>`).join("\n") : "Loading…"}</pre>
        <div class="btns">${j.status === "running" ? `<button class="btn xs danger" data-stop="${j.id}">${ICON.stop} Stop</button>` : ""}<button class="btn xs" data-copylog="${j.id}">${ICON.copy} Copy log</button></div>` : ""}</div>`;
  }).join("");
  const log = el.querySelector(".log");
  if (log && atBottom) log.scrollTop = log.scrollHeight;
  $$("[data-jid]", el).forEach((b) => b.onclick = () => { A.openJob = A.openJob === b.dataset.jid ? null : b.dataset.jid; renderJobs(); pollLog(); });
  $$("[data-stop]", el).forEach((b) => b.onclick = async () => { if (await ask("Stop this job?", "Its output so far stays on disk.", "Stop")) post("/api/job/stop", { id: b.dataset.stop }).catch((e) => toast(e.message, "error")); });
  $$("[data-copylog]", el).forEach((b) => b.onclick = () => copyText((A.logs[b.dataset.copylog] || { lines: [] }).lines.join("\n"), "Log copied"));
}

async function pollLog() {
  const id = A.openJob; if (!id) return;
  const j = A.jobs.find((x) => x.id === id); if (!j) return;
  const L = A.logs[id] || (A.logs[id] = { lines: [], n: 0, final: false });
  if (L.final) return;
  try {
    const d = await get(`/api/job?id=${id}&since=${L.n}`);
    L.lines.push(...d.lines); L.n = d.n;
    if (d.status !== "running") L.final = true;
    renderJobs();
  } catch {}
}

function renderSitePane() {
  const el = $("#site-pane"); if (!el || A.tab !== "site") return;
  const s = A.site;
  if (!s) { el.innerHTML = '<div class="empty">No site selected</div>'; return; }
  const c = s.calibration || {}, cp = s.camera_position;
  el.innerHTML = `<div class="card" style="margin-bottom:8px"><h3>${esc(s.name)}</h3>
      <dl class="kv"><dt>Config</dt><dd class="mono">${esc(s.config)}</dd>
      <dt>Video</dt><dd>${s.video.ok ? `${s.video.width}×${s.video.height}, ${dur(s.video.duration_s)}` : '<span class="st bad">missing</span>'}</dd>
      <dt>Aerial</dt><dd>${s.map.ok ? "found" : '<span class="st bad">missing</span>'}</dd>
      <dt>Camera</dt><dd class="num">${cp ? `E ${cp.E}, N ${cp.N}, ${cp.height_m} m up` : '<span class="st warn">not set</span>'}</dd>
      <dt>Calibration</dt><dd>${c.ok ? `<span class="st ${rmseClass(c.rmse_m) === "good" ? "ok" : rmseClass(c.rmse_m) === "warn" ? "warn" : "bad"}">${c.rmse_m != null ? (+c.rmse_m).toFixed(2) : "–"} m RMSE</span> · ${c.n_points} pairs` : '<span class="st warn">not yet</span>'}</dd>
      <dt>Area</dt><dd>${s.roi ? "outlined" : "whole frame"}</dd>
      <dt>Runs</dt><dd>${(s.runs || []).length}</dd></dl>
      <div class="btns"><button class="btn xs white" data-folder="${esc(s.config_path)}">${ICON.folder} Config folder</button></div></div>
    ${(c.checks || []).length ? `<div class="card"><h3>Calibration check</h3><div class="gallery" style="grid-template-columns:1fr">${c.checks.map((p, i) => `<button class="shot" data-ci="${i}"><img src="${thumb(p, 420)}" alt=""><span>${esc(base(p))}</span></button>`).join("")}</div></div>` : ""}`;
  bindFiles(el);
  $$("[data-ci]", el).forEach((b) => b.onclick = () => lightbox(c.checks.map((p) => ({ path: p })), +b.dataset.ci));
}

async function renderLinksPane() {
  const el = $("#links-pane"); if (!el || A.tab !== "links") return;
  if (!A.links) el.innerHTML = '<div class="empty">Looking…</div>';
  let L;
  try { L = await loadLinks(); } catch (e) { el.innerHTML = `<div class="note bad">${esc(e.message)}</div>`; return; }
  const run = A.running || {};
  const row = (icon, cls, title, sub, end) => `<div class="row" style="cursor:default"><span class="avatar ${cls}">${icon}</span><span style="min-width:0"><span class="row-title">${title}</span><span class="row-sub">${sub}</span></span><span class="row-end">${end}</span></div>`;
  el.innerHTML = `<div class="list">
      ${row(ICON.swirl, "c2", `Houdini ${esc(L.houdini ? L.houdini.version : "")}`, L.houdini ? esc(L.houdini.bin) : "not found", `<span class="st ${run.houdini ? "ok" : "none"}">${run.houdini ? "open" : "closed"}</span>`)}
      ${row(ICON.cube, "c1", `Rhino ${esc(L.rhino ? L.rhino.version : "")}`, L.rhino ? esc(L.rhino.exe) : "not found", `<span class="st ${run.rhino ? "ok" : "none"}">${run.rhino ? "open" : "closed"}</span>`)}
      ${row("GH", "c0", "Grasshopper plugin", L.grasshopper.installed ? `groundtrack ${esc(L.grasshopper.version || "")}` : "not installed", `<span class="st ${L.grasshopper.conflict ? "warn" : L.grasshopper.installed ? "ok" : "warn"}">${L.grasshopper.conflict ? "twice" : L.grasshopper.installed ? "installed" : "missing"}</span>`)}
    </div>
    <div class="group">Recent kits</div>
    <div class="list">${L.kits.slice(0, 5).map((k) => `<div class="row" style="cursor:default"><span class="avatar sm c${k.source === "Grasshopper" ? 4 : 0}">${ICON.swirl}</span><span style="min-width:0"><span class="row-title">${esc(k.site)}</span><span class="row-sub">${esc(k.source)} · ${k.modules.length} modules</span></span><span class="row-end">${k.scene ? `<button class="btn xs primary" data-open="${esc(k.scene)}">Open</button>` : ""}</span></div>`).join("") || '<div class="empty">No kits yet</div>'}</div>
    <div class="btns" style="padding:12px 4px"><button class="btn sm" id="lp-go">Houdini & Grasshopper →</button></div>`;
  bindFiles(el);
  $("#lp-go").onclick = () => goto(8);
}

/* ------------------------------------------------------------------ new site */
function newSiteDrawer() {
  drawer("New site", `<div class="d-title">New site</div><div class="d-sub">One clip of one place. Settings start from a template; everything can be changed later.</div>
    <div class="card"><div class="fields" style="grid-template-columns:1fr">
      <label class="field">Name<input id="ns-name" placeholder="e.g. trafalgar6377" spellcheck="false"></label>
      <div class="field">Template<div class="seg" id="ns-tpl"><button aria-selected="true" data-tpl="plaza">Plaza (people)</button><button aria-selected="false" data-tpl="motorway">Road (vehicles)</button></div></div>
      <label class="field">Video<div class="with"><input id="ns-video" spellcheck="false"><button class="btn sm" data-pick="video">Browse…</button></div></label>
      <label class="field">Aerial (GeoTIFF, EPSG:27700)<div class="with"><input id="ns-map" spellcheck="false"><button class="btn sm" data-pick="geotiff">Browse…</button></div></label>
    </div>
    <div class="btns"><button class="btn neon" id="ns-go">${ICON.plus} Create site</button></div></div>`, (b) => {
    let tpl = "plaza";
    $$("[data-tpl]", b).forEach((x) => x.onclick = () => { tpl = x.dataset.tpl; $$("[data-tpl]", b).forEach((y) => y.setAttribute("aria-selected", String(y === x))); segThumb($("#ns-tpl", b)); });
    $$("[data-pick]", b).forEach((x) => x.onclick = async () => { const p = await pickPath("file", x.dataset.pick); if (p) $(x.dataset.pick === "video" ? "#ns-video" : "#ns-map", b).value = p; });
    $("#ns-name", b).focus();
    $("#ns-go", b).onclick = async () => {
      try {
        const r = await post("/api/new-site", { site: $("#ns-name", b).value.trim(), template: tpl, video: $("#ns-video", b).value.trim(), geotiff: $("#ns-map", b).value.trim() });
        closeDrawer(); toast("Site created", "ok");
        await loadOverview(); selectSite(r.config, 1);
      } catch (e) { toast(e.message, "error"); }
    };
  });
}
$("#new-site").onclick = newSiteDrawer;

/* ------------------------------------------------------------------ menu, keys */
$("#menu-btn").onclick = (e) => { e.stopPropagation(); const m = $("#menu"); m.hidden = !m.hidden; $("#menu-btn").setAttribute("aria-expanded", String(!m.hidden)); };
document.addEventListener("click", (e) => { if (!$("#menu").hidden && !$("#menu").contains(e.target)) $("#menu").hidden = true; });
$$("[data-menu]").forEach((b) => b.onclick = async () => {
  $("#menu").hidden = true;
  const m = b.dataset.menu;
  if (m === "new-site") newSiteDrawer();
  else if (m === "open-project") openFolder(`${A.root}/sites`);
  else if (m === "add-library") addLibrary();
  else if (m === "refresh") { A.links = null; A.sources = null; await loadOverview(); await refreshSite(); renderWork(); toast("Refreshed"); }
});

document.addEventListener("keydown", (e) => {
  if (!$("#modal").hidden) { if (e.key === "Escape") A.modalDone && A.modalDone(false); if (e.key === "Enter") A.modalDone && A.modalDone(true); return; }
  if (!$("#lightbox").hidden) {
    if (e.key === "Escape") closeLB(); else if (e.key === "ArrowLeft") $("#lb-prev").click(); else if (e.key === "ArrowRight") $("#lb-next").click();
    return;
  }
  if (!$("#drawer").hidden) { if (e.key === "Escape") closeDrawer(); return; }
  const t = e.target;
  if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT") && e.key !== "Escape") return;
  if (A.mounted && A.mounted.key(e)) { e.preventDefault(); return; }
  if (e.key === "Escape" && isFocus()) goto(A.prevStep || 5);
});

/* ------------------------------------------------------------------ window frame */
// With the custom frame (desktop_frame.py) the top bar is the title bar: these are its buttons.
window.gtFrame = (st) => {
  const on = !!(st && st.custom), root = document.documentElement;
  root.classList.toggle("custom-frame", on);
  root.classList.toggle("maximized", on && !!st.maximized);
  $("#winctl").hidden = !on;
  $$("[data-edge]").forEach((e) => e.hidden = !on);
};
const winApi = () => window.pywebview && window.pywebview.api;
$$("[data-win]").forEach((b) => b.onclick = () => {
  const api = winApi(); if (!api) return;
  ({ min: () => api.minimize(), max: () => api.toggle_maximize(), close: () => api.close() })[b.dataset.win]();
});
$$("[data-edge]").forEach((e) => e.addEventListener("mousedown", (ev) => {
  if (ev.button === 0 && winApi()) { ev.preventDefault(); winApi().resize(e.dataset.edge); }
}));
window.addEventListener("pywebviewready", () => { const api = winApi(); if (api && api.frame) api.frame().then(window.gtFrame).catch(() => {}); });

/* ------------------------------------------------------------------ start */
(async function start() {
  try { await loadOverview(); }
  catch (e) { $("#work").innerHTML = `<div class="body"><div class="note bad">The local server did not answer: ${esc(e.message)}</div></div>`; return; }
  segThumb($("#tabs"));
  get("/api/composites").then((d) => { A.composites = d.composites; renderSteps(); }).catch(() => {});
  loadLinks().then(() => renderSteps()).catch(() => {});
  let cfg = null;
  try { cfg = localStorage.getItem("gt.config"); } catch {}
  if (!A.sites.some((s) => s.config === cfg)) cfg = (A.sites[0] || {}).config || null;
  if (cfg) await selectSite(cfg); else { renderHero(); renderWork(); }
  setInterval(() => { loadOverview().catch(() => {}); pollLog(); }, 2000);
})();
