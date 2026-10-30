/* motus Live, Realtime (plans/live_controller/10): one phone films, this laptop tracks.
   This page turns the mode on (/api/live/realtime start: the phone hub, the model loading in the
   background) and shows the recorder's state, the phones' code, the frame size and the recordings.
   Open Live (now) (or the site picker's Realtime · now) is the Live view of the mode, which draws
   each recording as it is tracked (step 3). Picking another site goes to the Live view, which
   turns the mode off. A saved recording opens as plain pre-tracked Live (ground C, the grid: a
   recording has no aerial). */

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

async function postJson(url, body) {
  const r = await fetch(url, { method: "POST", headers: { "X-Groundtrack": "1", "Content-Type": "application/json" },
                               body: JSON.stringify(body) });
  let d = {};
  try { d = await r.json(); } catch {}
  if (!r.ok || d.error) throw new Error(d.error || `${r.status}`);
  return d;
}

const clock = (s) => (s == null ? "–" : `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`);
const openRec = (run) => { location.href = `/live?run=${encodeURIComponent(run)}&ground=C`; };
let recSig = "";

// the recorder's frame size (the next recording's; the hub tells the phone), kept on this laptop
const SIZE_KEY = "motus-rt-long-side";
let longSide = (() => { try { return Number(localStorage.getItem(SIZE_KEY)) || 1920; } catch { return 1920; } })();
if (![1920, 1280].includes(longSide)) longSide = 1920;
function showSize(v) {
  $$("#rt-size button").forEach((b) => b.setAttribute("aria-selected", String(Number(b.dataset.size) === v)));
}
$$("#rt-size button").forEach((b) => b.onclick = async () => {
  const v = Number(b.dataset.size);
  if (v === longSide) return;
  longSide = v;
  showSize(v);
  try { localStorage.setItem(SIZE_KEY, String(v)); } catch {}
  try { show(await postJson("/api/live/realtime", { action: "size", long_side: v })); } catch (e) { $("#rt-state").title = e.message; }
});
showSize(longSide);

function show(st) {
  const rec = st.recording, cam = st.camera;
  const [text, cls] = st.mode === "error" ? [`Model error`, "bad"]
    : st.mode === "loading" ? ["Loading the model", "wait"]
    : rec && st.lost ? ["Recorder dropped out", "bad"]
    : rec ? ["Recording", "on"]
    : (st.saving || []).length ? ["Saving", "wait"]
    : ["Waiting for a recorder", ""];
  $("#rt-state").textContent = text;
  $("#rt-dot").className = `rt-dot ${cls}`;
  $("#rt-state").title = st.error || "";
  $("#rt-time").textContent = rec ? clock(st.seconds) : "–";
  $("#rt-in").textContent = rec ? `${st.fps_in}` : "–";
  $("#rt-tr").textContent = rec ? `${st.fps_tracked}` : "–";
  $("#rt-lag").textContent = rec && st.lag_ms != null ? `${st.lag_ms} ms` : "–";
  $("#rt-people").textContent = rec ? `${st.people}` : "–";
  $("#rt-cam").textContent = cam ? `${cam.tilt_deg}° down · roll ${cam.roll_deg}°` : rec ? "calibrating" : "–";
  $("#rt-cam").className = "small";
  $("#rt-end").disabled = !rec;
  const recs = st.recordings || [];
  const sig = JSON.stringify(recs.map((r) => [r.id, r.complete, r.growing, r.seconds, r.fps_tracked]));
  if (sig === recSig) return;                // rebuilt only when it changes (never under the pointer)
  recSig = sig;
  const el = $("#rt-list");
  el.innerHTML = "";
  if (!recs.length) el.innerHTML = `<p class="rt-empty">None yet.</p>`;
  for (const r of recs.slice(0, 12)) {
    const row = document.createElement("div");
    row.className = `rt-rec${r.growing ? " saving" : ""}`;
    const meta = r.growing ? "recording or saving…"
      : r.complete ? `${clock(r.seconds)} · ${r.tracks} tracks` +
        (r.fps_tracked != null ? ` · ${r.fps_tracked} fps tracked` : "") : (r.notes[0] || "no package");
    row.innerHTML = `<div><div class="name"></div><div class="meta"></div></div>`;
    row.querySelector(".name").textContent = r.title;
    row.querySelector(".meta").textContent = meta;
    if (r.complete) {
      const b = document.createElement("button");
      b.className = "btn sm";
      b.textContent = "Open in Live";
      b.onclick = () => openRec(r.run);
      row.appendChild(b);
    }
    el.appendChild(row);
  }
}

async function hub() {
  try {
    const d = await postJson("/api/live/hub", { action: "start" });
    const u = (d.phone_urls || []).find((x) => x.likely) || (d.phone_urls || [])[0];
    $("#qr").innerHTML = u ? u.qr_svg : "";
    $("#phone-url").textContent = u ? u.url : "No network address found";
    $("#phones").textContent = `${d.phones} phone${d.phones === 1 ? "" : "s"} connected` +
      (d.recorder ? " · a recorder" : "");
  } catch (e) { $("#phones").textContent = `Phone hub: ${e.message}`; }
}

async function sites() {
  let list = [];
  try { list = (await (await fetch("/api/live/sites")).json()).sites || []; } catch {}
  const el = $("#site-list");
  el.innerHTML = "";
  for (const e of list) {
    const b = document.createElement("button");
    b.textContent = `${e.title} · ${e.cameras}`;
    b.onclick = () => { location.href = `/live?site=${encodeURIComponent(e.site)}`; };
    el.appendChild(b);
  }
  const now = document.createElement("button");
  now.textContent = "Realtime · now";
  now.title = "The Live view, drawing each recording as it is tracked";
  now.setAttribute("aria-pressed", "true");
  now.onclick = () => { location.href = "/live?realtime=1"; };
  el.appendChild(now);
}

// the window (the app's own frame), as the Live view
const winApi = () => window.pywebview && window.pywebview.api;
window.gtFrame = (st) => {
  const on = !!(st && st.custom), root = document.documentElement;
  root.classList.toggle("custom-frame", on);
  root.classList.toggle("maximized", on && !!st.maximized);
  $("#winctl").hidden = !on;
  $$("[data-edge]").forEach((e) => { e.hidden = !on; });
};
$$("[data-win]").forEach((b) => b.onclick = () => {
  const api = winApi(); if (!api) return;
  ({ min: () => api.minimize(), max: () => api.toggle_maximize(), close: () => api.close() })[b.dataset.win]();
});
$$("[data-edge]").forEach((e) => e.addEventListener("mousedown", (ev) => {
  if (ev.button === 0 && winApi()) { ev.preventDefault(); winApi().resize(e.dataset.edge); }
}));
window.addEventListener("pywebviewready", () => { const api = winApi(); if (api && api.frame) api.frame().then(window.gtFrame).catch(() => {}); });

$("#back").onclick = async () => {
  try { await postJson("/api/live/realtime", { action: "stop" }); } catch {}
  location.href = "/";
};
$("#rt-live").onclick = () => { location.href = "/live?realtime=1"; };
$("#rt-end").onclick = async () => {
  $("#rt-end").disabled = true;
  try { show(await postJson("/api/live/realtime", { action: "end" })); } catch (e) { $("#rt-state").textContent = e.message; }
};

(async function start() {
  sites();
  try {
    show(await postJson("/api/live/realtime", { action: "start", long_side: longSide }));
  } catch (e) {
    $("#rt-state").textContent = `Realtime did not start: ${e.message}`;
    $("#rt-dot").className = "rt-dot bad";
    return;
  }
  hub();
  setInterval(async () => {
    try { show(await (await fetch("/api/live/realtime")).json()); } catch {}
  }, 500);
  setInterval(hub, 3000);
})();
