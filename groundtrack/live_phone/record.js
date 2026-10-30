// motus live: the phone as the recorder (realtime, plans/live_controller/10, steps 1-2). Only while
// the laptop is in the realtime mode (the hub says so: phone.js shows the Record button). The rear
// camera's frames go as JPEG over wss to the live hub (/ws/record) and on to the laptop only,
// where they are tracked and blurred before anything is shown; with each frame the phone's pose
// (pose v1 q). Nothing goes anywhere else.
// Calibration (step 2, no button): a level shows the horizon as the sensors see it; the laptop
// calibrates once the phone has been held still for a second (a tick), at the height shown
// (remembered on this phone; tap to change). It checks the people it sees against 1.7 m: when
// they are off, this page offers the height that fits and tilt nudges. A new height or tilt, or
// Recalibrate, is a new recording (a new pass).

const FPS = 30;                     // frames a second, at most (the laptop's package rate)
const LONG_SIDE = 1920;             // frames at most this wide; the laptop may ask for less (720p)
const QUALITY = 0.7;                // JPEG
const MAX_BUFFER = 1.5e6;           // bytes waiting in the socket: more, and a frame is skipped
const BACK_S = 10;                  // a dropped link: back within this, the recording goes on
const HEIGHT_KEY = "motus-rec-height";
export const HEIGHT = {def: 1.5, min: 0.3, max: 5};      // m (live_track.HEIGHT_M, HEIGHT_RANGE)
const TILT_MAX = 20;                // deg: the tilt nudge, at most (live_track.TILT_NUDGE_MAX)
const VFOV_HALF = Math.atan(540 / 1597.7);      // the lens's half height (live_track.FOCAL_1920)
const CAL_S = 1;                    // the still second (live_track.CAL_S)

/** [u32 little-endian header length][header JSON]: what goes before a frame's JPEG. */
export function frameHeader(head) {
  const json = new TextEncoder().encode(JSON.stringify(head));
  const out = new Uint8Array(4 + json.length);
  new DataView(out.buffer).setUint32(0, json.length, true);
  out.set(json, 4);
  return out;
}

/** The frame size for a camera of w x h: the long side at most longSide (the laptop's choice,
 *  320 to LONG_SIDE), both even. */
export function frameSize(w, h, longSide = LONG_SIDE) {
  const most = Math.min(LONG_SIDE, Math.max(320, Number(longSide) || LONG_SIDE));
  const s = Math.min(1, most / Math.max(w, h));
  const even = v => Math.max(2, 2 * Math.round(v * s / 2));
  return [even(w), even(h)];
}

/** A camera height in metres: in range, to the centimetre (not a number: the default). */
export function cleanHeight(v) {
  v = v == null || v === "" ? NaN : Number(v);
  if (!Number.isFinite(v)) return HEIGHT.def;
  return Math.round(Math.min(HEIGHT.max, Math.max(HEIGHT.min, v)) * 100) / 100;
}

/** The horizon on the picture for the phone's tilt (deg, + up) and roll (deg, + clockwise): its
 *  turn (deg, CSS rotate) and how far it sits above the middle, as a share of half the picture's
 *  height (the lens: live_track.FOCAL_1920). Tilted down, the horizon is above the middle. */
export function horizon(tiltUp, rollCw) {
  return {turn: -rollCw, up: -Math.tan(tiltUp * Math.PI / 180) / Math.tan(VFOV_HALF)};
}

if (typeof document !== "undefined") boot();

function boot() {
  const $ = id => document.getElementById(id);
  const ui = Object.fromEntries(["rec", "msg", "recview", "recvideo", "recdot", "recst", "rectime",
    "recfps", "rectr", "reclag", "reccal", "recmsg", "recstop", "lvline", "rechold", "recholdtxt",
    "recholdbar", "reccheck", "reccheckmsg", "recusehm", "recheight", "recheightval",
    "recheightset", "recrecal"].map(id => [id, $(id)]));
  if (!ui.rec) return;
  const K = new URLSearchParams(location.search).get("k") || "";
  let ws = null, stream = null, wake = null, phone = null;
  let lastQ = null, seq = 0, sent = 0, skipped = 0, busy = false, size = null, camSize = null;
  let started = false, recording = false, stopping = false, resume = false, lostAt = 0, retry = 0;
  let canvas = null, ctx = null, recId = null, lastShot = 0;
  let height = cleanHeight((() => { try { return localStorage.getItem(HEIGHT_KEY); } catch (e) { return null; } })());
  let tiltOffset = 0, recal = false, calAt = 0, recHeight = null, levelDue = false;
  showHeight();

  ui.rec.addEventListener("click", () => {
    say("");
    if (!window.isSecureContext) return say("Open the https link from the QR code.");
    // both asked from the tap: iPhones allow the motion sensors and the camera only so
    const asks = [];
    if (typeof DeviceOrientationEvent !== "undefined" &&
        typeof DeviceOrientationEvent.requestPermission === "function")
      asks.push(DeviceOrientationEvent.requestPermission().catch(() => "denied"));
    const cam = navigator.mediaDevices && navigator.mediaDevices.getUserMedia
      ? navigator.mediaDevices.getUserMedia({audio: false, video: {facingMode: {ideal: "environment"},
          width: {ideal: 1920}, height: {ideal: 1080}, frameRate: {ideal: FPS}}})
      : Promise.reject(new Error("no camera access in this browser"));
    Promise.all([cam, ...asks]).then(([s, motion]) => {
      stream = s;
      if (motion === "denied")
        note("No motion access: the laptop assumes the camera level, 10° down.");
      begin();
    }, e => say(`The camera did not open (${e && (e.name || e.message) || e}).`));
  });

  async function begin() {
    window.dispatchEvent(new CustomEvent("motus:record"));     // phone.js lets go of its player
    phone = await import(new URL(`phone.js${new URL(import.meta.url).search}`, import.meta.url));
    window.addEventListener("deviceorientation", e => {
      if (e.alpha == null || e.beta == null || e.gamma == null) return;
      let q = phone.cameraQuaternion(e.alpha, e.beta, e.gamma, phone.screenAngle());
      if (lastQ && q[0] * lastQ[0] + q[1] * lastQ[1] + q[2] * lastQ[2] + q[3] * lastQ[3] < 0)
        q = q.map(v => -v);
      lastQ = q;
      if (!levelDue) { levelDue = true; requestAnimationFrame(level); }
    });
    ui.recview.hidden = false;
    $("start").hidden = true;
    ui.recvideo.srcObject = stream;
    try { await ui.recvideo.play(); } catch (e) { /* autoplay muted: plays anyway */ }
    await new Promise(r => ui.recvideo.videoWidth ? r() : ui.recvideo.addEventListener("loadedmetadata", r, {once: true}));
    const vw = ui.recvideo.videoWidth, vh = ui.recvideo.videoHeight;
    if (vh > vw) note("Portrait: the laptop's camera is set for landscape. Done, turn the phone, Record again.");
    camSize = [vw, vh];                              // the frame size: when the hub says hello
    lock();
    connect();
    tick();
  }

  function say(text) { if (ui.msg) ui.msg.textContent = text; }
  function note(text) { ui.recmsg.textContent = text; ui.recmsg.hidden = !text; }
  function state(text, cls = "") {
    ui.recst.textContent = text;
    ui.recdot.className = `recdot ${cls}`;
  }

  async function lock() {
    try {
      if ("wakeLock" in navigator && document.visibilityState === "visible" && !wake) {
        wake = await navigator.wakeLock.request("screen");
        wake.addEventListener("release", () => { wake = null; });
      }
    } catch (e) { /* not granted: the screen may dim */ }
  }
  document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible" && stream) lock(); });

  // ---- the level: the horizon where the sensors put it, over the picture (object-fit: contain)
  function level() {
    levelDue = false;
    if (!lastQ || !phone) return;
    const [up, roll] = phone.tiltRoll(lastQ);
    const v = ui.recvideo, box = v.getBoundingClientRect();
    const ar = v.videoWidth && v.videoHeight ? v.videoWidth / v.videoHeight : 16 / 9;
    const half = Math.min(box.height, box.width / ar) / 2;
    const hz = horizon(up, roll), dy = Math.max(-half, Math.min(half, hz.up * half));
    ui.lvline.style.transform = `translateY(${(-dy).toFixed(1)}px) rotate(${hz.turn.toFixed(2)}deg)`;
    ui.lvline.classList.toggle("level", Math.abs(roll) < 1);
  }

  // ---- the height: remembered on this phone; a new one while recording asks for Recalibrate
  function showHeight() {
    ui.recheightval.textContent = `${height.toFixed(2)} m`;
    const ask = recording && recHeight != null && Math.abs(recHeight - height) > 0.004;
    ui.recrecal.classList.toggle("ask", ask);
    ui.recrecal.textContent = ask ? "Recalibrate at the new height" : "Recalibrate";
  }
  function setHeight(v) {
    height = cleanHeight(v);
    try { localStorage.setItem(HEIGHT_KEY, String(height)); } catch (e) { /* this page only */ }
    showHeight();
  }
  ui.recheight.addEventListener("click", () => { ui.recheightset.hidden = !ui.recheightset.hidden; });
  for (const b of ui.recheightset.querySelectorAll("[data-h]"))
    b.addEventListener("click", () => setHeight(height + Number(b.dataset.h)));

  // ---- the people check's nudges: each is a new calibration (a new recording)
  ui.recusehm.addEventListener("click", () => {
    const v = Number(ui.recusehm.dataset.h);
    if (Number.isFinite(v)) { setHeight(v); recalibrate(); }
  });
  for (const b of ui.reccheck.querySelectorAll("[data-tilt]"))
    b.addEventListener("click", () => {
      tiltOffset = Math.max(-TILT_MAX, Math.min(TILT_MAX, tiltOffset + Number(b.dataset.tilt)));
      recalibrate();
    });
  ui.recrecal.addEventListener("click", () => recalibrate());

  /* This recording is saved and a new one starts at the height and tilt now (the laptop waits
     for a still second again). */
  function recalibrate() {
    if (!recording || !ws || ws.readyState !== 1) return;
    recal = true;
    recording = false;
    ui.recrecal.disabled = ui.recstop.disabled = true;
    ui.reccheck.hidden = true;
    ws.send(JSON.stringify({type: "rec", action: "stop"}));
    state("recalibrating");
  }

  function sendStart() {
    if (ws && ws.readyState === 1)
      ws.send(JSON.stringify({type: "rec", action: "start", w: size[0], h: size[1], resume,
                              height_m: height, tilt_offset: tiltOffset}));
  }

  // ---- the link: one socket; a drop while recording comes back with "resume" (live_track.LOST_S)
  function connect() {
    if (ws && ws.readyState <= 1) return;
    const id = (() => { try { return localStorage.getItem("motus-live-id") || ""; } catch (e) { return ""; } })();
    const s = new WebSocket(`wss://${location.host}/ws/record?k=${encodeURIComponent(K)}&id=${encodeURIComponent(id)}`);
    ws = s;
    state(resume ? "reconnecting" : "connecting");
    s.onopen = () => { if (ws === s) retry = 0; };       // start: on the hub's hello (onMessage)
    s.onmessage = e => { if (ws === s) onMessage(e.data); };
    s.onclose = e => {
      if (ws !== s) return;
      ws = null;
      if (e.code === 4004) return end("Another phone is the recorder.");
      if (e.code === 4005) return end("The laptop left the realtime mode.");
      if (e.code === 4002) return end("This page is open elsewhere.");
      if (stopping) return end("");
      if (recording || recal) {                      // recalibrating: back with a new recording
        if (!lostAt) lostAt = performance.now();
        resume = recording;
        if (performance.now() - lostAt > BACK_S * 1000) return end("The link dropped for too long: the laptop saved what it had.");
        state("link dropped", "lost");
      } else if (!started) {
        return end("Realtime is off on the laptop, or the hub is not reachable.");
      }
      setTimeout(connect, Math.min(2000, 250 * 2 ** retry++));
    };
  }

  function onMessage(data) {
    let m;
    try { m = JSON.parse(data); } catch (e) { return; }
    if (m.type !== "rec") return;
    if (m.hello) {                                   // the laptop's frame size (1080p or 720p)
      if (!size) {
        size = frameSize(camSize[0], camSize[1], m.long_side);
        canvas = document.createElement("canvas");
        [canvas.width, canvas.height] = size;
        ctx = canvas.getContext("2d");
      }
      sendStart();
      return;
    }
    if (m.error) return note(m.error);
    if (m.started) {
      started = recording = true;
      recal = resume = false;
      recId = m.started;
      lostAt = calAt = 0;
      note("");
      ui.recstop.disabled = ui.recrecal.disabled = false;
      state("recording", "on");
      return;
    }
    if (m.stopped) {
      if (recal) { resume = false; sendStart(); return; }      // recalibrating: a new one at once
      return state("saving");
    }
    if (recording && m.recording && m.id === recId) {
      recHeight = m.height_m;
      showHeight();
      ui.rectime.textContent = clock(m.seconds);
      ui.rectr.textContent = `${m.fps_tracked ?? "–"} tracked`;
      ui.reclag.textContent = m.lag_ms != null ? `${m.lag_ms} ms` : "– ms";
      calibration(m);
    } else if (stopping && m.saved && m.saved.id === recId) {     // this recording, saved
      end(m.saved.complete ? `Saved: ${m.saved.seconds} s, ${m.saved.tracks} tracks. Open it on the laptop.`
                           : `Saved without a package: ${(m.saved.notes || []).join("; ")}`);
    }
  }

  /* Hold still (the laptop's still second as a bar), then a tick; once calibrated, the camera,
     and the nudges when the people it sees do not come out near 1.7 m. */
  function calibration(m) {
    const cam = m.camera, chk = m.check;
    if (m.mode === "loading") {
      ui.reccal.textContent = "the model is loading";
    } else if (!cam) {
      ui.reccal.textContent = "hold still: calibrating";
      ui.rechold.hidden = false;
      ui.rechold.classList.remove("done");
      ui.recholdtxt.textContent = "Hold still";
      ui.recholdbar.style.width = `${Math.round(100 * Math.min(1, (m.still_s || 0) / CAL_S))}%`;
    } else {
      if (!calAt) {                                  // the tick, for a moment
        calAt = performance.now();
        ui.rechold.hidden = false;
        ui.rechold.classList.add("done");
        ui.recholdtxt.textContent = "✓ Calibrated";
        ui.recholdbar.style.width = "100%";
      } else if (performance.now() - calAt > 1800) ui.rechold.hidden = true;
      const nudge = m.tilt_offset_deg ? ` (${m.tilt_offset_deg > 0 ? "+" : ""}${m.tilt_offset_deg}°)` : "";
      ui.reccal.textContent = `tilt ${cam.tilt_deg}°${nudge} · roll ${cam.roll_deg}° · ` +
        `${Number(cam.height_m).toFixed(2)} m` + (chk ? ` · people ${chk.median_m} m` : "");
    }
    const off = !!(cam && chk && !chk.ok);
    ui.reccheck.hidden = !off;
    if (off) {
      ui.reccheckmsg.textContent = `People look ${chk.median_m} m tall ` +
        `(${chk.error_pct > 0 ? "+" : ""}${chk.error_pct} %): the height or the tilt is off.`;
      ui.recusehm.dataset.h = String(chk.height_m);
      ui.recusehm.textContent = `Height ${Number(chk.height_m).toFixed(2)} m`;
      ui.recusehm.hidden = Math.abs(chk.height_m - height) < 0.01;
    }
  }

  ui.recstop.addEventListener("click", () => {
    if (!recording) return;
    recording = false;
    stopping = true;
    ui.recstop.disabled = ui.recrecal.disabled = true;
    if (ws && ws.readyState === 1) ws.send(JSON.stringify({type: "rec", action: "stop"}));
    state("saving");
  });

  function end(text) {
    recording = recal = false;
    if (stream) for (const t of stream.getTracks()) t.stop();
    stream = null;
    if (ws) { const s = ws; ws = null; s.close(1000); }
    state("ended");
    ui.rechold.hidden = ui.reccheck.hidden = true;
    ui.recrecal.disabled = true;
    ui.recstop.textContent = "Done";
    ui.recstop.disabled = false;
    ui.recstop.onclick = () => location.reload();       // back to Record / Manipulate
    note(text);
  }

  // ---- frames: the newest picture each time the last one is out (latest wins, never queued)
  function tick() {
    if (!stream) return;
    if (ui.recvideo.requestVideoFrameCallback) ui.recvideo.requestVideoFrameCallback(tick);
    else setTimeout(tick, 1000 / FPS);
    if (!recording || !ws || ws.readyState !== 1) return;
    const t = performance.now() / 1000;              // capture time: the laptop's frame numbers
    if (t - lastShot < 0.9 / FPS) return;
    if (busy || ws.bufferedAmount > MAX_BUFFER) { skipped++; return; }   // the stats count these
    lastShot = t;
    busy = true;
    const q = lastQ;
    ctx.drawImage(ui.recvideo, 0, 0, size[0], size[1]);
    canvas.toBlob(blob => {
      busy = false;
      if (!blob || !ws || ws.readyState !== 1 || !recording) return;
      const head = {seq: seq++, t: +t.toFixed(4), w: size[0], h: size[1],
                    q: q ? q.map(v => +v.toFixed(6)) : null, skip: skipped};
      ws.send(new Blob([frameHeader(head), blob]));
      sent++;
    }, "image/jpeg", QUALITY);
  }

  setInterval(() => {
    ui.recfps.textContent = `${sent} fps${size ? ` · ${Math.min(...size)}p` : ""}`;
    sent = 0;
  }, 1000);

  function clock(s) {
    s = Math.max(0, s || 0);
    return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  }
}
