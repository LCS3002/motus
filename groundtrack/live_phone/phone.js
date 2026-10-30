// groundtrack live: the phone as the camera (rotation) and a remote for the Live view.
// Sends pose v1 (plans/live_controller/00_README.md) to the live hub over wss. Motion sensors
// only: never the camera, nothing but poses and remote buttons leaves the phone.

const D = Math.PI / 180;

/** Hamilton product a·b of quaternions [w, x, y, z]. */
export function qmul(a, b) {
  const [aw, ax, ay, az] = a, [bw, bx, by, bz] = b;
  return [aw * bw - ax * bx - ay * by - az * bz,
          aw * bx + ax * bw + ay * bz - az * by,
          aw * by - ax * bz + ay * bw + az * bx,
          aw * bz + ax * by - ay * bx + az * bw];
}

function about(i, deg) {                       // rotation about axis i (0 x, 1 y, 2 z)
  const h = deg * D / 2, q = [Math.cos(h), 0, 0, 0];
  q[1 + i] = Math.sin(h);
  return q;
}

// camera (OpenCV: x right, y down, z forward) in screen axes (x right, y up, z out of the
// glass): 180° about x. The back camera looks out of the back, so forward = -screen z.
const CAMERA_FROM_SCREEN = [0, 1, 0, 0];

/**
 * deviceorientation (alpha, beta, gamma in degrees: W3C intrinsic Z-X'-Y') and the screen's
 * angle (screen.orientation.angle) -> the back camera's world_from_camera quaternion
 * [w, x, y, z], OpenCV axes, in the gravity frame (z up, heading arbitrary).
 * The screen's angle turns the device axes into the axes of the screen as shown, so in a
 * landscape hold camera x runs along the phone's long side.
 */
export function cameraQuaternion(alpha, beta, gamma, screenAngle) {
  let q = qmul(qmul(about(2, alpha), about(0, beta)), about(1, gamma));   // world_from_device
  q = qmul(q, about(2, -screenAngle));                                   // world_from_screen
  q = qmul(q, CAMERA_FROM_SCREEN);                                       // world_from_camera
  const n = Math.hypot(...q);
  return q.map(v => v / n);
}

/** The camera's x, y, z axes in the world (the columns of q's rotation matrix). */
export function axes(q) {
  const [w, x, y, z] = q;
  return [[1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)],
          [2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)],
          [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)]];
}

/** Tilt (up +) and roll (clockwise +) of the camera in degrees, against gravity. */
export function tiltRoll(q) {
  const [cx, cy, cz] = axes(q);
  return [Math.asin(Math.max(-1, Math.min(1, cz[2]))) / D, Math.atan2(-cx[2], -cy[2]) / D];
}

/** screen.orientation.angle (or the older window.orientation) in [0, 360). */
export function screenAngle() {
  const so = globalThis.screen && globalThis.screen.orientation;
  const a = so && typeof so.angle === "number" ? so.angle
    : typeof globalThis.orientation === "number" ? globalThis.orientation : 0;
  return ((a % 360) + 360) % 360;
}

// gain slider 0..1000 on a log scale, 0.1-3x (1x about two thirds along)
const G0 = Math.log(0.1), G1 = Math.log(3);
export const gainOf = s => Math.exp(G0 + (G1 - G0) * s / 1000);
export const sliderOf = g => Math.round(1000 * (Math.log(g) - G0) / (G1 - G0));

// --------------------------------------------------------------------------- page
if (typeof document !== "undefined") boot();

function boot() {
  const $ = id => document.getElementById(id);
  const ui = Object.fromEntries(["start", "go", "msg", "hubst", "conn", "dot", "hz", "rtt",
    "viewst", "spot", "spotbox", "who", "cam", "camname", "camarea",
    "camgo", "picker", "pback", "ptitle", "pnone", "pclose", "plist", "zero", "gain",
    "gainval", "play", "section", "space", "spacest", "scrub", "tcur", "tdur",
    "horizon", "tilt", "turn", "note", "rec", "recnote", "sensors"].map(id => [id, $(id)]));
  const K = new URLSearchParams(location.search).get("k") || "";
  const PID = pageId();
  // the fonts of desktop_web/tokens.css, added here: like every file of this page they are
  // fetched with the pairing code, which a url in phone.css cannot carry
  for (const [family, file, weight] of [["Geist", "Geist-Variable", "100 900"],
                                        ["Spline Sans Mono", "SplineSansMono-Variable", "300 700"]]) {
    try {
      const f = new FontFace(family, `url(fonts/${file}.woff2?k=${encodeURIComponent(K)})`, {weight});
      document.fonts.add(f);
      f.load().catch(() => {});                  // else the system font
    } catch (e) { /* no FontFace: the system font */ }
  }
  let ws = null, stopped = false, replaced = false, retry = 0, retryTimer = 0;
  let running = false, gotMotion = false, wake = null;
  let spaceDown = false, spaceDownAt = 0, spaceUpAt = -1e9, spaceTapAt = -1e9;
  let seq = 0, lastQ = null, budget = 1, lastT = 0, sent = 0;
  let pingId = 0, drawPending = false, sectionOn = false;
  let player = null, full = false, spotSig = "", shownSpot = null;     // the hub numbers phones 1-4
  const pings = new Map(), rtts = [];
  const state = {}, dragging = {scrub: false, gain: false};

  // ---- start: motion permission must be asked from the tap
  ui.go.addEventListener("click", () => {
    say("");
    if (!window.isSecureContext) return say("Open the https link from the QR code.");
    if (!K) return say("No pairing code: open the link from the QR code in the motus Live view.");
    const asks = [];
    if (typeof DeviceOrientationEvent !== "undefined" &&
        typeof DeviceOrientationEvent.requestPermission === "function")
      asks.push(DeviceOrientationEvent.requestPermission());
    if (typeof DeviceMotionEvent !== "undefined" &&
        typeof DeviceMotionEvent.requestPermission === "function")
      asks.push(DeviceMotionEvent.requestPermission().catch(() => "denied"));
    Promise.all(asks).then(res => {
      if (res.length && res[0] !== "granted")
        return say("Motion access was refused. Close Safari fully, open the link again and tap Allow.");
      window.addEventListener("deviceorientation", onOrient);
      running = true;
      ui.start.hidden = true;
      lockScreen();
      checkHold();
      setTimeout(() => { if (!gotMotion) note("No motion data from this phone yet."); }, 2000);
    }, e => say(`Motion access failed (${e && e.name || e}).`));
  });

  function say(text) { ui.msg.textContent = text; }
  function note(text) { ui.note.textContent = text; ui.note.hidden = !text; }

  async function lockScreen() {
    try {
      if ("wakeLock" in navigator && document.visibilityState === "visible" && !wake) {
        wake = await navigator.wakeLock.request("screen");
        wake.addEventListener("release", () => { wake = null; });
      }
    } catch (e) { /* not granted: the screen may dim */ }
  }

  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") {
      if (running) lockScreen();
      if (replaced) { replaced = false; note(""); }          // this page is the one in use again
      connect();
    } else {
      letGoSpace();
      if (lastQ) sendPose(lastQ, "lost");
    }
  });
  window.addEventListener("pagehide", () => letGoSpace());
  window.addEventListener("blur", () => letGoSpace());

  // ---- orientation -> pose, at most 60 Hz
  function onOrient(e) {
    if (e.alpha == null || e.beta == null || e.gamma == null) return;
    if (!gotMotion) { gotMotion = true; note(""); }
    let q = cameraQuaternion(e.alpha, e.beta, e.gamma, screenAngle());
    if (lastQ && q[0] * lastQ[0] + q[1] * lastQ[1] + q[2] * lastQ[2] + q[3] * lastQ[3] < 0)
      q = q.map(v => -v);                      // same hemisphere as the last one: no sign flips
    lastQ = q;
    const now = performance.now();
    budget = Math.min(2, budget + (now - lastT) * 0.06);    // 60 sends a second
    lastT = now;
    if (budget >= 1) {
      budget -= 1;
      sendPose(q, "ok");
    }
    draw();
  }

  function sendPose(q, status) {
    if (!ws || ws.readyState !== 1 || ws.bufferedAmount > 2048) return;   // late: skip, never queue
    ws.send(JSON.stringify({type: "pose", v: 1, src: "web", seq: seq++,
      t: performance.now() / 1000, q: q.map(v => +v.toFixed(6)), p: null, status}));
    sent++;
  }

  function draw() {
    if (drawPending) return;
    drawPending = true;
    requestAnimationFrame(() => {
      drawPending = false;
      const [tilt, roll] = tiltRoll(lastQ);
      // the horizon as the camera sees it: it drops as the camera tilts up, turns against the roll
      const y = Math.max(-1, Math.min(1, tilt / 45)) * 40;
      ui.horizon.setAttribute("transform", `translate(0 ${y.toFixed(1)}) rotate(${(-roll).toFixed(1)})`);
      ui.tilt.textContent = `tilt ${Math.round(tilt)}°  roll ${Math.round(roll)}°`;
    });
  }

  // ---- hub connection: one socket per page. connect() may be called from the retry timer,
  // the page coming back, a code check: it opens a socket only when none is open or opening
  // (a timer suspended in the background and the page coming back used to open two at once).
  function connect() {
    clearTimeout(retryTimer);
    retryTimer = 0;
    if (stopped || replaced || (ws && ws.readyState <= 1)) return;
    let s;
    try {
      s = new WebSocket(`wss://${location.host}/ws/phone?k=${encodeURIComponent(K)}&id=${PID}`);
    } catch (e) {
      return later();
    }
    ws = s;
    status("connecting");
    s.onopen = () => {
      if (ws !== s) return;
      retry = 0;
      status("connected");
      if (!spaceDown) control("space", false);                  // a new link starts let go
    };
    s.onmessage = e => { if (ws === s) onMessage(e.data); };
    s.onclose = e => {
      if (ws !== s) return;
      ws = null;
      player = null;
      showWho();
      dropSpace();
      showHub({views: 0});
      if (e.code === 4001) return codeChanged();
      if (e.code === 4002) {                     // the same page id connected again elsewhere
        replaced = true;
        status("in use on another page");
        return note("This link is open on another page. Come back to this one to use it here.");
      }
      if (e.code === 4003) {                     // four phones already: try again now and then
        full = true;
        status("four phones connected");
        note("Four phones are connected: this one joins when one of them leaves.");
        retry = Math.max(retry, 4);
        return later();
      }
      status("reconnecting");
      checkCode().then(ok => { if (ok) later(); });
    };
  }

  function later() {                             // at most one retry waiting
    if (retryTimer || ws || stopped || replaced) return;
    retryTimer = setTimeout(() => { retryTimer = 0; connect(); }, Math.min(3000, 300 * 2 ** retry++));
  }

  function pageId() {                            // kept across reloads: the hub keeps one socket per id
    const fresh = () => Array.from(crypto.getRandomValues(new Uint8Array(12)),
                                   b => b.toString(16).padStart(2, "0")).join("");
    try {
      let v = localStorage.getItem("motus-live-id");
      if (!/^[A-Za-z0-9_-]{8,64}$/.test(v || "")) {
        v = fresh();
        localStorage.setItem("motus-live-id", v);
      }
      return v;
    } catch (e) {
      return fresh();
    }
  }

  async function checkCode() {                 // a refused socket looks like any other failure
    try {
      const r = await fetch(`manifest.json?k=${encodeURIComponent(K)}`, {cache: "no-store"});
      if (r.status === 403) { codeChanged(); return false; }
    } catch (e) {
      if (!ws) status("hub not reachable");
    }
    return true;
  }

  function codeChanged() {
    stopped = true;
    status("pairing code changed");
    note("The pairing code changed: scan the new QR code in the motus Live view.");
  }

  function status(text) {
    ui.conn.textContent = ui.hubst.textContent = text;
    ui.dot.classList.toggle("ok", text === "connected");
  }

  function showHub(m) {
    ui.viewst.textContent = m.views ? "view open" : "view not open";
  }

  function onMessage(data) {
    let m;
    try { m = JSON.parse(data); } catch (e) { return; }
    if (m.type === "pong") {
      const t0 = pings.get(m.id);
      if (t0 !== undefined) {
        pings.delete(m.id);
        rtts.push(performance.now() - t0);
        if (rtts.length > 15) rtts.shift();
      }
    } else if (m.type === "state") {
      applyState(m);
    } else if (m.type === "hub") {
      if (typeof m.player === "number") {
        player = m.player;
        if (full) { full = false; note(""); }
        showWho();
      }
      showHub(m);
      showRealtime(m.realtime === true);
    }
  }

  // ---- realtime (plans/live_controller/10): the laptop's realtime mode offers Record (record.js)
  // beside Start (Manipulate). Record takes this page: it lets go of its player here.
  function showRealtime(on) {
    if (!running) {
      ui.rec.hidden = !on;
      ui.go.textContent = on ? "Manipulate" : "Start";
    }
    ui.recnote.hidden = !on;
    ui.sensors.hidden = on;
  }
  window.addEventListener("motus:record", () => {
    stopped = true;
    clearTimeout(retryTimer);
    retryTimer = 0;
    letGoSpace();
    const s = ws;
    ws = null;
    if (s) s.close(1000, "recording");
  });

  setInterval(() => {
    if (ws && ws.readyState === 1) {
      const id = ++pingId;
      pings.set(id, performance.now());
      if (pings.size > 20) pings.delete(pings.keys().next().value);
      ws.send(JSON.stringify({type: "ping", id}));
    }
    ui.hz.textContent = `${sent} Hz`;
    sent = 0;
    ui.rtt.textContent = rtts.length ? `${Math.round(median(rtts))} ms` : "– ms";
  }, 1000);

  function median(a) {
    const s = [...a].sort((x, y) => x - y), n = s.length;
    return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
  }

  // ---- remote
  function send(obj) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj)); }
  function control(action, value = null) { send({type: "control", action, value}); }

  ui.zero.addEventListener("click", () => {
    send({type: "zero"});
    ui.zero.classList.remove("flash");
    void ui.zero.offsetWidth;
    ui.zero.classList.add("flash");
  });
  ui.play.addEventListener("click", () => {
    state.playing = !state.playing;
    control(state.playing ? "play" : "pause");
    showState();
  });
  ui.section.addEventListener("click", () => {
    sectionOn = !sectionOn;
    control("section", sectionOn);
    showState();
  });

  // ---- space clutch: held while a finger is on the button (it turns the whole space), let go
  // on lift, cancel, the finger sliding off, the page hiding or the link dropping: never stuck.
  // A double-tap levels the space.
  ui.space.addEventListener("pointerdown", e => {
    e.preventDefault();
    try {        // touch captures the pointer implicitly: without it, sliding off fires pointerleave
      if (ui.space.hasPointerCapture(e.pointerId)) ui.space.releasePointerCapture(e.pointerId);
    } catch (err) { /* no capture to release */ }
    if (spaceDown || ui.space.classList.contains("off")) return;     // off: no spot of its own
    const now = performance.now();
    if (now - spaceTapAt < 320) {                // the second tap of a double-tap
      spaceTapAt = -1e9;
      control("level");
      ui.space.classList.remove("flash");
      void ui.space.offsetWidth;
      ui.space.classList.add("flash");
      return;
    }
    spaceDown = true;
    spaceDownAt = now;
    control("space", true);
    showSpace();
  });
  for (const type of ["pointerup", "pointercancel", "pointerleave"])
    ui.space.addEventListener(type, () => letGoSpace());
  ui.space.addEventListener("contextmenu", e => e.preventDefault());

  function letGoSpace() {
    if (!spaceDown) return;
    const now = performance.now();
    spaceDown = false;
    spaceUpAt = now;
    if (now - spaceDownAt < 250) spaceTapAt = now;              // a short tap: a second may follow
    control("space", false);
    showSpace();
  }

  function dropSpace() {                         // the link is gone: let go here (the hub lets go there)
    if (!spaceDown) return;
    spaceDown = false;
    spaceUpAt = performance.now();
    showSpace();
  }

  function showSpace() {
    // held elsewhere (the laptop's M) shows as on but not pressed; just after our own release
    // the view's state may still say held for a moment
    const elsewhere = !spaceDown && state.space_held === true && performance.now() - spaceUpAt > 600;
    const turned = !spaceDown && (state.space_tilted === true || elsewhere);
    ui.space.classList.toggle("held", spaceDown);
    ui.space.classList.toggle("turned", turned);
    ui.spacest.textContent = ui.space.classList.contains("off") ? "pick a camera"
      : spaceDown ? "turning the space" : elsewhere ? "held elsewhere"
      : turned ? "turned · double-tap: level" : "hold: turn the space";
  }

  // ---- cameras (the hub's spots). The view lists every camera of the site and, for a large
  // site, its areas (groups); the hub gives each camera to one phone, and the first phone the
  // start camera. Zero, gain and SPACE steer this phone's own camera; play and scrub are
  // everyone's. A small site: a plain list. A large one: an area, then a camera in it.
  const spotList = () => Array.isArray(state.spots) ? state.spots : [];
  const groupList = () => Array.isArray(state.groups) ? state.groups : [];
  const ownSpot = () => typeof state.spot === "string" ? state.spot : null;
  const takenBy = s => typeof s.player === "number" && s.player !== player ? s.player : null;
  const pick = {open: false, area: null, sig: ""};
  let asked, askedNote = false;                  // the camera last asked for (refused: say so)
  let siteName = null, hadSpots = false;         // the site last shown (only the laptop switches it)

  function claim(name) {
    letGoSpace();                                // SPACE never follows the phone to another camera
    asked = name;
    control("spot", name);
  }
  ui.spot.addEventListener("change", () => claim(ui.spot.value || null));
  ui.cam.addEventListener("click", () => openPicker());
  ui.pback.addEventListener("click", () => { pick.area = null; renderPicker(); });
  ui.pclose.addEventListener("click", () => closePicker());
  ui.pnone.addEventListener("click", () => { claim(null); closePicker(); });

  function showSpots() {
    const spots = spotList(), multi = spots.length > 0, grouped = multi && groupList().length > 0;
    const own = ownSpot();
    // a new site (the laptop switched: its name changed, and only that; the first name seen
    // just sets it): the hub let every camera go, so an old claim gets no answer
    const site = typeof state.site === "string" && state.site ? state.site : null;
    const newSite = site !== null && siteName !== null && site !== siteName;
    const firstSight = multi && !hadSpots;       // cameras where there were none
    const lost = shownSpot !== null && own === null && asked !== null;   // taken away, not let go
    if (site !== null) siteName = site;
    hadSpots = multi;
    if (newSite) {
      asked = undefined;
      if (askedNote) { askedNote = false; note(""); }
      pick.area = null;
    }
    ui.spotbox.hidden = grouped;                 // a small site (or none): a plain list
    ui.cam.hidden = !grouped;
    if (!grouped) {
      const sig = JSON.stringify([spots.map(s => [s.name, s.player]), player]);
      if (sig !== spotSig) {                     // rebuilt only when a holder changes
        spotSig = sig;
        const none = document.createElement("option");
        none.value = "";
        none.textContent = "–";
        ui.spot.replaceChildren(none, ...spots.map(s => {
          const o = document.createElement("option"), by = takenBy(s);
          o.value = s.name;
          o.textContent = by ? `${s.name} · player ${by}` : s.name;
          o.disabled = !!by;                     // one phone per camera
          return o;
        }));
      }
      ui.spot.value = own || "";
      ui.spot.disabled = !multi;                 // no cameras: one clip, nothing to pick
    } else spotSig = "";
    if (grouped) {
      const free = spots.filter(s => s.player == null).length;
      ui.camname.textContent = own || "pick a camera";
      ui.camarea.textContent = own ? areaOf(own) : `${free} free`;
      ui.camgo.textContent = own ? "change" : "";
      ui.cam.classList.toggle("none", !own);
    }
    if (!grouped) closePicker();
    else if ((newSite || firstSight || lost) && !own) openPicker();   // no camera (any more): pick
    else if (pick.open) renderPicker();
    if (asked !== undefined) {                   // the hub's answer to this phone's claim
      const s = spots.find(x => x.name === asked);
      if (own === asked) {
        asked = undefined;
        if (askedNote) { askedNote = false; note(""); }
      } else if (!s || takenBy(s)) {             // (a state still on its way is no answer)
        note(`${asked} is taken: pick another camera.`);
        askedNote = true;
        asked = undefined;
      }
    }
    const steer = !multi || own !== null;
    if (own !== shownSpot || !steer) letGoSpace();
    shownSpot = own;
    ui.zero.disabled = ui.gain.disabled = ui.gainval.disabled = !steer;
    ui.space.classList.toggle("off", !steer);
    showWho();
  }

  function areaOf(name) {
    const g = groupList().find(x => Array.isArray(x.spots) && x.spots.includes(name));
    const s = spotList().find(x => x.name === name);
    return g ? g.name : s && typeof s.group === "string" ? s.group : "";
  }

  function showWho() {
    ui.who.hidden = player === null;
    if (player !== null)
      ui.who.textContent = `player ${player}` + (spotList().length ? ` · ${ownSpot() || "pick a camera"}` : "");
  }

  function openPicker() {
    letGoSpace();                                // the sheet covers SPACE: never stuck under it
    Object.assign(pick, {open: true, area: null, sig: ""});
    ui.picker.hidden = false;
    renderPicker();
  }

  function closePicker() {
    pick.open = false;
    ui.picker.hidden = true;
  }

  function renderPicker() {                      // areas, or the cameras of one area
    const spots = spotList(), groups = groupList(), own = ownSpot();
    const byName = new Map(spots.map(s => [s.name, s]));
    const area = pick.area === null ? null : groups.find(g => g.name === pick.area) || null;
    if (!area) pick.area = null;
    const sig = JSON.stringify([pick.area, own, player, groups, spots.map(s => [s.name, s.player])]);
    if (sig === pick.sig) return;                // unchanged: never rebuilt under a finger
    pick.sig = sig;
    ui.pback.hidden = !area;
    ui.ptitle.textContent = area ? area.name : "Pick an area";
    ui.pnone.hidden = !own;
    if (!area) {
      ui.plist.replaceChildren(...groups.map(g => {
        const names = g.spots.filter(n => byName.has(n));
        const taken = names.filter(n => takenBy(byName.get(n))).length, mine = names.includes(own);
        const meta = (mine ? "yours · " : "") + `${names.length} camera${names.length === 1 ? "" : "s"}`
          + (taken ? ` · ${taken} taken` : "");
        return item(g.name, meta, {mine, full: taken === names.length},
                    () => { pick.area = g.name; renderPicker(); });
      }));
    } else {
      ui.plist.replaceChildren(...area.spots.filter(n => byName.has(n)).map(n => {
        const by = takenBy(byName.get(n)), mine = n === own;
        return item(n, mine ? "yours" : by ? `player ${by}` : "free", {mine, taken: !!by},
                    () => { if (!mine) claim(n); closePicker(); });
      }));
    }
  }

  function item(name, meta, {mine = false, taken = false, full = false}, onPick) {
    const b = document.createElement("button"), n = document.createElement("span"),
      m = document.createElement("span");
    b.type = "button";
    b.className = "pitem";
    n.className = "pname";
    n.textContent = name;
    m.className = "pmeta";
    m.textContent = meta;
    b.append(n, m);
    b.classList.toggle("mine", mine);            // this phone's camera (or its area)
    b.classList.toggle("full", full);            // an area with every camera taken
    b.disabled = taken;                          // one phone per camera: greyed
    b.addEventListener("click", onPick);
    return b;
  }

  const sendGain = throttle(g => control("gain", g), 50);
  const gainNow = () => {
    const g = gainOf(+ui.gain.value);
    return Math.abs(Math.log(g)) < 0.03 ? 1 : +g.toFixed(3);     // snaps to 1x
  };
  ui.gain.addEventListener("input", () => {
    dragging.gain = true;
    showGain(gainNow());
    sendGain(gainNow());
  });
  ui.gain.addEventListener("change", () => {
    dragging.gain = false;
    control("gain", gainNow());
  });
  ui.gainval.addEventListener("click", () => {
    ui.gain.value = sliderOf(1);
    showGain(1);
    control("gain", 1);
  });

  const sendSeek = throttle(t => control("seek", t), 66);
  ui.scrub.addEventListener("input", () => {
    dragging.scrub = true;
    ui.tcur.textContent = clock(+ui.scrub.value);
    sendSeek(+ui.scrub.value);
  });
  ui.scrub.addEventListener("change", () => {
    dragging.scrub = false;
    control("seek", +ui.scrub.value);
  });

  function applyState(st) {
    for (const k of Object.keys(state)) delete state[k];    // each state is whole (this phone's)
    Object.assign(state, st);
    if (typeof st.player === "number") player = st.player;
    if (typeof st.section === "boolean") sectionOn = st.section;
    if (typeof st.gain === "number" && !dragging.gain) {
      ui.gain.value = sliderOf(st.gain);
      showGain(st.gain);
    }
    showSpots();
    showState();
  }

  function showState() {
    ui.play.textContent = state.playing ? "Pause" : "Play";
    ui.play.classList.toggle("on", !!state.playing);
    ui.section.classList.toggle("on", sectionOn);
    showSpace();
    if (typeof state.duration === "number") {
      ui.scrub.max = state.duration;
      ui.scrub.disabled = !(state.duration > 0);
      ui.tdur.textContent = clock(state.duration);
    }
    if (typeof state.time === "number" && !dragging.scrub) {
      ui.scrub.value = state.time;
      ui.tcur.textContent = clock(state.time);
    }
  }

  function showGain(g) { ui.gainval.textContent = `${g.toFixed(2)}×`; }

  function clock(s) {
    s = Math.max(0, s || 0);
    return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  }

  function throttle(fn, ms) {                  // first call now, the last one after ms
    let last = 0, timer = 0, args;
    return (...a) => {
      args = a;
      const now = performance.now();
      if (now - last >= ms) {
        last = now;
        fn(...args);
      } else if (!timer) {
        timer = setTimeout(() => { timer = 0; last = performance.now(); fn(...args); }, ms - (now - last));
      }
    };
  }

  // ---- hold: landscape is the camera hold
  function checkHold() {
    const a = screenAngle();
    ui.turn.hidden = !running || a === 90 || a === 270;
  }
  window.addEventListener("orientationchange", checkHold);
  window.addEventListener("resize", checkHold);
  if (globalThis.screen && screen.orientation && screen.orientation.addEventListener)
    screen.orientation.addEventListener("change", checkHold);

  ui.gain.value = sliderOf(1);
  showGain(1);
  showState();
  if (K) connect();
  else say("No pairing code: open the link from the QR code in the motus Live view.");
}
