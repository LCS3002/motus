/* motus Live view (plans/live_controller/03, 08): a cinematic 3D view of a site's cameras. A
   virtual camera per camera in the scene, driven by a pose (a phone through the live hub, or the
   mouse), makes the machine redraw the tracks live while the clips play.

   One view that grows (08 b, 28 Oct): /live?run=<run folder> opens on that run's camera, the start
   camera, which the mouse controls and the first phone takes. Each further phone picks a camera of
   the site (/api/live/site: every Live package of the run's site, by place); it loads at its true
   place (its package's origin) with its panel. The Live panel's site picker switches site (the
   laptop only).
   Cameras vs Scene (29 Oct): a camera is in play while a phone holds it or while it is the mouse's
   (the marked camera no phone holds; when the last phone lets go, its camera). A camera out of play
   stays in the scene, locked where it was with its layer (L.parked: its video and panel go; 31 Oct:
   was frozen; now the padlock is the one way a camera stops), until Remove camera (a phone holding
   it is released, back to its picker). A phone that picks
   a camera in the scene takes it over. The panel: Cameras (the list, each camera's menu) and Scene
   (site, passes, sheet and exports, the look, help).
   Focus (29 Oct (4); O, or Shift+click a camera in the list or on its panel, or its menu; the same
   again: the whole scene): the view (plan or perspective) flies to frame that camera and its traces,
   the others stay. Whole scene (W): it flies back out to every camera of the collage. Solo (I, or
   the camera's menu; the same again: all): that camera alone in the main view, the rest hidden
   there, still running and recorded.

   Sandbox / collage (08 b, 29 Oct): each camera is its own layer on the site. A play (a pass)
   runs one shared clock from 0 to the end of the longest clip of the cameras in play (a shorter clip
   holds its last frame); at the end everything stops and the play is saved by itself as a pass
   (<start run>/live/passes/, groundtrack/live_passes.py). Play redraws only the cameras in play: a
   camera's layer is replaced when it starts drawing (Stack passes off) or added to (on, K); every
   other camera's layer stays. Undo takes back the last change (a play, a Load, a replay, a camera
   cleared): the cameras it changed get their layers back; Clear wipes all, X (or the camera's x)
   one camera. The passes' menu (L) loads a pass's layers back onto its cameras. Files always stay.

   Realtime (plans/live_controller/10, step 3; /live?realtime=1, the site picker's "Realtime ·
   now"): one phone records, this laptop tracks it, and this view draws it as it is tracked. The
   mode is turned on here; each recording opens as its own site once the laptop has calibrated it
   (Realtime, the camera called by its time). Its points come as they are tracked (whole frames
   only), its newest blurred picture fills the video plane, and the clock is now (no seek, no end):
   the other phones turn its camera as on any clip. The recorder's Stop: its last points, the clock
   runs to the end and the pass is saved (n = the package's points), its video takes the plane's
   place; the next recording opens once it is calibrated. People that come out the wrong height
   (the people check) turn the drag grid on: the ground as the camera puts it.

   The space clutch (08 a): while held (the phone's SPACE button, or M held on the laptop for the
   mouse) turning tilts the drawn crowd about the start camera's centre, with the hand; let go,
   turning steers the camera again and its drawing stays tilted; a double-tap glides it back to
   level. The machine's rules (08 c) judge what it draws (tracks.js).

   Each camera has its own camera, space, clutch, crowd and a tint of the violet; a phone acts on
   the camera it holds (pose, zero, gain, scale, SPACE); play / pause / scrub are shared. With more
   than one camera a column of panels (video boxes) shows the players' views, only on request (29 Oct
   (5): a camera's menu, Video; B all; none open by default; pop one out to its own window). The
   mouse, the keys and the HUD act on the focus camera (Tab: the next); the whole clip is the focus
   camera's.

   Keys: Space play, Z zero (the camera only), G / Shift+G gain, S section, H HUD, P panel,
   F full screen, V video (re-framed / glued / off), M hold: the space clutch (double-tap: level),
   C cause mode, U below the ground (veil / same), K stack passes, L the saved passes (a menu),
   O focus (this camera framed) / the whole scene, W the whole scene, I solo (this camera alone) / all,
   X clear this camera's layer, B the video boxes of the cameras in play, Tab the next camera, 1-4 views,
   arrows step a frame (Shift: 10).
   URL options (for checks and screenshots): site=<name> (its default start camera, instead of
   run=), players=6363,6367 (phones 1, 2.. on these cameras, without the hub), focus=<camera>,
   save=0 (no pass is saved), stack=1 (passes build up), ground=A|B|C, look=laptop|projector, video=reframed|glued|0,
   trails=performed|whole, view=1..4, pan= tilt= roll= (degrees, the mouse head),
   space=pan,tilt,roll (degrees: the crowd's space turned so, with the hand), under=veil|same|dim,
   orbit=x,y,z,tx,ty,tz, f= (frame) or t= (seconds), panel=0, hud=0, ref=1, cause=1, drag=1,
   section=along|across|horizontal, slab=1, play=1, hub=0 (no phone hub), bench=1 (measure fps;
   benchspace=1: with the clutch held), realtime=1 (instead of run= / site=). */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { prepareScene, spaceMatrix } from "./projection.js";
import { createProjectionPool, defaultWorkers } from "./projection_worker.js";
import { makeGround, makeCameras, makeDragGrid } from "./scene.js";
import { parseTracks, appendTracks, TrackView, Past } from "./tracks.js";
import { Timeline, fmtTime } from "./timeline.js";
import { Section } from "./section.js";
import { PoseLink, MouseController, PoseInput, cameraDelta, matToQuat, headMatrix, GAIN_MIN, GAIN_MAX, GAINS } from "./pose.js";
import { TINTS, hexRGB, freeTint, offsetFrom, findCamera } from "./spots.js";
import { packPass, unpackPass, padPass, fitsPackage, localIso, passLabel, mergeSnaps } from "./passes.js";
import { Panels } from "./panels.js";
import { startTips } from "./tips.js";
import { Recorder, machineCounts, CHANGED, HELD, LEVELLING, TILTED, PERFORMED, JOINED, SMOOTH, PLAYING, AT_END } from "./record.js";
import { Replay, readRecording, compareDrawings, rowCamera } from "./replay.js";

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
THREE.ColorManagement.enabled = false;          // colours are sRGB values, shown as written

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const params = new URLSearchParams(location.search);
const SCALES = [1, 2, 5, 10, 20, 50];           // position scale: 1 m of the phone = this many m
const LOOKS = {
  laptop: { line: 1, frame: 1, ghost: 0.55, virt: 0.95, drag: 0.38, ref: 0.5 },
  projector: { line: 1.6, frame: 1.5, ghost: 0.8, virt: 1, drag: 0.55, ref: 0.7 },
};
const TRANSITION_S = 0.8;
const DOUBLE_TAP_MS = 320, TAP_MS = 250;        // M: as the phone's SPACE button (live_phone/phone.js)
/* Performed trails stay in the world where they were drawn (a later turn of the space leaves the
   history: a performance leaves a 3D record). true: they ride with the space instead. */
const TRAILS_RIDE_WITH_SPACE = false;
const IDENTITY = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
const UNDER = ["same", "dim", "veil"];
/* Whole clip with several cameras: only the focus camera's (the others keep their performed
   trails). Measured (1920 x 1080, 4 Trafalgar cameras, every pose changing every frame, 4 panels):
   performed 11.6 / 13.7 ms of work per frame (p50 / p95); the whole clip for all four does not fit
   60 fps (frame p95 25 ms), for the focus camera only 12.0 / 14.1 ms. */
const WHOLE_ALL_SPOTS = false;
const SAVE = params.get("save") !== "0";       // passes saved by themselves (save=0: checks)
/* The sheet (09): this page renders one pass's view for its performance sheet (sheet=<pass id>,
   sw / sh: its size, build=0: only the view, layout=A|B|C) and posts it to the app; nothing else. */
const SHEET = params.has("sheet") ? { id: params.get("sheet"), names: new Set(), busy: false } : null;
/* Realtime (10, step 3): the recorder's recording drawn as it is tracked. RT: the laptop's status
   (st), the recording open (id), its points so far (n), done (the recorder stopped and its last
   points are in), the people check shown for (checked). */
const REALTIME = params.get("realtime") === "1" && !SHEET;
const RT = { st: null, id: null, n: 0, done: false, gone: false, opening: false, tBusy: false, pBusy: false, checked: null, refAt: 0 };
const RT_STATUS_MS = 250, RT_TRACKS_MS = 120, RT_PREVIEW_MS = 120, RT_REF_MS = 2000;

// ---------------------------------------------------------------------------- the stage
const canvas = $("#gl");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
renderer.localClippingEnabled = true;
renderer.setClearColor(0x000000, 1);
const scene3 = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(35, 1, 0.05, 4000);
const orbit = new OrbitControls(camera, canvas);
orbit.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.PAN, RIGHT: null };
orbit.enableDamping = true;
orbit.dampingFactor = 0.12;
orbit.zoomSpeed = 0.8;
orbit.maxPolarAngle = Math.PI;                  // see-through site: the orbit goes below the ground too
const video = $("#video");                      // the start camera's video

const S = {                                     // the view's settings
  trails: params.get("trails") === "whole" ? "whole" : "performed",
  ground: ["A", "B", "C"].includes(params.get("ground")) ? params.get("ground") : REALTIME ? "C" : "B",   // B: dark contours (the user's pick); a recording has no aerial: C
  look: params.get("look") === "projector" ? "projector" : "laptop",
  layers: { ref: params.get("ref") === "1" || params.get("truth") === "1", drag: params.get("drag") === "1",
            video: !["0", "off"].includes(params.get("video")), cameras: true, cause: params.get("cause") === "1" },
  video: params.get("video") === "glued" ? "glued" : "reframed",   // the video plane when shown (V)
  under: UNDER.includes(params.get("under")) ? params.get("under") : "veil",   // the crowd below the ground (U; the user's pick: veil)
  view: 2,
  stack: params.get("stack") === "1",           // passes stay in the view and build up (K; off by default)
};
/* The open site: {site, title, cams (/api/live/site cameras), groups, startName, startRun, origin0
   (the start package's origin: the scene's), notes, spots: [spot] (the cameras in the scene, the
   start camera first), focus, locked, players: [{player, spot}] (the hub's, with a camera),
   section, loading: Map(name -> promise), tints: Map(name -> tint index), loadTimes: {name: ms},
   extra: Map(run -> {meta, data}) (packages of saved passes' cameras not in the scene), keep (the
   camera that stays while no phone holds one: the last in use), solo (the camera shown alone in the
   main view, or null), framed (the camera Focus frames in the plan / perspective, or null: the whole
   scene; it follows solo), boxes: Set(name) (the cameras whose video box was asked for; a camera
   leaving play drops out), lastTint: Map(name -> tint index) (a camera back keeps its tint when free)}.
   A spot: one camera in the scene (its package, scene, tracks, cameras, ground, pose input, group
   at its offset, tint, video, whole-clip pool). */
let L = null;
/* A saved pass being replayed (startReplay): {q, m (its manifest), play (replay.js Replay), spots
   (Set: its cameras), byName, mode ("watch" | "check" | "render"), t0, speed, done}. */
let R = null;
let link = null, mouse = null, hubInfo = null, sites = [], recordings = [];
const lastStart = {};                           // site -> the start camera's run last used (this session)
const input = { get current() { return L && L.focus ? L.focus.inp : null; } };   // the focus camera's pose input
const panels = new Panels({ renderer, scene: scene3, host: $("#players"), view: panelView,
                            onFocus: (s, e) => { if (e && e.shiftKey) toggleFocus(s); else setFocus(s); } });
const timeline = new Timeline($("#timeline"), (what) => onClock(what), () => newPass());
const past = new Past();                        // the drawings of the passes before the current one
scene3.add(past.group);

function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  if (L && L.locked) camera.fov = lockedFov();
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);

// ---------------------------------------------------------------------------- loading
const enc = encodeURIComponent;
const pkg = (run, name) => `/live/pkg/${name}?run=${enc(run)}`;

async function fetchOk(url, kind) {
  const r = await fetch(url);
  if (!r.ok) {
    let msg = `${r.status}`;
    try { msg = (await r.json()).error || msg; } catch {}
    throw new Error(`${url.split("?")[0].split("/").pop()}: ${msg}`);
  }
  return kind === "json" ? r.json() : r.arrayBuffer();
}

async function postJson(url, body) {
  const r = await fetch(url, { method: "POST", headers: { "X-Groundtrack": "1", "Content-Type": "application/json" },
                               body: JSON.stringify(body) });
  let d = {};
  try { d = await r.json(); } catch {}
  if (!r.ok || d.error) throw new Error(d.error || `${r.status}`);
  return d;
}

/* A request with files, no base64: [u32 header length][the JSON header][the files one after
   another], the header naming each file and its size (desktop.py: /api/live/pass/save). */
async function postFiles(url, body, files) {
  const enc8 = new TextEncoder(), parts = [], list = [];
  for (const [name, data] of Object.entries(files)) {
    const b = typeof data === "string" ? enc8.encode(data) : new Uint8Array(data);
    parts.push(b); list.push({ file: name, bytes: b.byteLength });
  }
  const head = enc8.encode(JSON.stringify({ ...body, files: list }));
  const r = await fetch(url, { method: "POST", headers: { "X-Groundtrack": "1", "Content-Type": "application/octet-stream" },
                               body: new Blob([new Uint32Array([head.byteLength]), head, ...parts]) });
  let d = {};
  try { d = await r.json(); } catch {}
  if (!r.ok || d.error) throw Object.assign(new Error(d.error || `${r.status}`), { confirm: r.status === 409 && !!d.confirm });
  return d;
}

function loading(text, bad = false) {
  const el = $("#loading");
  el.classList.toggle("done", text === null);
  el.classList.toggle("bad", bad);
  if (text !== null) $("#loading-what").textContent = text;
}

/** A camera's package, fetched (nothing built yet). live: a recording's id while it records (its
    points so far, not its tracks.bin). */
async function fetchSpot(cam, live = null) {
  const run = cam.run;
  const [meta, terrainBuf, tracks, rules] = await Promise.all([
    fetchOk(pkg(run, "scene.json"), "json"), fetchOk(pkg(run, "terrain.f32")),
    live ? rtFetchTracks(live, 0) : fetchOk(pkg(run, "tracks.bin")),
    fetchOk(`/api/live/rules?run=${enc(run)}`, "json")]);
  if (!meta.video || !meta.video.blurred)
    throw new Error(`${cam.name}: this package's video is not blurred: it is not shown`);
  if (live) meta.frames = tracks.head.frames;
  const aerial = await new THREE.TextureLoader().loadAsync(pkg(run, "aerial.jpg"));
  return { meta, terrainBuf, trackBuf: live ? tracks.bytes : tracks, rules, aerial, live: live ? tracks.head : null };
}

/* A recording's points from point n0 on, while it records (the laptop's memory) or once saved (its
   package): {head: {id, n0, n, frames (whole so far), done}, bytes (tracks.bin records)}. */
async function rtFetchTracks(id, n0) {
  const buf = await fetchOk(`/api/live/realtime/tracks?id=${enc(id)}&from=${n0}`);
  const hl = new DataView(buf).getUint32(0, true);
  return { head: JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl))), bytes: buf.slice(4 + hl) };
}

function spotVideo(start) {                     // the start camera: the page's <video>; others: one each
  if (start) return video;
  const v = document.createElement("video");
  v.muted = true; v.playsInline = true; v.preload = "auto";
  v.className = "spot-video";
  document.body.appendChild(v);
  return v;
}

/** A camera's package built into a group at its offset (not in the scene yet). */
function buildSpot(raw, cam, start, tintI) {
  const { meta, terrainBuf, trackBuf, rules, aerial } = raw;
  const heights = new Float32Array(terrainBuf);
  const sc = prepareScene(meta, heights);
  const data = parseTracks(trackBuf, meta);
  const tint = TINTS[tintI], rgb = hexRGB(tint), vid = spotVideo(start);
  const ground = makeGround(sc, heights, aerial);
  // a recording while it records: its newest blurred picture (rtPreview) instead of a video
  const canvas = raw.live ? Object.assign(document.createElement("canvas"), { width: 640, height: 360 }) : null;
  if (canvas) canvas.getContext("2d").fillRect(0, 0, canvas.width, canvas.height);
  const cams = makeCameras(sc, canvas || vid, { tint: rgb });
  const tracks = new TrackView(sc, data, rules, (meta.tracks || {}).classes || {}, { tint: rgb });
  tracks.computeReference();
  tracks.setMode("performed");
  const drag = makeDragGrid(sc);
  // only the drawn crowd turns with the space (08 a); the site, the reference and the cameras stay
  tracks.crowd.matrixAutoUpdate = false;
  const offset = start ? [0, 0, 0] : offsetFrom(meta.origin, L.origin0);
  const group = new THREE.Group();
  group.position.set(...offset);
  group.add(ground.mesh, cams.group, drag.group, tracks.group);
  const inp = new PoseInput(sc);
  return { name: cam.name, cam, start, run: cam.run, meta, sc, data, rules, ground, cams, tracks, drag, inp,
           group, offset, tint, tintI, rgb, video: vid, growing: !!raw.live, canvas,
           pool: null, player: null, frameShown: -1, poseVersion: -1, joinFrame: 0,
           spaceKey: new Float64Array(9).fill(NaN), camKey: new Float64Array(12).fill(NaN), rebuild: false,
           wholeLocal: null, wholeOut: null, whole: { busy: false, again: false, lastMs: 0, n: 0, sumMs: 0, calls: 0, startMs: 0 } };
}

const spotMaterials = (s) => [s.ground.material, ...s.cams.materials(), ...s.tracks.materials(), ...s.drag.materials()];

function disposeSpot(s) {
  scene3.remove(s.group);
  s.ground.dispose(); s.cams.dispose(); s.tracks.dispose(); s.drag.dispose();
  if (s.pool) { s.pool.close(); s.pool = null; }
  stopVideo(s);
  if (s.video !== video) s.video.remove();
}

/** A camera's video let go (its decoder freed; the element stays for when it is in play again). */
function stopVideo(s) {
  s.video.pause(); s.video.removeAttribute("src"); s.video.load();
}

/** A camera's point (its own frame) in the scene: + its offset. */
const toScene = (s, p) => [p[0] + s.offset[0], p[1] + s.offset[1], p[2] + s.offset[2]];

function videoReady(v) {
  return new Promise((res) => {
    if (v.readyState >= 1) res();
    else { v.addEventListener("loadedmetadata", res, { once: true }); v.addEventListener("error", res, { once: true }); }
  });
}

const round = (v, k = 0) => Math.round(v * 10 ** k) / 10 ** k;

/* A camera into the scene (start: the site's start camera). Fetched, built, its video ready, then
   added, unless it is no longer wanted by then (its phone left). Its load time is kept (L.loadTimes). */
function addSpot(cam, start = false) {
  if (!L || spotOf(cam.name)) return Promise.resolve(spotOf(cam.name));
  if (L.loading.has(cam.name)) return L.loading.get(cam.name);
  const site = L, t0 = performance.now(), back = parkedOf(cam.name);
  let tintI = back ? back.tintI : 0;
  if (!back) {
    const used = [...L.tints].filter(([n]) => n !== cam.name).map(([, v]) => v), prev = L.lastTint.get(cam.name);
    tintI = start ? 0 : prev != null && !used.includes(prev) ? prev : freeTint(used);
  }
  L.tints.set(cam.name, tintI);
  L.lastTint.set(cam.name, tintI);
  const gone = () => L !== site || (!start && !wanted().has(cam.name)) || (back && !L.parked.includes(back));
  const job = (async () => {
    let s = back, t1 = t0, t2 = t0;
    if (!s) {
      const raw = await fetchSpot(cam, start && L.live ? L.live : null);
      t1 = performance.now();
      if (gone()) { raw.aerial.dispose(); return null; }
      if (start) L.origin0 = raw.meta.origin || null;
      s = buildSpot(raw, cam, start, tintI);
      t2 = performance.now();
    }
    if (!s.growing) {                              // (a recording while it records has no video yet)
      s.video.src = pkg(cam.run, "video.mp4");     // the blurred clip from the package only
      await videoReady(s.video);
    }
    const t3 = performance.now();
    const fail = !s.growing && s.video.error;
    if (gone() || fail) {
      if (!back) disposeSpot(s);
      else if (L.parked.includes(back)) stopVideo(back);
      if (fail) throw new Error(`${cam.name}: the video could not be played`);
      return null;
    }
    if (back) {                                    // a frozen camera in play again, where it stood
      L.parked = L.parked.filter((x) => x !== back);
      s.parked = false;
      s.tracks.resetPerformed();
      s.tracks.group.visible = true;
      s.poseVersion = -1; s.frameShown = -1; s.rebuild = true;
    } else {
      scene3.add(s.group);
      if (start) makeSection(s);
      L.section.register(spotMaterials(s));
      s.inp.setGain(1);
      s.inp.setScale(1);
    }
    timeline.add(s, { video: s.video, fps: s.meta.fps, frames: s.meta.frames, live: s.growing });
    s.joinFrame = timeline.time > 0 ? timeline.frameOf(s) : 0;          // it joins the pass now
    s.tracks.perfFrame = s.joinFrame - 1;
    L.spots.push(s);
    if (!start && P.cur) recJoin(s);              // it takes part in the pass from here
    lookSpot(s);
    layersSpot(s);
    spotTrails(s);
    const ms = { fetch: round(t1 - t0), build: round(t2 - t1), video: round(t3 - t2), total: round(performance.now() - t0) };
    L.loadTimes[cam.name] = ms;
    if (L.markNext === cam.name) { L.markNext = null; setFocus(s, false); }
    if (!start) flash(back ? `${cam.name}: in play` : `${cam.name}: in the scene (${ms.total} ms)`);
    return s;
  })();
  L.loading.set(cam.name, job);
  job.catch((e) => note(`${cam.name}: ${e.message}`)).finally(() => {
    if (L !== site) return;
    L.loading.delete(cam.name);
    if (L.markNext === cam.name && !spotOf(cam.name)) L.markNext = null;
    if (!spotOf(cam.name) && !parkedOf(cam.name)) L.tints.delete(cam.name);
    syncScene();                                    // the camera it replaces leaves now (never before)
    sceneChanged();
  });
  return job;
}

/* A camera leaves play (the start camera too; never the last one): what it drew in the current
   play stays in it (its layer); the camera stays in the scene, locked where it was (parked: its
   video and panel go), until it is removed (Remove camera: L.removing, then it goes). The clock
   follows the cameras in play. */
function removeSpot(s) {
  if (!L || !L.spots.includes(s) || L.spots.length < 2) return;
  const snap = drew(s) ? s.tracks.snapshot() : null;
  if (snap) {
    P.cur.left.set(s.name, merged(P.cur.left.get(s.name), passEntry(s, copySnap(snap))));
    addLayer(P.cur.key, s.name, s.data, snap, s.offset, s.tint, camCopy(s.inp.cam));
  }
  rec.event("camera_leave", { camera: s.name, player: s.player });
  L.spots = L.spots.filter((x) => x !== s);
  if (mKey.spot === s) { mKey.down = false; mKey.spot = null; }
  if (L.solo === s) L.solo = L.framed = null;      // alone no more: the whole collage
  if (L.focus === s) setFocus(L.spots[0], false);
  L.boxes.delete(s.name);                          // its video box closed (back in play: closed)
  refreshPlayers();                                // its panel goes first
  timeline.remove(s);
  const removing = L.removing.get(s.name);
  if (removing) { takeLayers(removing, s.name); L.removing.delete(s.name); dropSpot(s); }
  else park(s);
  syncGhosts();
  renderer.renderLists.dispose();
  sceneChanged();
}

/** A camera out of play: its camera still where it was, its video and panel gone. Locked (31 Oct,
    the user: one way a camera stops): one that a Load, an Undo or a replay's end left out of play
    comes in locked (L.autoLocked, until its padlock is used). */
function park(s) {
  if (!isLocked(s.name)) { L.lockedCams.add(s.name); L.autoLocked.add(s.name); }
  s.parked = true;
  s.player = null;
  stopVideo(s);
  if (s.pool) { s.pool.close(); s.pool = null; }
  parkLook(s);
  L.parked.push(s);
}

/** A parked camera shows its camera (and its ground) only: no video, no live drawing, no grid. */
function parkLook(s, k = 1) {
  lookSpot(s, k);
  layersSpot(s);
  s.cams.plane.visible = false;
  s.drag.group.visible = false;
  s.tracks.group.visible = false;
}

/** A camera gone from the view (in play or parked). */
function dropSpot(s) {
  if (!spotOf(s.name) && !L.loading.has(s.name)) L.tints.delete(s.name);
  L.parked = L.parked.filter((x) => x !== s);
  if (L.solo === s) L.solo = null;
  if (L.framed === s) L.framed = null;
  L.section.unregister(spotMaterials(s));
  disposeSpot(s);
}

/** The parked camera called so, or null. */
const parkedOf = (name) => (L && L.parked.find((s) => s.name === name)) || null;

/** Whether camera `name` has a layer in the view (a finished drawing of it). */
const hasLayer = (name) => past.items.some((it) => it.cam === name);

/** A camera's pose kept with a layer (its frozen camera): {R_wc, R_cw, C}. */
const camCopy = (c) => (c ? { R_wc: Float64Array.from(c.R_wc), R_cw: Float64Array.from(c.R_cw), C: Float64Array.from(c.C) } : null);

/* The frozen cameras follow their layers: a parked camera with layers sits at its newest layer's
   pose; a camera with a layer but neither in play nor parked (a Load, an Undo) is fetched and
   parked. (A camera stays until it is removed, with or without a layer.) */
function syncGhosts() {
  if (!L) return;
  const newest = new Map();
  for (const it of past.items) { const o = newest.get(it.cam); if (!o || it.key >= o.key) newest.set(it.cam, it); }
  for (const g of L.parked) {
    const it = newest.get(g.name);
    if (it && it.pose && g.posedBy !== it) { g.cams.update(it.pose); g.posedBy = it; }
  }
  if (R || SHEET) return;
  for (const [name, it] of newest) {
    if (spotOf(name) || parkedOf(name) || L.loading.has(name) || L.ghosts.has(name)) continue;
    const cam = L.cams.find((c) => c.name === name);
    if (cam) ghostSpot(cam, it.tintI);
  }
}

/** A camera that is not in play, fetched for its layer (or an Undo of its Remove): parked straight away. */
function ghostSpot(cam, tintI, pose = null, force = false) {
  const site = L;
  L.ghosts.add(cam.name);
  fetchSpot(cam).then((raw) => {
    if (L !== site) { raw.aerial.dispose(); return; }
    L.ghosts.delete(cam.name);
    if (spotOf(cam.name) || parkedOf(cam.name) || L.loading.has(cam.name) || !(force || hasLayer(cam.name))) { raw.aerial.dispose(); return; }
    const used = [...L.tints.values()];
    const ti = tintI != null && tintI >= 0 && !used.includes(tintI) ? tintI : L.lastTint.get(cam.name) ?? freeTint(used);
    L.tints.set(cam.name, ti);
    const s = buildSpot(raw, cam, false, ti);
    scene3.add(s.group);
    L.section.register(spotMaterials(s));
    s.cams.update(pose || s.inp.camera(clock()));  // at rest until its layer's pose
    park(s);
    syncGhosts();
    sceneChanged();
  }).catch((e) => { if (L === site) { L.ghosts.delete(cam.name); note(`${cam.name}: ${e.message}`); } });
}

function makeSection(s) {
  L.section = new Section({ scene: scene3, camera, dom: canvas, orbit, onChange: () => syncPanel(),
                            sc: { C: toScene(s, s.sc.C), yaw: s.sc.yaw, R_max: s.sc.R_max } });
  L.section.register(past.materials());
}

/* After a camera joined or left: the title, the list, the panels, the overview's framing, the
   phones' state. */
function sceneChanged() {
  if (!L) return;
  refreshPlayers();
  syncTitle();
  syncSpots();
  if (S.trails === "whole") for (const s of L.spots) spotTrails(s);
  if (S.view === 1 || S.view === 2) setView(S.view);
  publishState(true);
}

/* The site of `run` (its start camera): the old site's pass saved and ended, its cameras gone;
   the phones' cameras are released (the hub sees a new start camera). live: the recording's id
   while it records (realtime: its start camera grows). */
async function openSite(run, { live = null } = {}) {
  loading("the site");
  const info = await fetchOk(`/api/live/site?run=${enc(run)}`, "json");
  const start = info.cameras.find((c) => c.name === info.start_spot);
  if (L) endSite();
  L = { site: info.site, title: (sites.find((x) => x.site === info.site) || {}).title || info.site, cams: info.cameras,
        groups: info.groups || [], startName: info.start_spot, startRun: start.run, origin0: null, notes: info.notes || [],
        spots: [], focus: null, locked: null, players: [], section: null, loading: new Map(), tints: new Map(),
        loadTimes: {}, extra: new Map(), keep: info.start_spot, solo: null, framed: null, boxes: new Set(), lastTint: new Map(), parked: [], ghosts: new Set(),
        removing: new Map(), released: new Map(), markNext: null, live, hiddenCams: new Set(), lockedCams: new Set(),
        autoLocked: new Set(), inPlay: new Set([info.start_spot]) };
  P.plays = 0;
  lastStart[L.site] = start.run;
  loading("the machine at rest");
  const s = await addSpot(start, true);
  if (!s) throw new Error(`${start.name}: not loaded`);
  timeline.seekTime(0, false);
  passBegin();
  setFocus(s, false);
  if (mouse) mouse.send();                       // the mouse's pose (its first is its zero)
  applyLook();
  setLayers();
  syncPanel();
  syncSites();
  setView(S.view === 1 ? 1 : 2, false);
  if (info.notes && info.notes.length) note(info.notes.join("\n"));
  if (!REALTIME)                                  // (realtime: the page stays the mode's, a reload comes back to it)
    history.replaceState(null, "", `/live?run=${enc(start.run)}${location.search.replace(/^\?/, "&").replace(/&run=[^&]*/g, "").replace(/&site=[^&]*/g, "")}`);
  refreshPasses();
  publishState(true);
  return L;
}

/** The open site goes: its pass saved (when it has unsaved drawing), every camera disposed. */
function endSite(reason = "site") {
  timeline.pause();
  passEnd(reason);
  panels.clear();
  for (const s of L.spots) { timeline.remove(s); disposeSpot(s); }
  for (const s of L.parked) disposeSpot(s);
  if (L.section) { scene3.remove(L.section.pivot, L.section.helper); L.section.dispose(); }
  past.clear();
  P.steps = [];
  P.byKey.clear();
  renderer.renderLists.dispose();
  L = null;
}

/* The site picker: another site opens on its start camera (the one used last this session, else
   its best calibrated). Only the laptop switches sites. */
async function switchSite(name) {
  const e = sites.find((x) => x.site === name);
  if (!e || !L || e.site === L.site) return;
  if (R) stopReplay(false);
  rec.event("site", { value: e.site });
  try {
    await openSite(lastStart[e.site] || e.start_run);
    loading(null);
    flash(`Site: ${L.title}`);
  } catch (err) { loading(`${e.title}: ${err.message}`, true); }
}

function ensurePool(s) {
  if (!s.pool) s.pool = createProjectionPool(s.sc, Math.max(defaultWorkers(), 2));
  return s.pool;
}

/** The camera in the scene called so, or null. */
function spotOf(name) {
  if (!L || name == null) return null;
  return L.spots.find((s) => s.name === name) || null;
}

/* The focus camera: the mouse, the keys (M, Z, G), the panel's sliders, the HUD and the views 3 / 4
   act on it. */
function setFocus(s, notify = true) {
  if (!L || !s) return;
  L.focus = s;
  if (L.solo && L.solo !== s) { L.solo = L.framed = s; if (S.view === 1 || S.view === 2) setView(S.view); }   // the one shown alone follows
  timeline.setFocus(s);
  if (S.trails === "whole") for (const x of L.spots) spotTrails(x);
  document.documentElement.style.setProperty("--spot-tint", s.tint);
  $("#h-clip").textContent = `${s.name} · ${s.cam.clip}/${s.cam.run_name}`;
  showGain(s.inp.gain);
  showScale(s.inp.scale);
  sel.name = s.name;                             // the selected camera follows the marked one
  syncSpots();
  if (L.locked) setView(L.locked === "start" ? 3 : 4);
  if (notify) { flash(`Camera: ${s.name}`); syncHud(); publishState(); }
}

function nextFocus(d = 1) {
  const all = L ? [...L.spots, ...L.parked].filter((s) => s === L.focus || !isLocked(s.name)) : [];
  if (all.length < 2) return;
  const i = all.indexOf(L.focus);
  markCamera(all[(i + d + all.length) % all.length]);
}

/** Whether a phone holds camera `name` (one released by Remove no longer counts). */
const heldBy = (name) => (L ? L.players.find((p) => p.spot === name && L.released.get(name) !== p.player) || null : null);

/* A click on a camera (or Tab): the marked camera, which the mouse, the keys and the controls act
   on. One no phone holds becomes the mouse's camera: in play from here; the mouse's camera before
   stays in play where it stands (31 Oct, the user: they all run at once; only the padlock stops one,
   and a locked one is selected only). */
function markCamera(s) {
  if (!L || !s) return;
  if (isLocked(s.name)) { selectCamera(s.name); flash(`${s.name} is locked · its padlock unlocks it`); return; }
  if (L.spots.includes(s)) {
    setFocus(s);
    if (!heldBy(s.name) && L.keep !== s.name) { L.keep = s.name; syncScene(); }
    return;
  }
  if (!L.parked.includes(s)) return;
  L.inPlay.add(s.name);
  L.keep = s.name;
  L.markNext = s.name;
  flash(`${s.name}: the mouse's camera`);
  syncScene();
}

/* Remove camera (its menu): it leaves the scene with its layer (Undo brings them back; the files
   stay). A phone holding it is released (the hub: back to its camera picker, still connected). The
   last camera stays (the view is never empty). */
function removeCamera(name) {
  if (!L || R) return;
  const s = spotOf(name) || parkedOf(name);
  if (!s || L.removing.has(name)) return;
  if (isLocked(name)) { flash(`${name} is locked · unlock it to remove it`); return; }
  const others = [...L.spots, ...L.parked].filter((x) => x !== s && !L.removing.has(x.name));
  if (!others.length) { flash("The last camera stays (the view is never empty)"); return; }
  const holder = heldBy(name);
  rec.event("camera_remove", { camera: name, player: holder ? holder.player : null });
  if (holder) { L.released.set(name, holder.player); if (link) link.sendMessage({ type: "release", spot: name }); }
  const st = newStep("remove", { camera: name, tintI: s.tintI, pose: camCopy(s.parked && s.posedBy ? s.posedBy.pose : s.inp.cam) });
  takeLayers(st, name);
  L.hiddenCams.delete(name);
  L.inPlay.delete(name);
  if (L.solo === s) L.solo = null;
  if (L.framed === s) L.framed = null;
  if (s.parked) dropSpot(s);
  else {
    L.removing.set(name, st);
    if (L.keep === name) L.keep = null;
    if (L.focus === s || !L.spots.some((x) => x !== s)) {           // the marked camera goes: the next one is marked
      const next = L.spots.find((x) => x !== s) || others[0];
      if (L.spots.includes(next)) { setFocus(next, false); if (!heldBy(next.name)) L.keep = next.name; }
      else markCamera(next);
    }
    syncScene();
  }
  syncGhosts();
  flash(`${name}: removed${holder ? ` · P${holder.player} picks again` : ""} · Undo brings it back`);
  sceneChanged();
}

/* The layers' padlock (31 Oct, the user: "lock them there like a Photoshop layer"): a locked camera
   stays where it is. Nobody moves it: in play it leaves play (what it drew kept as its layer), its
   phone picks again and the phones see it locked; a click selects it only. Its layer stays: Clear
   layer, Clear all and Remove leave it, and Undo and Load too once it was locked here (a camera a
   Load, an Undo or a replay brought in locked, L.autoLocked, is theirs to change). Unlocked: in play,
   where it stands. One camera stays unlocked (the view always has one in play). Recorded. */
function isLocked(name) { return !!L && L.lockedCams.has(name); }

function setLock(name, on) {
  if (!L || !name) return;
  const s = spotOf(name) || parkedOf(name);
  if (!s || !!on === isLocked(name)) return;
  if (R) { flash("Lock after the replay"); return; }
  if (on) {
    if (s.growing) { flash(`${name} is recording: it cannot be locked`); return; }
    const free = [...L.spots, ...L.parked].filter((x) => x !== s && !isLocked(x.name) && !L.removing.has(x.name));
    if (!free.length) { flash("One camera stays unlocked: the view always has one in play"); return; }
    L.lockedCams.add(name);
    L.autoLocked.delete(name);
    const holder = heldBy(name);
    if (holder) { L.released.set(name, holder.player); if (link) link.sendMessage({ type: "release", spot: name }); }
    if (L.spots.includes(s)) {
      if (L.keep === name) L.keep = null;
      if (L.focus === s) {                       // the mouse moves on to another camera
        const next = L.spots.find((x) => x !== s && !isLocked(x.name)) || free[0];
        if (L.spots.includes(next)) { setFocus(next, false); if (!heldBy(next.name)) L.keep = next.name; }
        else markCamera(next);
      }
      syncScene();
    }
  } else {
    L.lockedCams.delete(name);
    L.autoLocked.delete(name);
    L.inPlay.add(name);
    syncScene();                                 // in play, where it stands
  }
  rec.event("lock", { camera: name, value: !!on });
  flash(on ? `${name}: locked · nobody moves it; Clear, Remove, Undo and Load leave its layer`
           : `${name}: unlocked · in play`);
  sceneChanged();
}

/** Whether camera `name` was locked here (its padlock), not brought in locked by a Load, an Undo or a replay. */
const lockedHere = (name) => isLocked(name) && !L.autoLocked.has(name);

/** The camera locked here whose layers step st would change (its own, or the ones it brings back), or null. */
function lockedBy(st) {
  const names = [...past.items.filter((it) => it.key === st.key).map((it) => it.cam), ...(st.replaced || []).map((d) => d.cam)];
  return names.find(lockedHere) || null;
}

/* The layers' eye (31 Oct): a camera and its layers hidden in the main view (it goes on drawing and
   is recorded; its phone's box stays). The collage's output leaves a hidden camera out; Output's
   Camera still gives it. */
function toggleHidden(name) {
  if (!L || !(spotOf(name) || parkedOf(name))) return;
  const off = !L.hiddenCams.has(name);
  if (off) L.hiddenCams.add(name); else L.hiddenCams.delete(name);
  flash(`${name}: ${off ? "hidden in the view · the eye shows it" : "shown"}`);
  syncSpots();
}

/* + Camera (31 Oct): one of the site's cameras into the scene from the laptop, as the mouse's camera
   (the others run on where they stand). One in the scene: marked. */
function addCamera(name) {
  if (!L || R || !name) return;
  const s = spotOf(name) || parkedOf(name);
  if (s) { markCamera(s); return; }
  if (!L.cams.some((c) => c.name === name) || L.loading.has(name)) return;
  L.inPlay.add(name);
  L.keep = name;
  L.markNext = name;
  sel.name = name;
  rec.event("camera_add", { camera: name });
  flash(`${name}: into the scene · the mouse's camera`);
  syncScene();
  syncSpots();
}

// ---------------------------------------------------------------------------- views
const tmpCam = new THREE.PerspectiveCamera();
const anim = { t0: 0, from: null, to: null };
let lastOverview = 2;                            // the plan (1) or perspective (2) last on: Focus flies into it

function startFwd(sc) { return [Math.sin(sc.yaw), Math.cos(sc.yaw)]; }

function lockedFov(sc = L.focus.sc, aspect = camera.aspect) {
  const K = sc.K;
  const vf = 2 * Math.atan(sc.H / 2 / K[4]), hf = 2 * Math.atan(sc.W / 2 / K[0]);
  return THREE.MathUtils.radToDeg(aspect >= sc.W / sc.H ? vf : 2 * Math.atan(Math.tan(hf / 2) / aspect));
}

/** The three.js orientation of an OpenCV camera (x right, y down, z forward). */
function cvQuat(R, q = new THREE.Quaternion()) {
  const m = new THREE.Matrix4().makeBasis(new THREE.Vector3(R[0], R[3], R[6]), new THREE.Vector3(-R[1], -R[4], -R[7]),
                                          new THREE.Vector3(-R[2], -R[5], -R[8]));
  return q.setFromRotationMatrix(m);
}

/* The overview: one camera (or the one Focus frames, or the one shown alone), about its view; more,
   every camera in the scene (the cameras' box out to each one's reach), north up. */
function viewPose(v) {
  let fs = L.focus;
  if (v === 3 || v === 4) {
    const cam = v === 3 ? fs.sc : fs.inp.cam;     // the cameras are drawn without the space's turn
    return { pos: new THREE.Vector3(...toScene(fs, cam.C)), quat: cvQuat(cam.R_wc), fov: lockedFov(), lock: v === 3 ? "start" : "virtual" };
  }
  let pos, target, fov = 35;
  const all = [...L.spots, ...L.parked];          // the collage's cameras: in play and locked
  const one = L.solo || (L.framed && all.includes(L.framed) ? L.framed : null);
  if (all.length <= 1 || one) {                    // one camera (or one framed / shown alone)
    fs = one || fs;
    const sc = fs.sc, R = sc.R_max, C = toScene(fs, sc.C), [fx, fy] = startFwd(sc), rx = fy, ry = -fx;
    if (v === 1) {
      target = new THREE.Vector3(C[0] + fx * R * 0.5, C[1] + fy * R * 0.5, 0);
      const h = 0.68 * R / Math.tan(THREE.MathUtils.degToRad(fov / 2));
      pos = target.clone().add(new THREE.Vector3(0, -0.001 * h, h));
    } else {
      target = new THREE.Vector3(C[0] + fx * R * 0.45, C[1] + fy * R * 0.45, 0);
      pos = new THREE.Vector3(C[0] - fx * R * 0.42 - rx * R * 0.42, C[1] - fy * R * 0.42 - ry * R * 0.42, R * 0.5);
    }
  } else {
    const xs = [], ys = [];
    for (const s of all) {
      const C = toScene(s, s.sc.C), half = Math.atan(s.sc.W / 2 / s.sc.K[0]), r = s.sc.R_max;
      xs.push(C[0]); ys.push(C[1]);
      for (const a of [-half, -half / 2, 0, half / 2, half]) { xs.push(C[0] + r * Math.sin(s.sc.yaw + a)); ys.push(C[1] + r * Math.cos(s.sc.yaw + a)); }
    }
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const t = Math.tan(THREE.MathUtils.degToRad(fov / 2)), W = window.innerWidth;
    const side = $("#panel").classList.contains("closed") ? 0 : 248, use = Math.max(0.4, (W - side) / W);
    const h = 1.2 * Math.max((y1 - y0) / 2 / t, (x1 - x0) / 2 / (t * camera.aspect * use));
    const shift = side / 2 / W * 2 * h * t * camera.aspect;          // centred in the stage left of the panel
    target = new THREE.Vector3((x0 + x1) / 2 + shift, (y0 + y1) / 2, 0);
    if (v === 1) pos = target.clone().add(new THREE.Vector3(0, -0.001 * h, h));
    else pos = target.clone().add(new THREE.Vector3(-0.42 * h, -0.62 * h, 0.62 * h));
  }
  tmpCam.position.copy(pos);
  tmpCam.up.set(0, 0, 1);
  tmpCam.lookAt(target);
  return { pos, quat: tmpCam.quaternion.clone(), fov, target };
}

function setView(v, animate = true) {
  if (!L || !L.focus) return;
  S.view = v;
  if (v === 1 || v === 2) lastOverview = v;
  const to = viewPose(v);
  $$("#views [data-view]").forEach((b) => b.setAttribute("aria-selected", String(+b.dataset.view === v)));
  orbit.enabled = false;
  orbit.locked = !!to.lock;
  L.locked = null;
  if (!animate) { finishView(to); return; }
  anim.from = { pos: camera.position.clone(), quat: camera.quaternion.clone(), fov: camera.fov };
  anim.to = to;
  anim.t0 = performance.now();
}

function finishView(to) {
  anim.to = null;
  camera.position.copy(to.pos);
  camera.quaternion.copy(to.quat);
  camera.fov = to.fov;
  camera.updateProjectionMatrix();
  if (to.lock) { L.locked = to.lock; orbit.locked = true; orbit.enabled = false; }
  else { orbit.target.copy(to.target); orbit.locked = false; orbit.enabled = !(L.section && L.section.dragging); orbit.update(); }
}

function stepView(now) {
  if (!anim.to) return;
  let k = Math.min(1, (now - anim.t0) / (TRANSITION_S * 1000));
  const e = k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;      // ease in-out
  const to = anim.to.lock === "virtual" ? viewPose(4) : anim.to;
  camera.position.lerpVectors(anim.from.pos, to.pos, e);
  camera.quaternion.slerpQuaternions(anim.from.quat, to.quat, e);
  camera.fov = anim.from.fov + (to.fov - anim.from.fov) * e;
  camera.updateProjectionMatrix();
  if (k >= 1) finishView(to);
}

/** Leave a locked camera view (3, 4) when the user orbits or zooms: orbit about a point ahead. */
function unlock() {
  if (!L || !L.locked) return;
  const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
  orbit.target.copy(camera.position).addScaledVector(fwd, L.focus.sc.R_max * 0.4);
  L.locked = null;
  orbit.locked = false;
  orbit.enabled = true;
  S.view = 0;
  $$("#views [data-view]").forEach((b) => b.setAttribute("aria-selected", "false"));
}
$("#stage").addEventListener("pointerdown", (e) => { if (e.button === 0 && !e.altKey && !e.shiftKey && L && L.locked && !anim.to) unlock(); }, true);

/* Shift+click a camera in the 3D view (31 Oct, the user): as a click on its layer, it is selected and
   the mouse moves it (a locked one: selected only); a double Shift+click (within PICK_MS): Focus, the
   view flies to it (again: the whole scene). The camera under the pointer (its frustum or its
   picture, the nearest in front) or whose lines pass within PICK_PX; a drag (Shift: the orbit's pan)
   picks nothing. */
const PICK_PX = 12, PICK_MS = 400;
const pickAt = { x: 0, y: 0, id: null, last: null, t: 0 };
$("#stage").addEventListener("pointerdown", (e) => {
  if (e.button === 0 && e.shiftKey) Object.assign(pickAt, { x: e.clientX, y: e.clientY, id: e.pointerId });
}, true);
window.addEventListener("pointerup", (e) => {
  if (e.pointerId !== pickAt.id) return;
  pickAt.id = null;
  if (!L || R || Math.hypot(e.clientX - pickAt.x, e.clientY - pickAt.y) > 5) return;
  const s = cameraAt(e.clientX, e.clientY);
  if (!s) return;
  const twice = pickAt.last === s && performance.now() - pickAt.t < PICK_MS;
  Object.assign(pickAt, { last: twice ? null : s, t: performance.now() });
  if (twice) { toggleFocus(s); selectCamera(s.name); } else markCamera(s);
});

/** The camera shown in the main view under the screen point (x, y): inside its drawn frustum (the
    rays from its centre to the image's corners, the image), the nearest in front; else the one whose
    lines pass nearest, within PICK_PX; null if none. */
function cameraAt(x, y) {
  const r = canvas.getBoundingClientRect(), v = new THREE.Vector3();
  const screen = (p, m) => {
    v.set(p[0], p[1], p[2]).applyMatrix4(m).project(camera);
    return v.z > -1 && v.z < 1 ? [r.left + (v.x + 1) / 2 * r.width, r.top + (1 - v.y) / 2 * r.height, v.z] : null;
  };
  let best = null, bd = PICK_PX, bz = Infinity;
  for (const s of [...L.spots, ...L.parked]) {
    const lines = s.cams.virt;
    if (!lines.corners || !lines.group.visible || L.hiddenCams.has(s.name) || (L.solo && L.solo !== s)) continue;
    const m = lines.group.matrixWorld, c = screen(lines.C, m), q = lines.corners.map((p) => screen(p, m));
    if (c && q.every(Boolean)) {
      const tris = [[q[0], q[1], q[2]], [q[0], q[2], q[3]], ...q.map((p, i) => [c, p, q[(i + 1) % 4]])];
      if (tris.some(([a, b, d]) => inTri(x, y, a, b, d))) {
        const z = Math.min(c[2], ...q.map((p) => p[2]));
        if (bd > 0 || z < bz) { bd = 0; bz = z; best = s; }
        continue;
      }
    }
    for (let i = 0; i < 4; i++)
      for (const [a, b] of [[c, q[i]], [q[i], q[(i + 1) % 4]]]) {
        if (!a || !b) continue;
        const d = segDist(x, y, a, b);
        if (d < bd) { bd = d; best = s; }
      }
  }
  return best;
}

/** Whether the point (x, y) is inside the screen triangle a-b-c (either winding). */
function inTri(x, y, a, b, c) {
  const s = (p, q) => (q[0] - p[0]) * (y - p[1]) - (q[1] - p[1]) * (x - p[0]);
  const d1 = s(a, b), d2 = s(b, c), d3 = s(c, a);
  return !((d1 < 0 || d2 < 0 || d3 < 0) && (d1 > 0 || d2 > 0 || d3 > 0));
}

/** The distance from the point (x, y) to the segment a-b (screen pixels). */
function segDist(x, y, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1], l2 = dx * dx + dy * dy;
  const t = l2 ? Math.max(0, Math.min(1, ((x - a[0]) * dx + (y - a[1]) * dy) / l2)) : 0;
  return Math.hypot(x - a[0] - t * dx, y - a[1] - t * dy);
}
$("#stage").addEventListener("wheel", () => { if (L && L.locked && !anim.to) unlock(); }, { capture: true, passive: true });
orbit.addEventListener("start", () => { if (anim.to) anim.to = null; if (S.view === 1 || S.view === 2) S.view = 0; });

// ---------------------------------------------------------------------------- the players' panels
const _pq = new THREE.Quaternion();
/* A panel: its camera's virtual camera (the re-framed video fills it), without that camera's own
   lines, the section's handles and the axis labels. */
function panelView(e, cam, aspect) {
  const s = e.spot, c = s.inp.cam;
  cam.position.set(...toScene(s, c.C));
  cam.quaternion.copy(cvQuat(c.R_wc, _pq));
  cam.aspect = aspect;
  cam.fov = lockedFov(s.sc, aspect);
  cam.updateProjectionMatrix();
  return [s.cams.virt.group, L.section.helper, L.section.pivot];
}

/** The cameras the scene wants: every phone's; while no phone holds one, the last in use (the
    mouse's: the view is never empty); a replay's and a sheet's. */
function wanted() {
  const w = new Set();
  for (const n of L.inPlay) if (!isLocked(n)) w.add(n);
  for (const p of L.players) if (L.cams.some((c) => c.name === p.spot) && L.released.get(p.spot) !== p.player && !isLocked(p.spot)) w.add(p.spot);
  if (L.keep && !isLocked(L.keep)) w.add(L.keep);
  if (R) for (const n of R.names) w.add(n);
  if (SHEET) for (const n of SHEET.names) w.add(n);
  for (const n of L.removing.keys()) w.delete(n);
  return w;
}

/* The phones' cameras ([{player, spot: name}], the hub's players): a camera a phone picked comes
   into play (one in the scene is taken over) and stays in play when no phone holds it any more (31
   Oct; was: frozen), until it is locked or removed. A phone that takes the mouse's camera takes it
   over; when the last phone lets go, its camera is the mouse's (the focus when it was one of theirs). */
function setPlayers(list) {
  if (!L) return;
  const before = new Set(L.players.map((p) => p.spot));
  L.players = (list || []).filter((p) => p && p.spot != null && L.cams.some((c) => c.name === p.spot));
  for (const [n, pl] of [...L.released]) if (!L.players.some((p) => p.spot === n && p.player === pl)) L.released.delete(n);
  for (const p of L.players) if (L.released.get(p.spot) !== p.player) L.inPlay.add(p.spot);
  if (L.keep && (heldBy(L.keep) || isLocked(L.keep))) L.keep = null;
  if (!L.players.some((p) => heldBy(p.spot) && !isLocked(p.spot)) && !L.keep)
    L.keep = [L.focus && before.has(L.focus.name) ? L.focus.name : null, ...[...before].filter((n) => spotOf(n)).reverse(),
              L.focus && L.focus.name, ...L.spots.map((s) => s.name)].find((n) => n && !isLocked(n)) || null;
  syncScene();
  refreshPlayers();
  syncSpots();
}

/* The scene to what it wants: the wanted cameras load; the others leave once one wanted camera is
   in (a camera taken over keeps the view until its successor has loaded; one that fails to load
   leaves the old one there). */
function syncScene() {
  if (!L) return;
  const w = wanted();
  for (const name of w) if (!spotOf(name) && !L.loading.has(name)) addSpot(L.cams.find((c) => c.name === name));
  if (!L.spots.some((s) => w.has(s.name))) return;
  for (const s of L.spots.slice()) if (!w.has(s.name)) removeSpot(s);
  if (L.solo && !L.spots.includes(L.solo) && !L.parked.includes(L.solo)) L.solo = null;
  if (L.framed && !L.spots.includes(L.framed) && !L.parked.includes(L.framed)) L.framed = null;
}

/* Who holds which camera in the scene; a panel (video box) per player with more than one camera in
   the scene (one camera: its own view is view 4), only for the cameras whose box was asked for. */
function refreshPlayers() {
  if (!L) return;
  for (const s of L.spots) s.player = null;
  for (const p of L.players) {
    const s = spotOf(p.spot);
    if (!s) continue;
    s.player = p.player;
    if (P.cur) { const h = P.cur.players.get(s.name) || new Set(); h.add(p.player); P.cur.players.set(s.name, h); }
  }
  const many = L.spots.length > 1;
  panels.set(many ? L.players.filter((p) => spotOf(p.spot) && L.boxes.has(p.spot)).map((p) => ({ player: p.player, spot: spotOf(p.spot) })) : []);
}

/** Whether camera s can have a video box: in play, held by a phone, more than one camera in play. */
const canBox = (s) => !!L && L.spots.includes(s) && L.spots.length > 1 && L.players.some((p) => p.spot === s.name);

/** A camera's video box (its menu's Video) open / closed. */
function toggleBox(s) {
  if (!L || !s) return;
  if (L.boxes.has(s.name)) L.boxes.delete(s.name);
  else if (canBox(s)) L.boxes.add(s.name);
  else { flash(`${s.name}: no video box (only a camera a phone holds, with others in play)`); return; }
  refreshPlayers();
  syncCamSel();
}

/** B: the video boxes of every camera in play a phone holds, shown (any closed) / hidden. */
function toggleBoxes() {
  if (!L) return;
  const can = L.spots.filter(canBox);
  if (!can.length) { flash("No video boxes: no phone holds a camera (with others in play)"); return; }
  const show = can.some((s) => !L.boxes.has(s.name));
  for (const s of can) { if (show) L.boxes.add(s.name); else L.boxes.delete(s.name); }
  refreshPlayers();
  syncCamSel();
  flash(show ? "Video boxes: shown · B: hidden" : "Video boxes: hidden");
}

// ---------------------------------------------------------------------------- passes
/* The collage's steps (08 b, "Sandbox / collage"): each change of the layers, oldest first (Undo
   takes back the newest): {key, kind ("play" | "load" | "replay" | "clear"), id + run (its saved
   pass), stacked (Stack on: it added to the layers), replaced: the layers it took off the view
   (null: too old to undo, UNDO_STEPS)}. P.cur: the play being drawn (a step too): also started,
   dirty (drawn since it was last saved), left: Map(name -> entry) (cameras that left during it),
   players: Map(name -> Set), claimed: the cameras that drew in it. The layers are `past`'s items
   (a camera's finished drawings: key = its step, cam, pose: its camera then); the play being drawn
   is in the cameras' own tracks until it ends. */
const P = { cur: null, steps: [], byKey: new Map(), seq: 0, plays: 0, saving: Promise.resolve(), saved: [], ended: false,
            ops: new Map() };                      // per saved pass "<run>|<id>": {what, text, path, busy, bad} (video, export)
const UNDO_STEPS = 24;                             // the steps Undo can take back (older: their layers stay, no undo)
/* The pass's recording (record.js; 05, 29 Oct: every pass records itself, the mouse too): each
   camera's pose, mapped camera, space and the machine's counts, the clock, and the events. */
const rec = new Recorder();
rec.clockFn = () => [performance.now(), timeline.time];
const hubSeen = new Map();                      // the hub's players: number -> camera (phone join / leave)

/** What the recording keeps about a camera (its package, its place in the pass). */
const camInfo = (s) => ({ id: s.cam.id, clip: s.cam.clip, run_name: s.cam.run_name, run: s.cam.run, origin: s.meta.origin || null,
                          offset: s.offset.map((v) => round(v, 3)), fps: s.meta.fps, frames: s.meta.frames, points: s.data.n,
                          tint: s.tint, ...(s.growing ? { growing: true } : {}) });

/** A camera takes part in the pass from its join frame (its drawing starts there). */
function recJoin(s, reason = null) {
  rec.camera(s.name, camInfo(s));
  rec.event("camera_join", { camera: s.name, player: s.player, value: reason ? { join_frame: s.joinFrame, reason } : { join_frame: s.joinFrame } },
            performance.now(), timeline.time);
  s.recJoined = true;
}

function passBegin() {
  P.plays++;
  P.cur = newStep("play", { run: L.startRun, started: localIso(), dirty: false, left: new Map(), players: new Map(), claimed: new Set() });
  rec.begin(performance.now(), P.cur.started);
  rec.event("pass_start", { value: { site: L.site, stack: S.stack, trails: S.trails, cameras: L.spots.map((s) => s.name),
                                     phones: [...hubSeen].map(([player, spot]) => ({ player, spot })) } }, performance.now(), timeline.time);
  for (const s of L.spots) recJoin(s);
  refreshPlayers();
  syncPasses();
}

/* The current pass ends (a new one, undo, clear, another site): saved when it drew, with its
   recording to here. */
function passEnd(reason) {
  if (!P.cur) return;
  rec.event("pass_end", { value: { reason, complete: timeline.atEnd } }, performance.now(), timeline.time);
  return hasDrawing(P.cur) ? savePass(P.cur, true) : P.saving;
}

/* Whether pass p has drawn anything: a camera that left during it, or one past the frame it began
   at (a line needs two frames). */
const hasDrawing = (p) => !!p && !!L && (p.left.size > 0 || L.spots.some(drew));

/** Whether a camera drew in the current pass (past the frame it joined at). */
const drew = (s) => s.tracks.perfFrame > s.joinFrame;

/** Every camera in the scene starts the pass again from 0. */
function resetSpots() {
  for (const s of L.spots) { s.tracks.resetPerformed(); s.joinFrame = 0; s.frameShown = -1; }
}

/** A camera's drawing in a pass, with what the saved form keeps about the camera. */
function passEntry(s, snap) {
  return { cam: s.cam, meta: s.meta, data: s.data, offset: s.offset, tint: s.tint, growing: !!s.growing, snap };
}

const copySnap = (sn) => ({ ...sn, X: sn.X.slice(), kind: sn.kind.slice(), cause: sn.cause.slice(), part: sn.part.slice() });

/** Two drawings of one camera in one pass (it left and came back): the later where it drew. */
function merged(a, b) {
  if (a) mergeSnaps(a.snap, b.snap);
  return b;
}

/** Every camera's drawing in pass p (the current one): Map(name -> entry). */
function passDrawings(p) {
  const out = new Map();
  for (const [name, e] of p.left) out.set(name, { ...e, snap: copySnap(e.snap) });
  for (const s of L.spots) {
    const snap = drew(s) ? s.tracks.snapshot() : null;
    if (snap) out.set(s.name, out.has(s.name) ? merged(out.get(s.name), passEntry(s, snap)) : passEntry(s, snap));
  }
  return out;
}

/* The pass saved (<start run>/live/passes/<id>/: groundtrack/live_passes.py), the same pass again
   when it was saved before. The drawing is taken now; the saves go one after another. */
function savePass(p, ending = false) {
  if (!SAVE || bench.on || !L || R) return P.saving;
  // a recording while it records: its pass is saved at the recorder's Stop, once its points are all
  // there; a pass that ends before (another site, undo, clear) with the points so far ("growing")
  if (!ending && L.spots.some((s) => s.growing)) return P.saving;
  const dr = passDrawings(p);
  if (!dr.size) return P.saving;
  const data = {}, cameras = [];
  const files = new Set();
  const safe = (id) => String(id).replace(/[^A-Za-z0-9_.-]/g, "_");
  for (const [name, e] of dr) {
    let file = `${safe(e.cam.id)}.bin`;
    for (let k = 2; files.has(file); k++) file = `${safe(e.cam.id)}_${k}.bin`;
    files.add(file);
    data[file] = packPass(e.snap);
    cameras.push({ name, id: e.cam.id, clip: e.cam.clip, run_name: e.cam.run_name, run: e.cam.run,
                   origin: e.meta.origin || null, offset: e.offset.map((v) => round(v, 3)), fps: e.meta.fps,
                   frames: e.meta.frames, points: e.data.n, performed_to: e.snap.upTo, file, tint: e.tint,
                   players: [...(p.players.get(name) || [])], ...(e.growing ? { growing: true } : {}) });
  }
  const manifest = { site: L.site, start_spot: L.startName, started: p.started, ended: localIso(), complete: timeline.atEnd,
                     clock_s: round(timeline.time, 3), duration_s: round(timeline.duration, 3), origin: L.origin0, cameras };
  if (p === P.cur) {                               // its recording, to here (taken now: the next pass records anew)
    const r = rec.snapshot((name, info) => `${safe(info ? info.id : name)}.rec`);
    Object.assign(data, r.files);
    manifest.recording = r.meta;
  }
  p.dirty = false;
  P.saving = P.saving.then(async () => {           // (true: this save went through)
    try {
      const r = await postFiles("/api/live/pass/save", { action: "save", run: p.run, id: p.id, manifest }, data);
      p.id = r.id;
      flash(`Pass saved · ${r.id}`);
      refreshPasses();
      return true;
    } catch (e) { p.dirty = true; note(`The pass was not saved: ${e.message}`); return false; }
  });
  return P.saving;
}

/* ---- the collage's layers */

/** A step: its key, in the steps (the newest last). */
function newStep(kind, o = {}) {
  const st = { key: ++P.seq, kind, id: null, run: null, stacked: S.stack, replaced: [], ...o };
  P.byKey.set(st.key, st);
  P.steps.push(st);
  const keys = new Set(past.items.map((it) => it.key));
  P.steps.forEach((x, i) => { if (i < P.steps.length - UNDO_STEPS) x.replaced = null; });
  P.steps = P.steps.filter((x) => x.replaced || x === st || x === P.cur || keys.has(x.key));
  for (const k of [...P.byKey.keys()]) if (!keys.has(k) && !P.steps.some((x) => x.key === k)) P.byKey.delete(k);
  return st;
}

/** A finished drawing of camera `name` as a layer of step `key` (pose: its camera then). */
function addLayer(key, name, data, snap, offset, tint, pose) {
  const it = past.add(key, name, data, snap, offset, hexRGB(tint));
  it.tintHex = tint; it.tintI = TINTS.indexOf(tint); it.pose = pose || null;
  return it;
}

const layerData = (it) => ({ key: it.key, cam: it.cam, data: it.data, snap: it.snap, offset: it.obj.position.toArray(),
                             tint: it.tintHex || TINTS[0], pose: it.pose });

/** Camera `name`'s layers (those `keep` does not keep) off the view, into step st (for Undo). */
function takeLayers(st, name, keep = () => false) {
  for (const it of past.items.filter((x) => x.cam === name && !keep(x))) {
    if (st.replaced) st.replaced.push(layerData(it));
    past.drop(it);
  }
}

/** Layers taken off the view back on. */
function restoreLayers(list) {
  for (const d of list || []) addLayer(d.key, d.cam, d.data, d.snap, d.offset, d.tint, d.pose);
}

/* A camera draws in this play for the first time: its layer is redrawn (Stack off: its layers of
   other steps go; on: they stay and this one adds to them). Its own earlier stint of this play
   stays (taken back mid-play: it goes on). */
function claim(s) {
  P.cur.claimed.add(s.name);
  if (!P.cur.stacked) takeLayers(P.cur, s.name, (it) => it.key === P.cur.key);
}

/** The play being drawn ends: what its cameras in play drew stays as their layers. A play that
    changed nothing leaves no step. */
function freezePlay() {
  const cur = P.cur;
  if (!hasDrawing(cur) && !(cur.replaced && cur.replaced.length)) {
    P.steps = P.steps.filter((x) => x !== cur);
    P.byKey.delete(cur.key);
    return;
  }
  for (const s of L.spots) {
    const snap = drew(s) ? s.tracks.snapshot() : null;
    if (snap) addLayer(cur.key, s.name, s.data, snap, s.offset, s.tint, camCopy(s.inp.cam));
  }
}

/** Step st taken back: its layers go, the ones it replaced come back (a removed camera too). */
function takeBack(st) {
  past.remove(st.key);
  restoreLayers(st.replaced);
  P.steps = P.steps.filter((x) => x !== st);
  P.byKey.delete(st.key);
  if (st.kind === "remove" && !spotOf(st.camera) && !parkedOf(st.camera) && !L.ghosts.has(st.camera)) {
    const cam = L.cams.find((c) => c.name === st.camera);
    if (cam) ghostSpot(cam, st.tintI, st.pose, true);
  }
}

/* X, or the camera's x in the list: camera `name`'s layer off the view (Undo brings it back). A
   camera in play that drew in this play goes on drawing from here: what it drew so far stays in
   the play (it is saved with it) as a stint that left (recorded as a leave and a join, so the
   replay draws the same). */
function clearCamera(name) {
  if (!L || R || !name) return;
  if (isLocked(name)) { flash(`${name} is locked · unlock it to clear it`); return; }
  const s = spotOf(name), live = !!s && drew(s);
  if (!hasLayer(name) && !live) { flash(`${name}: nothing to clear`); return; }
  rec.event("clear_camera", { camera: name, player: s ? s.player : null });
  const st = newStep("clear", { camera: name });
  if (live) {
    const snap = s.tracks.snapshot();
    P.cur.left.set(name, merged(P.cur.left.get(name), passEntry(s, copySnap(snap))));
    rec.event("camera_leave", { camera: name, player: s.player, value: { reason: "clear" } });
    st.replaced.push({ key: P.cur.key, cam: name, data: s.data, snap, offset: [...s.offset], tint: s.tint, pose: camCopy(s.inp.cam) });
    s.tracks.resetPerformed();
    s.joinFrame = timeline.time > 0 ? timeline.frameOf(s) : 0;
    s.tracks.perfFrame = s.joinFrame - 1;
    s.frameShown = -1; s.rebuild = true;
    recJoin(s, "clear");
  }
  takeLayers(st, name);
  syncGhosts();
  flash(`${name}: cleared · Undo brings it back`);
  syncSpots();
  syncPasses();
  publishState(true);
}

/* The collage as data (for the collage sheet, 09): each camera with a layer in the view, in the
   site's order: {camera (name), id, clip, run (its package), tint, offset, active (in play now),
   pose ({R_wc, C}: its frozen camera; null in play), layers: oldest first, each {run, id (its saved
   pass: run + id + camera find the drawing in it; null while the play is not saved yet), key, kind
   ("play" | "load" | "replay"), camera, stacked (drawn with Stack on, over the layers before), live
   (the play being drawn now), upTo, snap (the drawing, as packPass / Past take it), data (the
   package's tracks), offset, tint, pose ({R_wc, C}: the camera when it was drawn)}}. */
function collageLayers() {
  if (!L) return [];
  const by = new Map(), push = (name, l) => { if (!by.has(name)) by.set(name, []); by.get(name).push(l); };
  const arr = (a) => (a ? Array.from(a) : null), pose = (c) => c && { R_wc: arr(c.R_wc), C: arr(c.C) };
  for (const it of past.items) {
    const st = P.byKey.get(it.key) || {};
    push(it.cam, { run: st.run ?? null, id: st.id ?? null, key: it.key, kind: st.kind || "play", camera: it.cam,
                   stacked: !!st.stacked, live: false, upTo: it.snap.upTo, snap: it.snap, data: it.data,
                   offset: it.obj.position.toArray(), tint: it.tintHex || null, pose: pose(it.pose) });
  }
  if (!R && P.cur) for (const s of L.spots) if (drew(s)) {
    const snap = s.tracks.snapshot();
    push(s.name, { run: P.cur.run, id: P.cur.id, key: P.cur.key, kind: "play", camera: s.name, stacked: P.cur.stacked, live: true,
                   upTo: snap.upTo, snap, data: s.data, offset: [...s.offset], tint: s.tint, pose: pose(s.inp.cam) });
  }
  return L.cams.filter((c) => by.has(c.name)).map((c) => {
    const layers = by.get(c.name).sort((a, b) => a.key - b.key || a.live - b.live), s = spotOf(c.name) || parkedOf(c.name);
    return { camera: c.name, id: c.id, clip: c.clip, run: c.run, tint: s ? s.tint : layers[0].tint,
             offset: s ? [...s.offset] : layers[0].offset, active: !!spotOf(c.name),
             pose: spotOf(c.name) ? null : layers[layers.length - 1].pose, layers };
  });
}

/** The clock's news: at the end the pass is finished, and saved once every camera has drawn its
    last frame (after this frame's step). */
function onClock(what) {
  if (what === "end" && !R) P.ended = true;
  if (!R) rec.event(what, { value: what === "seek" ? round(timeline.time, 4) : what === "rate" ? timeline.rate : null },
                    performance.now(), timeline.time);
  publishState(true);
}

function passEnded() {
  P.ended = false;
  if (!L || !hasDrawing(P.cur)) return;
  savePass(P.cur);
  flash("The pass ended · Space: a new pass");
}

/* Play at the end: the play just drawn is saved (when it changed since) and stays as its cameras'
   layers; every camera in play starts a new play from 0 (it redraws its layer as it draws: Stack
   off, replaced; on, added to). */
function newPass() {
  if (!L || R) return;
  passEnd("new");
  freezePlay();
  resetSpots();
  passBegin();
}

/** K: Stack passes on / off (from the next play on; what is in the view stays). */
function setStack(on) {
  S.stack = !!on;
  rec.event("stack", { value: S.stack });
  syncPanel();
  flash(S.stack ? "Stack passes: on · a play adds to its cameras' layers" : "Stack passes: off · a play replaces its cameras' layers");
}

/** The play being drawn off the view (saved first: every pass is kept): its drawing goes, the
    layers it replaced come back, the clock back to 0, a new play begins. */
function dropCurrent(reason) {
  passEnd(reason);
  takeBack(P.cur);
  resetSpots();
  timeline.pause();
  timeline.seekTime(0, false);
  passBegin();
  syncGhosts();
}

/** Undo: the last change taken back (the play being drawn when it drew; else the step before:
    a play, a Load, a replay, a camera cleared). Files stay. */
function undoPass() {
  if (!L || R) return;
  rec.event("undo");
  let done = null;
  for (let i = P.steps.length - 1; i >= 0 && !done; i--) {
    const st = P.steps[i];
    if (st === P.cur) {
      if (hasDrawing(st)) {
        const locked = lockedBy(st);
        if (locked) { done = `${locked} is locked (unlock it to undo)`; break; }
        dropCurrent("undo"); done = "this play is off the view (it is saved)";
      }
      continue;
    }
    if (!st.replaced) break;                       // too old
    const locked = lockedBy(st);
    if (locked) { done = `${locked} is locked (unlock it to undo)`; break; }
    takeBack(st);
    done = st.kind === "clear" || st.kind === "remove" ? `${st.camera} is back` : st.kind === "play" ? "the play before is off the view" :
           `the ${st.kind === "load" ? "loaded" : "replayed"} pass ${st.id || ""} is off the view`;
  }
  syncGhosts();
  flash(done ? `Undo: ${done}` : "Nothing to undo");
  syncSpots();
  syncPasses();
  publishState(true);
}

/** Clear: every layer leaves the view (the play being drawn too), but the locked cameras'. Files stay. */
function clearPasses() {
  if (!L || R) return;
  rec.event("clear");
  const cur = P.cur, kept = past.items.filter((it) => isLocked(it.cam));
  for (const it of past.items.slice()) if (!kept.includes(it)) past.drop(it);
  const mine = kept.filter((it) => it.key === cur.key).map(layerData);    // locked during this play
  const steps = [...new Set(kept.map((it) => P.byKey.get(it.key)))].filter((st) => st && st !== cur);
  for (const st of steps) st.replaced = null;      // kept; no longer undone
  P.steps = [...steps, cur];
  cur.replaced = [];
  P.byKey = new Map(P.steps.map((st) => [st.key, st]));
  dropCurrent("clear");
  if (mine.length) {                               // they stay with the play they were drawn in
    cur.replaced = null;
    P.steps.splice(P.steps.length - 1, 0, cur);
    P.byKey.set(cur.key, cur);
    restoreLayers(mine);
    syncGhosts();
  }
  const names = [...new Set(kept.map((it) => it.cam))];
  flash(names.length ? `Clear: only the locked layers stay (${names.join(", ")})` : "Clear: nothing in the view");
  syncSpots();
  publishState(true);
}

/** A package's tracks for a saved pass's camera that is not in the scene (kept while the site is open). */
async function packageData(run) {
  const s = L.spots.find((x) => x.run === run) || L.parked.find((x) => x.run === run);
  if (s) return { meta: s.meta, data: s.data };
  if (!L.extra.has(run)) {
    const meta = await fetchOk(pkg(run, "scene.json"), "json");
    L.extra.set(run, { meta, data: parseTracks(await fetchOk(pkg(run, "tracks.bin")), meta) });
  }
  return L.extra.get(run);
}

/** Whether saved pass q is in the view: the play being drawn, or a layer of it. */
const ofPass = (q) => (st) => !!st && st.id === q.id && st.run === q.run;
const inView = (q) => ofPass(q)(P.cur) || past.items.some((it) => ofPass(q)(P.byKey.get(it.key)));
/** Whether each camera of saved pass q has its layer of q in the view. */
const allShown = (q) => ofPass(q)(P.cur) ||
  (q.cameras || []).length > 0 && q.cameras.every((n) => past.items.some((it) => it.cam === n && ofPass(q)(P.byKey.get(it.key))));

/** Each camera's camera at its last recorded row (a pass's recording): name -> {R_wc, R_cw, C}. */
function lastCameras(data) {
  const last = new Map();
  for (const [n, rows] of data.cams) {
    let j = -1;
    for (let r = 0; r < rows.n; r++) if (j < 0 || rows.k(r) >= rows.k(j)) j = r;
    if (j >= 0) last.set(n, rowCamera(rows, j));
  }
  return last;
}

/* Load: a saved pass's layers back on its cameras (a step: Stack off, they replace those cameras'
   layers; on, they add to them; the play being drawn stays as it is). Its cameras not in play
   come in locked at their last recorded pose. poses: name -> camera (else from its recording). */
async function loadPass(q, poses = null) {
  if (!L || R || allShown(q)) return;
  const site = L;
  try {
    const m = await fetchOk(`/api/live/pass?run=${enc(q.run)}&id=${enc(q.id)}`, "json");
    if (!poses && m.recording) {
      try { poses = lastCameras((await fetchRecording(q)).data); } catch { poses = null; }
    }
    const got = [], skipped = [];
    for (const c of m.cameras || []) {
      const run = c.path || c.run;
      const { meta, data } = await packageData(run);
      if (L !== site) return;
      if (!fitsPackage(c, data.n)) { skipped.push(c.name); continue; }   // the package changed since
      const snap = padPass(unpackPass(await fetchOk(`/live/passdata?run=${enc(q.run)}&id=${enc(m.id)}&file=${enc(c.file)}`), c.points, c.performed_to), data.n);
      if (L !== site) return;
      got.push({ c, data, snap, offset: offsetFrom(c.origin || meta.origin, L.origin0) });
    }
    if (R) return;
    const locked = got.filter((g) => lockedHere(g.c.name)).map((g) => g.c.name);
    if (locked.length === got.length) { flash(`Pass ${m.id}: its cameras are locked (${locked.join(", ")})`); return; }
    const st = newStep("load", { id: m.id, run: q.run });
    for (const g of got) {
      if (locked.includes(g.c.name)) continue;
      if (!st.stacked) takeLayers(st, g.c.name);
      addLayer(st.key, g.c.name, g.data, g.snap, g.offset, g.c.tint || TINTS[0], camCopy(poses && poses.get(g.c.name)));
    }
    syncGhosts();
    flash(`Pass ${m.id}: its layers on its cameras` + (skipped.length ? ` (not: ${skipped.join(", ")}, its package changed)` : "") +
          (locked.length ? ` (not on ${locked.join(", ")}: locked)` : ""));
    syncSpots();
    syncPasses();
  } catch (e) { note(`Pass ${q.id}: ${e.message}`); }
}

/** A saved pass deleted (its files); in the view it stays, no longer saved. */
async function deletePass(q) {
  try {
    await postJson("/api/live/pass", { action: "delete", run: q.run, id: q.id });
    for (const p of P.steps) if (p.id === q.id && p.run === q.run) { p.id = null; p.dirty = true; }
    flash(`Pass ${q.id} deleted`);
  } catch (e) { note(`Pass ${q.id}: ${e.message}`); }
  refreshPasses();
}

async function refreshPasses() {
  const l = L;
  if (!l) return;
  let saved;
  try { saved = (await fetchOk(`/api/live/passes?run=${enc(l.startRun)}`, "json")).passes || []; }
  catch { saved = []; }
  if (L !== l) return;                           // another site opened meanwhile
  P.saved = saved;
  syncPasses();
}

// ---------------------------------------------------------------------------- replay, render (05)
/** A saved pass's manifest and recording (record.js readRecording). */
async function fetchRecording(q) {
  const m = await fetchOk(`/api/live/pass?run=${enc(q.run)}&id=${enc(q.id)}`, "json");
  const rc = m.recording;
  if (!rc) throw new Error("this pass has no recording (saved before recordings, 29 Oct)");
  const get = (f) => fetchOk(`/live/passdata?run=${enc(q.run)}&id=${enc(m.id)}&file=${enc(f)}`);
  const [clockBuf, evBuf, ...camBufs] = await Promise.all([get(rc.clock.file), get(rc.events.file), ...rc.cameras.map((c) => get(c.file))]);
  const bufs = {};
  rc.cameras.forEach((c, i) => { bufs[c.file] = camBufs[i]; });
  return { m, data: readRecording(rc, clockBuf, bufs, new TextDecoder().decode(evBuf)) };
}

/* A saved pass drives the view (src "replay"): the live pass ends (saved), the pass's cameras come
   into the scene where they stood, each one's drawing starts again and follows the recording row
   by row. mode: "watch" (at the pace it was played; Space pauses, Esc stops), "check" (all at once)
   or "render" (renderPass steps it). The other cameras wait. */
async function startReplay(q, mode = "watch") {
  if (!L) throw new Error("no site open");
  if (R) stopReplay();
  const site = L;
  timeline.pause();
  passEnd("replay"); await P.saving;               // the live pass saved first: replaying it reads this save
  if (L !== site) return null;
  freezePlay();                                    // it stays as its cameras' layers (hidden while the replay runs)
  const { m, data } = await fetchRecording(q);
  if (L !== site) return null;
  const names = [...data.cams.keys()];
  const missing = names.filter((n) => !L.cams.some((c) => c.name === n));
  if (missing.length) throw new Error(`not cameras of this site: ${missing.join(", ")}`);
  R = { q, m, data, names, mode, spots: new Set(), byName: new Map(), t: 0, speed: 1, paused: false, play: null, cancel: false };
  await Promise.all(names.map((n) => addSpot(L.cams.find((c) => c.name === n))));
  if (L !== site || !R) return null;
  const tracks = new Map();
  for (const n of names) {
    const s = spotOf(n), info = m.recording.cameras.find((c) => c.name === n);
    if (!s || (info && !fitsPackage(info, s.data.n))) { stopReplay(); throw new Error(`${n}: ${s ? "its package changed since the pass" : "not loaded"}`); }
    R.spots.add(s); R.byName.set(n, s);
    s.tracks.setMode("performed");
    tracks.set(n, s.tracks);
  }
  for (const s of L.spots) { s.tracks.resetPerformed(); s.joinFrame = 0; s.frameShown = -1; s.rebuild = true; }
  R.play = new Replay(data, tracks);
  past.group.visible = false;                      // only the pass being replayed
  for (const g of L.parked) g.group.visible = false;
  timeline.seekTime(0, false);
  if (R.byName.has(L.focus.name) === false) setFocus(R.byName.get(names[0]) || L.focus, false);
  syncPasses();
  flash(`Replay · ${q.id}${mode === "watch" ? " · Space: pause, Esc: stop" : ""}`);
  return R;
}

/** The replayed cameras that moved: their camera, video plane and drawing on screen. */
function replayShow() {
  for (const name of R.play.take()) {
    const s = R.byName.get(name);
    if (!s) continue;
    const cam = R.play.cams.get(name), f = R.play.frame.get(name);
    s.cams.update(cam);
    if (S.layers.drag) s.drag.update(cam);
    s.tracks.build(f);
    s.frameShown = f;
  }
}

/* Watch: the recording at the pace it was played (its own clock), the videos kept on its shared
   clock as the live view keeps them. */
function stepReplay(dt) {
  if (!R.play || R.mode !== "watch") return;
  if (!R.paused) R.t += Math.min(dt, 0.1) * R.speed;
  R.play.advance(R.t);
  replayShow();
  const T = R.play.T, end = timeline.duration - 0.05;
  if (R.play.playing && !R.paused && T < end) {
    if (!timeline.playing || Math.abs(timeline.time - T) > 0.3) { timeline.seekTime(T, false); timeline.play(); }
  } else {
    if (timeline.playing) timeline.pause();
    if (Math.abs(timeline.time - T) > 1e-3) timeline.seekTime(T, false);
  }
  timeline.poll();
  if (R.play.done) finishReplay();
}

/** The replay ends: what it drew stays as its cameras' layers (as Load puts them), the live view
    goes on. */
function finishReplay() {
  if (!R) return null;
  const dr = R.play ? R.play.drawings() : new Map();
  if (dr.size) {
    const st = newStep("replay", { id: R.q.id, run: R.q.run });
    for (const [name, snap] of dr) {
      const s = R.byName.get(name);
      if (lockedHere(name)) continue;
      if (!st.stacked) takeLayers(st, name);
      addLayer(st.key, name, s.data, snap, s.offset, s.tint, camCopy(R.play.cams.get(name)));
    }
  }
  const out = dr;
  stopReplay();
  flash(`Replay ended · ${out.size ? "the pass stays in the view" : "nothing drawn"}`);
  return out;
}

/** Back to live: every camera's drawing starts again (a new pass), cameras no phone holds leave. */
function stopReplay() {
  if (!R) return;
  R.cancel = true;
  R = null;
  past.group.visible = true;
  for (const g of L.parked) g.group.visible = true;
  timeline.pause();
  for (const s of L.spots) { s.tracks.resetPerformed(); s.joinFrame = 0; s.frameShown = -1; s.poseVersion = -1; s.rebuild = true; spotTrails(s); }
  timeline.seekTime(0, false);
  passBegin();
  setPlayers(L.players);
  syncGhosts();
  syncPasses();
  publishState(true);
}

/* Check: the pass replayed all at once against what it saved (its .bin files) and the counts it
   recorded: {cameras: {name: {points, max_m, kind, nan, counts_rows, counts_bad}}, seconds}. */
async function checkReplay(q) {
  await startReplay(q, "check");
  const play = R.play, m = R.m, res = { cameras: {}, seconds: play.seconds, rows: 0 };
  const bad = new Map(), seen = new Map();
  play.onRow = (name, rows, r, tv) => {
    if (!(rows.flags(r) & PERFORMED)) return;
    const c = machineCounts(tv.d, tv.out, tv.perf.state, Math.min(rows.frame(r), tv.d.frames - 1), new Uint16Array(5));
    const want = rows.view(r).counts;
    seen.set(name, (seen.get(name) || 0) + 1);
    if (want.some((v, i) => v !== c[i])) bad.set(name, (bad.get(name) || 0) + 1);
  };
  play.run();
  const dr = play.drawings();
  for (const c of m.cameras || []) {
    const mine = dr.get(c.name);
    const saved = padPass(unpackPass(await fetchOk(`/live/passdata?run=${enc(q.run)}&id=${enc(m.id)}&file=${enc(c.file)}`), c.points, c.performed_to),
                          mine ? mine.kind.length : c.points);
    res.cameras[c.name] = mine ? { ...compareDrawings(mine, saved), counts_rows: seen.get(c.name) || 0, counts_bad: bad.get(c.name) || 0 }
                               : { missing: true };
  }
  finishReplay();
  return res;
}

/* Render: the pass replayed offline at a fixed rate into a video (05: 1920 x 1080, 30 fps, H.264):
   output frame j shows the recording at j / fps s, each camera's video on the frame it drew, the
   view as it is now (the orbit, or a preset; view 4 follows the focus camera's recorded camera).
   Each frame goes to the app as a PNG; the app encodes them (ffmpeg) into <start run>/live/renders/
   and deletes the PNGs unless keep. */
async function renderPass(q, { fps = 30, width = 1920, height = 1080, keep = false, onProgress = null } = {}) {
  await startReplay(q, "render");
  const r0 = R, N = Math.max(1, Math.round(R.play.seconds * fps));
  const viewName = L.locked ? (L.locked === "virtual" ? "virtual" : "start") : ({ 1: "plan", 2: "axo" }[S.view] || "orbit");
  const b = await postJson("/api/live/render", { action: "begin", run: q.run, id: q.id, fps, width, height, frames: N, view: viewName });
  const pr = renderer.getPixelRatio();
  renderer.setPixelRatio(1);
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  if (L.locked) camera.fov = lockedFov();
  camera.updateProjectionMatrix();
  let ok = false;
  try {
    for (let j = 0; j < N; j++) {
      if (R !== r0 || r0.cancel) throw new Error("render stopped");
      R.play.advance(j / fps);
      replayShow();
      await replayVideos();
      const fs = L.focus;
      if (L.locked === "virtual") {
        const c = R.play.cams.get(fs.name) || fs.inp.cam;
        camera.position.set(...toScene(fs, c.C)); cvQuat(c.R_wc, camera.quaternion);
      }
      dashes(fs);
      renderer.render(scene3, camera);
      const png = await new Promise((res) => canvas.toBlob(res, "image/png"));
      const rr = await fetch(`/api/live/render/frame?token=${enc(b.token)}&frame=${j}`, {
        method: "POST", headers: { "X-Groundtrack": "1", "Content-Type": "application/octet-stream" }, body: png });
      if (!rr.ok) throw new Error(`frame ${j}: ${rr.status}`);
      if (onProgress) onProgress(j + 1, N);
    }
    R.play.run();                                  // the rest (after the last output frame): the drawing complete
    ok = true;
  } finally {
    renderer.setPixelRatio(pr);
    resize();
    if (!ok) { postJson("/api/live/render", { action: "cancel", token: b.token }).catch(() => {}); if (R === r0) stopReplay(); }
  }
  finishReplay();
  return postJson("/api/live/render", { action: "finish", token: b.token, keep });
}

/** Render: each replayed camera's video on the frame it drew (seeked, then presented). */
async function replayVideos() {
  if (!S.layers.video) return;
  await Promise.all([...R.spots].map((s) => new Promise((res) => {
    const f = R.play.frame.get(s.name) | 0, v = s.video;
    if (s.renderFrame === f || !s.cams.plane.visible) { res(); return; }
    s.renderFrame = f;
    const done = () => { clearTimeout(tm); s.cams.texture.needsUpdate = true; res(); };
    const tm = setTimeout(done, 3000);
    if (!v.paused) v.pause();
    v.addEventListener("seeked", () => { if ("requestVideoFrameCallback" in v) v.requestVideoFrameCallback(() => done()); else done(); }, { once: true });
    v.currentTime = (f + 0.5) / s.meta.fps;
  })));
}

// ---------------------------------------------------------------------------- the sheet (09)
/* The sheet's large view, a fixed preset (the same for every sheet): from the south-south-west,
   about 40° above the ground, 35° high, framed to the reach of the pass's cameras (each one's view
   out to R_max on its ground), north away from the viewer. */
const SHEET_FROM = new THREE.Vector3(-0.42, -0.62, 0.62).normalize();
const SHEET_FOV = 35;
const SHEET_FILL = 0.92;                        // the reach fills this much of the frame
const SHEET_PX = [3048, 1016];                  // layout C's view (the default), 258 x 86 mm at 300 dpi (live_sheet.view_size)

function sheetPose(spots, aspect) {
  const pts = [];
  for (const s of spots) {
    const C = toScene(s, s.sc.C), half = Math.atan(s.sc.W / 2 / s.sc.K[0]), r = s.sc.R_max, z = s.offset[2];
    pts.push(new THREE.Vector3(...C));
    for (let i = 0; i <= 12; i++) {
      const a = s.sc.yaw - half + 2 * half * i / 12;
      pts.push(new THREE.Vector3(C[0] + r * Math.sin(a), C[1] + r * Math.cos(a), z));
    }
  }
  const target = new THREE.Box3().setFromPoints(pts).getCenter(new THREE.Vector3());
  const cam = new THREE.PerspectiveCamera(SHEET_FOV, aspect, 0.05, 1e5);
  cam.up.set(0, 0, 1);
  const v = new THREE.Vector3();
  const reach = (D) => {
    cam.position.copy(target).addScaledVector(SHEET_FROM, D);
    cam.lookAt(target);
    cam.updateMatrixWorld();
    cam.updateProjectionMatrix();
    let m = 0;
    for (const p of pts) { v.copy(p).project(cam); m = Math.max(m, Math.abs(v.x), Math.abs(v.y)); }
    return m;
  };
  let lo = 1, hi = 1e5;
  for (let i = 0; i < 60; i++) { const mid = Math.sqrt(lo * hi); if (reach(mid) > SHEET_FILL) lo = mid; else hi = mid; }
  reach(hi);
  return { pos: cam.position.clone(), quat: cam.quaternion.clone(), fov: SHEET_FOV, dist: hi };
}

/* The pass's finished drawing (its saved .bin files, as the list's Load shows it) alone, its cameras
   where they stood and each virtual camera at its last recorded row, no video (no people), from
   the fixed preset at width x height, lines widened for print: a PNG. For the sheet page only (it
   leaves the view as the sheet needs it). */
async function sheetView(q, { width, height }) {
  if (!L) throw new Error("no site open");
  const m = await fetchOk(`/api/live/pass?run=${enc(q.run)}&id=${enc(q.id)}`, "json");
  const names = (m.cameras || []).map((c) => c.name);
  const missing = names.filter((n) => !L.cams.some((c) => c.name === n));
  if (missing.length) throw new Error(`not cameras of this site: ${missing.join(", ")}`);
  SHEET.names = new Set(names);
  await Promise.all(names.map((n) => addSpot(L.cams.find((c) => c.name === n))));
  const last = m.recording ? lastCameras((await fetchRecording(q)).data) : new Map();
  await loadPass(q, last);
  if (!inView(q)) throw new Error("its drawing could not be loaded");
  SHEET.busy = true;
  const spots = names.map(spotOf).filter(Boolean);
  if (!spots.length) throw new Error("none of its cameras loaded");
  for (const s of L.parked) s.group.visible = false;
  for (const s of L.spots) {
    s.group.visible = SHEET.names.has(s.name);
    s.tracks.group.visible = false;               // the live drawing: not this pass's
    const c = last.get(s.name);
    if (c) s.cams.update(c);
    s.cams.virt.group.visible = !!c;              // without a recording: the start camera only
    s.cams.plane.visible = false;
  }
  if (L.section) L.section.setEnabled(false);
  const k = Math.max(1, width / 1500);
  applyLook(k);
  renderer.setPixelRatio(1);
  renderer.setSize(width, height, false);
  const pose = sheetPose(spots, width / height);
  camera.aspect = width / height;
  camera.fov = pose.fov;
  camera.position.copy(pose.pos);
  camera.quaternion.copy(pose.quat);
  camera.updateProjectionMatrix();
  const ppm = (height / 2) / (pose.dist * Math.tan(THREE.MathUtils.degToRad(pose.fov / 2)));
  for (const s of L.spots) {
    s.tracks.setDashScale(ppm);
    s.cams.virt.foot.material.dashSize = 6 * k / ppm; s.cams.virt.foot.material.gapSize = 5 * k / ppm;
  }
  past.setDashScale(ppm);
  renderer.render(scene3, camera);
  const png = await new Promise((res) => canvas.toBlob(res, "image/png"));
  if (!png) throw new Error("the render gave no image");
  return png;
}

/** The sheet page (sheet=<id>): the view rendered and posted to the app, which draws the sheet. */
async function sheetPage() {
  const q = { run: params.get("run"), id: SHEET.id };
  const size = (k, d) => Math.max(64, Math.min(8000, Math.round(+params.get(k) || d)));
  let msg;
  try {
    loading("Sheet · the view");
    const png = await sheetView(q, { width: size("sw", SHEET_PX[0]), height: size("sh", SHEET_PX[1]) });
    loading("Sheet · drawing it");
    const r = await postFiles("/api/live/sheet", { action: "view", run: q.run, id: q.id, build: params.get("build") !== "0",
                                                   layout: params.get("layout") || undefined }, { "view.png": await png.arrayBuffer() });
    msg = { type: "motus-sheet", run: q.run, id: q.id, ok: true, result: r };
  } catch (e) { msg = sheetFail(e.message); }
  loading(null);
  sheetAnswer(msg);
}

function sheetFail(message) {
  postFiles("/api/live/sheet", { action: "error", run: params.get("run"), id: SHEET.id, message }, {}).catch(() => {});
  return { type: "motus-sheet", run: params.get("run"), id: SHEET.id, ok: false, error: message };
}

function sheetAnswer(msg) {
  window.sheetDone = msg;
  if (window.parent !== window) window.parent.postMessage(msg, location.origin);
}

/* The Sheet button: the pass's sheet (light, layout C, PDF + PNG) into its folder. The view is
   rendered by a page of its own in a hidden frame (this view stays as it is). */
async function passSheet(q) {
  passOp(q, { what: "sheet", busy: true, text: "Sheet · rendering the view" });
  syncPasses();
  const f = document.createElement("iframe");
  f.className = "sheet-frame";
  f.setAttribute("aria-hidden", "true");
  f.tabIndex = -1;
  f.src = `/live?run=${enc(q.run)}&sheet=${enc(q.id)}&sw=${SHEET_PX[0]}&sh=${SHEET_PX[1]}&hub=0&save=0&video=0&panel=0&hud=0`;
  try {
    const msg = await new Promise((res, rej) => {
      const done = () => { clearTimeout(tm); window.removeEventListener("message", on); };
      const on = (e) => {
        const d = e.data;
        if (e.origin !== location.origin || !d || d.type !== "motus-sheet" || d.id !== q.id) return;
        done(); res(d);
      };
      const tm = setTimeout(() => { done(); rej(new Error("no answer in 5 min")); }, 300000);
      window.addEventListener("message", on);
      document.body.appendChild(f);
    });
    if (!msg.ok) throw new Error(msg.error || "failed");
    const fl = msg.result.files || {};
    passOp(q, { what: "sheet", text: `Sheet → ${Object.values(fl).map((x) => String(x.pdf).split(/[\\/]/).pop()).join(" · ")} (+ PNG)`,
                path: msg.result.dir || "" });
  } catch (e) { passOp(q, { what: "sheet", text: `Sheet · ${e.message}`, bad: true }); }
  finally { f.remove(); }
  syncPasses();
}

/* ---- the collage sheet and a camera's sheet (09, 29 Oct (2)): the layers as they are in the view
   (all, or one camera's), from the fixed preset at print size, rendered here in one go and the view
   put back; below, the score of the plays that made them (live_sheet.build_collage), into
   <run of the first layer>/live/sheets/. A play not saved yet is saved first. */
let collageBusy = false;
const hasSheet = (s) => hasLayer(s.name) || (!R && !!P.cur && L.spots.includes(s) && drew(s));

function collageShot(names, { width, height }) {
  const keep = new Set(names), all = [...L.spots, ...L.parked];
  const spots = all.filter((s) => keep.has(s.name));
  if (!spots.length) throw new Error("none of its cameras is in the view");
  const was = [], set = (o, v) => { was.push([o, o.visible]); o.visible = v; };
  const live = (s) => !R && !!P.cur && L.spots.includes(s) && drew(s);
  for (const s of all) {
    set(s.group, keep.has(s.name));
    set(s.tracks.group, keep.has(s.name) && live(s));   // the play being drawn: its layer, nothing else
    set(s.cams.plane, false);                     // no video (no people)
    set(s.drag.group, false);
  }
  for (const it of past.items) set(it.obj, keep.has(it.cam));
  const section = L.section && L.section.enabled, pr = renderer.getPixelRatio();
  try {
    if (section) L.section.setEnabled(false);
    const k = Math.max(1, width / 1500);
    applyLook(k);
    renderer.setPixelRatio(1);
    renderer.setSize(width, height, false);
    const pose = sheetPose(spots, width / height);
    const cam = new THREE.PerspectiveCamera(pose.fov, width / height, camera.near, camera.far);
    cam.up.set(0, 0, 1);
    cam.position.copy(pose.pos);
    cam.quaternion.copy(pose.quat);
    cam.updateMatrixWorld();
    cam.updateProjectionMatrix();
    const ppm = (height / 2) / (pose.dist * Math.tan(THREE.MathUtils.degToRad(pose.fov / 2)));
    for (const s of all) {
      s.tracks.setDashScale(ppm);
      s.cams.virt.foot.material.dashSize = 6 * k / ppm; s.cams.virt.foot.material.gapSize = 5 * k / ppm;
    }
    past.setDashScale(ppm);
    renderer.render(scene3, cam);
    return new Promise((res) => canvas.toBlob(res, "image/png"));   // the canvas as it is now
  } finally {
    for (let i = was.length - 1; i >= 0; i--) was[i][0].visible = was[i][1];
    if (section) L.section.setEnabled(true);
    applyLook();
    renderer.setPixelRatio(pr);
    resize();                                     // the next frame: the view as it was (its dashes too)
  }
}

async function collageSheet(name = null) {
  if (!L || collageBusy) return;
  const what = name ? `${name} · sheet` : "Collage sheet";
  const pick = () => collageLayers().filter((c) => (name ? c.camera === name : !L.hiddenCams.has(c.camera)));
  const unsaved = (cs) => cs.some((c) => c.layers.some((l) => !l.id));
  collageBusy = true;
  outSay(`${what} · starting`);
  syncOutput();
  try {
    let cs = pick();
    if (!cs.length) throw new Error(name ? "this camera has no layer" : "no layer shown in the view");
    if (unsaved(cs)) {                            // the play being drawn (or one still saving): saved first
      flash(`${what} · saving the play first`);
      outSay(`${what} · saving the play first`);
      if (!R && P.cur && hasDrawing(P.cur)) savePass(P.cur);
      await P.saving;
      cs = pick();
      if (unsaved(cs)) throw new Error("a play of it is not saved (saving off, or it failed)");
    }
    flash(`${what} · rendering the view`);
    outSay(`${what} · rendering the view`);
    const png = await collageShot(cs.map((c) => c.camera), { width: SHEET_PX[0], height: SHEET_PX[1] });
    if (!png) throw new Error("the render gave no image");
    const layers = cs.flatMap((c) => c.layers.map((l) => ({ run: l.run, id: l.id, camera: l.camera, key: l.key, kind: l.kind, stacked: l.stacked })));
    flash(`${what} · drawing it`);
    outSay(`${what} · drawing it`);
    const r = await postFiles("/api/live/sheet", { action: "collage", camera: name || undefined, site: L.site, layers },
                              { "view.png": await png.arrayBuffer() });
    const pdf = Object.values(r.files || {}).map((x) => String(x.pdf).split(/[\\/]/).pop()).join(" · ");
    flash(`${what} → ${shortPath(r.dir)} (${pdf} + PNG)`);
    note(`${what}: ${r.dir}`);
    outSay(`${what} → ${shortPath(r.dir)} (${pdf} + PNG)`, false, r.dir);
  } catch (e) { note(`${what}: ${e.message}`); flash(`${what} · ${e.message}`); outSay(`${what} · ${e.message}`, true); }
  finally { collageBusy = false; syncOutput(); }
}

/* ---- the collage's exports and a camera's (05, 29 Oct (3)): the layers (all, or one camera's) as
   the view shows them (a layer can be part of a pass) to Rhino, Houdini and GeoJSON in one go (fmt
   "all"; 30 Oct: or one of them, GeoJSON with the files always), by a job of the app (groundtrack
   live-layers-export) into <run of the first layer>/live/exports/<stamp>-<collage | camera id>/
   with layers.json. The play being drawn is saved first, so its layer names its pass. Rhino or
   Houdini open (the ones asked for): asked first. */
let exportBusy = false;
const FORMATS = { all: "Rhino, Houdini, GeoJSON", rhino: "Rhino", houdini: "Houdini", geojson: "GeoJSON" };

async function collageExport(name = null, fmt = "all", force = false) {
  if (!L || exportBusy) return;
  const what = `${name ? `${name} · export` : "Collage export"}${fmt === "all" ? "" : ` · ${FORMATS[fmt]}`}`;
  const pick = () => collageLayers().filter((c) => (name ? c.camera === name : !L.hiddenCams.has(c.camera)));
  exportBusy = true;
  outSay(`${what} · starting`);
  syncOutput();
  try {
    let cs = pick();
    if (!cs.length) throw new Error(name ? "this camera has no layer" : "no layer shown in the view");
    if (cs.some((c) => c.layers.some((l) => !l.id))) {
      flash(`${what} · saving the play first`);
      if (!R && P.cur && hasDrawing(P.cur)) savePass(P.cur);
      await P.saving;
      cs = pick();
    }
    const safe = (id) => String(id).replace(/[^A-Za-z0-9_.-]/g, "_"), layers = [], files = {};
    for (const c of cs) for (const l of c.layers) {
      const file = `${layers.length + 1}_${safe(c.id)}.bin`;
      files[file] = packPass(l.snap);
      layers.push({ camera: c.camera, camera_id: c.id, package: c.run, run: l.run, id: l.id, key: l.key, kind: l.kind,
                    stacked: l.stacked, up_to: l.upTo, file });
    }
    note(`${what} · ${layers.length} layer${layers.length === 1 ? "" : "s"} · starting`);
    let d;
    const rhino = fmt === "all" || fmt === "rhino", houdini = fmt === "all" || fmt === "houdini";
    try { d = await postFiles("/api/live/export/layers", { camera: name || undefined, site: L.site, layers, rhino, houdini, force }, files); }
    catch (e) {
      if (!e.confirm) throw e;
      note("");
      exportBusy = false;
      outSay("");
      if (confirm(e.message)) return await collageExport(name, fmt, true);
      syncOutput();
      return;
    }
    let j, n = 0;
    for (;;) {
      j = await fetchOk(`/api/job?id=${enc(d.id)}&since=${n}`, "json");
      n += (j.lines || []).length;
      if (j.status !== "running") break;
      note(`${what} · ${(j.last || "running").slice(0, 120)}`);
      outSay(`${what} · ${(j.last || "running").slice(0, 120)}`);
      await new Promise((res) => setTimeout(res, 700));
    }
    if (j.returncode !== 0) throw new Error((j.last || `exit ${j.returncode}`).slice(0, 200));
    flash(`${what} → ${shortPath(d.dir)} (${FORMATS[fmt]})`);
    note(`${what}: ${d.dir}`);
    outSay(`${what} → ${shortPath(d.dir)} (${FORMATS[fmt]})`, false, d.dir);
  } catch (e) { note(`${what}: ${e.message}`); flash(`${what} · ${e.message}`); outSay(`${what} · ${e.message}`, true); }
  finally { exportBusy = false; syncOutput(); }
}

// ---------------------------------------------------------------------------- the look
function lookSpot(s, k = 1) {                   // k: widths for print (sheetView)
  const K = LOOKS[S.look];
  s.ground.setLook(S.look);
  s.ground.setMode(S.ground);
  s.ground.setUnder(S.under);
  if (s.tracks.under !== S.under) { s.tracks.setUnder(S.under); s.rebuild = true; }
  s.tracks.setWidth(K.line * k);
  s.tracks.ref.material.opacity = K.ref;
  const set = (seg, w, op) => { seg.material.linewidth = seg.material.baseWidth * w * k; if (op != null) seg.material.opacity = op; };
  set(s.cams.ghost.frame, K.frame, K.ghost); set(s.cams.ghost.foot, K.frame, K.ghost);
  set(s.cams.virt.frame, K.frame, K.virt); set(s.cams.virt.foot, K.frame, K.virt);
  set(s.cams.virt.axes, K.frame, 1);
  s.drag.group.children[0].material.opacity = K.drag;
}

function applyLook(k = 1) {
  if (!L) return;
  for (const s of L.spots) lookSpot(s, k);
  for (const s of L.parked) parkLook(s, k);
  past.setWidth(LOOKS[S.look].line * k);
  past.setUnder(S.under === "veil" ? "veil" : "same");
  if (L.section) L.section.outline.material.linewidth = LOOKS[S.look].frame;
}

function layersSpot(s) {
  s.tracks.setRef(S.layers.ref);
  if (s.tracks.cause !== S.layers.cause) { s.tracks.setCause(S.layers.cause); s.rebuild = true; }
  s.drag.group.visible = S.layers.drag;
  s.cams.setVideoMode(S.layers.video ? S.video : "off");
  s.cams.plane.visible = S.layers.video && S.layers.cameras;
  s.cams.ghost.group.visible = S.layers.cameras;
  s.cams.virt.group.visible = S.layers.cameras;
  if (S.layers.drag) s.poseVersion = -1;        // draw it now
}

function setLayers() {
  if (!L) return;
  for (const s of L.spots) layersSpot(s);
  for (const s of L.parked) parkLook(s);
  past.setCause(S.layers.cause);
  $("#labels").hidden = !S.layers.cameras;
}

function setTrails(m) {
  if (S.trails !== m) rec.event("trails", { value: m });
  S.trails = m;
  if (!L) return;
  for (const s of L.spots) spotTrails(s);
  syncPanel();
}

/** What a camera draws: S.trails; with several cameras the whole clip only for the focus camera; a
    recording while it records: performed (its whole is not there yet). */
const trailsOf = (s) => (S.trails === "whole" && !s.growing && (L.spots.length <= 1 || WHOLE_ALL_SPOTS || s === L.focus) ? "whole" : "performed");

/* A camera to its mode. Its performed drawing of the pass stays while it shows the whole clip (back
   in performed it goes on from where it is). A camera that leaves the whole clip with several in
   the scene lets its pool's workers go. */
function spotTrails(s) {
  const m = trailsOf(s);
  if (s.tracks.mode === m) return;
  s.tracks.setMode(m);
  if (m === "performed") {
    s.rebuild = true;
    if (L.spots.length > 1 && s.pool && !WHOLE_ALL_SPOTS) { s.pool.close(); s.pool = null; }
  } else { ensurePool(s); s.whole.again = true; }
  s.frameShown = -1;
}

// ---------------------------------------------------------------------------- the space clutch
/* Held by any of its holders: "key" (M on the laptop), "phone <n>" (a phone's SPACE button),
   "url" / "bench" (checks). A double-tap glides the space back to level. One clutch per camera. */
const clock = () => performance.now() / 1000;

const recAt = (type, s, value = null) => rec.event(type, { camera: s ? s.name : null, player: s ? s.player : null, value },
                                                   performance.now(), timeline.time);

function spacePress(who, s = L && L.focus) {
  if (s && s.inp.press(who, clock())) { recAt("space_press", s, who); syncHud(); publishState(true); }
}

function spaceRelease(who, s = L && L.focus) {
  if (s && s.inp.release(who, clock())) { recAt("space_release", s, who); syncHud(); publishState(true); }
}

function spaceLevel(s = L && L.focus) {
  if (!s) return;
  recAt("level", s);
  s.inp.level(clock());
  flash(L.spots.length > 1 ? `${s.name}: space back to level` : "Space: back to level");
  publishState(true);
}

// M on the laptop: held = the clutch (for the mouse, on the focus camera), a double-tap levels
const mKey = { down: false, downAt: 0, tapAt: -1e9, spot: null };
function mDown() {
  if (mKey.down || !L) return;
  const now = performance.now();
  if (now - mKey.tapAt < DOUBLE_TAP_MS) { mKey.tapAt = -1e9; spaceLevel(); return; }
  mKey.down = true;
  mKey.downAt = now;
  mKey.spot = L.focus;
  spacePress("key", mKey.spot);
}

function mUp() {
  if (!mKey.down) return;
  mKey.down = false;
  const now = performance.now();
  if (now - mKey.downAt < TAP_MS) mKey.tapAt = now;
  if (L && L.spots.includes(mKey.spot)) spaceRelease("key", mKey.spot);
}
window.addEventListener("blur", () => mUp());     // never stuck

// ---------------------------------------------------------------------------- poses
/* A pose: the mouse's to the focus camera; a phone's to the camera it holds. */
function onPose(m) {
  if (!L) return;
  const s = m.src === "mouse" ? L.focus : spotOf(m.spot);
  if (s && !(R && R.spots.has(s)) && !isLocked(s.name)) s.inp.pose(m);   // a camera being replayed follows its recording
}

function zero(src = null, s = L && L.focus) {
  if (!s || (R && R.spots.has(s))) return;
  recAt("zero", s, src || s.inp.src);
  s.inp.zero(src);
}

function showGain(g) {
  $("#gain").value = String(g);
  $("#gain-v").textContent = `${g.toFixed(2)}×`;
}

function showScale(v) {
  const i = SCALES.reduce((b, x, k) => (Math.abs(x - v) < Math.abs(SCALES[b] - v) ? k : b), 0);
  $("#scale").value = String(i);
  $("#scale-v").textContent = `1 m = ${SCALES[i]} m`;
}

function setGain(g, s = L && L.focus) {
  g = Math.max(GAIN_MIN, Math.min(GAIN_MAX, Math.round(g * 100) / 100));
  if (s) { if (s.inp.gain !== g) recAt("gain", s, g); s.inp.setGain(g); }
  if (!L || s === shownSpot()) showGain(g);
}

function setScale(v, s = L && L.focus) {
  const i = SCALES.reduce((b, x, k) => (Math.abs(x - v) < Math.abs(SCALES[b] - v) ? k : b), 0);
  if (s) { if (s.inp.scale !== SCALES[i]) recAt("scale", s, SCALES[i]); s.inp.setScale(SCALES[i]); }
  if (!L || s === shownSpot()) showScale(SCALES[i]);
}

/* Smooth (08, 29 Oct (4), the user): per camera, off by default (J: the marked camera; its menu). On,
   its movement from the phone eases (pose.js SMOOTH_S, about 0.1 s behind). The pass records it (a
   "smooth" event, the SMOOTH flag on its rows) and its rows keep the eased camera: replays are exact. */
function setSmooth(on, s = L && L.focus) {
  if (s && !(R && R.spots.has(s))) {
    if (!!on !== s.inp.smooth) recAt("smooth", s, !!on);
    if (s.inp.setSmooth(on, clock())) flash(`${L.spots.length > 1 ? s.name + " · " : ""}Smooth: ${s.inp.smooth ? "on" : "off"}`);
  }
  syncCamSel();
}

/* the phone's remote (through the hub): the same actions as the keys. player: the phone's number,
   spot: the camera it holds. Space, level, gain and scale act on that camera; play, pause, seek,
   section are shared. (Phones never switch the clip or the site: the hub drops those. Record is
   ignored: recording comes with every pass, job 05.) */
function onControl(action, value, player = null, spot = null) {
  if (!L) return;
  const s = spotOf(spot);
  if (action === "space") { if (s) { if (value === true) spacePress(`phone ${player}`, s); else spaceRelease(`phone ${player}`, s); } }
  else if (action === "level") { if (s) spaceLevel(s); }
  else if (action === "play") timeline.play();
  else if (action === "pause") timeline.pause();
  else if (action === "seek" && typeof value === "number") timeline.seekTime(value);
  else if (action === "gain" && value > 0) { if (s) setGain(value, s); }
  else if (action === "scale" && value > 0) { if (s) setScale(value, s); }
  else if (action === "section") { L.section.setEnabled(!!value); syncPanel(); }
  publishState(true);
}

/* The state for the hub (and from it the phones): the shared fields, the focus camera's controls,
   and every camera of the site (each phone gets its own camera's gain, scale and space), its areas
   and the start camera (the first phone takes it), and the site (the site config's site: a new one
   releases the phones' cameras). */
function publishState(now = false) {
  if (!link || !L || !L.focus) return;
  const fs = L.focus.inp, start = L.spots.find((s) => s.start) || L.focus;
  link.sendState({
    clip: `${start.cam.clip}/${start.cam.run_name}`, clips: [], playing: timeline.playing,
    time: Math.round(timeline.time * 100) / 100, duration: Math.round(timeline.duration * 100) / 100, recording: true,
    gain: fs.gain, scale: fs.scale, src: fs.src || "none", status: fs.status, section: !!(L.section && L.section.enabled),
    space_held: fs.held, space_tilted: fs.tilted,
    spots: L.cams.map((c) => {
      const s = spotOf(c.name) || parkedOf(c.name), o = { name: c.name, clip: `${c.clip}/${c.run_name}`, in_scene: !!s,
                                      gain: s ? s.inp.gain : 1, scale: s ? s.inp.scale : 1,
                                      space_held: s ? s.inp.held : false, space_tilted: s ? s.inp.tilted : false };
      if (c.group) o.group = c.group;
      if (isLocked(c.name)) o.locked = true;
      return o;
    }),
    groups: L.groups, start_spot: L.startName, site: L.site,
  });
}

// ---------------------------------------------------------------------------- the phone hub
async function startHub() {
  if (params.get("hub") === "0") { $("#phones").textContent = $("#phones-n").textContent = "Off"; return; }
  try {
    const d = await postJson("/api/live/hub", { action: "start" });
    showHub(d);
    link = new PoseLink({
      url: d.view_url, onPose, onControl,
      onZero: (m) => { const s = spotOf(m && m.spot); if (s) zero(s.inp.phoneSrc, s); },
      onHub: (m) => {
        showPhones(m.phones);
        if (Array.isArray(m.players)) hubPlayers(m.players);
        if (Array.isArray(m.players) && !params.has("players")) setPlayers(m.players);
      },
      onStatus: (s) => {
        if (s === "connected") publishState(true);
        else if (L) for (const sp of L.spots) {                                        // the link dropped: let go
          if (sp.inp.releaseWhere((w) => w.startsWith("phone"), clock())) recAt("space_release", sp, "link");
        }
        syncHud();
      },
    });
  } catch (e) {
    $("#phones").textContent = $("#phones-n").textContent = "Off";
    $("#phone-note").textContent = `Phone link off: ${e.message}`;
  }
}

/** The hub's players against the last list: phones joining, leaving, picking a camera (recorded). */
function hubPlayers(list) {
  const now = new Map(list.filter((p) => p && p.player != null).map((p) => [p.player, p.spot ?? null]));
  for (const [n, sp] of now) {
    if (!hubSeen.has(n)) rec.event("phone_join", { player: n, camera: sp, value: { spot: sp } });
    else if (hubSeen.get(n) !== sp) rec.event("phone_spot", { player: n, camera: sp });
  }
  for (const [n, sp] of hubSeen) if (!now.has(n)) rec.event("phone_leave", { player: n, camera: sp });
  hubSeen.clear();
  for (const [n, sp] of now) hubSeen.set(n, sp);
}

function showHub(d) {
  hubInfo = d;
  const urls = d.phone_urls || [];
  const u = urls.find((x) => x.likely) || urls[0];
  $("#qr").innerHTML = u ? u.qr_svg : "";
  $("#phone-url").textContent = u ? u.url : "No network address found";
  $("#phone-note").textContent = urls.length > 1 ? `Also: ${urls.filter((x) => x !== u).map((x) => x.ip).join(", ")}` : "";
  showPhones(d.phones || 0);
}

function showPhones(n) {
  const el = $("#phones");
  el.textContent = `${n} phone${n === 1 ? "" : "s"} connected`;
  el.classList.toggle("on", n > 0);
  $("#phones-n").textContent = String(n);
  $("#phones-btn").classList.toggle("on", n > 0);
  if (hubInfo) hubInfo.phones = n;
}

$("#new-code").onclick = async () => {
  try { showHub(await postJson("/api/live/hub", { action: "new-code" })); }
  catch (e) { $("#phone-note").textContent = `New code failed: ${e.message}`; }
};

// ---------------------------------------------------------------------------- the panel
function syncPanel() {
  const pressed = (g, v) => $$(`[data-group="${g}"] button`).forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.v === v)));
  pressed("trails", S.trails);
  pressed("ground", S.ground);
  pressed("look", S.look);
  pressed("under", S.under);
  $$("[data-layer]").forEach((c) => { c.checked = !!S.layers[c.dataset.layer]; });
  $("#pass-stack").checked = S.stack;
  if (L && L.section) {
    const s = L.section;
    pressed("section", s.preset);
    pressed("secmode", s.tc.mode);
    $("#sec-on").checked = s.enabled;
    $("#sec-slab").checked = s.slab;
    $("#sec-h").value = String(s.height);
    $("#sec-h-v").textContent = `${s.height.toFixed(1)} m`;
    $("#sec-t").value = String(s.thickness);
    $("#sec-t-v").textContent = `${s.thickness.toFixed(1)} m`;
  }
}

function syncTitle() {
  if (!L) return;
  const n = L.spots.length, nT = L.spots.reduce((a, s) => a + s.data.nTracks, 0), nP = L.spots.reduce((a, s) => a + s.data.n, 0);
  $("#clip-name").textContent = L.title;
  const s0 = L.spots[0];
  $("#clip-sub").textContent = n === 1
    ? `${s0.name} · ${s0.cam.clip}/${s0.cam.run_name} · ${nT} tracks · ${nP.toLocaleString("en-GB")} points · ${s0.meta.frames} frames`
    : `${n} cameras · ${L.spots.map((s) => s.cam.id).join(" · ")} · ${nT} tracks · ${nP.toLocaleString("en-GB")} points`;
  document.title = `motus · Live · ${L.title}`;
}

/* The layers (31 Oct, Photoshop's; was the Cameras list): a row per camera in the scene (and
   loading), in the site's order: its eye (shown / hidden in the view), its tint, name and who moves
   it, its padlock. A click selects it and marks it (the mouse moves it); Shift+click selects it and
   focuses it (a locked camera stays locked). Below: + Camera, Remove, how many of the site's
   cameras are in, then the selected camera's block. */
const EYE_SVG = `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.1" aria-hidden="true"><path d="M1.5 8s2.4-4.2 6.5-4.2S14.5 8 14.5 8s-2.4 4.2-6.5 4.2S1.5 8 1.5 8z"/><circle class="on" cx="8" cy="8" r="1.9"/><path class="off" d="M3 13 13 3"/></svg>`;
const LOCK_SVG = `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.1" aria-hidden="true"><rect x="3.5" y="7.5" width="9" height="6"/><path class="shut" d="M5.5 7.5V5.2a2.5 2.5 0 0 1 5 0v2.3"/><path class="open" d="M5.5 7.5V5.2a2.5 2.5 0 0 1 5 0"/></svg>`;

function syncSpots() {
  const el = $("#spot-list");
  if (!L) { el.innerHTML = ""; delete el.dataset.key; $("#cam-sel").hidden = true; return; }
  const loads = new Set([...L.loading.keys(), ...L.ghosts].filter((n) => !spotOf(n) && !parkedOf(n)));
  const rows = L.cams.map((c) => {
    const s = spotOf(c.name) || parkedOf(c.name);
    return s ? { s, name: c.name, tint: s.tint, parked: !!s.parked }
      : loads.has(c.name) ? { s: null, name: c.name, tint: TINTS[L.tints.get(c.name) || 0] } : null;
  }).filter(Boolean);
  const key = rows.map((r) => `${r.name}:${r.s ? (r.parked ? 2 : 1) : 0}`).join("|");
  if (el.dataset.key !== key) {
    el.dataset.key = key;
    el.innerHTML = "";
    for (const r of rows) {
      const row = document.createElement("div");
      row.className = "layer";
      row.dataset.name = r.name;
      row.innerHTML = `<button class="lyr-eye">${EYE_SVG}</button><button class="spot"><span class="sw"></span><span class="nm"></span>` +
                      `<span class="who label"></span></button><button class="lyr-lock">${LOCK_SVG}</button>`;
      const [eye, b, lock] = row.children;
      b.style.setProperty("--tint", r.tint);
      b.querySelector(".nm").textContent = r.name;
      b.title = "Click: select it and move it with the mouse (one a phone holds: its phone moves it) · Shift+click or O: focus (the view frames it; W: the whole scene)";
      eye.title = "Shown in the view / hidden (it goes on drawing; the collage's output leaves it out)";
      lock.title = "Lock: nobody moves it (its phone picks again), and Clear, Remove, Undo and Load leave its layer / unlock";
      if (r.s) {
        b.onclick = (e) => { if (e.shiftKey) { toggleFocus(r.s); selectCamera(r.s.name); } else markCamera(r.s); };
        eye.onclick = () => toggleHidden(r.name);
        lock.onclick = () => setLock(r.name, !isLocked(r.name));
      } else eye.disabled = b.disabled = lock.disabled = true;
      el.appendChild(row);
    }
  }
  if (!selSpot()) sel.name = L.focus ? L.focus.name : null;
  el.querySelectorAll(".layer").forEach((row, k) => {
    const r = rows[k], [eye, b, lock] = row.children, locked = isLocked(r.name), off = L.hiddenCams.has(r.name);
    row.classList.toggle("selected", !!r.s && r.name === sel.name);
    row.classList.toggle("parked", !!r.parked);
    row.classList.toggle("locked", locked);
    row.classList.toggle("off", off);
    b.setAttribute("aria-pressed", String(!!r.s && r.s === L.focus));
    b.classList.toggle("solo", !!r.s && r.s === L.solo);
    eye.setAttribute("aria-pressed", String(!off));
    lock.setAttribute("aria-pressed", String(locked));
    b.querySelector(".who").textContent = !r.s ? "loading" : (r.s === L.solo ? "alone · " : "") + whoOf(r.s);
  });
  syncCamSel();
  const N = L.cams.length, n = L.spots.length + L.parked.length, F = [...L.spots, ...L.parked].filter((s) => isLocked(s.name)).length;
  $("#spot-sum").textContent = N > 1 ? `${n} of ${N} cameras in the scene (${n - F} in play` + (F ? `, ${F} locked)` : ")")
                                     : "The site's only camera";
  $("#cam-add").disabled = !!R || n + loads.size >= N;
}

/** Who moves camera s: a phone (P1..), the mouse (the marked one), a replay, or nobody (running where
    it stands; locked). */
const whoOf = (s) => (R && R.byName.has(s.name) ? "replay" : isLocked(s.name) ? "locked" : s.player != null ? `P${s.player}`
                      : s === L.focus ? "mouse" : "running");

/* The selected camera (30 Oct; was each camera's menu, its row's ⋯): the panel's camera block
   and Output's "Camera" act on it. It follows the marked camera (a click, Tab, a phone's panel);
   Shift+click selects a camera without marking it (a locked one stays locked). */
const sel = { name: null };
const selSpot = () => (L && sel.name && (spotOf(sel.name) || parkedOf(sel.name))) || null;
/** The camera the mouse's sliders show and set: the marked one (the mouse's, unless a phone holds it). */
const shownSpot = () => L && L.focus;

function selectCamera(name) {
  sel.name = name;
  syncSpots();
}

function syncCamSel() {
  const s = selSpot();
  $("#cam-sel").hidden = !s;
  $("#cm-remove").disabled = !s || !!R || isLocked(s.name) || L.spots.length + L.parked.length < 2;
  syncMouse();
  syncOutput();
  if (!s) return;
  const locked = isLocked(s.name);
  $("#cm-name").textContent = `${s.name} · ${whoOf(s)}`;
  $("#cm-focus").setAttribute("aria-pressed", String(L.framed === s));
  $("#cm-solo").setAttribute("aria-pressed", String(L.solo === s));
  $("#cm-video").setAttribute("aria-pressed", String(L.boxes.has(s.name) && canBox(s)));
  $("#cm-video").disabled = !canBox(s);
  $("#cm-smooth").checked = s.inp.smooth;
  $("#cm-smooth").disabled = !!(R && R.spots.has(s)) || locked;   // a replayed camera follows its recording
  $("#cm-clear").disabled = locked;
}

/* The selected camera's mouse controls (Zero, Gain, Scale) when the mouse moves it (the marked
   camera, no phone holds it); else what moves it. And the phones' popup: who holds which camera. */
function syncMouse() {
  const s = selSpot(), holder = s && heldBy(s.name), mine = !!s && s === L.focus && !holder && !isLocked(s.name);
  $("#cm-mouse").hidden = !mine;
  const why = !s || mine ? "" : isLocked(s.name) ? "Locked: nobody moves it, it does not draw and its layer stays. Its padlock puts it in play."
    : holder ? `P${holder.player} holds it: zero, gain and scale on its phone. Click a camera no phone holds to move it with the mouse.`
    : "Running where it stands: click it to move it with the mouse.";
  $("#cm-phone").hidden = !why;
  $("#cm-phone").textContent = why;
  const el = $("#holders");
  const lines = !L ? [] : L.players.filter((p) => p.spot && L.released.get(p.spot) !== p.player)
    .map((p) => [`P${p.player}`, p.spot, (spotOf(p.spot) || {}).tint]);
  const key = lines.map((x) => x.join(":")).join("|");
  if (el.dataset.key === key) return;
  el.dataset.key = key;
  el.innerHTML = "";
  for (const [who, cam, tint] of lines) {
    const row = document.createElement("span");
    row.innerHTML = `<b></b><i></i>`;
    row.firstChild.textContent = who;
    row.lastChild.textContent = cam;
    if (tint) row.lastChild.style.setProperty("--tint", tint);
    el.appendChild(row);
  }
}

/* The site (the panel's top, 31 Oct; was the top bar's): its menu the sites with Live packages and
   the saved realtime recordings (the open one marked; a recording opens on its own page); Realtime
   beside it, on / off, and while it is on, its setup page (the filming phone's code, the frame size,
   Stop). */
function menuItem(el, name, n, current, title, onclick) {
  const b = document.createElement("button");
  b.setAttribute("role", "menuitem");
  b.innerHTML = `<span class="dot"></span><span class="t"></span><span class="n"></span>`;
  b.querySelector(".t").textContent = name;
  b.querySelector(".n").textContent = n;
  b.setAttribute("aria-current", String(current));
  b.title = title;
  b.onclick = onclick;
  el.appendChild(b);
}

function syncSites() {
  const el = $("#site-list");
  el.innerHTML = "";
  const item = (name, n, current, title, onclick) => menuItem(el, name, n, current, title, () => { togglePop("site", false); onclick(); });
  for (const e of sites)
    item(e.title, `${e.cameras} camera${e.cameras === 1 ? "" : "s"}`, !REALTIME && !!L && e.site === L.site,
         `${e.title}: starts on ${e.start_id}` + (e.rmse != null ? ` (the best calibrated, ${e.rmse} m)` : "") + ", or the one used last",
         // (from realtime: the site's own page, which turns the mode off)
         () => { if (REALTIME) leaveTo(`/live?site=${enc(e.site)}`); else switchSite(e.site); });
  const norm = (p) => String(p || "").replace(/\\/g, "/").toLowerCase(), recs = recordings.slice(0, 12);
  if (recs.length) {
    el.appendChild(document.createElement("hr"));
    const h = document.createElement("h4");
    h.textContent = "Recordings";
    el.appendChild(h);
  }
  for (const r of recs) {
    item(r.title, r.complete ? fmtS(r.seconds) : "saving…", !REALTIME && !!L && norm(L.startRun) === norm(r.run),
         `${r.title}: filmed in Realtime` + (r.complete ? ` · ${fmtS(r.seconds)} · ${r.tracks} tracks` : " · still being saved"),
         () => leaveTo(`/live?run=${enc(r.run)}&ground=C`));
    if (!r.complete) el.lastChild.disabled = true;
  }
  $("#site-cur").textContent = REALTIME ? "Realtime" : L ? L.title : "Live";
  $("#rt-setup-row").hidden = !REALTIME;
  // realtime (plans/live_controller/10): one phone films, this laptop tracks and draws it now
  const rt = $("#rt-btn");
  rt.setAttribute("aria-pressed", String(REALTIME));
  rt.title = REALTIME ? "Realtime is on · off: back to the site used last (the pass drawn so far is saved first)"
                      : "Realtime: one phone films, this laptop tracks it and draws it here as it comes; the other phones turn its camera";
  if (REALTIME) $("#clip-name").textContent = "Realtime";
  else if (L && sites.some((x) => x.site === L.site))           // the start menu's Live opens it (a site, not a recording)
    try { localStorage.setItem("motus-live-site", L.site); } catch {}
}

/** The saved realtime recordings (newest first) for the site menu; it is rebuilt with them. */
async function refreshRecordings() {
  try { recordings = (await fetchOk("/api/live/realtime", "json")).recordings || []; } catch { return; }
  const open = !$("#site-menu").hidden;
  syncSites();
  if (open) ($("#site-menu").querySelector('[aria-current="true"]') || $("#site-menu").querySelector("button")).focus({ preventScroll: true });
}

/** Realtime on (its page), or off: the site used last. */
function toggleRealtime() {
  if (!REALTIME) { location.href = "/live?realtime=1"; return; }
  let last = null;
  try { last = localStorage.getItem("motus-live-site"); } catch {}
  const e = sites.find((x) => x.site === last) || sites[0];
  leaveTo(e ? `/live?site=${enc(e.site)}` : "/live/realtime");
}

/* The popups (31 Oct): the site menu and + Camera's (under their buttons), the phones' (bottom
   left). One open at a time; a click elsewhere or Esc closes it. */
const POPS = { site: ["#site-menu", "#site-btn"], cam: ["#cam-menu", "#cam-add"], phones: ["#phone", "#phones-btn"] };
const openPop = () => Object.keys(POPS).find((k) => !$(POPS[k][0]).hidden) || null;

function togglePop(name, on) {
  const m = $(POPS[name][0]), b = $(POPS[name][1]), open = on ?? m.hidden, was = openPop();
  if (open && was && was !== name) togglePop(was, false);
  if (open && name === "cam") syncCamMenu();
  if (open && name === "site") refreshRecordings();
  m.hidden = !open;
  b.setAttribute("aria-expanded", String(open));
  if (!open || name === "phones") return;
  const r = b.getBoundingClientRect();
  m.style.top = `${Math.round(r.bottom + 4)}px`;
  m.style.left = `${Math.round(Math.max(8, Math.min(r.left, innerWidth - m.offsetWidth - 8)))}px`;
  m.style.maxHeight = `${Math.max(160, Math.round(innerHeight - r.bottom - 16))}px`;
  (m.querySelector('[aria-current="true"]') || m.querySelector("button:not(:disabled)") || m).focus({ preventScroll: true });
}

/** + Camera's menu: the site's cameras not in the scene (a large site: by area). */
function syncCamMenu() {
  const el = $("#cam-list");
  el.innerHTML = "";
  if (!L) return;
  const free = L.cams.filter((c) => !spotOf(c.name) && !parkedOf(c.name) && !L.loading.has(c.name) && !L.ghosts.has(c.name));
  if (!free.length) { el.innerHTML = `<p class="hint">Every camera of the site is in the scene.</p>`; return; }
  const item = (c) => menuItem(el, c.name, "", false,
    `${c.name}: ${c.clip}/${c.run_name}` + (c.rmse != null ? ` (calibrated to ${c.rmse} m)` : "") + " · into the scene as the mouse's camera",
    () => { togglePop("cam", false); addCamera(c.name); });
  const done = new Set();
  for (const g of L.groups || []) {
    const cs = free.filter((c) => g.spots.includes(c.name) && !done.has(c.name));
    if (!cs.length) continue;
    const h = document.createElement("h4");
    h.textContent = g.name;
    el.appendChild(h);
    for (const c of cs) { done.add(c.name); item(c); }
  }
  const rest = free.filter((c) => !done.has(c.name));
  if (rest.length && done.size) el.appendChild(document.createElement("hr"));
  rest.forEach(item);
}

/* Away from this page (another site's page from realtime, a recording, the realtime setup): the pass
   drawn so far saved first (a recording still recording: with its points so far); a save that failed
   stays (its note says why). */
async function leaveTo(url) {
  if (L && !R && P.cur && hasDrawing(P.cur)) {
    flash("Saving the pass first");
    if ((await passEnd("site")) === false) return;
  }
  location.href = url;
}

/** The passes: how many are in the view, the saved ones (load, replay, delete, their output). */
function syncPasses() {
  if (!L || !P.cur) return;
  syncPassCount(true);
  timeline.setPass(P.plays, "The play being drawn (each play is saved as a pass)");
  const el = $("#pass-list");
  el.innerHTML = "";
  const busy = !!R || [...P.ops.values()].some((o) => o.busy && o.what === "video");
  for (const q of P.saved) {
    const row = document.createElement("div");
    row.className = "pass";
    row.dataset.id = q.id;
    const shown = allShown(q), rec = !!q.recorded, mine = R && R.q.id === q.id && R.q.run === q.run;
    const op = P.ops.get(opKey(q)) || {};
    row.innerHTML = `<span class="pl"></span>
      <div class="pass-do"><button class="btn sm" data-a="load"></button><button class="btn sm" data-a="replay"></button><button class="btn sm" data-a="del">Delete</button><button class="btn sm" data-a="out">Output</button></div>
      <p class="pass-op mono"></p>`;
    row.querySelector(".pl").textContent = passLabel(q);
    row.title = `${q.id} · ${(q.cameras || []).join(", ")}` + (rec ? ` · recorded ${fmtS(q.rec_s)}` : " · no recording (saved before 29 Oct)");
    const lb = row.querySelector('[data-a="load"]');
    lb.textContent = shown ? "Shown" : "Load";
    lb.disabled = shown || !!R;
    lb.title = shown ? "Its layers are on its cameras" : "Its layers back on its cameras (Stack off: in place of theirs; on: added)";
    lb.onclick = () => loadPass(q);
    const db = row.querySelector('[data-a="del"]');
    db.title = "Delete the pass and its files (click twice)";
    db.disabled = mine || !!op.busy;
    let armed = 0;
    db.onclick = () => {
      if (!armed) { db.textContent = "Sure?"; armed = setTimeout(() => { armed = 0; db.textContent = "Delete"; }, 3000); return; }
      clearTimeout(armed);
      deletePass(q);
    };
    const rb = row.querySelector('[data-a="replay"]');
    rb.textContent = mine ? "Stop" : "Replay";
    rb.disabled = !rec || (busy && !mine);
    rb.title = rec ? (mine ? "Back to live (Esc)" : "Play the pass back as it was drawn (Space pauses, Esc stops)") : "Saved before recordings: nothing to replay";
    rb.onclick = () => mine ? stopReplay() : startReplay(q, "watch").catch((e) => passOp(q, { what: "replay", text: e.message, bad: true }));
    const ob = row.querySelector('[data-a="out"]');
    ob.title = "Its sheet, video and exports: the panel's Output, on this pass";
    ob.onclick = () => { togglePassMenu(false); outputPass(q); };
    showOp(row, op);
    el.appendChild(row);
  }
  if (!P.saved.length) el.innerHTML = `<p class="hint">No saved pass yet.</p>`;
  syncOutput();
}

// ---------------------------------------------------------------------------- Output (30 Oct)
/* One place for what comes out: the collage (every layer in the view), the selected camera's layer
   or a saved pass; then its sheet, all exports in one go, or one of them (a pass: its video too).
   Was: the camera menu's Sheet / Exports, the Scene's Collage sheet / Export collage, each saved
   pass's row. */
const OUT = { scope: "all", pass: null };          // pass: opKey of the chosen saved pass

const outPassQ = () => P.saved.find((q) => opKey(q) === OUT.pass) || P.saved[0] || null;

function syncOutput() {
  if (!L) return;
  $$("#out-scope [data-scope]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.scope === OUT.scope)));
  const s = selSpot(), q = OUT.scope === "pass" ? outPassQ() : null;
  const pick = $("#out-pass");
  pick.hidden = OUT.scope !== "pass" || !P.saved.length;
  const key = P.saved.map(opKey).join("|");
  if (pick.dataset.key !== key) {
    pick.dataset.key = key;
    pick.innerHTML = "";
    for (const x of P.saved) {
      const o = document.createElement("option");
      o.value = opKey(x);
      o.textContent = passLabel(x);
      pick.appendChild(o);
    }
  }
  if (q) pick.value = opKey(q);
  const all = [...layerKeys()].map((k) => k.slice(k.indexOf("|") + 1)), shown = all.filter((c) => !L.hiddenCams.has(c));
  const n = shown.length, nc = new Set(shown).size, nh = new Set(all).size - nc;
  $("#out-what").textContent = OUT.scope === "all"
    ? (n ? `Every layer in the view: ${n} on ${nc} camera${nc === 1 ? "" : "s"}` + (nh ? ` (${nh} hidden camera${nh === 1 ? "" : "s"} left out)` : "")
         : nh ? "Every camera with a layer is hidden" : "No layer in the view yet")
    : OUT.scope === "camera" ? (s ? `${s.name}'s layer` + (hasSheet(s) ? "" : ": none yet") : "No camera selected")
    : q ? `${(q.cameras || []).join(", ")}` + (q.recorded ? ` · ${fmtS(q.rec_s)}` : " · no recording") : "No saved pass yet";
  const op = q ? P.ops.get(opKey(q)) || {} : null;
  const can = OUT.scope === "all" ? n > 0 : OUT.scope === "camera" ? !!s && hasSheet(s) : !!q;
  const busyX = OUT.scope === "pass" ? !!(op && op.busy) : exportBusy, busyS = OUT.scope === "pass" ? !!(op && op.busy) : collageBusy;
  $("#out-sheet").disabled = !can || busyS;
  for (const id of ["#out-all", "#out-rhino", "#out-houdini", "#out-geojson"]) $(id).disabled = !can || busyX;
  const vb = $("#out-video");
  vb.hidden = OUT.scope !== "pass";
  vb.disabled = !q || !q.recorded || !!R || [...P.ops.values()].some((o) => o.busy && o.what === "video");
  if (OUT.scope === "pass") showOp($("#scene-out"), op || {});
  else if (!OUT.said) showOp($("#scene-out"), {});
}

/** The output's own line (the collage's, a camera's): progress, then where the files are. */
function outSay(text, bad = false, path = "") {
  OUT.said = text ? { text, bad, path } : null;
  if (OUT.scope !== "pass") showOp($("#scene-out"), { text, bad, path });
}

/** A pass's Output button (Saved passes, L): Output on that pass, the panel open on it. */
function outputPass(q) {
  OUT.scope = "pass";
  OUT.pass = opKey(q);
  togglePanel(true);
  const sec = $("#scene-out");
  sec.scrollIntoView({ block: "nearest", behavior: "smooth" });
  sec.classList.remove("flash");
  void sec.offsetWidth;
  sec.classList.add("flash");
  syncOutput();
}

function bindOutput() {
  $$("#out-scope [data-scope]").forEach((b) => b.onclick = () => {
    OUT.scope = b.dataset.scope;
    if (OUT.scope !== "pass" && OUT.said) showOp($("#scene-out"), OUT.said);
    syncOutput();
  });
  $("#out-pass").onchange = (e) => { OUT.pass = e.target.value; syncOutput(); };
  const go = (fn) => () => {
    const q = outPassQ(), s = selSpot();
    if (OUT.scope === "pass") { if (q) fn.pass(q); }
    else if (OUT.scope === "camera") { if (s) fn.layers(s.name); }
    else fn.layers(null);
  };
  $("#out-sheet").onclick = go({ pass: passSheet, layers: (name) => collageSheet(name) });
  for (const what of ["all", "rhino", "houdini", "geojson"])
    $(`#out-${what}`).onclick = go({ pass: (q) => passExport(q, what), layers: (name) => collageExport(name, what) });
  $("#out-video").onclick = () => { const q = outPassQ(); if (q && OUT.scope === "pass") passVideo(q); };
}

// ---------------------------------------------------------------------------- per pass: video, export (05)
const RENDER = { fps: 30, width: 1920, height: 1080 };
const opKey = (q) => `${q.run}|${q.id}`;
const shortPath = (p) => String(p || "").split(/[\\/]live[\\/]/).pop();     // from the run's live/ on (the full one: its tooltip)
const fmtS = (s) => { s = Math.round(+s || 0); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; };

function showOp(row, op) {
  const el = row.querySelector(".pass-op");
  el.hidden = !op.text;
  el.textContent = op.text || "";
  el.classList.toggle("bad", !!op.bad);
  el.title = op.path || "";
}

/** A pass's progress or result under its row and in Output while it shows that pass (no rebuild). */
function passOp(q, op) {
  P.ops.set(opKey(q), op);
  const row = [...$$("#pass-list .pass")].find((r) => r.dataset.id === q.id);
  if (row) showOp(row, op);
  if (OUT.scope === "pass" && outPassQ() === q) showOp($("#scene-out"), op);
}

async function passVideo(q) {
  passOp(q, { what: "video", busy: true, text: "Video · starting" });
  syncPasses();
  try {
    const r = await renderPass(q, { ...RENDER, onProgress: (j, n) => passOp(q, { what: "video", busy: true, text: `Video · frame ${j} / ${n}` }) });
    passOp(q, { what: "video", text: `Video · ${r.frames} frames, ${fmtS(r.seconds)} → ${shortPath(r.path)}`, path: r.path });
  } catch (e) { passOp(q, { what: "video", text: `Video · ${e.message}`, bad: true }); }
  syncPasses();
}

/* An export as a job of the app (groundtrack live-pass-export): its last line while it runs, then
   where the files are. Rhino or Houdini open: asked first (the model / scene is rewritten). */
async function passExport(q, what, force = false) {
  const label = { all: "All", rhino: "Rhino", houdini: "Houdini", geojson: "GeoJSON" }[what];
  passOp(q, { what, busy: true, text: `${label} · starting` });
  syncPasses();
  try {
    let j;
    const body = { run: q.run, id: q.id, rhino: what === "rhino" || what === "all", houdini: what === "houdini" || what === "all", force };
    const r = await fetch("/api/live/export", { method: "POST", headers: { "Content-Type": "application/json", "X-Groundtrack": "1" }, body: JSON.stringify(body) });
    const d = await r.json().catch(() => ({}));
    if (r.status === 409 && d.confirm) {
      P.ops.delete(opKey(q));
      if (confirm(d.error)) return passExport(q, what, true);
      syncPasses();
      return;
    }
    if (!r.ok || d.error) throw new Error(d.error || `${r.status}`);
    let lines = [];
    for (;;) {
      j = await fetchOk(`/api/job?id=${enc(d.id)}&since=${lines.length}`, "json");
      lines = lines.concat(j.lines || []);
      if (j.status !== "running") break;
      passOp(q, { what, busy: true, text: `${label} · ${(j.last || "running").slice(0, 120)}` });
      await new Promise((res) => setTimeout(res, 700));
    }
    if (j.returncode !== 0) throw new Error((j.last || lines.filter((l) => l.trim()).pop() || `exit ${j.returncode}`).slice(0, 200));
    const dir = d.dir || "", sub = { all: "", rhino: "rhino", houdini: "houdini_pop", geojson: "data" }[what];
    const path = dir && sub ? `${dir}${dir.includes("\\") ? "\\" : "/"}${sub}` : dir;
    passOp(q, { what, text: `${label} → ${shortPath(path) || "done"}`, path });
  } catch (e) { passOp(q, { what, text: `${label} · ${e.message}`, bad: true }); }
  syncPasses();
}

function bindPanel() {
  $$("[data-group] button").forEach((b) => b.onclick = () => {
    const g = b.parentElement.dataset.group, v = b.dataset.v;
    if (g === "trails") setTrails(v);
    else if (g === "ground") { S.ground = v; applyLook(); }
    else if (g === "look") { S.look = v; applyLook(); }
    else if (g === "under") { S.under = v; applyLook(); }
    else if (g === "section" && L) { L.section.setPreset(v); if (!L.section.enabled) L.section.setEnabled(true); }
    else if (g === "secmode" && L) L.section.setMode(v);
    syncPanel();
    publishState();
  });
  $$("[data-layer]").forEach((c) => c.onchange = () => { S.layers[c.dataset.layer] = c.checked; setLayers(); });
  $("#zero").onclick = () => zero(null, shownSpot());
  $("#gain").oninput = (e) => { setGain(+e.target.value, shownSpot()); publishState(); };
  $("#scale").oninput = (e) => { setScale(SCALES[+e.target.value], shownSpot()); publishState(); };
  bindOutput();
  $("#cm-focus").onclick = () => { const s = selSpot(); if (s) toggleFocus(s); };
  $("#cm-solo").onclick = () => { const s = selSpot(); if (s) toggleSolo(s); };
  $("#cm-video").onclick = () => { const s = selSpot(); if (s) toggleBox(s); };
  $("#whole-scene").onclick = () => wholeScene();
  $("#cm-clear").onclick = () => { const s = selSpot(); if (s) clearCamera(s.name); };
  $("#cm-smooth").onchange = (e) => { const s = selSpot(); if (s) setSmooth(e.target.checked, s); else syncCamSel(); };
  for (const k of Object.keys(POPS)) $(POPS[k][1]).onclick = () => togglePop(k);
  document.addEventListener("pointerdown", (e) => {      // a popup: a click elsewhere closes it
    const o = openPop();
    if (o && !e.target.closest(POPS[o].join(", "))) togglePop(o, false);
  });
  $("#rt-btn").onclick = () => toggleRealtime();
  $("#rt-setup").onclick = () => leaveTo("/live/realtime");
  bindFolds();
  let armed = 0;
  $("#cm-remove").onclick = () => {
    const s = selSpot(), b = $("#cm-remove");
    if (!s) return;
    if (!armed) { b.textContent = "Sure?"; armed = setTimeout(() => { armed = 0; b.textContent = "Remove"; }, 3000); return; }
    clearTimeout(armed); armed = 0; b.textContent = "Remove";
    removeCamera(s.name);
  };
  $("#sec-on").onchange = (e) => { L.section.setEnabled(e.target.checked); publishState(); };
  $("#sec-slab").onchange = (e) => { L.section.setSlab(e.target.checked); syncPanel(); };
  $("#sec-h").oninput = (e) => { L.section.setHeight(+e.target.value); syncPanel(); };
  $("#sec-t").oninput = (e) => { L.section.setThickness(+e.target.value); syncPanel(); };
  $("#sec-flip").onclick = () => L.section.flip();
  $("#pass-undo").onclick = () => undoPass();
  $("#pass-clear").onclick = () => clearPasses();
  $("#pass-stack").onchange = (e) => setStack(e.target.checked);
  $("#pass-menu-btn").onclick = () => togglePassMenu();
  $("#pass-menu-close").onclick = () => togglePassMenu(false);
  $$("#views [data-view]").forEach((b) => b.onclick = () => setView(+b.dataset.view));
  $("#panel-btn").onclick = () => togglePanel();
  $("#fs-btn").onclick = () => toggleFullscreen();
  // back where Live was opened from: its run in the app, else the start menu
  let fromRun = false;
  try { fromRun = !!sessionStorage.getItem("gt.run"); } catch {}
  $("#back").textContent = fromRun ? "\u2190 Run" : "\u2190 Menu";
  $("#back").title = fromRun ? "Back to the run" : "Back to the start menu";
  $("#back").onclick = () => { location.href = "/"; };
}

/* The panel's folded parts (Display, Help; 30 Oct): open / closed, remembered on this computer. */
function bindFolds() {
  for (const part of $$("[data-fold]")) {
    const b = part.querySelector(".fold-h"), body = $(`#${b.getAttribute("aria-controls")}`), key = `motus-live-fold-${part.dataset.fold}`;
    const set = (open) => { b.setAttribute("aria-expanded", String(open)); body.hidden = !open; };
    let open = false;
    try { open = localStorage.getItem(key) === "1"; } catch {}
    set(open);
    b.onclick = () => {
      open = body.hidden;
      set(open);
      try { localStorage.setItem(key, open ? "1" : "0"); } catch {}
    };
  }
}

function togglePanel(on) {
  const p = $("#panel"), open = on ?? p.classList.contains("closed");
  p.classList.toggle("closed", !open);
  $("#panel-btn").setAttribute("aria-pressed", String(open));
}

/** L: the saved passes' menu (Load, Replay, Video, Sheet, Delete, exports) open / closed. */
function togglePassMenu(on) {
  const m = $("#pass-menu"), open = on ?? m.hidden;
  m.hidden = !open;
  $("#pass-menu-btn").setAttribute("aria-pressed", String(open));
  if (open) { refreshPasses(); m.focus({ preventScroll: true }); }
}

/* Focus (29 Oct (4), after the phone test): the view flies to frame s and its traces (the plan or
   the perspective, whichever is on; from a camera's own view or a free orbit, the last of the two);
   the other cameras stay. s in play becomes the marked camera; a locked one can be framed too (with
   Solo on, the one shown alone follows). null, the whole scene: it flies back out to every camera
   of the collage (Solo off). */
function setFramed(s, notify = true) {
  if (!L) return;
  L.framed = s && (L.spots.includes(s) || L.parked.includes(s)) ? s : null;
  if (!L.framed) L.solo = null;
  else if (L.solo && L.solo !== L.framed) L.solo = L.framed;
  if (L.framed && !L.framed.parked && L.focus !== L.framed) setFocus(L.framed, false);
  syncSpots();
  setView(S.view === 1 || S.view === 2 ? S.view : lastOverview);
  if (notify) flash(L.framed ? `Focus: ${L.framed.name} · O or W: the whole scene` : "The whole scene");
}

const toggleFocus = (s) => setFramed(L && L.framed === s ? null : s);
const wholeScene = () => setFramed(null);

/* Solo (29 Oct; was Focus until 29 Oct (4)): s alone in the main view (its camera, video and layers)
   against the whole collage; the rest is hidden there only: the cameras in play go on drawing and
   are recorded, their panels stay. null: the whole collage again. s in play becomes the marked
   camera; a locked one can be shown alone too. */
function setSolo(s, notify = true) {
  if (!L) return;
  L.solo = s && (L.spots.includes(s) || L.parked.includes(s)) ? s : null;
  L.framed = L.solo;
  if (L.solo && !L.solo.parked && L.focus !== L.solo) setFocus(L.solo, false);
  syncSpots();
  if (S.view === 1 || S.view === 2) setView(S.view);
  if (notify) flash(L.solo ? `${L.solo.name} alone · I or its menu: all` : "The whole collage");
}

const toggleSolo = (s) => setSolo(L && L.solo === s ? null : s);

/** The main view's render without the cameras the eye hid, and with Solo on, with only the solo
    camera (the panels draw everything after). */
const viewHidden = [];
function viewHide(hide) {
  if (!hide) { for (const o of viewHidden) o.visible = true; viewHidden.length = 0; return; }
  if (!L || (!L.solo && !L.hiddenCams.size)) return;
  const out = (name) => (L.solo && name !== L.solo.name) || L.hiddenCams.has(name);
  const hideIt = (o) => { if (o.visible) { o.visible = false; viewHidden.push(o); } };
  for (const s of [...L.spots, ...L.parked]) if (out(s.name)) hideIt(s.group);
  for (const it of past.items) if (out(it.cam)) hideIt(it.obj);
}

// ---------------------------------------------------------------------------- window, full screen
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

let fsOn = false;
async function toggleFullscreen() {
  const api = winApi();
  if (api && api.fullscreen) { await api.fullscreen(); fsOn = !fsOn; }
  else if (!document.fullscreenElement) { await document.documentElement.requestFullscreen().catch(() => {}); fsOn = !!document.fullscreenElement; }
  else { await document.exitFullscreen().catch(() => {}); fsOn = false; }
  document.documentElement.classList.toggle("fs", fsOn);
}
document.addEventListener("fullscreenchange", () => {
  if (!winApi() || !winApi().fullscreen) { fsOn = !!document.fullscreenElement; document.documentElement.classList.toggle("fs", fsOn); }
});
// in full screen the bar peeks in when the pointer reaches the top
window.addEventListener("pointermove", (e) => { $("#bar").classList.toggle("peek", fsOn && e.clientY < 52); }, { passive: true });

// ---------------------------------------------------------------------------- keys
window.addEventListener("keydown", (e) => {
  const t = e.target;
  if (t && (t.tagName === "INPUT" && t.type !== "range" && t.type !== "checkbox" || t.tagName === "SELECT" || t.tagName === "TEXTAREA")) return;
  if (e.ctrlKey || e.metaKey) return;
  const pop = openPop();
  if (pop) {                                       // a popup: Esc closes it; in a menu the arrows move
    if (e.key === "Escape") { togglePop(pop, false); $(POPS[pop][1]).focus(); e.preventDefault(); return; }
    if (pop !== "phones") {
      const items = $$(`${POPS[pop][0]} button:not(:disabled)`), i = items.indexOf(document.activeElement), down = e.key === "ArrowDown";
      if ((down || e.key === "ArrowUp") && items.length) {
        items[i < 0 ? (down ? 0 : items.length - 1) : (i + (down ? 1 : items.length - 1)) % items.length].focus();
        e.preventDefault();
        return;
      }
      if (e.key === "Enter" || e.key === " ") return;              // the item's own click
      togglePop(pop, false);
    }
  }
  if (!L || !L.focus) return;
  const k = e.key, gain = L.focus.inp.gain;
  if (R) {                                         // a replay: Space pauses it, Esc stops it
    if (k === "Escape" && !$("#pass-menu").hidden) togglePassMenu(false);       // the menu first
    else if (k === " " && R.mode === "watch") { R.paused = !R.paused; flash(R.paused ? "Replay paused" : "Replay"); }
    else if (k === "Escape" && R.mode === "watch") { stopReplay(); flash("Replay stopped"); }
    else if (!"hHpPfFlL1234".includes(k)) return;
    if (k === " " || k === "Escape") { e.preventDefault(); syncPasses(); return; }
  }
  if (k === " ") timeline.toggle();
  else if (k === "z" || k === "Z") zero();
  else if (k === "g") setGain(GAINS.find((x) => x > gain + 1e-6) ?? GAINS[0]);      // the next step up (3: back to 0.1)
  else if (k === "G") setGain(GAINS.findLast((x) => x < gain - 1e-6) ?? GAINS[GAINS.length - 1]);
  else if (k === "k" || k === "K") setStack(!S.stack);
  else if (k === "l" || k === "L") togglePassMenu();
  else if (k === "Escape" && !$("#pass-menu").hidden) togglePassMenu(false);
  else if (k === "o" || k === "O") toggleFocus(L.focus);
  else if (k === "w" || k === "W") wholeScene();
  else if (k === "i" || k === "I") setSolo(L.solo ? null : L.focus);
  else if (k === "b" || k === "B") toggleBoxes();
  else if (k === "x" || k === "X") clearCamera((L.solo || L.focus).name);
  else if (k === "j" || k === "J") setSmooth(!L.focus.inp.smooth);
  else if (k === "s" || k === "S") { L.section.toggle(); syncPanel(); }
  else if (k === "h" || k === "H") $("#hud").hidden = !$("#hud").hidden;
  else if (k === "p" || k === "P") togglePanel();
  else if (k === "f" || k === "F") toggleFullscreen();
  else if (k === "v" || k === "V") cycleVideo();
  else if (k === "m" || k === "M") { if (!e.repeat) mDown(); }
  else if (k === "c" || k === "C") toggleCause();
  else if (k === "u" || k === "U") cycleUnder();
  else if (k === "Tab" && L.spots.length > 1) nextFocus(e.shiftKey ? -1 : 1);
  else if (k >= "1" && k <= "4") setView(+k);
  else if (k === "ArrowRight") timeline.step(e.shiftKey ? 10 : 1);
  else if (k === "ArrowLeft") timeline.step(e.shiftKey ? -10 : -1);
  else if (k === "Alt") { /* the roll drag: no menu */ }
  else return;
  e.preventDefault();
  publishState();
});
window.addEventListener("keyup", (e) => { if (e.key === "m" || e.key === "M") mUp(); });

/* C: cause mode. The gaps Live caused violet dashes, the footage's white, the solid lines dimmed;
   off (default), every gap is white (the machine does not know who caused it). */
function toggleCause() {
  S.layers.cause = !S.layers.cause;
  setLayers();
  syncPanel();
  flash(`Cause: ${S.layers.cause ? "on (violet: caused in Live)" : "off"}`);
}

/* U: the crowd below the ground: through an opaque ground, faintly (veil, the default) <-> the same as
   above. ("dim", darker below, only by URL: under=dim.) */
function cycleUnder() {
  S.under = S.under === "veil" ? "same" : "veil";
  applyLook();
  syncPanel();
  flash(`Below the ground: ${{ same: "the same", dim: "dimmer", veil: "through the ground" }[S.under]}`);
}

/* V: re-framed (the recording fixed in the world, the violet frame a window over it) -> glued
   (the old way, the picture turns with the hand) -> off. */
function cycleVideo() {
  if (!S.layers.video) { S.layers.video = true; S.video = "reframed"; }
  else if (S.video === "reframed") S.video = "glued";
  else S.layers.video = false;
  setLayers();
  syncPanel();
  flash(`Video: ${S.layers.video ? (S.video === "reframed" ? "re-framed" : "glued") : "off"}`);
}

// ---------------------------------------------------------------------------- HUD
function note(text) { $("#h-note").textContent = text || ""; }
let flashTimer = 0;
function flash(text) {
  note(text);
  clearTimeout(flashTimer);
  flashTimer = setTimeout(() => { if ($("#h-note").textContent === text) note(""); }, 1800);
}
const fps = { n: 0, t0: performance.now(), value: 0, work: 0, workSum: 0 };

function syncHud() {
  if (!L || !L.focus || $("#hud").hidden) return;
  const fs = L.focus, inp = fs.inp, d = cameraDelta(fs.sc, inp.cam), f = timeline.frameOf(fs);
  const sgn = (v, n = 1) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(n)}`;
  $("#h-time").textContent = `${fmtTime(timeline.time)}  f ${f}/${fs.meta.frames}` +
    (timeline.live ? (timeline.playing ? "  now" : "  paused · Space: now") : timeline.playing ? "" : timeline.atEnd ? "  end" : "  paused");
  $("#h-pan").textContent = `${sgn(d.pan)}°`;
  $("#h-tilt").textContent = `${sgn(d.tilt)}°`;
  $("#h-roll").textContent = `${sgn(d.roll)}°`;
  $("#h-pos").textContent = d.move.map((v) => sgn(v, 2)).join(" ") + " m";
  $("#h-gain").textContent = `${inp.gain.toFixed(2)}×`;
  $("#h-scale").textContent = `1 m = ${inp.scale} m`;
  const st = fs.growing && RT.st && RT.st.id === RT.id ? RT.st : null;          // the recorder's, while it records
  $("#h-spot").textContent = `${fs.name} · ${fs.player != null ? "player " + fs.player : "no phone"}` + (L.spots.length > 1 ? " · Tab: next" : "") +
    (st ? ` · tracking ${st.fps_tracked} fps · ${st.lag_ms ?? "–"} ms` + (st.lost ? " · the recorder dropped out" : "") : "");
  const hub = link ? (link.status === "connected" ? `hub · ${hubInfo ? hubInfo.phones : 0} phone${hubInfo && hubInfo.phones === 1 ? "" : "s"}` : `hub ${link.status}`) : "no hub";
  const r = inp.src && inp.rates[inp.src];
  const hz = r ? ` ${performance.now() - r.last > 1500 ? 0 : Math.round(r.hz)} Hz` : "";     // 0 when it went quiet
  $("#h-src").textContent = `${inp.src || "none"} · ${inp.status}${hz} · ${hub}`;
  $("#h-src").classList.toggle("violet", !!inp.src);
  const c = fs.tracks.counts(f);
  $("#h-drawn").textContent = (c ? `${c.placed} placed · ${c.guessed} guessed · ${c.dropped} dropped · ` : "") +
    trailsWord(fs) + (S.layers.cause ? " · cause" : "") +
    (fs.rules.source === "defaults" ? " · rules · defaults" : "");    // the run has no config_used.yaml
  const ang = inp.tilted ? THREE.MathUtils.radToDeg(inp.clutch.angle).toFixed(1) + "°" : "";
  $("#h-space").textContent = inp.held ? `held · ${ang || "0.0°"}` : inp.levelling ? `levelling · ${ang}` : inp.tilted ? `turned · ${ang}` : "level";
  $("#h-space").classList.toggle("violet", inp.held || inp.tilted);
  const pool = L.spots.filter((s) => s.pool && s.whole.n), rules = L.spots.reduce((a, s) => a + s.tracks.stats.rules, 0);
  $("#h-fps").textContent = `${fps.value.toFixed(0)} · ${fps.work.toFixed(1)} ms` +
    (S.trails === "whole" && pool.length ? ` · pool ${(pool.reduce((a, s) => a + s.whole.sumMs / s.whole.n, 0)).toFixed(1)} ms` : "") +
    ` · rules ${rules.toFixed(1)} ms`;
  syncSpots();
}

/** The HUD's word for what the focus camera draws (several: the whole clip is the focus's). */
const trailsWord = (s) => (L.spots.length > 1 && S.trails === "whole" && !WHOLE_ALL_SPOTS ? "whole (this camera)" : trailsOf(s));

// ---------------------------------------------------------------------------- the frame loop
const ppmTarget = new THREE.Vector3();

const drawSpace = (s) => (TRAILS_RIDE_WITH_SPACE ? IDENTITY : s.inp.space);   // what the tracks are turned by

/* Whole clip: the camera's pool projects every point with its current camera; the rules and the
   space's turn follow here (tracks.judgeWhole, turnWhole). */
function requestWhole(s) {
  const W = s.whole;
  if (W.busy) { W.again = true; return; }
  W.busy = true; W.again = false;
  const site = L, t0 = performance.now(), cam = s.inp.cam;
  if (!s.wholeOut) s.wholeOut = { X: new Float32Array(3 * s.data.n), state: new Uint8Array(s.data.n) };
  ensurePool(s).project(cam, s.data.X, s.wholeOut).then((out) => {
    if (L !== site || !L.spots.includes(s)) return;
    W.lastMs = performance.now() - t0;
    if (++W.calls === 1) W.startMs = W.lastMs;     // the first call starts the workers: kept apart
    else { W.sumMs += W.lastMs; W.n++; if (W.n > 120) { W.sumMs = W.lastMs; W.n = 1; } }
    s.wholeLocal = out;
    if (s.tracks.mode !== "whole") return;
    s.tracks.judgeWhole(out);
    s.tracks.turnWhole(drawSpace(s));
    const f = timeline.frameOf(s);
    s.tracks.build(f);
    s.frameShown = f;
  }).catch((e) => note(`Whole clip (${s.name}): ${e.message}`)).finally(() => {
    if (L !== site || !L.spots.includes(s)) return;
    W.busy = false;
    if (W.again) requestWhole(s);
  });
}

/** One camera, one frame: its pose, cameras, crowd. */
function stepSpot(s) {
  const f = timeline.frameOf(s), inp = s.inp;
  inp.tick(clock());                             // a glide back to level goes on (an easing camera too, Smooth)
  let camChanged = false, spaceChanged = false;
  if (inp.version !== s.poseVersion) {
    s.poseVersion = inp.version;
    const cam = inp.camera(clock());
    s.cams.update(cam);
    if (S.layers.drag) s.drag.update(cam);
    camChanged = keyChanged(s.camKey, cam.R_cw, 0) | keyChanged(s.camKey, cam.C, 9);
    spaceChanged = keyChanged(s.spaceKey, inp.space, 0);
    if (spaceChanged && TRAILS_RIDE_WITH_SPACE) {
      s.tracks.crowd.matrix.set(...spaceMatrix(inp.space, s.sc.C));
      s.tracks.crowd.matrixWorldNeedsUpdate = true;
      spaceChanged = false;                        // the crowd rides in its group: nothing to redraw
    }
  }
  s.recChanged = camChanged || spaceChanged;
  if (s.tracks.mode === "performed") {
    const changed = s.tracks.perform(f, inp.cam, drawSpace(s), camChanged || spaceChanged);
    if (changed) P.cur.dirty = true;
    if (changed || f !== s.frameShown || s.rebuild) { s.tracks.build(f); s.frameShown = f; s.rebuild = false; }
  } else {
    if (camChanged || s.whole.again) requestWhole(s);
    else if ((spaceChanged || s.rebuild) && s.tracks.rWhole) {
      s.tracks.turnWhole(drawSpace(s));
      s.tracks.build(f); s.frameShown = f; s.rebuild = false;
    } else if (f !== s.frameShown && s.tracks.whole) { s.tracks.frameOnly(f); s.frameShown = f; }
  }
}

/** Dots of the dotted lines: about the same size on screen at the orbit target. */
function dashes(fs) {
  const dist = L.locked ? fs.sc.R_max * 0.5 : camera.position.distanceTo(orbit.target);
  const H = R && R.mode === "render" ? renderer.domElement.height : renderer.domElement.clientHeight;
  const ppm = (H / 2) / (Math.max(dist, 1) * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)));
  for (const s of [...L.spots, ...L.parked]) {
    s.tracks.setDashScale(ppm);
    s.cams.virt.foot.material.dashSize = 6 / ppm; s.cams.virt.foot.material.gapSize = 5 / ppm;
  }
  past.setDashScale(ppm);
}

/** The camera's state this frame into the pass's recording (a row when it changed). */
function recordSpot(s) {
  const inp = s.inp, perf = s.tracks.mode === "performed", f = timeline.frameOf(s);
  const state = perf ? s.tracks.perf.state : s.wholeLocal ? s.wholeLocal.state : null;
  const c = s.recCounts || (s.recCounts = new Uint16Array(5));
  machineCounts(s.data, s.tracks.out, state, Math.min(f, s.data.frames - 1), c);
  const flags = (s.recChanged ? CHANGED : 0) | (inp.held ? HELD : 0) | (inp.levelling ? LEVELLING : 0) |
                (inp.tilted ? TILTED : 0) | (perf ? PERFORMED : 0) | (s.recJoined ? JOINED : 0) | (inp.smooth ? SMOOTH : 0);
  rec.row(s.name, { cam: inp.cam, S: drawSpace(s), frame: f, pose: inp.src ? inp.latest[inp.src] : null, gain: inp.gain,
                    scale: inp.scale, counts: c, player: s.player, src: inp.src, status: inp.status, flags });
  s.recJoined = false;
}

let last = performance.now(), lastHud = 0;
function frame(now) {
  requestAnimationFrame(frame);
  const dt = (now - last) / 1000;
  last = now;
  if (!L || !L.focus) return;
  if (R && R.mode === "render") return;            // renderPass draws the frames
  if (SHEET && SHEET.busy) return;                 // sheetView draws its one frame
  const w0 = performance.now();
  if (mouse) mouse.tick(dt);
  if (bench.on) bench.drive(now);
  if (R) stepReplay(dt);
  else {
    timeline.poll();
    rec.frame(performance.now(), timeline.time);
    for (const s of L.spots) { stepSpot(s); recordSpot(s); if (!P.cur.claimed.has(s.name) && drew(s)) claim(s); }
    rec.commit(timeline.rate, (timeline.playing ? PLAYING : 0) | (timeline.atEnd ? AT_END : 0));
    if (P.ended) passEnded();
  }
  stepView(now);
  const fs = L.focus;
  if (L.locked === "virtual" && !anim.to) {
    camera.position.set(...toScene(fs, fs.inp.cam.C));
    cvQuat(fs.inp.cam.R_wc, camera.quaternion);
  } else if (!L.locked && orbit.enabled) orbit.update();
  dashes(fs);
  viewHide(true);
  renderer.render(scene3, camera);
  viewHide(false);
  if (panels.entries.length) {
    const W = renderer.domElement.clientWidth, H = renderer.domElement.clientHeight;
    const open = !$("#panel").classList.contains("closed");
    panels.layout(W, H, { top: (fsOn ? 0 : 40) + 12, bottom: 56, right: open ? 248 : 0 });
    panels.render(H);
  }
  placeLabels();
  const work = performance.now() - w0;
  fps.n++; fps.workSum += work;
  if (now - fps.t0 >= 1000) {
    fps.value = fps.n * 1000 / (now - fps.t0); fps.work = fps.workSum / fps.n;
    fps.n = 0; fps.workSum = 0; fps.t0 = now;
  }
  if (bench.on) bench.sample(now, work);
  if (now - lastHud > 200) { lastHud = now; syncHud(); syncPassCount(); publishState(); }
}

/** The layers' count in the panel follows the drawing (a camera's drawing of a step: one layer). */
let passShown = -1;
function syncPassCount(force = false) {
  const n = layerKeys().size;
  if (n !== passShown || force) { passShown = n; $("#pass-n").textContent = `${n} layer${n === 1 ? "" : "s"}`; syncOutput(); }
}

/** The layers in the view, "key|camera" each (the play being drawn too). */
function layerKeys() {
  const keys = new Set(past.items.map((it) => `${it.key}|${it.cam}`));
  if (L && !R && P.cur) for (const s of L.spots) if (drew(s)) keys.add(`${P.cur.key}|${s.name}`);
  return keys;
}

/** Whether a changed: copies a into key at o (a key starts as NaN: changed). */
function keyChanged(key, a, o) {
  let changed = false;
  for (let i = 0; i < a.length; i++) if (key[o + i] !== a[i]) { key[o + i] = a[i]; changed = true; }
  return changed;
}

function placeLabels() {
  if (!S.layers.cameras || !L) return;
  const spans = $("#labels").children, w = renderer.domElement.clientWidth, h = renderer.domElement.clientHeight;
  const cut = L.section && L.section.enabled ? L.section.active : null, fs = L.solo || L.focus;   // the camera shown alone
  fs.cams.virt.axisTips.forEach((tip, i) => {
    const p = ppmTarget.copy(tip).add(fs.group.position);                     // the focus camera's
    const kept = !cut || cut.every((pl) => pl.distanceToPoint(p) >= 0);     // the section hides it too
    p.project(camera);
    const vis = kept && ppmTarget.z > -1 && ppmTarget.z < 1 && Math.abs(ppmTarget.x) < 1.2 && Math.abs(ppmTarget.y) < 1.2;
    spans[i].style.display = vis ? "" : "none";
    if (vis) spans[i].style.transform = `translate(${(ppmTarget.x + 1) / 2 * w + 6}px, ${(1 - ppmTarget.y) / 2 * h - 6}px)`;
  });
}

// ---------------------------------------------------------------------------- bench (?bench=1)
/* Plays the clips with every camera's head swinging (pan +-30 deg, tilt +-4 deg, a new pose every
   frame, as a phone at 60 Hz per camera, each a little out of phase), performed then whole clip,
   and keeps fps and work per frame (and the rules' time, all cameras). benchspace=1: with every
   camera's clutch held. No pass is saved while it runs; the clock starts again at the end. */
const bench = {
  on: false, frames: [], works: [], rules: [], seq: 0,
  drive(now) {
    const t = (now - this.t0) / 1000, D = Math.PI / 180;
    L.spots.forEach((s, i) => {
      const ph = i * 0.7;
      s.inp.pose({ type: "pose", v: 1, src: "bench", seq: ++this.seq, t, status: "ok", p: [0, 0, 0],
                   q: matToQuat(headMatrix(30 * D * Math.sin(2 * Math.PI * t / 4 + ph), 4 * D * Math.sin(2 * Math.PI * t / 3 + ph), 0)) });
    });
    if (!timeline.playing) { timeline.seekTime(0, false); timeline.play(); }
  },
  sample(now, work) {
    this.frames.push(now); this.works.push(work);
    this.rules.push(L.spots.reduce((a, s) => a + s.tracks.stats.rules, 0));
  },
  async run(seconds = 8) {
    const space = params.get("benchspace") === "1";
    const out = { site: L.site, clip: L.spots.map((s) => s.cam.id).join(","), spots: L.spots.length, space,
                  points: L.spots.reduce((a, s) => a + s.data.n, 0), tracks: L.spots.reduce((a, s) => a + s.data.nTracks, 0),
                  panels: panels.entries.length, past_segments: past.segments,
                  size: [renderer.domElement.width, renderer.domElement.height], dpr: renderer.getPixelRatio(),
                  workers: Math.max(defaultWorkers(), 2), whole_all: L.spots.length <= 1 || WHOLE_ALL_SPOTS, shared: false };
    timeline.setRate(1);
    this.on = true;
    timeline.seekTime(0, false);
    timeline.play();
    this.t0 = performance.now();
    if (space) for (const s of L.spots) spacePress("bench", s);
    for (const mode of ["performed", "whole"]) {
      setTrails(mode);
      await new Promise((r) => setTimeout(r, 1500));
      this.frames = []; this.works = []; this.rules = [];
      for (const s of L.spots) { s.whole.sumMs = 0; s.whole.n = 0; }
      const calls0 = L.spots.reduce((a, s) => a + s.whole.calls, 0);
      const v0 = timeline.presented;
      await new Promise((r) => setTimeout(r, seconds * 1000));
      const fr = this.frames, dts = fr.slice(1).map((t, i) => t - fr[i]).sort((a, b) => a - b), ws = this.works.slice().sort((a, b) => a - b);
      const rs = this.rules.slice().sort((a, b) => a - b);
      const q = (a, p) => a.length ? +a[Math.min(a.length - 1, Math.floor(p * a.length))].toFixed(2) : null;
      const pooled = L.spots.filter((s) => s.whole.n);
      out[mode] = { fps: +((fr.length - 1) / ((fr[fr.length - 1] - fr[0]) / 1000)).toFixed(1), frame_ms_p50: q(dts, 0.5),
                    frame_ms_p95: q(dts, 0.95), work_ms_p50: q(ws, 0.5), work_ms_p95: q(ws, 0.95),
                    rules_ms_p50: q(rs, 0.5), rules_ms_p95: q(rs, 0.95),
                    video_fps: +((timeline.presented - v0) / seconds).toFixed(1),
                    pool_ms_per_spot: pooled.length ? +(pooled.reduce((a, s) => a + s.whole.sumMs / s.whole.n, 0) / pooled.length).toFixed(2) : null,
                    pool_calls: L.spots.reduce((a, s) => a + s.whole.calls, 0) - calls0,
                    pool_start_ms: mode === "whole" ? +Math.max(...L.spots.map((s) => s.whole.startMs)).toFixed(1) : null };
      const p0 = L.spots.find((s) => s.pool);
      if (p0) out.shared = p0.pool.shared;
    }
    if (space) for (const s of L.spots) spaceRelease("bench", s);
    setTrails("performed");
    timeline.pause();
    this.on = false;
    window.__bench = out;
    note(`bench ${JSON.stringify(out)}`);
    return out;
  },
};

// ---------------------------------------------------------------------------- realtime (10, step 3)
/** The recording being drawn as it records (or just stopped), or null. */
const rtSpot = () => (L && L.spots.find((s) => s.growing)) || null;

/* The mode on (the hub, the model: as the realtime page does) and the three polls: the laptop's
   status, the points, the picture. */
async function rtBegin() {
  let longSide = 1920;
  try { longSide = Number(localStorage.getItem("motus-rt-long-side")) || 1920; } catch {}
  try {
    RT.st = await postJson("/api/live/realtime", { action: "start", long_side: [1920, 1280].includes(longSide) ? longSide : 1920 });
  } catch (e) { loading(`Realtime did not start: ${e.message}`, true); return; }
  rtWaiting();
  rtStatus();
  rtTracks();
  rtPreview();
}

/** The laptop's status: a calibrated recording not open yet opens (the last one's pass saved first). */
async function rtStatus() {
  try { RT.st = await fetchOk("/api/live/realtime?light=1", "json"); } catch { RT.st = null; }
  const st = RT.st;
  if (st && RT.id && !RT.done && st.saved && st.saved.id === RT.id && !st.saved.complete) RT.gone = true;   // nothing to draw
  if (st && st.recording && st.calibrated && st.id && st.id !== RT.id && !RT.opening && !R && (!RT.id || RT.done || RT.gone))
    rtOpen(st);
  if (st) rtCheck(st);
  if (!L) rtWaiting();
  setTimeout(rtStatus, RT_STATUS_MS);
}

async function rtOpen(st) {
  RT.opening = true;
  try {
    await openSite(st.run, { live: st.id });
    RT.id = st.id; RT.n = L.spots[0].data.n; RT.done = RT.gone = false;
    loading(null);
    $("#loading").classList.remove("rt-wait");
    timeline.play();                               // now
    flash(`Realtime · ${L.spots[0].name}`);
  } catch (e) {
    note(`Realtime: ${e.message}`);                // (again at the next status: its package may be a moment late)
  } finally { RT.opening = false; }
}

/** No recording open yet: what the laptop waits for (the panel stays usable). */
function rtWaiting() {
  const st = RT.st, url = $("#phone-url").textContent;
  const text = !st || st.mode === "off" ? "Realtime · starting"
    : st.mode === "error" ? `Realtime · the model did not load: ${st.error}`
    : st.mode === "loading" ? "Realtime · the model is loading"
    : st.recording && !st.calibrated ? "Realtime · recording: the phone held still for a second calibrates it"
    : `Realtime · waiting for a recorder: on a phone open ${url || "the phone link (panel)"}, press Record`;
  loading(text, !!st && st.mode === "error");
  $("#loading").classList.add("rt-wait");
}

/** The points since the last ones, whole frames only (a recording that stopped: up to its last). */
async function rtTracks() {
  const s = rtSpot();
  if (s && RT.id && !RT.tBusy && !RT.opening) {
    RT.tBusy = true;
    const id = RT.id;
    try {
      const { head, bytes } = await rtFetchTracks(id, RT.n);
      if (id === RT.id && rtSpot() === s && head.n0 === RT.n) rtGrow(s, head, bytes);
    } catch { /* saving: its package is not there yet */ }
    RT.tBusy = false;
  }
  setTimeout(rtTracks, RT_TRACKS_MS);
}

function rtGrow(s, head, bytes) {
  if (bytes.byteLength || head.frames > s.data.frames) {
    s.data = appendTracks(s.data, bytes, head.frames);
    s.tracks.grow(s.data);
    RT.n = s.data.n;
    s.meta.frames = s.data.frames;
    s.meta.duration_s = s.data.frames / s.meta.fps;
    rec.camera(s.name, camInfo(s));                // the pass's recording: its points so far
    timeline.setLive(s, s.data.frames);
    if (S.layers.ref && performance.now() - RT.refAt > RT_REF_MS) { RT.refAt = performance.now(); s.tracks.refreshReference(); }
    syncTitle();
  }
  if (head.done) rtDone(s);
}

/* The recorder stopped and the last points are in: the clock runs to the end (the pass is saved
   there, with the package's n), the recording's video takes the picture's place. */
async function rtDone(s) {
  if (!s.growing) return;
  s.growing = false;
  RT.done = true;
  try { s.meta = { ...(await fetchOk(pkg(s.run, "scene.json"), "json")), frames: s.data.frames }; } catch {}
  rec.camera(s.name, camInfo(s));
  s.tracks.refreshReference();
  timeline.endLive(s, s.data.frames);
  syncTitle();
  flash("The recorder stopped · the pass is saved at the end");
  s.video.src = pkg(s.run, "video.mp4");
  await videoReady(s.video);
  if (s.video.error || !L || !L.spots.includes(s)) return;
  timeline.attach(s, s.video);
  s.cams.setSource(s.video);
}

/** The recording's newest blurred picture into its plane (while it records). */
async function rtPreview() {
  const s = rtSpot();
  if (s && RT.id && !RT.pBusy && S.layers.video) {
    RT.pBusy = true;
    try {
      const r = await fetch(`/api/live/realtime/preview?id=${enc(RT.id)}`);
      if (r.status === 200 && rtSpot() === s) {
        const img = await createImageBitmap(await r.blob()), c = s.canvas;
        if (c.width !== img.width || c.height !== img.height) { c.width = img.width; c.height = img.height; }
        c.getContext("2d").drawImage(img, 0, 0);
        img.close();
        s.cams.texture.needsUpdate = true;
      }
    } catch { /* the next one */ }
    RT.pBusy = false;
  }
  setTimeout(rtPreview, RT_PREVIEW_MS);
}

/* The people check (step 2): people that come out far from 1.7 m mean the height or the tilt is
   off; the drag grid shows the ground as this camera puts it (once per recording; the phone
   offers the fix). */
function rtCheck(st) {
  const c = st.recording && st.id === RT.id ? st.check : null;
  if (!c || c.ok || RT.checked === st.id || !L) return;
  RT.checked = st.id;
  if (!S.layers.drag) { S.layers.drag = true; setLayers(); syncPanel(); }
  note(`People look ${c.median_m} m tall (${c.error_pct > 0 ? "+" : ""}${c.error_pct} %): the camera's height or tilt is off · ` +
       "the drag grid shows the ground as it is placed · the phone offers a fix (a new recording)");
}

// ---------------------------------------------------------------------------- start
function applyParams() {
  const num = (k) => (params.has(k) ? +params.get(k) : null);
  if (params.has("space")) {                     // the space turned so (with the hand), the camera back at start in it
    const [sp, st, sr] = params.get("space").split(",").map((v) => +v || 0);
    mouse.set(0, 0, 0);
    input.current.press("url", clock());
    mouse.set(sp, st, sr);
    input.current.release("url", clock());
    mouse.set(0, 0, 0);
    zero("mouse");
  }
  if (num("pan") !== null || num("tilt") !== null || num("roll") !== null) mouse.set(num("pan") || 0, num("tilt") || 0, num("roll") || 0);
  if (params.has("section")) { L.section.setPreset(params.get("section")); L.section.setEnabled(true); }
  if (params.get("slab") === "1") L.section.setSlab(true);
  if (params.has("h")) L.section.setHeight(num("h"));
  if (params.get("panel") === "0") togglePanel(false);
  if (params.get("hud") === "0") $("#hud").hidden = true;
  if (params.has("gain")) setGain(num("gain"));
  if (params.has("view")) setView(+params.get("view"), false);
  if (params.has("orbit")) {                     // orbit=x,y,z,tx,ty,tz: the orbit camera (checks)
    const o = params.get("orbit").split(",").map(Number);
    if (o.length === 6 && o.every(Number.isFinite)) {
      unlock(); anim.to = null;
      camera.position.set(o[0], o[1], o[2]); orbit.target.set(o[3], o[4], o[5]); orbit.update();
    }
  }
  if (params.has("f")) timeline.seekFrame(num("f"));
  else if (params.has("t")) timeline.seekTime(num("t"));
  if (params.get("play") === "1") timeline.play();
  syncPanel();
}

/** players=6363,6367: phones 1, 2.. on these cameras (checks, without the hub); waits for them. */
async function devPlayers(words) {
  const list = [];
  for (const w of words.split(",")) {
    const c = findCamera(w, L.cams);
    if (c) list.push({ player: list.length + 1, spot: c.name });
    else note(`players: no camera ${w}`);
  }
  setPlayers(list);
  await Promise.all([...L.loading.values()].map((p) => p.catch(() => null)));
  sceneChanged();
}

(async function start() {
  startTips();
  bindPanel();
  resize();
  // the mouse head: its poses go through the same handler as the phone's (to the focus camera)
  mouse = new MouseController($("#stage"), onPose, { onDrag: () => {} });
  if (!SHEET && !REALTIME) postJson("/api/live/realtime", { action: "stop" }).catch(() => {});   // a site: realtime off
  try { sites = (await fetchOk("/api/live/sites", "json")).sites || []; } catch { sites = []; }
  if (!SHEET) refreshRecordings();
  let run = params.get("run");
  if (!run && params.has("site")) {
    const e = sites.find((x) => x.site === params.get("site"));
    if (e) run = e.start_run;
  }
  if (REALTIME) {                                  // the recordings open as they come (rtStatus)
    syncSites();
    rtBegin();
  } else {
    if (!run) { loading("No run: open Live from a run in the app.", true); return; }
    try {
      await openSite(run);
    } catch (e) {
      loading(e.message, true);
      if (SHEET) sheetAnswer(sheetFail(e.message));
      return;
    }
  }
  resize();
  if (L && params.has("players")) await devPlayers(params.get("players"));
  if (L && params.has("focus")) {
    const c = findCamera(params.get("focus"), L.cams), s = c && spotOf(c.name);
    if (s) setFocus(s, false);
  }
  if (L && (params.has("players") || params.has("focus"))) setView(S.view === 1 ? 1 : 2, false);
  if (L) applyParams();
  requestAnimationFrame(frame);
  if (L) loading(null);
  publishState();
  startHub();
  window.live = { get L() { return L; }, S, P, past, timeline, input, mouse, setView, setTrails, setGain, setScale, setSmooth, zero, bench,
                  get link() { return link; }, spacePress, spaceRelease, spaceLevel, toggleCause, cycleUnder, onControl, onPose,
                  setFocus, setPlayers, panels, camera, orbit, undoPass, clearPasses, clearCamera, collageLayers, setSolo, setFramed, toggleBox, toggleBoxes, markCamera, removeCamera, selectCamera, addCamera, setLock, toggleHidden, cameraAt, loadPass, deletePass, switchSite, setStack,
                  refreshPasses, get sites() { return sites; }, startReplay, stopReplay, finishReplay, checkReplay, renderPass,
                  passExport, passSheet, sheetView, collageSheet, collageExport, get rec() { return rec; }, get R() { return R; }, RT, ready: true };
  if (SHEET) sheetPage();
  if (params.get("bench") === "1") setTimeout(() => bench.run(+(params.get("secs") || 8)), 1500);
})();
