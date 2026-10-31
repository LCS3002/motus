/* Live view: the timeline (bottom bar, hides itself) and the shared clock (08 b, "Passes", 28 Oct).
   One clock T for every camera in the scene, from 0 to the end of the longest clip: no looping. A
   shorter clip holds its last frame until then. At the end everything stops (onChange("end"));
   play from there starts a new pass, every clip together from 0 (onNewPass first). A seek sets the
   same time for all (clamped per clip). Each video is kept on T: a small drift is made up by playing
   it a little faster or slower (NUDGE), only a large one is a seek (RESYNC_S: a seek to the middle of
   a keyframe interval is slow, and seeks every second keep several videos behind for good); each
   camera's tracks follow the frame its own video shows (requestVideoFrameCallback:
   mediaTime x fps), so the tracks, the video plane and the scrub bar agree to the frame. Cameras
   join and leave (add / remove, by key).
   A recording while it records (plans/live_controller/10, step 3: add with live, no video): the
   clock is "now", LIVE_DELAY_S behind the frames tracked so far (setLive) and never past them; no
   seek, no end. Pause freezes it, play goes back to now. endLive: the recorder stopped, its
   frames are all there; the clock runs on to their end as for any clip, and "end" comes. A video
   may take its place then (attach). */

import { clipTime, loopDiff } from "./spots.js";

const RATES = [0.25, 0.5, 1, 1.5, 2];
const HIDE_MS = 2500;
const RESYNC_S = 1.0;          // a video further than this from the clock is put back on it (a seek)
const RESYNC_EVERY_MS = 2500;  // at most this often per video (a seek stalls it a while)
const NUDGE_S = 0.06;          // further than this: played faster / slower until it is back on the clock
const NUDGE = 1.25;            // by this much
const STEP = 30;               // the scrub bar's steps per second
export const LIVE_DELAY_S = 0.2;   // a recording's clock: this far behind its newest tracked frame
const LIVE_JUMP_S = 1.0;       // further than this from there: put back (else eased, LIVE_EASE)
const LIVE_EASE = 0.5;         // the clock's rate, at most this much faster / slower to get there

export function fmtTime(s) {
  if (!(s >= 0)) s = 0;
  const m = Math.floor(s / 60), r = s - 60 * m;
  return `${String(m).padStart(2, "0")}:${r.toFixed(2).padStart(5, "0")}`;
}

export class Timeline {
  /* el: the bar; onChange(what): play / pause / seek / rate / end; onNewPass(): play at the end,
     before the clock goes back to 0. */
  constructor(el, onChange, onNewPass = null) {
    this.el = el;
    this.onChange = onChange;
    this.onNewPass = onNewPass;
    this.c = new Map();                 // key -> {video, fps, frames, duration, frame, fixAt, alive}
    this.T = 0;                         // the shared clock (s)
    this.T0 = 0; this.wall0 = 0;        // while playing: T = T0 + (now - wall0) x rate
    this.rate = 1;
    this.isPlaying = false;
    this.focus = null;
    this.presented = 0;                 // frames the focus camera's video presented (the bench)
    this.scrubbing = false;
    this.longest = 0;
    this.liveKey = null;                // the recording while it records (its entry: live)
    this.lastPoll = 0;
    this.noRVFC = typeof HTMLVideoElement === "undefined" || !("requestVideoFrameCallback" in HTMLVideoElement.prototype);
    this._ui();
  }

  get playing() { return this.isPlaying; }
  get time() { return this.T; }
  get duration() { return this.longest; }
  /** Whether a recording is being recorded (the clock is "now"). */
  get live() { return this.liveKey !== null; }
  /** The clock at the end of the longest clip (play starts a new pass); never while live. */
  get atEnd() { return !this.live && this.longest > 0 && this.T >= this.longest - 1e-3; }
  get _f() { return this.c.get(this.focus) || this.c.values().next().value || { fps: 30, frames: 1, frame: 0 }; }
  get fps() { return this._f.fps; }
  get frames() { return this._f.frames; }
  /** The focus camera's frame on screen (the HUD, stepping). */
  get frame() { return this._f.frame; }
  frameOf(key) { const k = this.c.get(key); return k ? k.frame : 0; }
  /** A camera's time on the shared clock (its last frame once the clock is past its end). */
  timeOf(key) { const k = this.c.get(key); return k ? clipTime(this.T, k.duration, false) : 0; }
  setFocus(key) { if (this.c.has(key)) this.focus = key; }
  has(key) { return this.c.has(key); }

  _frameAt(k) { return Math.max(0, Math.min(k.frames - 1, Math.floor(clipTime(this.T, k.duration, false) * k.fps))); }
  _held(k) { return this.T >= k.duration - 0.5 / k.fps; }                 // past its last frame: held

  /** A camera joins: its video goes to the clock's time (and plays when the clock does). live: a
      recording while it records (no video; frames: those tracked so far). */
  add(key, { video, fps, frames, live = false }) {
    if (this.c.has(key)) return;
    const k = { video: live ? null : video, fps: +fps || 30, frames: Math.max(1, frames | 0), frame: 0, fixAt: 0, alive: true };
    k.duration = k.frames / k.fps;
    this.c.set(key, k);
    if (this.focus === null) this.focus = key;
    if (live) this.liveKey = key;
    this._lengths();
    k.frame = this._frameAt(k);
    if (k.video) this._attach(key, k);
    this._show();
  }

  /** The recording's frames tracked so far (live), or all of them (endLive: the clock is a clip's
      again, and runs on to their end). */
  setLive(key, frames) {
    const k = this.c.get(key);
    if (!k) return;
    k.frames = Math.max(1, frames | 0);
    k.duration = k.frames / k.fps;
    this._lengths();
  }

  endLive(key, frames) {
    if (key !== this.liveKey) return;
    this.liveKey = null;
    this.setLive(key, frames);                          // (the scrub works again)
    this.T0 = this.T; this.wall0 = performance.now();
    this._show();
  }

  /** A video for a camera that had none (a recording once its video is made). */
  attach(key, video) {
    const k = this.c.get(key);
    if (!k || k.video || key === this.liveKey) return;
    k.video = video;
    this._attach(key, k);
    this._show();
  }

  _attach(key, k) {
    const video = k.video, f = k.frame;
    video.loop = false;
    video.playbackRate = this.rate;
    if (this.isPlaying && !this._held(k)) {
      video.currentTime = this.T;
      k.fixAt = performance.now() + RESYNC_EVERY_MS;
      const p = video.play(); if (p && p.catch) p.catch(() => {});
    } else video.currentTime = (f + 0.5) / k.fps;
    if (!this.noRVFC) {
      const tick = (now, md) => {
        if (!k.alive) return;
        k.frame = Math.max(0, Math.min(k.frames - 1, Math.round(md.mediaTime * k.fps)));
        if (key === this.focus) this.presented++;
        video.requestVideoFrameCallback(tick);
      };
      video.requestVideoFrameCallback(tick);
    }
  }

  /** A camera leaves (its video stops; the caller removes it). */
  remove(key) {
    const k = this.c.get(key);
    if (!k) return;
    k.alive = false;
    if (k.video) k.video.pause();
    this.c.delete(key);
    if (key === this.liveKey) this.liveKey = null;
    if (this.focus === key) this.focus = this.c.size ? this.c.keys().next().value : null;
    this._lengths();
    if (this.T > this.longest) this.T = this.longest;
    this.T0 = this.T; this.wall0 = performance.now();
    this._show();
  }

  _lengths() {
    this.longest = Math.max(0, ...[...this.c.values()].map((k) => k.duration));
    this.scrub.max = String(Math.max(0, Math.round(this.longest * STEP)));
    this.durEl.textContent = fmtTime(this.longest);
    this.scrub.disabled = this.live;                    // a recording while it records: no seek
  }

  /** Each render: the clock moves on, the videos are kept on it; at the end everything stops. */
  poll() {
    const now = performance.now(), dt = Math.min(0.1, Math.max(0, (now - (this.lastPoll || now)) / 1000));
    this.lastPoll = now;
    if (this.isPlaying && this.live) {                 // now: behind the newest frame, eased
      const k = this.c.get(this.liveKey), edge = k.frames / k.fps, target = Math.max(0, edge - LIVE_DELAY_S);
      let T = this.T;
      if (Math.abs(target - T) > LIVE_JUMP_S) T = target;
      else T += dt * (1 + Math.max(-LIVE_EASE, Math.min(LIVE_EASE, 2 * (target - T))));
      this.T = Math.max(0, Math.min(T, edge));
      this.T0 = this.T; this.wall0 = now;
    } else if (this.isPlaying) {
      this.T = this.T0 + (now - this.wall0) / 1000 * this.rate;
      if (this.T >= this.longest) {
        this.T = this.longest;
        this._stop();
        for (const k of this.c.values()) { this._hold(k); k.frame = k.frames - 1; }   // every clip on its last frame now
        this._show();
        this._sync("end");
        return;
      }
    }
    for (const k of this.c.values()) {
      const v = k.video;
      if (!v) { k.frame = this._frameAt(k); continue; }   // no video (a recording): the clock's frame
      if (this.isPlaying) {
        if (this._held(k) || v.ended) this._hold(k);      // (an ended video's play() would start it again)
        else {
          if (v.paused) { const p = v.play(); if (p && p.catch) p.catch(() => {}); }
          if (!v.seeking) {
            const d = loopDiff(v.currentTime, this.T, 0);           // + ahead of the clock, - behind
            if (now > k.fixAt && Math.abs(d) > RESYNC_S) {
              v.currentTime = this.T;
              k.fixAt = now + RESYNC_EVERY_MS;
            }
            const r = Math.abs(d) > NUDGE_S ? this.rate * (d < 0 ? NUDGE : 1 / NUDGE) : this.rate;
            if (v.playbackRate !== r) v.playbackRate = r;
          }
        }
      }
      if (this.noRVFC) k.frame = Math.max(0, Math.min(k.frames - 1, Math.floor(v.currentTime * k.fps + 1e-3)));
    }
    this._show();
  }

  /** A clip past its end: paused on its last frame. */
  _hold(k) {
    const v = k.video, last = (k.frames - 0.5) / k.fps;
    if (!v) return;
    if (!v.paused) v.pause();
    if (!v.ended && !v.seeking && Math.abs(v.currentTime - last) > 1 / k.fps) v.currentTime = last;
  }

  _stop() {
    this.isPlaying = false;
    for (const k of this.c.values()) if (k.video) { k.video.pause(); k.video.playbackRate = this.rate; }
  }

  play() {
    if (this.isPlaying || !this.c.size) return;
    if (this.atEnd) {                                   // a new pass: every clip from 0
      if (this.onNewPass) this.onNewPass();
      this.T = 0;
    }
    if (this.live) {                                   // back to now
      const k = this.c.get(this.liveKey);
      this.T = Math.max(0, k.frames / k.fps - LIVE_DELAY_S);
    }
    this.T0 = this.T; this.wall0 = this.lastPoll = performance.now();
    this.isPlaying = true;
    for (const k of this.c.values()) {
      if (!k.video) continue;
      if (this._held(k)) { this._hold(k); continue; }
      k.video.currentTime = this.T;
      k.fixAt = performance.now() + RESYNC_EVERY_MS;
      const p = k.video.play(); if (p && p.catch) p.catch(() => {});
    }
    this._sync("play");
  }

  pause() {
    if (!this.isPlaying) return;
    if (!this.live) this.T = Math.min(this.T0 + (performance.now() - this.wall0) / 1000 * this.rate, this.longest);
    this._stop();
    this._show();
    this._sync("pause");
  }

  toggle() { if (this.isPlaying) this.pause(); else this.play(); }

  /** The same time for every clip, clamped per clip (each shows its frame at t). */
  seekTime(t, notify = true) {
    if (this.live) return;                              // a recording while it records: now only
    this.T = Math.max(0, Math.min(this.longest, +t || 0));
    this.T0 = this.T; this.wall0 = performance.now();
    for (const k of this.c.values()) {
      const f = this._frameAt(k);
      if (!k.video) { k.frame = f; continue; }
      if (this.isPlaying && !this._held(k)) k.video.currentTime = this.T;
      else { if (!k.video.paused) k.video.pause(); k.video.currentTime = (f + 0.5) / k.fps; }
      k.frame = f;                                      // now (a paused video may present it late)
    }
    this._show();
    if (notify && this.onChange) this.onChange("seek");
  }

  /** Frame f of the focus camera. */
  seekFrame(f) { this.seekTime((Math.max(0, Math.round(f)) + 0.5) / this.fps); }

  step(df) { this.pause(); this.seekTime(this.T + df / this.fps); }

  setRate(r) {
    r = RATES.reduce((a, b) => (Math.abs(b - r) < Math.abs(a - r) ? b : a), 1);
    if (this.isPlaying) { this.T = this.T0 + (performance.now() - this.wall0) / 1000 * this.rate; this.T0 = this.T; this.wall0 = performance.now(); }
    this.rate = r;
    for (const k of this.c.values()) if (k.video) k.video.playbackRate = r;
    this.rateEl.value = String(r);
    this.onChange && this.onChange("rate");
  }

  /** The pass's number on the bar (live.js). */
  setPass(n, title = "") {
    this.passEl.textContent = n ? `Pass ${n}` : "";
    this.passEl.title = title;
  }

  _sync(what) {
    this.playEl.setAttribute("aria-pressed", String(this.playing));
    this.playEl.title = this.playing ? "Pause (Space)" : this.atEnd ? "A new pass (Space)" : "Play (Space)";
    this.onChange && this.onChange(what);
  }

  _show() {
    const step = Math.round(this.T * STEP);
    if (!this.scrubbing && this.scrub.value !== String(step)) this.scrub.value = String(step);
    this.timeEl.textContent = fmtTime(this.T);
    const k = this._f;
    this.frameEl.textContent = `${String(k.frame).padStart(String(k.frames).length, "0")} / ${k.frames}`;
  }

  _ui() {
    const el = this.el;
    el.innerHTML = `
      <button class="tbtn play" data-t="play" aria-pressed="false" title="Play (Space): every camera, one clock">
        <svg class="i-play" viewBox="0 0 12 12"><path d="M3 1.8v8.4L10 6z" fill="currentColor"/></svg>
        <svg class="i-pause" viewBox="0 0 12 12"><path d="M3 2h2v8H3zM7 2h2v8H7z" fill="currentColor"/></svg>
      </button>
      <span class="t-pass label" data-t="pass"></span>
      <span class="t-time num" data-t="time">00:00.00</span>
      <input class="scrub" data-t="scrub" type="range" min="0" max="0" step="1" value="0" aria-label="Time" title="Drag to go through the clip (← →: a frame, Shift: 10)">
      <span class="t-frame num" data-t="frame" title="The focus camera's frame"></span>
      <span class="t-dur num" data-t="dur" title="The longest clip: the pass ends here"></span>
      <label class="t-rate" title="The clock's speed: slower or faster than the footage"><span class="label">Rate</span>
        <select data-t="rate">${RATES.map((r) => `<option value="${r}" ${r === 1 ? "selected" : ""}>${r}×</option>`).join("")}</select></label>`;
    const q = (k) => el.querySelector(`[data-t="${k}"]`);
    this.playEl = q("play"); this.timeEl = q("time"); this.scrub = q("scrub"); this.frameEl = q("frame");
    this.rateEl = q("rate"); this.durEl = q("dur"); this.passEl = q("pass");
    this.playEl.onclick = () => this.toggle();
    this.rateEl.onchange = () => this.setRate(+this.rateEl.value);
    let wasPlaying = false;
    this.scrub.addEventListener("pointerdown", () => { this.scrubbing = true; wasPlaying = this.playing; this.pause(); });
    this.scrub.addEventListener("input", () => this.seekTime(+this.scrub.value / STEP));
    const end = () => { if (!this.scrubbing) return; this.scrubbing = false; if (wasPlaying && !this.atEnd) this.play(); };
    this.scrub.addEventListener("pointerup", end);
    this.scrub.addEventListener("change", end);
    let timer = 0;
    const show = () => {
      el.classList.remove("hidden");
      clearTimeout(timer);
      timer = setTimeout(() => { if (!el.matches(":hover") && !this.scrubbing && !this.pinned) el.classList.add("hidden"); }, HIDE_MS);
    };
    if (typeof window !== "undefined") window.addEventListener("pointermove", show, { passive: true });
    el.addEventListener("pointerleave", show);
    this.show = show;
    show();
  }
}
