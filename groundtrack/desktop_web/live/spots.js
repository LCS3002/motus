/* Live view, cameras in one scene (plans/live_controller/08 b, changed 28 Oct): the view opens on
   one run's camera (the start camera) and loads the site's other cameras as phones pick them
   (/api/live/site, groundtrack/live_site.py). Each sits at its package's origin, one shared clock,
   a tint of the violet each. Pure helpers (no DOM, no three.js): live.js builds the scene with
   them, the tests check them in node. */

/* The controller's violet (tokens.css --violet-line, #9b82d9) and three tints of its hue, one per
   camera in the scene: bluer, redder, more magenta. Light enough on black; never orange. */
export const TINTS = ["#9b82d9", "#8b8fe6", "#b787dc", "#c98bd2"];

/** "#rrggbb" -> [r, g, b] in 0..1. */
export function hexRGB(hex) {
  const v = parseInt(String(hex).replace("#", ""), 16);
  return [((v >> 16) & 255) / 255, ((v >> 8) & 255) / 255, (v & 255) / 255];
}

/** The hue of [r, g, b] (0..1), degrees. */
export function hue([r, g, b]) {
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b), d = mx - mn;
  if (d === 0) return 0;
  const h = mx === r ? ((g - b) / d) % 6 : mx === g ? (b - r) / d + 2 : (r - g) / d + 4;
  return (60 * h + 360) % 360;
}

/** The tint for a camera joining the scene: the first one no camera in it has (all taken: the
    least used, first first). used: the tints' indices in the scene. */
export function freeTint(used) {
  const n = TINTS.map((_, i) => used.filter((u) => u === i).length);
  return n.indexOf(Math.min(...n));
}

/* A package's offset in the scene, from scene.json origin {E, N, datum_z}: (E - E0, N - N0,
   datum_z - datum_z0), the scene's origin o0 being the start camera's (a package without an
   origin, or no o0: at the scene's). */
export function offsetFrom(o, o0) {
  if (!o || !o0) return [0, 0, 0];
  return [+o.E - +o0.E, +o.N - +o0.N, (+o.datum_z || 0) - (+o0.datum_z || 0)];
}

/** The camera a word of the URL means (players=, focus=): its name, its number, its clip or
    "<clip>/<run name>"; null when none. cams: /api/live/site cameras. */
export function findCamera(word, cams) {
  const w = String(word || "").trim().toLowerCase();
  if (!w) return null;
  return cams.find((c) => [c.name, c.id, c.clip, `${c.clip}/${c.run_name}`].some((x) => String(x).toLowerCase() === w)) || null;
}

/* The shared clock (08 b): every clip plays from one time T; past its end a clip holds its last
   frame (playing = false: clamped). (playing = true loops at the clip's own length: the first
   version, before passes.) */
export function clipTime(T, duration, playing) {
  if (!(duration > 0)) return 0;
  if (playing) return ((T % duration) + duration) % duration;
  return Math.max(0, Math.min(T, duration - 1e-3));
}

/** a - b on a loop of length d, in (-d/2, d/2]. */
export function loopDiff(a, b, d) {
  let x = a - b;
  if (d > 0) { x = ((x % d) + d) % d; if (x > d / 2) x -= d; }
  return x;
}

/* The players' panels: a column at the right of the stage (left of the settings panel when it is
   open), each at the clips' aspect (w / h), stacked from the top, shrunk to fit. CSS pixels, from
   the top left: [{x, y, w, h}]. */
export function panelRects(n, W, H, { top = 60, bottom = 56, right = 0, margin = 12, gap = 8, aspect = 16 / 9,
                                      maxW = 420, frac = 0.24 } = {}) {
  if (n <= 0) return [];
  let w = Math.min(maxW, Math.max(160, frac * W));
  const avail = H - top - bottom;
  let h = w / aspect;
  if (n * h + (n - 1) * gap > avail) { h = Math.max(40, (avail - (n - 1) * gap) / n); w = h * aspect; }
  const x = W - right - margin - w;
  return Array.from({ length: n }, (_, i) => ({ x: Math.round(x), y: Math.round(top + i * (h + gap)), w: Math.round(w), h: Math.round(h) }));
}
