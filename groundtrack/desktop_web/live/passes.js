/* Live view, passes (plans/live_controller/08 b, "Passes", 28 Oct): pure helpers (no DOM, no
   three.js). A pass is one play-through of the shared clock; per camera it keeps what the machine
   drew (the rules' verdict on the drawn points, rules.js Rules.out). The saved form (one .bin per
   camera, groundtrack/live_passes.py): column-wise little-endian, n = the package's tracks.bin
   records, 16 bytes a point: X float32 x 3n, part uint16 x n, kind uint8 x n, cause uint8 x n. */

export const BYTES_PER_POINT = 16;

/** A copy of a camera's drawing: {X, part, kind, cause, upTo} (upTo: the last frame drawn). */
export function snapshot(out, upTo) {
  return { X: Float32Array.from(out.X), part: Uint16Array.from(out.part), kind: Uint8Array.from(out.kind),
           cause: Uint8Array.from(out.cause), upTo };
}

/** A drawing -> the .bin's bytes. */
export function packPass(snap) {
  const n = snap.kind.length, buf = new ArrayBuffer(BYTES_PER_POINT * n);
  new Float32Array(buf, 0, 3 * n).set(snap.X);
  new Uint16Array(buf, 12 * n, n).set(snap.part);
  new Uint8Array(buf, 14 * n, n).set(snap.kind);
  new Uint8Array(buf, 15 * n, n).set(snap.cause);
  return buf;
}

/** The .bin's bytes -> a drawing (n points; throws when the size is not 16 n). */
export function unpackPass(buf, n, upTo = -1) {
  if (buf.byteLength !== BYTES_PER_POINT * n) throw new Error(`a pass file of ${buf.byteLength} bytes for ${n} points`);
  return { X: new Float32Array(buf.slice(0, 12 * n)), part: new Uint16Array(buf.slice(12 * n, 14 * n)),
           kind: new Uint8Array(buf.slice(14 * n, 15 * n)), cause: new Uint8Array(buf.slice(15 * n, 16 * n)), upTo };
}

/** Two drawings of one camera in one pass (it left and came back): into later, the earlier where
    the later has nothing yet (kind 255). Returns later. */
export function mergeSnaps(earlier, later) {
  const x = later, y = earlier, n = x.kind.length;
  for (let i = 0; i < n; i++) {
    if (x.kind[i] !== 255 || y.kind[i] === 255) continue;
    x.kind[i] = y.kind[i]; x.cause[i] = y.cause[i]; x.part[i] = y.part[i];
    x.X[3 * i] = y.X[3 * i]; x.X[3 * i + 1] = y.X[3 * i + 1]; x.X[3 * i + 2] = y.X[3 * i + 2];
  }
  x.upTo = Math.max(x.upTo, y.upTo);
  return x;
}

/** Whether a drawing has anything drawn. */
export const drawn = (snap) => !!snap && snap.upTo >= 0;

/** Bytes -> base64 (the save's JSON body). */
export function toBase64(buf) {
  const b = new Uint8Array(buf);
  let s = "";
  for (let i = 0; i < b.length; i += 0x8000) s += String.fromCharCode.apply(null, b.subarray(i, i + 0x8000));
  return btoa(s);
}

/** A local ISO time ("2026-10-28T19:14:02"), as the pass files keep it. */
export function localIso(d = new Date()) {
  const p = (v, k = 2) => String(v).padStart(k, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

/** A saved pass in the list: "28 Oct 19:14 · 3 cameras · 0:41" (+ " · stopped" when the clock
    did not reach the end). */
export function passLabel(p) {
  const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/.exec(p.ended || p.started || "");
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const when = m ? `${+m[3]} ${months[+m[2] - 1]} ${m[4]}:${m[5]}` : p.id;
  const n = (p.cameras || []).length, s = Math.round(+p.clock_s || 0);
  return `${when} · ${n} camera${n === 1 ? "" : "s"} · ${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}` +
    (p.complete ? "" : " · stopped");
}
