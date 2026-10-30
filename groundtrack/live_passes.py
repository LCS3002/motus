"""Live view: passes (plans/live_controller/08 b, "Passes", 28 Oct).

A pass is one play-through of the site's shared clock, 0 to the end of the longest clip: what
every camera in the scene drew in it. The view saves every finished pass by itself (and the
current one when the site is switched), under the start camera's run:

    <run>/live/passes/<id>/pass.json      the manifest (below)
    <run>/live/passes/<id>/<camera>.bin   per camera: what it drew, one record per point

<id> is the local time the pass was first saved, "YYYYMMDD-HHMMSS" ("-2", "-3" when taken).
Nothing is written anywhere else; the view's Clear and Undo never delete files, only the list's
Delete does (this pass's folder: its pass.json and .bin files).

<camera>.bin is column-wise, little-endian, n = the camera package's tracks.bin records (row i is
record i: its track id, frame and time are there), 16 bytes a point:
    X      float32 x 3n   where the machine drew the point, metres, in the package's own frame
                          (scene.json origin; + cameras[k].offset: in the pass's frame, the start
                          camera's package); NaN: not drawn (not reached, or the clip was shorter)
    part   uint16 x n     the identity piece the rules gave it (a new identity breaks the line)
    kind   uint8 x n      0 solid, 1 predicted, 2 stitched, 3 interpolated, 4 dropped, 255 not yet
    cause  uint8 x n      0 none, 1 the footage's own guess, 2 caused in Live
(rules.js Rules.out after the pass: the machine's own verdict on what was drawn, 08 c.)

pass.json: {"format": "motus-live-pass", "version": 2 (1: no recording), "id", "run" (the run
holding it), "site", "start_spot", "started", "ended" (local ISO times), "saved" (the server's),
"complete" (the clock reached the end), "clock_s" (where it stopped), "duration_s" (the longest
clip), "origin" (the pass frame: the start package's scene.json origin), "layout" (the record
layout above), "cameras": [{"name", "id", "clip", "run_name", "run" (project-relative where
possible), "origin", "offset" [dE, dN, dz], "fps", "frames", "points" (n), "performed_to" (the
last frame drawn), "file", "tint", "players"}], "recording" (version 2, below)}.
Readers ignore keys they don't know.

The recording (job 05, 29 Oct: a session is a pass; every pass records itself, the mouse too).
Everything needed to replay, render, export and score the pass, written by the view each frame
something changed (a pose, the clock, a camera's frame, the machine's counts, an event); between
two rows everything held. "recording": {"format": "motus-live-rec", "version": 1, "started"
(local ISO), "clock": {"file": "clock.rec", "rows"}, "events": {"file": "events.jsonl", "n"},
"cameras": [{"name", "id", "clip", "run", "origin", "offset", "fps", "frames", "points", "tint",
"file": "<camera id>.rec", "rows", "players"}] (every camera that took part, drawn or not),
"clock_fields", "camera_fields" (the numpy dtypes below as [name, type, shape, offset])}.
    clock.rec     CLOCK_DTYPE rows, 24 bytes: one per recorded frame k (k = the row number)
        t f8      s since the pass began (the view's clock: the performance's time axis)
        T f8      the shared clip clock, s (play / pause / seek show here)
        rate f4   the playback rate
        flags u1  1 playing, 2 at the end
    <camera>.rec  CAM_DTYPE rows, 232 bytes, one per recorded frame this camera changed in:
        R_wc f8 3x3  the virtual camera, world_from_camera (OpenCV axes), the package's frame,
                     in the space's own frame (as mapped then: replays never re-map)
        C f8 3       its centre (m)
        S f8 3x3     the space's turn about the start camera's centre C0 (08 a): a point is drawn
                     at C0 + S (G(X) - C0), G the ground-mode projection with R_wc, C
        k u4         the clock row
        frame i4     the clip frame the camera drew (its video's)
        q f4 4       the raw pose of the driving source, world_from_camera [w, x, y, z] (NaN: none)
        p f4 3       its position (NaN: none)
        gain f4, scale f4
        placed, guessed, dropped, lost, new u2   the machine's counts at that frame, as shown then
                     (placed: solid; guessed: white dashes; dropped: nothing drawn; lost: points
                     the projection lost, behind / out of view / past the horizon; new: identity
                     pieces that begin at that frame, a track's first drawn point or a new identity)
        player u1    the phone holding the camera (1-4; 0: none, the laptop's mouse)
        src u1       SOURCES (0 none, 1 web, 2 mouse, 3 replay, 4 bench, 5 webcam, 6 freed,
                     7 arkit, 8 url; 255 another)
        status u1    0 ok, 1 limited, 2 lost
        flags u1     1 changed (the camera or the space changed: the frame was projected again),
                     2 the clutch held, 4 levelling, 8 the space turned, 16 performed trails (0: the
                     whole clip, nothing performed), 32 the camera (re)joined here, 64 Smooth on
                     (R_wc, C are the eased camera; 0, as in older passes: it followed the pose)
        seq u4       the raw pose's seq
    events.jsonl  one JSON object a line: {"k" (the clock row it came before), "t", "T", "type",
                  "player", "camera", "value"}; types EVENTS: pass_start {site, stack, cameras},
                  pass_end {reason: end / new / undo / clear / site / replay / close}, end (the
                  clock reached the end), play, pause, seek (T), rate, space_press / space_release
                  (who), level, zero (src), gain, scale, trails, stack, site (the new site),
                  camera_join {join_frame} / camera_leave, phone_join {spot} / phone_leave /
                  phone_spot, undo, clear, smooth (the camera's Smooth: true / false).
A replay (live/replay.js) applies the camera rows in k order, each camera starting at its
camera_join's join_frame (its drawing reset there), and redraws exactly what the pass drew.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from pathlib import Path

import numpy as np

FORMAT = "motus-live-pass"
VERSION = 2
BYTES_PER_POINT = 16
ID_RE = re.compile(r"^\d{8}-\d{6}(-\d{1,3})?$")
FILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\.bin$")
LAYOUT = ("column-wise little-endian, n = tracks.bin records: X float32 x 3n (m, the package's frame; "
          "NaN not drawn), part uint16 x n, kind uint8 x n (0 solid, 1 predicted, 2 stitched, "
          "3 interpolated, 4 dropped, 255 not yet), cause uint8 x n (0 none, 1 footage, 2 live)")
MAX_CAMERAS = 64

REC_FORMAT = "motus-live-rec"
REC_VERSION = 1
REC_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\.rec$")
EVENTS_FILE = "events.jsonl"
CLOCK_DTYPE = np.dtype({"names": ["t", "T", "rate", "flags"], "formats": ["<f8", "<f8", "<f4", "u1"],
                        "offsets": [0, 8, 16, 20], "itemsize": 24})
CAM_DTYPE = np.dtype({
    "names": ["R_wc", "C", "S", "k", "frame", "q", "p", "gain", "scale", "placed", "guessed", "dropped",
              "lost", "new", "player", "src", "status", "flags", "seq"],
    "formats": [("<f8", (3, 3)), ("<f8", 3), ("<f8", (3, 3)), "<u4", "<i4", ("<f4", 4), ("<f4", 3), "<f4",
                "<f4", "<u2", "<u2", "<u2", "<u2", "<u2", "u1", "u1", "u1", "u1", "<u4"],
    "offsets": [0, 72, 96, 168, 172, 176, 192, 204, 208, 212, 214, 216, 218, 220, 224, 225, 226, 227, 228],
    "itemsize": 232})
SOURCES = ("none", "web", "mouse", "replay", "bench", "webcam", "freed", "arkit", "url")
STATUSES = ("ok", "limited", "lost")
CLOCK_PLAYING, CLOCK_AT_END = 1, 2
CHANGED, HELD, LEVELLING, TILTED, PERFORMED, JOINED, SMOOTH = 1, 2, 4, 8, 16, 32, 64
EVENTS = ("pass_start", "pass_end", "end", "play", "pause", "seek", "rate", "space_press", "space_release",
          "level", "zero", "gain", "scale", "trails", "stack", "site", "camera_join", "camera_leave",
          "phone_join", "phone_leave", "phone_spot", "undo", "clear", "smooth")
MAX_REC_BYTES = 1 << 30


def dtype_fields(dt: np.dtype) -> list:
    """A dtype as [name, type, shape, offset] (pass.json: readable without this module)."""
    out = []
    for name in dt.names:
        sub, off = dt.fields[name][:2]
        base = sub.base if sub.shape else sub
        out.append([name, base.str, list(sub.shape), off])
    return out


def passes_dir(run: Path) -> Path:
    return Path(run) / "live" / "passes"


def _folder(run: Path, pass_id: str) -> Path:
    if not isinstance(pass_id, str) or not ID_RE.match(pass_id):
        raise ValueError("not a pass id")
    return passes_dir(run) / pass_id


def new_id(run: Path, now: float | None = None) -> str:
    base = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
    pid, k = base, 2
    while (passes_dir(run) / pid).exists():
        pid, k = f"{base}-{k}", k + 1
    return pid


def _write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _rel(p: str, root: Path | None) -> str:
    try:
        return Path(p).resolve().relative_to(root).as_posix() if root else str(p)
    except (ValueError, OSError, TypeError):
        return str(p)


def _want(ok: bool, what: str) -> None:
    if not ok:
        raise ValueError(what)


def parse_files(body: bytes) -> tuple[dict, dict]:
    """A request with files (live.js postFiles): [u32 little-endian header length][the JSON header,
    "files": [{"file", "bytes"}]][the files one after another] -> (header, {file: bytes})."""
    body = memoryview(body)
    if len(body) < 4:
        raise ValueError("an empty request")
    hl = int.from_bytes(body[:4], "little")
    if hl > len(body) - 4:
        raise ValueError("a header longer than the request")
    head = json.loads(bytes(body[4:4 + hl]))
    _want(isinstance(head, dict), "the header: an object")
    files, pos = {}, 4 + hl
    for f in head.pop("files", None) or []:
        name, n = str(f.get("file", "")), int(f.get("bytes", -1))
        if n < 0 or pos + n > len(body) or name in files:
            raise ValueError(f"{name or 'a file'}: not in the request")
        files[name] = bytes(body[pos:pos + n])
        pos += n
    if pos != len(body):
        raise ValueError(f"{len(body) - pos} bytes after the files")
    return head, files


def save_pass(run: Path, manifest: dict, data: dict, pass_id: str | None = None,
              root: Path | None = None) -> dict:
    """Saves a pass under <run>/live/passes/ (a new id, or pass_id again: the same pass finished
    once more). data: {file: bytes or base64 text}, one per camera of the manifest. root: paths in
    the manifest are written relative to it. Returns the pass's summary."""
    run = Path(run)
    if not (run / "live" / "scene.json").is_file():
        raise FileNotFoundError(f"no live package in {run}")
    cams = manifest.get("cameras") if isinstance(manifest, dict) else None
    if not isinstance(cams, list) or not cams or len(cams) > MAX_CAMERAS:
        raise ValueError("a pass needs its cameras")
    blobs: dict[str, bytes] = {}

    def blob(f):
        b = data.get(f) if isinstance(data, dict) else None
        if isinstance(b, str):
            b = base64.b64decode(b, validate=True)
        _want(isinstance(b, (bytes, bytearray, memoryview)), f"{f}: missing")
        return bytes(b)

    for c in cams:
        f, n = c.get("file"), c.get("points")
        if not isinstance(f, str) or not FILE_RE.match(f) or f in blobs:
            raise ValueError(f"bad camera file {f!r}")
        if not isinstance(n, int) or n < 0:
            raise ValueError(f"{f}: points")
        b = blob(f)
        if len(b) != BYTES_PER_POINT * n:
            raise ValueError(f"{f}: {BYTES_PER_POINT} bytes a point")
        blobs[f] = b
    rec = manifest.get("recording")
    if rec is not None:
        rec = _check_recording(rec, blob, blobs)
    if set(data) - set(blobs):
        raise ValueError("data for a file the pass doesn't list")
    if pass_id:
        folder = _folder(run, pass_id)
        if not (folder / "pass.json").is_file():
            raise FileNotFoundError(f"no pass {pass_id}")
    else:
        pass_id = new_id(run)
        folder = _folder(run, pass_id)
    folder.mkdir(parents=True, exist_ok=True)
    for f, b in blobs.items():
        _write(folder / f, b)
    for old in folder.iterdir():                         # a camera (or a recording) no longer in it
        if old.is_file() and old.name not in blobs and _own(old.name) and old.name != "pass.json":
            old.unlink()
    m = dict(manifest)
    rel = lambda c: {**c, "run": _rel(c["run"], root) if c.get("run") else c.get("run")}
    m.update(format=FORMAT, version=VERSION if rec else 1, id=pass_id, run=_rel(str(run), root),
             saved=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()), layout=LAYOUT,
             cameras=[rel(c) for c in cams])
    if rec:
        m["recording"] = {**rec, "cameras": [rel(c) for c in rec["cameras"]]}
    _write(folder / "pass.json", json.dumps(m, indent=1).encode("utf-8"))
    return summary(m, folder, run)


def _own(name: str) -> bool:
    """The files a pass writes (and their .tmp while writing)."""
    name = name.removesuffix(".tmp")
    return bool(FILE_RE.match(name) or REC_RE.match(name)) or name in ("pass.json", EVENTS_FILE)


def _check_recording(rec: dict, blob, blobs: dict) -> dict:
    """The manifest's "recording" against its files (into blobs); returns it with the format's
    keys filled in."""
    _want(isinstance(rec, dict), "recording: an object")

    def take(f, size=None):
        if not isinstance(f, str) or f in blobs or not (REC_RE.match(f) or f == EVENTS_FILE):
            raise ValueError(f"recording: bad file {f!r}")
        b = blob(f)
        if size is not None and len(b) != size:
            raise ValueError(f"{f}: {len(b)} bytes, not {size}")
        if len(b) > MAX_REC_BYTES:
            raise ValueError(f"{f}: too large")
        blobs[f] = b
        return b

    clock = rec.get("clock") or {}
    rows = clock.get("rows")
    if not isinstance(rows, int) or rows < 0:
        raise ValueError("recording: clock rows")
    take(clock.get("file"), rows * CLOCK_DTYPE.itemsize)
    cams = rec.get("cameras")
    if not isinstance(cams, list) or len(cams) > MAX_CAMERAS:
        raise ValueError("recording: cameras")
    for c in cams:
        n = c.get("rows") if isinstance(c, dict) else None
        if not isinstance(n, int) or n < 0:
            raise ValueError("recording: camera rows")
        take(c.get("file"), n * CAM_DTYPE.itemsize)
    ev = rec.get("events") or {}
    lines = take(ev.get("file")).decode("utf-8").splitlines()
    for ln in lines:
        _want(isinstance(json.loads(ln), dict), "events: one object a line")
    if ev.get("n") != len(lines):
        raise ValueError("events: n")
    return {**rec, "format": REC_FORMAT, "version": REC_VERSION,
            "clock_fields": dtype_fields(CLOCK_DTYPE), "camera_fields": dtype_fields(CAM_DTYPE)}


def summary(m: dict, folder: Path, run: Path) -> dict:
    size = sum(p.stat().st_size for p in folder.iterdir() if p.is_file())
    rec = m.get("recording") or {}
    return {"id": m.get("id"), "run": str(run), "site": m.get("site"), "started": m.get("started"),
            "ended": m.get("ended"), "complete": bool(m.get("complete")), "clock_s": m.get("clock_s"),
            "duration_s": m.get("duration_s"), "start_spot": m.get("start_spot"),
            "cameras": [c.get("name") for c in m.get("cameras") or []], "bytes": size,
            "recorded": bool(rec), "rec_s": rec.get("seconds"),
            "players": sorted({p for c in rec.get("cameras") or [] for p in c.get("players") or []})}


def read_pass(run: Path, pass_id: str) -> dict:
    f = _folder(Path(run), pass_id) / "pass.json"
    if not f.is_file():
        raise FileNotFoundError(f"no pass {pass_id}")
    m = json.loads(f.read_text(encoding="utf-8"))
    if m.get("format") != FORMAT:
        raise ValueError(f"{pass_id}: not a pass")
    if int(m.get("version") or 1) > VERSION:
        raise ValueError(f"{pass_id}: a newer pass format ({m.get('version')})")
    return m


def pass_files(m: dict) -> set:
    """The files a pass's manifest lists: the drawings and the recording."""
    out = {c.get("file") for c in m.get("cameras") or []}
    rec = m.get("recording") or {}
    if rec:
        out |= {(rec.get("clock") or {}).get("file"), (rec.get("events") or {}).get("file")}
        out |= {c.get("file") for c in rec.get("cameras") or []}
    return {f for f in out if isinstance(f, str) and _own(f) and f != "pass.json" and not f.endswith(".tmp")}


def pass_file(run: Path, pass_id: str, name: str) -> Path:
    """One file of a pass: a camera's .bin, or its recording's (only what the manifest lists)."""
    m = read_pass(run, pass_id)
    if name not in pass_files(m):
        raise FileNotFoundError(name)
    p = _folder(Path(run), pass_id) / name
    if not p.is_file():
        raise FileNotFoundError(name)
    return p


def read_recording(run: Path, pass_id: str) -> dict | None:
    """A pass's recording: {"meta" (pass.json's "recording"), "clock" (CLOCK_DTYPE rows),
    "cameras": {name: CAM_DTYPE rows}, "events": [dict]}; None for a pass without one (version 1)."""
    m = read_pass(run, pass_id)
    rec = m.get("recording")
    if not rec:
        return None
    folder = _folder(Path(run), pass_id)
    clock = np.fromfile(folder / rec["clock"]["file"], CLOCK_DTYPE)
    cams = {c["name"]: np.fromfile(folder / c["file"], CAM_DTYPE) for c in rec.get("cameras") or []}
    ev = (folder / rec["events"]["file"]).read_text(encoding="utf-8").splitlines()
    return {"meta": rec, "manifest": m, "clock": clock, "cameras": cams, "events": [json.loads(e) for e in ev]}


def delete_pass(run: Path, pass_id: str) -> None:
    """Deletes a pass: its pass.json, drawings and recording, then its folder when empty (its
    exports, in pass_<id>/, stay: they may be open in Rhino or Houdini)."""
    folder = _folder(Path(run), pass_id)
    if not (folder / "pass.json").is_file():
        raise FileNotFoundError(f"no pass {pass_id}")
    for p in folder.iterdir():
        if p.is_file() and _own(p.name):
            p.unlink()
    try:
        folder.rmdir()
    except OSError:
        pass                                              # something else in it (the exports): kept


def list_passes(runs: list[Path]) -> list[dict]:
    """The passes saved under these runs, newest first."""
    out, seen = [], set()
    for run in runs:
        d = passes_dir(Path(run))
        if not d.is_dir() or d.resolve() in seen:
            continue
        seen.add(d.resolve())
        for f in d.glob("*/pass.json"):
            if not ID_RE.match(f.parent.name):
                continue
            try:
                m = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if m.get("format") == FORMAT:
                out.append(summary(m, f.parent, Path(run)))
    return sorted(out, key=lambda s: (str(s.get("ended") or ""), str(s["id"])), reverse=True)
