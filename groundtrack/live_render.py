"""Live view: a saved pass rendered into a video (plans/live_controller/05, "Render").

The view replays the pass offline at a fixed rate (live.js renderPass): output frame j shows the
recording at j / fps s. Each frame comes here as a PNG, in order or not, and is kept under
<start run>/live/renders/.frames-<pass id>-<token>/%06d.png; once every frame is in, ffmpeg encodes
them (H.264, yuv420p, constant rate: no frame dropped or doubled) into
<start run>/live/renders/<pass id>_<view>.mp4, ffprobe counts the frames, and the PNGs are deleted
unless asked to keep them. Nothing leaves the machine.
"""

from __future__ import annotations

import json
import re
import secrets
import shutil
import subprocess
import threading
import time
from pathlib import Path

RENDERS = "renders"
PNG = b"\x89PNG\r\n\x1a\n"
MAX_FRAME_BYTES = 40 * 1024 * 1024
MAX_FRAMES = 30 * 60 * 30                           # 30 min at 30 fps
STALE_S = 6 * 3600                                   # a frames folder this old, no render using it: removed
VIEW_RE = re.compile(r"^[a-z0-9_-]{1,24}$")
ID_RE = re.compile(r"^[0-9]{8}-[0-9]{6}(-[0-9]+)?$")


def renders_dir(run: Path) -> Path:
    return Path(run) / "live" / RENDERS


def probe_frames(path: Path) -> dict:
    """ffprobe: the video's frame count (decoded), rate, size, length and codec."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames,r_frame_rate,width,height,codec_name,pix_fmt:format=duration",
         "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    st = (d.get("streams") or [{}])[0]
    num, _, den = str(st.get("r_frame_rate", "0/1")).partition("/")
    return {"frames": int(st.get("nb_read_frames") or 0), "fps": float(num) / float(den or 1),
            "width": int(st.get("width") or 0), "height": int(st.get("height") or 0),
            "codec": st.get("codec_name"), "pix_fmt": st.get("pix_fmt"),
            "seconds": float((d.get("format") or {}).get("duration") or 0)}


def encode(frames_dir: Path, fps: int, out: Path) -> None:
    """%06d.png at fps -> H.264 MP4 (every frame once, in order)."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is not on PATH (needed to render a pass)")
    tmp = out.with_name(out.stem + ".part.mp4")
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(fps), "-start_number", "0",
         "-i", str(frames_dir / "%06d.png"), "-c:v", "libx264", "-preset", "medium", "-crf", "16",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)],
        capture_output=True, text=True, check=False)
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg: {r.stderr.strip()[-400:]}")
    tmp.replace(out)


class Renders:
    """The renders in progress (one token each)."""

    def __init__(self):
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def begin(self, run: Path, pass_id: str, fps: int, width: int, height: int, frames: int,
              view: str = "orbit") -> dict:
        from .live_passes import read_pass

        read_pass(run, pass_id)                     # a saved pass of this run (raises otherwise)
        if not ID_RE.match(pass_id):
            raise ValueError(f"not a pass id: {pass_id!r}")
        fps, width, height, frames = int(fps), int(width), int(height), int(frames)
        if not (1 <= fps <= 120 and 16 <= width <= 7680 and 16 <= height <= 4320 and 1 <= frames <= MAX_FRAMES):
            raise ValueError(f"fps 1-120, size 16-7680 x 16-4320, frames 1-{MAX_FRAMES}")
        if width % 2 or height % 2:
            raise ValueError("width and height must be even (yuv420p)")
        view = str(view or "orbit")
        if not VIEW_RE.match(view):
            raise ValueError(f"not a view name: {view!r}")
        token = secrets.token_hex(8)
        with self._lock:
            mine = {j["dir"] for j in self._jobs.values()}
        for old in renders_dir(run).glob(".frames-*"):       # left by a render the app never finished
            if old.is_dir() and old not in mine and time.time() - old.stat().st_mtime > STALE_S:
                shutil.rmtree(old, ignore_errors=True)
        d = renders_dir(run) / f".frames-{pass_id}-{token}"
        d.mkdir(parents=True, exist_ok=False)
        with self._lock:
            self._jobs[token] = {"run": Path(run), "id": pass_id, "fps": fps, "width": width, "height": height,
                                 "frames": frames, "view": view, "dir": d, "got": set()}
        return {"token": token, "frames": frames, "dir": str(d)}

    def _job(self, token: str) -> dict:
        with self._lock:
            j = self._jobs.get(str(token))
        if j is None:
            raise KeyError("no such render (finished, cancelled or the app restarted)")
        return j

    def frame(self, token: str, index: int, data: bytes) -> dict:
        j = self._job(token)
        index = int(index)
        if not 0 <= index < j["frames"]:
            raise ValueError(f"frame {index} outside 0-{j['frames'] - 1}")
        if len(data) > MAX_FRAME_BYTES or not data.startswith(PNG):
            raise ValueError("a frame is one PNG")
        w, h = int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
        if (w, h) != (j["width"], j["height"]):
            raise ValueError(f"frame {index} is {w} x {h}, not {j['width']} x {j['height']}")
        p = j["dir"] / f"{index:06d}.png"
        tmp = p.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(p)
        with self._lock:
            j["got"].add(index)
            n = len(j["got"])
        return {"ok": True, "got": n, "frames": j["frames"]}

    def finish(self, token: str, keep: bool = False) -> dict:
        j = self._job(token)
        missing = sorted(set(range(j["frames"])) - j["got"])
        if missing:
            raise ValueError(f"{len(missing)} frames missing (first {missing[0]})")
        out = renders_dir(j["run"]) / f"{j['id']}_{j['view']}.mp4"
        encode(j["dir"], j["fps"], out)
        info = probe_frames(out)
        if info["frames"] != j["frames"]:
            raise RuntimeError(f"{out.name}: {info['frames']} frames, {j['frames']} rendered")
        with self._lock:
            self._jobs.pop(token, None)
        if not keep:
            shutil.rmtree(j["dir"], ignore_errors=True)
        return {"ok": True, "path": str(out), "frames": info["frames"], "fps": info["fps"],
                "seconds": round(info["frames"] / j["fps"], 3), "width": info["width"], "height": info["height"],
                "codec": info["codec"], "frames_dir": str(j["dir"]) if keep else None}

    def cancel(self, token: str) -> dict:
        with self._lock:
            j = self._jobs.pop(str(token), None)
        if j is not None:
            shutil.rmtree(j["dir"], ignore_errors=True)
        return {"ok": True}

    def cancel_all(self) -> None:
        for t in list(self._jobs):
            self.cancel(t)
