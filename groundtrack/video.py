"""Small OpenCV video helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    fps: float
    n_frames: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0


def video_info(path: str | Path, fps_override: float | None = None) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video {path}")
    fps = float(fps_override or cap.get(cv2.CAP_PROP_FPS) or 0)
    if not fps or fps > 1000:
        raise ValueError(f"{path}: video reports no usable frame rate; set `fps:` in the config.")
    info = VideoInfo(str(path), fps, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                     int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return info


def read_frame(path: str | Path, frame: int = 0) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not read frame {frame} from {path}")
    return img


def point_in_polygon(points: np.ndarray, polygon) -> np.ndarray:
    """Vectorised test of (N, 2) pixel points against a polygon [[x, y], ...]."""
    from matplotlib.path import Path as MplPath

    pts = np.asarray(points, float).reshape(-1, 2)
    if len(pts) == 0:
        return np.zeros(0, bool)
    return MplPath(np.asarray(polygon, float)).contains_points(pts, radius=1e-9)


class VideoSink:
    """Write BGR frames to an .mp4. Uses ffmpeg's H.264 encoder through a pipe when ffmpeg
    is installed (fast, small files that play everywhere), else OpenCV's MPEG-4 writer.
    keyint: a keyframe at least every keyint frames (x264's own default: 250); a video that is
    seeked (the Live view's) wants it short: a seek decodes from the keyframe before."""

    def __init__(self, path, fps: float, size: tuple[int, int], crf: int = 19, keyint: int | None = None):
        import shutil
        import subprocess

        self.path = str(path)
        w, h = size
        self.size = (w - w % 2, h - h % 2)          # H.264 / yuv420p needs even dimensions
        self.proc = None
        self.writer = None
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            import tempfile

            self._err = tempfile.TemporaryFile()        # ffmpeg's own messages, for errors
            self.proc = subprocess.Popen(
                [ffmpeg, "-y", "-loglevel", "error", "-hide_banner",
                 "-f", "rawvideo", "-pix_fmt", "bgr24",
                 "-s", f"{self.size[0]}x{self.size[1]}", "-r", f"{fps:.6f}", "-i", "-",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
                 *(["-g", str(int(keyint))] if keyint else []),
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", self.path],
                stdin=subprocess.PIPE, stderr=self._err)
        else:
            self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                          self.size)

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[1] != self.size[0] or frame.shape[0] != self.size[1]:
            frame = frame[: self.size[1], : self.size[0]]
        if self.proc is not None:
            try:
                self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            except OSError as e:          # ffmpeg stopped: say why, not just "Invalid argument"
                raise RuntimeError(f"ffmpeg stopped writing {self.path} "
                                   f"(exit {self.proc.poll()}): {self._ffmpeg_says()}") from e
        else:
            self.writer.write(frame)

    def _ffmpeg_says(self) -> str:
        try:
            self.proc.wait(timeout=5)
            self._err.seek(0)
            return self._err.read().decode("utf-8", "replace").strip()[-600:] or "no message"
        except Exception:  # noqa: BLE001
            return "no message"

    def release(self) -> None:
        if self.proc is not None:
            self.proc.stdin.close()
            if self.proc.wait() != 0:
                raise RuntimeError(f"ffmpeg failed on {self.path}: {self._ffmpeg_says()}")
        elif self.writer is not None:
            self.writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
