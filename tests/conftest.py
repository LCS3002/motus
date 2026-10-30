"""Shared fixtures: a synthetic oblique camera with an exactly known ground homography."""

from __future__ import annotations

import numpy as np
import pytest

from groundtrack.demo import ORIGIN, SyntheticCamera  # noqa: F401  (re-exported for tests)


@pytest.fixture
def key_frames():
    """path -> the numbers of a video's keyframes (ffprobe; skipped without it)."""
    import shutil
    import subprocess

    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe is not installed")

    def frames(path):
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "frame=key_frame", "-of", "csv=p=0", str(path)],
                             capture_output=True, text=True, check=True).stdout
        return [i for i, v in enumerate(out.split()) if v.startswith("1")]

    return frames


@pytest.fixture
def cam():
    """Camera 12 m up at local (0, -20), pitched 30 deg down, looking north, f = 1400 px."""
    return SyntheticCamera()


@pytest.fixture
def ground_points():
    """8 well spread calibration points on the ground, 20-75 m in front of the camera."""
    local = np.array([[-8, 0], [8, 0], [-12, 15], [12, 15], [-15, 35], [15, 35], [-5, 55],
                      [6, 50]], float)
    return local + ORIGIN


@pytest.fixture
def cfg():
    from groundtrack.config import default_config

    return default_config()
