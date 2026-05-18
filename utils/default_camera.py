"""Load baked default camera (no Camera class)."""

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAMERA_NPZ = REPO_ROOT / "assets" / "default_camera.npz"
DEFAULT_CAMERA_TXT = REPO_ROOT / "assets" / "default_camera.txt"


def load_default_camera(path=None):
    path = Path(path or DEFAULT_CAMERA_NPZ)
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}
