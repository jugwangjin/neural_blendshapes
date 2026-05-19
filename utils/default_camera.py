"""Load / bake default camera (no Camera class)."""

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMERA_NPZ = REPO_ROOT / "assets" / "default_camera.npz"
DEFAULT_CAMERA_TXT = REPO_ROOT / "assets" / "default_camera.txt"


def load_default_camera(path=None):
    path = Path(path or DEFAULT_CAMERA_NPZ)
    if not path.is_file():
        return None
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def save_default_camera(R, t, K, path=None):
    """Write ``K_mean``, ``R_mean``, ``t_mean`` for ``FixedCamera.from_default_npz``."""
    path = Path(path or DEFAULT_CAMERA_NPZ)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(path),
        K_mean=np.asarray(K, dtype=np.float64),
        R_mean=np.asarray(R, dtype=np.float64),
        t_mean=np.asarray(t, dtype=np.float64).reshape(3),
    )
    return path
