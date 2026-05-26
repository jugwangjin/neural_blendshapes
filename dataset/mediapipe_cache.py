"""Load precomputed MediaPipe + face_alignment frame cache."""

from pathlib import Path

import numpy as np
import torch


def load_frame_npz(path):
    d = np.load(path, allow_pickle=True)
    if "valid" in d and not bool(d["valid"]):
        raise ValueError(f"invalid frame cache: {path}")
    return {
        "mp_blendshape": torch.tensor(d["mp_blendshape"], dtype=torch.float32),
        "mp_landmarks_2d": torch.tensor(d["mp_landmarks_2d"], dtype=torch.float32),
        "mp_landmarks_3d": torch.tensor(d.get("mp_landmarks_3d", np.zeros((478, 3))), dtype=torch.float32),
        "mp_valid": torch.tensor(d.get("mp_valid", np.ones(478)), dtype=torch.float32),
        "mp_pose_raw": torch.tensor(d.get("mp_pose_raw", np.zeros(6)), dtype=torch.float32),
        "mp_transform_matrix": torch.tensor(
            d.get("mp_transform_matrix", np.eye(4, dtype=np.float32)),
            dtype=torch.float32,
        ),
        "pose_feat": torch.tensor(d.get("pose_feat", np.zeros(11)), dtype=torch.float32),
        "landmark_fa": torch.tensor(d.get("landmark_fa", np.zeros((68, 4))), dtype=torch.float32),
    }


def default_frame_dict(device, image_size=512):
    """Synthetic frame for smoke tests when cache is missing."""
    return {
        "image": torch.rand(3, image_size, image_size, device=device),
        "mask": torch.ones(1, image_size, image_size, device=device),
        "mp_blendshape": torch.rand(52, device=device),
        "mp_landmarks_2d": torch.rand(478, 2, device=device),
        "mp_landmarks_3d": torch.zeros(478, 3, device=device),
        "mp_valid": torch.ones(478, device=device),
        "mp_pose_raw": torch.zeros(6, device=device),
        "frame_idx": 0,
    }


def list_cached_frames(cache_dir):
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return []
    out = []
    for p in sorted(cache_dir.glob("**/*.npz")):
        d = np.load(p, allow_pickle=True)
        if "valid" in d and not bool(d["valid"]):
            continue
        out.append(p)
    return out
