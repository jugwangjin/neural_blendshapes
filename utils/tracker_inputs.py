"""Sparse MediaPipe 478 indices for split tracker MLP inputs."""

import torch

# Pose residual: stable head-frame anchors (2D normalized UV).
POSE_LMK_MP = [
    1,    # nose tip
    6,    # nose bridge / between eyes
    10,   # forehead
    33,   # left eye outer
    133,  # left eye inner
    152,  # chin
    263,  # right eye outer
    362,  # right eye inner
    61,   # left mouth corner
    291,  # right mouth corner
]

# Gaze: head-facing + iris (468–477).
GAZE_FACE_DIR_MP = [
    1,
    6,
    33,
    133,
    159,  # left eye lower
    263,
    362,
    386,  # right eye lower
    10,
    152,
]

GAZE_IRIS_MP = list(range(468, 478))


def _reshape_lmk(mp_landmarks_2d, B, device, dtype):
    if mp_landmarks_2d.ndim == 3:
        return mp_landmarks_2d.to(device=device, dtype=dtype)
    return mp_landmarks_2d.reshape(B, 478, 2).to(device=device, dtype=dtype)


def gather_mp_landmarks(mp_landmarks_2d, indices, device=None, dtype=None):
    """``indices`` → [B, len(indices)*2] flattened (x,y interleaved per point)."""
    if mp_landmarks_2d is None:
        raise ValueError("mp_landmarks_2d is required for sparse landmark inputs")
    B = mp_landmarks_2d.shape[0]
    device = device or mp_landmarks_2d.device
    dtype = dtype or mp_landmarks_2d.dtype
    lmk = _reshape_lmk(mp_landmarks_2d, B, device, dtype)
    idx = torch.tensor(indices, device=lmk.device, dtype=torch.long)
    pts = lmk[:, idx, :]
    return pts.reshape(B, -1)
