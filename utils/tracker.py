"""Tracker MLP: sparse MP landmark inputs, 2D canonicalization, blendshape gating."""

import torch

# Pose residual trunk — stable head-frame anchors (normalized MP UV).
POSE_LMK_MP = [
    1,
    6,
    10,
    33,
    133,
    152,
    263,
    362,
    61,
    291,
]


def smoothstep(x, lo, hi):
    t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _reshape_lmk(mp_landmarks_2d, B, device, dtype):
    if mp_landmarks_2d.ndim == 3:
        return mp_landmarks_2d.to(device=device, dtype=dtype)
    return mp_landmarks_2d.reshape(B, 478, 2).to(device=device, dtype=dtype)


_indices_cache = {}

def get_cached_indices_tensor(indices, device):
    key = (tuple(indices), str(device))
    if key not in _indices_cache:
        _indices_cache[key] = torch.tensor(indices, device=device, dtype=torch.long)
    return _indices_cache[key]


def gather_mp_landmarks(mp_landmarks_2d, indices, device=None, dtype=None):
    """``indices`` → [B, len(indices)*2] flattened (x,y per point)."""
    if mp_landmarks_2d is None:
        raise ValueError("mp_landmarks_2d is required for sparse landmark inputs")
    B = mp_landmarks_2d.shape[0]
    device = device or mp_landmarks_2d.device
    dtype = dtype or mp_landmarks_2d.dtype
    lmk = _reshape_lmk(mp_landmarks_2d, B, device, dtype)
    idx = get_cached_indices_tensor(indices, lmk.device)
    pts = lmk[:, idx, :]
    return pts.reshape(B, -1)


def landmarks_3d_to_camera_xy(lmk3d, world_to_cam):
    """lmk3d [B, N, 3] → camera-plane xy (depth dropped for expr trunk)."""
    cam = torch.matmul(lmk3d, world_to_cam.transpose(-1, -2))
    return cam[..., :2]


def landmarks_2d_canonical(mp_landmarks_2d, anchor_mp=None):
    """Normalized MP UV → nose-centered, inter-eye scale invariant [B, 478, 2]."""
    if mp_landmarks_2d.ndim == 2:
        B = mp_landmarks_2d.shape[0]
        lmk = mp_landmarks_2d.reshape(B, 478, 2)
    else:
        lmk = mp_landmarks_2d
    if anchor_mp is None:
        anchor_mp = (1, 6, 33, 263)
    idx = get_cached_indices_tensor(anchor_mp, lmk.device)
    center = lmk[:, idx, :].mean(dim=1, keepdim=True)
    le = lmk[:, 33:34, :]
    re = lmk[:, 263:264, :]
    scale = (re - le).norm(dim=-1, keepdim=True).clamp(min=1e-4)
    return (lmk - center) / scale.unsqueeze(-1)
