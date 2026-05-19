"""
ICT-FaceKit Multi-PIE 68 facial landmarks (vertex indices).

Source: USC-ICT/ICT-FaceKit README — «Multi-PIE 68 point facial landmarks» with
jawline substitution for contour points 0–7 (right) and 9–16 (left).

https://github.com/USC-ICT/ICT-FaceKit
"""

from __future__ import annotations

import numpy as np

# Official Multi-PIE 68 (contour uses brow/cheek anchors, not extended jawline)
MULTIPIE_68_OFFICIAL = [
    1225, 1888, 1052, 367, 1719, 1722, 2199, 1447, 966, 3661, 4390, 3927, 3924,
    2608, 3272, 4088, 3443, 268, 493, 1914, 2044, 1401, 3615, 4240, 4114, 2734,
    2509, 978, 4527, 4942, 4857, 1140, 2075, 1147, 4269, 3360, 1507, 1542, 1537,
    1528, 1518, 1511, 3742, 3751, 3756, 3721, 3725, 3732, 5708, 5695, 2081, 0,
    4275, 6200, 6213, 6346, 6461, 5518, 5957, 5841, 5702, 5711, 5533, 6216, 6207,
    6470, 5517, 5966,
]

RIGHT_JAWLINE_68 = [1278, 1272, 12, 1834, 243, 781, 2199, 1447]
LEFT_JAWLINE_68 = [3661, 4390, 3022, 2484, 4036, 2253, 3490, 3496]

# Recommended for FLAME/ICT alignment (README jawline substitutes 0–7, 9–16)
LANDMARK_INDICES_MULTIPIE_68_JAWLINE = (
    RIGHT_JAWLINE_68
    + [966]
    + LEFT_JAWLINE_68
    + MULTIPIE_68_OFFICIAL[17:]
)

LANDMARK_START_FLAME_PAIRING = 17

# iBUG / Multi-PIE 68 topology (for texture QA lines)
FACIAL_68_CONNECTIONS = [
    list(range(0, 17)),
    list(range(17, 22)),
    list(range(22, 27)),
    list(range(27, 31)),
    list(range(31, 36)),
    list(range(36, 42)),
    list(range(42, 48)),
    [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 48],
    [60, 61, 62, 63, 64, 65, 66, 67, 60],
]


def landmark_indices_for_asset(*, use_jawline=True):
    if use_jawline:
        return list(LANDMARK_INDICES_MULTIPIE_68_JAWLINE)
    return list(MULTIPIE_68_OFFICIAL)


def landmark_jawline_vertex_indices(landmark_indices, landmark_start=LANDMARK_START_FLAME_PAIRING):
    """Multi-PIE protocol indices ``0 .. landmark_start-1`` (jaw contour on ICT mesh)."""
    return np.asarray(landmark_indices, dtype=np.int64)[: int(landmark_start)]


def landmark_inner_vertex_indices(landmark_indices, landmark_start=LANDMARK_START_FLAME_PAIRING):
    """Protocol ``landmark_start .. 67`` — pairs with FLAME static embedding (51 inner pts)."""
    return np.asarray(landmark_indices, dtype=np.int64)[int(landmark_start) :]


def validate_landmark_indices(indices, n_verts, *, label="landmark_indices"):
    idx = np.asarray(indices, dtype=np.int64)
    if idx.shape[0] != 68:
        raise ValueError(f"{label}: expected 68 indices, got {len(idx)}")
    if idx.min() < 0 or idx.max() >= n_verts:
        raise ValueError(
            f"{label}: vertex index out of range [0, {n_verts}): min={idx.min()} max={idx.max()}"
        )
    return idx


def assert_matches_ict_facekit_readme(indices):
    """Raise if list diverges from repo canonical jawline-substituted 68."""
    ref = np.asarray(LANDMARK_INDICES_MULTIPIE_68_JAWLINE, dtype=np.int64)
    got = np.asarray(indices, dtype=np.int64)
    if got.shape != ref.shape or not np.all(got == ref):
        diff = np.where(got != ref)[0]
        raise ValueError(
            f"landmark_indices mismatch vs ICT-FaceKit README at positions {diff[:10].tolist()}..."
        )
