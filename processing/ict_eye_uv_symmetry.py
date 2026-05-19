"""
Check whether left/right eyeball UV charts share orientation or need u-mirror on the right eye.

Used at npy bake time; result stored as eye_uv_mirror_right_u for EyeTextureGaussians.
"""

from __future__ import annotations

import numpy as np


def _valid_vertex_uv(vertex_uvs, indices):
    idx = np.asarray(indices, dtype=np.int64)
    uv = np.asarray(vertex_uvs, dtype=np.float64)[idx]
    valid = uv[:, 0] >= 0
    return uv[valid]


def _normalize_uv_box(uv):
    mn = uv.min(axis=0)
    mx = uv.max(axis=0)
    span = mx - mn
    span[span < 1e-8] = 1.0
    return (uv - mn) / span


def analyze_eye_uv_symmetry(vertex_uvs, left_eyeball_indices, right_eyeball_indices):
    """
    Compare per-eyeball local UV (uv_neutral_mesh on 3D verts).

    Returns dict with mirror_right_u recommendation and diagnostics.
    """
    left_uv = _valid_vertex_uv(vertex_uvs, left_eyeball_indices)
    right_uv = _valid_vertex_uv(vertex_uvs, right_eyeball_indices)

    if left_uv.size == 0 or right_uv.size == 0:
        return {
            "mirror_right_u": False,
            "left_n": int(left_uv.shape[0]),
            "right_n": int(right_uv.shape[0]),
            "note": "empty eye UV",
        }

    left_n = _normalize_uv_box(left_uv)
    right_n = _normalize_uv_box(right_uv)
    right_mirror_u = right_n.copy()
    right_mirror_u[:, 0] = 1.0 - right_mirror_u[:, 0]

    centroid_direct = float(np.linalg.norm(left_n.mean(axis=0) - right_n.mean(axis=0)))
    centroid_mirror = float(np.linalg.norm(left_n.mean(axis=0) - right_mirror_u.mean(axis=0)))

    # Per-point: nearest-neighbor in normalized space is expensive; use mean + std alignment.
    std_direct = float(np.linalg.norm(left_n.std(axis=0) - right_n.std(axis=0)))
    std_mirror = float(np.linalg.norm(left_n.std(axis=0) - right_mirror_u.std(axis=0)))

    score_direct = centroid_direct + 0.25 * std_direct
    score_mirror = centroid_mirror + 0.25 * std_mirror
    mirror_right_u = score_mirror < score_direct

    return {
        "mirror_right_u": bool(mirror_right_u),
        "left_n": int(left_uv.shape[0]),
        "right_n": int(right_uv.shape[0]),
        "left_uv_u_range": [float(left_uv[:, 0].min()), float(left_uv[:, 0].max())],
        "right_uv_u_range": [float(right_uv[:, 0].min()), float(right_uv[:, 0].max())],
        "centroid_err_direct": centroid_direct,
        "centroid_err_mirror_u": centroid_mirror,
        "score_direct": score_direct,
        "score_mirror_u": score_mirror,
    }


def print_eye_uv_symmetry_report(report):
    print("\n========== Eye UV symmetry (shared bank) ==========")
    print(f"  left VT={report.get('left_n')}  right VT={report.get('right_n')}")
    if "left_uv_u_range" in report:
        print(f"  left  u range: {report['left_uv_u_range']}")
        print(f"  right u range: {report['right_uv_u_range']}")
    if "score_direct" in report:
        print(
            f"  alignment score direct={report['score_direct']:.4f}  "
            f"mirror_u={report['score_mirror_u']:.4f}"
        )
    mirror = report.get("mirror_right_u", False)
    print(f"  eye_uv_mirror_right_u = {mirror}")
    if mirror:
        print("  -> EyeTextureGaussians will apply u' = 1 - u on right eye UV.")
    else:
        print("  -> Left/right charts appear co-oriented; no u-mirror.")
    print("===================================================\n")
