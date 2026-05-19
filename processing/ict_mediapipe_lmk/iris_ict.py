"""
Bake MediaPipe iris landmarks directly on ICT eyeball components.

FLAME has no disconnected eyeball; iris points sampled on FLAME face and projected
to ICT often land on eyelid skin. This module defines iris targets on ICT eyeball only.
"""

import numpy as np

from processing.ict_mediapipe_lmk.constants import LEFT_IRIS_MP, RIGHT_IRIS_MP
from processing.ict_mediapipe_lmk.landmarks import project_points_to_mesh_bary

# center + up/down/left/right in local tangent frame (unit directions, scaled by radius)
IRIS_LOCAL_DIRS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.35, 0.0],
        [0.35, 0.0, 0.0],
        [0.0, -0.35, 0.0],
        [-0.35, 0.0, 0.0],
    ],
    dtype=np.float64,
)


def _eyeball_tri_mask(faces, eyeball_vertex_ids, num_verts):
    vmask = np.zeros(num_verts, dtype=bool)
    vmask[np.asarray(eyeball_vertex_ids, dtype=np.int64)] = True
    return np.all(vmask[faces], axis=1)


def _local_frame(verts):
    center = verts.mean(axis=0)
    x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(verts[:, 0].std()) < 1e-6:
        x = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    z = np.cross(x, verts[0] - center)
    if np.linalg.norm(z) < 1e-8:
        z = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    z = z / (np.linalg.norm(z) + 1e-8)
    x = np.cross(verts[1] - center, z)
    x = x / (np.linalg.norm(x) + 1e-8)
    y = np.cross(z, x)
    y = y / (np.linalg.norm(y) + 1e-8)
    return center, x, y, z


def iris_query_points_on_eyeball(vertices, eyeball_vertex_ids):
    """5 query points in world space on the eyeball component (center + 4 ring)."""
    eye = np.asarray(eyeball_vertex_ids, dtype=np.int64)
    verts = np.asarray(vertices, dtype=np.float64)[eye]
    center, ax, ay, az = _local_frame(verts)
    radii = np.linalg.norm(verts - center, axis=1)
    radius = float(np.median(radii)) * 0.45

    points = []
    for d in IRIS_LOCAL_DIRS:
        offset = d[0] * ax + d[1] * ay + d[2] * az
        if np.linalg.norm(d) < 1e-6:
            points.append(center)
        else:
            points.append(center + offset * radius)
    return np.stack(points, axis=0)


def bake_iris_landmarks_ict(vertices, faces, left_eyeball_indices, right_eyeball_indices):
    """
    Returns dict with keys left_iris_mp, right_iris_mp, points_ict [10,3] ordered
    left 5 then right 5.
    """
    num_verts = vertices.shape[0]
    left_pts = iris_query_points_on_eyeball(vertices, left_eyeball_indices)
    right_pts = iris_query_points_on_eyeball(vertices, right_eyeball_indices)

    left_mask = _eyeball_tri_mask(faces, left_eyeball_indices, num_verts)
    right_mask = _eyeball_tri_mask(faces, right_eyeball_indices, num_verts)

    left_f, left_b, left_e = project_points_to_mesh_bary(left_pts, vertices, faces, left_mask)
    right_f, right_b, right_e = project_points_to_mesh_bary(right_pts, vertices, faces, right_mask)

    return {
        "left_iris_mp": np.array(LEFT_IRIS_MP, dtype=np.int64),
        "right_iris_mp": np.array(RIGHT_IRIS_MP, dtype=np.int64),
        "left_face_idx": left_f,
        "left_bary": left_b,
        "left_error": left_e,
        "right_face_idx": right_f,
        "right_bary": right_b,
        "right_error": right_e,
        "left_points_ict": left_pts,
        "right_points_ict": right_pts,
    }
