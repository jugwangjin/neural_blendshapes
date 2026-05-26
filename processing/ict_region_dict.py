"""
Build ICT region index arrays for assets/ict_facekit_torch.npy.

Default asset: full head parts #0–#16 (vertices 0:26719), including ``M_EyeOcclusion``.
Official #0–#8 (0:24591) remains available via ``build_official_region_indices``.
"""

from __future__ import annotations

import numpy as np

ASSET_SCHEMA_VERSION = 7
OFFICIAL_PART_SPLITS = [9409, 11248, 13294, 13678, 14062, 17039, 21451, 23021, 24591]
VERTEX_COUNT_STANDARD = 24591

# ICT-FaceKit README — full light model (parts #0–#16, vertices 0:26718).
OFFICIAL_FULL_PART_SPLITS = [
    9409,
    11248,
    13294,
    13678,
    14062,
    17039,
    21451,
    23021,
    24591,
    24795,
    24999,
    25023,
    25047,
    25199,
    25351,
    26035,
    26719,
]
VERTEX_COUNT_FULL = 26719

PART_FACE_SKIN = 0
PART_HEAD_NECK = 1
PART_MOUTH_SOCKET = 2
PART_EYE_SOCKET_L = 3
PART_EYE_SOCKET_R = 4
PART_GUMS_TONGUE = 5
PART_TEETH = 6
PART_EYEBALL_L = 7
PART_EYEBALL_R = 8


def _range(a, b):
    return list(range(a, b))


def vertex_parts_from_splits(n_verts, splits):
    vertex_parts = [0] * n_verts
    for i, end in enumerate(splits):
        start = 0 if i == 0 else splits[i - 1]
        for v in range(start, end):
            vertex_parts[v] = i
    return vertex_parts


def indices_for_parts(vertex_parts, *part_ids):
    vp = np.asarray(vertex_parts, dtype=np.int64)
    return np.where(np.isin(vp, list(part_ids)))[0].tolist()


def build_official_region_indices():
    """
    Region lists for the standard 24591-vertex ICT mesh (README parts #0–#8).
    """
    skin_face = _range(0, 9409)
    head_neck = _range(9409, 11248)
    mouth_socket = _range(11248, 13294)
    eye_socket_left = _range(13294, 13678)
    eye_socket_right = _range(13678, 14062)
    gums_tongue = _range(14062, 17039)
    teeth = _range(17039, 21451)
    left_eyeball = _range(21451, 23021)
    right_eyeball = _range(23021, 24591)

    face_indices = skin_face + mouth_socket + gums_tongue + teeth
    not_face_indices = head_neck
    eyeball_indices = left_eyeball + right_eyeball
    head_indices = face_indices + not_face_indices
    surface_sample_vertex_indices = (
        skin_face
        + head_neck
        + mouth_socket
        + eye_socket_left
        + eye_socket_right
        + gums_tongue
    )

    return {
        "skin_face_indices": skin_face,
        "head_neck_indices": head_neck,
        "mouth_socket_indices": mouth_socket,
        "eye_socket_left_indices": eye_socket_left,
        "eye_socket_right_indices": eye_socket_right,
        "mouth_interior_vertex_indices": gums_tongue,
        "gums_tongue_indices": gums_tongue,
        "teeth_indices": teeth,
        "left_eyeball_indices": left_eyeball,
        "right_eyeball_indices": right_eyeball,
        "left_iris_indices": [],
        "right_iris_indices": [],
        "face_indices": face_indices,
        "not_face_indices": not_face_indices,
        "eyeball_indices": eyeball_indices,
        "head_indices": head_indices,
        "surface_sample_vertex_indices": surface_sample_vertex_indices,
    }


def build_full_head_region_indices():
    """
    Parts #0–#8 (official) plus #9–#16 (lacrimal, eye blend, eye occlusion, eyelashes).

    Eye-occlusion verts (13–14) are kept for ``M_EyeOcclusion`` surface Gaussians;
    lacrimal / eye-blend / eyelashes are stored but gated off in ``expr_regions``.
    """
    r = build_official_region_indices()
    lacrimal = _range(24591, 24795)
    eye_blend = _range(24795, 25047)
    eye_occ_left = _range(25047, 25199)
    eye_occ_right = _range(25199, 25351)
    eyelashes_left = _range(25351, 26035)
    eyelashes_right = _range(26035, 26719)
    r.update(
        {
            "lacrimal_indices": lacrimal,
            "eye_blend_indices": eye_blend,
            "left_eye_occlusion_indices": eye_occ_left,
            "right_eye_occlusion_indices": eye_occ_right,
            "eyelashes_left_indices": eyelashes_left,
            "eyelashes_right_indices": eyelashes_right,
            "auxiliary_part_indices": (
                lacrimal
                + eye_blend
                + eye_occ_left
                + eye_occ_right
                + eyelashes_left
                + eyelashes_right
            ),
        }
    )
    return r


def build_region_dict(
    vertices,
    vertex_parts,
    face_indices,
    not_face_indices,
    eyeball_indices,
    parts_split,
    asset_variant=None,
):
    """Merge metadata for npy (region lists + schema fields)."""
    n_verts = int(np.asarray(vertices).shape[0])
    if n_verts >= VERTEX_COUNT_FULL - 1:
        regions = build_full_head_region_indices()
        if asset_variant is None:
            asset_variant = "full_head_26719"
        parts_split = list(parts_split or OFFICIAL_FULL_PART_SPLITS)
    else:
        regions = build_official_region_indices()
        if asset_variant is None:
            asset_variant = "official_24591"
        parts_split = list(parts_split or OFFICIAL_PART_SPLITS)
    if n_verts != VERTEX_COUNT_STANDARD and n_verts < VERTEX_COUNT_FULL - 1:
        regions["left_eyeball_indices"], regions["right_eyeball_indices"] = _split_eyeballs_by_x(
            vertices, eyeball_indices
        )

    return {
        "asset_variant": asset_variant,
        "asset_schema_version": ASSET_SCHEMA_VERSION,
        "vertex_count": n_verts,
        "parts_split": parts_split,
        "official_part_splits": list(OFFICIAL_PART_SPLITS),
        "full_head_part_splits": list(OFFICIAL_FULL_PART_SPLITS),
        **regions,
    }


def _split_eyeballs_by_x(vertices, eyeball_indices):
    eye = np.asarray(eyeball_indices, dtype=np.int64)
    if eye.size == 0:
        return [], []
    xs = np.asarray(vertices, dtype=np.float64)[eye, 0]
    med = float(np.median(xs))
    return eye[xs <= med].tolist(), eye[xs > med].tolist()
