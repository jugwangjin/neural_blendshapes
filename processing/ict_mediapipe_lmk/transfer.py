"""Project FLAME MediaPipe points onto fitted ICT mesh and build embedding."""

import numpy as np

from ict_mediapipe_lmk.constants import ICT_LEFT_IRIS_VERTICES, ICT_RIGHT_IRIS_VERTICES
from ict_mediapipe_lmk.landmarks import project_points_to_mesh_bary


def _iris_vertex_mask(num_verts, eyeball_indices, side):
    if num_verts > 24590:
        sl = ICT_LEFT_IRIS_VERTICES if side == "left" else ICT_RIGHT_IRIS_VERTICES
        mask = np.zeros(num_verts, dtype=bool)
        mask[sl] = True
        return mask

    eyeball = np.asarray(eyeball_indices, dtype=np.int64)
    mid = len(eyeball) // 2
    subset = eyeball[:mid] if side == "left" else eyeball[mid:]
    mask = np.zeros(num_verts, dtype=bool)
    mask[subset] = True
    return mask


def _face_vertex_mask(num_verts, face_indices):
    mask = np.zeros(num_verts, dtype=bool)
    mask[np.asarray(face_indices, dtype=np.int64)] = True
    return mask


def transfer_mediapipe_to_ict(
    mp_pack,
    v_ict_fit,
    f_ict,
    face_indices,
    eyeball_indices,
):
    num_verts = v_ict_fit.shape[0]
    face_mask = _face_vertex_mask(num_verts, face_indices)
    face_tri_mask = np.all(face_mask[f_ict], axis=1)

    left_mask = _iris_vertex_mask(num_verts, eyeball_indices, "left")
    right_mask = _iris_vertex_mask(num_verts, eyeball_indices, "right")
    left_tri_mask = np.all(left_mask[f_ict], axis=1)
    right_tri_mask = np.all(right_mask[f_ict], axis=1)

    all_face_idx = []
    all_bary = []
    all_error = []
    all_type = []
    all_source = []

    skin_n = len(mp_pack["skin_mp_ids"])
    skin_points = mp_pack["skin_points_flame"]
    f_idx, bary, err = project_points_to_mesh_bary(
        skin_points, v_ict_fit, f_ict, face_mask=face_tri_mask
    )
    all_face_idx.append(f_idx)
    all_bary.append(bary)
    all_error.append(err)
    all_type.extend(["face"] * skin_n)
    all_source.extend(["metrical-tracker"] * skin_n)

    left_n = len(mp_pack["left_iris_mp"])
    left_points = mp_pack["points_flame"][skin_n : skin_n + left_n]
    f_idx, bary, err = project_points_to_mesh_bary(
        left_points, v_ict_fit, f_ict, face_mask=left_tri_mask
    )
    all_face_idx.append(f_idx)
    all_bary.append(bary)
    all_error.append(err)
    all_type.extend(["left_iris"] * left_n)
    all_source.extend(["iris_hardcoded"] * left_n)

    right_n = len(mp_pack["right_iris_mp"])
    right_start = skin_n + left_n
    right_points = mp_pack["points_flame"][right_start : right_start + right_n]
    f_idx, bary, err = project_points_to_mesh_bary(
        right_points, v_ict_fit, f_ict, face_mask=right_tri_mask
    )
    all_face_idx.append(f_idx)
    all_bary.append(bary)
    all_error.append(err)
    all_type.extend(["right_iris"] * right_n)
    all_source.extend(["iris_hardcoded"] * right_n)

    return {
        "mp_landmark_indices": mp_pack["mp_ids"].astype(np.int64),
        "ict_lmk_face_idx": np.concatenate(all_face_idx),
        "ict_lmk_b_coords": np.concatenate(all_bary, axis=0),
        "transfer_error": np.concatenate(all_error),
        "ict_lmk_target_type": np.array(all_type, dtype=object),
        "source": np.array(all_source, dtype=object),
    }
