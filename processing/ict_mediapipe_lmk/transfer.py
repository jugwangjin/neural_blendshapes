"""Component-aware MediaPipe → ICT barycentric transfer (face + eyelid only).

Iris landmarks 468–477 are merged from eye_transplant.run_eye_transplant() in bake script.
"""

import numpy as np

from processing.ict_mediapipe_lmk.landmark_routing import IRIS_MP_SET, classify_mp_landmark, geometry_chart_id
from processing.ict_mediapipe_lmk.landmarks import project_points_to_mesh_bary


def _vertex_id_mask(num_verts, vertex_ids):
    mask = np.zeros(num_verts, dtype=bool)
    mask[np.asarray(vertex_ids, dtype=np.int64)] = True
    return mask


def _tri_mask_all_vertices_in(faces, vertex_ids, num_verts):
    return np.all(_vertex_id_mask(num_verts, vertex_ids)[faces], axis=1)


def _tri_mask_any_vertex_in(faces, vertex_ids, num_verts):
    return np.any(_vertex_id_mask(num_verts, vertex_ids)[faces], axis=1)


def build_projection_masks(f_ict, regions, num_verts):
    eyeball = regions.get("eyeball_indices", [])
    left_eye = regions.get("left_eyeball_indices", [])
    right_eye = regions.get("right_eyeball_indices", [])
    if len(left_eye) == 0 and len(eyeball) > 0:
        mid = len(eyeball) // 2
        left_eye, right_eye = eyeball[:mid], eyeball[mid:]
    left_socket = regions.get("eye_socket_left_indices", [])
    right_socket = regions.get("eye_socket_right_indices", [])

    surface = regions.get("surface_sample_vertex_indices")
    if surface is None:
        surface = (
            regions.get("skin_face_indices", [])
            + regions.get("head_neck_indices", regions.get("not_face_indices", []))
            + regions.get("mouth_socket_indices", [])
            + regions.get("gums_tongue_indices", regions.get("mouth_interior_vertex_indices", []))
        )

    surface = [v for v in surface if v not in set(eyeball)]
    face_mask = _tri_mask_all_vertices_in(f_ict, surface, num_verts)
    face_mask &= ~_tri_mask_any_vertex_in(f_ict, eyeball, num_verts)

    eye_v = set(eyeball)
    left_eyelid_verts = list(left_socket) + [v for v in regions.get("skin_face_indices", []) if v not in eye_v]
    right_eyelid_verts = list(right_socket) + [v for v in regions.get("skin_face_indices", []) if v not in eye_v]
    left_eyelid_mask = _tri_mask_all_vertices_in(f_ict, left_eyelid_verts, num_verts)
    left_eyelid_mask &= ~_tri_mask_any_vertex_in(f_ict, eyeball, num_verts)
    right_eyelid_mask = _tri_mask_all_vertices_in(f_ict, right_eyelid_verts, num_verts)
    right_eyelid_mask &= ~_tri_mask_any_vertex_in(f_ict, eyeball, num_verts)

    return {"face": face_mask, "left_eyelid": left_eyelid_mask, "right_eyelid": right_eyelid_mask}


def validate_embedding(embedding, regions):
    types = embedding["ict_lmk_target_type"]
    mp_ids = embedding["mp_landmark_indices"]
    bad = []
    for mp_idx, t in zip(mp_ids, types):
        route = classify_mp_landmark(int(mp_idx))
        if route == "left_iris" and t != "left_iris":
            bad.append((int(mp_idx), route, t))
        if route == "right_iris" and t != "right_iris":
            bad.append((int(mp_idx), route, t))
        if route == "left_eyelid" and t in ("left_iris", "right_iris", "right_eyelid"):
            bad.append((int(mp_idx), route, t))
        if route == "right_eyelid" and t in ("right_iris", "left_iris", "left_eyelid"):
            bad.append((int(mp_idx), route, t))
        if route == "face" and t in ("left_iris", "right_iris"):
            bad.append((int(mp_idx), route, t))
    if bad:
        print(f"WARNING: {len(bad)} landmark routing mismatches (first 10): {bad[:10]}")
    else:
        print("embedding routing validation: OK")
    return bad


def transfer_mediapipe_to_ict(
    mp_pack,
    v_ict_fit,
    f_ict,
    face_indices,
    eyeball_indices,
    left_eyeball_indices=None,
    right_eyeball_indices=None,
    left_iris_indices=None,
    right_iris_indices=None,
    eye_socket_left_indices=None,
    eye_socket_right_indices=None,
    surface_sample_vertex_indices=None,
    skin_face_indices=None,
    head_neck_indices=None,
    mouth_socket_indices=None,
    gums_tongue_indices=None,
    v_ict_neutral=None,
):
    """
    Face + eyelid landmarks only (skips MP iris 468–477).
    Iris is added via eye_transplant.merge_iris_into_embedding().
    """
    num_verts = v_ict_fit.shape[0]
    regions = {
        "face_indices": face_indices,
        "eyeball_indices": eyeball_indices,
        "left_eyeball_indices": left_eyeball_indices or [],
        "right_eyeball_indices": right_eyeball_indices or [],
        "eye_socket_left_indices": eye_socket_left_indices or [],
        "eye_socket_right_indices": eye_socket_right_indices or [],
        "surface_sample_vertex_indices": surface_sample_vertex_indices,
        "skin_face_indices": skin_face_indices or [],
        "head_neck_indices": head_neck_indices or [],
        "mouth_socket_indices": mouth_socket_indices or [],
        "gums_tongue_indices": gums_tongue_indices or [],
    }
    masks = build_projection_masks(f_ict, regions, num_verts)

    all_face_idx = []
    all_bary = []
    all_error = []
    all_type = []
    all_source = []
    all_chart = []

    mp_ids = mp_pack["mp_ids"]
    points_flame = mp_pack["points_flame"]

    for i, mp_idx in enumerate(mp_ids):
        mp_idx = int(mp_idx)
        if mp_idx in IRIS_MP_SET:
            continue

        route = classify_mp_landmark(mp_idx)
        pt = points_flame[i : i + 1]
        if route == "left_eyelid":
            f_mask = masks["left_eyelid"]
            tname = "left_eyelid"
        elif route == "right_eyelid":
            f_mask = masks["right_eyelid"]
            tname = "right_eyelid"
        else:
            f_mask = masks["face"]
            tname = "face"

        f_idx, bary, err = project_points_to_mesh_bary(pt, v_ict_fit, f_ict, face_mask=f_mask)
        all_face_idx.append(f_idx)
        all_bary.append(bary)
        all_error.append(err)
        all_type.append(tname)
        all_source.append("metrical-tracker")
        all_chart.append(geometry_chart_id(tname))

    embedding = {
        "mp_landmark_indices": np.array(
            [int(m) for m in mp_ids if int(m) not in IRIS_MP_SET], dtype=np.int64
        ),
        "ict_lmk_face_idx": np.concatenate(all_face_idx),
        "ict_lmk_b_coords": np.concatenate(all_bary, axis=0),
        "transfer_error": np.concatenate(all_error),
        "ict_lmk_target_type": np.array(all_type, dtype=object),
        "source": np.array(all_source, dtype=object),
        "geometry_chart_id": np.array(all_chart, dtype=np.int32),
    }
    return embedding
