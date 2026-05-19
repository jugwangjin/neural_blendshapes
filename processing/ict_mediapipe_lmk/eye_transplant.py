"""
FLAME eyeball similarity ``s,T`` (R=I) → ICT eyeball barycentric iris landmark transplant.

Per side:
  bidirectional chamfer(FLAME eyeball, ICT M_Sclera*∩eyeball) + 2 anchor pairs:
    front: FLAME MP iris center ↔ ICT sclera UV (0.5, 0.5)
    back:  reflected through eyeball centroid (front–back axis lock)
  project fitted FLAME iris pentagon → ICT sclera surface
"""

import numpy as np

from processing.ict_mediapipe_lmk.constants import (
    LEFT_IRIS_FLAME,
    LEFT_IRIS_MP,
    RIGHT_IRIS_FLAME,
    RIGHT_IRIS_MP,
)
from processing.ict_mediapipe_lmk.eye_rigid_align import fit_eye_rigid_chamfer_torch
from processing.ict_mediapipe_lmk.landmark_routing import CHART_LEFT_EYE, CHART_RIGHT_EYE, IRIS_MP_SET
from processing.ict_mediapipe_lmk.landmarks import project_points_to_mesh_bary
from utils.eye_chart import (
    eyeball_back_pole_3d,
    sclera_chart_point_3d,
    sclera_eyeball_face_mask,
    sclera_eyeball_vertex_indices,
)
from processing.ict_mediapipe_lmk.mesh_components import (
    check_projected_faces_are_in_eye,
    extract_flame_eye_components,
    extract_submesh,
    local_indices,
)

FLAME_LEFT_IRIS_SEEDS = np.array(LEFT_IRIS_FLAME, dtype=np.int64)
FLAME_RIGHT_IRIS_SEEDS = np.array(RIGHT_IRIS_FLAME, dtype=np.int64)
MP_LEFT_IRIS = np.array(LEFT_IRIS_MP, dtype=np.int64)
MP_RIGHT_IRIS = np.array(RIGHT_IRIS_MP, dtype=np.int64)
IRIS_MP_CENTER = {"left": int(LEFT_IRIS_MP[0]), "right": int(RIGHT_IRIS_MP[0])}


def _has_face_material_name(ict):
    if ict is None:
        return False
    if isinstance(ict, dict):
        return "face_material_name" in ict
    return hasattr(ict, "face_material_name")


def _eyeball_tri_mask(f_ict, eye_vertex_ids, num_verts):
    vmask = np.zeros(num_verts, dtype=bool)
    vmask[np.asarray(eye_vertex_ids, dtype=np.int64)] = True
    return np.all(vmask[f_ict], axis=1)


def _sclera_chamfer_target(v_ict, f_ict, ict_ref, side, ict_eye_ids):
    ch = "L" if side == "left" else "R"
    vids = sclera_eyeball_vertex_indices(ict_ref, ch, f_ict, ict_eye_ids, len(v_ict))
    if vids.size == 0:
        raise ValueError(f"[{side}] no sclera∩eyeball vertices for eye rigid target")
    return np.asarray(v_ict[vids], dtype=np.float64), int(vids.size)


def _flame_mediapipe_iris_center(v_flame, flame_iris_seed_global):
    """MP iris center on FLAME: first seed = MP 468 (L) / 473 (R) in ``constants``."""
    seeds = np.asarray(flame_iris_seed_global, dtype=np.int64)
    return np.asarray(v_flame[seeds[0]], dtype=np.float64)


def _flame_eyeball_back_pole(fl_eye_v, flame_iris_center):
    """Back pole on FLAME eyeball submesh (opposite to iris center through centroid)."""
    return eyeball_back_pole_3d(fl_eye_v, np.arange(len(fl_eye_v)), flame_iris_center)


def _log_iris_uv_on_sclera_chart(ict_npy_dict, face_idx, bary, side):
    tuv = np.asarray(ict_npy_dict["triangle_uv_local"], dtype=np.float64)
    fi = np.asarray(face_idx, dtype=np.int64)
    bary = np.asarray(bary, dtype=np.float64)
    uv = (tuv[fi] * bary[:, :, None]).sum(axis=1)
    center = np.array([0.5, 0.5], dtype=np.float64)
    d_center = np.linalg.norm(uv - center, axis=1)
    print(
        f"  [{side}] iris UV on M_Sclera* (triangle_uv_local): "
        f"u=[{uv[:, 0].min():.4f},{uv[:, 0].max():.4f}] "
        f"v=[{uv[:, 1].min():.4f},{uv[:, 1].max():.4f}] "
        f"dist(0.5,0.5) mean={d_center.mean():.4f} max={d_center.max():.4f}"
    )
    if d_center.max() < 0.03:
        print(
            f"  [{side}] NOTE: landmarks sit near chart center (0.5,0.5) — "
            "expected on filled sclera disk (pupil = UV center). Use *_iris5_texture_zoom.png for QA."
        )
    return uv


def transplant_one_eye(
    side,
    v_flame,
    f_flame,
    flame_eye_ids,
    flame_iris_seed_global,
    mp_iris_ids,
    v_ict,
    f_ict,
    ict_eye_ids,
    ict_iris_ids,
    device,
    ict=None,
    eye_rigid_iters=300,
    eye_rigid_lr=1e-2,
    eye_w_chamfer=1.0,
    eye_w_anchor=200.0,
):
    print(f"\n=== {side.upper()} EYE transplant (s,T only R=I + bidirectional chamfer) ===")

    fl_eye_v, fl_eye_f, _, fl_g2l = extract_submesh(v_flame, f_flame, flame_eye_ids)
    ict_eye_v, ict_eye_f, _, _ = extract_submesh(v_ict, f_ict, ict_eye_ids)

    fl_iris_local = local_indices(flame_iris_seed_global, fl_g2l)
    flame_iris_center = _flame_mediapipe_iris_center(v_flame, flame_iris_seed_global)
    mp_center = IRIS_MP_CENTER[side]
    print(f"  [{side}] FLAME anchor: MP iris center {mp_center} @ V={int(flame_iris_seed_global[0])}")

    ict_ref = ict if ict is not None else {}
    ict_ch = "L" if side == "left" else "R"
    if _has_face_material_name(ict_ref):
        tgt_chamfer_v, n_sclera_v = _sclera_chamfer_target(v_ict, f_ict, ict_ref, side, ict_eye_ids)
        ict_front = sclera_chart_point_3d(v_ict, f_ict, ict_ref, ict_ch, uv_target=(0.5, 0.5))
        ict_back = eyeball_back_pole_3d(v_ict, ict_eye_ids, ict_front)
        print(
            f"  [{side}] ICT anchors: front=M_Sclera* UV(0.5,0.5) back=centroid-reflect | "
            f"chamfer(bidirectional) target {n_sclera_v} sclera verts"
        )
    else:
        tgt_chamfer_v = ict_eye_v
        ict_front = ict_eye_v.mean(axis=0)
        ict_back = eyeball_back_pole_3d(v_ict, ict_eye_ids, ict_front)
        print(f"  [{side}] WARNING: no face_material_name — full eyeball chamfer + centroid anchors")

    flame_back = _flame_eyeball_back_pole(fl_eye_v, flame_iris_center)
    src_anchors = np.stack([flame_iris_center, flame_back], axis=0)
    tgt_anchors = np.stack([ict_front, ict_back], axis=0)
    axis_len = float(np.linalg.norm(ict_front - ict_back))
    print(
        f"  [{side}] FLAME back pole (eyeball submesh) | front–back ICT axis length={axis_len:.6f}"
    )

    fl_eye_fit_v, s, R, T, aligned_anchors = fit_eye_rigid_chamfer_torch(
        src_v=fl_eye_v,
        tgt_v=tgt_chamfer_v,
        src_anchors=src_anchors,
        tgt_anchors=tgt_anchors,
        iters=eye_rigid_iters,
        lr=eye_rigid_lr,
        w_chamfer=eye_w_chamfer,
        w_anchor=eye_w_anchor,
        device=str(device),
    )
    print(f"  [{side}] eye s,T result: s={s:.6f} (R=I)")

    fitted_iris_points = fl_eye_fit_v[fl_iris_local]
    spread_fit = np.linalg.norm(fitted_iris_points - fitted_iris_points.mean(axis=0), axis=1)
    print(
        f"  [{side}] fitted iris spread: mean={spread_fit.mean():.6f} "
        f"min={spread_fit.min():.6f} max={spread_fit.max():.6f}"
    )
    print(
        f"  [{side}] anchor dist after rigid: front={np.linalg.norm(aligned_anchors[0] - ict_front):.6f} "
        f"back={np.linalg.norm(aligned_anchors[1] - ict_back):.6f}"
    )

    if _has_face_material_name(ict):
        proj_mask = sclera_eyeball_face_mask(
            ict, "L" if side == "left" else "R", f_ict, ict_eye_ids, len(v_ict)
        )
        n_sclera = int(proj_mask.sum())
        if n_sclera == 0:
            raise ValueError(
                f"[{side}] no M_Sclera* triangles on eyeball verts — check face_material_name / npy"
            )
        print(
            f"  [{side}] iris MP → M_Sclera* ({n_sclera} tris, filled disk; M_Iris* annulus excluded). "
            f"Eye 3DGS UV uses sclera front hemisphere+ (see utils.eye_chart.sclera_sampling_face_indices)."
        )
    else:
        proj_mask = _eyeball_tri_mask(f_ict, ict_eye_ids, len(v_ict))
        print(f"  [{side}] WARNING: no face_material_name — iris projected to full eyeball mesh")

    ict_face_idx, ict_bary, dist = project_points_to_mesh_bary(
        fitted_iris_points, v_ict, f_ict, face_mask=proj_mask
    )

    check_projected_faces_are_in_eye(ict_face_idx, f_ict, ict_eye_ids, side=side)

    if ict is not None and isinstance(ict, dict) and "triangle_uv_local" in ict:
        _log_iris_uv_on_sclera_chart(ict, ict_face_idx, ict_bary, side)

    target_type = f"{side}_iris"
    print(f"  mp ids: {mp_iris_ids.tolist()}")
    print(f"  transfer dist: mean={dist.mean():.6f} max={dist.max():.6f}")

    return {
        "mp_landmark_indices": mp_iris_ids.astype(np.int64),
        "ict_lmk_face_idx": ict_face_idx.astype(np.int64),
        "ict_lmk_b_coords": ict_bary.astype(np.float32),
        "transfer_error": dist.astype(np.float32),
        "ict_lmk_target_type": np.array([target_type] * len(mp_iris_ids), dtype=object),
        "source": np.array(["flame_eye_rigid"] * len(mp_iris_ids), dtype=object),
        "geometry_chart_id": np.array(
            [CHART_LEFT_EYE if side == "left" else CHART_RIGHT_EYE] * len(mp_iris_ids),
            dtype=np.int32,
        ),
        "mesh_debug": {
            "side": side,
            "flame_eye_fit_vertices": np.asarray(fl_eye_fit_v, dtype=np.float64),
            "flame_eye_fit_faces": np.asarray(fl_eye_f, dtype=np.int64),
            "flame_eye_canonical_vertices": np.asarray(fl_eye_v, dtype=np.float64),
            "ict_eyeball_vertices": np.asarray(ict_eye_v, dtype=np.float64),
            "ict_eyeball_faces": np.asarray(ict_eye_f, dtype=np.int64),
            "ict_eyeball_vertex_ids_global": np.asarray(ict_eye_ids, dtype=np.int64),
            "flame_eye_vertex_ids_global": np.asarray(flame_eye_ids, dtype=np.int64),
            "fitted_iris_points": np.asarray(fitted_iris_points, dtype=np.float64),
            "flame_iris_center": np.asarray(flame_iris_center, dtype=np.float64),
            "flame_eyeball_back": np.asarray(flame_back, dtype=np.float64),
            "ict_sclera_uv_center": np.asarray(ict_front, dtype=np.float64),
            "ict_eyeball_back": np.asarray(ict_back, dtype=np.float64),
            "flame_alignment_s": float(s),
            "flame_alignment_R": np.asarray(R, dtype=np.float64),
            "flame_alignment_T": np.asarray(T, dtype=np.float64),
            "flame_iris_seed_global": np.asarray(flame_iris_seed_global, dtype=np.int64),
        },
    }


def ensure_ict_in_flame_space_for_eyes(v_ict, ict_npy_dict):
    """
    Eye transplant expects ICT in FLAME space (``jawOpen`` + npy ``flame_alignment_*``).
    Re-apply from ``neutral_mesh`` when the passed mesh still looks like raw neutral.
    """
    from processing.ict_flame_similarity import (
        apply_ict_to_flame_space,
        has_flame_alignment,
        ict_mesh_with_jaw,
    )

    if ict_npy_dict is None:
        return np.asarray(v_ict, dtype=np.float64)
    neutral = np.asarray(ict_npy_dict["neutral_mesh"], dtype=np.float64)
    v = np.asarray(v_ict, dtype=np.float64)
    v_jaw = ict_mesh_with_jaw(neutral, ict_npy_dict)
    if np.linalg.norm(v - neutral) < 1e-5:
        v = apply_ict_to_flame_space(
            neutral, ict_npy_dict, use_final_alignment=has_flame_alignment(ict_npy_dict)
        )
        print("  eye transplant: applied npy jaw+flame_alignment to neutral_mesh")
    elif np.linalg.norm(v - v_jaw) < 1e-5:
        v = apply_ict_to_flame_space(
            v_jaw, ict_npy_dict, use_final_alignment=has_flame_alignment(ict_npy_dict)
        )
        print("  eye transplant: applied npy flame_alignment on jaw-open mesh")
    return v


def log_eyeball_flame_space_check(v_flame, v_ict, regions, flame_left_ids, flame_right_ids):
    left_ict = np.asarray(regions["left_eyeball_indices"], dtype=np.int64)
    right_ict = np.asarray(regions["right_eyeball_indices"], dtype=np.int64)
    for name, fl_ids, ic_ids in (
        ("L", flame_left_ids, left_ict),
        ("R", flame_right_ids, right_ict),
    ):
        c_fl = v_flame[fl_ids].mean(axis=0)
        c_ic = v_ict[ic_ids].mean(axis=0)
        dist = float(np.linalg.norm(c_fl - c_ic))
        print(f"  eyeball center {name}: FLAME↔ICT dist={dist:.6f} m (after global npy align)")


def run_eye_transplant(
    v_flame,
    f_flame,
    v_ict,
    f_ict,
    regions,
    device,
    ict_npy_dict=None,
    eye_rigid_iters=300,
    eye_rigid_lr=1e-2,
    eye_w_chamfer=1.0,
    eye_w_anchor=200.0,
):
    left_ict = np.asarray(regions["left_eyeball_indices"], dtype=np.int64)
    right_ict = np.asarray(regions["right_eyeball_indices"], dtype=np.int64)
    if left_ict.size == 0 or right_ict.size == 0:
        eye = np.asarray(regions.get("eyeball_indices", []), dtype=np.int64)
        mid = len(eye) // 2
        if left_ict.size == 0:
            left_ict = eye[:mid]
        if right_ict.size == 0:
            right_ict = eye[mid:]

    v_ict = ensure_ict_in_flame_space_for_eyes(v_ict, ict_npy_dict)

    flame_left, flame_right = extract_flame_eye_components(
        v_flame, f_flame, FLAME_LEFT_IRIS_SEEDS, FLAME_RIGHT_IRIS_SEEDS
    )
    log_eyeball_flame_space_check(v_flame, v_ict, regions, flame_left, flame_right)

    ict_ref = ict_npy_dict if ict_npy_dict is not None else regions

    left_pack = transplant_one_eye(
        "left",
        v_flame,
        f_flame,
        flame_left,
        FLAME_LEFT_IRIS_SEEDS,
        MP_LEFT_IRIS,
        v_ict,
        f_ict,
        left_ict,
        regions.get("left_iris_indices", []),
        device,
        ict=ict_ref,
        eye_rigid_iters=eye_rigid_iters,
        eye_rigid_lr=eye_rigid_lr,
        eye_w_chamfer=eye_w_chamfer,
        eye_w_anchor=eye_w_anchor,
    )
    right_pack = transplant_one_eye(
        "right",
        v_flame,
        f_flame,
        flame_right,
        FLAME_RIGHT_IRIS_SEEDS,
        MP_RIGHT_IRIS,
        v_ict,
        f_ict,
        right_ict,
        regions.get("right_iris_indices", []),
        device,
        ict=ict_ref,
        eye_rigid_iters=eye_rigid_iters,
        eye_rigid_lr=eye_rigid_lr,
        eye_w_chamfer=eye_w_chamfer,
        eye_w_anchor=eye_w_anchor,
    )

    return {
        "mp_landmark_indices": np.concatenate(
            [left_pack["mp_landmark_indices"], right_pack["mp_landmark_indices"]]
        ),
        "ict_lmk_face_idx": np.concatenate(
            [left_pack["ict_lmk_face_idx"], right_pack["ict_lmk_face_idx"]]
        ),
        "ict_lmk_b_coords": np.concatenate(
            [left_pack["ict_lmk_b_coords"], right_pack["ict_lmk_b_coords"]], axis=0
        ),
        "transfer_error": np.concatenate([left_pack["transfer_error"], right_pack["transfer_error"]]),
        "ict_lmk_target_type": np.concatenate(
            [left_pack["ict_lmk_target_type"], right_pack["ict_lmk_target_type"]]
        ),
        "source": np.concatenate([left_pack["source"], right_pack["source"]]),
        "geometry_chart_id": np.concatenate(
            [left_pack["geometry_chart_id"], right_pack["geometry_chart_id"]]
        ),
        "left": left_pack,
        "right": right_pack,
        "flame_left_eye_indices": flame_left,
        "flame_right_eye_indices": flame_right,
    }


def merge_iris_into_embedding(face_embedding, eye_embedding):
    """Replace iris MP 468–477 in face embedding with eye-transplant results."""
    mp = face_embedding["mp_landmark_indices"]
    keep = np.array([int(m) not in IRIS_MP_SET for m in mp], dtype=bool)

    def cat(key):
        return np.concatenate([face_embedding[key][keep], eye_embedding[key]])

    merged = {
        "mp_landmark_indices": cat("mp_landmark_indices").astype(np.int64),
        "ict_lmk_face_idx": cat("ict_lmk_face_idx").astype(np.int64),
        "ict_lmk_b_coords": cat("ict_lmk_b_coords"),
        "transfer_error": cat("transfer_error"),
        "ict_lmk_target_type": np.concatenate(
            [face_embedding["ict_lmk_target_type"][keep], eye_embedding["ict_lmk_target_type"]]
        ),
        "source": np.concatenate([face_embedding["source"][keep], eye_embedding["source"]]),
        "geometry_chart_id": np.concatenate(
            [face_embedding["geometry_chart_id"][keep], eye_embedding["geometry_chart_id"]]
        ).astype(np.int32),
    }
    print(
        f"merged embedding: {keep.sum()} face/eyelid + {len(eye_embedding['mp_landmark_indices'])} iris "
        f"= {len(merged['mp_landmark_indices'])} total"
    )
    return merged
