"""
Iris MP 468–477 → ICT ``M_EyeOcclusion`` via shared center normal ray cast.

Per eye:
  - FLAME iris center vertex (``LEFT/RIGHT_IRIS_FLAME[0]``) → outward unit normal ``n``
  - All 5 FLAME iris 3D points cast the **same** direction ``n`` (not per-point normals)
  - Ray–triangle hit on ``M_EyeOcclusion`` only; barycentric coords on hit triangle

ICT mesh: jawOpen + ``flame_alignment_*`` only (no face NICP, no per-eye rigid).
"""

from __future__ import annotations

import numpy as np
import trimesh

from processing.ict_flame_similarity import apply_ict_to_flame_space, has_flame_alignment
from processing.ict_mediapipe_lmk.constants import (
    LEFT_IRIS_FLAME,
    LEFT_IRIS_MP,
    RIGHT_IRIS_FLAME,
    RIGHT_IRIS_MP,
)
from processing.ict_mediapipe_lmk.landmark_routing import CHART_LEFT_EYE, CHART_RIGHT_EYE
from processing.ict_mediapipe_lmk.landmarks import barycentric_coords, validate_lmk_face_indices
from utils.ict_regions import eye_occlusion_layout_face_indices


def ict_vertices_for_iris_bake(ict_npy_dict):
    """Neutral + jawOpen + npy FLAME map (``ict.expression_reference_verts()`` space)."""
    return apply_ict_to_flame_space(
        ict_npy_dict["neutral_mesh"],
        ict_npy_dict,
        use_final_alignment=has_flame_alignment(ict_npy_dict),
    )


def _vertex_normals_np(vertices, faces):
    import torch

    from utils.mesh_ops import vertex_normals

    v = torch.tensor(np.asarray(vertices, dtype=np.float32))
    f = torch.tensor(np.asarray(faces, dtype=np.int64), dtype=torch.long)
    return vertex_normals(v, f).detach().cpu().numpy()


def _iris_center_normal(v_flame, f_flame, center_vertex_idx):
    vn = _vertex_normals_np(v_flame, f_flame)
    n = vn[int(center_vertex_idx)].astype(np.float64)
    n_norm = float(np.linalg.norm(n))
    if n_norm < 1e-12:
        raise ValueError(f"degenerate iris center normal at vertex {center_vertex_idx}")
    return n / n_norm


def _occlusion_global_face_indices(f_ict, ict_ref, n_verts):
    class _Ref:
        face_material_name = ict_ref.get("face_material_name")

    fi = eye_occlusion_layout_face_indices(_Ref)
    if fi.size == 0:
        raise ValueError(
            "no M_EyeOcclusion faces — rebuild full-head ict_facekit_torch.npy with face_material_name"
        )
    tri = f_ict[fi]
    if np.any(tri >= n_verts) or np.any(tri < 0):
        raise ValueError("occlusion face indices out of vertex range")
    return fi.astype(np.int64)


def _ray_hits_along_direction(origins, direction, vertices, occ_face_idx, f_ict):
    """
    origins [N,3], shared unit direction [3].
    Returns hits [N,3], global_face_idx [N], local_tri [N], miss mask [N].
    """
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    origins = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
    n_rays = origins.shape[0]

    occ_faces = f_ict[occ_face_idx]
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices, dtype=np.float64), faces=occ_faces, process=False)
    dirs = np.tile(direction.reshape(1, 3), (n_rays, 1))

    locs, ray_ids, tri_ids = mesh.ray.intersects_location(
        ray_origins=origins,
        ray_directions=dirs,
    )

    hits = np.full((n_rays, 3), np.nan, dtype=np.float64)
    face_out = np.full(n_rays, -1, dtype=np.int64)
    miss = np.ones(n_rays, dtype=bool)

    locs = np.asarray(locs, dtype=np.float64)
    ray_ids = np.asarray(ray_ids, dtype=np.int64)
    tri_ids = np.asarray(tri_ids, dtype=np.int64)
    global_fi = np.asarray(occ_face_idx, dtype=np.int64)

    for i in range(n_rays):
        m = ray_ids == i
        if not m.any():
            continue
        pts = locs[m]
        tri_local = tri_ids[m]
        t = (pts - origins[i]) @ direction
        forward = t > 1e-6
        if forward.any():
            j = int(np.where(forward)[0][np.argmin(t[forward])])
        else:
            j = int(np.argmax(t))
        hits[i] = pts[j]
        face_out[i] = int(global_fi[tri_local[j]])
        miss[i] = False

    return hits, face_out, miss


def _bary_on_hit_faces(hits, face_idx, vertices, faces):
    bary = np.zeros((len(hits), 3), dtype=np.float64)
    for i, fi in enumerate(face_idx):
        if fi < 0 or np.isnan(hits[i, 0]):
            continue
        tri = faces[int(fi)]
        a, b, c = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        bary[i] = barycentric_coords(hits[i], a, b, c)
    return bary.astype(np.float32)


def _transfer_error(hits, origins):
    d = np.linalg.norm(hits - origins, axis=1)
    d[np.isnan(d)] = 0.0
    return d.astype(np.float32)


def bake_iris_one_side(
    side,
    v_flame,
    f_flame,
    v_ict,
    f_ict,
    ict_npy_dict,
    *,
    flame_iris_vertex_ids,
    mp_iris_ids,
    chart_id,
):
    flame_iris_vertex_ids = np.asarray(flame_iris_vertex_ids, dtype=np.int64)
    mp_iris_ids = np.asarray(mp_iris_ids, dtype=np.int64)
    center_v = int(flame_iris_vertex_ids[0])

    n = _iris_center_normal(v_flame, f_flame, center_v)
    origins = np.asarray(v_flame[flame_iris_vertex_ids], dtype=np.float64)

    occ_fi = _occlusion_global_face_indices(f_ict, ict_npy_dict, len(v_ict))
    hits, face_idx, miss = _ray_hits_along_direction(origins, n, v_ict, occ_fi, f_ict)

    if miss.any():
        print(
            f"  WARNING [{side}] iris ray miss on {int(miss.sum())}/{len(miss)} points "
            f"(direction=n_center, material=M_EyeOcclusion)"
        )

    bary = _bary_on_hit_faces(hits, face_idx, v_ict, f_ict)
    dist = _transfer_error(hits, origins)

    print(
        f"  [{side}] iris ray→occlusion: center_v={center_v} n={n.tolist()} "
        f"occ_faces={len(occ_fi)} dist mean={dist.mean():.6f} max={dist.max():.6f}"
    )

    target_type = f"{side}_iris"
    return {
        "mp_landmark_indices": mp_iris_ids,
        "ict_lmk_face_idx": face_idx.astype(np.int64),
        "ict_lmk_b_coords": bary,
        "transfer_error": dist,
        "ict_lmk_target_type": np.array([target_type] * len(mp_iris_ids), dtype=object),
        "source": np.array(["flame_iris_center_normal_ray_occ"] * len(mp_iris_ids), dtype=object),
        "geometry_chart_id": np.full(len(mp_iris_ids), int(chart_id), dtype=np.int32),
        "ray_direction": n,
        "flame_origins": origins,
        "hit_points": hits,
    }


def run_iris_ray_to_occlusion(
    v_flame,
    f_flame,
    v_ict,
    f_ict,
    ict_npy_dict,
):
    """
    Build eye_embedding dict for ``merge_iris_into_embedding`` (10 iris landmarks).

    ``v_ict`` may be NICP-fitted face mesh; if ``None``, uses jawOpen+FLAME map from npy.
    """
    if v_ict is None:
        v_ict = ict_vertices_for_iris_bake(ict_npy_dict)
        print("  iris bake ICT mesh: jawOpen + flame_alignment (no NICP)")
    else:
        print("  iris bake ICT mesh: caller-provided vertices")

    left = bake_iris_one_side(
        "left",
        v_flame,
        f_flame,
        v_ict,
        f_ict,
        ict_npy_dict,
        flame_iris_vertex_ids=LEFT_IRIS_FLAME,
        mp_iris_ids=np.array(LEFT_IRIS_MP, dtype=np.int64),
        chart_id=CHART_LEFT_EYE,
    )
    right = bake_iris_one_side(
        "right",
        v_flame,
        f_flame,
        v_ict,
        f_ict,
        ict_npy_dict,
        flame_iris_vertex_ids=RIGHT_IRIS_FLAME,
        mp_iris_ids=np.array(RIGHT_IRIS_MP, dtype=np.int64),
        chart_id=CHART_RIGHT_EYE,
    )

    mp = np.concatenate([left["mp_landmark_indices"], right["mp_landmark_indices"]])
    face_idx = np.concatenate([left["ict_lmk_face_idx"], right["ict_lmk_face_idx"]])
    bary = np.concatenate([left["ict_lmk_b_coords"], right["ict_lmk_b_coords"]], axis=0)
    validate_lmk_face_indices(f_ict, face_idx, label="iris_ray_occlusion")

    return {
        "mp_landmark_indices": mp,
        "ict_lmk_face_idx": face_idx,
        "ict_lmk_b_coords": bary,
        "transfer_error": np.concatenate([left["transfer_error"], right["transfer_error"]]),
        "ict_lmk_target_type": np.concatenate([left["ict_lmk_target_type"], right["ict_lmk_target_type"]]),
        "source": np.concatenate([left["source"], right["source"]]),
        "geometry_chart_id": np.concatenate([left["geometry_chart_id"], right["geometry_chart_id"]]).astype(
            np.int32
        ),
        "left": left,
        "right": right,
    }
