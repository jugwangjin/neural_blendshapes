"""
ICT eye texture charts (OBJ usemtl).

- ``M_ScleraLeft`` / ``M_ScleraRight``: filled disk in local UV — eye Gaussians + iris MP landmarks.
- ``M_IrisLeft`` / ``M_IrisRight``: annulus (empty center) — not used for UV sampling at pupil.
"""

from __future__ import annotations

import numpy as np
import torch

from processing.ict_obj_materials import normalize_material_name
from utils.barycentric import barycentric_coords_2d
from utils.ict_regions import eyeball_left_vertices, eyeball_right_vertices
from utils.uv_mesh import UVMesh

SCLERA_MAT = {"L": "M_ScleraLeft", "R": "M_ScleraRight"}
IRIS_MAT = {"L": "M_IrisLeft", "R": "M_IrisRight"}


def _face_material_names(ict):
    if hasattr(ict, "face_material_name"):
        return np.asarray(ict.face_material_name, dtype=object)
    if isinstance(ict, dict):
        return np.asarray(ict["face_material_name"], dtype=object)
    raise AttributeError("ict needs face_material_name from ict_facekit_torch.npy")


def _normalize_names(names):
    return np.array([normalize_material_name(str(x)) for x in names], dtype=object)


def sclera_face_mask(ict, side):
    """Boolean [F] — triangles on the sclera texture chart (not iris annulus)."""
    names = _normalize_names(_face_material_names(ict))
    return names == SCLERA_MAT[side]


def iris_face_mask(ict, side):
    names = _normalize_names(_face_material_names(ict))
    return names == IRIS_MAT[side]


def sclera_face_indices(ict, side):
    return np.where(sclera_face_mask(ict, side))[0].astype(np.int64)


def sclera_face_indices_torch(ict, side, device):
    return torch.tensor(sclera_face_indices(ict, side), dtype=torch.long, device=device)


def _face_normals_np(verts, faces):
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(ln, 1e-8)


def sclera_forward_axis(ict, side):
    """Unit vector from eyeball center toward visible sclera (chart center in 3D)."""
    eye_ids = np.asarray(eyeball_ids_for_side(ict, side), dtype=np.int64)
    verts = np.asarray(
        ict.neutral_mesh[0].detach().cpu().numpy()
        if hasattr(ict.neutral_mesh, "detach")
        else ict.neutral_mesh[0],
        dtype=np.float64,
    )
    center = verts[eye_ids].mean(axis=0)
    fi, bary = sclera_chart_center_bary(ict, side)
    faces = np.asarray(ict.faces if hasattr(ict, "faces") else ict["faces"], dtype=np.int64)
    tri = verts[faces[fi]]
    pole = (tri * bary[:, None]).sum(axis=0)
    fwd = pole - center
    ln = np.linalg.norm(fwd)
    if ln < 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return (fwd / ln).astype(np.float64)


def sclera_front_face_indices(ict, side, min_dot=-0.15):
    """
  ``M_Sclera*`` ∩ eyeball triangles whose normals face the visible pole.

  ``min_dot=-0.15`` is slightly wider than a strict hemisphere (front-facing views).
    """
    faces = np.asarray(ict.faces if hasattr(ict, "faces") else ict["faces"], dtype=np.int64)
    n_verts = int(faces.max()) + 1
    eye_ids = eyeball_ids_for_side(ict, side)
    base = sclera_eyeball_face_mask(ict, side, faces, eye_ids, n_verts)
    fi = np.where(base)[0]
    if fi.size == 0:
        return sclera_face_indices(ict, side)

    verts = np.asarray(
        ict.neutral_mesh[0].detach().cpu().numpy()
        if hasattr(ict.neutral_mesh, "detach")
        else ict.neutral_mesh[0],
        dtype=np.float64,
    )
    forward = sclera_forward_axis(ict, side)
    fn = _face_normals_np(verts, faces[fi])
    dot = (fn * forward).sum(axis=1)
    keep = fi[dot >= min_dot]
    if keep.size == 0:
        order = np.argsort(-dot)
        keep = fi[order[: max(1, fi.size // 4)]]
    return keep.astype(np.int64)


def sclera_front_face_indices_torch(ict, side, device, min_dot=-0.15):
    return torch.tensor(
        sclera_front_face_indices(ict, side, min_dot=min_dot),
        dtype=torch.long,
        device=device,
    )


def eyeball_face_mask(faces, eyeball_vertex_ids, n_verts):
    vmask = np.zeros(n_verts, dtype=bool)
    vmask[np.asarray(eyeball_vertex_ids, dtype=np.int64)] = True
    return np.all(vmask[np.asarray(faces, dtype=np.int64)], axis=1)


def sclera_eyeball_face_mask(ict, side, faces, eyeball_vertex_ids, n_verts):
    return sclera_face_mask(ict, side) & eyeball_face_mask(faces, eyeball_vertex_ids, n_verts)


def sclera_eyeball_vertex_indices(ict, side, faces, eyeball_vertex_ids, n_verts):
    """Unique vertex indices on ``M_Sclera*`` ∩ eyeball (outer disk layer, not ``M_Iris*`` annulus)."""
    mask = sclera_eyeball_face_mask(ict, side, faces, eyeball_vertex_ids, n_verts)
    if not np.any(mask):
        return np.array([], dtype=np.int64)
    vids = np.unique(faces[mask].reshape(-1)).astype(np.int64)
    eye_set = set(np.asarray(eyeball_vertex_ids, dtype=np.int64).tolist())
    return np.array([v for v in vids if int(v) in eye_set], dtype=np.int64)


def build_sclera_uv_mesh(ict, side, device, face_idx=None):
    """UVMesh restricted to ``M_Sclera*`` triangles (local UV disk, center valid)."""
    if face_idx is None:
        fi = sclera_face_indices_torch(ict, side, device)
    else:
        fi = face_idx
    return UVMesh(
        verts=ict.neutral_mesh[0].to(device),
        faces=ict.faces.to(device),
        verts_uvs=ict.uvs.to(device),
        faces_uvs=ict.uv_faces.to(device),
        active_face_idx=fi,
    )


def sclera_chart_center_bary(ict, side, uv_target=(0.5, 0.5)):
    """
    Face + bary on ``M_Sclera*`` whose local UV contains the chart center (pupil on sclera map).
    """
    fi_all = sclera_face_indices(ict, side)
    if fi_all.size == 0:
        raise ValueError(f"No {SCLERA_MAT[side]} faces in face_material_name")

    if hasattr(ict, "triangle_uv_local"):
        local = np.asarray(ict.triangle_uv_local, dtype=np.float64)[fi_all]
    elif isinstance(ict, dict) and "triangle_uv_local" in ict:
        local = np.asarray(ict["triangle_uv_local"], dtype=np.float64)[fi_all]
    else:
        faces = np.asarray(ict.faces if hasattr(ict, "faces") else ict["faces"])
        uvs = np.asarray(ict.uvs if hasattr(ict, "uvs") else ict["uvs"])
        uf = np.asarray(ict.uv_faces if hasattr(ict, "uv_faces") else ict["uv_faces"])
        local = uvs[uf[fi_all]]

    target = np.asarray(uv_target, dtype=np.float64)
    centroids = local.mean(axis=1)
    j = int(np.argmin(np.linalg.norm(centroids - target, axis=1)))
    fi = int(fi_all[j])
    tri = local[j]
    bary = barycentric_coords_2d(target, tri[0], tri[1], tri[2]).astype(np.float32)
    return fi, bary


def sclera_chart_point_3d(verts, faces, ict, side, uv_target=(0.5, 0.5)):
    """3D point on ``M_Sclera*`` at chart-local UV."""
    fi, bary = sclera_chart_center_bary(ict, side, uv_target=uv_target)
    tri = np.asarray(verts, dtype=np.float64)[faces[fi]]
    return (tri * bary[:, None]).sum(axis=0).astype(np.float64)


def eyeball_back_pole_3d(verts, eyeball_vertex_ids, front_pole):
    """
    Back pole: reflect ``front_pole`` through eyeball centroid, snap to nearest eyeball vertex.

    Pairs with sclera UV (0.5,0.5) front pole for front–back rigid constraint.
    """
    eye_ids = np.asarray(eyeball_vertex_ids, dtype=np.int64)
    v = np.asarray(verts, dtype=np.float64)
    center = v[eye_ids].mean(axis=0)
    back = 2.0 * center - np.asarray(front_pole, dtype=np.float64).reshape(3)
    eye_v = v[eye_ids]
    j = int(np.argmin(np.linalg.norm(eye_v - back, axis=1)))
    return eye_v[j].astype(np.float64)


def sclera_pole_uv(ict, side):
    """UV (seam ``uvs`` space) at sclera chart center."""
    fi, bary = sclera_chart_center_bary(ict, side)
    uf = np.asarray(ict.uv_faces if hasattr(ict, "uv_faces") else ict["uv_faces"])
    uvs = np.asarray(ict.uvs if hasattr(ict, "uvs") else ict["uvs"])
    tri_uv = uvs[uf[fi]]
    return (tri_uv * bary[:, None]).sum(axis=0)


def eyeball_ids_for_side(ict, side):
    if side == "L":
        return list(eyeball_left_vertices(ict))
    return list(eyeball_right_vertices(ict))
