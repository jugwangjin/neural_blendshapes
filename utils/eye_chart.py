"""
ICT eye charts (OBJ ``usemtl``).

**Train:** ``utils/ict_regions`` + ``utils/sampling`` (sclera/occlusion surface Gaussians).

**Legacy:** ``legacy/eye_uv_slide`` — ``embed_chart_uv_on_mesh``, ``build_sclera_uv_mesh``, ``TextureSpaceMeshes``.

- ``M_ScleraLeft`` / ``M_ScleraRight``: sclera disk chart
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
EYEBALL_MAT = {"L": "M_EyeballLeft", "R": "M_EyeballRight"}
IRIS_MAT = {"L": "M_IrisLeft", "R": "M_IrisRight"}
IRIS_MAT_NAMES = frozenset({IRIS_MAT["L"], IRIS_MAT["R"]})


def _tensor_to_numpy(x, dtype=None):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)
    if dtype is not None:
        return x.astype(dtype, copy=False)
    return x


def _ict_attr(ict, name):
    if hasattr(ict, name):
        return getattr(ict, name)
    return ict[name]


def _ict_faces_np(ict):
    return _tensor_to_numpy(_ict_attr(ict, "faces"), np.int64)


def _ict_reference_verts_np(ict):
    """Render-space reference verts (``canonical`` if set, else raw ``neutral_mesh``)."""
    if hasattr(ict, "canonical") and _ict_attr(ict, "canonical") is not None:
        return _tensor_to_numpy(_ict_attr(ict, "canonical")[0], np.float64)
    return _tensor_to_numpy(_ict_attr(ict, "neutral_mesh")[0], np.float64)


def _ict_reference_verts_torch(ict, device):
    if hasattr(ict, "canonical") and _ict_attr(ict, "canonical") is not None:
        return _ict_attr(ict, "canonical")[0].to(device)
    return _ict_attr(ict, "neutral_mesh")[0].to(device)


def _face_material_names(ict):
    if hasattr(ict, "face_material_name"):
        return _tensor_to_numpy(_ict_attr(ict, "face_material_name"), object)
    if isinstance(ict, dict):
        return np.asarray(ict["face_material_name"], dtype=object)
    raise AttributeError("ict needs face_material_name from ict_facekit_torch.npy")


def _normalize_names(names):
    return np.array([normalize_material_name(str(x)) for x in names], dtype=object)


def is_iris_material(name):
    """``M_Iris*`` chart / material — excluded from eye Gaussian UV and sclera UVMesh."""
    n = normalize_material_name(str(name))
    if n in IRIS_MAT_NAMES:
        return True
    low = n.lower()
    return "iris" in low and "sclera" not in low


def is_sclera_material(name, side):
    return normalize_material_name(str(name)) == SCLERA_MAT[side]


def is_eyeball_material(name, side):
    return normalize_material_name(str(name)) == EYEBALL_MAT[side]


def eye_texture_material_mask(ict, side):
    """``M_Sclera*`` or ``M_Eyeball*`` (never ``M_Iris*``)."""
    names = _normalize_names(_face_material_names(ict))
    out = np.zeros(len(names), dtype=bool)
    for i, nm in enumerate(names):
        if is_iris_material(nm):
            continue
        if is_sclera_material(nm, side) or is_eyeball_material(nm, side):
            out[i] = True
    return out


def sclera_face_mask(ict, side):
    """Boolean [F] — ``M_Sclera*`` only (never ``M_Iris*`` annulus chart)."""
    names = _normalize_names(_face_material_names(ict))
    return np.array([is_sclera_material(n, side) for n in names], dtype=bool)


def iris_face_mask(ict, side):
    names = _normalize_names(_face_material_names(ict))
    return np.array([is_iris_material(n) for n in names], dtype=bool)


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
    verts = _ict_reference_verts_np(ict)
    center = verts[eye_ids].mean(axis=0)
    fi, bary = sclera_chart_center_bary(ict, side)
    faces = _ict_faces_np(ict)
    tri = verts[faces[fi]]
    pole = (tri * bary[:, None]).sum(axis=0)
    fwd = pole - center
    ln = np.linalg.norm(fwd)
    if ln < 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return (fwd / ln).astype(np.float64)


def eyeball_chart_face_mask(ict, side, faces, eyeball_vertex_ids, n_verts):
    """All eyeball-part triangles except ``M_Iris*`` annulus chart."""
    return eyeball_face_mask(faces, eyeball_vertex_ids, n_verts) & ~iris_face_mask(ict, side)


def eyeball_front_hemisphere_face_indices(ict, side, min_dot=0.0):
    """
    Eyeball triangles on the forward hemisphere (about ``sclera_forward_axis``).

    Uses vertex directions from eyeball center: keep tri if any corner has
    ``dot(normalize(v - center), forward) >= min_dot``. ``min_dot=0`` = closed
    front hemisphere (not the small ``M_Sclera`` UV disk cap).
    """
    faces = _ict_faces_np(ict)
    n_verts = int(faces.max()) + 1
    eye_ids = eyeball_ids_for_side(ict, side)
    base = eyeball_chart_face_mask(ict, side, faces, eye_ids, n_verts)
    fi = np.where(base)[0]
    if fi.size == 0:
        return sclera_face_indices(ict, side)

    verts = _ict_reference_verts_np(ict)
    center = verts[np.asarray(eye_ids, dtype=np.int64)].mean(axis=0)
    forward = sclera_forward_axis(ict, side)

    tri = faces[fi]
    vc = verts[tri] - center
    vn = np.linalg.norm(vc, axis=2, keepdims=True)
    vn = np.maximum(vn, 1e-8)
    vd = (vc / vn * forward.reshape(1, 1, 3)).sum(axis=2)
    keep = vd.max(axis=1) >= float(min_dot)
    out = fi[keep]
    if out.size == 0:
        order = np.argsort(-vd.max(axis=1))
        out = fi[order[: max(1, fi.size // 4)]]
    return out.astype(np.int64)


def sclera_front_face_indices(ict, side, min_dot=-0.15):
    """Deprecated alias — use ``eyeball_front_hemisphere_face_indices``."""
    return eyeball_front_hemisphere_face_indices(ict, side, min_dot=min_dot)


def sclera_front_face_indices_torch(ict, side, device, min_dot=-0.15):
    return torch.tensor(
        sclera_front_face_indices(ict, side, min_dot=min_dot),
        dtype=torch.long,
        device=device,
    )


def sclera_sampling_face_indices(ict, side, min_front_dot=0.0, hemisphere_only=True):
    """
    Triangles for eye Gaussian UV / ``UVMesh``.

    - Geometry: eyeball part #7/#8, ``M_Iris*`` excluded.
    - ``hemisphere_only=True``: full **forward hemisphere** of the eyeball (``min_front_dot`` on
      radial dot product; default ``0`` = half-sphere toward camera pole).
    - ``hemisphere_only=False``: entire eyeball chart (front + back).
    """
    faces = _ict_faces_np(ict)
    n_verts = int(faces.max()) + 1
    eye_ids = eyeball_ids_for_side(ict, side)
    if hemisphere_only:
        return eyeball_front_hemisphere_face_indices(ict, side, min_dot=min_front_dot)
    return np.where(eyeball_chart_face_mask(ict, side, faces, eye_ids, n_verts))[0].astype(np.int64)


def sclera_sampling_face_indices_torch(ict, side, device, min_front_dot=-0.15, hemisphere_only=True):
    return torch.tensor(
        sclera_sampling_face_indices(
            ict, side, min_front_dot=min_front_dot, hemisphere_only=hemisphere_only
        ),
        dtype=torch.long,
        device=device,
    )


def eyeball_face_mask(faces, eyeball_vertex_ids, n_verts):
    vmask = np.zeros(n_verts, dtype=bool)
    vmask[np.asarray(eyeball_vertex_ids, dtype=np.int64)] = True
    return np.all(vmask[np.asarray(faces, dtype=np.int64)], axis=1)


def sclera_eyeball_face_mask(ict, side, faces, eyeball_vertex_ids, n_verts):
    mat = eye_texture_material_mask(ict, side)
    return mat & eyeball_face_mask(faces, eyeball_vertex_ids, n_verts)


def sclera_eyeball_vertex_indices(ict, side, faces, eyeball_vertex_ids, n_verts):
    """Unique vertex indices on ``M_Sclera*`` ∩ eyeball (outer disk layer, not ``M_Iris*`` annulus)."""
    mask = sclera_eyeball_face_mask(ict, side, faces, eyeball_vertex_ids, n_verts)
    if not np.any(mask):
        return np.array([], dtype=np.int64)
    vids = np.unique(faces[mask].reshape(-1)).astype(np.int64)
    eye_set = set(np.asarray(eyeball_vertex_ids, dtype=np.int64).tolist())
    return np.array([v for v in vids if int(v) in eye_set], dtype=np.int64)


def build_sclera_uv_mesh(
    ict,
    side,
    device,
    face_idx=None,
    min_front_dot=0.0,
    hemisphere_only=True,
):
    """
    ``UVMesh`` on forward eyeball hemisphere (``M_Iris*`` excluded).

    Matches ``sample_sclera_uv`` triangle set.
    """
    if face_idx is None:
        fi = sclera_sampling_face_indices_torch(
            ict,
            side,
            device,
            min_front_dot=min_front_dot,
            hemisphere_only=hemisphere_only,
        )
    else:
        fi = face_idx
    return UVMesh(
        verts=_ict_reference_verts_torch(ict, device),
        faces=_ict_attr(ict, "faces").to(device),
        verts_uvs=_ict_attr(ict, "uvs").to(device),
        faces_uvs=_ict_attr(ict, "uv_faces").to(device),
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
        local = _tensor_to_numpy(_ict_attr(ict, "triangle_uv_local"), np.float64)[fi_all]
    elif isinstance(ict, dict) and "triangle_uv_local" in ict:
        local = _tensor_to_numpy(ict["triangle_uv_local"], np.float64)[fi_all]
    else:
        faces = _ict_faces_np(ict)
        uvs = _tensor_to_numpy(_ict_attr(ict, "uvs"))
        uf = _tensor_to_numpy(_ict_attr(ict, "uv_faces"))
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
    """Chart-local UV at sclera disk center (pupil), typically near ``(0.5, 0.5)``."""
    return np.asarray([0.5, 0.5], dtype=np.float32)


def _triangle_uv_local_torch(ict, device):
    local = _ict_attr(ict, "triangle_uv_local")
    if torch.is_tensor(local):
        return local.to(device=device, dtype=torch.float32)
    return torch.tensor(local, device=device, dtype=torch.float32)


def sclera_local_triangles(ict, side, device, min_front_dot=0.0, hemisphere_only=True):
    """
    Sclera chart triangle soup for one eye.

    Returns ``mesh_face_per_tri`` [T] (global ICT face id per chart triangle) and
    ``tri_local_uv`` [T,3,2] (corner UVs in ``M_Sclera*`` chart space).
    """
    mesh_face_per_tri = sclera_sampling_face_indices_torch(
        ict, side, device, min_front_dot=min_front_dot, hemisphere_only=hemisphere_only
    )
    local = _triangle_uv_local_torch(ict, device)
    tri_local_uv = local[mesh_face_per_tri.long()]
    return mesh_face_per_tri, tri_local_uv


def sclera_local_uv_to_face_bary(
    ict,
    side,
    uv_points,
    device,
    min_front_dot=0.0,
    hemisphere_only=True,
    eps=1e-4,
):
    """
    Chart UV → mesh ``(face_idx, bary)`` for one sclera hemisphere.

    Pipeline:
      1. ``uv_points`` → chart ``tri_idx`` + ``bary`` (UV-space triangle)
      2. ``tri_idx`` → global mesh ``face_idx`` via ``mesh_face_per_tri``
      3. same ``bary`` on mesh triangle (``sample_surface``)
    """
    from utils.uv_mesh import chart_uv_to_mesh_face_bary

    mesh_face_per_tri, tri_local_uv = sclera_local_triangles(
        ict, side, device, min_front_dot=min_front_dot, hemisphere_only=hemisphere_only
    )
    return chart_uv_to_mesh_face_bary(uv_points, mesh_face_per_tri, tri_local_uv, eps=eps)


def layout_bary_to_local_uv(ict, face_idx, bary, device):
    """Mesh ``(face_idx, bary)`` → chart-local UV (``triangle_uv_local``)."""
    local = _triangle_uv_local_torch(ict, device)
    tri = local[face_idx.long()]
    return (tri * bary.unsqueeze(-1)).sum(dim=1)


def chart_uv_for_side(uv_chart, side, mirror_right_u=False):
    """Canonical chart UV → chart coords used on ``side`` (mirror U on R when enabled)."""
    uv = uv_chart
    if side == "R" and mirror_right_u:
        uv = torch.stack([1.0 - uv[:, 0], uv[:, 1]], dim=-1)
    return uv


def embed_chart_uv_on_mesh(
    ict,
    side,
    uv_chart,
    device,
    mirror_right_u=False,
    min_front_dot=0.0,
    hemisphere_only=True,
    eps=1e-4,
):
    """
    Shared chart UV [G,2] → per-side mesh ``(face_idx, bary)``.

    UV coords → (UV chart tri + bary) → mesh face lookup → same bary on 3D triangle.
    L/R use different ``mesh_face_per_tri`` tables; Gaussian index ``i`` shares ``uv[i]``.
    """
    uv_side = chart_uv_for_side(uv_chart, side, mirror_right_u=mirror_right_u)
    return sclera_local_uv_to_face_bary(
        ict,
        side,
        uv_side,
        device,
        min_front_dot=min_front_dot,
        hemisphere_only=hemisphere_only,
        eps=eps,
    )


def eyeball_ids_for_side(ict, side):
    if side == "L":
        return list(eyeball_left_vertices(ict))
    return list(eyeball_right_vertices(ict))
