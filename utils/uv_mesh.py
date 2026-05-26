"""
UV surface parametrization for mesh-embedded Gaussians (SplattingAvatar-style).

Uses separate verts_uvs / faces_uvs (ICT seam-safe).
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch

from utils.barycentric import barycentric_coords_2d


@dataclass
class UVMesh:
    verts: torch.Tensor          # [V, 3]
    faces: torch.Tensor          # [F, 3] full topology
    verts_uvs: torch.Tensor      # [VT, 2]
    faces_uvs: torch.Tensor      # [F, 3] indices into verts_uvs
    active_face_idx: torch.Tensor = None  # subset of faces for this texture space

    @classmethod
    def from_ict_facekit(cls, ict, device="cpu"):
        return cls(
            verts=ict.neutral_mesh[0].to(device),
            faces=ict.faces.to(device),
            verts_uvs=ict.uvs.to(device),
            faces_uvs=ict.uv_faces.to(device),
            active_face_idx=None,
        )


def _triangle_uv_coords(uv_mesh: UVMesh):
    return uv_mesh.verts_uvs[uv_mesh.faces_uvs]


def _search_faces(uv_mesh: UVMesh):
    if uv_mesh.active_face_idx is not None:
        return uv_mesh.active_face_idx.cpu().numpy()
    return np.arange(uv_mesh.faces_uvs.shape[0])


def _barycentric_2d_batch(p, a, b, c, eps=1e-4):
    """p [G,F,2], a,b,c [1,F,2] -> w [G,F,3]."""
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = (v0 * v0).sum(-1)
    d01 = (v0 * v1).sum(-1)
    d11 = (v1 * v1).sum(-1)
    d20 = (v2 * v0).sum(-1)
    d21 = (v2 * v1).sum(-1)
    denom = (d00 * d11 - d01 * d01).clamp(min=eps * eps)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return torch.stack([u, v, w], dim=-1)


@dataclass
class ChartTriangleHit:
    """Step 1: UV point located on a chart triangle (UV-space index + barycentric)."""

    tri_idx: torch.Tensor   # [G] row into ``tri_local_uv`` / ``mesh_face_per_tri``
    bary: torch.Tensor      # [G, 3] same weights used on mesh corners


def uv_points_to_chart_triangle_bary(
    uv_points: torch.Tensor,
    tri_local_uv: torch.Tensor,
    eps: float = 1e-4,
) -> ChartTriangleHit:
    """
    Step 1 — UV coords → UV-space triangle + barycentric.

    ``tri_local_uv`` [T, 3, 2]: corner UVs of each chart triangle (e.g. ``triangle_uv_local[fi]``).
    Returns which chart triangle ``tri_idx[g]`` and barycentric weights on that triangle.
    """
    uv = uv_points.detach().float()
    device = uv.device
    tri_uv = tri_local_uv.to(device=device, dtype=uv.dtype)

    g, f = uv.shape[0], tri_uv.shape[0]
    if g == 0:
        empty_l = torch.zeros(0, dtype=torch.long, device=device)
        return ChartTriangleHit(tri_idx=empty_l, bary=torch.zeros(0, 3, dtype=uv.dtype, device=device))

    a = tri_uv[:, 0].unsqueeze(0)
    b = tri_uv[:, 1].unsqueeze(0)
    c = tri_uv[:, 2].unsqueeze(0)
    p = uv.unsqueeze(1)
    bary_all = _barycentric_2d_batch(p, a, b, c, eps=eps)
    score = bary_all.min(dim=-1).values
    tri_idx = score.argmax(dim=1)
    br_out = bary_all[torch.arange(g, device=device), tri_idx]
    br_out = br_out.clamp(min=0.0)
    br_out = br_out / br_out.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    miss = score[torch.arange(g, device=device), tri_idx] < -eps
    if miss.any():
        tri_idx_m, bary_m = _closest_uv_triangle_hit(uv[miss], tri_uv, eps=eps)
        tri_idx[miss] = tri_idx_m
        br_out[miss] = bary_m

    return ChartTriangleHit(tri_idx=tri_idx, bary=br_out)


def _closest_uv_triangle_hit(uv_points, tri_uv, eps=1e-4):
    """UV outside all triangles → nearest point on closest triangle (not chart centroid)."""
    uv = uv_points.detach().float()
    device = uv.device
    tri_uv = tri_uv.to(device=device, dtype=uv.dtype)
    g, t = uv.shape[0], tri_uv.shape[0]
    if g == 0:
        zl = torch.zeros(0, dtype=torch.long, device=device)
        return zl, torch.zeros(0, 3, dtype=uv.dtype, device=device)

    a = tri_uv[:, 0].unsqueeze(0)
    b = tri_uv[:, 1].unsqueeze(0)
    c = tri_uv[:, 2].unsqueeze(0)
    p = uv.unsqueeze(1)
    bary_all = _barycentric_2d_batch(p, a, b, c, eps=eps)
    w = bary_all.clamp(min=0.0)
    w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    closest = (
        tri_uv[:, 0].unsqueeze(0) * w[..., 0:1]
        + tri_uv[:, 1].unsqueeze(0) * w[..., 1:2]
        + tri_uv[:, 2].unsqueeze(0) * w[..., 2:3]
    )
    dist = (p - closest).pow(2).sum(-1)
    tri_idx = dist.argmin(dim=1)
    br_out = bary_all[torch.arange(g, device=device), tri_idx]
    br_out = br_out.clamp(min=0.0)
    br_out = br_out / br_out.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return tri_idx, br_out


def chart_triangle_to_mesh_face(
    tri_idx: torch.Tensor,
    mesh_face_per_tri: torch.Tensor,
) -> torch.Tensor:
    """
    Step 2 — chart triangle index → global mesh ``face_idx``.

    ``mesh_face_per_tri[t]`` is the ICT mesh face for chart triangle ``t``.
    Barycentric from step 1 is unchanged on the corresponding mesh triangle.
    """
    return mesh_face_per_tri.to(device=tri_idx.device, dtype=torch.long)[tri_idx.long()]


def chart_uv_to_mesh_face_bary(
    uv_points: torch.Tensor,
    mesh_face_per_tri: torch.Tensor,
    tri_local_uv: torch.Tensor,
    eps: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Full pipeline (chart-local UV on one material, e.g. ``M_ScleraLeft``):

      UV coords
        → (chart ``tri_idx``, ``bary``)     # UV-space face + barycentric
        → mesh ``face_idx`` via lookup table
        → same ``bary`` on mesh triangle     # 3D via ``sample_surface(verts, faces, …)``
    """
    hit = uv_points_to_chart_triangle_bary(uv_points, tri_local_uv, eps=eps)
    mesh_fi = chart_triangle_to_mesh_face(hit.tri_idx, mesh_face_per_tri)
    return mesh_fi, hit.bary


def local_uv_triangles_to_face_bary(uv_points, face_indices, tri_local_uv, eps=1e-4):
    """Alias: ``chart_uv_to_mesh_face_bary`` with per-triangle global face ids."""
    return chart_uv_to_mesh_face_bary(uv_points, face_indices, tri_local_uv, eps=eps)


def uv_to_face_bary(uv_points, uv_mesh: UVMesh, eps=1e-4):
    """
    uv_points: [G, 2] in texture space
    returns face_idx [G], bary [G, 3] indexing full face array
    """
    tri_uv_all = _triangle_uv_coords(uv_mesh)
    if uv_mesh.active_face_idx is not None:
        search_fi = uv_mesh.active_face_idx
        tri_uv = tri_uv_all[search_fi]
    else:
        search_fi = torch.arange(tri_uv_all.shape[0], device=tri_uv_all.device, dtype=torch.long)
        tri_uv = tri_uv_all
    return local_uv_triangles_to_face_bary(uv_points, search_fi, tri_uv, eps=eps)


def _lookup_face_bary(uv, uv_mesh, uvh_module, cache_key=None):
    if uvh_module is None:
        return uv_to_face_bary(uv, uv_mesh)

    key = cache_key or "default"
    cache_fi_attr = f"cached_face_idx_{key}"
    cache_br_attr = f"cached_bary_{key}"
    cached_fi = getattr(uvh_module, cache_fi_attr, None)
    cached_br = getattr(uvh_module, cache_br_attr, None)

    reproject = cached_fi is None
    if cached_fi is not None and uvh_module.training:
        step = getattr(uvh_module, "_uv_lookup_step", 0) + 1
        uvh_module._uv_lookup_step = step
        every = max(1, int(getattr(uvh_module, "reproject_uv_every", 1)))
        if step % every == 0:
            reproject = True

    if reproject:
        fi, br = uv_to_face_bary(uv, uv_mesh)
        setattr(uvh_module, cache_fi_attr, fi)
        setattr(uvh_module, cache_br_attr, br)
        return fi, br

    return cached_fi, cached_br


def surface_points_from_uvh(uv, h, uv_mesh: UVMesh, uvh_module=None, mesh_verts=None):
    """
    uv: [G, 2], h: [G, 1]
    X = S(u,v) + h * N(u,v)   (h=0 for eyeball texture spaces)
    uvh_module: optional UVHGaussians — caches face_idx/bary between steps.
    mesh_verts: optional [V,3] posed mesh (defaults to ``uv_mesh.verts``).
    """
    from utils.barycentric import sample_normals, sample_surface
    from utils.mesh_ops import compute_vertex_normals

    face_idx, bary = _lookup_face_bary(uv, uv_mesh, uvh_module, cache_key=getattr(uvh_module, "_uv_cache_key", None))
    verts = mesh_verts if mesh_verts is not None else uv_mesh.verts
    if verts.ndim == 3:
        verts = verts[0]
    v_normals = compute_vertex_normals(verts, uv_mesh.faces)
    p = sample_surface(verts, uv_mesh.faces, face_idx, bary)
    n = sample_normals(v_normals, uv_mesh.faces, face_idx, bary)
    xyz = p + h * n
    return xyz, face_idx, bary, n
