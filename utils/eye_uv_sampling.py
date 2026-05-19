"""Sample eye Gaussians on ``M_Sclera*`` UV disk (center included; ``M_Iris*`` annulus excluded)."""

import math

import numpy as np
import torch

from utils.barycentric import bary_to_uv_coords, uniform_barycentric_samples
from utils.eye_chart import (
    build_sclera_uv_mesh,
    eyeball_ids_for_side,
    is_iris_material,
    sclera_forward_axis,
    sclera_sampling_face_indices_torch,
)
from utils.mesh_ops import closest_points_on_triangles


def _tangent_basis(forward, device, dtype):
    forward = forward / forward.norm().clamp(min=1e-8)
    up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
    if forward.dot(up).abs() > 0.95:
        up = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    right = torch.linalg.cross(forward, up, dim=0)
    right = right / right.norm().clamp(min=1e-8)
    up2 = torch.linalg.cross(right, forward, dim=0)
    return right, up2


def uniform_hemisphere_directions(n, forward, device, dtype=torch.float32):
    """Solid-angle uniform directions on the forward hemisphere (``forward`` = pole)."""
    forward = forward.reshape(3).to(device=device, dtype=dtype)
    forward = forward / forward.norm().clamp(min=1e-8)
    right, up = _tangent_basis(forward, device, dtype)
    phi = (2.0 * math.pi) * torch.rand(n, device=device, dtype=dtype)
    cos_theta = torch.rand(n, device=device, dtype=dtype)
    sin_theta = torch.sqrt((1.0 - cos_theta * cos_theta).clamp(min=0.0))
    dirs = cos_theta[:, None] * forward + sin_theta[:, None] * (
        torch.cos(phi)[:, None] * right + torch.sin(phi)[:, None] * up
    )
    return dirs / dirs.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def _distribute_sample_counts(n, n_tris, device):
    if n_tris == 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    if n <= n_tris:
        k = torch.zeros(n_tris, dtype=torch.long, device=device)
        k[:n] = 1
        return k
    base = n // n_tris
    rem = n % n_tris
    k = torch.full((n_tris,), base, dtype=torch.long, device=device)
    if rem > 0:
        k[:rem] += 1
    return k


def _face_material_names_np(ict):
    from utils.eye_chart import _face_material_names, _normalize_names

    return _normalize_names(_face_material_names(ict))


def _assert_sclera_sampling_tris(ict, side, tri_ids):
    names = _face_material_names_np(ict)
    fi = tri_ids.detach().cpu().numpy().astype(np.int64)
    for f in fi:
        if is_iris_material(names[int(f)]):
            raise ValueError(
                f"eye sampling includes iris material face {int(f)} ({names[int(f)]!r}); "
                f"use M_Sclera{side} hemisphere only"
            )


def _sample_sclera_uv_triangle(ict, side, n, device, min_front_dot, hemisphere_only):
    """Legacy: per-triangle uniform bary (clusters at chart pole)."""
    from utils.barycentric import bary_to_uv_coords as _b2u

    tri_ids = sclera_sampling_face_indices_torch(
        ict, side, device, min_front_dot=min_front_dot, hemisphere_only=hemisphere_only
    )
    _assert_sclera_sampling_tris(ict, side, tri_ids)
    mesh = build_sclera_uv_mesh(ict, side, device, face_idx=tri_ids)

    if tri_ids.numel() == 0:
        return torch.rand(n, 2, device=device, dtype=torch.float32)

    k_each = _distribute_sample_counts(n, tri_ids.numel(), device)
    fi = torch.repeat_interleave(tri_ids, k_each)
    n_samples = int(k_each.sum().item())
    bary = uniform_barycentric_samples(n_samples, device)
    uv = _b2u(fi, bary, mesh.faces_uvs, mesh.verts_uvs)
    if uv.shape[0] > n:
        uv = uv[:n]
    elif uv.shape[0] < n:
        extra = torch.rand(n - uv.shape[0], 2, device=device, dtype=uv.dtype)
        uv = torch.cat([uv, extra], dim=0)
    return uv


def _triangle_areas(v0, v1, v2):
    return 0.5 * torch.linalg.cross(v1 - v0, v2 - v0, dim=-1).norm(dim=-1).clamp(min=1e-12)


def sample_sclera_layout(
    ict,
    side,
    n,
    device,
    min_front_dot=0.0,
    hemisphere_only=True,
    mode="hemisphere",
):
    """
    Eye Gaussian layout: ``(uv, face_idx, bary)``.

    **3D positions must use ``face_idx`` + ``bary``** (``sample_surface``). Do not
    re-resolve atlas ``uv`` with ``uv_to_face_bary`` — misses collapse to chart center.
    """
    if mode == "triangle":
        return _sample_sclera_layout_triangle(
            ict, side, n, device, min_front_dot, hemisphere_only
        )
    if mode == "hemisphere_snap":
        return _sample_sclera_layout_hemisphere_snap(
            ict, side, n, device, min_front_dot, hemisphere_only
        )
    return _sample_sclera_layout_hemisphere(
        ict, side, n, device, min_front_dot, hemisphere_only
    )


def _layout_empty(n, device):
    zf = torch.zeros(0, dtype=torch.long, device=device)
    zb = torch.zeros(0, 3, device=device, dtype=torch.float32)
    zu = torch.zeros(0, 2, device=device, dtype=torch.float32)
    return zu, zf, zb


def _sample_sclera_layout_hemisphere(ict, side, n, device, min_front_dot, hemisphere_only):
    from utils.eye_chart import _ict_reference_verts_torch, layout_bary_to_local_uv

    tri_ids = sclera_sampling_face_indices_torch(
        ict, side, device, min_front_dot=min_front_dot, hemisphere_only=hemisphere_only
    )
    _assert_sclera_sampling_tris(ict, side, tri_ids)
    if tri_ids.numel() == 0:
        return _layout_empty(n, device)

    verts = _ict_reference_verts_torch(ict, device)
    faces = ict.faces.to(device)
    v0 = verts[faces[tri_ids, 0]]
    v1 = verts[faces[tri_ids, 1]]
    v2 = verts[faces[tri_ids, 2]]
    areas = _triangle_areas(v0, v1, v2)
    pick = torch.multinomial(areas, n, replacement=True)
    fi = tri_ids[pick]
    bary = uniform_barycentric_samples(n, device)
    uv = layout_bary_to_local_uv(ict, fi, bary, device)
    return uv, fi.long(), bary


def _sample_sclera_layout_triangle(ict, side, n, device, min_front_dot, hemisphere_only):
    from utils.eye_chart import layout_bary_to_local_uv

    tri_ids = sclera_sampling_face_indices_torch(
        ict, side, device, min_front_dot=min_front_dot, hemisphere_only=hemisphere_only
    )
    _assert_sclera_sampling_tris(ict, side, tri_ids)
    if tri_ids.numel() == 0:
        return _layout_empty(n, device)

    k_each = _distribute_sample_counts(n, tri_ids.numel(), device)
    fi = torch.repeat_interleave(tri_ids, k_each)
    bary = uniform_barycentric_samples(int(k_each.sum().item()), device)
    if fi.shape[0] > n:
        fi, bary = fi[:n], bary[:n]
    uv = layout_bary_to_local_uv(ict, fi, bary, device)
    return uv, fi.long(), bary


def _sample_sclera_layout_hemisphere_snap(ict, side, n, device, min_front_dot, hemisphere_only):
    from utils.eye_chart import _ict_reference_verts_torch

    tri_ids = sclera_sampling_face_indices_torch(
        ict, side, device, min_front_dot=min_front_dot, hemisphere_only=hemisphere_only
    )
    _assert_sclera_sampling_tris(ict, side, tri_ids)
    if tri_ids.numel() == 0:
        return _layout_empty(n, device)

    verts = _ict_reference_verts_torch(ict, device)
    faces = ict.faces.to(device)
    eye_ids = torch.tensor(eyeball_ids_for_side(ict, side), dtype=torch.long, device=device)
    center = verts[eye_ids].mean(dim=0)
    forward = torch.tensor(sclera_forward_axis(ict, side), device=device, dtype=torch.float32)
    radius = (verts[eye_ids] - center).norm(dim=-1).mean().clamp(min=1e-6)

    dirs = uniform_hemisphere_directions(n, forward, device)
    targets = center + dirs * radius

    v0 = verts[faces[tri_ids, 0]]
    v1 = verts[faces[tri_ids, 1]]
    v2 = verts[faces[tri_ids, 2]]
    _, bary_all, dist_sq = closest_points_on_triangles(targets, v0, v1, v2)
    best = dist_sq.argmin(dim=1)
    fi = tri_ids[best]
    bary = bary_all[torch.arange(n, device=device), best]
    from utils.eye_chart import layout_bary_to_local_uv

    uv = layout_bary_to_local_uv(ict, fi, bary, device)
    return uv, fi.long(), bary


def _sample_sclera_uv_hemisphere(ict, side, n, device, min_front_dot, hemisphere_only):
    uv, _, _ = _sample_sclera_layout_hemisphere(ict, side, n, device, min_front_dot, hemisphere_only)
    return uv


def _sample_sclera_uv_hemisphere_snap(ict, side, n, device, min_front_dot, hemisphere_only):
    """Legacy: uniform 3D hemisphere dirs, snap to nearest triangle (clusters if chart is small)."""
    from utils.eye_chart import _ict_reference_verts_torch

    tri_ids = sclera_sampling_face_indices_torch(
        ict, side, device, min_front_dot=min_front_dot, hemisphere_only=hemisphere_only
    )
    _assert_sclera_sampling_tris(ict, side, tri_ids)
    if tri_ids.numel() == 0:
        return torch.rand(n, 2, device=device, dtype=torch.float32)

    verts = _ict_reference_verts_torch(ict, device)
    faces = ict.faces.to(device)
    eye_ids = torch.tensor(eyeball_ids_for_side(ict, side), dtype=torch.long, device=device)
    center = verts[eye_ids].mean(dim=0)
    forward = torch.tensor(sclera_forward_axis(ict, side), device=device, dtype=torch.float32)
    radius = (verts[eye_ids] - center).norm(dim=-1).mean().clamp(min=1e-6)

    dirs = uniform_hemisphere_directions(n, forward, device)
    targets = center + dirs * radius

    v0 = verts[faces[tri_ids, 0]]
    v1 = verts[faces[tri_ids, 1]]
    v2 = verts[faces[tri_ids, 2]]
    _, bary_all, dist_sq = closest_points_on_triangles(targets, v0, v1, v2)
    best = dist_sq.argmin(dim=1)
    fi = tri_ids[best]
    bary = bary_all[torch.arange(n, device=device), best]
    return bary_to_uv_coords(fi, bary, ict.uv_faces.to(device), ict.uvs.to(device))


def sample_shared_sclera_layout(
    ict,
    n,
    device,
    min_front_dot=0.0,
    hemisphere_only=True,
    mode="hemisphere",
    mirror_right_u=True,
):
    """
    Shared chart ``uv`` from left hemisphere sample; R mesh embed = same ``uv`` (mirrored U).

    Do **not** independently resample R — that breaks index-wise correspondence.
    """
    from utils.eye_chart import embed_chart_uv_on_mesh  # noqa: PLC0415 — avoid import cycle at module load

    uv, fi_l, bary_l = sample_sclera_layout(
        ict, "L", n, device, min_front_dot, hemisphere_only, mode=mode
    )
    fi_r, bary_r = embed_chart_uv_on_mesh(
        ict,
        "R",
        uv,
        device,
        mirror_right_u=mirror_right_u,
        min_front_dot=min_front_dot,
        hemisphere_only=hemisphere_only,
    )
    return uv, fi_l, bary_l, fi_r, bary_r


def sample_shared_sclera_uv(
    ict,
    n,
    device,
    min_front_dot=0.0,
    hemisphere_only=True,
    mode="hemisphere",
):
    uv, _, _, _, _ = sample_shared_sclera_layout(
        ict, n, device, min_front_dot, hemisphere_only, mode=mode
    )
    return uv


def sample_sclera_uv(
    ict,
    side,
    n,
    device,
    min_front_dot=0.0,
    mode="hemisphere",
    hemisphere_only=True,
):
    """
    UV seeds for ``EyeTextureGaussians``.

    - ``M_Iris*`` excluded; eyeball part geometry (sclera + eyeball charts).
    - ``hemisphere_only=True``: sample on **full forward hemisphere** of eyeball mesh.
    - ``mode="hemisphere"``: area-weighted bary on those triangles (default).
    - ``mode="hemisphere_snap"``: legacy 3D dir + nearest-triangle snap.
    - ``mode="triangle"``: uniform count per triangle (pole cluster on UV disk).
    """
    uv, _, _ = sample_sclera_layout(
        ict, side, n, device, min_front_dot, hemisphere_only, mode=mode
    )
    return uv
