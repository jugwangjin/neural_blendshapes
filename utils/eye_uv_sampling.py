"""Sample eye Gaussians on ``M_Sclera*`` UV disk (center included; ``M_Iris*`` annulus excluded)."""

import torch

from utils.barycentric import bary_to_uv_coords, uniform_barycentric_samples
from utils.eye_chart import build_sclera_uv_mesh, sclera_front_face_indices_torch


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


def sample_sclera_uv(ict, side, n, device, min_front_dot=-0.15):
    """
    Uniform barycentric samples on front ``M_Sclera*`` ∩ eyeball triangles.

    Uses normals vs. sclera-pole axis (``min_front_dot`` default −0.15 ≈ front hemisphere+).
    Iris annulus chart ``M_Iris*`` is excluded (empty center).
    """
    tri_ids = sclera_front_face_indices_torch(ict, side, device, min_dot=min_front_dot)
    mesh = build_sclera_uv_mesh(ict, side, device, face_idx=tri_ids)

    if tri_ids.numel() == 0:
        return torch.rand(n, 2, device=device, dtype=torch.float32)

    k_each = _distribute_sample_counts(n, tri_ids.numel(), device)
    fi = torch.repeat_interleave(tri_ids, k_each)
    n_samples = int(k_each.sum().item())
    bary = uniform_barycentric_samples(n_samples, device)
    uv = bary_to_uv_coords(fi, bary, mesh.faces_uvs, mesh.verts_uvs)
    if uv.shape[0] > n:
        uv = uv[:n]
    elif uv.shape[0] < n:
        extra = torch.rand(n - uv.shape[0], 2, device=device, dtype=uv.dtype)
        uv = torch.cat([uv, extra], dim=0)
    return uv
