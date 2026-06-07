"""
SplattingAvatar-style mesh pose for embedded Gaussians (rotation + scale).

Matches ``reference_codes/SplattingAvatar/utils/map.py``:
  - per-face ``deform_Rt @ inv(cano_Rt)`` from triangle TBN projections (Eq.4 family)
  - area-weighted per-vertex quaternion
  - barycentric interpolation of corner quats per Gaussian
  - ``q_pose = q_mesh * q_can``; ``scale = exp(log_s) * (A_pose / A_cano)``
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import matrix_to_quaternion, quaternion_multiply


def _normalize_quat_wxyz(q):
    return q / torch.sqrt(torch.sum(q * q, dim=-1, keepdim=True) + 1e-12)


def triangle_tbn(triangles: torch.Tensor) -> torch.Tensor:
    """SA ``map.tbn``: columns are X, Y, Z (tangent frame axes)."""
    a, b, c = triangles.unbind(-2)
    n = F.normalize(torch.cross(b - a, c - a, dim=-1), dim=-1)
    d = b - a
    x = F.normalize(torch.cross(d, n, dim=-1), dim=-1)
    y = F.normalize(torch.linalg.cross(d, x, dim=-1), dim=-1)
    z = F.normalize(d, dim=-1)
    return torch.stack([x, y, z], dim=-1)


def triangle_to_projection(triangles: torch.Tensor) -> torch.Tensor:
    """[B, F, 3, 3] corner positions -> [B, F, 4, 4] rigid transforms."""
    r = triangle_tbn(triangles)
    t = triangles.unbind(-2)[0]
    batch, n_face = r.shape[0], r.shape[1]
    rt = (
        torch.eye(4, device=triangles.device, dtype=triangles.dtype)
        .view(1, 1, 4, 4)
        .expand(batch, n_face, 4, 4)
        .clone()
    )
    rt[:, :, 0:3, 0:3] = r
    rt[:, :, 0:3, 3] = t
    return rt


def calc_per_face_relative_rotation(
    cano_verts: torch.Tensor,
    faces: torch.Tensor,
    mesh_verts: torch.Tensor,
) -> torch.Tensor:
    """[F, 3, 3] = R_deform @ R_cano^{-1} (SplattingAvatar ``calc_per_face_Rt``)."""
    cano_tri = cano_verts[faces].unsqueeze(0)
    def_tri = mesh_verts[faces].unsqueeze(0)
    cano_rt = triangle_to_projection(cano_tri)[0]
    def_rt = triangle_to_projection(def_tri)[0]
    rel_rt = torch.einsum("fij,fjk->fik", def_rt, torch.inverse(cano_rt))
    return rel_rt[:, :3, :3]


def calc_face_areas(mesh_verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    v = mesh_verts[faces]
    n = torch.cross(v[:, 2] - v[:, 1], v[:, 0] - v[:, 1], dim=1)
    return n.norm(dim=-1, keepdim=True) * 0.5


def per_vertex_quaternion_from_mesh(
    cano_verts: torch.Tensor,
    faces: torch.Tensor,
    cano_face_areas: torch.Tensor,
    mesh_verts: torch.Tensor,
) -> torch.Tensor:
    """SA ``PerVertQuaternion`` (non-numpy): area-weighted sum then normalize."""
    per_face_r = calc_per_face_relative_rotation(cano_verts, faces, mesh_verts)
    per_face_q = matrix_to_quaternion(per_face_r)

    n_verts = cano_verts.shape[0]
    device = mesh_verts.device
    verts_q = torch.zeros(n_verts, 4, device=device, dtype=per_face_q.dtype)
    verts_q = verts_q.index_add(0, faces[:, 0], cano_face_areas * per_face_q)
    verts_q = verts_q.index_add(0, faces[:, 1], cano_face_areas * per_face_q)
    verts_q = verts_q.index_add(0, faces[:, 2], cano_face_areas * per_face_q)
    bad = verts_q.isnan().any(dim=-1)
    if bad.any():
        verts_q = verts_q.clone()
        verts_q[bad] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=verts_q.dtype)
    return F.normalize(verts_q, eps=1e-6, dim=-1)


def barycentric_vertex_quaternion(
    per_vert_quat: torch.Tensor,
    faces: torch.Tensor,
    face_idx: torch.Tensor,
    bary: torch.Tensor,
) -> torch.Tensor:
    """SA ``base_quat``: barycentric blend of the three corner vertex quats on each triangle."""
    tri_q = per_vert_quat[faces[face_idx.long()]]
    q = torch.einsum("njk,nj->nk", tri_q, bary)
    return _normalize_quat_wxyz(q)


def face_area_scale_ratio(
    cano_face_areas: torch.Tensor,
    mesh_verts: torch.Tensor,
    faces: torch.Tensor,
    damping: float = 1e-4,
) -> torch.Tensor:
    areas = calc_face_areas(mesh_verts, faces)
    return (areas + damping) / (cano_face_areas + damping)


class MeshGaussianPoseHelper(nn.Module):
    """Caches canonical mesh geometry for SplattingAvatar-style pose maps."""

    def __init__(self, cano_verts: torch.Tensor, faces: torch.Tensor):
        super().__init__()
        self.register_buffer("cano_verts", cano_verts.detach().float())
        self.register_buffer("faces", faces.detach().long())
        areas = calc_face_areas(self.cano_verts, self.faces)
        self.register_buffer("cano_face_areas", areas)

    def forward(self, mesh_verts: torch.Tensor):
        mesh_verts = mesh_verts.reshape(-1, 3).to(
            device=self.cano_verts.device, dtype=self.cano_verts.dtype
        )
        per_vert_q = per_vertex_quaternion_from_mesh(
            self.cano_verts, self.faces, self.cano_face_areas, mesh_verts
        )
        face_scale = face_area_scale_ratio(self.cano_face_areas, mesh_verts, self.faces)
        return per_vert_q, face_scale

    def gaussian_mesh_quaternion(
        self, mesh_verts: torch.Tensor, face_idx: torch.Tensor, bary: torch.Tensor
    ) -> torch.Tensor:
        per_vert_q, _ = self.forward(mesh_verts)
        return barycentric_vertex_quaternion(per_vert_q, self.faces, face_idx, bary)

    def gaussian_scale_ratio(
        self, mesh_verts: torch.Tensor, face_idx: torch.Tensor
    ) -> torch.Tensor:
        _, face_scale = self.forward(mesh_verts)
        return face_scale[face_idx.long()]
