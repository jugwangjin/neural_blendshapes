"""Mesh geometry helpers (torch)."""

import torch


def vertex_normals(verts, faces):
    """verts [V,3], faces [F,3] -> vn [V,3]."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=-1)
    fn_len = torch.sqrt(torch.sum(fn**2, dim=-1, keepdim=True) + 1e-12)
    fn = fn / fn_len.clamp(min=1e-8)
    vn = torch.zeros_like(verts)
    vn.index_add_(0, faces[:, 0], fn)
    vn.index_add_(0, faces[:, 1], fn)
    vn.index_add_(0, faces[:, 2], fn)
    vn_len = torch.sqrt(torch.sum(vn**2, dim=-1, keepdim=True) + 1e-12)
    return vn / vn_len.clamp(min=1e-8)


compute_vertex_normals = vertex_normals


def barycentric_3d(p, a, b, c, eps=1e-12):
    """p,a,b,c broadcast to common leading shape; returns [..., 3] barycentric."""
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = (v0 * v0).sum(dim=-1)
    d01 = (v0 * v1).sum(dim=-1)
    d11 = (v1 * v1).sum(dim=-1)
    d20 = (v2 * v0).sum(dim=-1)
    d21 = (v2 * v1).sum(dim=-1)
    denom = (d00 * d11 - d01 * d01).clamp(min=eps)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return torch.stack([u, v, w], dim=-1)


def closest_points_on_triangles(points, tri_v0, tri_v1, tri_v2):
    """
    Closest point on each triangle (plane projection + clamped bary).

    points [N, 3], tri_v* [T, 3] -> closest [N, T, 3], bary [N, T, 3], dist_sq [N, T].
    """
    n_pts = points.shape[0]
    n_tri = tri_v0.shape[0]
    p = points[:, None, :].expand(n_pts, n_tri, 3)
    a = tri_v0[None, :, :].expand(n_pts, n_tri, 3)
    b = tri_v1[None, :, :].expand(n_pts, n_tri, 3)
    c = tri_v2[None, :, :].expand(n_pts, n_tri, 3)
    w = barycentric_3d(p, a, b, c)
    w = w.clamp(min=0.0)
    w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    closest = a * w[..., 0:1] + b * w[..., 1:2] + c * w[..., 2:3]
    dist_sq = ((p - closest) ** 2).sum(dim=-1)
    return closest, w, dist_sq


def rotation_matrix_to_6d(R):
    """3×3 rotation → 6D (Zhou), inverse of ``rotation_6d_to_matrix`` (columns b1, b2)."""
    return torch.cat([R[..., 0], R[..., 1]], dim=-1)


def rotation_6d_to_matrix(r6):
    """r6: [..., 6] -> [..., 3, 3] (Zhou et al.)."""
    a1 = r6[..., 0:3]
    a2 = r6[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def apply_rigid(verts, R, t, scale=None):
    if scale is not None:
        if scale.ndim == 0:
            verts = verts * scale
        else:
            verts = verts * scale.view(-1, 1, 1)
    return verts @ R.transpose(-1, -2) + t[:, None, :]


def apply_rigid_about_centroid(verts, R, t):
    c = verts.mean(dim=1, keepdim=True)
    return (verts - c) @ R.transpose(-1, -2) + c + t[:, None, :]
