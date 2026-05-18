"""Mesh geometry helpers (torch)."""

import torch


def vertex_normals(verts, faces):
    """verts [V,3], faces [F,3] -> vn [V,3]."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=-1)
    fn = fn / fn.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    vn = torch.zeros_like(verts)
    vn.index_add_(0, faces[:, 0], fn)
    vn.index_add_(0, faces[:, 1], fn)
    vn.index_add_(0, faces[:, 2], fn)
    return vn / vn.norm(dim=-1, keepdim=True).clamp(min=1e-8)
