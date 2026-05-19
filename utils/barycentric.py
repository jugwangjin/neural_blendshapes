"""Barycentric sampling on triangle meshes (torch + numpy)."""

import numpy as np
import torch


def sample_barycentric_numpy(vertices, faces, face_idx, bary):
    tri = faces[face_idx]
    pts = vertices[tri]
    return (pts * bary[:, :, None]).sum(axis=1)


def vertices2landmarks_barycentric(vertices, faces, face_idx, bary):
    """Alias for vertices2landmarks (barycentric ICT surface sampling)."""
    return vertices2landmarks(vertices, faces, face_idx, bary)


def vertices2landmarks(vertices, faces, lmk_face_idx, lmk_bary_coords):
    """
    vertices: [B, V, 3]
    faces: [F, 3]
    lmk_face_idx: [L]
    lmk_bary_coords: [L, 3]
    returns: [B, L, 3]
    """
    tri = faces[lmk_face_idx]
    lmk_vertices = vertices[:, tri]
    return (lmk_vertices * lmk_bary_coords[None, :, :, None]).sum(dim=2)


def barycentric_coords_2d(p, a, b, c):
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.array([u, v, w], dtype=np.float64)


def sample_surface(vertices, faces, face_idx, bary):
    """vertices [V,3] or [B,V,3]; face_idx [N]; bary [N,3]."""
    if vertices.ndim == 2:
        tri = faces[face_idx]
        pts = vertices[tri]
        return (pts * bary[:, :, None]).sum(dim=1)
    tri = faces[face_idx]
    pts = vertices[:, tri]
    return (pts * bary[None, :, :, None]).sum(dim=2)


def sample_normals(vertex_normals, faces, face_idx, bary):
    return sample_surface(vertex_normals, faces, face_idx, bary)


def uniform_barycentric_samples(n, device, dtype=torch.float32):
    """Area-uniform random barycentric weights on a triangle, shape [n, 3]."""
    r1 = torch.rand(n, device=device, dtype=dtype)
    r2 = torch.rand(n, device=device, dtype=dtype)
    s = torch.sqrt(r1)
    b0 = 1.0 - s
    b1 = s * (1.0 - r2)
    b2 = s * r2
    return torch.stack([b0, b1, b2], dim=-1)


def bary_to_uv_coords(face_idx, bary, uv_faces, uvs):
    """Map mesh (face_idx, bary) → UV [N, 2] using per-corner vt indices."""
    fi = face_idx.long()
    tri_uv_idx = uv_faces[fi]
    tri_uv = uvs[tri_uv_idx]
    return (tri_uv * bary[:, :, None]).sum(dim=1)
