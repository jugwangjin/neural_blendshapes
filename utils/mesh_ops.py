"""Lightweight mesh ops for UVH (no xatlas / nvdiffrast)."""

import torch


def compute_face_normals(vertices, faces):
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    n = torch.cross(v1 - v0, v2 - v0, dim=-1)
    return torch.nn.functional.normalize(n, dim=-1)


def compute_vertex_normals(vertices, faces):
    face_normals = compute_face_normals(vertices, faces)
    v_normals = torch.zeros_like(vertices)
    v_normals.index_add_(0, faces[:, 0], face_normals)
    v_normals.index_add_(0, faces[:, 1], face_normals)
    v_normals.index_add_(0, faces[:, 2], face_normals)
    return torch.nn.functional.normalize(v_normals, dim=-1)


def face_centers(vertices, faces):
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    return (v0 + v1 + v2) / 3.0


def build_edges(faces):
    edges = set()
    for f in faces.tolist():
        for i in range(3):
            a, b = f[i], f[(i + 1) % 3]
            edges.add((min(a, b), max(a, b)))
    return torch.tensor(list(edges), dtype=torch.long)


def laplacian_uniform(vertices, faces):
    """Combinatorial uniform Laplacian (dense)."""
    v = vertices.shape[0]
    adj = torch.zeros(v, v, device=vertices.device, dtype=vertices.dtype)
    for f in faces:
        for i in range(3):
            for j in range(3):
                if i != j:
                    adj[f[i], f[j]] = 1.0
    deg = adj.sum(dim=1).clamp(min=1.0)
    lap = torch.diag(deg) - adj
    return lap / deg.unsqueeze(1)
