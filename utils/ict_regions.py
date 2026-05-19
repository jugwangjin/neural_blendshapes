"""
ICT region helpers — source of truth is assets/ict_facekit_torch.npy index arrays.

Do not hardcode official ICT README vertex ranges at runtime; use ict.face_indices,
ict.surface_sample_vertex_indices, ict.eyeball_indices, etc. from the loaded npy.
"""

import numpy as np
import torch


def _as_long_tensor(ids, device):
    if torch.is_tensor(ids):
        return ids.to(device=device, dtype=torch.long)
    return torch.tensor(list(ids), device=device, dtype=torch.long)


def _vertex_mask(n_verts, ids, device):
    m = torch.zeros(n_verts, dtype=torch.bool, device=device)
    m[_as_long_tensor(ids, device)] = True
    return m


def filter_triangles_all_vertices_in(faces, allowed_vertex_ids, device=None):
    """Triangles whose three vertices are all in allowed_vertex_ids."""
    device = device or faces.device
    if torch.is_tensor(allowed_vertex_ids) and allowed_vertex_ids.dtype == torch.bool:
        allowed = allowed_vertex_ids.to(device)
    else:
        n_verts = int(faces.max().item()) + 1
        allowed = _vertex_mask(n_verts, allowed_vertex_ids, device)
    tri = faces.long()
    return torch.where(allowed[tri].all(dim=1))[0]


def filter_triangles_exclude_vertices(faces, exclude_vertex_ids, device=None):
    device = device or faces.device
    n_verts = int(exclude_vertex_ids.max()) + 1 if len(exclude_vertex_ids) else 0
    excluded = _vertex_mask(n_verts, exclude_vertex_ids, device)
    tri = faces.long()
    keep = ~excluded[tri].any(dim=1)
    return torch.where(keep)[0]


def surface_allowed_vertices(ict):
    if hasattr(ict, "surface_sample_vertex_indices"):
        return ict.surface_sample_vertex_indices
    face = list(ict.face_indices)
    head = list(ict.not_face_indices)
    eye = set(ict.eyeball_indices)
    teeth = set(getattr(ict, "teeth_indices", []))
    out = [i for i in face + head if i not in eye and i not in teeth]
    return out


def classify_surface_triangles_batch(tri_ids, faces, ict, device):
    """
    Vectorized region tags for mesh triangle indices.

    Returns int64 codes: 0 mouth_interior, 1 mouth_socket, 2 eye_socket,
    3 head, 4 face, -1 skip (eyeball/teeth).
    """
    fi = tri_ids.long()
    tri = faces[fi]
    n_verts = int(faces.max().item()) + 1

    def any_vertex_in(ids):
        if ids is None or len(ids) == 0:
            return torch.zeros(fi.shape[0], dtype=torch.bool, device=device)
        m = _vertex_mask(n_verts, ids, device)
        return m[tri].any(dim=1)

    def all_vertices_in(ids):
        m = _vertex_mask(n_verts, ids, device)
        return m[tri].all(dim=1)

    # Teeth: no surface Gaussians (gums substitute visually); eyeball → eye UV Gaussians.
    skip = any_vertex_in(ict.eyeball_indices) | any_vertex_in(getattr(ict, "teeth_indices", []))
    gums = any_vertex_in(
        getattr(ict, "mouth_interior_vertex_indices", getattr(ict, "gums_tongue_indices", []))
    )
    mouth_sock = any_vertex_in(getattr(ict, "mouth_socket_indices", []))
    eye_sock = any_vertex_in(getattr(ict, "eye_socket_left_indices", [])) | any_vertex_in(
        getattr(ict, "eye_socket_right_indices", [])
    )
    head = all_vertices_in(ict.not_face_indices)

    code = torch.full((fi.shape[0],), 4, dtype=torch.long, device=device)
    code[head] = 3
    code[eye_sock] = 2
    code[mouth_sock] = 1
    code[gums] = 0
    code[skip] = -1
    return code


def classify_surface_triangle(fi, faces, ict, device):
    """Return 'skip' | 'mouth_interior' | 'mouth_socket' | 'eye_socket' | 'head' | 'face'."""
    tri = faces[fi].tolist()
    teeth = set(getattr(ict, "teeth_indices", []))
    gums = set(getattr(ict, "mouth_interior_vertex_indices", getattr(ict, "gums_tongue_indices", [])))
    mouth_sock = set(getattr(ict, "mouth_socket_indices", []))
    eye_sock = set(getattr(ict, "eye_socket_left_indices", [])) | set(
        getattr(ict, "eye_socket_right_indices", [])
    )
    head = set(ict.not_face_indices)
    eye = set(ict.eyeball_indices)

    if any(v in eye for v in tri):
        return "skip"
    if any(v in teeth for v in tri):
        return "skip"
    if any(v in gums for v in tri):
        return "mouth_interior"
    if any(v in mouth_sock for v in tri):
        return "mouth_socket"
    if any(v in eye_sock for v in tri):
        return "eye_socket"
    if all(v in head for v in tri):
        return "head"
    return "face"


def eyeball_left_vertices(ict):
    if hasattr(ict, "left_eyeball_indices"):
        return ict.left_eyeball_indices
    return list(range(21451, 23021))


def eyeball_right_vertices(ict):
    if hasattr(ict, "right_eyeball_indices"):
        return ict.right_eyeball_indices
    return list(range(23021, 24591))


def iris_vertices(ict, side):
    if side == "L":
        return list(getattr(ict, "left_iris_indices", []))
    return list(getattr(ict, "right_iris_indices", []))


def sclera_vertices(ict, side):
    """
    Vertices touched by ``M_Sclera*`` triangles (preferred over eyeball−iris heuristic).

    Falls back to eyeball minus iris vertex ids when ``face_material_name`` is missing.
    """
    if hasattr(ict, "face_material_name") or (
        isinstance(ict, dict) and "face_material_name" in ict
    ):
        from utils.eye_chart import sclera_face_indices

        faces = ict.faces if hasattr(ict, "faces") else ict["faces"]
        fi = sclera_face_indices(ict, side)
        if fi.size > 0:
            return np.unique(np.asarray(faces, dtype=np.int64)[fi].reshape(-1)).tolist()

    if side == "L":
        eye = list(eyeball_left_vertices(ict))
    else:
        eye = list(eyeball_right_vertices(ict))
    iris = set(iris_vertices(ict, side))
    return [v for v in eye if v not in iris]
