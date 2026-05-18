"""Fixed surface Gaussian layout: per-triangle barycentric samples."""

import torch

from utils.texture_spaces import PART_FACE, filter_face_indices


def bary_samples_per_triangle(k, device, dtype=torch.float32):
    r1 = torch.rand(k, device=device, dtype=dtype)
    r2 = torch.rand(k, device=device, dtype=dtype)
    s = torch.sqrt(r1)
    b0 = 1.0 - s
    b1 = s * (1.0 - r2)
    b2 = s * r2
    return torch.stack([b0, b1, b2], dim=-1)


def build_surface_gaussian_layout(faces, vertex_parts, k_per_face, device=None):
    """
    Sample K Gaussians per allowed face triangle (non-eyeball ICT parts).
    Returns face_idx [G], bary [G,3], active_face_indices [F_allowed].
    """
    device = device or faces.device
    allowed_faces = filter_face_indices(faces, vertex_parts, PART_FACE)
    if allowed_faces.numel() == 0:
        empty_f = torch.zeros(0, dtype=torch.long, device=device)
        empty_b = torch.zeros(0, 3, device=device, dtype=torch.float32)
        return empty_f, empty_b, allowed_faces

    chunks_f = []
    chunks_b = []
    for fi in allowed_faces.tolist():
        b = bary_samples_per_triangle(k_per_face, device)
        chunks_f.append(torch.full((k_per_face,), fi, dtype=torch.long, device=device))
        chunks_b.append(b)
    face_idx = torch.cat(chunks_f, dim=0)
    bary = torch.cat(chunks_b, dim=0)
    return face_idx, bary, allowed_faces


def count_surface_gaussians(faces, vertex_parts, k_per_face):
    allowed = filter_face_indices(faces, vertex_parts, PART_FACE)
    return int(allowed.numel()) * k_per_face
