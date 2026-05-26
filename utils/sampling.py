"""Fixed surface Gaussian layout from ICT npy region index arrays."""

import torch

from utils.barycentric import uniform_barycentric_samples
from utils.ict_texture_maps import bary_to_texture_chart_uv
from utils.ict_regions import (
    classify_surface_triangles_batch,
    surface_layout_triangle_ids,
)

def _k_table_tensor(
    k_mouth_interior,
    k_mouth_socket,
    k_eye_socket,
    k_head,
    k_face,
    k_eyeball_sclera,
    k_eye_occlusion,
    device,
):
    return torch.tensor(
        [
            k_mouth_interior,
            k_mouth_socket,
            k_eye_socket,
            k_head,
            k_face,
            k_eyeball_sclera,
            k_eye_occlusion,
        ],
        dtype=torch.long,
        device=device,
    )


def build_surface_gaussian_layout(
    ict,
    faces,
    k_face=8,
    k_head=8,
    k_mouth_socket=1,
    k_mouth_interior=2,
    k_eye_socket=1,
    k_eyeball_sclera=4,
    k_eye_occlusion=4,
    device=None,
):
    """
    Sample Gaussians on surface triangles (skin, sockets, sparse sclera, eye occlusion).

    Sclera / ``M_EyeOcclusion`` use ``k_*`` per face like skin; ``is_h_pin`` forces h→0 offset.
    ICT eye-occlusion blendshapes deform these triangles — no UV chart / gaze slide.

    Returns ``face_idx, bary, tri_ids, uv, is_gum, is_h_pin``.
    """
    device = device or faces.device
    tri_ids = surface_layout_triangle_ids(ict, faces, device=device)

    if tri_ids.numel() == 0:
        empty_f = torch.zeros(0, dtype=torch.long, device=device)
        empty_b = torch.zeros(0, 3, device=device, dtype=torch.float32)
        empty_uv = torch.zeros(0, 2, device=device, dtype=torch.float32)
        empty_gum = torch.zeros(0, dtype=torch.bool, device=device)
        empty_pin = torch.zeros(0, dtype=torch.bool, device=device)
        return empty_f, empty_b, tri_ids, empty_uv, empty_gum, empty_pin

    codes = classify_surface_triangles_batch(tri_ids, faces, ict, device)
    k_tab = _k_table_tensor(
        k_mouth_interior,
        k_mouth_socket,
        k_eye_socket,
        k_head,
        k_face,
        k_eyeball_sclera,
        k_eye_occlusion,
        device,
    )
    k_per_tri = k_tab[codes.clamp(min=0)]
    active = (codes >= 0) & (k_per_tri > 0)
    if not active.any():
        empty_f = torch.zeros(0, dtype=torch.long, device=device)
        empty_b = torch.zeros(0, 3, device=device, dtype=torch.float32)
        empty_uv = torch.zeros(0, 2, device=device, dtype=torch.float32)
        empty_gum = torch.zeros(0, dtype=torch.bool, device=device)
        empty_pin = torch.zeros(0, dtype=torch.bool, device=device)
        return empty_f, empty_b, tri_ids, empty_uv, empty_gum, empty_pin

    fi = tri_ids[active]
    k_each = k_per_tri[active]
    tri_codes = codes[active]
    face_idx = torch.repeat_interleave(fi, k_each)
    is_gum = torch.repeat_interleave(tri_codes == 0, k_each)
    is_h_pin = torch.repeat_interleave((tri_codes == 5) | (tri_codes == 6), k_each)
    n_samples = int(k_each.sum().item())
    bary = uniform_barycentric_samples(n_samples, device)

    if hasattr(ict, "triangle_uv_local") or (
        hasattr(ict, "uvs") and hasattr(ict, "uv_faces")
    ):
        uv = bary_to_texture_chart_uv(face_idx, bary, ict)
    else:
        uv = torch.zeros(n_samples, 2, device=device, dtype=torch.float32)

    return face_idx, bary, tri_ids, uv, is_gum, is_h_pin


def count_surface_gaussians(
    ict,
    faces,
    k_face=8,
    k_head=8,
    k_mouth_socket=1,
    k_mouth_interior=2,
    k_eye_socket=1,
    k_eyeball_sclera=4,
    k_eye_occlusion=4,
):
    device = faces.device
    tri_ids = surface_layout_triangle_ids(ict, faces, device=device)
    if tri_ids.numel() == 0:
        return 0
    codes = classify_surface_triangles_batch(tri_ids, faces, ict, device)
    k_tab = _k_table_tensor(
        k_mouth_interior,
        k_mouth_socket,
        k_eye_socket,
        k_head,
        k_face,
        k_eyeball_sclera,
        k_eye_occlusion,
        device,
    )
    k_per_tri = k_tab[codes.clamp(min=0)]
    return int(k_per_tri[codes >= 0].sum().item())
