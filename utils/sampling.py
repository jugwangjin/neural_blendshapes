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
    k_teeth,
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
            k_teeth,
        ],
        dtype=torch.long,
        device=device,
    )


def build_surface_gaussian_layout(
    ict,
    faces,
    k_face=4,
    k_head=4,
    k_mouth_socket=1,
    k_mouth_interior=2,
    k_teeth=1,
    k_eye_socket=1,
    k_eyeball_sclera=4,
    k_eye_occlusion=4,
    k_face_loose_factor=0.5,
    device=None,
    face_center_init=False,
):
    """
    Sample Gaussians on surface triangles (skin, sockets, sparse sclera, eye occlusion).

    Sclera / ``M_EyeOcclusion`` use ``k_*`` per face; ``is_h_pin`` tags eyeball/occlusion charts.
    ICT eye-occlusion blendshapes deform these triangles — no UV chart / gaze slide.

    Returns ``face_idx, bary, tri_ids, uv, is_gum, is_h_pin, is_teeth``.
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
        k_teeth,
        device,
    )
    k_per_tri = k_tab[codes.clamp(min=0, max=k_tab.shape[0] - 1)]
    # Policy: do not initialize sclera Gaussians.
    k_per_tri = k_per_tri.clone()
    k_per_tri[codes == 5] = 0

    # Tight-face vs loose-face sampling:
    # - tight face triangles (all vertices inside ``ict.skin_face_indices``) keep ``k_face``
    # - remaining face triangles (still code==4) get reduced sampling (default: half)
    if k_face_loose_factor is not None and float(k_face_loose_factor) < 1.0 and int(k_face) > 0:
        skin = getattr(ict, "skin_face_indices", None)
        if skin is not None and len(skin) > 0:
            n_verts = int(faces.max().item()) + 1
            skin_mask = torch.zeros(n_verts, dtype=torch.bool, device=device)
            skin_mask[torch.as_tensor(list(skin), dtype=torch.long, device=device)] = True
            tri = faces[tri_ids.long()].long()
            tight = skin_mask[tri].all(dim=1)
            is_face = codes == 4
            loose_face = is_face & (~tight)
            if loose_face.any():
                if float(k_face_loose_factor) == 0.5:
                    k_loose = max(1, int(k_face) // 2)
                else:
                    k_loose = max(1, int(round(float(k_face) * float(k_face_loose_factor))))
                k_per_tri = k_per_tri.clone()
                k_per_tri[loose_face] = int(k_loose)

    active = (codes >= 0) & (k_per_tri > 0)
    if not active.any():
        empty_f = torch.zeros(0, dtype=torch.long, device=device)
        empty_b = torch.zeros(0, 3, device=device, dtype=torch.float32)
        empty_uv = torch.zeros(0, 2, device=device, dtype=torch.float32)
        empty_gum = torch.zeros(0, dtype=torch.bool, device=device)
        empty_pin = torch.zeros(0, dtype=torch.bool, device=device)
        empty_teeth = torch.zeros(0, dtype=torch.bool, device=device)
        return empty_f, empty_b, tri_ids, empty_uv, empty_gum, empty_pin, empty_teeth

    fi = tri_ids[active]
    tri_codes = codes[active]

    if face_center_init:
        # Center-init default: 1 Gaussian per triangle.
        # Exception: eye-occlusion(code==6) keeps per-face multiplicity (k_eye_occlusion)
        # to increase initial coverage in this thin region.
        k_each = k_per_tri[active]
        repeats = torch.ones_like(k_each)
        is_eye_occ = tri_codes == 6
        repeats[is_eye_occ] = k_each[is_eye_occ].clamp(min=1)

        face_idx = torch.repeat_interleave(fi, repeats)
        is_gum = torch.repeat_interleave((tri_codes == 0) | (tri_codes == 7), repeats)
        is_teeth = torch.repeat_interleave(tri_codes == 7, repeats)
        is_h_pin = torch.repeat_interleave((tri_codes == 5) | (tri_codes == 6), repeats)
        n_samples = int(repeats.sum().item())
        bary = torch.full((n_samples, 3), 1.0 / 3.0, device=device, dtype=torch.float32)
    else:
        k_each = k_per_tri[active]
        face_idx = torch.repeat_interleave(fi, k_each)
        is_gum = torch.repeat_interleave((tri_codes == 0) | (tri_codes == 7), k_each)
        is_teeth = torch.repeat_interleave(tri_codes == 7, k_each)
        is_h_pin = torch.repeat_interleave((tri_codes == 5) | (tri_codes == 6), k_each)
        n_samples = int(k_each.sum().item())
        bary = uniform_barycentric_samples(n_samples, device)

    if hasattr(ict, "triangle_uv_local") or (
        hasattr(ict, "uvs") and hasattr(ict, "uv_faces")
    ):
        uv = bary_to_texture_chart_uv(face_idx, bary, ict)
    else:
        uv = torch.zeros(n_samples, 2, device=device, dtype=torch.float32)

    return face_idx, bary, tri_ids, uv, is_gum, is_h_pin, is_teeth


def count_surface_gaussians(
    ict,
    faces,
    k_face=4,
    k_head=4,
    k_mouth_socket=1,
    k_mouth_interior=2,
    k_teeth=1,
    k_eye_socket=1,
    k_eyeball_sclera=4,
    k_eye_occlusion=4,
    k_face_loose_factor=0.5,
    face_center_init=False,
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
        k_teeth,
        device,
    )
    k_per_tri = k_tab[codes.clamp(min=0, max=k_tab.shape[0] - 1)]
    # Policy: do not initialize sclera Gaussians.
    k_per_tri = k_per_tri.clone()
    k_per_tri[codes == 5] = 0

    if k_face_loose_factor is not None and float(k_face_loose_factor) < 1.0 and int(k_face) > 0:
        skin = getattr(ict, "skin_face_indices", None)
        if skin is not None and len(skin) > 0:
            n_verts = int(faces.max().item()) + 1
            skin_mask = torch.zeros(n_verts, dtype=torch.bool, device=device)
            skin_mask[torch.as_tensor(list(skin), dtype=torch.long, device=device)] = True
            tri = faces[tri_ids.long()].long()
            tight = skin_mask[tri].all(dim=1)
            is_face = codes == 4
            loose_face = is_face & (~tight)
            if loose_face.any():
                if float(k_face_loose_factor) == 0.5:
                    k_loose = max(1, int(k_face) // 2)
                else:
                    k_loose = max(1, int(round(float(k_face) * float(k_face_loose_factor))))
                k_per_tri = k_per_tri.clone()
                k_per_tri[loose_face] = int(k_loose)

    active = (codes >= 0) & (k_per_tri > 0)
    if face_center_init:
        return int(active.sum().item())
    return int(k_per_tri[active].sum().item())
