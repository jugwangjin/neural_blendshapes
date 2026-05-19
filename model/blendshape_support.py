"""ICT expression blendshape support masks (GaussianBlendshapes-style consistency)."""

import torch

from utils.smoothstep import smoothstep


def dilate_vertex_mask(mask, faces, n_ring=2):
    """Max-pool mask over mesh adjacency (triangle faces)."""
    v = mask.clone()
    f = faces.long()
    for _ in range(n_ring):
        face_max = v[f].amax(dim=1)
        pooled = v.clone()
        pooled.scatter_reduce_(0, f.reshape(-1), face_max.repeat_interleave(3), reduce="amax", include_self=True)
        v = pooled
    return v


def precompute_expression_support(
    ict,
    quantile=0.95,
    support_lo=0.05,
    support_hi=0.20,
    dilate_rings=2,
):
    """
    Returns:
      mag: [E, V] ICT expression mode magnitude per vertex
      support: [E, V] soft support in [0, 1]
    """
    modes = ict.expression_shape_modes[0]
    mag = modes.norm(dim=-1)
    q = torch.quantile(mag, quantile, dim=1, keepdim=True)
    m = mag / (q + 1e-8)
    support = smoothstep(m, support_lo, support_hi)
    faces = ict.faces
    dilated = [dilate_vertex_mask(support[e], faces, n_ring=dilate_rings) for e in range(support.shape[0])]
    support = torch.stack(dilated, dim=0)
    return mag, support


def build_mp_gates(ict, region_weight, mag_e, support_e):
    """
    Map ICT expression support to MediaPipe AU indices.

    gate: [J, V] = support[ict_j] * region_weight
    mag: [J, V] = mag[ict_j]
    """
    mp_to_ict = torch.tensor(ict.mediapipe_to_ict, dtype=torch.long)
    gate = support_e[mp_to_ict] * region_weight.unsqueeze(0)
    mag = mag_e[mp_to_ict]
    return gate, mag, mp_to_ict
