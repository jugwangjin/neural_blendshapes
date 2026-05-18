"""ICT expression blendshape support masks (GaussianBlendshapes-style consistency)."""

import torch

from utils.smoothstep import smoothstep


def dilate_vertex_mask(mask, faces, n_ring=2):
    """Max-pool mask over mesh adjacency (triangle faces)."""
    v = mask.clone()
    for _ in range(n_ring):
        pooled = v.clone()
        for f in faces:
            m = v[f].max()
            pooled[f[0]] = torch.maximum(pooled[f[0]], m)
            pooled[f[1]] = torch.maximum(pooled[f[1]], m)
            pooled[f[2]] = torch.maximum(pooled[f[2]], m)
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
    dilated = []
    for e in range(support.shape[0]):
        dilated.append(dilate_vertex_mask(support[e], faces, n_ring=dilate_rings))
    support = torch.stack(dilated, dim=0)
    return mag, support


def build_mp_gates(ict, region_weight, mag_e, support_e):
    """
    Map ICT expression support to MediaPipe AU indices.

    gate: [J, V] = support[ict_j] * region_weight
    mag: [J, V] = mag[ict_j]
    """
    mp_to_ict = torch.tensor(ict.mediapipe_to_ict, dtype=torch.long)
    j_count = len(mp_to_ict)
    v = region_weight.shape[0]
    gate = torch.zeros(j_count, v, dtype=region_weight.dtype)
    mag = torch.zeros(j_count, v, dtype=mag_e.dtype)
    for j, ei in enumerate(mp_to_ict):
        ei = int(ei)
        gate[j] = support_e[ei] * region_weight
        mag[j] = mag_e[ei]
    return gate, mag, mp_to_ict
