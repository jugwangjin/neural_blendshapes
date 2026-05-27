"""ICT expression blendshape support masks (GaussianBlendshapes-style consistency)."""

import torch

from utils.tracker import smoothstep


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
    alpha=0.1,
    dilate_rings=2,
    **kwargs,  # For backward-compatibility with unused kwargs (quantile, support_lo, support_hi)
):
    """
    Normalized Soft Mask Gating based on raw blendshape displacement norm.
    Uses the formula min( ||B_j,v|| / (max_v ||B_j,v|| * alpha), 1.0 ).
    Dilated by dilate_rings to support neighboring vertex skin sliding.
    
    Returns:
      mag: [E, V] ICT expression mode magnitude per vertex
      support: [E, V] soft support gate in [0, 1]
    """
    modes = ict.expression_shape_modes[0]
    mag = modes.norm(dim=-1) # [E, V]
    max_mag = mag.amax(dim=1, keepdim=True) # [E, 1]
    
    # min( ||B_j,v|| / (max_v ||B_j,v|| * alpha), 1.0 ) soft-clamping
    support = torch.clamp(mag / (max_mag * alpha + 1e-8), 0.0, 1.0)
    
    faces = ict.faces
    if dilate_rings > 0:
        dilated = [dilate_vertex_mask(support[e], faces, n_ring=dilate_rings) for e in range(support.shape[0])]
        support = torch.stack(dilated, dim=0)
        
    return mag, support


def build_mp_gates(ict, region_weight, mag_e, support_e, mp_to_ict):
    """
    Map ICT expression support to MediaPipe AU indices.

    gate: [J, V] = support[ict_j] * region_weight (per-vertex region gate)
    mag: [J, V] = mag[ict_j]
    """
    gate = support_e[mp_to_ict] * region_weight.unsqueeze(0)
    mag = mag_e[mp_to_ict]
    return gate, mag
