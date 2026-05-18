"""
h regularization: only skin+eyeball (anchor surface) → h ≈ 0.

Non-anchor regions (hair, accessory, etc.) are not penalized.
"""

import torch


def charbonnier(x, eps=1e-3):
    return torch.sqrt(x * x + eps * eps)


def loss_h_anchor_surface(h, is_anchor_surface, eyeball_mask=None):
    """
    h: [G, 1]
    is_anchor_surface: [G] bool — face skin anchor (h→0)
    eyeball_mask: optional [G] bool — eye texture Gaussians (h already 0, skip)
    """
    if eyeball_mask is not None:
        is_anchor_surface = is_anchor_surface & (~eyeball_mask)
    mask = is_anchor_surface.float()
    if mask.sum() < 1:
        return h.sum() * 0.0
    r = h.squeeze(-1)[is_anchor_surface]
    return charbonnier(r).mean()
