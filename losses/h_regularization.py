"""
h regularization: anchor surface (legacy) or semantic class-conditioned prior.
"""

import torch

from gaussian_splatting.semantic import DEFAULT_H_SIGMA, h_prior_tensors


def charbonnier(x, eps=1e-3):
    return torch.sqrt(x * x + eps * eps)


def loss_h_semantic(h, sem_prob, class_sigma=None, class_weight=None):
    """
    h: [G, 1]
    sem_prob: [G, K] softmax over semantic classes
    class_sigma: [K] allowed |h| scale per class
    class_weight: [K] loss multiplier (accessory=0 → no prior)
    """
    if class_sigma is None or class_weight is None:
        sigma_t, weight_t = h_prior_tensors(h.device, h.dtype)
    else:
        sigma_t = torch.tensor(class_sigma, device=h.device, dtype=h.dtype)
        weight_t = torch.tensor(class_weight, device=h.device, dtype=h.dtype)

    r = h.squeeze(-1).abs().unsqueeze(-1)
    per_class = charbonnier(r / sigma_t.view(1, -1)) * weight_t.view(1, -1)
    return (sem_prob * per_class).sum(dim=-1).mean()


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
