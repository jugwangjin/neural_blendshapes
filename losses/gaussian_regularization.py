"""Gaussian surface regularization (semantic-weighted; no global scale/opacity-to-0.5 prior)."""

import torch

from rendering.semantic import SEMANTIC_CLASS_INDEX


def loss_opacity_toward_one(
    opacity,
    sem_prob,
    target: float = 1.0,
    w_skin: float = 1.0,
    w_other: float = 0.05,
):
    """
    Pull opacity toward ``target`` (default 1).
    Strong on **skin** Gaussians; weak elsewhere (hair, accessory, bg).
    """
    op = torch.sigmoid(opacity.reshape(-1))
    skin_w = sem_prob[:, SEMANTIC_CLASS_INDEX["skin"]]
    w = w_skin * skin_w + w_other * (1.0 - skin_w)
    err = (op - target).pow(2)
    return (err * w).sum() / w.sum().clamp(min=1e-6)


def loss_geometry_log_scale(log_scale, max_scale: float = 0.008):
    """Penalize ``exp(log_scale)`` above ``max_scale`` (all surface Gaussians)."""
    scale = torch.exp(log_scale).amax(dim=-1)
    excess = torch.relu(scale - max_scale)
    return excess.pow(2).mean()


def loss_scaling_regularization(
    log_scale,
    thresh_scaling_max: float = 0.008,
    thresh_scaling_ratio: float = 10.0,
):
    """
    Reference-style scaling regularization:
    penalize Gaussians that are both too large and too anisotropic.
    """
    s = torch.exp(log_scale)
    max_vals = s.max(dim=-1).values
    min_vals = s.min(dim=-1).values.clamp(min=1e-8)
    ratio = max_vals / min_vals
    bad = (max_vals > thresh_scaling_max) & (ratio > thresh_scaling_ratio)
    if bad.any():
        return max_vals[bad].mean()
    return torch.zeros((), device=log_scale.device, dtype=log_scale.dtype)
