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
    op = opacity.reshape(-1)
    skin_w = sem_prob[:, SEMANTIC_CLASS_INDEX["skin"]]
    w = w_skin * skin_w + w_other * (1.0 - skin_w)
    err = (op - target).pow(2)
    return (err * w).sum() / w.sum().clamp(min=1e-6)


def loss_geometry_log_scale(
    log_scale,
    sem_prob,
    max_scale: float = 0.05,
    skin_only: bool = True,
):
    """Penalize large 3D scales; default only where rendered semantic is **skin** (face/neck skin)."""
    scale = torch.exp(log_scale).amax(dim=-1)
    if skin_only:
        w = sem_prob[:, SEMANTIC_CLASS_INDEX["skin"]]
    else:
        w = torch.ones(scale.shape[0], device=scale.device, dtype=scale.dtype)
    excess = torch.relu(scale - max_scale)
    return (excess.pow(2) * w).sum() / w.sum().clamp(min=1e-6)
