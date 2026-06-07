"""Gaussian surface regularization (semantic-weighted; no global scale/opacity-to-0.5 prior)."""

import torch

from rendering.semantic import SEMANTIC_CLASS_INDEX


def loss_opacity_sparsity(opacity):
    """SplattingAvatar ``lambda_sparsity``: mean activated opacity over all Gaussians."""
    return torch.sigmoid(opacity.reshape(-1)).mean()


def loss_opacity_uniform(opacity, target: float = 1.0):
    """Pull every Gaussian opacity toward ``target`` in [0,1] (``opacity`` = logit parameter)."""
    op = torch.sigmoid(opacity.reshape(-1))
    return (op - target).pow(2).mean()


def loss_opacity_toward_one_masked(opacity, mask, target: float = 1.0):
    """Pull opacity toward ``target`` only on masked Gaussians."""
    op = torch.sigmoid(opacity.reshape(-1))
    m = mask.reshape(-1).to(dtype=op.dtype, device=op.device)
    if m.numel() != op.numel():
        raise ValueError("mask size must match number of Gaussians")
    denom = m.sum().clamp(min=1.0)
    return ((op - target).pow(2) * m).sum() / denom


def loss_opacity_toward_one(
    opacity,
    sem_features,
    target: float = 1.0,
    w_skin: float = 1.0,
    w_other: float = 0.05,
):
    """
    Pull opacity toward ``target`` (default 1).
    Strong on **others** (face/head) Gaussians; weak on mouth interior / eye occlusion.
    """
    op = torch.sigmoid(opacity.reshape(-1))
    others_w = sem_features[:, SEMANTIC_CLASS_INDEX["others"]]
    w = w_skin * others_w + w_other * (1.0 - others_w)
    err = (op - target).pow(2)
    return (err * w).sum() / w.sum().clamp(min=1e-6)


def loss_geometry_log_scale(log_scale, max_scale: float = 0.004):
    """Penalize max axis ``exp(log_scale)`` only above ``max_scale`` (small splats free)."""
    scale = torch.exp(log_scale)
    if scale.ndim > 1:
        scale = scale.max(dim=-1).values
    excess = torch.relu(scale - max_scale)
    return excess.pow(2).mean()


def loss_scaling_regularization(
    log_scale,
    thresh_scaling_max: float = 0.008,
    thresh_scaling_ratio: float = 10.0,
):
    """
    GB-style isotropic + magnitude regularization, weighted by Gaussian size.

    Every Gaussian contributes, but small splats are down-weighted via a smooth
    size factor (no ReLU cutoff). Large / stretched Gaussians dominate the loss.
    """
    ref = max(float(thresh_scaling_max), 1e-8)
    ratio_ref = max(float(thresh_scaling_ratio) - 1.0, 1e-8)

    s = torch.exp(log_scale)
    max_vals = s.max(dim=-1).values
    min_vals = s.min(dim=-1).values.clamp(min=1e-8)
    stretch = max_vals / min_vals

    # r = max_axis / ref; size_w in [0, 1), ~0 for tiny splats, ~1 for large ones
    r = max_vals / ref
    size_w = r.pow(2) / (1.0 + r.pow(2))

    aniso = ((stretch - 1.0) / ratio_ref).pow(2) 
    mag = r.pow(2)
    return (size_w * (aniso + mag)).mean()


def _expr_coeff_1d(expr_coeff, n_channels, device, dtype):
    if expr_coeff is None:
        return torch.zeros(n_channels, device=device, dtype=dtype)
    c = expr_coeff
    if c.ndim == 2:
        c = c.mean(dim=0)
    c = c.reshape(-1).to(device=device, dtype=dtype)
    if c.shape[0] != n_channels:
        raise ValueError(f"expr_coeff length {c.shape[0]} != {n_channels} expression channels")
    return c


def color_expression_activation_contrib(color_expression, support, expr_coeff):
    """
    Per-Gaussian per-channel RGB delta actually added in forward (before sum over k):

      Δc_{n,k} = c_k · s_{n,k} · w_{n,k}   →  [N, K, 3]
    """
    c = _expr_coeff_1d(expr_coeff, color_expression.shape[1], color_expression.device, color_expression.dtype)
    return color_expression * support.unsqueeze(-1) * c.view(1, -1, 1)


def loss_color_expression_activation_sparse(color_expression, support, expr_coeff):
    """Global L1 on activated deltas Δc_{n,k} = c_k · s_{n,k} · w_{n,k}."""
    contrib = color_expression_activation_contrib(color_expression, support, expr_coeff)
    return contrib.abs().mean()


def loss_color_expression_activation_per_coeff(color_expression, support, expr_coeff):
    """
    **Per blendshape coefficient k** (채널별): when c_k is active, keep spatial footprint small.

    mean_k Σ_{n,rgb} |Δc_{n,k}| — each AU's color change should use few Gaussians, not the
    whole face. This is NOT ``few k active'' (that is tracker/deformer); it is
    ``few Gaussians per active k''.
    """
    contrib = color_expression_activation_contrib(color_expression, support, expr_coeff)
    per_k = contrib.abs().sum(dim=(0, 2))
    return per_k.mean()


def loss_color_expression_activation_per_gaussian(color_expression, support, expr_coeff):
    """
    **Per Gaussian n**: mean_n Σ_{k,rgb} |Δc_{n,k}| — each splat should not mix many AUs.

    Complements per-coeff loss (orthogonal axis: spatial sparsity across k at each point).
    """
    contrib = color_expression_activation_contrib(color_expression, support, expr_coeff)
    per_n = contrib.abs().sum(dim=(1, 2))
    return per_n.mean()


# Back-compat alias
loss_color_expression_activation_group = loss_color_expression_activation_per_coeff
