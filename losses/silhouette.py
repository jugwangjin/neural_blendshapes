"""Rendered alpha vs foreground silhouette (binary L2 or precomputed EDT fields)."""

import torch


def _alpha_nchw(alpha, channels=1):
    if alpha.ndim == 4:
        a = alpha[:, :channels]
    elif alpha.ndim == 3:
        a = alpha.unsqueeze(0)
    else:
        raise ValueError(f"expected alpha [B,1,H,W] or [B,H,W], got {alpha.shape}")
    return a


def _resize_to_pred(field, pred_shape):
    if field.shape[-2:] == pred_shape[-2:]:
        return field
    return torch.nn.functional.interpolate(
        field, size=pred_shape[-2:], mode="bilinear", align_corners=False
    )


def loss_silhouette(render_alpha, target_mask, *, use_l1: bool = False):
    """Alpha vs tight matting mask. L2 (GB) or L1 (sharper α at boundaries, smaller grad when far)."""
    pred = _alpha_nchw(render_alpha)
    tgt = target_mask
    if tgt.ndim == 3:
        tgt = tgt.unsqueeze(0)
    if tgt.shape[1] != 1:
        tgt = tgt[:, :1]
    if pred.shape[-2:] != tgt.shape[-2:]:
        tgt = _resize_to_pred(tgt, pred.shape)
    diff = pred - tgt
    if use_l1:
        return diff.abs().mean()
    return diff.pow(2).mean()


def silhouette_edt_distance_fields(batch, cfg, render_alpha):
    """``(dist_out, dist_in)`` at the same resolution as ``render_alpha`` (dataset cache)."""
    alpha = _alpha_nchw(render_alpha)
    h, w = int(alpha.shape[-2]), int(alpha.shape[-1])
    ds = int(getattr(cfg, "mask_edt_downsample", 4))
    normalize = bool(getattr(cfg, "mask_edt_normalize", False))

    d_out = batch.get("mask_dist_out")
    d_in = batch.get("mask_dist_in")
    if d_out is None or d_in is None:
        return None, None
    if d_out.ndim == 3:
        d_out = d_out.unsqueeze(0)
    if d_in.ndim == 3:
        d_in = d_in.unsqueeze(0)
    if d_out.shape[-2:] != (h, w):
        d_out = _resize_to_pred(d_out.to(device=alpha.device, dtype=alpha.dtype), alpha.shape)
        d_in = _resize_to_pred(d_in.to(device=alpha.device, dtype=alpha.dtype), alpha.shape)
    return d_out, d_in


def loss_silhouette_edt(
    render_alpha,
    mask_dist_out,
    mask_dist_in,
    *,
    w_ext: float = 1.0,
    w_int: float = 1.0,
    max_dist_px: float = 50.0,
):
    """
    Truncated GT distance fields × rendered alpha (no EDT on ``render_alpha``).

    ``mask_dist_*``: ``[B,1,H,W]`` pixel EDT from ``dataset.mask_distance_cache``.
    """
    alpha = _alpha_nchw(render_alpha)
    d_out = mask_dist_out
    d_in = mask_dist_in
    if d_out.ndim == 3:
        d_out = d_out.unsqueeze(0)
    if d_in.ndim == 3:
        d_in = d_in.unsqueeze(0)
    if d_out.shape[1] != 1:
        d_out = d_out[:, :1]
    if d_in.shape[1] != 1:
        d_in = d_in[:, :1]
    d_out = _resize_to_pred(d_out.to(device=alpha.device, dtype=alpha.dtype), alpha.shape)
    d_in = _resize_to_pred(d_in.to(device=alpha.device, dtype=alpha.dtype), alpha.shape)

    cap = max(float(max_dist_px), 1e-6)
    d_out_norm = torch.clamp(d_out, 0.0, cap) / cap
    d_in_norm = torch.clamp(d_in, 0.0, cap) / cap

    loss_ext = (alpha * d_out_norm).mean()
    loss_int = ((1.0 - alpha).pow(2) * d_in_norm).mean()
    return w_ext * loss_ext + w_int * loss_int
