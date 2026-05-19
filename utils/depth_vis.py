"""Depth map visualization for sanity / debug renders."""

import numpy as np
import torch


def depth_alpha_from_render(depth_out):
    return depth_out["depth"][0, 0], depth_out["alpha"][0, 0]


def _percentile_range(depth, mask, lo_pct, hi_pct):
    v = depth[mask]
    if v.numel() < 16:
        v = depth.reshape(-1)
    lo = torch.quantile(v.float(), lo_pct / 100.0)
    hi = torch.quantile(v.float(), hi_pct / 100.0)
    if (hi - lo).abs() < 1e-8:
        hi = lo + 1e-6
    return lo, hi


def depth_normalized(depth, alpha, lo_pct=2.0, hi_pct=98.0, alpha_thresh=0.05):
    mask = alpha > alpha_thresh
    lo, hi = _percentile_range(depth, mask, lo_pct, hi_pct)
    t = ((depth - lo) / (hi - lo)).clamp(0.0, 1.0)
    return t, mask


def turbo_rgb(t01_np):
    import matplotlib.pyplot as plt

    rgba = plt.get_cmap("turbo")(t01_np)
    return (rgba[..., :3] * 255.0).astype(np.uint8)


def depth_vis_images(depth, alpha, lo_pct=2.0, hi_pct=98.0, alpha_thresh=0.05):
    """
    Returns uint8 H×W×3 turbo colormap and H×W grayscale (background black).
    """
    t, mask = depth_normalized(depth, alpha, lo_pct, hi_pct, alpha_thresh)
    t_np = t.detach().float().cpu().numpy()
    m_np = mask.detach().cpu().numpy()

    color = turbo_rgb(t_np)
    color[~m_np] = 0

    gray = (t_np * 255.0).astype(np.uint8)
    gray[~m_np] = 0

    return color, gray


def overlay_rgb_depth(rgb_uint8, depth_color_uint8, alpha=0.55):
    """Blend RGB sanity image with depth colormap (depth weighted higher)."""
    w = float(alpha)
    return np.clip(rgb_uint8.astype(np.float32) * (1.0 - w) + depth_color_uint8.astype(np.float32) * w, 0, 255).astype(
        np.uint8
    )
