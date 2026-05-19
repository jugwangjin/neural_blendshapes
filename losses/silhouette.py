"""Rendered alpha vs foreground silhouette (MP / dataset mask)."""

import torch


def loss_silhouette(render_alpha, target_mask):
    """
    L2 on composited alpha vs binary/soft foreground mask ``[1,H,W]`` or ``[B,1,H,W]``.
    """
    pred = render_alpha
    tgt = target_mask
    if pred.ndim == 4:
        pred = pred[:, :1]
    elif pred.ndim == 3:
        pred = pred.unsqueeze(0)
    if tgt.ndim == 3:
        tgt = tgt.unsqueeze(0)
    if tgt.shape[1] != 1:
        tgt = tgt[:, :1]
    if pred.shape[-2:] != tgt.shape[-2:]:
        tgt = torch.nn.functional.interpolate(
            tgt, size=pred.shape[-2:], mode="bilinear", align_corners=False
        )
    return (pred - tgt).pow(2).mean()
