"""Thin LPIPS wrapper used only when a stage enables perceptual loss."""

import torch

# One LPIPS instance per (device, net); not re-instantiated each training step.
_LPIPS_MODELS = {}


def _lpips_model(device, net: str):
    import lpips

    key = (str(device), str(net))
    if key not in _LPIPS_MODELS:
        model = lpips.LPIPS(net=net).to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _LPIPS_MODELS[key] = model
    return _LPIPS_MODELS[key]


def loss_lpips(pred_rgb, target_rgb, mask=None, net: str = "alex"):
    """LPIPS on RGB tensors in [0,1], optionally foreground-masked."""
    pred = pred_rgb.clamp(0.0, 1.0)
    target = target_rgb.clamp(0.0, 1.0)
    if mask is not None:
        m = mask.to(device=pred.device, dtype=pred.dtype)
        if m.ndim == 2:
            m = m.unsqueeze(0).unsqueeze(0)
        elif m.ndim == 3:
            m = m.unsqueeze(1) if m.shape[0] == pred.shape[0] else m.unsqueeze(0)
        while m.ndim < pred.ndim:
            m = m.unsqueeze(0)
        pred = pred * m
        target = target * m
    pred = pred * 2.0 - 1.0
    target = target * 2.0 - 1.0
    return _lpips_model(pred.device, net)(pred, target).mean()
