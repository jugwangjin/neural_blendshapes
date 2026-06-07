"""Small helpers shared by loss orchestration."""

from contextlib import nullcontext

import torch

from rendering.pack import surface_avatar_out


def image_size_from_cfg(cfg, batch):
    from losses.cfg_access import get_loss_weight

    sz = get_loss_weight(cfg, "image_size", None)
    if sz is not None:
        return int(sz)
    img = batch.get("image")
    if img is not None and img.ndim >= 3:
        return int(img.shape[-1])
    return 512


def align_batch_device(batch, device):
    dev = torch.device(device)
    return {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def surface_sem_features(avatar_out):
    surf = avatar_out.get("surface")
    if surf is None:
        return None
    return surf.get("sem_features")


def loss_section(timer, name):
    if timer is None:
        return nullcontext()
    return timer.section(f"loss/{name}")


# Re-export for callers that used surface_avatar_out via train_losses
__all__ = [
    "align_batch_device",
    "image_size_from_cfg",
    "loss_section",
    "surface_avatar_out",
    "surface_sem_features",
]
