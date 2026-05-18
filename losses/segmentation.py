"""Image-space segmentation losses on gsplat-rendered semantic features."""

import torch
import torch.nn.functional as F


def loss_segmentation_logits(render_sem, target_label, ignore_index=-1):
    """
    render_sem: [B, K, H, W] composited class features (softmax probs)
    target_label: [B, H, W] int64 class ids
    """
    logits = render_sem.clamp(1e-6, 1.0).log()
    return F.cross_entropy(logits, target_label.long(), ignore_index=ignore_index)


def loss_segmentation_soft(render_sem, target_onehot):
    """
    render_sem: [B, K, H, W]
    target_onehot: [B, K, H, W]
    """
    return F.l1_loss(render_sem, target_onehot)
