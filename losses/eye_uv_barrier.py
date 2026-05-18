"""Soft penalty when UV slides outside [0, 1] texture space."""

import torch


def soft_uv_box_barrier(uv, margin=0.0):
    below = torch.relu(margin - uv)
    above = torch.relu(uv - (1.0 - margin))
    return (below + above).pow(2).mean()
