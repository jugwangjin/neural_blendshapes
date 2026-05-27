"""
Precomputed EDT distance fields from a fixed GT foreground mask.

Used by silhouette loss (not computed from rendered alpha):
  - ``dist_out``: 0 inside foreground, distance to fg boundary outside
  - ``dist_in``: 0 outside foreground, distance to boundary from inside (hole filling)

Compute once per frame at dataset load; training only loads tensors.
"""

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def _fg_mask_hw(mask) -> np.ndarray:
    m = mask.numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    return (m >= 0.5).astype(np.uint8)


def _resize_mask_np(fg: np.ndarray, h: int, w: int) -> np.ndarray:
    if fg.shape[0] == h and fg.shape[1] == w:
        return fg
    import cv2

    return (cv2.resize(fg.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) >= 0.5).astype(np.uint8)


def _edt_pair(fg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bg = 1 - fg
    # distance to nearest foreground (0): positive outside, 0 inside fg
    dist_out = distance_transform_edt(bg.astype(np.float64)).astype(np.float32)
    # distance to nearest background (0): positive inside fg, 0 outside
    dist_in = distance_transform_edt(fg.astype(np.float64)).astype(np.float32)
    return dist_out, dist_in


def compute_mask_distance_fields(
    mask,
    image_size: int,
    *,
    downsample: int = 4,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build ``[1, H, W]`` distance maps at ``image_size``.

    ``downsample``: EDT on ``image_size // downsample``, then bilinear upsample (1 = full res).
    """
    fg = _fg_mask_hw(mask)
    h = w = int(image_size)
    ds = max(1, int(downsample))

    if ds > 1:
        sh, sw = max(1, h // ds), max(1, w // ds)
        fg_small = _resize_mask_np(fg, sh, sw)
        dist_out, dist_in = _edt_pair(fg_small)
        import cv2

        dist_out = cv2.resize(dist_out, (w, h), interpolation=cv2.INTER_LINEAR)
        dist_in = cv2.resize(dist_in, (w, h), interpolation=cv2.INTER_LINEAR)
        fg_full = _resize_mask_np(fg, h, w)
        dist_out = dist_out * (1.0 - fg_full)
        dist_in = dist_in * fg_full
    else:
        fg_full = _resize_mask_np(fg, h, w)
        dist_out, dist_in = _edt_pair(fg_full)

    if normalize:
        scale = float(max(dist_out.max(), dist_in.max(), 1e-6))
        dist_out = dist_out / scale
        dist_in = dist_in / scale

    return (
        torch.tensor(dist_out, dtype=torch.float32).unsqueeze(0),
        torch.tensor(dist_in, dtype=torch.float32).unsqueeze(0),
    )
