"""
Image-space normal supervision (GT from ``processing/face_normals``).

Primary term follows ``reference_codes/normal.py``:
Laplacian of GT vs predicted normals in camera space, L1 on the difference, masked.
"""

import torch
import torch.nn.functional as F

# OpenCV-style camera / image axis fix (reference ``normal_loss``).
_R_IMAGE_FLIP = torch.tensor(
    [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
    dtype=torch.float32,
)


def _laplacian_kernel(device, dtype):
    k = torch.tensor(
        [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
        device=device,
        dtype=dtype,
    )
    return k.view(1, 1, 3, 3).repeat(3, 1, 1, 1) / 4.0


def _to_bchw(normal, device, dtype):
    if normal.ndim == 3:
        normal = normal.unsqueeze(0)
    if normal.shape[1] == 3:
        return normal.to(device=device, dtype=dtype)
    if normal.shape[-1] == 3:
        return normal.permute(0, 3, 1, 2).to(device=device, dtype=dtype)
    raise ValueError(f"normal must be [B,3,H,W] or [B,H,W,3], got {normal.shape}")


def _to_bhwc(normal, device, dtype):
    if normal.ndim == 3:
        normal = normal.unsqueeze(0)
    if normal.shape[-1] == 3:
        return normal.to(device=device, dtype=dtype)
    if normal.shape[1] == 3:
        return normal.permute(0, 2, 3, 1).to(device=device, dtype=dtype)
    raise ValueError(f"normal must be [B,3,H,W] or [B,H,W,3], got {normal.shape}")


def camera_R_batch(camera, batch_size: int, device, dtype):
    """``[B,3,3]`` with ``R.T`` per view (matches reference ``stack([c.R.T ...])``)."""
    if camera is None:
        return torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
    R = camera.R
    if not isinstance(R, torch.Tensor):
        R = torch.tensor(R, device=device, dtype=dtype)
    else:
        R = R.to(device=device, dtype=dtype)
    if R.ndim == 2:
        R = R.T.unsqueeze(0).expand(batch_size, -1, -1)
    elif R.ndim == 3:
        R = R.transpose(-1, -2)
    else:
        raise ValueError(f"camera.R must be [3,3] or [B,3,3], got {R.shape}")
    return R


def normals_world_to_camera(normals_bhwc, camera, device, dtype):
    """
    World / mesh normals ``[B,H,W,3]`` → camera space with image-axis flip.

    Same composition as ``reference_codes/normal.py`` ``normal_loss``.
    """
    n = _to_bhwc(normals_bhwc, device, dtype)
    B = n.shape[0]
    R_cam = camera_R_batch(camera, B, device, dtype)
    n = torch.einsum("bhwc,bcj->bhwj", n, R_cam)
    R_flip = _R_IMAGE_FLIP.to(device=device, dtype=dtype)
    n = torch.einsum("bhwc,cj->bhwj", n, R_flip.T)
    return n


def normal_supervision_mask(batch, render_alpha=None, device=None, dtype=None):
    """
    Foreground mask for normal loss.

    Prefer ``skin_tight`` (exclude hair/cloth); multiply render alpha when present.
  Optionally drop mouth interior + eyes from ``part_label``.
    """
    if batch is None:
        return None

    if batch.get("skin_tight") is not None:
        m = batch["skin_tight"]
    elif batch.get("full_face_region_mask") is not None:
        m = batch["full_face_region_mask"]
    elif batch.get("mask") is not None:
        m = batch["mask"]
    else:
        return None

    if m.ndim == 2:
        m = m.unsqueeze(0)
    if m.ndim == 3:
        m = m.unsqueeze(1)
    if m.shape[1] != 1:
        m = m[:, :1]

    if device is not None:
        m = m.to(device=device, dtype=dtype)

    pl = batch.get("part_label")
    if pl is not None:
        if pl.ndim == 2:
            pl = pl.unsqueeze(0)
        exclude = torch.zeros_like(m)
        for pid in (4, 5, 6, 11):
            exclude = exclude + (pl == pid).float().unsqueeze(1)
        m = m * (1.0 - exclude.clamp(0.0, 1.0))

    if render_alpha is not None:
        a = render_alpha
        if a.ndim == 2:
            a = a.unsqueeze(0)
        if a.ndim == 3:
            a = a.unsqueeze(1)
        if a.shape[1] != 1:
            a = a[:, :1]
        if device is not None:
            a = a.to(device=device, dtype=dtype)
        m = m * a

    return (m > 0).to(dtype=m.dtype)


def loss_normal_laplacian(
    pred_normal,
    gt_normal,
    camera=None,
    batch=None,
    render_alpha=None,
    supervision_mask=None,
    gt_normal_valid=None,
):
    """
    ``mean(|L(gt) - L(pred)|)`` on masked pixels (reference ``normal_loss``).

    ``gt_normal``: ``[B,3,H,W]`` in [-1, 1] (from ``load_gt_normal``).
    ``pred_normal``: world-space normals from the renderer (``[B,3,H,W]`` or ``[B,H,W,3]``).
    """
    device = gt_normal.device
    dtype = gt_normal.dtype

    gt = _to_bchw(gt_normal, device, dtype)
    if gt.max() <= 1.0 + 1e-5 and gt.min() >= 0.0 - 1e-5:
        gt = gt * 2.0 - 1.0

    pred_world = _to_bhwc(pred_normal, device, dtype)
    pred = normals_world_to_camera(pred_world, camera, device, dtype).permute(0, 3, 1, 2)

    lap_k = _laplacian_kernel(device, dtype)
    gt_lap = F.conv2d(gt, lap_k, padding=1, groups=3)
    pred_lap = F.conv2d(pred, lap_k, padding=1, groups=3)

    if supervision_mask is not None:
        mask = supervision_mask
    else:
        mask = normal_supervision_mask(batch, render_alpha, device=device, dtype=dtype)
    if mask is None:
        mask = torch.ones(gt.shape[0], 1, gt.shape[2], gt.shape[3], device=device, dtype=dtype)

    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[1] != 1:
        mask = mask[:, :1]

    if gt_normal_valid is None and batch is not None:
        gt_normal_valid = batch.get("gt_normal_valid")
    if gt_normal_valid is not None:
        v = gt_normal_valid.reshape(gt.shape[0], -1)[:, 0]
        mask = mask * v.view(gt.shape[0], 1, 1, 1)

    num_valid = mask.sum()
    if num_valid < 1:
        return torch.zeros((), device=device, dtype=dtype)

    err = torch.abs(gt_lap - pred_lap) * mask
    return err.sum() / num_valid.clamp(min=1.0)
