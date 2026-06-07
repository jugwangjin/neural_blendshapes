"""
FLARE-style masked image metrics (PSNR / SSIM / LPIPS).

Ported from ``old/flare/metrics/metrics.py``:
  - white background compositing
  - optional cloth/necklace exclusion via semantic part ids 15/16
  - masked RMSE denominator over full image (H×W×C)
  - SSIM on zero-padded masked tensors
  - LPIPS on masked RGB
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from losses.rgb import ssim as _ssim_map


def _as_bchw_mask(mask: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    m = mask.to(device=pred.device, dtype=pred.dtype)
    if m.ndim == 2:
        m = m.unsqueeze(0).unsqueeze(0)
    elif m.ndim == 3:
        m = m.unsqueeze(1) if m.shape[0] == pred.shape[0] else m.unsqueeze(0)
    while m.ndim < 4:
        m = m.unsqueeze(0)
    if m.shape[1] != 1:
        m = m[:, :1]
    return m.clamp(0.0, 1.0)


def refine_mask_no_cloth(mask: torch.Tensor, part_label: torch.Tensor | None) -> torch.Tensor:
    """Zero necklace (15) and cloth (16) in mask — matches FLARE ``no_cloth_mask``."""
    if part_label is None:
        return mask
    m = mask.clone()
    sem = part_label
    if sem.ndim == 4:
        sem = sem[:, 0]
    elif sem.ndim == 3 and sem.shape[0] == m.shape[0]:
        pass
    elif sem.ndim == 2:
        sem = sem.unsqueeze(0)
    exclude = ((sem == 15) | (sem == 16)).to(dtype=m.dtype)
    if exclude.ndim == 2:
        exclude = exclude.unsqueeze(0)
    if exclude.ndim == 3 and m.ndim == 4:
        exclude = exclude.unsqueeze(1)
    return m * (1.0 - exclude)


def apply_white_background(rgb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``rgb * mask + white * (1 - mask)`` — [B,3,H,W] and [B,1,H,W]."""
    m = _as_bchw_mask(mask, rgb)
    return rgb * m + 1.0 * (1.0 - m)


def img_mse_masked(pred, gt, mask, *, error_type: str = "mse", use_mask: bool = True):
    """FLARE ``img_mse`` with ``use_mask=True`` (errors masked, mean over full image)."""
    assert pred.dim() == 4
    bsize = pred.size(0)
    if error_type == "mae":
        all_errors = (pred - gt).abs()
    else:
        all_errors = (pred - gt).square()

    m = _as_bchw_mask(mask, pred)
    if use_mask:
        nc = pred.size(1)
        nnz = torch.sum(torch.ones_like(m.reshape(bsize, -1)), 1) * nc
        all_errors = m.expand(-1, nc, -1, -1) * all_errors
        errors = all_errors.reshape(bsize, -1).sum(1) / nnz
    else:
        errors = all_errors.reshape(bsize, -1).mean(1)

    if error_type == "rmse":
        errors = errors.sqrt()
    return errors


def img_psnr_from_rmse(rmse: torch.Tensor, *, max_val: float = 1.0) -> torch.Tensor:
    eps = 1e-8
    return 20.0 * torch.log10(torch.tensor(max_val, device=rmse.device) / (rmse + eps))


@torch.no_grad()
def lpips_masked(pred, gt, mask, *, net: str = "alex"):
    from losses.lpips_loss import loss_lpips

    m = _as_bchw_mask(mask, pred)
    return loss_lpips(pred, gt, mask=m, net=net)


@torch.no_grad()
def flare_image_metrics_batch(
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    mask: torch.Tensor,
    *,
    part_label: torch.Tensor | None = None,
    no_cloth_mask: bool = True,
    use_mask: bool = True,
    lpips_net: str = "alex",
):
    """
    One batch (typically B=1) of linear RGB in [0,1].

    Returns list of per-item metric dicts.
    """
    pred = pred_rgb.clamp(0.0, 1.0)
    gt = gt_rgb.clamp(0.0, 1.0)
    m = _as_bchw_mask(mask, pred)
    if no_cloth_mask:
        m = refine_mask_no_cloth(m, part_label)

    if use_mask and float((pred * m).sum().item()) == 0.0:
        nan = float("nan")
        return [{"psnr": nan, "ssim": nan, "lpips": nan, "mse": nan, "rmse": nan, "skipped": True}]

    gt_comp = apply_white_background(gt, m)
    pred_comp = apply_white_background(pred, m) if use_mask else pred

    mse = img_mse_masked(pred_comp, gt_comp, m, error_type="mse", use_mask=use_mask)
    rmse = img_mse_masked(pred_comp, gt_comp, m, error_type="rmse", use_mask=use_mask)
    psnr = img_psnr_from_rmse(rmse)

    if use_mask:
        mb = m.bool()
        pred_ssim = pred_comp.clone()
        gt_ssim = gt_comp.clone()
        pred_ssim[~mb.expand_as(pred_ssim)] = 0.0
        gt_ssim[~mb.expand_as(gt_ssim)] = 0.0
        ssim_val = _ssim_map(pred_ssim, gt_ssim)
    else:
        ssim_val = _ssim_map(pred_comp, gt_comp)

    lp = lpips_masked(pred_comp, gt_comp, m, net=lpips_net)

    out = []
    b = pred.shape[0]
    for i in range(b):
        out.append(
            {
                "psnr": float(psnr[i].item()),
                "ssim": float(ssim_val.item() if b == 1 else ssim_val),
                "lpips": float(lp.item() if b == 1 else lp),
                "mse": float(mse[i].item()),
                "rmse": float(rmse[i].item()),
                "skipped": False,
            }
        )
    return out


class ImageMetricsAccumulator:
    """Per-run mean/std for PSNR / SSIM / LPIPS on test frames."""

    def __init__(self):
        self._psnr: list[float] = []
        self._ssim: list[float] = []
        self._lpips: list[float] = []
        self._mse: list[float] = []
        self._n_frames = 0
        self._n_skipped = 0

    def add_frame(self, stats: dict):
        self._n_frames += 1
        if stats.get("skipped") or not math.isfinite(stats.get("psnr", float("nan"))):
            self._n_skipped += 1
            return
        self._psnr.append(stats["psnr"])
        self._ssim.append(stats["ssim"])
        self._lpips.append(stats["lpips"])
        self._mse.append(stats["mse"])

    def summary(self) -> dict:
        def _agg(vals: list[float]) -> tuple[float, float]:
            if not vals:
                return float("nan"), float("nan")
            t = torch.tensor(vals, dtype=torch.float64)
            std = float(t.std(unbiased=False).item()) if t.numel() > 1 else 0.0
            return float(t.mean().item()), std

        psnr_m, psnr_s = _agg(self._psnr)
        ssim_m, ssim_s = _agg(self._ssim)
        lp_m, lp_s = _agg(self._lpips)
        mse_m, mse_s = _agg(self._mse)
        return {
            "n_frames": self._n_frames,
            "n_frames_valid": len(self._psnr),
            "n_frames_skipped": self._n_skipped,
            "psnr_mean": psnr_m,
            "psnr_std": psnr_s,
            "ssim_mean": ssim_m,
            "ssim_std": ssim_s,
            "lpips_mean": lp_m,
            "lpips_std": lp_s,
            "mse_mean": mse_m,
            "mse_std": mse_s,
        }
