"""Unit tests for FLARE-style masked image metrics."""

import torch

from eval.flare_image_metrics import (
    ImageMetricsAccumulator,
    apply_white_background,
    flare_image_metrics_batch,
    refine_mask_no_cloth,
)


def test_white_background():
    rgb = torch.zeros(1, 3, 4, 4)
    mask = torch.ones(1, 1, 4, 4)
    mask[:, :, 2:, :] = 0.0
    out = apply_white_background(rgb, mask)
    assert out[0, 0, 0, 0] == 0.0
    assert out[0, 0, 3, 0] == 1.0


def test_refine_mask_no_cloth():
    mask = torch.ones(1, 1, 8, 8)
    part = torch.zeros(1, 8, 8, dtype=torch.long)
    part[0, 2, 2] = 16
    m = refine_mask_no_cloth(mask, part)
    assert m[0, 0, 2, 2] == 0.0
    assert m[0, 0, 0, 0] == 1.0


def test_flare_metrics_identical_images():
    pred = torch.rand(1, 3, 64, 64)
    gt = pred.clone()
    mask = torch.ones(1, 1, 64, 64)
    stats = flare_image_metrics_batch(pred, gt, mask, no_cloth_mask=False)[0]
    assert stats["mse"] < 1e-8
    assert stats["psnr"] > 99.0
    assert stats["ssim"] > 0.99
    assert stats["lpips"] < 0.02


def test_image_metrics_accumulator_std():
    acc = ImageMetricsAccumulator()
    acc.add_frame({"psnr": 30.0, "ssim": 0.9, "lpips": 0.1, "mse": 0.01, "skipped": False})
    acc.add_frame({"psnr": 28.0, "ssim": 0.85, "lpips": 0.12, "mse": 0.02, "skipped": False})
    s = acc.summary()
    assert abs(s["psnr_mean"] - 29.0) < 1e-4
    assert s["psnr_std"] > 0.0
