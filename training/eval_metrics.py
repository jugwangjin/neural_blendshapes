"""Eval-set image metrics (GT vs final render)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from dataset import build_train_dataset, collate_batch, move_batch_to_device
from losses.rgb import l1_loss, ssim


def _psnr(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8) -> float:
    mse = (pred - gt).pow(2).mean()
    return float((10.0 * torch.log10(1.0 / mse.clamp(min=eps))).item())


def _masked_tensor(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return x
    m = mask.to(device=x.device, dtype=x.dtype)
    while m.ndim < x.ndim:
        m = m.unsqueeze(1)
    return x * m


@torch.no_grad()
def _lpips_metric(pred: torch.Tensor, gt: torch.Tensor, mask=None, net: str = "alex") -> float:
    from losses.lpips_loss import loss_lpips

    return float(loss_lpips(pred, gt, mask=mask, net=net).item())


@torch.no_grad()
def compute_render_metrics(
    cfg,
    render_dir: Path,
    *,
    device: torch.device | None = None,
    max_frames: int = 0,
    lpips_net: str = "alex",
) -> dict:
    """
    Compare ``render_dir/{stem}.png`` preds against eval-set GT images.

    Returns per-frame rows + means for l1, psnr, ssim, lpips.
    """
    render_dir = Path(render_dir)
    if not render_dir.is_dir():
        raise FileNotFoundError(render_dir)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_ds = build_train_dataset(cfg, train=False)
    if len(eval_ds) == 0:
        raise RuntimeError("eval dataset is empty")

    n_cap = len(eval_ds) if max_frames <= 0 else min(len(eval_ds), max_frames)
    rows = []

    for i in range(n_cap):
        batch = move_batch_to_device(collate_batch([eval_ds[i]]), device)
        paths = batch.get("path", [f"frame_{i:05d}"])
        stem = Path(paths[0]).stem
        pred_path = render_dir / f"{stem}.png"
        if not pred_path.is_file():
            continue

        gt = batch["image"].clamp(0.0, 1.0)
        pred_bgr = cv2.imread(str(pred_path), cv2.IMREAD_UNCHANGED)
        if pred_bgr is None:
            continue
        if pred_bgr.shape[-1] == 4:
            pred_bgr = pred_bgr[:, :, :3]
        pred_rgb = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        pred = torch.from_numpy(pred_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

        mask = batch.get("mask")
        gt_m = _masked_tensor(gt, mask)
        pred_m = _masked_tensor(pred, mask)

        row = {
            "stem": stem,
            "path": str(paths[0]),
            "l1": float(l1_loss(pred_m, gt_m).item()),
            "psnr": _psnr(pred_m, gt_m),
            "ssim": float(ssim(pred_m, gt_m).item()),
            "lpips": _lpips_metric(pred_m, gt_m, mask=mask, net=lpips_net),
        }
        rows.append(row)

    if not rows:
        raise RuntimeError(f"no matched pred/gt pairs under {render_dir}")

    means = {
        "l1": float(np.mean([r["l1"] for r in rows])),
        "psnr": float(np.mean([r["psnr"] for r in rows])),
        "ssim": float(np.mean([r["ssim"] for r in rows])),
        "lpips": float(np.mean([r["lpips"] for r in rows])),
        "n_frames": len(rows),
    }
    return {"rows": rows, "means": means}


def find_final_render_dir(output_root: Path) -> Path | None:
    """Prefer ``renders/final_eval/step_* /render``; else latest stage-end render."""
    root = Path(output_root) / "renders"
    candidates = sorted(root.glob("final_eval/step_*/render"), key=lambda p: p.parent.name)
    if candidates:
        return candidates[-1]
    stage_dirs = sorted(root.glob("*/step_*/render"), key=lambda p: (p.parent.parent.name, p.parent.name))
    if stage_dirs:
        return stage_dirs[-1]
    return None


def write_metrics_report(
    sweep_root: Path,
    results: list[dict],
    *,
    csv_name: str = "sweep_metrics.csv",
    md_name: str = "sweep_report.md",
):
    sweep_root = Path(sweep_root)
    sweep_root.mkdir(parents=True, exist_ok=True)

    ranked = sorted(results, key=lambda r: (-r["means"]["psnr"], r["means"]["lpips"]))
    csv_path = sweep_root / csv_name
    fieldnames = [
        "run_id",
        "output_root",
        "n_frames",
        "l1",
        "psnr",
        "ssim",
        "lpips",
        "overrides_json",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in ranked:
            m = r["means"]
            w.writerow(
                {
                    "run_id": r["run_id"],
                    "output_root": r["output_root"],
                    "n_frames": m["n_frames"],
                    "l1": f"{m['l1']:.6f}",
                    "psnr": f"{m['psnr']:.4f}",
                    "ssim": f"{m['ssim']:.6f}",
                    "lpips": f"{m['lpips']:.6f}",
                    "overrides_json": json.dumps(r.get("overrides", {}), sort_keys=True),
                }
            )

    lines = [
        "# Sweep metrics report",
        "",
        f"Runs: {len(ranked)}",
        "",
        "| rank | run_id | PSNR ↑ | SSIM ↑ | LPIPS ↓ | L1 ↓ | frames |",
        "|------|--------|--------|--------|---------|------|--------|",
    ]
    for i, r in enumerate(ranked, start=1):
        m = r["means"]
        lines.append(
            f"| {i} | `{r['run_id']}` | {m['psnr']:.3f} | {m['ssim']:.4f} | {m['lpips']:.4f} | {m['l1']:.4f} | {m['n_frames']} |"
        )
    lines.append("")
    lines.append("## Overrides (top 5 by PSNR)")
    lines.append("")
    for r in ranked[:5]:
        lines.append(f"### `{r['run_id']}`")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(r.get("overrides", {}), indent=2, sort_keys=True))
        lines.append("```")
        lines.append("")

    md_path = sweep_root / md_name
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path
