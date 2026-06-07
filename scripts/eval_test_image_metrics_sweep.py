#!/usr/bin/env python
"""
Test-set PSNR / SSIM / LPIPS for all completed sweep runs (4 ablations).

FLARE-style masked metrics (white background, cloth/necklace excluded).
Writes per-run JSON + per-log-root Excel summary.

Usage:
  python scripts/eval_test_image_metrics_sweep.py --ablation all --dry-run
  python scripts/eval_test_image_metrics_sweep.py --device cuda
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from tqdm import tqdm

from dataset import move_batch_to_device
from dataset.dataset_util import format_splits_label, normalize_split_names
from eval.flare_image_metrics import ImageMetricsAccumulator, flare_image_metrics_batch
from eval.tracking_eval_common import (
    build_test_loader,
    final_stage_spec,
    find_final_checkpoint,
    infer_ablation_from_output_root,
    load_final_run_stack,
    render_personalized_rgb,
)
from run_status import ABLATION_LOG_ROOTS, is_run_complete, log_root_for_ablation


def _device_from_cli(device: str | None) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def eval_run(
    output_root: Path,
    *,
    device: torch.device,
    skip_existing: bool,
    dry_run: bool,
    no_cloth_mask: bool,
    num_workers: int = 0,
    lpips_net: str = "alex",
) -> dict | None:
    if not is_run_complete(output_root):
        print(f"skip (incomplete): {output_root}")
        return None

    out_json = output_root / "test_image_metrics.json"
    if skip_existing and out_json.is_file():
        return json.loads(out_json.read_text(encoding="utf-8"))

    ckpt_path = find_final_checkpoint(output_root)
    ablation = infer_ablation_from_output_root(output_root)
    spec = final_stage_spec(output_root)
    print(f"\n=== test metrics {output_root.name} ({ablation}) ===")

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = deepcopy(payload["cfg"])
    cfg.output_root = Path(output_root)
    eval_scenes = normalize_split_names(cfg.eval_split)
    eval_label = format_splits_label(cfg.eval_split)

    if dry_run:
        loader, n = build_test_loader(cfg, num_workers=0)
        print(f"  eval_split={eval_label} frames={n}")
        return {
            "run_name": output_root.name,
            "ablation": ablation,
            "eval_split": eval_label,
            "n_frames": n,
        }

    if len(eval_scenes) == 0:
        print("  skip: empty eval_split")
        return None

    stack = load_final_run_stack(cfg, ckpt_path, device)
    loader, n_frames = build_test_loader(cfg, num_workers=num_workers)
    if n_frames == 0:
        print("  skip: no test frames")
        return None

    acc = ImageMetricsAccumulator()
    per_frame = []
    for batch in tqdm(loader, total=n_frames, desc=output_root.name):
        batch = move_batch_to_device(batch, device)
        pred = render_personalized_rgb(stack, batch, spec)
        gt = batch["image"].clamp(0.0, 1.0)
        mask = batch["mask"]
        part_label = batch.get("part_label")

        stats = flare_image_metrics_batch(
            pred,
            gt,
            mask,
            part_label=part_label,
            no_cloth_mask=no_cloth_mask,
            use_mask=True,
            lpips_net=lpips_net,
        )[0]
        acc.add_frame(stats)
        per_frame.append(
            {
                "frame_name": batch["frame_name"][0],
                "img_path": batch["img_path"][0],
                **stats,
            }
        )

    summary = acc.summary()
    result = {
        "run_name": output_root.name,
        "ablation": ablation,
        "checkpoint": str(ckpt_path),
        "eval_split": eval_label,
        "input_dir": str(cfg.input_dir),
        "no_cloth_mask": no_cloth_mask,
        "lpips_net": lpips_net,
        **summary,
        "per_frame": per_frame,
    }
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"  PSNR={summary['psnr_mean']:.2f}±{summary['psnr_std']:.2f}  "
        f"SSIM={summary['ssim_mean']:.4f}  LPIPS={summary['lpips_mean']:.4f}"
    )
    return result


def summary_row(r: dict) -> dict:
    return {
        "ablation": r["ablation"],
        "run_name": r["run_name"],
        "eval_split": r.get("eval_split", ""),
        "input_dir": r.get("input_dir", ""),
        "n_frames": r.get("n_frames", r.get("n_frames_valid", 0)),
        "n_frames_valid": r.get("n_frames_valid", 0),
        "psnr_mean": r.get("psnr_mean"),
        "psnr_std": r.get("psnr_std"),
        "ssim_mean": r.get("ssim_mean"),
        "ssim_std": r.get("ssim_std"),
        "lpips_mean": r.get("lpips_mean"),
        "lpips_std": r.get("lpips_std"),
        "mse_mean": r.get("mse_mean"),
        "mse_std": r.get("mse_std"),
    }


def write_excel(rows: list[dict], path: Path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame([summary_row(r) for r in rows])
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        for r in rows:
            sheet = r["run_name"][:31]
            frame_df = pd.DataFrame(r.get("per_frame", []))
            if not frame_df.empty:
                frame_df.to_excel(writer, sheet_name=sheet, index=False)
    print(f"Excel: {path}")


def iter_log_roots(ablation: str):
    if ablation == "all":
        for root in ABLATION_LOG_ROOTS.values():
            yield root
        return
    yield log_root_for_ablation(ablation)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ablation", default="all", choices=["all", *sorted(ABLATION_LOG_ROOTS.keys())])
    p.add_argument("--run-name", default=None)
    p.add_argument("--log-root", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-cloth-mask", action="store_true", help="Disable semantic cloth/necklace mask exclusion.")
    p.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"])
    p.add_argument(
        "--excel-path",
        default=None,
        help="Combined Excel path (default: <log_root>/test_image_metrics_summary.xlsx).",
    )
    args = p.parse_args()

    device = _device_from_cli(args.device)
    roots = [Path(args.log_root)] if args.log_root else list(iter_log_roots(args.ablation))
    total = 0

    for log_root in roots:
        if not log_root.is_dir():
            continue
        rows = []
        for run_dir in sorted(log_root.iterdir()):
            if not run_dir.is_dir():
                continue
            if args.run_name and not fnmatch.fnmatch(run_dir.name, args.run_name):
                continue
            row = eval_run(
                run_dir,
                device=device,
                skip_existing=args.skip_existing,
                dry_run=args.dry_run,
                no_cloth_mask=not args.no_cloth_mask,
                num_workers=args.num_workers,
                lpips_net=args.lpips_net,
            )
            if row and "psnr_mean" in row:
                rows.append(row)
                total += 1
        if rows and not args.dry_run:
            xlsx = Path(args.excel_path) if args.excel_path else log_root / "test_image_metrics_summary.xlsx"
            write_excel(rows, xlsx)

    print(f"\nDone: {total} run(s).")


if __name__ == "__main__":
    main()
