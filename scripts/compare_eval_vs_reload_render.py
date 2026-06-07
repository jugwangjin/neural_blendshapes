#!/usr/bin/env python
"""
Compare training eval PNG (``renders/{stage}/step_*/render/``) vs checkpoint reload.

Same layout as ``training/eval_render.py`` + ``Config.eval_render_dir`` (= ``output_root/renders``).

Usage:
  python scripts/compare_eval_vs_reload_render.py \\
    --output-root /Bean/log/gwangjin/2026/neural_blendshapes_10/bala

  # explicit: stage_3 final (default when --checkpoint omitted)
  python scripts/compare_eval_vs_reload_render.py \\
    --output-root .../bala --checkpoint stage_3_expression_detail_end_step_037500.pt --frame-stem 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import collate_batch, move_batch_to_device
from dataset.image_dataset import ImageDataset
from training.eval_render import (
    list_training_render_pred_stems,
    training_render_pred_dir,
    training_render_stage_dir,
)
from training.inference_render import render_gsplat_from_batch
from training.render_load import find_final_checkpoint, load_run_for_render


def _load_eval_pred_bgr(eval_png: Path) -> np.ndarray:
    img = cv2.imread(str(eval_png), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(eval_png)
    if img.ndim == 3 and img.shape[2] == 4:
        bgr = img[:, :, :3]
        a = img[:, :, 3:4].astype(np.float32) / 255.0
        bgr = (bgr.astype(np.float32) * a).astype(np.uint8)
    return bgr


def _render_bgr(stack, batch, *, composite: bool = True) -> np.ndarray:
    gs = render_gsplat_from_batch(
        stack.avatar,
        stack.renderer,
        stack.camera,
        stack.tracker,
        batch,
        stack.spec,
        composite=composite,
    )
    from dataset.dataset_util import rgb_to_srgb

    rgb = gs["rgb"][0].detach().float().clamp(0, 1).cpu()
    hwc = (rgb_to_srgb(rgb.permute(1, 2, 0)).numpy() * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(hwc, cv2.COLOR_RGB2BGR)


def _find_eval_pred(output_root: Path, stage: str, step: int, stem: str) -> Path | None:
    p = training_render_pred_dir(output_root, stage, step) / f"{stem}.png"
    return p if p.is_file() else None


def _find_frame_batch(cfg, stem: str, device):
    ds = ImageDataset(cfg, train=False, synthetic_if_empty=False, distribution_boost=False)
    if len(ds) == 0:
        ds = ImageDataset(cfg, train=True, synthetic_if_empty=False, distribution_boost=False)

    def matches(item) -> bool:
        fn = item.get("frame_name") or Path(item["img_path"]).stem
        if fn == stem or Path(item["img_path"]).stem == stem:
            return True
        if stem.isdigit() and fn.isdigit():
            return int(fn) == int(stem)
        return False

    for j in range(len(ds)):
        item = ds[j]
        if matches(item):
            batch = collate_batch([item])
            return move_batch_to_device(batch, device), j

    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_batch)
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        fn = batch["frame_name"][0]
        img_stem = Path(batch["img_path"][0]).stem
        if fn == stem or img_stem == stem:
            return batch, None
        if stem.isdigit() and str(fn).isdigit() and int(fn) == int(stem):
            return batch, None
    raise RuntimeError(f"frame stem not found in dataset: {stem}")


def main():
    p = argparse.ArgumentParser(description="Compare training renders/ vs reload gsplat.")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument(
        "--frame-stem",
        default=None,
        help="Stem under renders/{stage}/step_*/render/ (e.g. 100). Default: first available.",
    )
    p.add_argument("--list-stems", action="store_true", help="List render/ stems and exit.")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    if args.device not in ("auto", "cpu"):
        device = torch.device(args.device)

    output_root = Path(args.output_root)
    ckpt_path = find_final_checkpoint(output_root, args.checkpoint)
    stack = load_run_for_render(output_root, checkpoint=ckpt_path, device=device)
    step = stack.global_step
    stage = stack.spec.name
    stage_dir = training_render_stage_dir(output_root, stage, step)
    pred_dir = training_render_pred_dir(output_root, stage, step)
    stems = list_training_render_pred_stems(output_root, stage, step)

    if args.list_stems or args.frame_stem is None:
        print(f"renders stage dir: {stage_dir}")
        print(f"pred dir: {pred_dir}")
        if not stems:
            print("  (empty — no PNGs under render/)")
            if not args.list_stems:
                p.error("no render stems; pass --frame-stem or run training eval first")
            return
        for s in stems[:30]:
            print(f"  {s}")
        if len(stems) > 30:
            print(f"  ... +{len(stems) - 30} more")
        if args.list_stems:
            return
        args.frame_stem = stems[0]
        print(f"using first stem: {args.frame_stem}")

    eval_png = _find_eval_pred(output_root, stage, step, args.frame_stem)
    if eval_png is None:
        print(f"eval pred missing: {pred_dir / (args.frame_stem + '.png')}")
        print(f"available stems (first 10): {stems[:10]}")
        raise SystemExit(1)
    print(f"training pred: {eval_png}")

    batch, _ = _find_frame_batch(stack.cfg, args.frame_stem, device)
    reload_bgr = _render_bgr(stack, batch, composite=True)
    out_dir = output_root / "reload_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    reload_path = out_dir / f"{args.frame_stem}_reload.png"
    cv2.imwrite(str(reload_path), reload_bgr)
    print(f"reload render: {reload_path}")

    eval_bgr = _load_eval_pred_bgr(eval_png)
    if eval_bgr.shape != reload_bgr.shape:
        reload_bgr = cv2.resize(reload_bgr, (eval_bgr.shape[1], eval_bgr.shape[0]))
    diff = np.abs(eval_bgr.astype(np.float32) - reload_bgr.astype(np.float32))
    mae = float(diff.mean())
    mx = float(diff.max())
    side = np.concatenate([eval_bgr, reload_bgr, diff.astype(np.uint8)], axis=1)
    cmp_path = out_dir / f"{args.frame_stem}_eval_vs_reload.jpg"
    cv2.imwrite(str(cmp_path), side)
    print(f"  MAE={mae:.3f} max={mx:.1f}  -> {cmp_path}")
    if mae < 2.0:
        print("OK — reload matches training render")
    else:
        print("MISMATCH — reload differs from training render PNG")


if __name__ == "__main__":
    main()
