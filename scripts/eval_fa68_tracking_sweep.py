#!/usr/bin/env python
"""
FA-68 landmark error: face_alignment GT vs ICT mesh landmarks.

Independent of MediaPipe — measures raw vs personalized tracker on train ∪ eval.

Usage:
  python scripts/eval_fa68_tracking_sweep.py --ablation all --dry-run
  python scripts/eval_fa68_tracking_sweep.py --device cuda --skip-existing
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from tqdm import tqdm

from dataset import move_batch_to_device
from eval.fa68_landmark_metrics import Fa68ErrorAccumulator, fa68_landmark_error_batch
from eval.tracking_eval_common import (
    build_all_frames_loader,
    build_personalized_tracker_out,
    build_raw_tracker_out,
    final_stage_spec,
    find_final_checkpoint,
    infer_ablation_from_output_root,
    mesh_from_tracker_out,
)
from model.build import build_deformer, build_ict, build_tracker
from run_status import ABLATION_LOG_ROOTS, is_run_complete, log_root_for_ablation
from training.checkpoint_io import load_checkpoint
from utils.camera import load_training_camera


def _device_from_cli(device: str | None) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def eval_run(
    output_root: Path,
    *,
    device: torch.device,
    score_thresh: float,
    skip_existing: bool,
    dry_run: bool,
    num_workers: int = 0,
) -> dict | None:
    if not is_run_complete(output_root):
        print(f"skip (incomplete): {output_root}")
        return None

    out_json = output_root / "fa68_landmark_eval.json"
    if skip_existing and out_json.is_file():
        return json.loads(out_json.read_text(encoding="utf-8"))

    ckpt_path = find_final_checkpoint(output_root)
    ablation = infer_ablation_from_output_root(output_root)
    spec = final_stage_spec(output_root)
    print(f"\n=== FA68 {output_root.name} ({ablation}) ===")

    if dry_run:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        _, n = build_all_frames_loader(payload["cfg"], num_workers=0)
        print(f"  frames: {n}")
        return {"run_name": output_root.name, "ablation": ablation, "n_frames": n}

    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = deepcopy(payload["cfg"])
    cfg.output_root = Path(output_root)

    ict = build_ict(cfg, device)
    tracker = build_tracker(cfg, ict, device)
    deformer = build_deformer(cfg, ict, device)
    load_checkpoint(ckpt_path, tracker=tracker, deformer=deformer, map_location=device)
    tracker.eval()
    deformer.eval()
    camera = load_training_camera(
        ict.expression_reference_verts(),
        path=cfg.camera_npz,
        width=cfg.image_size,
        height=cfg.image_size,
        device=device,
    )
    loader, n_frames = build_all_frames_loader(cfg, num_workers=num_workers)

    acc_raw = Fa68ErrorAccumulator()
    acc_pers = Fa68ErrorAccumulator()
    for batch in tqdm(loader, total=n_frames, desc=output_root.name):
        batch = move_batch_to_device(batch, device)
        if batch.get("landmark") is None:
            continue
        mesh_raw = mesh_from_tracker_out(deformer, build_raw_tracker_out(tracker, batch, spec), spec)
        mesh_pers = mesh_from_tracker_out(
            deformer, build_personalized_tracker_out(tracker, batch, spec), spec
        )
        acc_raw.add_frame(
            fa68_landmark_error_batch(
                mesh_raw, ict, batch["landmark"], camera, cfg.image_size, score_thresh=score_thresh
            )[0]
        )
        acc_pers.add_frame(
            fa68_landmark_error_batch(
                mesh_pers, ict, batch["landmark"], camera, cfg.image_size, score_thresh=score_thresh
            )[0]
        )

    result = {
        "run_name": output_root.name,
        "ablation": ablation,
        "checkpoint": str(ckpt_path),
        "score_thresh": score_thresh,
        "raw": acc_raw.summary(),
        "personalized": acc_pers.summary(),
    }
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"  raw RMSE={result['raw']['rmse_px']:.3f}px  "
        f"pers={result['personalized']['rmse_px']:.3f}px"
    )
    return result


def write_summary_csv(rows: list[dict], path: Path):
    if not rows:
        return
    fields = [
        "ablation",
        "run_name",
        "raw_mse_px",
        "raw_rmse_px",
        "raw_per_frame_mse_std",
        "pers_mse_px",
        "pers_rmse_px",
        "pers_per_frame_mse_std",
        "n_frames",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            raw, pers = r["raw"], r["personalized"]
            w.writerow(
                {
                    "ablation": r["ablation"],
                    "run_name": r["run_name"],
                    "raw_mse_px": raw["mse_px"],
                    "raw_rmse_px": raw["rmse_px"],
                    "raw_per_frame_mse_std": raw["per_frame_mse_std"],
                    "pers_mse_px": pers["mse_px"],
                    "pers_rmse_px": pers["rmse_px"],
                    "pers_per_frame_mse_std": pers["per_frame_mse_std"],
                    "n_frames": raw["n_frames"],
                }
            )


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
    p.add_argument("--score-thresh", type=float, default=0.3)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    args = p.parse_args()

    device = _device_from_cli(args.device)
    roots = [Path(args.log_root)] if args.log_root else list(iter_log_roots(args.ablation))
    all_rows = []

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
                score_thresh=args.score_thresh,
                skip_existing=args.skip_existing,
                dry_run=args.dry_run,
                num_workers=args.num_workers,
            )
            if row and "raw" in row:
                rows.append(row)
                all_rows.append(row)
        if rows and not args.dry_run:
            write_summary_csv(rows, log_root / "fa68_landmark_eval_summary.csv")

    print(f"Done: {len(all_rows)} run(s).")


if __name__ == "__main__":
    main()
