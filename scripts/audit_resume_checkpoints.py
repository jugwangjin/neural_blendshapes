#!/usr/bin/env python
"""
Simulate resume_from_checkpoint for each stage-end ckpt (avatar n after load).

Usage:
  python scripts/audit_resume_checkpoints.py --output-root .../bala
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.build import build_deformer, build_ict, build_tracker
from training.checkpoint_io import avatar_n_from_state_dict
from training.resume import resume_from_checkpoint


def main():
    p = argparse.ArgumentParser(description="Audit resume avatar counts per checkpoint")
    p.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args()

    ckpt_dir = Path(args.output_root) / "checkpoints"
    paths = sorted(ckpt_dir.glob("stage_*_end_step_*.pt"))
    if not paths:
        raise SystemExit(f"no checkpoints in {ckpt_dir}")

    device = torch.device("cpu")
    print(f"output_root: {args.output_root}\n")

    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        n_disk = avatar_n_from_state_dict(payload["avatar"])
        cfg = payload["cfg"]
        cfg.output_root = Path(args.output_root)

        ict = build_ict(cfg, device)
        tracker = build_tracker(cfg, ict, device)
        deformer = build_deformer(cfg, ict, device)
        avatar, step, meta = resume_from_checkpoint(
            path,
            ict=ict,
            deformer=deformer,
            tracker=tracker,
            device=device,
            cfg=cfg,
            payload=payload,
        )
        ok = int(avatar.n_gaussians) == n_disk == meta["n_gaussians"]
        flag = "" if ok else "  MISMATCH"
        print(
            f"{path.name}: disk n={n_disk} resume n={avatar.n_gaussians} "
            f"step={step} stage={meta['stage']}{flag}"
        )


if __name__ == "__main__":
    main()
