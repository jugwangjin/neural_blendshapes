#!/usr/bin/env python
"""
Verify checkpoint avatar tensors reload faithfully (color, log_scale, n_gaussians).

Usage:
  python scripts/verify_gaussian_checkpoint_load.py \\
    --output-root /Bean/log/gwangjin/2026/neural_blendshapes_10/bala
  python scripts/verify_gaussian_checkpoint_load.py \\
    --output-root .../bala --checkpoint stage_3_expression_detail_end_step_037500.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from model.build import sh_dim_from_avatar_state
from training.render_load import find_final_checkpoint, load_run_for_render


def _dc_sigmoid_mean(avatar) -> float:
    color = avatar.color
    if color.ndim == 3:
        dc = color[:, 0, :]
    else:
        dc = color
    return float(torch.sigmoid(dc).mean().item())


def main():
    p = argparse.ArgumentParser(description="Verify avatar checkpoint load fidelity.")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    if args.device not in ("auto", "cpu"):
        device = torch.device(args.device)

    output_root = Path(args.output_root)
    ckpt_path = find_final_checkpoint(output_root, args.checkpoint)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_av = payload["avatar"]
    n_ckpt = int(ckpt_av["face_idx"].shape[0])
    sh_ckpt = sh_dim_from_avatar_state(ckpt_av)

    warn_if_avatar_color_init_like(ckpt_av, tag=f"load {ckpt_path.name}")

    stack = load_run_for_render(output_root, checkpoint=ckpt_path, device=device)
    avatar = stack.avatar

    print(f"\ncheckpoint: {ckpt_path.name}")
    print(f"  stage={payload.get('stage')} step={payload.get('step')}")
    print(f"  ckpt n={n_ckpt} sh_dim={sh_ckpt}")
    print(f"  loaded n={avatar.n_gaussians} sh_dim={avatar.sh_dim}")
    print(f"  render spec: {stack.spec.name} sh_degree={stack.spec.sh_degree}")

    color_diff = (avatar.color - ckpt_av["color"].to(avatar.color.device, dtype=avatar.color.dtype)).abs().max().item()
    ls_diff = (
        avatar.log_scale - ckpt_av["log_scale"].to(avatar.log_scale.device, dtype=avatar.log_scale.dtype)
    ).abs().max().item()
    dc_mean = _dc_sigmoid_mean(avatar)

    print(f"  color_max_diff={color_diff:.3e} log_scale_max_diff={ls_diff:.3e}")
    print(f"  color_dc_sigmoid_mean={dc_mean:.4f} log_scale_mean={avatar.log_scale.mean().item():.4f}")

    ok = (
        avatar.n_gaussians == n_ckpt
        and avatar.sh_dim == sh_ckpt
        and color_diff <= 1e-5
        and ls_diff <= 1e-5
    )
    if dc_mean < 0.05:
        print(
            "  WARNING: color_dc_sigmoid_mean very low — face may look dark/rainbow from SH bands "
            "(check eval_render PNG vs reload; not necessarily a load bug)"
        )
    if ok:
        print("\nOK — trained color and log_scale restored")
    else:
        print("\nFAIL — avatar tensors did not match checkpoint")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
