#!/usr/bin/env python
"""
Audit checkpoint avatar color — is saved state trained or still init-like?

``color_max_diff=0`` on reload only proves bytes round-trip; this script checks
whether the **saved tensors themselves** look like GB random init (U(0,1/255) DC).

Usage:
  python scripts/audit_checkpoint_color.py \\
    --output-root /Bean/log/gwangjin/2026/neural_blendshapes_10/bala
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from training.checkpoint_io import avatar_color_stats, avatar_n_from_state_dict, warn_if_avatar_color_init_like


def main():
    p = argparse.ArgumentParser(description="Audit checkpoint avatar color stats.")
    p.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args()

    ckpt_dir = Path(args.output_root) / "checkpoints"
    paths = sorted(ckpt_dir.glob("stage_*_end_step_*.pt"))
    if not paths:
        raise SystemExit(f"no checkpoints under {ckpt_dir}")

    print(f"{'checkpoint':48s} {'n':>8s}  {'dc_mean':>8s}  {'dc_max':>8s}  {'sh_rest':>8s}  init?")
    print("-" * 95)
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        av = payload["avatar"]
        n = avatar_n_from_state_dict(av)
        s = avatar_color_stats(av)
        init_like = (
            s["dc_sigmoid_mean"] < 0.02
            and s["dc_sigmoid_max"] < 0.08
            and s["sh_rest_rms"] < 0.05
        )
        flag = "YES" if init_like else "no"
        print(
            f"{path.name:48s} {n:8d}  {s['dc_sigmoid_mean']:8.4f}  {s['dc_sigmoid_max']:8.4f}  "
            f"{s['sh_rest_rms']:8.4f}  {flag}"
        )
        if init_like:
            warn_if_avatar_color_init_like(av, tag=path.name)

    print(
        "\nInterpretation:\n"
        "  init?=YES  → DC + sh_rest both init-like (bootstrap / pre-RGB).\n"
        "               Stage 2 with sh_rest≈0.24 is trained color — not init.\n"
        "  init?=no   → sh trained. eval OK + reload broken was h_trainable (load bug, fixed).\n"
        "               Compare: scripts/compare_eval_vs_reload_render.py"
    )


if __name__ == "__main__":
    main()
