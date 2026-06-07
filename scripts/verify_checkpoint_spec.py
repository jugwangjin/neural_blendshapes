#!/usr/bin/env python
"""Compare checkpoint metadata vs codes snapshot vs current-code schedule."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.render_load import final_stage_spec, find_final_checkpoint
from training.stage_spec_io import (
    load_final_stage_spec_from_codes,
    resolve_render_stage_spec,
    stage_spec_from_dict,
    stage_spec_to_dict,
)


def _brief(spec, label: str):
    if spec is None:
        print(f"{label}: (none)")
        return
    d = stage_spec_to_dict(spec)
    print(
        f"{label}: sh_degree={d.get('sh_degree')} "
        f"expr_deform={d.get('train_expression_deform')} "
        f"color_expr={d.get('train_color_expression')} "
        f"color_pose={d.get('train_color_pose')}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=None)
    args = p.parse_args()

    ckpt = find_final_checkpoint(args.output_root, args.checkpoint)
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    av = payload["avatar"]
    color = av["color"]

    print(f"checkpoint: {ckpt.name}")
    print(f"  step={payload.get('step')} stage={payload.get('stage')}")
    print(f"  n_gaussians={av['face_idx'].shape[0]} color={tuple(color.shape)}")
    print(f"  cfg.sh_degree={getattr(payload['cfg'], 'sh_degree', None)}")
    print(f"  has stage_spec in ckpt: {'stage_spec' in payload}")

    ckpt_spec = stage_spec_from_dict(payload["stage_spec"]) if "stage_spec" in payload else None
    codes_spec = load_final_stage_spec_from_codes(args.output_root)
    current_spec = final_stage_spec(args.output_root)
    render_spec = resolve_render_stage_spec(payload, args.output_root)

    _brief(ckpt_spec, "ckpt stage_spec")
    _brief(codes_spec, "codes STAGE_SCHEDULE[-1]")
    _brief(current_spec, "current-code schedule[-1]")
    _brief(render_spec, "render will use")

    if "stage_spec" in payload and payload.get("stage") == "3_expression_detail":
        print("\nOK: render spec from checkpoint stage_spec.")
    elif codes_spec is not None:
        print("\nOK for existing run: render spec from codes/STAGE_SCHEDULE.json.")
    else:
        print("\nWARN: no stage_spec or codes dump — render may not match training.")


if __name__ == "__main__":
    main()
