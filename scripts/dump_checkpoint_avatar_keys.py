#!/usr/bin/env python
"""Print avatar keys/stats inside a checkpoint .pt (debug save/load)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from training.checkpoint_io import avatar_color_stats, format_avatar_color_stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    args = p.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    av = payload["avatar"]
    print(f"step={payload.get('step')} stage={payload.get('stage')} n_gaussians={payload.get('n_gaussians')}")
    print(f"avatar keys ({len(av)}):")
    for k in sorted(av.keys()):
        v = av[k]
        if torch.is_tensor(v):
            print(f"  {k:28s} {tuple(v.shape)} {v.dtype} mean={v.float().mean().item():.6f}")
        else:
            print(f"  {k:28s} {type(v).__name__}")
    print(format_avatar_color_stats(av))
    s = avatar_color_stats(av)
    print(f"log_scale mean={av['log_scale'].float().mean().item():.4f} (init would be ~0 → exp=1)")


if __name__ == "__main__":
    main()
