"""
Precompute MediaPipe blendshapes + landmarks per frame → .npz cache.

Usage (repo root):
  python scripts/precompute_mediapipe.py --input_dir DATA --scene MVI_1797 --out cache/mediapipe
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--out", type=Path, default=Path("cache/mediapipe"))
    args = parser.parse_args()
    out_dir = args.out / args.scene
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"TODO: run MediaPipe on frames under {args.input_dir / args.scene}")
    print(f"Write per-frame npz to {out_dir} with keys:")
    print("  mp_blendshape [52], mp_landmarks_2d [478,2], mp_pose_raw [6], mp_valid [478]")


if __name__ == "__main__":
    main()
