"""
Bake assets/default_camera.npz from ICT canonical mesh (FLAME-aligned render space).

  python scripts/bake_default_camera.py
  python scripts/bake_default_camera.py --out assets/default_camera.npz --image-size 512
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Config
from model.ict_model import ICTFaceKitTorch
from utils.camera import FixedCamera
from utils.camera import save_default_camera


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "assets" / "default_camera.npz")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--fov-deg", type=float, default=35.0)
    args = parser.parse_args()

    cfg = Config()
    ict = ICTFaceKitTorch(npy_dir=str(cfg.ict_npy))
    verts = ict.canonical[0]
    cam = FixedCamera.from_mesh_bounds(
        verts, width=args.image_size, height=args.image_size, fov_deg=args.fov_deg
    )
    print(
        "  note: sanity/train apply with_view_correction(yaw=180, roll=180) at runtime; "
        "npz stores mesh-bounds camera only"
    )
    K = cam.K.numpy()
    R = cam.R.numpy()
    t = cam.t.numpy()
    out = save_default_camera(R, t, K, args.out)
    print(f"wrote {out}")
    print(f"  fx={cam.fx:.2f} fy={cam.fy:.2f} cx={cam.cx:.2f} cy={cam.cy:.2f}")
    print(f"  t={t.tolist()}")


if __name__ == "__main__":
    main()
