"""
Precompute MediaPipe + face_alignment cache for a FLARE split.

Usage (repo root):
  python scripts/precompute_mediapipe.py --input_dir /Bean/data/.../bala/bala --split train
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Config
from dataset.dataset_util import format_splits_label, list_split_images, normalize_split_names
from dataset.frame_processor import build_split_cache
from dataset.image_dataset import ImageDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--face_landmarker",
        type=Path,
        default=Path("assets/face_landmarker.task"),
    )
    args = parser.parse_args()

    cfg = Config()
    cfg.input_dir = args.input_dir
    cfg.rebuild_mp_cache = args.rebuild
    if args.out is not None:
        cfg.mp_cache_dir = args.out
    cfg.face_landmarker_task = args.face_landmarker

    images = list_split_images(cfg.input_dir, args.split)
    split_label = format_splits_label(args.split)
    print(f"{len(images)} images under {cfg.input_dir}/{{{split_label}}}/image")
    build_split_cache(cfg, args.split, images, rebuild=args.rebuild)

    if set(normalize_split_names(args.split)) != set(normalize_split_names(cfg.train_split)):
        print(
            f"skip bshapes_mode (pass all train scenes "
            f"{normalize_split_names(cfg.train_split)!r} to bake calibration)"
        )
        return

    ImageDataset(cfg, train=True, distribution_boost=False)


if __name__ == "__main__":
    main()
