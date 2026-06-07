#!/usr/bin/env python3
"""
NeRFBlendShape → FLARE layout: copy jpg frames, MODNet matte, face parsing.

Source::

    {input_dir}/id1/ori_imgs/1.jpg  (numeric sort, not lexicographic)

Output::

    {output_dir}/nbs_id1/nbs_id1/train/image/*.png   # all frames
    {output_dir}/nbs_id1/nbs_id1/test/image/*.png    # last ``test_tail`` frames

Train/test overlap is allowed (driving vis only).

Run from repo root::

    python processing/process_video/process_nerfblendshape.py \\
        --input-dir /path/to/nerfblendshape_root \\
        --output-dir /Bean/data/gwangjin/2024/nbshapes/flare_2
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROC_ROOT = Path(__file__).resolve().parent
if str(PROC_ROOT) not in sys.path:
    sys.path.insert(0, str(PROC_ROOT))

from flare_image_preprocess import (
    DEFAULT_MODNET_CKPT,
    copy_frame_list,
    process_scene,
    split_paths,
    subject_scene_root,
)
from frame_sequence_io import collect_images

ORI_IMG_DIR_NAMES = ("ori_imgs", "org_imgs")


def _ori_imgs_dir(identity_dir: Path) -> Path:
    for name in ORI_IMG_DIR_NAMES:
        d = identity_dir / name
        if d.is_dir():
            return d
    raise FileNotFoundError(
        f"no ori_imgs/org_imgs under {identity_dir} (tried {ORI_IMG_DIR_NAMES})"
    )


def discover_id_dirs(input_dir: Path) -> list[tuple[int, Path]]:
    out = []
    for p in sorted(input_dir.iterdir()):
        if not p.is_dir():
            continue
        m = re.fullmatch(r"id(\d+)", p.name, flags=re.IGNORECASE)
        if m is None:
            continue
        out.append((int(m.group(1)), p))
    out.sort(key=lambda x: x[0])
    return out


def process_identity(
    id_num: int,
    identity_dir: Path,
    output_dir: Path,
    *,
    test_tail: int,
    modnet_ckpt: Path,
    parser_ckpt: Path | None,
    device: str | None,
    skip_copy: bool,
    skip_matte: bool,
    skip_parse: bool,
) -> None:
    frames_dir = _ori_imgs_dir(identity_dir)
    all_frames = collect_images(frames_dir)
    tail = int(test_tail)
    if tail <= 0:
        raise ValueError("test_tail must be positive")
    test_frames = all_frames[-tail:] if len(all_frames) >= tail else all_frames
    if len(test_frames) < tail:
        print(f"  warn: only {len(test_frames)} test frames (< {tail})")

    subject_name = f"nbs_id{id_num}"
    scene = subject_scene_root(output_dir, subject_name)

    if not skip_copy:
        train_paths = split_paths(scene, "train")
        test_paths = split_paths(scene, "test")
        print(f"[id{id_num}] copy train: {len(all_frames)} -> {train_paths['image']}")
        copy_frame_list(all_frames, train_paths["image"])
        print(f"[id{id_num}] copy test: {len(test_frames)} -> {test_paths['image']}")
        copy_frame_list(test_frames, test_paths["image"])

    process_scene(
        scene,
        modnet_ckpt=modnet_ckpt,
        parser_ckpt=parser_ckpt,
        device=device,
        skip_matte=skip_matte,
        skip_parse=skip_parse,
    )
    print(f"[done] {subject_name} -> {scene}")


def main():
    p = argparse.ArgumentParser(
        description="NeRFBlendShape: copy frames + MODNet image matte + face parsing"
    )
    p.add_argument("--input-dir", type=Path, required=True, help="Root with id1, id2, ...")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Bean/data/gwangjin/2024/nbshapes/flare_2"),
        help="flare_2 root; writes nbs_id{N}/nbs_id{N}/{train,test}/...",
    )
    p.add_argument(
        "--test-tail",
        type=int,
        default=500,
        help="Last N frames for test split (may overlap train)",
    )
    p.add_argument("--modnet-ckpt", type=Path, default=DEFAULT_MODNET_CKPT)
    p.add_argument("--parser-ckpt", type=Path, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--skip-copy", action="store_true")
    p.add_argument("--skip-matte", action="store_true")
    p.add_argument("--skip-parse", action="store_true")
    p.add_argument(
        "--ids",
        type=int,
        nargs="*",
        default=None,
        help="Subset of id numbers (default: all id* under input-dir)",
    )
    args = p.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)

    id_dirs = discover_id_dirs(input_dir)
    if args.ids is not None and len(args.ids) > 0:
        want = set(int(i) for i in args.ids)
        id_dirs = [(n, d) for n, d in id_dirs if n in want]
    if len(id_dirs) == 0:
        raise FileNotFoundError(f"no id* directories under {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for id_num, identity_dir in id_dirs:
        process_identity(
            id_num,
            identity_dir,
            output_dir,
            test_tail=int(args.test_tail),
            modnet_ckpt=args.modnet_ckpt,
            parser_ckpt=args.parser_ckpt,
            device=args.device,
            skip_copy=args.skip_copy,
            skip_matte=args.skip_matte,
            skip_parse=args.skip_parse,
        )


if __name__ == "__main__":
    main()
