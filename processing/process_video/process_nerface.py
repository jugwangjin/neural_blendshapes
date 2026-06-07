#!/usr/bin/env python3
"""
NeRFace → FLARE layout: copy train/test PNGs, MODNet matte, face parsing.

Source::

    {input_dir}/person_1/train/f_0001.png
    {input_dir}/person_1/test/f_0001.png

Output (``Config.input_dir`` = ``.../flare_2/nf_01/nf_01``)::

    {output_dir}/nf_01/nf_01/train/image/f_0001.png
    {output_dir}/nf_01/nf_01/train/mask/f_0001.png
    {output_dir}/nf_01/nf_01/train/semantic/...
    ...

Run from repo root::

    python processing/process_video/process_nerface.py \\
        --input-dir /Bean/data/gwangjin/2024/nbshapes/nerface \\
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

FRAME_RE = re.compile(r"^f_(\d+)\.png$", re.IGNORECASE)


def stem_int_nerface(path: Path) -> int:
    m = FRAME_RE.match(path.name)
    if m is None:
        raise ValueError(f"expected f_XXXX.png, got {path.name}")
    return int(m.group(1))


def collect_nerface_frames(split_dir: Path) -> list[Path]:
    paths = [p for p in split_dir.glob("*.png") if FRAME_RE.match(p.name)]
    return sorted(paths, key=stem_int_nerface)


def discover_person_dirs(input_dir: Path) -> list[tuple[int, Path]]:
    out = []
    for p in sorted(input_dir.iterdir()):
        if not p.is_dir():
            continue
        m = re.fullmatch(r"person_(\d+)", p.name, flags=re.IGNORECASE)
        if m is None:
            continue
        out.append((int(m.group(1)), p))
    out.sort(key=lambda x: x[0])
    return out


def process_person(
    person_num: int,
    person_dir: Path,
    output_dir: Path,
    *,
    modnet_ckpt: Path,
    parser_ckpt: Path | None,
    device: str | None,
    skip_copy: bool,
    skip_matte: bool,
    skip_parse: bool,
) -> None:
    subject_name = f"nf_{person_num:02d}"
    scene = subject_scene_root(output_dir, subject_name)

    if not skip_copy:
        for split in ("train", "test"):
            split_dir = person_dir / split
            if not split_dir.is_dir():
                raise FileNotFoundError(split_dir)
            frames = collect_nerface_frames(split_dir)
            paths = split_paths(scene, split)
            print(f"[person_{person_num}] copy {split}: {len(frames)} -> {paths['image']}")
            copy_frame_list(frames, paths["image"])

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
        description="NeRFace: copy frames + MODNet image matte + face parsing (FLARE layout)"
    )
    p.add_argument("--input-dir", type=Path, required=True, help="Root with person_1, person_2, ...")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Bean/data/gwangjin/2024/nbshapes/flare_2"),
        help="flare_2 root; writes nf_XX/nf_XX/{train,test}/image|mask|semantic/",
    )
    p.add_argument("--modnet-ckpt", type=Path, default=DEFAULT_MODNET_CKPT)
    p.add_argument("--parser-ckpt", type=Path, default=None)
    p.add_argument("--device", type=str, default=None, help="cuda | cpu")
    p.add_argument("--skip-copy", action="store_true")
    p.add_argument("--skip-matte", action="store_true")
    p.add_argument("--skip-parse", action="store_true")
    p.add_argument(
        "--persons",
        type=int,
        nargs="*",
        default=None,
        help="Subset of person indices (default: all person_* under input-dir)",
    )
    args = p.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)

    person_dirs = discover_person_dirs(input_dir)
    if args.persons is not None and len(args.persons) > 0:
        want = set(int(x) for x in args.persons)
        person_dirs = [(n, d) for n, d in person_dirs if n in want]
    if len(person_dirs) == 0:
        raise FileNotFoundError(f"no person_* directories under {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for person_num, person_dir in person_dirs:
        process_person(
            person_num,
            person_dir,
            output_dir,
            modnet_ckpt=args.modnet_ckpt,
            parser_ckpt=args.parser_ckpt,
            device=args.device,
            skip_copy=args.skip_copy,
            skip_matte=args.skip_matte,
            skip_parse=args.skip_parse,
        )


if __name__ == "__main__":
    main()
