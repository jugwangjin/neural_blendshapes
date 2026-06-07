#!/usr/bin/env python3
"""
Crop + matte + frame extraction + face parsing for FLARE-style subject folders.

Input (example ``/Bean/data/.../flare_2/justin/``)::

    justin/train.mp4
    justin/test.mp4
    ...

Output::

    justin/justin/train/image/1.png
    justin/justin/train/mask/1.png
    justin/justin/train/semantic/1.png
    justin/justin/train/semantic_color/1.png

Intermediate videos (same level as source mp4)::

    train_cropped.mp4
    train_cropped_matte.mp4

Run from repo root::

    python processing/process_video/process_video.py \\
        --dataset-dir /Bean/data/gwangjin/2024/nbshapes/flare_2/justin

Requires: ffmpeg, MODNet + face-parsing.PyTorch under ``processing/process_video/``.
See ``processing/process_video/README.md``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROC_VIDEO_ROOT = Path(__file__).resolve().parent
if str(PROC_VIDEO_ROOT) not in sys.path:
    sys.path.insert(0, str(PROC_VIDEO_ROOT))
REPO_ROOT = PROC_VIDEO_ROOT.parent.parent
MODNET_ROOT = PROC_VIDEO_ROOT / "MODNet"
DEFAULT_MODNET_CKPT = MODNET_ROOT / "pretrained" / "modnet_webcam_portrait_matting.ckpt"


def _run(cmd: list[str], *, cwd: Path | None = None):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd is not None else None)


def discover_source_videos(dataset_dir: Path, names: list[str] | None) -> list[Path]:
    """``*.mp4`` in dataset_dir, excluding ``*_cropped*`` intermediates."""
    if names:
        out = []
        for name in names:
            p = dataset_dir / name
            if not p.suffix:
                p = p.with_suffix(".mp4")
            if not p.is_file():
                raise FileNotFoundError(p)
            out.append(p)
        return out

    out = []
    for p in sorted(dataset_dir.glob("*.mp4")):
        if "_cropped" in p.stem:
            continue
        out.append(p)
    return out


def cropped_path(video: Path) -> Path:
    return video.with_name(f"{video.stem}_cropped.mp4")


def matte_path(cropped: Path) -> Path:
    return cropped.with_name(f"{cropped.stem}_matte.mp4")


def scene_root(dataset_dir: Path, video_stem: str) -> Path:
    dataset_name = dataset_dir.name
    return dataset_dir / dataset_name / video_stem


def ffmpeg_crop_resize(
    src: Path,
    dst: Path,
    *,
    crop: str,
    resize: int,
    fps: int,
):
    vf = f"fps={fps},crop={crop},scale={resize}:{resize}"
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-vf",
            vf,
            "-c:v",
            "libx264",
            str(dst),
        ]
    )


def ffmpeg_extract_frames(video: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "%d.png")
    _run(["ffmpeg", "-y", "-i", str(video), "-q:v", "2", pattern])


def run_modnet_matte(cropped_video: Path, fps: int, ckpt: Path):
    if not MODNET_ROOT.is_dir():
        raise FileNotFoundError(
            f"MODNet not found at {MODNET_ROOT}. See processing/process_video/README.md"
        )
    if not ckpt.is_file():
        raise FileNotFoundError(f"MODNet checkpoint missing: {ckpt}")

    env = dict(os.environ)
    prev = env.get("PYTHONPATH", "")
    mod_path = str(MODNET_ROOT)
    env["PYTHONPATH"] = mod_path if not prev else f"{mod_path}{os.pathsep}{prev}"

    cmd = [
        sys.executable,
        "-m",
        "demo.video_matting.custom.run",
        "--video",
        str(cropped_video.resolve()),
        "--result-type",
        "matte",
        "--fps",
        str(fps),
    ]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(MODNET_ROOT), env=env)


def run_face_parsing(image_dir: Path, scene: Path, ckpt: Path, device: str | None):
    from parse_faces import DEFAULT_CKPT, parse_image_dir

    ckpt_use = ckpt if ckpt is not None else DEFAULT_CKPT
    parse_image_dir(
        image_dir,
        scene / "semantic",
        scene / "semantic_color",
        ckpt=ckpt_use,
        device=device,
    )


def process_one_video(
    video: Path,
    dataset_dir: Path,
    *,
    crop: str,
    resize: int,
    fps: int,
    modnet_ckpt: Path,
    parser_ckpt: Path | None,
    device: str | None,
    skip_crop: bool,
    skip_matte: bool,
    skip_frames: bool,
    skip_parse: bool,
):
    stem = video.stem
    scene = scene_root(dataset_dir, stem)
    cropped = cropped_path(video)
    matte = matte_path(cropped)

    if not skip_crop:
        ffmpeg_crop_resize(video, cropped, crop=crop, resize=resize, fps=fps)
    elif not cropped.is_file():
        raise FileNotFoundError(f"--skip-crop but missing {cropped}")

    if not skip_matte:
        run_modnet_matte(cropped, fps, modnet_ckpt)
    elif not matte.is_file():
        raise FileNotFoundError(f"--skip-matte but missing {matte}")

    image_dir = scene / "image"
    mask_dir = scene / "mask"
    if not skip_frames:
        ffmpeg_extract_frames(cropped, image_dir)
        ffmpeg_extract_frames(matte, mask_dir)
    elif not image_dir.is_dir():
        raise FileNotFoundError(f"--skip-frames but missing {image_dir}")

    if not skip_parse:
        run_face_parsing(image_dir, scene, parser_ckpt, device)

    print(f"[done] {stem} → {scene}")


def main():
    p = argparse.ArgumentParser(
        description="Crop/matte videos and build FLARE-style image/mask/semantic folders"
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Folder containing source .mp4 files (e.g. .../flare_2/justin)",
    )
    p.add_argument(
        "--videos",
        nargs="*",
        default=None,
        help="Subset of video stems or filenames (default: all top-level .mp4)",
    )
    p.add_argument(
        "--crop",
        type=str,
        default="1080:1080:420:0",
        help="ffmpeg crop filter (w:h:x:y), adjust per capture",
    )
    p.add_argument("--resize", type=int, default=512)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--modnet-ckpt", type=Path, default=DEFAULT_MODNET_CKPT)
    p.add_argument(
        "--parser-ckpt",
        type=Path,
        default=None,
        help="face-parsing checkpoint (default: submodules/.../res/cp/79999_iter.pth)",
    )
    p.add_argument("--device", type=str, default=None, help="cuda | cpu for face parsing")
    p.add_argument("--skip-crop", action="store_true")
    p.add_argument("--skip-matte", action="store_true")
    p.add_argument("--skip-frames", action="store_true")
    p.add_argument("--skip-parse", action="store_true")
    args = p.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(dataset_dir)

    videos = discover_source_videos(dataset_dir, args.videos)
    if len(videos) == 0:
        raise FileNotFoundError(f"No source .mp4 in {dataset_dir}")

    print(f"dataset_dir={dataset_dir}  videos={[v.name for v in videos]}")
    for video in videos:
        process_one_video(
            video,
            dataset_dir,
            crop=args.crop,
            resize=args.resize,
            fps=args.fps,
            modnet_ckpt=args.modnet_ckpt,
            parser_ckpt=args.parser_ckpt,
            device=args.device,
            skip_crop=args.skip_crop,
            skip_matte=args.skip_matte,
            skip_frames=args.skip_frames,
            skip_parse=args.skip_parse,
        )


if __name__ == "__main__":
    main()
