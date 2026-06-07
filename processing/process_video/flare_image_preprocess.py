"""FLARE-style image layout: copy frames → MODNet matte → face parsing (no video)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

PROC_VIDEO_ROOT = Path(__file__).resolve().parent
MODNET_ROOT = PROC_VIDEO_ROOT / "MODNet"
DEFAULT_MODNET_CKPT = MODNET_ROOT / "pretrained" / "modnet_webcam_portrait_matting.ckpt"


def subject_scene_root(output_dir: Path, subject_name: str) -> Path:
    """``{output_dir}/nf_01/nf_01`` (matches ``process_video.scene_root``)."""
    return Path(output_dir) / subject_name / subject_name


def split_paths(scene_root: Path, split: str) -> dict[str, Path]:
    base = scene_root / split
    return {
        "image": base / "image",
        "mask": base / "mask",
        "semantic": base / "semantic",
        "semantic_color": base / "semantic_color",
    }


def copy_frame_as_png(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    im = Image.open(src)
    if im.mode != "RGB":
        im = im.convert("RGB")
    im.save(dst, format="PNG")


def copy_frame_list(frames: list[Path], image_dir: Path) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    for src in frames:
        dst = image_dir / f"{src.stem}.png"
        copy_frame_as_png(src, dst)


def run_modnet_matte_dir(
    image_dir: Path,
    mask_dir: Path,
    ckpt: Path,
    *,
    ref_size: int = 512,
    device: str | None = None,
) -> None:
    """
    Per-image MODNet matte (``demo/image_matting/colab/inference.py`` logic).

    Reads ``image_dir/*.png``, writes grayscale mattes to ``mask_dir/<stem>.png``.
    """
    if not MODNET_ROOT.is_dir():
        raise FileNotFoundError(f"MODNet not found at {MODNET_ROOT}")
    if not ckpt.is_file():
        raise FileNotFoundError(f"MODNet checkpoint missing: {ckpt}")

    root = str(MODNET_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from src.models.modnet import MODNet

    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(image_dir.glob("*.png"))
    if len(paths) == 0:
        raise FileNotFoundError(f"No PNG in {image_dir}")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    im_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)
    weights = torch.load(str(ckpt), map_location=dev)
    modnet.load_state_dict(weights)
    modnet.to(dev)
    modnet.eval()

    for img_path in paths:
        print(f"  MODNet: {img_path.name}", flush=True)
        im = Image.open(img_path).convert("RGB")
        im_tensor = im_transform(im)
        im_tensor = im_tensor[None, :, :, :]
        im_b, im_c, im_h, im_w = im_tensor.shape
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            else:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w
        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32
        im_tensor = F.interpolate(im_tensor, size=(im_rh, im_rw), mode="area")
        im_tensor = im_tensor.to(dev)

        with torch.no_grad():
            _, _, matte = modnet(im_tensor, True)

        matte = F.interpolate(matte, size=(im_h, im_w), mode="area")
        matte_np = (matte[0, 0].cpu().numpy() * 255.0).astype("uint8")
        Image.fromarray(matte_np, mode="L").save(mask_dir / f"{img_path.stem}.png")


def run_face_parsing_split(
    image_dir: Path,
    semantic_dir: Path,
    semantic_color_dir: Path,
    *,
    ckpt: Path | None = None,
    device: str | None = None,
    inference_size: int = 512,
) -> None:
    from parse_faces import DEFAULT_CKPT, parse_image_dir

    parse_image_dir(
        image_dir,
        semantic_dir,
        semantic_color_dir,
        ckpt=ckpt if ckpt is not None else DEFAULT_CKPT,
        device=device,
        inference_size=inference_size,
    )


def process_split(
    paths: dict[str, Path],
    *,
    modnet_ckpt: Path,
    parser_ckpt: Path | None,
    device: str | None,
    skip_matte: bool,
    skip_parse: bool,
) -> None:
    if not skip_matte:
        run_modnet_matte_dir(
            paths["image"],
            paths["mask"],
            modnet_ckpt,
            device=device,
        )
    if not skip_parse:
        run_face_parsing_split(
            paths["image"],
            paths["semantic"],
            paths["semantic_color"],
            ckpt=parser_ckpt,
            device=device,
        )


def process_scene(
    scene_root: Path,
    *,
    modnet_ckpt: Path = DEFAULT_MODNET_CKPT,
    parser_ckpt: Path | None = None,
    device: str | None = None,
    skip_matte: bool = False,
    skip_parse: bool = False,
    splits: tuple[str, ...] = ("train", "test"),
) -> None:
    for split in splits:
        paths = split_paths(scene_root, split)
        if not paths["image"].is_dir() or len(list(paths["image"].glob("*.png"))) == 0:
            raise FileNotFoundError(f"missing frames: {paths['image']}")
        print(f"  [{split}] matte + parse ({len(list(paths['image'].glob('*.png')))} frames)", flush=True)
        process_split(
            paths,
            modnet_ckpt=modnet_ckpt,
            parser_ckpt=parser_ckpt,
            device=device,
            skip_matte=skip_matte,
            skip_parse=skip_parse,
        )
