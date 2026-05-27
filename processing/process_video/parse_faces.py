"""
BiSeNet face parsing on a folder of RGB frames.

Writes FLARE-compatible outputs per frame:
  - ``semantic/<stem>.png``      uint8 part id (0–18), training loader uses ``imageio`` mode ``F``
  - ``semantic_color/<stem>.png``  RGB overlay visualization

Expects submodule at ``processing/process_video/submodules/face-parsing.PyTorch``
with checkpoint ``res/cp/79999_iter.pth``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

PROC_VIDEO_ROOT = Path(__file__).resolve().parent
PARSER_ROOT = PROC_VIDEO_ROOT / "submodules" / "face-parsing.PyTorch"
DEFAULT_CKPT = PARSER_ROOT / "res" / "cp" / "79999_iter.pth"

PART_COLORS = [
    [0, 0, 0],
    [255, 0, 0],
    [255, 85, 0],
    [255, 170, 0],
    [255, 0, 85],
    [255, 0, 170],
    [0, 255, 0],
    [85, 255, 0],
    [170, 255, 0],
    [0, 255, 85],
    [0, 255, 170],
    [0, 0, 255],
    [85, 0, 255],
    [170, 0, 255],
    [0, 85, 255],
    [0, 170, 255],
    [255, 255, 0],
    [255, 255, 85],
    [255, 255, 170],
    [255, 0, 255],
    [255, 85, 255],
    [255, 170, 255],
    [0, 255, 255],
    [85, 255, 255],
    [170, 255, 255],
]


def _import_bisenet():
    if not PARSER_ROOT.is_dir():
        raise FileNotFoundError(
            f"face-parsing.PyTorch not found at {PARSER_ROOT}. "
            "See processing/process_video/README.md"
        )
    root = str(PARSER_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from model import BiSeNet

    return BiSeNet


def _color_overlay(rgb: np.ndarray, parsing: np.ndarray) -> np.ndarray:
    """RGB uint8 [H,W,3] + part ids → blended visualization (BGR for cv2 write)."""
    vis = rgb.copy().astype(np.uint8)
    color = np.zeros((*parsing.shape, 3), dtype=np.uint8) + 255
    max_cls = int(parsing.max())
    for pi in range(1, max_cls + 1):
        mask = parsing == pi
        if not mask.any():
            continue
        c = PART_COLORS[pi % len(PART_COLORS)]
        color[mask] = c
    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    out = cv2.addWeighted(vis_bgr, 0.4, color, 0.6, 0)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def _load_net(ckpt: Path, device: torch.device):
    BiSeNet = _import_bisenet()
    net = BiSeNet(n_classes=19)
    state = torch.load(str(ckpt), map_location=device)
    net.load_state_dict(state)
    net.to(device)
    net.eval()
    to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    return net, to_tensor


@torch.no_grad()
def parse_image_dir(
    image_dir: Path,
    semantic_dir: Path,
    semantic_color_dir: Path,
    *,
    ckpt: Path = DEFAULT_CKPT,
    device: str | None = None,
    inference_size: int = 512,
):
    image_dir = Path(image_dir)
    semantic_dir = Path(semantic_dir)
    semantic_color_dir = Path(semantic_color_dir)
    semantic_dir.mkdir(parents=True, exist_ok=True)
    semantic_color_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(image_dir.glob("*.png"))
    if len(paths) == 0:
        raise FileNotFoundError(f"No PNG frames in {image_dir}")

    if not ckpt.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    net, to_tensor = _load_net(ckpt, dev)

    for img_path in paths:
        pil = Image.open(img_path).convert("RGB")
        ow, oh = pil.size
        resized = pil.resize((inference_size, inference_size), Image.BILINEAR)
        batch = to_tensor(resized).unsqueeze(0).to(dev)
        logits = net(batch)[0]
        parsing = logits.squeeze(0).cpu().numpy().argmax(0).astype(np.uint8)
        parsing = cv2.resize(parsing, (ow, oh), interpolation=cv2.INTER_NEAREST)

        stem = img_path.stem
        imageio.imwrite(str(semantic_dir / f"{stem}.png"), parsing)
        rgb = np.array(pil, dtype=np.uint8)
        color = _color_overlay(rgb, parsing)
        imageio.imwrite(str(semantic_color_dir / f"{stem}.png"), color)


def main():
    p = argparse.ArgumentParser(description="Face parsing on extracted frames")
    p.add_argument("--image-dir", type=Path, required=True)
    p.add_argument("--semantic-dir", type=Path, required=True)
    p.add_argument("--semantic-color-dir", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--inference-size", type=int, default=512)
    args = p.parse_args()
    parse_image_dir(
        args.image_dir,
        args.semantic_dir,
        args.semantic_color_dir,
        ckpt=args.ckpt,
        device=args.device,
        inference_size=args.inference_size,
    )
    print(f"Wrote semantics for {len(list(args.image_dir.glob('*.png')))} frames")


if __name__ == "__main__":
    main()
