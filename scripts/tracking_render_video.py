"""GT / gsplat RGB / tracking frame folders → mp4 (``render_tracking_sweep.py``)."""

from __future__ import annotations

from pathlib import Path

import imageio
import numpy as np
import torch

from dataset.dataset_util import rgb_to_srgb
from processing.process_video.frame_sequence_io import collect_png_frames, images_to_mp4

TRACKING_FRAME_TAGS = ("gt", "rgb", "raw", "personalized")


def tracking_frame_dir(output_root: Path, scene: str, tag: str) -> Path:
    return output_root / f"tracking_render_{scene}_{tag}"


def discover_tracking_scenes(output_root: Path) -> list[str]:
    scenes = []
    for p in sorted(output_root.glob("tracking_render_*_raw")):
        scene = p.name[len("tracking_render_") : -len("_raw")]
        if scene:
            scenes.append(scene)
    return scenes


@torch.no_grad()
def save_gt_raw_frame(path: Path, img_path: Path, *, image_size: int):
    """Dataset PNG as-is (resize only); no mask matte or background composite."""
    from dataset.dataset_util import _load_img
    from dataset.image_dataset import _resize_chw

    path.parent.mkdir(parents=True, exist_ok=True)
    img = _load_img(Path(img_path))
    chw = _resize_chw(img, int(image_size))
    hwc = rgb_to_srgb(chw.permute(1, 2, 0).clamp(0, 1)).numpy()
    arr = (hwc * 255.0).round().astype("uint8")
    imageio.imwrite(str(path), arr)


@torch.no_grad()
def save_gt_frame(path: Path, image_chw, mask_chw=None, *, background: float = 1.0):
    """Linear RGB CHW [0,1] → PNG on white (mask composite)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    x = image_chw.detach().float().clamp(0, 1).cpu()
    if x.ndim == 4:
        x = x[0]
    hwc = rgb_to_srgb(x.permute(1, 2, 0)).numpy()
    if mask_chw is not None:
        m = mask_chw.detach().float().clamp(0, 1).cpu()
        if m.ndim == 4:
            m = m[0]
        if m.ndim == 3 and m.shape[0] == 1:
            m = m[0]
        alpha = m.numpy()[..., None]
        hwc = hwc * alpha + float(background) * (1.0 - alpha)
    arr = (hwc.clip(0, 1) * 255.0).round().astype("uint8")
    imageio.imwrite(str(path), arr)


@torch.no_grad()
def save_gsplat_frame(
    path: Path,
    rgb_chw,
    alpha_chw=None,
    *,
    transparent: bool = False,
    composited_on_black: bool = True,
):
    """
    Gsplat linear RGB CHW → PNG.

    ``transparent=False``: sRGB RGB on black (same as training ``eval_render`` pred).
    ``transparent=True``: straight-alpha RGBA PNG.

    When ``composited_on_black=True`` (default, ``renderer(composite=True)``), unpremultiply
    rgb/alpha for straight RGBA. When ``False`` (``renderer(composite=False)``), rgb is gsplat
    accum and alpha is separate — convert to sRGB without dividing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = rgb_chw.detach().float().clamp(0, 1).cpu()
    if rgb.ndim == 4:
        rgb = rgb[0]
    hwc = rgb.permute(1, 2, 0)

    if transparent:
        if alpha_chw is None:
            raise ValueError("alpha_chw required when transparent=True")
        alpha = alpha_chw.detach().float().clamp(0, 1).cpu()
        if alpha.ndim == 4:
            alpha = alpha[0]
        if alpha.ndim == 3 and alpha.shape[0] == 1:
            alpha = alpha[0]
        a = alpha.numpy()
        if composited_on_black:
            eps = 1e-6
            straight = hwc.numpy() / np.maximum(a[..., None], eps)
            straight = rgb_to_srgb(torch.from_numpy(straight).clamp(0, 1)).numpy()
        else:
            straight = rgb_to_srgb(hwc).numpy()
        rgba = np.concatenate([straight, a[..., None]], axis=-1)
        arr = (rgba.clip(0, 1) * 255.0).round().astype("uint8")
        imageio.imwrite(str(path), arr)
        return

    hwc_srgb = rgb_to_srgb(hwc).numpy()
    arr = (hwc_srgb.clip(0, 1) * 255.0).round().astype("uint8")
    imageio.imwrite(str(path), arr)


def assemble_tracking_videos(
    output_root: Path,
    *,
    fps: int = 25,
    video_codec: str | None = None,
    scenes: list[str] | None = None,
) -> list[Path]:
    """Encode ``tracking_render_{scene}_{gt,rgb,raw,personalized}/`` → ``tracking_video_*.mp4``."""
    output_root = Path(output_root)
    scene_list = scenes if scenes is not None else discover_tracking_scenes(output_root)
    written: list[Path] = []
    for scene in scene_list:
        for tag in TRACKING_FRAME_TAGS:
            frame_dir = tracking_frame_dir(output_root, scene, tag)
            if not frame_dir.is_dir():
                continue
            frames = collect_png_frames(frame_dir)
            if len(frames) == 0:
                continue
            out_mp4 = output_root / f"tracking_video_{scene}_{tag}.mp4"
            images_to_mp4(frames, out_mp4, fps=fps, video_codec=video_codec)
            written.append(out_mp4)
            print(f"  video: {out_mp4.name} ({len(frames)} frames @ {fps} fps)")
    return written
