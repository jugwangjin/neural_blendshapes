"""
INSTA-style matting mask tightening.

Face-parsing (BiSeNet, 19 classes, CelebAMask-HQ labels) writes ``seg_mask/*.png`` as
pseudo-color maps. INSTA ``postprocess.py`` removes pixels where the saved map has
``R == 90`` in BGR — that color encodes parsing class **16 = cloth**.

FLARE ``semantic/*.png`` uses the same part ids (0=bg, 1=skin, …, 16=cloth, 17=hair).

Tight mask: ``matting_alpha * (1 - exclude_region)``, with median + dilation on the
exclude region (same as INSTA postprocess).
"""

from pathlib import Path

import numpy as np
import torch

# Shared label id (FLARE / CelebAMask-HQ / INSTA face-parsing patch).
FLARE_PART_CLOTH = 16
# BGR ``seg_mask`` R-channel value for cloth in INSTA ``patches/face-parsing`` colormap.
INSTA_SEGMASK_CLOTH_R_VALUE = 90


def _morph_exclude_mask(exclude_hw: np.ndarray, median_ksize: int, dilate_iters: int) -> np.ndarray:
    import cv2
    from scipy import ndimage

    ex = exclude_hw.astype(np.uint8)
    if ex.ndim == 3:
        ex = ex[..., 0]
    if median_ksize and median_ksize > 1:
        k = int(median_ksize)
        if k % 2 == 0:
            k += 1
        ex = cv2.medianBlur(ex, k)
    if dilate_iters and dilate_iters > 0:
        ex = (ndimage.binary_dilation(ex, iterations=int(dilate_iters)) > 0).astype(np.uint8)
    return ex


def exclude_mask_from_insta_seg_mask(
    seg_mask_path: Path,
    median_ksize: int = 5,
    dilate_iters: int = 3,
    cloth_r_value: int = INSTA_SEGMASK_CLOTH_R_VALUE,
) -> np.ndarray:
    """
    INSTA ``postprocess.py``: ``seg_mask`` BGR, channel 2 (R) == 90 → cloth to remove.
    """
    import cv2

    img = cv2.imread(str(seg_mask_path), cv2.IMREAD_UNCHANGED)
    cloth = (img[:, :, 2] == int(cloth_r_value)).astype(np.uint8)
    return _morph_exclude_mask(cloth, median_ksize, dilate_iters)


def exclude_mask_from_flare_semantic(
    semantic_path: Path,
    height: int,
    width: int,
    exclude_part_ids,
    median_ksize: int = 5,
    dilate_iters: int = 3,
) -> np.ndarray:
    """FLARE part-id png → binary exclude mask (H, W) uint8."""
    import cv2
    import imageio

    part = imageio.imread(str(semantic_path), mode="F")
    if part.ndim == 3:
        part = part[..., 0]
    if part.shape[0] != height or part.shape[1] != width:
        part = cv2.resize(
            part.astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
    ids = np.asarray(list(exclude_part_ids), dtype=np.int64)
    exclude = np.isin(part.astype(np.int64), ids).astype(np.uint8)
    return _morph_exclude_mask(exclude, median_ksize, dilate_iters)


def apply_insta_tight_matting_mask(matting_mask, paths: dict, cfg) -> torch.Tensor:
    """
    ``matting_mask``: [H,W,1] float in [0,1]. Returns tightened mask.

    Prefers ``seg_mask`` (INSTA colormap) when present; else FLARE ``semantic`` part ids.
    """
    if not getattr(cfg, "tight_mask_from_semantic", False):
        return matting_mask

    median_ksize = int(getattr(cfg, "tight_mask_median_ksize", 5))
    dilate_iters = int(getattr(cfg, "tight_mask_exclude_dilate_iters", 3))
    exclude_parts = list(getattr(cfg, "tight_mask_exclude_parts", [FLARE_PART_CLOTH]))

    height = int(matting_mask.shape[0])
    width = int(matting_mask.shape[1])

    seg_mask_path = paths.get("seg_mask")
    if seg_mask_path is not None and Path(seg_mask_path).is_file():
        exclude = exclude_mask_from_insta_seg_mask(
            Path(seg_mask_path),
            median_ksize=median_ksize,
            dilate_iters=dilate_iters,
        )
    else:
        semantic_path = paths.get("semantic")
        if semantic_path is None or not Path(semantic_path).is_file():
            return matting_mask
        exclude = exclude_mask_from_flare_semantic(
            Path(semantic_path),
            height,
            width,
            exclude_parts,
            median_ksize=median_ksize,
            dilate_iters=dilate_iters,
        )

    import cv2

    if exclude.shape[0] != height or exclude.shape[1] != width:
        exclude = cv2.resize(exclude, (width, height), interpolation=cv2.INTER_NEAREST)

    m = matting_mask.numpy() if isinstance(matting_mask, torch.Tensor) else matting_mask
    if m.ndim == 3:
        m = m[..., 0]
    m = np.clip(m.astype(np.float32) * (1.0 - exclude.astype(np.float32)), 0.0, 1.0)
    if bool(getattr(cfg, "tight_mask_keep_mouth_interior", True)):
        semantic_path = paths.get("semantic")
        if semantic_path is not None and Path(semantic_path).is_file():
            import cv2
            import imageio
            from scipy import ndimage

            part = imageio.imread(str(semantic_path), mode="F")
            if part.ndim == 3:
                part = part[..., 0]
            if part.shape[0] != height or part.shape[1] != width:
                part = cv2.resize(
                    part.astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
            mouth = (part.astype(np.int64) == 11)
            keep_dilate = int(getattr(cfg, "tight_mask_keep_mouth_dilate_iters", 1))
            if keep_dilate > 0:
                mouth = ndimage.binary_dilation(mouth, iterations=keep_dilate)
            m = np.maximum(m, mouth.astype(np.float32))
    return torch.tensor(m[..., None], dtype=torch.float32)
