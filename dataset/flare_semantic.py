"""
FLARE ``semantic/*.png`` — raw part-id map + derived masks for image-space losses.

Part ids (see ``dataset_util._load_semantic`` comments):
  0 bg, 1 skin, 2/3 brows, 4/5 eyes, 6 glasses, 7/8 ears, 9 earring, 10 nose,
  11 mouth interior, 12/13 lips, 14 neck, 15 necklace, 16 cloth, 17 hair, 18 hat
"""

from pathlib import Path

import cv2
import imageio
import numpy as np
import torch

# Image-space h regularization tiers (FLARE part ids).
# strong: skin + eye | brow | mouth | misc (weakest)
H_REG_SKIN_PARTS = frozenset({1, 7, 8, 10, 12, 13, 14})
H_REG_EYE_PARTS = frozenset({4, 5})
H_REG_BROW_PARTS = frozenset({2, 3})
H_REG_MISC_PARTS = frozenset({6, 9, 15, 16, 17, 18})
H_REG_MOUTH_PARTS = frozenset({11})
H_REG_SKIP_PARTS = frozenset({0})

FLARE_PART_NAMES = {
    0: "background",
    1: "skin",
    2: "l_brow",
    3: "r_brow",
    4: "r_eye",
    5: "l_eye",
    6: "eye_g",
    7: "l_ear",
    8: "r_ear",
    9: "ear_r",
    10: "nose",
    11: "mouth",
    12: "u_lip",
    13: "l_lip",
    14: "neck",
    15: "neck_l",
    16: "cloth",
    17: "hair",
    18: "hat",
}


def _resize_part_label(part_hw: np.ndarray, image_size: int) -> np.ndarray:
    h, w = part_hw.shape[:2]
    if h == image_size and w == image_size:
        return part_hw.astype(np.int64)
    return cv2.resize(
        part_hw.astype(np.float32),
        (image_size, image_size),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.int64)


def load_semantic_part_label(path, image_size: int) -> torch.Tensor:
    """Raw FLARE part id per pixel, ``[H, W]`` int64."""
    img = imageio.imread(str(path), mode="F")
    if img.ndim == 3:
        img = img[..., 0]
    part = _resize_part_label(img, image_size)
    return torch.tensor(part, dtype=torch.long)


def masks_from_part_label(part_label: torch.Tensor):
    """
    Derived binary masks from part ids.

    Returns dict of ``[1,H,W]`` float32 in {0,1}:
      ``h_reg_skin``, ``h_reg_eye``, ``h_reg_brow``, ``h_reg_misc``, ``h_reg_mouth``, ``skin_tight``
    """
    dev = part_label.device
    pl = part_label

    def _mask(parts):
        return torch.isin(pl, torch.tensor(sorted(parts), device=dev, dtype=pl.dtype))

    skin = _mask(H_REG_SKIN_PARTS)
    eye = _mask(H_REG_EYE_PARTS)
    brow = _mask(H_REG_BROW_PARTS)
    misc = _mask(H_REG_MISC_PARTS)
    mouth = _mask(H_REG_MOUTH_PARTS)
    skin_tight = torch.isin(
        pl,
        torch.tensor([1, 2, 3, 10, 12, 13], device=dev, dtype=pl.dtype),
    )
    fg = pl != 0
    return {
        "h_reg_skin": skin.float()[None],
        "h_reg_eye": eye.float()[None],
        "h_reg_brow": brow.float()[None],
        "h_reg_misc": misc.float()[None],
        "h_reg_mouth": mouth.float()[None],
        "h_reg_skip": (pl == 0).float()[None],
        "skin_tight": skin_tight.float()[None],
        "fg": fg.float()[None],
    }


def part_onehot(part_label: torch.Tensor, num_parts: int = 19) -> torch.Tensor:
    """``[K,H,W]`` float one-hot (K=num_parts)."""
    k = int(max(num_parts, int(part_label.max().item()) + 1))
    return torch.nn.functional.one_hot(part_label.long(), num_classes=k).permute(2, 0, 1).float()


def load_flare_semantic_tensors(path, image_size: int):
    """
    Load semantic png → part label + legacy 8-ch soft map + h-reg masks.

    Legacy 8-ch kept for existing ``w_seg`` / ``skin_mask`` paths.
    """
    from dataset.dataset_util import _load_semantic

    part_label = load_semantic_part_label(path, image_size)
    sem8 = _load_semantic(path)
    if sem8.shape[0] != image_size or sem8.shape[1] != image_size:
        import cv2

        arr = sem8.numpy()
        arr = cv2.resize(arr, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        sem8 = torch.tensor(arr, dtype=torch.float32)

    masks = masks_from_part_label(part_label)
    seg_label = sem8.argmax(dim=-1).long()
    out = {
        "part_label": part_label,
        "semantic8": sem8,
        "seg_label": seg_label,
        "skin_mask": masks["skin_tight"],
        "part_onehot": part_onehot(part_label),
    }
    out.update(masks)
    return out


def load_flare_semantic_tensors_from_path(path, image_size: int):
    path = Path(path)
    if not path.is_file():
        return None
    return load_flare_semantic_tensors(path, image_size)
