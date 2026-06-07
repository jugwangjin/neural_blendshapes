"""
FLARE ``semantic/*.png`` — raw part-id map + derived masks for image-space losses.

Part ids (see ``dataset_util._load_semantic`` comments):
  0 bg, 1 skin, 2/3 brows, 4/5 eyes, 6 glasses, 7/8 ears, 9 earring, 10 nose,
  11 mouth interior, 12/13 lips, 14 neck, 15 necklace, 16 cloth, 17 hair, 18 hat

H regularization (image-space ``accum_h``):
  - **Label path**: ``h_reg_label_eye_occlusion`` — FLARE parts 4/5 only.
  - **Segmentation path** (legacy 8-ch ``semantic8``): face mesh-stick, mouth, loose tiers.
"""

from pathlib import Path

import cv2
import imageio
import numpy as np
import torch

# FLARE part ids with no ICT / 3-class supervision (skip in seg losses).
FLARE_PART_EYEGLASSES = 6
SEG_EXCLUDE_FLARE_PARTS = frozenset({FLARE_PART_EYEGLASSES})

H_REG_LABEL_EYE_OCCLUSION_PARTS = frozenset({4, 5})

H_REG_SEG_LOOSE_HAIR_PARTS = frozenset({17, 18})
H_REG_SEG_LOOSE_GLASSES_PARTS = frozenset({6})
H_REG_SEG_LOOSE_MISC_PARTS = frozenset({7, 8, 9, 15})
H_REG_SEG_NECK_PARTS = frozenset({14})

# ICT-FaceKit full-face-area (FLARE): skin + brows + eyes + nose + lips + mouth interior + ears + neck.
FULL_FACE_REGION_PARTS = frozenset({1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14})

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


def _part_mask(part_label: torch.Tensor, parts: frozenset[int]) -> torch.Tensor:
    dev = part_label.device
    return torch.isin(
        part_label,
        torch.tensor(sorted(parts), device=dev, dtype=part_label.dtype),
    )


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
    part = np.rint(_resize_part_label(img, image_size)).astype(np.int64)
    return torch.tensor(part, dtype=torch.long)


def h_reg_label_eye_occlusion_mask(part_label: torch.Tensor) -> torch.Tensor:
    """FLARE part ids 4/5 → ``[1,H,W]`` float mask (semantic **label** h-reg path)."""
    return _part_mask(part_label, H_REG_LABEL_EYE_OCCLUSION_PARTS).float()[None]


def h_reg_masks_from_segmentation(
    semantic8: torch.Tensor,
    part_label: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    H-reg tiers from legacy 8-ch segmentation + part ids for classes sem8 does not split.

    ``semantic8``: ``[H,W,8]`` from ``dataset_util._load_semantic``.
    """
    sem = semantic8.float()
    if sem.ndim != 3 or sem.shape[-1] != 8:
        raise ValueError(f"semantic8 must be [H,W,8], got {tuple(sem.shape)}")

    face = sem[..., 1].clamp(0, 1)
    mouth = sem[..., 5].clamp(0, 1)

    hair = _part_mask(part_label, H_REG_SEG_LOOSE_HAIR_PARTS)
    glasses = _part_mask(part_label, H_REG_SEG_LOOSE_GLASSES_PARTS)
    misc = _part_mask(part_label, H_REG_SEG_LOOSE_MISC_PARTS)
    neck = _part_mask(part_label, H_REG_SEG_NECK_PARTS)

    return {
        "h_reg_seg_face": face[None],
        "h_reg_seg_mouth": mouth[None],
        "h_reg_seg_neck": neck.float()[None],
        "h_reg_seg_hair": hair.float()[None],
        "h_reg_seg_glasses": glasses.float()[None],
        "h_reg_seg_misc": misc.float()[None],
    }


def region_masks_from_part_label(part_label: torch.Tensor):
    """Non-h-reg derived masks (face region, matting fg, legacy skin_tight)."""
    dev = part_label.device
    pl = part_label

    skin_tight = torch.isin(
        pl,
        torch.tensor([1, 2, 3, 10, 12, 13], device=dev, dtype=pl.dtype),
    )
    full_face = _part_mask(pl, FULL_FACE_REGION_PARTS)
    fg = pl != 0
    return {
        "full_face_region_mask": full_face.float()[None],
        "skin_tight": skin_tight.float()[None],
        "fg": fg.float()[None],
        "h_reg_skip": (pl == 0).float()[None],
    }


def build_h_reg_masks(part_label: torch.Tensor, semantic8: torch.Tensor) -> dict[str, torch.Tensor]:
    out = h_reg_masks_from_segmentation(semantic8, part_label)
    out["h_reg_label_eye_occlusion"] = h_reg_label_eye_occlusion_mask(part_label)
    return out


def part_onehot(part_label: torch.Tensor, num_parts: int = 19) -> torch.Tensor:
    """``[K,H,W]`` float one-hot (K=num_parts)."""
    k = int(max(num_parts, int(part_label.max().item()) + 1))
    return torch.nn.functional.one_hot(part_label.long(), num_classes=k).permute(2, 0, 1).float()


def mask_gt_semantic_by_matting(sem: dict, mask: torch.Tensor) -> dict:
    """
    Restrict FLARE semantic GT to the same support as matting α (when tight mask is on).

    Outside ``mask``: ``part_label`` → 0; h-reg / region maps recomputed.
    """
    m = mask[0] if mask.ndim == 3 else mask
    m = m.float()
    fg = m > 0.5

    pl = sem["part_label"]
    pl = torch.where(fg, pl, torch.zeros_like(pl))
    sem["part_label"] = pl

    derived = region_masks_from_part_label(pl)
    for key in (
        "full_face_region_mask",
        "skin_tight",
        "fg",
        "h_reg_skip",
    ):
        if key in derived:
            sem[key] = derived[key]
    sem["skin_mask"] = derived["skin_tight"]

    if "semantic8" in sem:
        sem8 = sem["semantic8"]
        if sem8.ndim == 3 and sem8.shape[-1] == 8:
            sem8 = sem8 * m.unsqueeze(-1)
        else:
            sem8 = sem8 * m.unsqueeze(0)
        sem["semantic8"] = sem8
        sem["seg_label"] = sem8.argmax(dim=-1).long()
        sem.update(build_h_reg_masks(pl, sem8))

    sem["part_onehot"] = part_onehot(pl)
    return sem


def load_flare_semantic_tensors(path, image_size: int):
    """
    Load semantic png → part label + legacy 8-ch soft map + h-reg masks.

    Legacy 8-ch kept for existing ``w_seg`` / ``skin_mask`` paths.
    """
    from dataset.dataset_util import _load_semantic

    part_label = load_semantic_part_label(path, image_size)
    sem8 = _load_semantic(path)
    if sem8.shape[0] != image_size or sem8.shape[1] != image_size:
        arr = sem8.numpy()
        arr = cv2.resize(arr, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        sem8 = torch.tensor(arr, dtype=torch.float32)

    regions = region_masks_from_part_label(part_label)
    h_reg = build_h_reg_masks(part_label, sem8)
    seg_label = sem8.argmax(dim=-1).long()
    out = {
        "part_label": part_label,
        "semantic8": sem8,
        "seg_label": seg_label,
        "skin_mask": regions["skin_tight"],
        "part_onehot": part_onehot(part_label),
    }
    out.update(regions)
    out.update(h_reg)
    return out


def flare_part_label_to_semantic_class(part_label: torch.Tensor) -> torch.Tensor:
    """
    Map FLARE part ids (0–18) → 3-class mesh seg ids for ``w_seg``.

    - ``mouth_interior``: FLARE 11 (oral cavity).
    - ``eye_occlusion``: FLARE 4/5 (r/l eye).
    - ``others``: skin, brows, nose, lips 12/13, ears, neck, cloth, hair/hat, …
    - **Eyeglasses (6)** → ``SEMANTIC_IGNORE_INDEX`` (no ICT geometry; skipped in seg loss).

    Out-of-range ids (bad png / colormap) → ``SEMANTIC_IGNORE_INDEX``.
    Background 0 is dropped via valid mask (``part_label != 0``).
    """
    from rendering.semantic import SEMANTIC_CLASS_INDEX, SEMANTIC_IGNORE_INDEX

    dev = part_label.device
    pl = part_label.long()
    others = SEMANTIC_CLASS_INDEX["others"]
    tbl = torch.full((19,), others, device=dev, dtype=torch.long)
    tbl[4] = SEMANTIC_CLASS_INDEX["eye_occlusion"]
    tbl[5] = SEMANTIC_CLASS_INDEX["eye_occlusion"]
    tbl[11] = SEMANTIC_CLASS_INDEX["mouth_interior"]
    tbl[FLARE_PART_EYEGLASSES] = SEMANTIC_IGNORE_INDEX
    in_range = (pl >= 0) & (pl <= 18)
    pl_safe = pl.clamp(0, 18)
    out = tbl[pl_safe]
    out = torch.where(in_range, out, torch.full_like(out, SEMANTIC_IGNORE_INDEX))
    return out


def load_flare_semantic_tensors_from_path(path, image_size: int):
    path = Path(path)
    if not path.is_file():
        return None
    return load_flare_semantic_tensors(path, image_size)
