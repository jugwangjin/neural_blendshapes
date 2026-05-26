"""
Landmark-only replay of metrical-tracker ``generate_dataset.py`` crop + squareify.

Uses ``metrical-tracker/image.py:get_bbox`` (same as dataset generator).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from processing.paths import METRICAL_ROOT

_METRICAL_IMAGE = METRICAL_ROOT / "image.py"
if not _METRICAL_IMAGE.is_file():
    raise FileNotFoundError(f"metrical-tracker image.py not found: {_METRICAL_IMAGE}")

_mt = str(METRICAL_ROOT)
if _mt not in sys.path:
    sys.path.insert(0, _mt)

from image import get_bbox  # noqa: E402


def metrical_crop_landmarks(
    lmks_uv: np.ndarray,
    image_hw: tuple[int, int],
    *,
    bb_scale: float = 2.5,
    out_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map 2D landmarks through crop (``get_bbox``) + pad-to-square + resize to ``out_size``.

    ``image_hw``: (height, width) as in metrical ``cfg.image_size``.
    Returns (lmks_out [N,2], bbox [xb_min, xb_max, yb_min, yb_max]).
    """
    lmks = np.asarray(lmks_uv, dtype=np.float64)
    h, w = int(image_hw[0]), int(image_hw[1])
    dummy = np.zeros((h, w, 3), dtype=np.uint8)
    bbox = get_bbox(dummy, lmks.astype(np.int32), bb_scale=float(bb_scale))
    xb_min, xb_max, yb_min, yb_max = bbox
    out = lmks.copy()
    out[:, 0] -= float(xb_min)
    out[:, 1] -= float(yb_min)
    crop_w = float(xb_max - xb_min)
    crop_h = float(yb_max - yb_min)
    max_wh = max(crop_w, crop_h)
    hp = int((max_wh - crop_w) / 2)
    vp = int((max_wh - crop_h) / 2)
    out[:, 0] += hp
    out[:, 1] += vp
    scale = float(out_size) / max_wh
    out *= scale
    return out, bbox


def target_landmark_half_span(out_size: int, bb_scale: float) -> float:
    """
    If landmarks are symmetric about the crop center, max half-extent after resize is
    ``out_size / (2 * bb_scale)`` (see metrical ``get_bbox`` + square resize).
    """
    return float(out_size) / (2.0 * float(bb_scale))
