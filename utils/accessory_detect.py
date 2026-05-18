"""Detect accessory presence in segmentation caches before enabling AccessoryGaussians."""

from pathlib import Path

import numpy as np
from PIL import Image

from rendering.semantic import SEMANTIC_CLASS_INDEX


def segmentation_has_accessory(
    segmentation_dir: Path,
    scene_names,
    accessory_class_id=None,
    min_pixel_ratio=0.0005,
    max_frames_per_scene=20,
):
    """
    Return True if any frame has enough accessory-labeled pixels.
    accessory_class_id: int label in seg PNG; default uses SEMANTIC_CLASS_INDEX['accessory'].
    """
    if accessory_class_id is None:
        accessory_class_id = SEMANTIC_CLASS_INDEX["accessory"]
    seg_dir = Path(segmentation_dir)
    if not seg_dir.is_dir():
        return False

    for scene in scene_names:
        scene_dir = seg_dir / scene
        if not scene_dir.is_dir():
            continue
        paths = sorted(scene_dir.glob("*_seg.png"))[:max_frames_per_scene]
        for p in paths:
            arr = np.array(Image.open(p))
            if arr.ndim == 3:
                arr = arr[..., 0]
            total = arr.size
            if total == 0:
                continue
            ratio = float((arr == accessory_class_id).sum()) / total
            if ratio >= min_pixel_ratio:
                return True
    return False
