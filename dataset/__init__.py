"""Dataset loaders for training (default: ``ImageDataset`` / image-split layout)."""

from dataset.collate import collate_batch, move_batch_to_device
from dataset.image_dataset import ImageDataset
from dataset.video_dataset import VideoDataset


def build_train_dataset(cfg, train=True):
    """
    Default ``dataset_type`` → ``ImageDataset`` (``dataset/image_dataset.py``).

    Layout: ``cfg.input_dir / {scene} / {image,mask,semantic,...}`` — split fields may be a
    scene name or list of scenes (merged index).
    (on-disk folder convention only — not the archived shader stack in ``legacy/``).
    Legacy ``mp_npz`` → ``VideoDataset`` (per-scene npz under ``mp_cache_dir``).
    """
    kind = getattr(cfg, "dataset_type", "flare")
    if kind in ("flare", "image"):
        return ImageDataset(
            cfg,
            train=train,
            synthetic_if_empty=not train,
            distribution_boost=train,
        )
    if kind == "mp_npz":
        return VideoDataset(
            cfg,
            train=train,
            synthetic_if_empty=not train,
            au_active_boost=False,
        )
    raise ValueError(f"unknown dataset_type={kind!r}; use 'flare' (ImageDataset) or 'mp_npz'")


__all__ = [
    "ImageDataset",
    "VideoDataset",
    "build_train_dataset",
    "collate_batch",
    "move_batch_to_device",
]
