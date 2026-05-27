"""Batch collation for ``train.py`` (ImageDataset / mp_npz keys)."""

import numpy as np
import torch

# Tensor keys consumed by tracker / loss (must share training device).
LOSS_BATCH_TENSOR_KEYS = (
    "image",
    "mask",
    "mask_dist_out",
    "mask_dist_in",
    "mp_blendshape",
    "mp_blendshape_raw",
    "mp_landmarks_2d",
    "mp_landmarks_3d",
    "mp_valid",
    "mp_pose_raw",
    "mp_transform_matrix",
    "pose_feat",
    "landmark",
    "seg_label",
    "skin_mask",
    "h_reg_skin",
    "h_reg_eye",
    "h_reg_brow",
    "h_reg_misc",
    "h_reg_mouth",
    "semantic_fg",
    "part_label",
    "part_onehot",
    "seg_onehot",
    "world_to_cam",
)


def move_batch_to_device(batch, device):
    """Move every ``torch.Tensor`` in a collated batch to ``device``."""
    dev = torch.device(device)
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(dev, non_blocking=(dev.type == "cuda"))
        else:
            out[k] = v
    return out


def collate_batch(items):
    """Training stack (``image``, ``mask``, MP tensors, optional seg)."""
    batch = {}
    for key in items[0]:
        if key in ("path", "frame_idx", "img_path", "frame_name"):
            batch[key] = [x[key] for x in items]
            continue
        vals = [x[key] for x in items]
        v0 = vals[0]
        if isinstance(v0, torch.Tensor):
            batch[key] = torch.stack(vals, dim=0)
        elif isinstance(v0, np.ndarray):
            batch[key] = torch.from_numpy(np.stack(vals, axis=0))
        else:
            batch[key] = vals
    if "image" not in batch and "img" in batch:
        batch["image"] = batch["img"]
    return batch
