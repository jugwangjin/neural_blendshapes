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
    "full_face_region_mask",
    "h_reg_label_eye_occlusion",
    "h_reg_seg_face",
    "h_reg_seg_mouth",
    "h_reg_seg_neck",
    "h_reg_seg_hair",
    "h_reg_seg_glasses",
    "h_reg_seg_misc",
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


def batch_has_gt_normal(batch) -> bool:
    """True if the collated batch has at least one frame with a GT normal file."""
    if batch.get("gt_normal") is None:
        return False
    valid = batch.get("gt_normal_valid")
    if valid is None:
        return True
    return bool(valid.reshape(-1).sum().item() > 0)


def collate_batch(items):
    """Training stack (``image``, ``mask``, MP tensors, optional seg)."""
    batch = {}
    skip_keys = {"gt_normal", "gt_normal_valid"}
    all_keys = set()
    for x in items:
        all_keys.update(x.keys())
    all_keys -= skip_keys

    for key in sorted(all_keys):
        if key in ("path", "frame_idx", "dataset_frame_idx", "img_path", "frame_name"):
            batch[key] = [x[key] for x in items]
            continue
        vals = [x[key] for x in items if key in x]
        if len(vals) != len(items):
            continue
        v0 = vals[0]
        if isinstance(v0, torch.Tensor):
            batch[key] = torch.stack(vals, dim=0)
        elif isinstance(v0, np.ndarray):
            batch[key] = torch.from_numpy(np.stack(vals, axis=0))
        else:
            batch[key] = vals
    if "image" not in batch and "img" in batch:
        batch["image"] = batch["img"]

    valids = [bool(x.get("gt_normal_valid", False)) for x in items]
    batch["gt_normal_valid"] = torch.tensor(valids, dtype=torch.float32)
    if any(valids):
        ref = next(x["gt_normal"] for x in items if x.get("gt_normal_valid"))
        gt_stack = []
        for x, ok in zip(items, valids):
            if ok:
                gt_stack.append(x["gt_normal"])
            else:
                gt_stack.append(torch.zeros_like(ref))
        batch["gt_normal"] = torch.stack(gt_stack, dim=0)
    return batch
