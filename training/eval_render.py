"""Render eval scenes after each training stage (no grad)."""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import build_train_dataset, collate_batch, move_batch_to_device


def _chw_to_bgr_uint8(t):
    x = t.detach().float().clamp(0, 1).cpu()
    if x.ndim == 4:
        x = x[0]
    hwc = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(hwc, cv2.COLOR_RGB2BGR)


@torch.no_grad()
def render_eval_set(
    cfg,
    spec,
    tracker,
    avatar,
    renderer,
    camera,
    device,
    *,
    out_dir: Path,
    global_step: int,
    max_frames: int = 0,
    eval_loader=None,
):
    """
    Save ``gt | pred`` PNGs under ``out_dir / {stage_name} /``.

    ``max_frames``: 0 = all eval frames; else cap count.
    """
    if eval_loader is None:
        eval_ds = build_train_dataset(cfg, train=False)
        n = len(eval_ds)
        if n == 0:
            split = getattr(cfg, "flare_eval_split", "test")
            print(
                f"eval render skipped [{spec.name}]: no frames in "
                f"{cfg.input_dir}/{split}/image"
            )
            return

        if max_frames > 0:
            n = min(n, max_frames)

        loader = DataLoader(
            eval_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_batch,
            num_workers=0,
        )
    else:
        loader = eval_loader
        eval_ds = loader.dataset
        n = len(eval_ds)
        if n == 0:
            return
        if max_frames > 0:
            n = min(n, max_frames)

    stage_dir = Path(out_dir) / spec.name / f"step_{global_step:06d}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for batch in loader:
        if saved >= n:
            break

        batch = move_batch_to_device(batch, device)

        corr = tracker(
            mp_blendshape=batch["mp_blendshape"],
            mp_landmarks_2d=batch.get("mp_landmarks_2d"),
            mp_landmarks_3d=batch.get("mp_landmarks_3d"),
            world_to_cam=batch.get("world_to_cam"),
            mp_pose_raw=batch.get("mp_pose_raw"),
            mp_transform_matrix=batch.get("mp_transform_matrix"),
            force_gamma_one=spec.fix_gamma_at_one,
        )
        pose_weight_fixed = 1.0 if spec.pose_weight_one else None
        avatar_out = avatar(
            tracker_out=corr,
            apply_expression_deform=spec.train_expression_deform,
            use_pose_scale=spec.apply_pose_scale,
            pose_weight_fixed=pose_weight_fixed,
            rotate_about_centroid=spec.pose_rotate_about_centroid,
            pose_zero_tz=spec.pose_zero_tz,
        )
        render = renderer(avatar_out, camera, render_semantic=False)
        pred_bgr = _chw_to_bgr_uint8(render["rgb"])
        gt_bgr = _chw_to_bgr_uint8(batch["image"])

        paths = batch.get("path", [f"frame_{saved:05d}"])
        stem = Path(paths[0]).stem if paths else f"frame_{saved:05d}"
        out_path = stage_dir / f"{stem}_compare.png"
        compare = np.concatenate([gt_bgr, pred_bgr], axis=1)
        cv2.imwrite(str(out_path), compare)
        saved += 1

    print(f"eval render [{spec.name}]: {saved} frames -> {stage_dir}")
