"""Render eval scenes after each training stage (no grad)."""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import build_train_dataset, collate_batch, move_batch_to_device
from dataset.dataset_util import rgb_to_srgb


def _chw_to_bgra_uint8(t, alpha_t=None):
    """CHW linear RGB [0,1] → BGRA uint8 for display (sRGB gamma before *255)."""
    x = t.detach().float().clamp(0, 1).cpu()
    if x.ndim == 4:
        x = x[0]
    hwc = rgb_to_srgb(x.permute(1, 2, 0)).numpy()
    hwc = (np.clip(hwc, 0, 1) * 255.0).round().astype(np.uint8)
    bgr = cv2.cvtColor(hwc, cv2.COLOR_RGB2BGR)
    if alpha_t is not None:
        a = alpha_t.detach().float().clamp(0, 1).cpu()
        if a.ndim == 4:
            a = a[0]
        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]
        a_hw = (a.numpy() * 255.0).round().astype(np.uint8)
        if a_hw.shape[:2] == bgr.shape[:2]:
            return np.concatenate([bgr, a_hw[..., None]], axis=-1)
    a_hw = np.full(bgr.shape[:2] + (1,), 255, dtype=np.uint8)
    return np.concatenate([bgr, a_hw], axis=-1)


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
    Save eval PNGs under ``out_dir / {stage_name} / step_{global_step} /``:

    - ``{stem}_compare.png`` — GT | pred side-by-side (sRGB for display)
    - ``render/{stem}.png`` — pred only, no GT

    Dataset GT and renderer output are linear RGB; both are converted with
    ``rgb_to_srgb`` before writing PNGs.

    ``max_frames``: 0 = all eval frames; else cap count.
    """
    if eval_loader is None:
        eval_ds = build_train_dataset(cfg, train=False)
        n = len(eval_ds)
        if n == 0:
            from dataset.dataset_util import format_splits_label

            split_label = format_splits_label(getattr(cfg, "eval_split", "test"))
            print(
                f"eval render skipped [{spec.name}]: no frames in "
                f"{cfg.input_dir}/{{{split_label}}}/image"
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
    render_dir = stage_dir / "render"
    render_dir.mkdir(parents=True, exist_ok=True)

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
            enable_color_pose=getattr(spec, "train_color_pose", False),
            enable_color_expression=getattr(spec, "train_color_expression", False),
        )
        render = renderer(avatar_out, camera, render_semantic=False)
        pred_bgra = _chw_to_bgra_uint8(render["rgb"], render.get("alpha"))
        gt_bgra = _chw_to_bgra_uint8(batch["image"], batch.get("mask"))

        paths = batch.get("path", [f"frame_{saved:05d}"])
        stem = Path(paths[0]).stem if paths else f"frame_{saved:05d}"
        cv2.imwrite(str(stage_dir / f"{stem}_compare.png"), np.concatenate([gt_bgra, pred_bgra], axis=1))
        cv2.imwrite(str(render_dir / f"{stem}.png"), pred_bgra)
        saved += 1

    print(
        f"eval render [{spec.name}]: {saved} frames -> {stage_dir} "
        f"(compare + render/)"
    )
