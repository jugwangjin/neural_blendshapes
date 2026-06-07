"""Shared gsplat inference path (matches ``training/eval_render.py``)."""

from __future__ import annotations

import torch


@torch.no_grad()
def tracker_out_from_batch(tracker, batch, spec):
    """Same tracker call as ``render_eval_set``."""
    return tracker(
        mp_blendshape=batch["mp_blendshape"],
        mp_landmarks_2d=batch.get("mp_landmarks_2d"),
        mp_landmarks_3d=batch.get("mp_landmarks_3d"),
        world_to_cam=batch.get("world_to_cam"),
        mp_pose_raw=batch.get("mp_pose_raw"),
        mp_transform_matrix=batch.get("mp_transform_matrix"),
        force_gamma_one=spec.fix_gamma_at_one,
        use_global_translation_param=getattr(spec, "use_global_translation_param", False),
        additive_gamma_correction=getattr(spec, "additive_gamma_correction", False),
    )


@torch.no_grad()
def render_gsplat_from_tracker_out(
    avatar,
    renderer,
    camera,
    tracker_out,
    spec,
    *,
    composite: bool = True,
):
    """Same avatar + renderer path as ``render_eval_set``."""
    pose_weight_fixed = 1.0 if spec.pose_weight_one else None
    avatar_out = avatar(
        tracker_out=tracker_out,
        apply_expression_deform=spec.train_expression_deform,
        use_pose_scale=spec.apply_pose_scale,
        pose_weight_fixed=pose_weight_fixed,
        rotate_about_centroid=spec.pose_rotate_about_centroid,
        pose_zero_tz=spec.pose_zero_tz,
        enable_color_pose=getattr(spec, "train_color_pose", False),
        enable_color_expression=getattr(spec, "train_color_expression", False),
    )
    return renderer(avatar_out, camera, render_semantic=False, composite=composite)


@torch.no_grad()
def render_gsplat_from_batch(
    avatar,
    renderer,
    camera,
    tracker,
    batch,
    spec,
    *,
    composite: bool = True,
):
    corr = tracker_out_from_batch(tracker, batch, spec)
    return render_gsplat_from_tracker_out(
        avatar, renderer, camera, corr, spec, composite=composite
    )
