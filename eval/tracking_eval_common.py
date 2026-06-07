"""Shared helpers for tracker-only mesh visualization and evaluation."""

from __future__ import annotations

import torch


@torch.no_grad()
def mesh_from_tracker_out_pure(deformer, tracker_out, spec):
    """
    Tracker-only posed mesh: ICT expression + rigid pose from tracker.

    Excludes template_mlp, expr_mlp, and pose_weight_net (uniform w=1 rigid).
    """
    pose_scale = tracker_out.get("pose_scale") if spec.apply_pose_scale else None
    out = deformer(
        mp_coeffs_corr=tracker_out["coeffs"],
        expression_weights=tracker_out.get("ict_expression_weights"),
        pose_rotation_6d=tracker_out["pose_residual"],
        pose_translation=tracker_out["translation_residual"],
        pose_translation_global=tracker_out.get("translation_global"),
        pose_scale=pose_scale,
        c_eff=None,
        apply_expression_deform=False,
        apply_template_delta=False,
        pose_weight_fixed=1.0,
        rotate_about_centroid=spec.pose_rotate_about_centroid,
        pose_zero_tz=spec.pose_zero_tz,
    )
    return out["verts_posed"]
