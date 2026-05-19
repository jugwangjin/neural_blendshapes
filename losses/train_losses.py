"""Unified training losses for MP + UVH + 3DGS stack."""

import torch

from losses.eye_uv_barrier import soft_uv_box_barrier
from losses.gaussian_regularization import loss_opacity, loss_scale
from losses.h_regularization import loss_h_anchor_surface, loss_h_semantic
from losses.segmentation import loss_segmentation_logits, loss_segmentation_soft
from losses.iris_landmark import loss_iris_landmarks_2d
from losses.mediapipe_landmark_478 import loss_mediapipe_landmarks_478
from losses.rgb import l1_loss
from losses.silhouette import loss_silhouette


def _get_w(cfg, key, default=0.0):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _silhouette_weight(cfg):
    """``w_silhouette`` primary; ``w_mask`` / ``w_mp_mask`` kept as aliases."""
    w = _get_w(cfg, "w_silhouette", 0.0)
    if w <= 0.0:
        w = _get_w(cfg, "w_mask", 0.0)
    if w <= 0.0:
        w = _get_w(cfg, "w_mp_mask", 0.0)
    return w


def compute_losses(
    cfg,
    batch,
    render,
    avatar_out,
    camera,
    mp_embedding,
    ict_faces,
    corr=None,
    deformer=None,
    expr_delta=None,
    avatar=None,
):
    losses = {}

    if render is not None and "rgb" in render and batch.get("image") is not None:
        losses["rgb"] = l1_loss(render["rgb"], batch["image"])

    if render is not None and "alpha" in render:
        w_sil = _silhouette_weight(cfg)
        if batch.get("mask") is None:
            if w_sil > 0.0:
                raise ValueError(
                    "silhouette loss enabled (w_silhouette>0) but batch has no 'mask' — "
                    "check dataset *_mask.png caches"
                )
        else:
            losses["silhouette"] = loss_silhouette(render["alpha"], batch["mask"])

    if render is not None and (
        render.get("semantic_prob") is not None or render.get("semantic") is not None
    ):
        pred_sem = render.get("semantic_prob", render["semantic"])
        if batch.get("seg_label") is not None and _get_w(cfg, "w_seg", 0.0) > 0:
            target = batch["seg_label"]
            if target.ndim == 4:
                target = target[0]
            if target.ndim == 3:
                target = target[0]
            losses["seg"] = loss_segmentation_logits(pred_sem, target)
        elif batch.get("seg_onehot") is not None and _get_w(cfg, "w_seg", 0.0) > 0:
            target = batch["seg_onehot"]
            if target.ndim == 3:
                target = target.unsqueeze(0)
            losses["seg"] = loss_segmentation_soft(pred_sem, target)

    if batch.get("mp_landmarks_2d") is not None and mp_embedding is not None:
        mesh_xyz = avatar_out.get("mesh_xyz")
        if mesh_xyz is None:
            raise ValueError("mp_lmk loss requires avatar_out['mesh_xyz'] (ICT deformed vertices)")
        losses["mp_lmk"] = loss_mediapipe_landmarks_478(
            mesh_xyz,
            ict_faces,
            batch["mp_landmarks_2d"],
            mp_embedding,
            camera,
            cfg.image_size,
            mp_valid=batch.get("mp_valid"),
            mp_ids=mp_embedding.get("_mp_ids"),
            face_idx=mp_embedding.get("_face_idx"),
            bary=mp_embedding.get("_bary"),
        )

    iris_xyz = avatar_out.get("iris_control_xyz")
    if (
        iris_xyz is not None
        and iris_xyz.numel() > 0
        and batch.get("mp_landmarks_2d") is not None
    ):
        if iris_xyz.ndim == 3:
            iris_xyz = iris_xyz[0]
        mp_uv = batch["mp_landmarks_2d"]
        if mp_uv.ndim == 3:
            mp_uv = mp_uv[0]
        losses["iris"] = loss_iris_landmarks_2d(iris_xyz, mp_uv, camera, cfg.image_size)

    n_face = avatar_out["surface"]["xyz"].shape[0]
    eyeball_mask = torch.zeros(avatar_out["h"].shape[0], dtype=torch.bool, device=avatar_out["h"].device)
    eyeball_mask[n_face:] = True
    if avatar_out.get("sem_prob") is not None and _get_w(cfg, "w_h", 0.0) > 0:
        h_scale = None
        if avatar is not None and getattr(avatar, "h_sigma_scale", None) is not None:
            h_scale = avatar.h_sigma_scale
        losses["h"] = loss_h_semantic(
            avatar_out["h"],
            avatar_out["sem_prob"],
            class_sigma=getattr(cfg, "h_class_sigma", None),
            class_weight=getattr(cfg, "h_class_weight", None),
            h_sigma_scale=h_scale,
        )
    else:
        losses["h"] = loss_h_anchor_surface(
            avatar_out["h"],
            avatar_out["is_anchor_surface"],
            eyeball_mask=eyeball_mask,
        )

    losses["eye_uv"] = soft_uv_box_barrier(avatar_out["eyes"]["left"]["uv"]) + soft_uv_box_barrier(
        avatar_out["eyes"]["right"]["uv"]
    )

    losses["scale"] = loss_scale(avatar_out["scale"])
    losses["opacity"] = loss_opacity(avatar_out["opacity"])

    if (
        avatar is not None
        and getattr(avatar, "sem_logits", None) is not None
        and getattr(avatar, "sem_anchor", None) is not None
        and _get_w(cfg, "w_sem_anchor", 0.0) > 0
    ):
        from rendering.gaussian_semantics import loss_semantic_anchor

        losses["sem_anchor"] = loss_semantic_anchor(
            avatar.sem_logits, avatar.sem_anchor, avatar.sem_frozen_dims
        )

    if corr is not None and corr.get("gamma") is not None and _get_w(cfg, "w_gamma_prior") > 0:
        losses["gamma_prior"] = torch.log(corr["gamma"].clamp(min=1e-4)).pow(2).mean()

    if corr is not None and _get_w(cfg, "w_pose_prior") > 0:
        pr = corr["pose_residual"]
        tr = corr["translation_residual"]
        losses["pose_prior"] = pr.pow(2).mean() + tr.pow(2).mean()
        if _get_w(cfg, "apply_pose_scale", False) and corr.get("pose_scale") is not None:
            losses["pose_prior"] = losses["pose_prior"] + (corr["pose_scale"] - 1.0).pow(2).mean()

    if corr is not None and _get_w(cfg, "w_pose_tz", 0.0) > 0:
        losses["pose_tz"] = corr["translation_residual"][..., 2].pow(2).mean()

    if corr is not None and _get_w(cfg, "w_gaze_residual", 0.0) > 0:
        from utils.gaze_uv import gaze_residual_prior_loss

        losses["gaze_residual"] = gaze_residual_prior_loss(corr["gaze_residual_left"]) + gaze_residual_prior_loss(
            corr["gaze_residual_right"]
        )

    if deformer is not None and expr_delta is not None and corr is not None:
        c_raw = corr.get("coeffs_raw", corr["coeffs"])
        reg = deformer.regularization_loss(corr["coeffs"], c_raw, expr_delta=expr_delta)
        if _get_w(cfg, "w_expr_neutral", 0.0) > 0:
            losses["expr_neutral"] = reg["expr_neutral"]
        if _get_w(cfg, "w_expr_leak", 0.0) > 0:
            losses["expr_leak"] = reg["expr_leak"]
        if _get_w(cfg, "w_expr_amp", 0.0) > 0:
            losses["expr_amp"] = reg["expr_amp"]
        if _get_w(cfg, "w_expr_amp", 0.0) > 0 and "expr_socket" in reg:
            losses["expr_socket"] = reg["expr_socket"]
        if _get_w(cfg, "w_expr_deform_reg", 0.0) > 0:
            losses["expr_deform_reg"] = (
                reg["expr_neutral"] + reg["expr_leak"] + reg["expr_amp"] + reg["expr_socket"]
            )

    w_tpl = _get_w(cfg, "w_template_smooth", _get_w(cfg, "w_identity_smooth", 0.0))
    if deformer is not None and w_tpl > 0:
        losses["template_smooth"] = deformer.template_regularization_loss()

    w_silhouette = _silhouette_weight(cfg)
    terms = [
        ("rgb", "w_rgb"),
        ("mp_lmk", "w_mp_lmk"),
        ("silhouette", None),
        ("seg", "w_seg"),
        ("iris", "w_iris"),
        ("h", "w_h"),
        ("eye_uv", "w_eye_uv_barrier"),
        ("scale", "w_scale"),
        ("opacity", "w_opacity"),
        ("gamma_prior", "w_gamma_prior"),
        ("pose_prior", "w_pose_prior"),
        ("pose_tz", "w_pose_tz"),
        ("gaze_residual", "w_gaze_residual"),
        ("expr_deform_reg", "w_expr_deform_reg"),
        ("expr_neutral", "w_expr_neutral"),
        ("expr_leak", "w_expr_leak"),
        ("expr_amp", "w_expr_amp"),
        ("expr_socket", "w_expr_amp"),
        ("sem_anchor", "w_sem_anchor"),
        ("template_smooth", None),
        ("identity_smooth", "w_identity_smooth"),
    ]
    ref = avatar_out["xyz"]
    total = torch.zeros((), device=ref.device, dtype=ref.dtype)
    for key, wkey in terms:
        if key not in losses:
            continue
        if key == "silhouette":
            w = w_silhouette
        elif key == "template_smooth":
            w = w_tpl
        else:
            w = _get_w(cfg, wkey, 0.0)
        total = total + w * losses[key]
    losses["total"] = total
    return losses
