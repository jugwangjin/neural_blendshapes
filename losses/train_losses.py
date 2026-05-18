"""Unified training losses for MP + UVH + 3DGS stack."""

import torch

from losses.eye_uv_barrier import soft_uv_box_barrier
from losses.gaussian_regularization import loss_opacity, loss_scale
from losses.h_regularization import loss_h_anchor_surface, loss_h_semantic
from losses.segmentation import loss_segmentation_logits, loss_segmentation_soft
from losses.iris_landmark import loss_iris_landmarks_2d
from losses.rgb import l1_loss


def loss_mediapipe_landmarks_2d(pred_xyz, mp_uv, mp_embedding, faces, camera, image_size, mp_valid=None):
    """
    pred_xyz: [B, V, 3] mesh vertices (world)
    mp_uv: [B, 478, 2] in [0, 1]
    mp_embedding: dict with ict_lmk_face_idx, ict_lmk_b_coords, mp_landmark_indices
    """
    from utils.barycentric import vertices2landmarks

    face_idx = torch.tensor(mp_embedding["ict_lmk_face_idx"], dtype=torch.long, device=pred_xyz.device)
    bary = torch.tensor(mp_embedding["ict_lmk_b_coords"], dtype=torch.float32, device=pred_xyz.device)
    lmk_xyz = vertices2landmarks(pred_xyz, faces, face_idx, bary)

    proj = camera.project_world_points(lmk_xyz.reshape(-1, 3)).reshape(pred_xyz.shape[0], -1, 2)
    pred_uv = proj / image_size
    mp_ids = mp_embedding["mp_landmark_indices"]
    n = min(pred_uv.shape[1], mp_uv.shape[1], len(mp_ids))
    target = mp_uv[:, :n, :]
    pred = pred_uv[:, :n, :]
    err = (pred - target).pow(2)
    if mp_valid is not None:
        w = mp_valid[:, :n, None]
        err = err * w
    return err.mean()


def _get_w(cfg, key, default=0.0):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


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
    expr_deform=None,
    expr_delta=None,
):
    losses = {}

    if render is not None and "rgb" in render and batch.get("image") is not None:
        losses["rgb"] = l1_loss(render["rgb"], batch["image"])

    if batch.get("mask") is not None and render is not None and "alpha" in render:
        losses["mask"] = (render["alpha"] - batch["mask"]).pow(2).mean()

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

    if batch.get("mp_landmarks_2d") is not None:
        losses["mp_lmk"] = loss_mediapipe_landmarks_2d(
            avatar_out.get("mesh_xyz", avatar_out["xyz"].unsqueeze(0)),
            batch["mp_landmarks_2d"],
            mp_embedding,
            ict_faces,
            camera,
            cfg.image_size,
            mp_valid=batch.get("mp_valid"),
        )

    if avatar_out.get("iris_control_xyz") is not None and batch.get("mp_landmarks_2d") is not None:
        iris_xyz = avatar_out["iris_control_xyz"]
        if iris_xyz.ndim == 3:
            iris_xyz = iris_xyz[0]
        mp_uv = batch["mp_landmarks_2d"]
        if mp_uv.ndim == 3:
            mp_uv = mp_uv[0]
        losses["iris"] = loss_iris_landmarks_2d(iris_xyz, mp_uv, camera, cfg.image_size)

    n_face = avatar_out["face"]["xyz"].shape[0]
    eyeball_mask = torch.zeros(avatar_out["h"].shape[0], dtype=torch.bool, device=avatar_out["h"].device)
    eyeball_mask[n_face:] = True
    if avatar_out.get("sem_prob") is not None and _get_w(cfg, "w_h", 0.0) > 0:
        losses["h"] = loss_h_semantic(
            avatar_out["h"],
            avatar_out["sem_prob"],
            class_sigma=getattr(cfg, "h_class_sigma", None),
            class_weight=getattr(cfg, "h_class_weight", None),
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

    face_mod = avatar_out.get("face")
    if (
        face_mod is not None
        and getattr(face_mod, "sem_logits", None) is not None
        and getattr(face_mod, "sem_anchor", None) is not None
        and _get_w(cfg, "w_sem_anchor", 0.0) > 0
    ):
        from rendering.gaussian_semantics import loss_semantic_anchor

        losses["sem_anchor"] = loss_semantic_anchor(
            face_mod.sem_logits, face_mod.sem_anchor, face_mod.sem_frozen_dims
        )

    if corr is not None and corr.get("gamma") is not None and _get_w(cfg, "w_gamma_prior") > 0:
        losses["gamma_prior"] = torch.log(corr["gamma"].clamp(min=1e-4)).pow(2).mean()

    if corr is not None and _get_w(cfg, "w_pose_prior") > 0:
        pr = corr["pose_residual"]
        tr = corr["translation_residual"]
        losses["pose_prior"] = pr.pow(2).mean() + tr.pow(2).mean()

    if corr is not None and _get_w(cfg, "w_gaze_residual", 0.0) > 0:
        from utils.gaze_uv import gaze_residual_prior_loss

        losses["gaze_residual"] = gaze_residual_prior_loss(corr["gaze_residual_left"]) + gaze_residual_prior_loss(
            corr["gaze_residual_right"]
        )

    if expr_deform is not None and expr_delta is not None:
        c_raw = corr.get("coeffs_raw", corr["coeffs"])
        reg = expr_deform.regularization_loss(corr["coeffs"], c_raw)
        if _get_w(cfg, "w_expr_neutral", 0.0) > 0:
            losses["expr_neutral"] = reg["expr_neutral"]
        if _get_w(cfg, "w_expr_leak", 0.0) > 0:
            losses["expr_leak"] = reg["expr_leak"]
        if _get_w(cfg, "w_expr_amp", 0.0) > 0:
            losses["expr_amp"] = reg["expr_amp"]
        if _get_w(cfg, "w_expr_deform_reg", 0.0) > 0:
            losses["expr_deform_reg"] = (
                reg["expr_neutral"] + reg["expr_leak"] + reg["expr_amp"]
            )

    w_tpl = _get_w(cfg, "w_template_smooth", _get_w(cfg, "w_identity_smooth", 0.0))
    if deformer is not None and w_tpl > 0:
        off = deformer.template_offset
        losses["template_smooth"] = off.pow(2).mean()

    w_mask = _get_w(cfg, "w_mask", _get_w(cfg, "w_mp_mask", 0.0))
    terms = [
        ("rgb", "w_rgb"),
        ("mp_lmk", "w_mp_lmk"),
        ("mask", None),
        ("seg", "w_seg"),
        ("iris", "w_iris"),
        ("h", "w_h"),
        ("eye_uv", "w_eye_uv_barrier"),
        ("scale", "w_scale"),
        ("opacity", "w_opacity"),
        ("gamma_prior", "w_gamma_prior"),
        ("pose_prior", "w_pose_prior"),
        ("gaze_residual", "w_gaze_residual"),
        ("expr_deform_reg", "w_expr_deform_reg"),
        ("expr_neutral", "w_expr_neutral"),
        ("expr_leak", "w_expr_leak"),
        ("expr_amp", "w_expr_amp"),
        ("sem_anchor", "w_sem_anchor"),
        ("template_smooth", None),
        ("identity_smooth", "w_identity_smooth"),
    ]
    ref = avatar_out["xyz"]
    total = torch.zeros((), device=ref.device, dtype=ref.dtype)
    for key, wkey in terms:
        if key not in losses:
            continue
        if key == "mask":
            w = w_mask
        elif key == "template_smooth":
            w = w_tpl
        else:
            w = _get_w(cfg, wkey, 0.0)
        total = total + w * losses[key]
    losses["total"] = total
    return losses
