"""Training losses — basic test stack + optional extras via weights."""

import torch

from losses.gaussian_regularization import (
    loss_geometry_log_scale,
    loss_opacity_toward_one,
    loss_scaling_regularization,
)
from losses.h_regularization import loss_h_image_space
from losses.mediapipe_landmark_478 import loss_mediapipe_landmarks_478
from losses.pie68_jaw_landmark import loss_pie68_jawline
from losses.rgb import l1_loss
from losses.silhouette import loss_silhouette, loss_silhouette_edt
from rendering.pack import surface_avatar_out


def _get_w(cfg, key, default=0.0):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _silhouette_weight(cfg):
    w = _get_w(cfg, "w_silhouette", 0.0)
    if w <= 0.0:
        w = _get_w(cfg, "w_mask", 0.0)
    if w <= 0.0:
        w = _get_w(cfg, "w_mp_mask", 0.0)
    return w


def _image_size(cfg, batch):
    sz = _get_w(cfg, "image_size", None)
    if sz is not None:
        return int(sz)
    img = batch.get("image")
    if img is not None and img.ndim >= 3:
        return int(img.shape[-1])
    return 512


def _align_batch_device(batch, device):
    dev = torch.device(device)
    return {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _surface_sem_prob(avatar_out):
    surf = avatar_out.get("surface")
    if surf is None:
        return None
    return surf.get("sem_prob")


def compute_losses(
    cfg,
    batch,
    render,
    avatar_out,
    camera,
    mp_lmk_emb,
    ict_faces,
    pie68_jaw_vertex_idx=None,
    corr=None,
    deformer=None,
    expr_delta=None,
    avatar=None,
    renderer=None,
):
    ref = avatar_out["xyz"]
    batch = _align_batch_device(batch, ref.device)
    image_size = _image_size(cfg, batch)
    losses = {}

    if render is not None and "rgb" in render and batch.get("image") is not None:
        losses["rgb"] = l1_loss(render["rgb"], batch["image"])

    w_sil = _silhouette_weight(cfg)
    if render is not None and "alpha" in render and w_sil > 0:
        use_edt = bool(_get_w(cfg, "silhouette_use_edt", False))
        if (
            use_edt
            and batch.get("mask_dist_out") is not None
            and batch.get("mask_dist_in") is not None
        ):
            losses["silhouette"] = loss_silhouette_edt(
                render["alpha"],
                batch["mask_dist_out"],
                batch["mask_dist_in"],
                w_ext=_get_w(cfg, "silhouette_edt_w_ext", 1.0),
                w_int=_get_w(cfg, "silhouette_edt_w_int", 1.0),
                max_dist_px=_get_w(cfg, "silhouette_edt_max_dist_px", 50.0),
            )
        elif batch.get("mask") is not None:
            losses["silhouette"] = loss_silhouette(render["alpha"], batch["mask"])

    if batch.get("mp_landmarks_2d") is not None and mp_lmk_emb is not None and _get_w(cfg, "w_mp_lmk") > 0:
        mesh_xyz = avatar_out.get("mesh_xyz")
        if mesh_xyz is None:
            raise ValueError("mp_lmk loss requires avatar_out['mesh_xyz']")
        losses["mp_lmk"] = loss_mediapipe_landmarks_478(
            mesh_xyz,
            ict_faces,
            batch["mp_landmarks_2d"],
            mp_lmk_emb,
            camera,
            image_size,
            mp_valid=batch.get("mp_valid"),
            iris_weight=_get_w(cfg, "mp_lmk_iris_weight", 2.5),
        )

    if (
        pie68_jaw_vertex_idx is not None
        and batch.get("landmark") is not None
        and _get_w(cfg, "w_pie68_jaw") > 0
    ):
        mesh_xyz = avatar_out.get("mesh_xyz")
        if mesh_xyz is None:
            raise ValueError("pie68_jaw loss requires avatar_out['mesh_xyz']")
        losses["pie68_jaw"] = loss_pie68_jawline(
            mesh_xyz,
            pie68_jaw_vertex_idx,
            batch["landmark"],
            camera,
            image_size,
        )

    if (
        renderer is not None
        and _get_w(cfg, "w_h") > 0
        and batch.get("h_reg_skin") is not None
    ):
        surf_out = surface_avatar_out(avatar_out)
        h_render = renderer.render_expected_signal(surf_out, camera, surf_out["h"])
        losses["h"] = loss_h_image_space(
            h_render["accum"],
            h_render["alpha"],
            batch,
            cfg,
            _get_w,
        )

    sem_prob = _surface_sem_prob(avatar_out)

    if avatar is not None and _get_w(cfg, "w_geometry") > 0:
        losses["geometry"] = loss_geometry_log_scale(
            avatar.surface.log_scale,
            max_scale=_get_w(cfg, "geometry_max_scale", 0.008),
        )

    if avatar is not None and _get_w(cfg, "w_scaling", 0.0) > 0:
        losses["scaling"] = loss_scaling_regularization(
            avatar.surface.log_scale,
            thresh_scaling_max=_get_w(cfg, "thresh_scaling_max", 0.008),
            thresh_scaling_ratio=_get_w(cfg, "thresh_scaling_ratio", 10.0),
        )

    if sem_prob is not None and avatar is not None and _get_w(cfg, "w_opacity") > 0:
        losses["opacity"] = loss_opacity_toward_one(
            avatar.surface.opacity,
            sem_prob,
            target=_get_w(cfg, "opacity_target", 1.0),
            w_skin=_get_w(cfg, "opacity_w_skin", 1.0),
            w_other=_get_w(cfg, "opacity_w_other", 0.05),
        )

    w_opacity_decay = _get_w(cfg, "w_opacity_decay", 0.0)
    if avatar is not None and w_opacity_decay > 0:
        codes = getattr(avatar.surface, "face_region_code", None)
        if codes is None:
            from utils.ict_regions import classify_surface_triangles_batch
            codes = classify_surface_triangles_batch(
                avatar.surface.face_idx,
                ict_faces,
                avatar.ict,
                avatar.surface.opacity.device,
            )
        # Apply L1 decay ONLY to head (3), face (4), sclera (5), and eye occlusion (6)
        # Protects mouth interior (0), mouth socket (1), eye socket (2)
        decay_mask = (codes == 3) | (codes == 4) | (codes == 5) | (codes == 6)
        if decay_mask.any():
            op_decay = torch.sigmoid(avatar.surface.opacity[decay_mask])
            losses["opacity_decay"] = op_decay.mean()

    if corr is not None and corr.get("gamma") is not None and _get_w(cfg, "w_gamma_prior") > 0:
        losses["gamma_prior"] = torch.log(corr["gamma"].clamp(min=1e-4)).pow(2).mean()

    if corr is not None and _get_w(cfg, "w_pose_prior") > 0:
        rot_delta = corr.get("pose_rotation_delta", corr["pose_residual"])
        losses["pose_prior"] = rot_delta.pow(2).mean() + corr["translation_residual"].pow(2).mean()

    if corr is not None and _get_w(cfg, "w_pose_tz", 0.0) > 0:
        losses["pose_tz"] = corr["translation_residual"][..., 2].pow(2).mean()

    if deformer is not None and expr_delta is not None and corr is not None and _get_w(cfg, "w_expr_deform_reg", 0.0) > 0:
        from losses.deformer_regularization import deformer_regularization_loss

        c_raw = corr.get("coeffs_raw", corr["coeffs"])
        reg = deformer_regularization_loss(deformer, corr["coeffs"], c_raw, expr_delta=expr_delta)
        losses["expr_deform_reg"] = (
            reg["expr_neutral"] + reg["expr_leak"] + reg["expr_amp"] + reg["expr_socket"]
        )

    w_tpl = _get_w(cfg, "w_template_smooth", _get_w(cfg, "w_identity_smooth", 0.0))
    if deformer is not None and w_tpl > 0:
        from losses.deformer_regularization import template_smooth_loss

        losses["template_smooth"] = template_smooth_loss(deformer)

    if (
        avatar is not None
        and getattr(avatar.surface, "sem_logits", None) is not None
        and _get_w(cfg, "w_sem_anchor", 0.0) > 0
    ):
        from rendering.gaussian_semantics import loss_semantic_anchor

        losses["sem_anchor"] = loss_semantic_anchor(
            avatar.surface.sem_logits, avatar.surface.sem_anchor, avatar.surface.sem_frozen_dims
        )

    if render is not None and batch.get("seg_label") is not None and _get_w(cfg, "w_seg", 0.0) > 0:
        from losses.segmentation import loss_segmentation_logits

        pred_sem = render.get("semantic_prob", render.get("semantic"))
        if pred_sem is not None:
            target = batch["seg_label"]
            if target.ndim == 4:
                target = target[0]
            if target.ndim == 3:
                target = target[0]
            target = target.to(device=pred_sem.device, dtype=torch.long)
            losses["seg"] = loss_segmentation_logits(pred_sem, target)

    terms = [
        ("rgb", "w_rgb"),
        ("silhouette", None),
        ("mp_lmk", "w_mp_lmk"),
        ("pie68_jaw", "w_pie68_jaw"),
        ("h", "w_h"),
        ("geometry", "w_geometry"),
        ("scaling", "w_scaling"),
        ("opacity", "w_opacity"),
        ("opacity_decay", "w_opacity_decay"),
        ("gamma_prior", "w_gamma_prior"),
        ("pose_prior", "w_pose_prior"),
        ("pose_tz", "w_pose_tz"),
        ("expr_deform_reg", "w_expr_deform_reg"),
        ("sem_anchor", "w_sem_anchor"),
        ("seg", "w_seg"),
        ("template_smooth", None),
        ("identity_smooth", "w_identity_smooth"),
    ]
    total = torch.zeros((), device=ref.device, dtype=ref.dtype)
    for key, wkey in terms:
        if key not in losses:
            continue
        if key == "silhouette":
            w = w_sil
        elif key == "template_smooth":
            w = w_tpl
        elif key == "identity_smooth":
            w = _get_w(cfg, "w_identity_smooth", 0.0)
        else:
            w = _get_w(cfg, wkey, 0.0)
        total = total + w * losses[key]
    losses["total"] = total
    return losses
