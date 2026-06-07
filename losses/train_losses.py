"""Training losses — orchestrates per-term losses and weighted sum."""

import torch

from losses.avatar_masks import (
    eyelash_vertex_ids as _get_eyelash_vidx,
    face_region_target_mask as _face_region_target_mask,
    non_tight_face_mask as _non_tight_face_mask,
    tight_face_mask as _tight_face_mask,
)
from losses.gaussian_regularization import (
    loss_color_expression_activation_per_coeff,
    loss_color_expression_activation_per_gaussian,
    loss_color_expression_activation_sparse,
    loss_geometry_log_scale,
    loss_opacity_sparsity,
    loss_opacity_toward_one_masked,
    loss_opacity_toward_one,
    loss_opacity_uniform,
    loss_scaling_regularization,
)
from losses.h_regularization import loss_h_image_space
from losses.loss_weights import (
    aggregate_weighted_losses,
    get_loss_weight as _get_w,
    silhouette_weight as _silhouette_weight,
)
from losses.mediapipe_landmark_478 import loss_mediapipe_landmarks_478
from losses.pie68_jaw_landmark import loss_pie68_jawline
from losses.rgb import rgb_l1_ssim_loss
from losses.silhouette import (
    loss_silhouette,
    loss_silhouette_edt,
    silhouette_edt_distance_fields,
)
from losses.train_util import (
    align_batch_device as _align_batch_device,
    image_size_from_cfg as _image_size,
    loss_section as _loss_section,
    surface_avatar_out,
    surface_sem_features as _surface_sem_features,
)
from dataset.collate import batch_has_gt_normal


def compute_losses(
    cfg,
    batch,
    render,
    avatar_out,
    camera,
    mp_lmk_emb,
    ict_faces,
    pie68_jaw_vertex_idx=None,
    pie68_protocol_idx=None,
    corr=None,
    deformer=None,
    expr_delta=None,
    avatar=None,
    renderer=None,
    timer=None,
):
    ref = avatar_out["xyz"]
    batch = _align_batch_device(batch, ref.device)
    image_size = _image_size(cfg, batch)
    losses = {}

    with _loss_section(timer, "rgb"):
        if render is not None and "rgb" in render and batch.get("image") is not None:
            rgb_pred = render["rgb"]
            rgb_total, rgb_l1, rgb_dssim = rgb_l1_ssim_loss(
                rgb_pred,
                batch["image"],
                lambda_ssim=_get_w(cfg, "rgb_ssim_lambda", 0.2),
            )
            losses["rgb"] = rgb_total
            losses["rgb_l1"] = rgb_l1
            losses["rgb_dssim"] = rgb_dssim
            if _get_w(cfg, "w_lpips", 0.0) > 0:
                from losses.lpips_loss import loss_lpips

                losses["lpips"] = loss_lpips(
                    rgb_pred,
                    batch["image"],
                    mask=batch.get("mask"),
                    net=str(_get_w(cfg, "lpips_net", "alex")),
                )

    with _loss_section(timer, "normal"):
        if (
            _get_w(cfg, "w_normal", 0.0) > 0
            and batch_has_gt_normal(batch)
            and render is not None
            and render.get("normal") is not None
        ):
            from losses.normal import loss_normal_laplacian

            pred_n = render["normal"]
            render_alpha = render.get("alpha")
            gt_n = batch["gt_normal"]
            losses["normal"] = loss_normal_laplacian(
                pred_n,
                gt_n,
                camera=camera,
                batch=batch,
                render_alpha=render_alpha,
            )

    w_sil = _silhouette_weight(cfg)
    with _loss_section(timer, "silhouette"):
        if w_sil > 0 and render is not None and "alpha" in render:
            sil_alpha = render.get("silhouette_alpha", render["alpha"])
            use_edt = bool(_get_w(cfg, "silhouette_use_edt", False))
            d_out, d_in = None, None
            if use_edt:
                d_out, d_in = silhouette_edt_distance_fields(batch, cfg, sil_alpha)
            if use_edt and d_out is not None and d_in is not None:
                losses["silhouette"] = loss_silhouette_edt(
                    sil_alpha,
                    d_out,
                    d_in,
                    w_ext=_get_w(cfg, "silhouette_edt_w_ext", 1.0),
                    w_int=_get_w(cfg, "silhouette_edt_w_int", 1.0),
                    max_dist_px=_get_w(cfg, "silhouette_edt_max_dist_px", 50.0),
                )
            elif batch.get("mask") is not None:
                losses["silhouette"] = loss_silhouette(
                    sil_alpha,
                    batch["mask"],
                    use_l1=bool(_get_w(cfg, "silhouette_l1", False)),
                )

    with _loss_section(timer, "mp_lmk"):
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
                mouth_weight=_get_w(cfg, "mp_lmk_mouth_weight", 2.0),
                lmk_metric=_get_w(cfg, "lmk_distance_metric", "smooth_l1"),
                lmk_eps=_get_w(cfg, "lmk_charbonnier_eps", 1e-4),
                lmk_wing_w_px=_get_w(cfg, "lmk_wing_w_px", 10.0),
                lmk_wing_eps_px=_get_w(cfg, "lmk_wing_eps_px", 2.0),
            )

    with _loss_section(timer, "pie68_jaw"):
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
                protocol_idx=pie68_protocol_idx,
                lmk_metric=_get_w(cfg, "lmk_distance_metric", "smooth_l1"),
                lmk_eps=_get_w(cfg, "lmk_charbonnier_eps", 1e-4),
                lmk_wing_w_px=_get_w(cfg, "lmk_wing_w_px", 10.0),
                lmk_wing_eps_px=_get_w(cfg, "lmk_wing_eps_px", 2.0),
            )

    with _loss_section(timer, "mesh_silhouette"):
        if (
            batch.get("mask") is not None
            and _get_w(cfg, "w_mesh_silhouette", 0.0) > 0
            and avatar_out.get("mesh_xyz") is not None
        ):
            from losses.mesh_silhouette import loss_mesh_silhouette

            mesh_use_edt = _get_w(cfg, "mesh_silhouette_use_edt", None)
            if mesh_use_edt is None:
                mesh_use_edt = bool(_get_w(cfg, "silhouette_use_edt", False))
            else:
                mesh_use_edt = bool(mesh_use_edt)
            losses["mesh_silhouette"] = loss_mesh_silhouette(
                avatar_out["mesh_xyz"],
                ict_faces,
                camera,
                batch["mask"],
                image_size=image_size,
                downsample=1,
                exclude_vertex_ids=_get_eyelash_vidx(avatar),
                cull_backfaces=bool(_get_w(cfg, "mesh_silhouette_cull_backfaces", False)),
                cull_flip=bool(_get_w(cfg, "mesh_silhouette_cull_flip", False)),
                backface_curl_weight=_get_w(cfg, "mesh_backface_curl_weight", 0.0),
                use_edt=mesh_use_edt,
                cfg=cfg,
                batch=batch,
            )

    with _loss_section(timer, "mesh_semantic"):
        if (
            batch.get("part_label") is not None
            and _get_w(cfg, "w_mesh_seg", 0.0) > 0
            and avatar_out.get("mesh_xyz") is not None
            and avatar is not None
            and getattr(avatar, "ict", None) is not None
        ):
            from losses.mesh_semantic import loss_mesh_semantic

            ict = avatar.ict
            losses["mesh_semantic"] = loss_mesh_semantic(
                avatar_out["mesh_xyz"],
                ict_faces,
                ict,
                camera,
                batch["part_label"],
                image_size=image_size,
                downsample=4,
                exclude_vertex_ids=_get_eyelash_vidx(avatar),
                cull_backfaces=bool(_get_w(cfg, "mesh_silhouette_cull_backfaces", False)),
                cull_flip=bool(_get_w(cfg, "mesh_silhouette_cull_flip", False)),
            )

    with _loss_section(timer, "h"):
        if (
            renderer is not None
            and _get_w(cfg, "w_h") > 0
            and batch.get("h_reg_seg_face") is not None
        ):
            surf_out = surface_avatar_out(avatar_out)
            h_sig = surf_out["h"]
            if avatar is not None:
                scale_teeth = float(_get_w(cfg, "h_teeth_h_loss_scale", 1.0))
                is_teeth = getattr(avatar.surface, "is_teeth", None)
                if scale_teeth != 1.0 and is_teeth is not None and bool(is_teeth.any()):
                    w = torch.where(
                        is_teeth.reshape(-1),
                        h_sig.new_tensor(scale_teeth),
                        h_sig.new_tensor(1.0),
                    )
                    h_sig = h_sig * w.unsqueeze(-1)
                scale_occ = float(_get_w(cfg, "h_eye_occlusion_h_loss_scale", 1.0))
                codes = getattr(avatar.surface, "face_region_code", None)
                if scale_occ != 1.0 and codes is not None:
                    on_occ = codes.reshape(-1) == 6
                    if on_occ.any():
                        w = torch.where(
                            on_occ,
                            h_sig.new_tensor(scale_occ),
                            h_sig.new_tensor(1.0),
                        )
                        h_sig = h_sig * w.unsqueeze(-1)
            h_render = renderer.render_expected_signal(surf_out, camera, h_sig)
            losses["h"] = loss_h_image_space(
                h_render["accum"],
                h_render["alpha"],
                batch,
                cfg,
                _get_w,
            )

    sem_feat = _surface_sem_features(avatar_out)

    with _loss_section(timer, "geometry"):
        if avatar is not None and _get_w(cfg, "w_geometry") > 0:
            # log_scale parameter; losses apply exp (same as render forward).
            losses["geometry"] = loss_geometry_log_scale(
                avatar.surface.log_scale,
                max_scale=_get_w(cfg, "geometry_max_scale", 0.004),
            )

    with _loss_section(timer, "scaling"):
        if avatar is not None and _get_w(cfg, "w_scaling", 0.0) > 0:
            losses["scaling"] = loss_scaling_regularization(
                avatar.surface.log_scale,
                thresh_scaling_max=_get_w(cfg, "thresh_scaling_max", 0.008),
                thresh_scaling_ratio=_get_w(cfg, "thresh_scaling_ratio", 10.0),
            )

    surf = avatar.surface if avatar is not None else None
    with _loss_section(timer, "color_expr"):
        if surf is not None and getattr(surf, "color_expression", None) is not None:
            expr_support = surf._color_expression_support()
            expr_coeff = None
            if corr is not None:
                expr_coeff = corr.get("coeffs")
            w_sparse = _get_w(cfg, "w_color_expr_sparse", 0.0)
            w_per_coeff = _get_w(cfg, "w_color_expr_group_sparse", 0.0)
            w_per_gauss = _get_w(cfg, "w_color_expr_per_gaussian", 0.0)
            if w_sparse > 0:
                losses["color_expr_sparse"] = loss_color_expression_activation_sparse(
                    surf.color_expression, expr_support, expr_coeff
                )
            if w_per_coeff > 0:
                losses["color_expr_per_coeff"] = loss_color_expression_activation_per_coeff(
                    surf.color_expression, expr_support, expr_coeff
                )
            if w_per_gauss > 0:
                losses["color_expr_per_gaussian"] = loss_color_expression_activation_per_gaussian(
                    surf.color_expression, expr_support, expr_coeff
                )

    with _loss_section(timer, "opacity"):
        if avatar is not None and _get_w(cfg, "w_opacity") > 0:
            target = _get_w(cfg, "opacity_target", 1.0)
            # avatar.surface.opacity is logit; losses apply sigmoid (same as render forward).
            if sem_feat is not None:
                losses["opacity"] = loss_opacity_toward_one(
                    avatar.surface.opacity,
                    sem_feat,
                    target=target,
                    w_skin=_get_w(cfg, "opacity_w_skin", 1.0),
                    w_other=_get_w(cfg, "opacity_w_other", 0.05),
                )
            else:
                losses["opacity"] = loss_opacity_uniform(avatar.surface.opacity, target=target)

        if avatar is not None and _get_w(cfg, "w_opacity_loose", 0.0) > 0:
            loose_mask = _non_tight_face_mask(avatar)
            if loose_mask is not None and loose_mask.any():
                losses["opacity_loose"] = loss_opacity_toward_one_masked(
                    avatar.surface.opacity,
                    loose_mask,
                    target=_get_w(cfg, "opacity_loose_target", 1.0),
                )

        if avatar is not None and _get_w(cfg, "w_opacity_headneck", 0.0) > 0:
            tight_mask = _tight_face_mask(avatar)
            if tight_mask is not None and tight_mask.any():
                losses["opacity_headneck"] = loss_opacity_toward_one_masked(
                    avatar.surface.opacity,
                    tight_mask,
                    target=1.0,
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

    with _loss_section(timer, "prior"):
        if corr is not None and corr.get("gamma") is not None and _get_w(cfg, "w_gamma_prior") > 0:
            g = corr["gamma"]
            if corr.get("additive_gamma"):
                losses["gamma_prior"] = g.pow(2).mean()
            else:
                losses["gamma_prior"] = torch.log(g.clamp(min=1e-4)).pow(2).mean()

        if corr is not None and _get_w(cfg, "w_pose_prior") > 0:
            rot_delta = corr.get("pose_rotation_delta", corr["pose_residual"])
            t_global = corr.get("translation_global")
            pose_prior = rot_delta.pow(2).mean() + corr["translation_residual"].pow(2).mean()
            if t_global is not None:
                pose_prior = pose_prior + t_global.pow(2).mean()
            losses["pose_prior"] = pose_prior

        if corr is not None and _get_w(cfg, "w_pose_tz", 0.0) > 0:
            tz = corr["translation_residual"][..., 2].pow(2)
            t_global = corr.get("translation_global")
            if t_global is not None:
                tz = tz + t_global[..., 2].pow(2)
            losses["pose_tz"] = tz.mean()

    with _loss_section(timer, "expr_deform_reg"):
        if deformer is not None and expr_delta is not None and corr is not None and _get_w(cfg, "w_expr_deform_reg", 0.0) > 0:
            from losses.deformer_regularization import deformer_regularization_loss

            c_raw = corr.get("coeffs_raw", corr["coeffs"])
            reg = deformer_regularization_loss(deformer, corr["coeffs"], c_raw, expr_delta=expr_delta)
            losses["expr_deform_reg"] = (
                reg["expr_neutral"] + reg["expr_leak"] + reg["expr_amp"] + reg["expr_socket"]
            )

    w_tpl = _get_w(cfg, "w_template_smooth", _get_w(cfg, "w_identity_smooth", 0.0))
    w_tpl_lap = _get_w(cfg, "w_template_laplacian", 0.0)
    w_tpl_scale = _get_w(cfg, "w_template_scale_prior", 0.0)
    train_template = bool(getattr(cfg, "train_template_deformer", False))
    with _loss_section(timer, "template"):
        if deformer is not None and train_template:
            from losses.deformer_regularization import (
                template_laplacian_loss,
                template_scale_prior,
                template_smooth_loss,
            )

            if w_tpl > 0:
                losses["template_smooth"] = template_smooth_loss(deformer)
            if w_tpl_lap > 0:
                losses["template_laplacian"] = template_laplacian_loss(deformer)
            if w_tpl_scale > 0:
                losses["template_scale"] = template_scale_prior(deformer)

    w_id_prior = _get_w(cfg, "w_identity_prior", 0.0)
    train_identity = bool(getattr(cfg, "train_ict_identity", False))
    with _loss_section(timer, "identity_prior"):
        if deformer is not None and train_identity and w_id_prior > 0:
            from losses.deformer_regularization import identity_prior_loss

            losses["identity_prior"] = identity_prior_loss(deformer)

    with _loss_section(timer, "seg"):
        if render is not None and _get_w(cfg, "w_seg", 0.0) > 0:
            from losses.segmentation import (
                exclude_flare_parts_from_mask,
                loss_lip_mouth_leak,
                loss_segmentation_l1,
                loss_segmentation_l2,
            )

            pred_sem = render.get("semantic")
            sem_alpha = render.get("semantic_alpha")
            part_label_hw = batch.get("part_label")
            if pred_sem is not None:
                if part_label_hw is not None:
                    from dataset.flare_semantic import flare_part_label_to_semantic_class

                    target = flare_part_label_to_semantic_class(part_label_hw)
                elif batch.get("seg_label") is not None:
                    target = batch["seg_label"]
                    if target.ndim == 4:
                        target = target[0]
                    if target.ndim == 3:
                        target = target[0]
                else:
                    target = None
                if target is not None:
                    target = target.to(device=pred_sem.device, dtype=torch.long)
                    if target.ndim == 2:
                        target = target.unsqueeze(0)
                    h_pred, w_pred = pred_sem.shape[-2:]
                    if target.shape[-2:] != (h_pred, w_pred):
                        target = torch.nn.functional.interpolate(
                            target.unsqueeze(1).float(),
                            size=(h_pred, w_pred),
                            mode="nearest",
                        ).squeeze(1).long()
                    alpha_min = float(_get_w(cfg, "seg_alpha_min", 0.02))
                    valid = None
                    if sem_alpha is not None:
                        sa = sem_alpha[:, 0] if sem_alpha.ndim == 4 else sem_alpha
                        valid = sa > alpha_min
                    if part_label_hw is not None:
                        pl = part_label_hw.to(device=pred_sem.device)
                        if pl.ndim == 2:
                            pl = pl.unsqueeze(0)
                        if pl.shape[-2:] != (h_pred, w_pred):
                            pl = torch.nn.functional.interpolate(
                                pl.unsqueeze(1).float(),
                                size=(h_pred, w_pred),
                                mode="nearest",
                            ).squeeze(1).long()
                        fg = pl != 0
                        valid = fg if valid is None else valid & fg
                        valid = exclude_flare_parts_from_mask(pl, valid)
                    if bool(_get_w(cfg, "seg_l1", False)):
                        losses["seg"] = loss_segmentation_l1(
                            pred_sem, target, valid_mask=valid
                        )
                    else:
                        losses["seg"] = loss_segmentation_l2(
                            pred_sem, target, valid_mask=valid
                        )
                    if (
                        part_label_hw is not None
                        and _get_w(cfg, "w_lip_mouth_leak", 0.0) > 0
                    ):
                        losses["lip_mouth_leak"] = loss_lip_mouth_leak(
                            pred_sem,
                            part_label_hw,
                            valid_mask=valid,
                        )

    with _loss_section(timer, "face_region"):
        if (
            renderer is not None
            and avatar is not None
            and avatar_out.get("surface") is not None
            and _get_w(cfg, "w_face_region", 0.0) > 0
        ):
            target_face = _face_region_target_mask(batch)
            if target_face is not None:
                from losses.segmentation import loss_full_face_region

                region_render = renderer.render_full_face_region(
                    avatar_out, camera, ict_faces, avatar.ict
                )
                losses["face_region"] = loss_full_face_region(
                    region_render,
                    target_face,
                    alpha_min=_get_w(cfg, "face_region_alpha_min", 0.02),
                )

    with _loss_section(timer, "sparsity"):
        lambda_sparsity = _get_w(cfg, "lambda_sparsity", 0.0)
        if avatar is not None and lambda_sparsity > 0:
            losses["sparsity"] = loss_opacity_sparsity(avatar.surface.opacity)

    with _loss_section(timer, "aggregate"):
        aggregate_weighted_losses(losses, cfg, ref_tensor=ref, w_silhouette=w_sil)
    return losses

