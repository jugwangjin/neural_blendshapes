"""Apply StageSpec: freeze/unfreeze modules and build optimizers."""

from types import SimpleNamespace

import torch
import torch.nn as nn

from training.stages import StageSpec


def stage_needs_rasterization(cfg) -> bool:
    """True when any loss term requires a full gsplat RGB/alpha (or seg) pass."""
    if getattr(cfg, "w_rgb", 0.0) > 0:
        return True
    if getattr(cfg, "w_lpips", 0.0) > 0:
        return True
    if getattr(cfg, "w_normal", 0.0) > 0:
        return True
    w_sil = getattr(cfg, "w_silhouette", 0.0)
    if w_sil <= 0:
        w_sil = getattr(cfg, "w_mask", 0.0)
    if w_sil <= 0:
        w_sil = getattr(cfg, "w_mp_mask", 0.0)
    if w_sil > 0:
        return True
    if getattr(cfg, "w_seg", 0.0) > 0:
        return True
    if getattr(cfg, "w_lip_mouth_leak", 0.0) > 0:
        return True
    if getattr(cfg, "w_h", 0.0) > 0:
        return True
    return False


def stage_needs_surface_forward(cfg) -> bool:
    """True when avatar must run surface Gaussian layout (not just deformed mesh)."""
    if stage_needs_rasterization(cfg):
        return True
    for key in (
        "w_geometry",
        "w_opacity",
        "w_opacity_headneck",
        "w_opacity_decay",
        "w_face_region",
        "w_mesh_silhouette",
        "w_mesh_seg",
        "w_color_expr_sparse",
        "w_color_expr_group_sparse",
        "w_color_expr_per_gaussian",
    ):
        if getattr(cfg, key, 0.0) > 0:
            return True
    return False


def stage_loss_cfg(spec: StageSpec):
    return SimpleNamespace(
        w_rgb=spec.w_rgb,
        rgb_ssim_lambda=getattr(spec, "rgb_ssim_lambda", 0.2),
        w_mp_lmk=spec.w_mp_lmk,
        w_pie68_jaw=spec.w_pie68_jaw,
        w_silhouette=spec.w_silhouette if spec.w_silhouette > 0 else spec.w_mask,
        silhouette_detach_covariance=getattr(spec, "silhouette_detach_covariance", False),
        silhouette_l1=getattr(spec, "silhouette_l1", False),
        w_mesh_silhouette=getattr(spec, "w_mesh_silhouette", 0.0),
        w_mesh_seg=getattr(spec, "w_mesh_seg", 0.0),
        mesh_seg_stop_local=getattr(spec, "mesh_seg_stop_local", 0),
        mesh_backface_curl_weight=getattr(spec, "mesh_backface_curl_weight", 0.0),
        lmk_distance_metric=getattr(spec, "lmk_distance_metric", "smooth_l1"),
        lmk_charbonnier_eps=getattr(spec, "lmk_charbonnier_eps", 1e-4),
        lmk_wing_w_px=getattr(spec, "lmk_wing_w_px", 10.0),
        lmk_wing_eps_px=getattr(spec, "lmk_wing_eps_px", 2.0),
        w_mp_mask=spec.w_silhouette if spec.w_silhouette > 0 else spec.w_mask,
        w_mask=spec.w_silhouette if spec.w_silhouette > 0 else spec.w_mask,
        w_seg=spec.w_seg,
        seg_l1=getattr(spec, "seg_l1", False),
        seg_alpha_min=getattr(spec, "seg_alpha_min", 0.02),
        w_h=spec.w_h,
        w_geometry=spec.w_geometry,
        w_opacity=spec.w_opacity,
        w_opacity_loose=getattr(spec, "w_opacity_loose", 0.0),
        w_opacity_headneck=getattr(spec, "w_opacity_headneck", 0.0),
        w_face_region=getattr(spec, "w_face_region", 0.0),
        face_region_alpha_min=getattr(spec, "face_region_alpha_min", 0.02),
        lambda_sparsity=getattr(spec, "lambda_sparsity", 0.0),
        w_lpips=getattr(spec, "w_lpips", 0.0),
        w_normal=getattr(spec, "w_normal", 0.0),
        w_lip_mouth_leak=getattr(spec, "w_lip_mouth_leak", 0.0),
        lpips_net=getattr(spec, "lpips_net", "alex"),
        w_gamma_prior=spec.w_gamma_prior,
        h_w_skin=getattr(spec, "h_w_skin", 2.4),
        h_w_nose=getattr(spec, "h_w_nose", 1.4),
        h_w_eye=getattr(spec, "h_w_eye", 2.8),
        h_w_brow=getattr(spec, "h_w_brow", 1.4),
        h_w_neck=getattr(spec, "h_w_neck", 1.4),
        h_w_cloth=getattr(spec, "h_w_cloth", 0.1),
        h_w_hair=getattr(spec, "h_w_hair", 0.015),
        h_w_glasses=getattr(spec, "h_w_glasses", 0.006),
        h_w_misc=getattr(spec, "h_w_misc", 1.4),
        h_w_mouth=getattr(spec, "h_w_mouth", 0.0),
        h_teeth_h_loss_scale=getattr(spec, "h_teeth_h_loss_scale", 1.0),
        h_eye_occlusion_h_loss_scale=getattr(spec, "h_eye_occlusion_h_loss_scale", 2.5),
        h_alpha_min=getattr(spec, "h_alpha_min", 0.08),
        geometry_max_scale=getattr(spec, "geometry_max_scale", 0.004),
        thresh_scaling_max=getattr(spec, "thresh_scaling_max", 0.008),
        thresh_scaling_ratio=getattr(spec, "thresh_scaling_ratio", 10.0),
        opacity_target=getattr(spec, "opacity_target", 1.0),
        opacity_loose_target=getattr(spec, "opacity_loose_target", 1.0),
        opacity_w_skin=getattr(spec, "opacity_w_skin", 1.0),
        opacity_w_other=getattr(spec, "opacity_w_other", 0.05),
        w_pose_prior=spec.w_pose_prior,
        w_pose_tz=spec.w_pose_tz,
        apply_pose_scale=spec.apply_pose_scale,
        w_expr_deform_reg=spec.w_expr_deform_reg,
        w_expr_neutral=spec.w_expr_neutral,
        w_expr_leak=spec.w_expr_leak,
        w_expr_amp=spec.w_expr_amp,
        w_identity_smooth=spec.w_template_smooth,
        w_template_smooth=spec.w_template_smooth,
        w_template_laplacian=getattr(spec, "w_template_laplacian", 0.0),
        w_template_scale_prior=getattr(spec, "w_template_scale_prior", 0.0),
        train_template_deformer=spec.train_template_deformer,
        train_ict_identity=getattr(spec, "train_ict_identity", False),
        w_identity_prior=getattr(spec, "w_identity_prior", 0.0),
        w_opacity_decay=getattr(spec, "w_opacity_decay", 0.0),
        w_color_expr_sparse=getattr(spec, "w_color_expr_sparse", 0.0),
        w_color_expr_group_sparse=getattr(spec, "w_color_expr_group_sparse", 0.0),
        w_color_expr_per_gaussian=getattr(spec, "w_color_expr_per_gaussian", 0.0),
    )


def _set_requires_grad(module, flag):
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = flag


def _set_surface_trainable(
    surface,
    appearance,
    geometry,
    train_h=False,
    train_color_pose=False,
    train_color_expression=False,
    geom_scale=1.0,
):
    if appearance:
        surface.color.requires_grad = True
        surface.color_pose.requires_grad = bool(train_color_pose)
        surface.color_expression.requires_grad = bool(train_color_expression)
        surface.opacity.requires_grad = True
    else:
        surface.color.requires_grad = False
        surface.color_pose.requires_grad = False
        surface.color_expression.requires_grad = False
        surface.opacity.requires_grad = False

    if geometry:
        surface.h.requires_grad = bool(train_h)
        surface.log_scale.requires_grad = True
        surface.rotation.requires_grad = True
        if hasattr(surface, "bary_uv"):
            surface.bary_uv.requires_grad = True
    else:
        surface.h.requires_grad = False
        surface.log_scale.requires_grad = False
        surface.rotation.requires_grad = False
        if hasattr(surface, "bary_uv"):
            surface.bary_uv.requires_grad = False

def apply_h_constraint(avatar, train_h: bool):
    """When ``train_h`` is False: pin ``h`` to 0 except mouth socket + teeth offsets."""
    surf = avatar.surface
    surf.h_trainable = bool(train_h)
    if not train_h:
        with torch.no_grad():
            keep = surf.face_region_code == 1
            if hasattr(surf, "is_teeth"):
                keep = keep | surf.is_teeth
            else:
                keep = keep | (surf.face_region_code == 7)
            surf.h.data[~keep] = 0.0


def apply_inference_forward_flags(avatar, spec):
    """
    Restore avatar forward flags from ``StageSpec`` after checkpoint load.

    ``h_trainable`` is not in ``state_dict``; default ``False`` ignores saved ``h``
    and breaks gsplat vs training ``eval_render``.
    """
    train_h = bool(getattr(spec, "train_gaussian_h", False))
    apply_h_constraint(avatar, train_h)
    return dict(h_trainable=bool(avatar.surface.h_trainable))


def apply_stage_requires_grad(spec, tracker, deformer, avatar):
    _set_requires_grad(tracker, False)
    _set_requires_grad(deformer, False)
    _set_requires_grad(avatar, False)

    train_gamma = spec.train_gamma or spec.train_tracker
    train_pose = spec.train_pose_residual or spec.train_tracker

    if train_gamma:
        for p in tracker.expr_trunk.parameters():
            p.requires_grad = True
        for p in tracker.head_gamma.parameters():
            p.requires_grad = True

    if train_pose:
        for p in tracker.pose_trunk.parameters():
            p.requires_grad = True
        for p in tracker.head_pose.parameters():
            p.requires_grad = True
    if spec.train_pose_scale:
        tracker.log_pose_scale.requires_grad = True
    if getattr(spec, "train_global_translation", False):
        tracker.global_translation.requires_grad = True
    if spec.train_pose_weight:
        for p in deformer.pose_weight_net.parameters():
            p.requires_grad = True

    if spec.train_template_deformer:
        for p in deformer.template_mlp.parameters():
            p.requires_grad = True

    if getattr(spec, "train_ict_identity", False) and isinstance(
        deformer.identity_weights, nn.Parameter
    ):
        deformer.identity_weights.requires_grad = True

    if spec.train_expression_deform:
        for p in deformer.expr_mlp.parameters():
            p.requires_grad = True

    g = spec.geometry_lr_scale
    surf = avatar.surface
    train_h = getattr(spec, "train_gaussian_h", False)
    _set_surface_trainable(
        surf,
        spec.train_gaussian_appearance,
        spec.train_gaussian_geometry,
        train_h=train_h,
        train_color_pose=getattr(spec, "train_color_pose", False),
        train_color_expression=getattr(spec, "train_color_expression", False),
        geom_scale=g,
    )
    apply_h_constraint(avatar, train_h)


def build_optimizers(spec, tracker, deformer, avatar, cfg=None):
    mesh_groups = []
    gaussian_groups = []
    gscale = spec.geometry_lr_scale

    tracker_params = [p for p in tracker.parameters() if p.requires_grad]
    if tracker_params:
        mesh_groups.append({"params": tracker_params, "lr": spec.lr_tracker})

    if spec.train_pose_weight:
        lr_pose_weight = spec.lr_pose_weight
        if cfg is not None:
            lr_pose_weight = float(getattr(cfg, "lr_pose_weight", lr_pose_weight))
        mesh_groups.append(
            {
                "params": [p for p in deformer.pose_weight_net.parameters() if p.requires_grad],
                "lr": lr_pose_weight,
            }
        )

    if spec.train_template_deformer:
        tpl_params = [p for p in deformer.template_mlp.parameters() if p.requires_grad]
        if tpl_params:
            mesh_groups.append(
                {
                    "params": tpl_params,
                    "lr": spec.lr_template,
                    "geometry_decay": True,
                }
            )

    if (
        getattr(spec, "train_ict_identity", False)
        and isinstance(deformer.identity_weights, nn.Parameter)
        and deformer.identity_weights.requires_grad
    ):
        mesh_groups.append(
            {"params": [deformer.identity_weights], "lr": getattr(spec, "lr_identity", 1e-2)}
        )

    if spec.train_expression_deform:
        expr_params = []
        for p in deformer.expr_mlp.parameters():
            if p.requires_grad:
                expr_params.append(p)
        if expr_params:
            mesh_groups.append(
                {"params": expr_params, "lr": spec.lr_expr_deform, "geometry_decay": True}
            )

    def add_surface_groups(surf):
        lr_color = spec.lr_gaussian_color
        lr_color_aux = lr_color * 0.25  # gsplat shN_lr = sh0_lr / 20; pose/expr detail slower
        if surf.color.requires_grad:
            gaussian_groups.append({"params": [surf.color], "lr": lr_color})
        if hasattr(surf, "color_pose") and surf.color_pose.requires_grad:
            gaussian_groups.append({"params": [surf.color_pose], "lr": lr_color_aux})
        if hasattr(surf, "color_expression") and surf.color_expression.requires_grad:
            gaussian_groups.append({"params": [surf.color_expression], "lr": lr_color_aux})
        if surf.opacity.requires_grad:
            gaussian_groups.append({"params": [surf.opacity], "lr": spec.lr_gaussian_opacity})
        if surf.h.requires_grad:
            gaussian_groups.append(
                {"params": [surf.h], "lr": spec.lr_gaussian_h, "geometry_decay": True}
            )
        if surf.log_scale.requires_grad:
            gaussian_groups.append(
                {"params": [surf.log_scale], "lr": spec.lr_gaussian_scale * gscale}
            )
        rot_lr = getattr(spec, "lr_gaussian_rotation", spec.lr_gaussian_scale) * gscale
        if surf.rotation.requires_grad:
            gaussian_groups.append({"params": [surf.rotation], "lr": rot_lr})
        if hasattr(surf, "bary_uv") and surf.bary_uv.requires_grad:
            gaussian_groups.append(
                {"params": [surf.bary_uv], "lr": spec.lr_gaussian_h, "geometry_decay": True}
            )
    add_surface_groups(avatar.surface)

    for optim_groups in (mesh_groups, gaussian_groups):
        for group in optim_groups:
            if group.get("geometry_decay", False):
                group["initial_lr"] = group["lr"]

    mesh_optim = torch.optim.Adam(mesh_groups) if mesh_groups else None
    gaussian_optim = torch.optim.Adam(gaussian_groups) if gaussian_groups else None
    return mesh_optim, gaussian_optim


def geometry_lr_decay_mult(spec, stage_local: int) -> float:
    """
    Stage-local exponential decay (GB ``position_lr_final / position_lr_init``).

    Before ``geometry_lr_decay_start_frac * steps``: mult = 1.
    After that: ``lr = lr_init * (final_mult ** t)`` with ``t`` in [0, 1] over the decay window.
    """
    if spec.steps <= 0:
        return 1.0
    start_frac = float(getattr(spec, "geometry_lr_decay_start_frac", 0.0))
    start_frac = min(max(start_frac, 0.0), 1.0)
    final_mult = float(getattr(spec, "geometry_lr_decay_final_mult", 0.01))
    step = min(max(stage_local, 0), spec.steps)
    decay_start = int(spec.steps * start_frac)
    if step <= decay_start:
        return 1.0
    decay_steps = max(spec.steps - decay_start, 1)
    t = (step - decay_start) / float(decay_steps)
    return final_mult**t


def apply_geometry_lr_decay(mesh_optim, gaussian_optim, spec, stage_local: int):
    if not getattr(spec, "geometry_lr_decay", False):
        return
    mult = geometry_lr_decay_mult(spec, stage_local)
    for optim in (mesh_optim, gaussian_optim):
        if optim is None:
            continue
        for group in optim.param_groups:
            if not group.get("geometry_decay", False):
                continue
            init_lr = group.get("initial_lr", group["lr"])
            group["lr"] = init_lr * mult


def clip_optimizer_grads(*optims, max_norm):
    params = []
    for optim in optims:
        if optim is None:
            continue
        for group in optim.param_groups:
            for p in group["params"]:
                if p.requires_grad and p.grad is not None:
                    params.append(p)
    if params:
        torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)


def tracker_out_for_training(spec: StageSpec, corr: dict) -> dict:
    """Training forward coeffs. ``fix_gamma_at_one`` is inference/viz-only (see ``tracker`` call sites)."""
    if getattr(spec, "use_ict_raw_coeffs", False):
        return {**corr, "coeffs": corr["coeffs_raw"]}
    return corr


def init_training_state(avatar, cfg=None):
    from model.gaussian_h_init import init_teeth_h

    with torch.no_grad():
        avatar.surface.h.zero_()
        teeth_r = float(getattr(cfg, "teeth_h_radius", 0.0)) if cfg is not None else 0.0
        if teeth_r > 0:
            init_teeth_h(avatar.surface, avatar.ict, teeth_r)
    apply_h_constraint(avatar, getattr(avatar.surface, "h_trainable", False))
