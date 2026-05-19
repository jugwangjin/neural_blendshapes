"""Apply StageSpec: freeze/unfreeze modules and build optimizers."""

from types import SimpleNamespace

import torch

from rendering.semantic import h_prior_tensors
from training.stages import StageSpec


def stage_loss_cfg(spec: StageSpec):
    h_sigma, h_weight = h_prior_tensors(torch.device("cpu"))
    return SimpleNamespace(
        w_rgb=spec.w_rgb,
        w_mp_lmk=spec.w_mp_lmk,
        w_silhouette=spec.w_silhouette if spec.w_silhouette > 0 else spec.w_mask,
        w_mp_mask=spec.w_silhouette if spec.w_silhouette > 0 else spec.w_mask,
        w_mask=spec.w_silhouette if spec.w_silhouette > 0 else spec.w_mask,
        w_seg=spec.w_seg,
        w_iris=spec.w_iris,
        w_h=spec.w_h,
        w_eye_uv_barrier=spec.w_eye_uv_barrier,
        w_scale=spec.w_scale,
        w_opacity=spec.w_opacity,
        w_gamma_prior=spec.w_gamma_prior,
        w_pose_prior=spec.w_pose_prior,
        w_pose_tz=spec.w_pose_tz,
        apply_pose_scale=spec.apply_pose_scale,
        w_gaze_residual=spec.w_gaze_residual,
        w_expr_deform_reg=spec.w_expr_deform_reg,
        w_expr_neutral=spec.w_expr_neutral,
        w_expr_leak=spec.w_expr_leak,
        w_expr_amp=spec.w_expr_amp,
        w_sem_anchor=spec.w_sem_anchor,
        w_identity_smooth=spec.w_template_smooth,
        w_template_smooth=spec.w_template_smooth,
        h_class_sigma=h_sigma.numpy().tolist(),
        h_class_weight=h_weight.numpy().tolist(),
        mask_accessory_in_seg=not spec.train_accessory,
    )


def _set_requires_grad(module, flag):
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = flag


def _set_surface_trainable(surface, appearance, geometry, semantic, geom_scale=1.0):
    if appearance:
        surface.color.requires_grad = True
        surface.opacity.requires_grad = True
    else:
        surface.color.requires_grad = False
        surface.opacity.requires_grad = False

    if geometry:
        surface.h.requires_grad = True
        surface.log_scale.requires_grad = True
        surface.rotation.requires_grad = True
    else:
        surface.h.requires_grad = False
        surface.log_scale.requires_grad = False
        surface.rotation.requires_grad = False

    if semantic and surface.sem_logits is not None:
        surface.sem_logits.requires_grad = True
    elif surface.sem_logits is not None:
        surface.sem_logits.requires_grad = False


def _set_eye_gaussians_trainable(eyes, appearance, train_gaze):
    """
    Shared ``EyeTextureGaussians``: ``uv`` / ``h`` are fixed buffers (sclera chart).
    Train appearance + optional ``gaze_refine_*`` only — never slide base UV.
    """
    if appearance:
        eyes.color.requires_grad = True
        eyes.opacity.requires_grad = True
        eyes.log_scale.requires_grad = True
        eyes.rotation.requires_grad = True
    else:
        eyes.color.requires_grad = False
        eyes.opacity.requires_grad = False
        eyes.log_scale.requires_grad = False
        eyes.rotation.requires_grad = False

    if eyes.gaze_refine_left is not None:
        eyes.gaze_refine_left.requires_grad = train_gaze
    if eyes.gaze_refine_right is not None:
        eyes.gaze_refine_right.requires_grad = train_gaze


def _set_accessory_trainable(acc, train):
    if acc is None or acc.n_gaussians == 0:
        return
    for p in acc.parameters():
        p.requires_grad = train


def apply_stage_requires_grad(spec, tracker, deformer, avatar):
    _set_requires_grad(tracker, False)
    _set_requires_grad(deformer, False)
    _set_requires_grad(avatar, False)

    if spec.train_gamma and not spec.fix_gamma_at_one:
        for p in tracker.expr_trunk.parameters():
            p.requires_grad = True
        for p in tracker.head_gamma.parameters():
            p.requires_grad = True
    if spec.train_tracker:
        for p in tracker.expr_trunk.parameters():
            p.requires_grad = True
        for p in tracker.head_gamma.parameters():
            p.requires_grad = True

    if spec.train_pose_residual or spec.train_tracker:
        for p in tracker.pose_trunk.parameters():
            p.requires_grad = True
        for p in tracker.head_pose.parameters():
            p.requires_grad = True
        if spec.train_pose_scale:
            tracker.log_pose_scale.requires_grad = True

    if spec.train_pose_weight:
        for p in deformer.pose_weight_net.parameters():
            p.requires_grad = True

    if spec.train_template_deformer:
        for p in deformer.template_mlp.parameters():
            p.requires_grad = True
        deformer.log_max_template_delta.requires_grad = True

    if spec.train_expression_deform:
        for p in deformer.expr_au_embed.parameters():
            p.requires_grad = True
        for p in deformer.expr_mlp.parameters():
            p.requires_grad = True

    g = spec.geometry_lr_scale
    surf = avatar.surface
    _set_surface_trainable(
        surf,
        spec.train_gaussian_appearance,
        spec.train_gaussian_geometry,
        spec.train_gaussian_semantic,
        geom_scale=g,
    )
    if spec.train_gaussian_geometry and g < 1.0:
        surf.h.requires_grad = spec.train_gaussian_geometry

    _set_eye_gaussians_trainable(
        avatar.eyes,
        spec.train_gaussian_appearance,
        train_gaze=spec.train_eye_gaze,
    )

    if spec.train_eye_gaze:
        for p in tracker.gaze_trunk.parameters():
            p.requires_grad = True
        for p in tracker.head_gaze_l.parameters():
            p.requires_grad = True
        for p in tracker.head_gaze_r.parameters():
            p.requires_grad = True

    _set_accessory_trainable(avatar.accessory, spec.train_accessory)


def build_optimizers(spec, tracker, deformer, avatar):
    mesh_groups = []
    gaussian_groups = []
    gscale = spec.geometry_lr_scale

    tracker_params = [p for p in tracker.parameters() if p.requires_grad]
    if tracker_params:
        mesh_groups.append({"params": tracker_params, "lr": spec.lr_tracker})

    if spec.train_pose_weight:
        mesh_groups.append(
            {
                "params": [p for p in deformer.pose_weight_net.parameters() if p.requires_grad],
                "lr": spec.lr_pose_weight,
            }
        )

    if spec.train_template_deformer:
        tpl_params = [p for p in deformer.template_mlp.parameters() if p.requires_grad]
        if deformer.log_max_template_delta.requires_grad:
            tpl_params.append(deformer.log_max_template_delta)
        if tpl_params:
            mesh_groups.append({"params": tpl_params, "lr": spec.lr_template})

    if spec.train_expression_deform:
        expr_params = []
        for p in deformer.expr_au_embed.parameters():
            if p.requires_grad:
                expr_params.append(p)
        for p in deformer.expr_mlp.parameters():
            if p.requires_grad:
                expr_params.append(p)
        if expr_params:
            mesh_groups.append({"params": expr_params, "lr": spec.lr_expr_deform})

    def add_surface_groups(surf):
        if surf.color.requires_grad:
            gaussian_groups.append({"params": [surf.color], "lr": spec.lr_gaussian_color})
        if surf.opacity.requires_grad:
            gaussian_groups.append({"params": [surf.opacity], "lr": spec.lr_gaussian_opacity})
        if surf.h.requires_grad:
            gaussian_groups.append({"params": [surf.h], "lr": spec.lr_gaussian_h * gscale})
        if surf.log_scale.requires_grad:
            gaussian_groups.append({"params": [surf.log_scale], "lr": spec.lr_gaussian_scale * gscale})
        if surf.rotation.requires_grad:
            gaussian_groups.append({"params": [surf.rotation], "lr": spec.lr_gaussian_scale * gscale})
        if surf.sem_logits is not None and surf.sem_logits.requires_grad:
            gaussian_groups.append({"params": [surf.sem_logits], "lr": spec.lr_gaussian_scale * gscale})

    add_surface_groups(avatar.surface)

    eye_mod = avatar.eyes
    if eye_mod.color.requires_grad:
        gaussian_groups.append({"params": [eye_mod.color], "lr": spec.lr_gaussian_color})
    if eye_mod.opacity.requires_grad:
        gaussian_groups.append({"params": [eye_mod.opacity], "lr": spec.lr_gaussian_opacity})
    if eye_mod.log_scale.requires_grad:
        gaussian_groups.append({"params": [eye_mod.log_scale], "lr": spec.lr_gaussian_scale * gscale})
    if eye_mod.rotation.requires_grad:
        gaussian_groups.append({"params": [eye_mod.rotation], "lr": spec.lr_gaussian_scale * gscale})

    gaze_params = []
    if avatar.eyes.gaze_refine_left is not None and avatar.eyes.gaze_refine_left.requires_grad:
        gaze_params.append(avatar.eyes.gaze_refine_left)
    if avatar.eyes.gaze_refine_right is not None and avatar.eyes.gaze_refine_right.requires_grad:
        gaze_params.append(avatar.eyes.gaze_refine_right)
    if gaze_params:
        gaussian_groups.append({"params": gaze_params, "lr": spec.lr_eye_gaze})

    acc = avatar.accessory
    if acc is not None and acc.n_gaussians > 0:
        acc_params = [p for p in acc.parameters() if p.requires_grad]
        if acc_params:
            gaussian_groups.append({"params": acc_params, "lr": spec.lr_accessory})

    mesh_optim = torch.optim.Adam(mesh_groups) if mesh_groups else None
    gaussian_optim = torch.optim.Adam(gaussian_groups) if gaussian_groups else None
    return mesh_optim, gaussian_optim


def init_training_state(avatar):
    with torch.no_grad():
        avatar.surface.h.zero_()
