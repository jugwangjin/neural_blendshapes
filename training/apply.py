"""Apply StageSpec: freeze/unfreeze modules and build optimizers."""

from types import SimpleNamespace

import torch

from gaussian_splatting.semantic import h_prior_tensors
from training.stages import StageSpec


def stage_loss_cfg(spec: StageSpec):
    h_sigma, h_weight = h_prior_tensors(torch.device("cpu"))
    return SimpleNamespace(
        w_rgb=spec.w_rgb,
        w_mp_lmk=spec.w_mp_lmk,
        w_mp_mask=spec.w_mask,
        w_mask=spec.w_mask,
        w_seg=spec.w_seg,
        w_iris=spec.w_iris,
        w_h=spec.w_h,
        w_eye_uv_barrier=spec.w_eye_uv_barrier,
        w_scale=spec.w_scale,
        w_opacity=spec.w_opacity,
        w_gamma_prior=spec.w_gamma_prior,
        w_pose_prior=spec.w_pose_prior,
        w_expr_deform_reg=spec.w_expr_deform_reg,
        w_expr_neutral=spec.w_expr_neutral,
        w_expr_leak=spec.w_expr_leak,
        w_expr_amp=spec.w_expr_amp,
        w_sem_anchor=spec.w_sem_anchor,
        w_identity_smooth=spec.w_template_smooth,
        w_template_smooth=spec.w_template_smooth,
        h_class_sigma=h_sigma.numpy().tolist(),
        h_class_weight=h_weight.numpy().tolist(),
    )


def _set_requires_grad(module, flag):
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = flag


def _freeze_uvh_geometry(module):
    module.uv.requires_grad = False
    module.log_scale.requires_grad = False
    module.rotation.requires_grad = False
    if module.fixed_h is None:
        module.h.requires_grad = False


def _set_uvh_trainable(module, appearance, geometry, semantic, geom_lr_scale=1.0):
    if appearance:
        module.color.requires_grad = True
        module.opacity.requires_grad = True
    else:
        module.color.requires_grad = False
        module.opacity.requires_grad = False

    if geometry:
        module.uv.requires_grad = True
        module.log_scale.requires_grad = True
        module.rotation.requires_grad = True
        if module.fixed_h is None:
            module.h.requires_grad = True
    else:
        _freeze_uvh_geometry(module)

    if semantic and module.sem_logits is not None:
        module.sem_logits.requires_grad = True
    elif module.sem_logits is not None:
        module.sem_logits.requires_grad = False


def apply_stage_requires_grad(spec, tracker, deformer, avatar, expr_deform):
    _set_requires_grad(tracker, False)
    _set_requires_grad(deformer, False)
    _set_requires_grad(avatar, False)
    _set_requires_grad(expr_deform, False)

    need_trunk = (
        spec.train_pose_residual
        or (spec.train_gamma and not spec.fix_gamma_at_one)
        or spec.train_tracker
    )
    if need_trunk:
        for p in tracker.trunk.parameters():
            p.requires_grad = True
    if spec.train_pose_residual or spec.train_tracker:
        for p in tracker.head_pose.parameters():
            p.requires_grad = True
        for p in tracker.head_trans.parameters():
            p.requires_grad = True
    if spec.train_gamma and not spec.fix_gamma_at_one:
        for p in tracker.head_gamma.parameters():
            p.requires_grad = True

    if spec.train_pose_weight:
        for p in deformer.pose_weight_net.parameters():
            p.requires_grad = True

    if spec.train_template_deformer:
        deformer.template_offset.requires_grad = True

    if spec.train_expression_deform and expr_deform is not None:
        _set_requires_grad(expr_deform, True)

    g = spec.geometry_lr_scale
    _set_uvh_trainable(
        avatar.face,
        spec.train_gaussian_appearance,
        spec.train_gaussian_geometry,
        spec.train_gaussian_semantic,
        geom_lr_scale=g,
    )
    for side in (avatar.eyes.left, avatar.eyes.right):
        _set_uvh_trainable(
            side,
            spec.train_gaussian_appearance,
            False,
            spec.train_gaussian_semantic,
        )
        if spec.train_eye_gaze:
            side.uv.requires_grad = True

    if avatar.eyes.gaze_refine_left is not None:
        avatar.eyes.gaze_refine_left.requires_grad = spec.train_eye_gaze
    if avatar.eyes.gaze_refine_right is not None:
        avatar.eyes.gaze_refine_right.requires_grad = spec.train_eye_gaze

    if spec.train_eye_gaze:
        for p in tracker.head_gaze_l.parameters():
            p.requires_grad = True
        for p in tracker.head_gaze_r.parameters():
            p.requires_grad = True
        if not need_trunk:
            for p in tracker.trunk.parameters():
                p.requires_grad = True


def build_optimizers(spec, tracker, deformer, avatar, expr_deform):
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

    if spec.train_template_deformer and deformer.template_offset.requires_grad:
        mesh_groups.append({"params": [deformer.template_offset], "lr": spec.lr_template})

    if spec.train_expression_deform and expr_deform is not None:
        mesh_groups.append(
            {
                "params": [p for p in expr_deform.parameters() if p.requires_grad],
                "lr": spec.lr_expr_deform,
            }
        )

    def add_uvh_groups(mod):
        if mod.color.requires_grad:
            gaussian_groups.append({"params": [mod.color], "lr": spec.lr_gaussian_color})
        if mod.opacity.requires_grad:
            gaussian_groups.append({"params": [mod.opacity], "lr": spec.lr_gaussian_opacity})
        if mod.uv.requires_grad:
            gaussian_groups.append({"params": [mod.uv], "lr": spec.lr_gaussian_uv * gscale})
        if mod.log_scale.requires_grad:
            gaussian_groups.append({"params": [mod.log_scale], "lr": spec.lr_gaussian_scale * gscale})
        if mod.rotation.requires_grad:
            gaussian_groups.append({"params": [mod.rotation], "lr": spec.lr_gaussian_scale * gscale})
        if mod.fixed_h is None and mod.h.requires_grad:
            gaussian_groups.append({"params": [mod.h], "lr": spec.lr_gaussian_h * gscale})
        if (
            mod.sem_logits is not None
            and mod.sem_logits.requires_grad
            and mod.sem_prob_fixed is None
        ):
            gaussian_groups.append({"params": [mod.sem_logits], "lr": spec.lr_gaussian_uv * gscale})

    add_uvh_groups(avatar.face)
    for side in (avatar.eyes.left, avatar.eyes.right):
        add_uvh_groups(side)

    gaze_params = []
    if avatar.eyes.gaze_refine_left is not None and avatar.eyes.gaze_refine_left.requires_grad:
        gaze_params.append(avatar.eyes.gaze_refine_left)
    if avatar.eyes.gaze_refine_right is not None and avatar.eyes.gaze_refine_right.requires_grad:
        gaze_params.append(avatar.eyes.gaze_refine_right)
    if gaze_params:
        gaussian_groups.append({"params": gaze_params, "lr": spec.lr_eye_gaze})

    mesh_optim = torch.optim.Adam(mesh_groups) if mesh_groups else None
    gaussian_optim = torch.optim.Adam(gaussian_groups) if gaussian_groups else None
    return mesh_optim, gaussian_optim


def init_training_state(ava