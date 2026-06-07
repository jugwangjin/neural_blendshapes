"""Single training iteration: tracker → avatar → render → losses → backward → optim."""

from contextlib import nullcontext
from dataclasses import dataclass

import torch

from dataset import move_batch_to_device
from dataset.collate import batch_has_gt_normal
from losses.train_losses import compute_losses
from training.apply import (
    apply_geometry_lr_decay,
    apply_h_constraint,
    stage_needs_rasterization,
    stage_needs_surface_forward,
    tracker_out_for_training,
)
from training.bootstrap_debug import maybe_save_bootstrap_debug
from training.mouth_coeff_debug import log_small_mouth_coeff_batch, mouth_debug_active


@dataclass
class TrainStepState:
    cfg: object
    spec: object
    stack: object
    batch: dict
    loss_cfg: object
    mesh_optim: object
    gaussian_optim: object
    stage_local: int
    global_step: int
    timer: object = None


def run_tracker(state: TrainStepState):
    spec = state.spec
    tracker = state.stack.tracker
    batch = state.batch
    train_tracker_now = (
        spec.train_tracker
        or spec.train_gamma
        or spec.train_pose_residual
        or spec.train_pose_scale
        or getattr(spec, "train_global_translation", False)
    )
    tracker_inputs = dict(
        mp_blendshape=batch["mp_blendshape"],
        mp_landmarks_2d=batch.get("mp_landmarks_2d"),
        mp_landmarks_3d=batch.get("mp_landmarks_3d"),
        world_to_cam=batch.get("world_to_cam"),
        mp_pose_raw=batch.get("mp_pose_raw"),
        mp_transform_matrix=batch.get("mp_transform_matrix"),
        force_gamma_one=False,
        use_global_translation_param=getattr(spec, "use_global_translation_param", False),
        additive_gamma_correction=getattr(spec, "additive_gamma_correction", False),
    )
    if not train_tracker_now:
        with torch.no_grad():
            corr = tracker(**tracker_inputs)
    else:
        corr = tracker(**tracker_inputs)
    return tracker_out_for_training(spec, corr)


def run_avatar_and_render(state: TrainStepState, corr):
    spec = state.spec
    loss_cfg = state.loss_cfg
    cfg = state.cfg
    avatar = state.stack.avatar
    renderer = state.stack.renderer
    camera = state.stack.camera

    pose_weight_fixed = 1.0 if spec.pose_weight_one else None
    need_render = stage_needs_rasterization(loss_cfg)
    need_surface = stage_needs_surface_forward(loss_cfg)

    avatar_out = avatar(
        tracker_out=corr,
        apply_expression_deform=spec.train_expression_deform,
        use_pose_scale=spec.apply_pose_scale,
        pose_weight_fixed=pose_weight_fixed,
        rotate_about_centroid=spec.pose_rotate_about_centroid,
        pose_zero_tz=spec.pose_zero_tz,
        skip_surface=not need_surface,
        enable_color_pose=getattr(spec, "train_color_pose", False),
        enable_color_expression=getattr(spec, "train_color_expression", False),
    )
    expr_delta = avatar_out.get("expr_delta")

    render = None
    if need_render:
        render_semantic = (
            (
                loss_cfg.w_seg > 0
                or getattr(loss_cfg, "w_lip_mouth_leak", 0.0) > 0
            )
            and cfg.n_semantic_classes > 0
        )
        render = renderer(
            avatar_out,
            camera,
            render_semantic=render_semantic,
        )
        if (
            getattr(loss_cfg, "silhouette_detach_covariance", False)
            and getattr(loss_cfg, "w_silhouette", 0.0) > 0
        ):
            sil = renderer.render_silhouette_alpha(
                avatar_out,
                camera,
                detach_covariance=True,
            )
            render["silhouette_alpha"] = sil["alpha"]
        if (
            getattr(loss_cfg, "w_normal", 0.0) > 0
            and batch_has_gt_normal(state.batch)
            and avatar_out.get("surface") is not None
            and avatar_out["surface"].get("normals") is not None
        ):
            normal_render = renderer.render_expected_signal(
                avatar_out,
                camera,
                avatar_out["surface"]["normals"],
                signal_dim=3,
            )
            render["normal"] = normal_render["expected"]

    return avatar_out, expr_delta, render


def run_train_step(state: TrainStepState):
    """
    One optimizer step. Returns ``(losses, corr, avatar_out, render)``.
    Mutates ``state.batch`` (moved to device) and runs optimizers when configured.
    """
    t = state.timer
    spec = state.spec
    cfg = state.cfg
    stack = state.stack

    batch = move_batch_to_device(state.batch, stack.device)
    state.batch = batch

    with (t.section("tracker") if t else nullcontext()):
        corr = run_tracker(state)

    if mouth_debug_active(spec, state.stage_local, cfg):
        stem = None
        for key in ("frame_name", "stem", "path"):
            if batch.get(key) is not None:
                s = batch[key]
                stem = s[0] if isinstance(s, (list, tuple)) else s
                if hasattr(stem, "item"):
                    stem = stem.item()
                break
        log_small_mouth_coeff_batch(
            cfg=cfg,
            spec=spec,
            stage_local=state.stage_local,
            global_step=state.global_step,
            batch=batch,
            corr=corr,
            indices=stack.mouth_debug_idx,
            stem=stem,
        )

    with (t.section("avatar_forward") if t else nullcontext()):
        avatar_out, expr_delta, render = run_avatar_and_render(state, corr)

    with (t.section("losses") if t else nullcontext()):
        losses = compute_losses(
            state.loss_cfg,
            batch,
            render,
            avatar_out,
            stack.camera,
            stack.mp_lmk_emb,
            stack.ict_faces,
            pie68_jaw_vertex_idx=stack.pie68_vertex_idx,
            pie68_protocol_idx=stack.pie68_protocol_idx,
            corr=corr,
            deformer=stack.deformer,
            expr_delta=expr_delta,
            avatar=stack.avatar,
            renderer=stack.renderer,
            timer=t,
        )

    mesh_optim = state.mesh_optim
    gaussian_optim = state.gaussian_optim
    if mesh_optim is not None:
        mesh_optim.zero_grad(set_to_none=True)
    if gaussian_optim is not None:
        gaussian_optim.zero_grad(set_to_none=True)
    apply_geometry_lr_decay(mesh_optim, gaussian_optim, spec, state.stage_local)

    densify = stack.densify_strategy
    with (t.section("densify_pre_backward") if t else nullcontext()):
        if spec.name in cfg.gaussian_densify_stages and render is not None:
            densify.pre_backward(
                state.global_step,
                render,
                avatar=stack.avatar,
                stage_name=spec.name,
                stage_local=state.stage_local,
            )

    with (t.section("backward") if t else nullcontext()):
        losses["total"].backward()

    with (t.section("densify_post_backward") if t else nullcontext()):
        if (
            spec.name in cfg.gaussian_densify_stages
            and gaussian_optim is not None
            and render is not None
        ):
            densify.post_backward(
                state.global_step,
                stack.avatar,
                render,
                stage_name=spec.name,
                stage_local=state.stage_local,
            )

    maybe_save_bootstrap_debug(state, avatar_out, corr, batch)

    with (t.section("optimizer_mesh") if t else nullcontext()):
        if mesh_optim is not None:
            mesh_optim.step()

    if gaussian_optim is not None:
        densify_mesh = avatar_out.get("mesh_xyz") if avatar_out is not None else None
        with (t.section("densify_pre_optimizer") if t else nullcontext()):
            if spec.name in cfg.gaussian_densify_stages:
                densify.pre_optimizer_step(
                    state.global_step,
                    stack.avatar,
                    gaussian_optim,
                    stack.ict_faces,
                    stack.ict,
                    stage_name=spec.name,
                    stage_local=state.stage_local,
                    mesh_verts=densify_mesh,
                    tracker=stack.tracker,
                    apply_pose_scale=spec.apply_pose_scale,
                )
        with (t.section("optimizer_gaussian") if t else nullcontext()):
            gaussian_optim.step()
        with (t.section("triangle_walk") if t else nullcontext()):
            if (
                spec.train_gaussian_geometry
                and state.global_step % max(1, cfg.gaussian_triangle_walk_every) == 0
            ):
                stack.triangle_walker.step(stack.avatar.surface, gaussian_optim)
        with (t.section("densify_post_optimizer") if t else nullcontext()):
            if spec.name in cfg.gaussian_densify_stages:
                densify.post_optimizer_step(
                    state.global_step,
                    stack.avatar,
                    gaussian_optim,
                    stack.ict_faces,
                    stack.ict,
                    stage_name=spec.name,
                    stage_local=state.stage_local,
                    mesh_verts=densify_mesh,
                    tracker=stack.tracker,
                    apply_pose_scale=spec.apply_pose_scale,
                )
        if not getattr(spec, "train_gaussian_h", False):
            apply_h_constraint(stack.avatar, False)

    return losses, corr, avatar_out, render
