"""
MediaPipe → tracker MLP → ICT deformer → surface/eye Gaussians → gsplat.

Run from repo root:
  python train.py
  python train.py --input-dir /path/to/subject --train-split MVI_1814 MVI_1810 --eval-split MVI_1812

CLI flags use hyphens (--output-root). See docs/guides/training.md.
"""

import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"
os.environ["LANG"] = "C.UTF-8"
os.environ["LC_ALL"] = "C.UTF-8"

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import Config
from training.apply import apply_stage_requires_grad, build_optimizers, stage_loss_cfg
from training.build_stack import build_training_stack
from training.checkpoint_io import avatar_n_from_state_dict, save_checkpoint
from training.cli import add_base_train_arguments, apply_base_train_cli
from training.code_dump import dump_training_code
from training.eval_render import render_eval_set
from training.loss_cfg import hydrate_stage_loss_cfg, update_stage_local_loss_weights
from training.loss_logger import LossAnalysisLogger, print_losses, tqdm_postfix
from training.loss_overrides import (
    apply_loss_overrides,
    load_loss_overrides,
    resolve_training_schedule,
)
from training.resume import resolve_existing_input_dir
from training.stages import describe_stage_trainables, iter_stages, total_training_steps
from training.dataloader_util import restart_loader_iter
from training.train_step import TrainStepState, run_train_step
from utils.seed import set_seed


def _code_dump_dir(cfg: Config) -> Path:
    return cfg.codes_dir / datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_train_cli():
    p = argparse.ArgumentParser(description="Train MediaPipe → ICT → 3DGS avatar")
    add_base_train_arguments(p)
    return p.parse_args()


def _print_stage_trainables(spec, cfg):
    for line in describe_stage_trainables(spec):
        print(f"  trainable: {line}")
    if spec.train_pose_weight:
        lr_pw = float(getattr(cfg, "lr_pose_weight", spec.lr_pose_weight))
        print(f"  pose_weight_net: trainable (lr={lr_pw}, pose_weight_one={spec.pose_weight_one})")
    if getattr(spec, "train_ict_identity", False):
        print(f"  ict identity_weights: trainable (lr={spec.lr_identity})")
    if spec.train_template_deformer:
        print(f"  template_mlp: trainable (lr={spec.lr_template})")


def _run_stage(
    *,
    cfg,
    spec,
    stage_idx,
    stage_start,
    stack,
    global_step,
    loss_logger,
    no_mid_eval,
    no_stage_end_eval,
):
    device = stack.device
    if spec.steps <= 0:
        return global_step, None
    stage_end = stage_start + spec.steps
    if stage_end <= global_step:
        print(f"\n=== Skip stage {stage_idx}: {spec.name} (checkpoint already past step {stage_end}) ===")
        return global_step, None

    print(f"\n=== Stage {stage_idx}: {spec.name} ({spec.steps} steps) ===")
    print(spec.description)
    n_avatar = stack.avatar.n_gaussians
    print(f"  avatar n_gaussians={n_avatar} at stage start")
    resume_meta = getattr(stack, "resume_meta", None)
    stage_local = global_step - stage_start
    if resume_meta is not None and stage_local == 0:
        n_resume = int(resume_meta["n_gaussians"])
        if n_avatar != n_resume:
            raise RuntimeError(
                f"avatar reset before stage {spec.name}: in-memory n={n_avatar} != "
                f"resume ckpt n={n_resume} ({resume_meta['path'].name})"
            )
        print(
            f"  resume ok: n={n_resume} matches {resume_meta['path'].name} "
            f"(global_step={global_step}, ckpt stage={resume_meta['stage']})"
        )
    elif resume_meta is None and spec.name == "2_coarse_mesh":
        s0 = sorted(cfg.checkpoint_dir.glob("stage_0_*_end_step_*.pt"))
        if s0:
            n0 = avatar_n_from_state_dict(
                torch.load(s0[-1], map_location="cpu", weights_only=False)["avatar"]
            )
            print(f"  continuous run: n={n_avatar} (stage_0 ckpt n={n0}, delta {n_avatar - n0:+d})")
    elif resume_meta is None and spec.name == "3_expression_detail":
        s2 = sorted(cfg.checkpoint_dir.glob("stage_2_coarse_mesh_end_step_*.pt"))
        if s2:
            n2 = avatar_n_from_state_dict(
                torch.load(s2[-1], map_location="cpu", weights_only=False)["avatar"]
            )
            print(
                f"  continuous run: n={n_avatar} (stage_2 ckpt n={n2}, delta {n_avatar - n2:+d})"
            )
            if n_avatar != n2:
                print(
                    "  NOTE: n != stage_2 ckpt before any stage_3 step — avatar changed between "
                    "stage_2 save and stage_3 start (not resume path; check separate train invocations)"
                )

    stack.renderer.set_sh_degree(spec.sh_degree)
    cfg.sh_degree = spec.sh_degree

    apply_stage_requires_grad(spec, stack.tracker, stack.deformer, stack.avatar)
    mesh_optim, gaussian_optim = build_optimizers(spec, stack.tracker, stack.deformer, stack.avatar, cfg)
    if getattr(spec, "geometry_lr_decay", False):
        mult = float(getattr(spec, "geometry_lr_decay_final_mult", 0.01))
        start_frac = float(getattr(spec, "geometry_lr_decay_start_frac", 0.0))
        start_note = (
            f", decay from {start_frac:.0%} of stage"
            if start_frac > 0.0
            else ", decay from stage start"
        )
        print(
            f"  geometry LR decay: final_mult={mult}{start_note} "
            f"(h, bary_uv, template_mlp, expr_mlp; color/opacity/scale/rot/tracker fixed LR)"
        )
    _print_stage_trainables(spec, cfg)
    if spec.name in cfg.gaussian_densify_stages:
        stack.densify_strategy.reset_state(len(stack.avatar.surface.h), stack.avatar.surface.h.device)

    loss_cfg = hydrate_stage_loss_cfg(stage_loss_cfg(spec), spec, cfg)

    pbar = tqdm(
        total=spec.steps,
        initial=stage_local,
        desc=f"stage {stage_idx} {spec.name}",
        unit="step",
        dynamic_ncols=True,
    )
    loader_iter = iter(stack.loader)
    for _ in range(stage_local, spec.steps):
        if stage_local >= spec.steps:
            break
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = restart_loader_iter(stack.loader, loader_iter)
            batch = next(loader_iter)
        stage_local += 1
        global_step += 1
        update_stage_local_loss_weights(loss_cfg, spec, stage_local)

        losses, _, _, _ = run_train_step(
            TrainStepState(
                cfg=cfg,
                spec=spec,
                stack=stack,
                batch=batch,
                loss_cfg=loss_cfg,
                mesh_optim=mesh_optim,
                gaussian_optim=gaussian_optim,
                stage_local=stage_local,
                global_step=global_step,
            )
        )

        if (
            stack.loader.dataset is not None
            and losses.get("rgb_l1") is not None
            and batch.get("dataset_frame_idx") is not None
        ):
            fi = batch["dataset_frame_idx"]
            frame_j = int(fi[0] if isinstance(fi, (list, tuple)) else fi)
            stack.loader.dataset.update_rgb_loss_ema(frame_j, losses["rgb_l1"].item())

        pbar.update(1)
        if global_step % cfg.log_every == 0 or stage_local >= spec.steps:
            pbar.set_postfix(tqdm_postfix(losses, global_step), refresh=False)
            if global_step % cfg.log_every == 0:
                print_losses(losses, global_step, spec.name)
                densify_stats = {}
                if spec.name in cfg.gaussian_densify_stages:
                    densify_stats = stack.densify_strategy.analysis_snapshot(
                        global_step,
                        stage_name=spec.name,
                        stage_local=stage_local,
                        surf=stack.avatar.surface,
                    )
                loss_logger.log(
                    global_step=global_step,
                    stage_name=spec.name,
                    stage_local=stage_local,
                    losses=losses,
                    mesh_optim=mesh_optim,
                    gaussian_optim=gaussian_optim,
                    densify_stats=densify_stats,
                )

        if not no_mid_eval and global_step % int(cfg.eval_render_interval) == 0:
            render_eval_set(
                cfg,
                spec,
                stack.tracker,
                stack.avatar,
                stack.renderer,
                stack.camera,
                device,
                out_dir=cfg.eval_render_dir,
                global_step=global_step,
                max_frames=cfg.eval_max_frames,
                eval_loader=stack.eval_loader,
                deformer=stack.deformer,
                **stack.eval_render_viz,
            )

    pbar.close()
    if hasattr(stack.loader.dataset, "save_rgb_loss_ema"):
        stack.loader.dataset.save_rgb_loss_ema()
    n_before_save = stack.avatar.n_gaussians
    print(f"  stage end: avatar n_gaussians={n_before_save} (before save)")
    stage_ckpt = cfg.checkpoint_dir / f"stage_{spec.name}_end_step_{global_step:06d}.pt"
    save_checkpoint(
        stage_ckpt,
        global_step=global_step,
        stage_name=spec.name,
        tracker=stack.tracker,
        deformer=stack.deformer,
        avatar=stack.avatar,
        cfg=cfg,
        spec=spec,
        extra={"stage_end": True, "stage_steps": spec.steps},
    )
    if not no_stage_end_eval:
        render_eval_set(
            cfg,
            spec,
            stack.tracker,
            stack.avatar,
            stack.renderer,
            stack.camera,
            device,
            out_dir=cfg.eval_render_dir,
            global_step=global_step,
            max_frames=cfg.eval_max_frames,
            eval_loader=stack.eval_loader,
            deformer=stack.deformer,
            **stack.eval_render_viz,
        )
    return global_step, spec


def main():
    args = parse_train_cli()
    cfg = apply_base_train_cli(Config(), args)
    set_seed(cfg.seed, deterministic=cfg.deterministic)
    cfg.input_dir = resolve_existing_input_dir(cfg.input_dir)
    from processing.ict_mediapipe_lmk.embedding_io import resolve_embedding_path

    cfg.mp_embedding = resolve_embedding_path(cfg.mp_embedding)
    assert cfg.batch_size == 1, "avatar/render path is single-mesh; set batch_size=1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.eval_render_dir.mkdir(parents=True, exist_ok=True)

    loss_overrides = load_loss_overrides(args.loss_overrides)
    base_schedule = resolve_training_schedule(loss_overrides)
    schedule = apply_loss_overrides(cfg, base_schedule, loss_overrides)
    cfg.iterations = total_training_steps(schedule)
    no_mid_eval = args.no_mid_eval_render or args.final_eval_only
    no_stage_end_eval = args.no_stage_end_eval or args.final_eval_only

    print(f"seed={cfg.seed} deterministic={cfg.deterministic}")
    code_dir = dump_training_code(ROOT, _code_dump_dir(cfg), cfg, schedule)
    if args.loss_overrides is not None:
        shutil.copy2(args.loss_overrides, code_dir / "loss_overrides.json")
        print(f"loss overrides: copied to {code_dir / 'loss_overrides.json'}")

    stack, global_step = build_training_stack(cfg, device, resume_path=args.resume)
    loss_logger = LossAnalysisLogger(cfg.output_root / "analysis" / "loss_log.jsonl")
    current_spec = None

    for stage_idx, spec, stage_start, _stage_end in iter_stages(schedule):
        global_step, ended_spec = _run_stage(
            cfg=cfg,
            spec=spec,
            stage_idx=stage_idx,
            stage_start=stage_start,
            stack=stack,
            global_step=global_step,
            loss_logger=loss_logger,
            no_mid_eval=no_mid_eval,
            no_stage_end_eval=no_stage_end_eval,
        )
        if ended_spec is not None:
            current_spec = ended_spec

    if args.final_eval_only and current_spec is not None and stack.eval_loader is not None:
        from dataclasses import replace

        final_spec = replace(current_spec, name="final_eval")
        print(f"\n=== Final eval render (step {global_step}) ===")
        render_eval_set(
            cfg,
            final_spec,
            stack.tracker,
            stack.avatar,
            stack.renderer,
            stack.camera,
            device,
            out_dir=cfg.eval_render_dir,
            global_step=global_step,
            max_frames=cfg.eval_max_frames,
            eval_loader=stack.eval_loader,
            deformer=stack.deformer,
            save_checkpoint_pt=False,
            **stack.eval_render_viz,
        )

    print(f"\nDone. Total steps: {global_step}. Last stage: {current_spec.name if current_spec else 'n/a'}")


if __name__ == "__main__":
    main()
