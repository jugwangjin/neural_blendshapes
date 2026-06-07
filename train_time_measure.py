"""
Profile train.py per-component wall time (CUDA-synchronized).

  python train_time_measure.py --resume checkpoints/stage_1_....pt --input-dir /path/to/subject

Defaults: stage ``2_coarse_mesh``, 100 measured iters, 10 warmup iters.
"""

import argparse
import os
import sys
from contextlib import nullcontext
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
from training.cli import add_base_train_arguments, apply_base_train_cli
from training.code_dump import dump_training_code
from training.component_timer import ComponentTimer
from training.loss_cfg import hydrate_stage_loss_cfg, update_stage_local_loss_weights
from training.loss_logger import print_losses, tqdm_postfix
from training.loss_overrides import apply_loss_overrides, load_loss_overrides
from training.resume import resolve_existing_input_dir
from training.stages import STAGE_SCHEDULE, iter_stages, total_training_steps
from training.dataloader_util import restart_loader_iter
from training.train_step import TrainStepState, run_train_step
from utils.seed import set_seed


def _code_dump_dir(cfg):
    return cfg.codes_dir / datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_train_cli():
    p = argparse.ArgumentParser(description="Profile training step component timings")
    add_base_train_arguments(p)
    p.add_argument(
        "--time-measure-stage",
        type=str,
        default="2_coarse_mesh",
        help="Stage name to profile (default: 2_coarse_mesh)",
    )
    p.add_argument("--time-measure-iters", type=int, default=100)
    p.add_argument("--time-measure-warmup", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_train_cli()
    cfg = apply_base_train_cli(Config(), args)
    set_seed(cfg.seed, deterministic=cfg.deterministic)
    cfg.input_dir = resolve_existing_input_dir(cfg.input_dir)
    from processing.ict_mediapipe_lmk.embedding_io import resolve_embedding_path

    cfg.mp_embedding = resolve_embedding_path(cfg.mp_embedding)
    assert cfg.batch_size == 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.eval_render_dir.mkdir(parents=True, exist_ok=True)

    schedule = apply_loss_overrides(cfg, STAGE_SCHEDULE, load_loss_overrides(args.loss_overrides))
    cfg.iterations = total_training_steps(schedule)

    print(f"seed={cfg.seed} deterministic={cfg.deterministic}")
    print(
        f"time measure: stage={args.time_measure_stage} "
        f"iters={args.time_measure_iters} warmup={args.time_measure_warmup}"
    )
    dump_training_code(ROOT, _code_dump_dir(cfg), cfg, schedule)

    stack, global_step = build_training_stack(cfg, device, resume_path=args.resume)
    measure_done = False

    for stage_idx, spec, stage_start, stage_end in iter_stages(schedule):
        if spec.steps <= 0:
            continue
        if stage_end <= global_step:
            print(f"\n=== Skip stage {stage_idx}: {spec.name} (checkpoint already past step {stage_end}) ===")
            continue

        print(f"\n=== Stage {stage_idx}: {spec.name} ({spec.steps} steps) ===")
        measure_active = spec.name == args.time_measure_stage
        if not measure_active:
            if measure_done:
                print(f"\n=== Skip stage {stage_idx}: {spec.name} (time measure complete) ===")
            continue

        timer = ComponentTimer(warmup=args.time_measure_warmup) if measure_active else None
        if measure_active:
            print(
                f"[TimeMeasure] profiling stage={spec.name} "
                f"iters={args.time_measure_iters} warmup={args.time_measure_warmup}"
            )

        stack.renderer.set_sh_degree(spec.sh_degree)
        cfg.sh_degree = spec.sh_degree
        apply_stage_requires_grad(spec, stack.tracker, stack.deformer, stack.avatar)
        mesh_optim, gaussian_optim = build_optimizers(spec, stack.tracker, stack.deformer, stack.avatar, cfg)
        if spec.name in cfg.gaussian_densify_stages:
            stack.densify_strategy.reset_state(len(stack.avatar.surface.h), stack.avatar.surface.h.device)

        loss_cfg = hydrate_stage_loss_cfg(stage_loss_cfg(spec), spec, cfg)
        stage_local = global_step - stage_start
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
            t = timer
            if t is not None:
                t.begin_iter()

            with (t.section("data") if t else nullcontext()):
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
                    timer=t,
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
            if global_step % cfg.log_every == 0:
                pbar.set_postfix(tqdm_postfix(losses, global_step), refresh=False)
                print_losses(losses, global_step, spec.name)

            if (
                measure_active
                and timer is not None
                and timer.n_recorded >= args.time_measure_iters
            ):
                out_json = cfg.output_root / "analysis" / "time_measure.json"
                timer.print_report(spec.name, args.time_measure_iters, out_path=out_json)
                measure_done = True
                break

        pbar.close()
        if measure_done:
            print("[TimeMeasure] done.")
            return

    print(f"\nDone. Total steps: {global_step}.")


if __name__ == "__main__":
    main()
