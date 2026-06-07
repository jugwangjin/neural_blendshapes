"""Shared CLI helpers for train.py and train_time_measure.py."""

import argparse
from pathlib import Path

from config import Config


def parse_split_cli(values):
    """
    argparse ``nargs='*'`` → split name(s).

    - ``MVI_1814 MVI_1810`` → list
    - ``MVI_1812`` → str
    - ``MVI_1814,MVI_1810`` (one token) → list
    """
    if values is None:
        return None
    if len(values) == 0:
        return None
    if len(values) == 1 and "," in values[0]:
        parts = [s.strip() for s in values[0].split(",") if s.strip()]
        return parts[0] if len(parts) == 1 else parts
    if len(values) == 1:
        return values[0]
    return list(values)


def add_base_train_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-dir", type=Path, default=None, help="Subject root (scene folders underneath)")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--train-split", "--flare-train-split",
        dest="train_split",
        nargs="*",
        metavar="SCENE",
        help="Train scene folder name(s), space-separated or comma in one arg",
    )
    parser.add_argument(
        "--eval-split", "--flare-eval-split",
        dest="eval_split",
        nargs="*",
        metavar="SCENE",
        help="Eval scene folder name(s), space-separated or comma in one arg",
    )
    parser.add_argument("--rebuild-mp-cache", action="store_true")
    parser.add_argument(
        "--gaussian-grow-option",
        choices=("grad2d", "gradrgb"),
        default=None,
        help="Densify grow signal: grad2d (viewspace) or gradrgb (color param grad)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Global RNG seed (default: Config.seed)")
    parser.add_argument(
        "--no-deterministic",
        action="store_true",
        help="Allow nondeterministic CUDA ops (faster; less reproducible)",
    )
    parser.add_argument(
        "--loss-overrides",
        type=Path,
        default=None,
        help="JSON file with config/basic_loss/stage loss overrides (sweep)",
    )
    parser.add_argument(
        "--no-mid-eval-render",
        action="store_true",
        help="Disable periodic eval render (see config eval_render_interval)",
    )
    parser.add_argument(
        "--no-stage-end-eval",
        action="store_true",
        help="Disable eval render at the end of each training stage",
    )
    parser.add_argument(
        "--final-eval-only",
        action="store_true",
        help="Render eval set once after all stages complete (implies --no-mid-eval-render)",
    )
    parser.add_argument(
        "--no-eval-checkpoint",
        action="store_true",
        help="Do not save extra eval_step_*.pt during render_eval_set",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Stage-end checkpoint .pt (e.g. checkpoints/stage_1_bootstrap_template_end_step_007500.pt)",
    )


def apply_base_train_cli(cfg: Config, args) -> Config:
    if args.input_dir is not None:
        cfg.input_dir = args.input_dir
    if args.output_root is not None:
        cfg.output_root = args.output_root
    train_split = parse_split_cli(args.train_split)
    if train_split is not None:
        cfg.train_split = train_split
    eval_split = parse_split_cli(args.eval_split)
    if eval_split is not None:
        cfg.eval_split = eval_split
    if args.rebuild_mp_cache:
        cfg.rebuild_mp_cache = True
    if args.gaussian_grow_option is not None:
        cfg.gaussian_grow_option = args.gaussian_grow_option
    if args.seed is not None:
        cfg.seed = args.seed
    if args.no_deterministic:
        cfg.deterministic = False
    if args.no_eval_checkpoint:
        cfg.save_eval_checkpoint = False
    return cfg
