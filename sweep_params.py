#!/usr/bin/env python
"""
Hyperparameter sweep for a single subject (default: Config() values).

Runs ``train.py`` sequentially with loss/config overrides, saves stage-end
checkpoints only, renders once at training completion, then aggregates GT metrics.

Usage:
    python sweep_params.py --dry-run
    python sweep_params.py --gpus 0
    python sweep_params.py --gpus 0 --report-only
    python sweep_params.py --gpus 0 --start-from 3
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from config import Config
from training.eval_metrics import (
    compute_render_metrics,
    find_final_render_dir,
    write_metrics_report,
)


_CFG = Config()
SUBJECT_INPUT_DIR = str(_CFG.input_dir)
SUBJECT_TRAIN_SPLIT = _CFG.train_split
SUBJECT_EVAL_SPLIT = _CFG.eval_split
LOG_ROOT = Path(_CFG.output_root).parent
RUN_PREFIX = Path(_CFG.output_root).name


def run_output_root(log_root: Path, run_id: str) -> Path:
    """``<log_root>/<run_prefix>_{run_id}``."""
    return Path(log_root) / f"{RUN_PREFIX}_{run_id}"


def _exp(run_id: str, description: str, overrides: dict) -> dict:
    return {"run_id": run_id, "description": description, "overrides": overrides}


def build_experiments() -> list[dict]:
    """One-axis sweeps over selected 5 knobs only."""
    base_stage3 = {"w_lpips": 0.005, "lpips_start_local": 10000}

    return [
        _exp("baseline", "current defaults", {}),
        _exp("w_rgb_3p0", "weaker photometric", {"basic_loss": {"w_rgb": 3.0}}),
        _exp("w_rgb_5p0", "stronger photometric", {"basic_loss": {"w_rgb": 5.0}}),
        _exp("w_sil_2p0", "weaker silhouette", {"basic_loss": {"w_silhouette": 2.0}}),
        _exp("w_sil_4p0", "stronger silhouette", {"basic_loss": {"w_silhouette": 4.0}}),
        _exp("w_scaling_0p10", "weaker scale regularization", {"basic_loss": {"w_scaling": 0.10}}),
        _exp("w_scaling_0p30", "stronger scale regularization", {"basic_loss": {"w_scaling": 0.30}}),
        _exp(
            "tpl_smooth_1e3",
            "template smoothness stronger",
            {"basic_loss": {"w_template_smooth": 1e-4}},
        ),
        _exp(
            "tpl_smooth_1e4",
            "template smoothness weaker",
            {"basic_loss": {"w_template_smooth": 1e-6}},
        ),
        # _exp(
        #     "tpl_lap_1e3",
        #     "template laplacian stronger",
        #     {"basic_loss": {"w_template_laplacian": 1e-6}},
        # ),
        # _exp(
        #     "tpl_lap_1e4",
        #     "template laplacian weaker",
        #     {"basic_loss": {"w_template_laplacian": 1e-8}},
        # ),
        _exp("w_lpips_0p0", "no late LPIPS", {"stages": {"3_expression_detail": {**base_stage3, "w_lpips": 0.0}}}),
        _exp(
            "w_lpips_0p01",
            "stronger late LPIPS",
            {"stages": {"3_expression_detail": {**base_stage3, "w_lpips": 0.01}}},
        ),
        _exp("split_bary_0p06", "lower split bary noise", {"config": {"gaussian_split_bary_noise_gb_match": False, "gaussian_split_bary_noise": 0.06}}),
        _exp("split_bary_0p18", "higher split bary noise", {"config": {"gaussian_split_bary_noise_gb_match": False, "gaussian_split_bary_noise": 0.18}}),
        # --- Gaussian quality (large/overlapping/blurry) A-B isolation vs new defaults ---
        _exp(
            "gauss_legacy",
            "revert all 6 quality knobs to pre-fix behavior",
            {
                "config": {
                    "gaussian_grow_grad2d_face_scale": 5.0,
                    "gaussian_reset_stage_local": {},
                    "gaussian_prune_world_scale_ratio": 0.1,
                    "gaussian_prune_screen_after_local": 3000,
                    "gaussian_densify_stages": ["2_coarse_mesh"],
                    "gaussian_densify_stage_local": {"2_coarse_mesh": [1, 15000]},
                    "gaussian_scale_max_clamp_factor": 0.0,
                },
                "stages": {
                    "2_coarse_mesh": {"w_lpips": 0.0},
                    "3_expression_detail": {"w_lpips": 0.005, "lpips_start_local": 10000},
                },
            },
        ),
        _exp(
            "face_scale_1p0",
            "most aggressive densify (threshold = SA 0.0002)",
            {"config": {"gaussian_grow_grad2d_face_scale": 1.0}},
        ),
        _exp(
            "face_scale_2p5",
            "milder densify",
            {"config": {"gaussian_grow_grad2d_face_scale": 2.5}},
        ),
        _exp(
            "no_opacity_reset",
            "disable opacity reset only",
            {"config": {"gaussian_reset_stage_local": {}}},
        ),
        _exp(
            "world_prune_0p1",
            "looser world-scale prune only",
            {"config": {"gaussian_prune_world_scale_ratio": 0.1}},
        ),
        _exp(
            "scale_clamp_off",
            "disable scale hard cap only",
            {"config": {"gaussian_scale_max_clamp_factor": 0.0}},
        ),
        _exp(
            "scale_clamp_3x",
            "tighter scale hard cap (3x max_scale)",
            {"config": {"gaussian_scale_max_clamp_factor": 3.0}},
        ),
        _exp(
            "no_stage3_densify",
            "densify only in coarse mesh (no stage-3 densify)",
            {
                "config": {
                    "gaussian_densify_stages": ["2_coarse_mesh"],
                    "gaussian_densify_stage_local": {"2_coarse_mesh": [1, 15000]},
                }
            },
        ),
    ]


def _write_overrides_json(path: Path, overrides: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(overrides, f, indent=2, sort_keys=True)


def _run_train(
    exp: dict,
    output_root: Path,
    overrides_path: Path,
    gpu: str,
    dry_run: bool,
    extra_args: list[str],
):
    cmd = [
        "python",
        "train.py",
        "--input-dir",
        SUBJECT_INPUT_DIR,
        "--output-root",
        str(output_root),
        "--train-split",
        *([SUBJECT_TRAIN_SPLIT] if isinstance(SUBJECT_TRAIN_SPLIT, str) else SUBJECT_TRAIN_SPLIT),
        "--eval-split",
        *([SUBJECT_EVAL_SPLIT] if isinstance(SUBJECT_EVAL_SPLIT, str) else SUBJECT_EVAL_SPLIT),
        "--loss-overrides",
        str(overrides_path),
        "--final-eval-only",
        "--no-mid-eval-render",
        "--no-eval-checkpoint",
    ]
    cmd.extend(extra_args)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONIOENCODING"] = "utf-8"
    env["LANG"] = "C.UTF-8"
    env["LC_ALL"] = "C.UTF-8"

    print(f"\n=== [{exp['run_id']}] {exp['description']} ===")
    print(f"output_root: {output_root}")
    print(f"CUDA_VISIBLE_DEVICES={gpu} {' '.join(cmd)}")

    if dry_run:
        return

    subprocess.run(cmd, env=env, check=True)


def _metrics_for_run(exp: dict, output_root: Path, max_frames: int) -> dict:
    from config import Config

    cfg = Config()
    cfg.input_dir = Path(SUBJECT_INPUT_DIR)
    cfg.train_split = SUBJECT_TRAIN_SPLIT
    cfg.eval_split = SUBJECT_EVAL_SPLIT
    cfg.output_root = output_root

    render_dir = find_final_render_dir(output_root)
    if render_dir is None:
        raise FileNotFoundError(f"no final render dir under {output_root / 'renders'}")

    metrics = compute_render_metrics(cfg, render_dir, max_frames=max_frames)
    return {
        "run_id": exp["run_id"],
        "description": exp["description"],
        "output_root": str(output_root),
        "render_dir": str(render_dir),
        "overrides": exp["overrides"],
        "means": metrics["means"],
        "rows": metrics["rows"],
    }


def _load_saved_meta(run_dir: Path) -> dict | None:
    meta_path = run_dir / "sweep_meta.json"
    if not meta_path.is_file():
        return None
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _discover_sweep_run_dirs(log_root: Path) -> list[Path]:
    """``<run_prefix>_*`` run directories under ``log_root`` that contain ``sweep_meta.json``."""
    log_root = Path(log_root)
    runs = []
    for p in sorted(log_root.iterdir()):
        if not p.is_dir():
            continue
        if not p.name.startswith(f"{RUN_PREFIX}_"):
            continue
        if (p / "sweep_meta.json").is_file():
            runs.append(p)
    return runs


def main():
    parser = argparse.ArgumentParser(description="Loss hyperparameter sweep")
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-only", action="store_true", help="Skip training; metrics + report only")
    parser.add_argument("--start-from", type=int, default=0, help="Start experiment index")
    parser.add_argument("--max-runs", type=int, default=0, help="Limit number of experiments (0=all)")
    parser.add_argument(
        "--max-eval-frames",
        type=int,
        default=0,
        help="Cap eval frames for metrics (0=all)",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=LOG_ROOT,
        help="Parent log dir; each run -> log_root/<run_prefix>_{run_id}",
    )
    args, extra_args = parser.parse_known_args()

    experiments = build_experiments()
    if args.start_from > 0:
        experiments = experiments[args.start_from :]
    if args.max_runs > 0:
        experiments = experiments[: args.max_runs]

    log_root = Path(args.log_root)
    log_root.mkdir(parents=True, exist_ok=True)
    report_dir = log_root / f"{RUN_PREFIX}_sweep_report"

    results = []
    gpu_cycle = list(args.gpus)

    for i, exp in enumerate(experiments):
        run_dir = run_output_root(log_root, exp["run_id"])
        overrides_path = run_dir / "loss_overrides.json"
        _write_overrides_json(overrides_path, exp["overrides"])
        meta = {
            "run_id": exp["run_id"],
            "description": exp["description"],
            "overrides": exp["overrides"],
            "input_dir": SUBJECT_INPUT_DIR,
            "output_root": str(run_dir),
        }
        (run_dir / "sweep_meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        gpu = gpu_cycle[i % len(gpu_cycle)]

        if not args.report_only:
            _run_train(exp, run_dir, overrides_path, gpu, args.dry_run, extra_args)

        if args.dry_run:
            continue

        try:
            result = _metrics_for_run(exp, run_dir, args.max_eval_frames)
            results.append(result)
            (run_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "run_id": result["run_id"],
                        "means": result["means"],
                        "render_dir": result["render_dir"],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            m = result["means"]
            print(
                f"  metrics: PSNR={m['psnr']:.3f} SSIM={m['ssim']:.4f} "
                f"LPIPS={m['lpips']:.4f} L1={m['l1']:.4f} ({m['n_frames']} frames)"
            )
        except Exception as e:
            print(f"  metrics FAILED for {exp['run_id']}: {e}")

    if args.dry_run:
        print(f"\n[DRY-RUN] {len(experiments)} experiment(s) under {log_root}/{RUN_PREFIX}_*")
        return

    if not results and args.report_only:
        for run_dir in _discover_sweep_run_dirs(log_root):
            meta = _load_saved_meta(run_dir)
            if meta is None:
                continue
            exp = {
                "run_id": meta["run_id"],
                "description": meta.get("description", ""),
                "overrides": meta.get("overrides", {}),
            }
            try:
                results.append(_metrics_for_run(exp, run_dir, args.max_eval_frames))
            except Exception as e:
                print(f"metrics skip {run_dir.name}: {e}")

    if results:
        csv_path, md_path = write_metrics_report(report_dir, results)
        print(f"\nSweep report written:\n  {csv_path}\n  {md_path}")
    else:
        print("\nNo metrics collected.")


if __name__ == "__main__":
    main()
