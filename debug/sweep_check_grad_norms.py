#!/usr/bin/env python
"""
Sweep grad-norm diagnostics: runs debug/check_grad_norms.py across configs_tmp/ subjects,
but only when a specific checkpoint exists.

Default checkpoint gate:
  /Bean/log/gwangjin/2026/neural_blendshapes/{run_name}/checkpoints/stage_1_coarse_mesh_end_step_020000.pt

Usage:
  python sweep_check_grad_norms.py --dry-run
  python sweep_check_grad_norms.py --gpus 0 1 2 3
  python sweep_check_grad_norms.py --pattern "*.txt" --exclude "*debug*" "*local*"
  python sweep_check_grad_norms.py --ckpt-name "stage_1_coarse_mesh_end_step_020000.pt"
"""

import sys
import os
import ast
import argparse
import subprocess
import queue
from pathlib import Path
from threading import Thread


def parse_config_txt(file_path: Path) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    config = {}
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()

        if (val.startswith("'") and val.endswith("'")) or (val.startswith('"') and val.endswith('"')):
            val = val[1:-1]

        if val.startswith("[") and val.endswith("]"):
            try:
                parsed_list = ast.literal_eval(val)
                if isinstance(parsed_list, list):
                    val = parsed_list
            except Exception:
                inner = val[1:-1].strip()
                val = [p.strip().strip("'").strip('"') for p in inner.split(",") if p.strip()]

        config[key] = val
    return config


def _ckpt_path(run_name: str, ckpt_name: str) -> Path:
    return Path(f"/Bean/log/gwangjin/2026/neural_blendshapes/{run_name}/checkpoints/{ckpt_name}")


def worker_gpu(gpu_id: str, task_queue: queue.Queue, dry_run: bool, extra_args: list):
    while True:
        try:
            task = task_queue.get_nowait()
        except queue.Empty:
            break

        config_path, config, ckpt = task
        run_name = str(config["run_name"])
        input_dir = config.get("input_dir")
        train_dir = config.get("train_dir")
        eval_dir = config.get("eval_dir")
        output_root = f"/Bean/log/gwangjin/2026/neural_blendshapes/{run_name}"

        cmd = ["python", "debug/check_grad_norms.py"]
        if input_dir:
            cmd.extend(["--input-dir", str(input_dir)])
        cmd.extend(["--output-root", output_root])
        if train_dir:
            cmd.append("--train-split")
            cmd.extend(train_dir if isinstance(train_dir, list) else [train_dir])
        if eval_dir:
            cmd.append("--eval-split")
            cmd.extend(eval_dir if isinstance(eval_dir, list) else [eval_dir])
        if extra_args:
            cmd.extend(extra_args)

        print(f"\n[GPU {gpu_id}] Grad-norm task: {config_path.name}")
        print(f"  run_name: {run_name}")
        print(f"  ckpt:     {ckpt}")
        print(f"  command:  CUDA_VISIBLE_DEVICES={gpu_id} {' '.join(cmd)}\n")

        if not dry_run:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            env["PYTHONIOENCODING"] = "utf-8"
            env["LANG"] = "C.UTF-8"
            env["LC_ALL"] = "C.UTF-8"
            try:
                subprocess.run(cmd, env=env, check=True)
                print(f"[GPU {gpu_id}] Completed: {config_path.name}")
            except subprocess.CalledProcessError as e:
                print(f"[GPU {gpu_id}] Error: {config_path.name} ({e})")
        else:
            print(f"[GPU {gpu_id}] [DRY-RUN] skipped execution")

        task_queue.task_done()


def main():
    p = argparse.ArgumentParser(description="Sweep debug/check_grad_norms.py across subjects with checkpoint gate")
    p.add_argument("--gpus", nargs="+", default=["0"])
    p.add_argument("--pattern", default="*.txt")
    p.add_argument("--exclude", nargs="+", default=["*copy*", "*debug*", "*local*"])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--ckpt-name", default="stage_0_bootstrap_pose_end_step_005000.pt")
    args, extra_args = p.parse_known_args()

    configs_dir = Path("configs_tmp")
    if not configs_dir.is_dir():
        print("Error: 'configs_tmp' directory does not exist in workspace root.")
        sys.exit(1)

    all_files = list(configs_dir.glob(args.pattern))
    tasks = []
    skipped = []

    for f in all_files:
        if any(f.match(ex) for ex in args.exclude):
            continue
        try:
            cfg = parse_config_txt(f)
            if "run_name" not in cfg or "input_dir" not in cfg:
                continue
            run_name = str(cfg["run_name"])
            ckpt = _ckpt_path(run_name, args.ckpt_name)
            if ckpt.is_file():
                tasks.append((f, cfg, str(ckpt)))
            else:
                skipped.append((f.name, str(ckpt)))
        except Exception as e:
            print(f"Warning: Failed to parse '{f.name}' ({e})")

    if not tasks:
        print("No tasks found (no matching configs with checkpoint present).")
        if skipped:
            print("Skipped (missing checkpoint):")
            for name, ckpt in skipped:
                print(f" - {name}: {ckpt}")
        sys.exit(0)

    print(f"\nFound {len(tasks)} grad-norm task(s) with checkpoint '{args.ckpt_name}':")
    for path, cfg, ckpt in tasks:
        print(f" - {path.name} (run_name: {cfg['run_name']})")
    print(f"Configured GPUs: {', '.join(args.gpus)}")
    if skipped:
        print(f"Skipped {len(skipped)} config(s) without checkpoint.")
    if args.dry_run:
        print("!!! DRY-RUN MODE: No commands will actually be executed !!!")
    print("==================================================\n")

    q = queue.Queue()
    for t in tasks:
        q.put(t)

    threads = []
    for gpu in args.gpus:
        th = Thread(target=worker_gpu, args=(gpu, q, args.dry_run, extra_args), daemon=True)
        th.start()
        threads.append(th)
    for th in threads:
        th.join()

    print("\nAll grad-norm sweep tasks completed.")


if __name__ == "__main__":
    main()

