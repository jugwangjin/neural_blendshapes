#!/usr/bin/env python
"""
Sweep Configs: Runs train.py sequentially or in parallel across multiple GPUs 
by sweeping through all config files inside the configs_tmp/ directory.

Usage:
    # Dry run to see what configs are found and what commands will be executed
    python sweep_configs.py --dry-run

    # Run sequentially on a single GPU (GPU 0)
    python sweep_configs.py --gpus 0

    # Run in parallel across multiple GPUs (e.g., GPU 0, 1, 2)
    python sweep_configs.py --gpus 0 1 2

    # Filter specific configs
    python sweep_configs.py --gpus 0 1 --pattern "ablation_*.txt"
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
    """Parses key-value assignments from custom config txt format."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    config = {}
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()

            # Clean outer quotes
            if (val.startswith("'") and val.endswith("'")) or (val.startswith('"') and val.endswith('"')):
                val = val[1:-1]

            # Try to parse list structure
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


def worker_gpu(gpu_id: str, task_queue: queue.Queue, dry_run: bool, extra_args: list):
    """Worker thread that pulls config tasks and runs them on a specific GPU."""
    while True:
        try:
            # Non-blocking get so threads can exit when queue is empty
            task = task_queue.get_nowait()
        except queue.Empty:
            break

        config_path, config = task
        run_name = config["run_name"]
        input_dir = config["input_dir"]
        train_dir = config.get("train_dir")
        eval_dir = config.get("eval_dir")

        output_root = f"/Bean/log/gwangjin/2026/neural_blendshapes/{run_name}"

        # Construct python command
        cmd = ["python", "train.py"]
        if input_dir:
            cmd.extend(["--input-dir", str(input_dir)])
        cmd.extend(["--output-root", output_root])

        if train_dir:
            cmd.append("--train-split")
            if isinstance(train_dir, list):
                cmd.extend(train_dir)
            else:
                cmd.append(train_dir)

        if eval_dir:
            cmd.append("--eval-split")
            if isinstance(eval_dir, list):
                cmd.extend(eval_dir)
            else:
                cmd.append(eval_dir)

        # Forward extra args
        if extra_args:
            cmd.extend(extra_args)

        cmd_str = " ".join(cmd)
        print(f"\n[GPU {gpu_id}] Starting task: {config_path.name}")
        print(f"  run_name:    {run_name}")
        print(f"  output_root: {output_root}")
        print(f"  command:     CUDA_VISIBLE_DEVICES={gpu_id} {cmd_str}\n")

        if not dry_run:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            env["PYTHONIOENCODING"] = "utf-8"
            env["LANG"] = "C.UTF-8"
            env["LC_ALL"] = "C.UTF-8"
            try:
                subprocess.run(cmd, env=env, check=True)
                print(f"[GPU {gpu_id}] Completed task: {config_path.name} successfully.")
            except subprocess.CalledProcessError as e:
                print(f"[GPU {gpu_id}] Error running task: {config_path.name} ({e})")
            except Exception as e:
                print(f"[GPU {gpu_id}] Unexpected error on task: {config_path.name} ({e})")
        else:
            print(f"[GPU {gpu_id}] [DRY-RUN] Bypassed execution of {config_path.name}")

        task_queue.task_done()


def main():
    parser = argparse.ArgumentParser(description="Sweep configurations across GPUs")
    parser.add_argument("--gpus", nargs="+", default=["0"], help="List of GPU indices to distribute tasks across (e.g. 0 1 2)")
    parser.add_argument("--pattern", default="*.txt", help="Glob pattern to select configs (e.g. '00*.txt' or 'ablation_*.txt')")
    parser.add_argument("--exclude", nargs="+", default=["*copy*", "*debug*", "*local*"], help="Glob patterns of files to exclude")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    args, extra_args = parser.parse_known_args()

    configs_dir = Path("configs_tmp")
    if not configs_dir.is_dir():
        print("Error: 'configs_tmp' directory does not exist in workspace root.")
        sys.exit(1)

    # Find matching config files
    all_files = list(configs_dir.glob(args.pattern))
    valid_tasks = []

    for f in all_files:
        # Check exclusion patterns
        should_exclude = False
        for ex in args.exclude:
            if f.match(ex):
                should_exclude = True
                break
        if should_exclude:
            continue

        try:
            cfg = parse_config_txt(f)
            if "run_name" in cfg and "input_dir" in cfg:
                valid_tasks.append((f, cfg))
        except Exception as e:
            print(f"Warning: Failed to parse '{f.name}' ({e})")

    if not valid_tasks:
        print(f"No valid configs containing 'run_name' and 'input_dir' found in '{configs_dir}' matching patterns.")
        sys.exit(0)

    print(f"\nFound {len(valid_tasks)} configuration task(s) to sweep:")
    for path, cfg in valid_tasks:
        print(f" - {path.name} (run_name: {cfg['run_name']})")
    print(f"Configured GPUs: {', '.join(args.gpus)}")
    if args.dry_run:
        print("!!! DRY-RUN MODE: No commands will actually be executed !!!")
    print("==================================================\n")

    # Load tasks into thread-safe Queue
    task_queue = queue.Queue()
    for task in valid_tasks:
        task_queue.put(task)

    # Spawn worker threads (one per specified GPU)
    threads = []
    for gpu in args.gpus:
        t = Thread(target=worker_gpu, args=(gpu, task_queue, args.dry_run, extra_args), daemon=True)
        t.start()
        threads.append(t)

    # Wait for all threads to finish
    for t in threads:
        t.join()

    print("\nAll sweep tasks completed.")


if __name__ == "__main__":
    main()
