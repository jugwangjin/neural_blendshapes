#!/usr/bin/env python
"""
Helper script to run train.py using settings parsed from a config txt file.
Usage:
    python run_config.py <config_name_or_path> [extra_train_args...]
Example:
    python run_config.py 001
    python run_config.py configs_tmp/justin.txt --rebuild-mp-cache
"""

import sys
import os
import re
import ast
import subprocess
from pathlib import Path


def parse_config_txt(file_path: Path) -> dict:
    """Parses key-value assignments from the custom config txt format."""
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

            # Clean outer quotes if any
            if (val.startswith("'") and val.endswith("'")) or (val.startswith('"') and val.endswith('"')):
                val = val[1:-1]

            # Try to parse list structure
            if val.startswith("[") and val.endswith("]"):
                try:
                    parsed_list = ast.literal_eval(val)
                    if isinstance(parsed_list, list):
                        val = parsed_list
                except Exception:
                    # Fallback simple parser for list
                    inner = val[1:-1].strip()
                    val = [p.strip().strip("'").strip('"') for p in inner.split(",") if p.strip()]

            config[key] = val
    return config


def main():
    if len(sys.argv) < 2:
        print("Error: Please specify a config name or path.")
        print("Usage: python run_config.py <config_name_or_path> [extra_train_args...]")
        sys.exit(1)

    config_arg = sys.argv[1]
    extra_args = sys.argv[2:]

    # Resolve config file path
    config_path = Path(config_arg)
    if not config_path.is_file():
        # Try appending .txt
        if not config_arg.endswith(".txt"):
            config_path = Path(f"{config_arg}.txt")
        
        # Try looking inside configs_tmp/ directory
        if not config_path.is_file():
            config_path = Path("configs_tmp") / config_path.name

    if not config_path.is_file():
        print(f"Error: Config file not found at '{config_arg}' or 'configs_tmp/{config_path.name}'")
        sys.exit(1)

    print(f"Reading config: {config_path}")
    config = parse_config_txt(config_path)

    # Extract required fields
    run_name = config.get("run_name")
    input_dir = config.get("input_dir")
    train_dir = config.get("train_dir")
    eval_dir = config.get("eval_dir")

    if not run_name:
        print("Error: 'run_name' not found in config.")
        sys.exit(1)

    # Construct output_root path
    output_root = f"/Bean/log/gwangjin/2026/neural_blendshapes/{run_name}"

    print("\n--- Extracted Settings ---")
    print(f"run_name:    {run_name}")
    print(f"input_dir:   {input_dir}")
    print(f"train_dir:   {train_dir}")
    print(f"eval_dir:    {eval_dir}")
    print(f"output_root: {output_root}")
    print("--------------------------\n")

    # Build the train.py command
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

    # Forward any additional CLI arguments
    if extra_args:
        cmd.extend(extra_args)

    # Force UTF-8 encoding in standard I/O and subprocesses
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["LANG"] = "C.UTF-8"
    env["LC_ALL"] = "C.UTF-8"

    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
