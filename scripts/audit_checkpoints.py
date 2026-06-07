#!/usr/bin/env python
"""
List Gaussian counts in every stage-end checkpoint vs loss_log densify stats.

Usage (repo root, WSL/docker):
  python scripts/audit_checkpoints.py --output-root /Bean/log/.../bala
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datetime import datetime

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.build import estimate_init_gaussian_count, sh_dim_from_avatar_state
from training.checkpoint_io import avatar_n_from_state_dict


def _fmt_mtime(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _ckpt_stats(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    av = payload["avatar"]
    color = av["color"]
    if color.ndim == 3:
        dc = color[:, 0, :]
    else:
        dc = color
    dc_sig = torch.sigmoid(dc).float().mean().item()
    n = avatar_n_from_state_dict(av)
    n_meta = payload.get("n_gaussians")
    return {
        "path": path.name,
        "step": int(payload.get("step", 0)),
        "stage": payload.get("stage", "?"),
        "n_gaussians": n,
        "n_meta": int(n_meta) if n_meta is not None else None,
        "sh_dim": sh_dim_from_avatar_state(av),
        "color_shape": tuple(color.shape),
        "log_scale_mean": float(av["log_scale"].float().mean().item()),
        "color_dc_sigmoid_mean": dc_sig,
        "mtime": path.stat().st_mtime,
    }


def _loss_log_densify_peak(log_path: Path) -> tuple[int, int]:
    if not log_path.is_file():
        return 0, 0
    peak_n = 0
    peak_step = 0
    with log_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for key in ("densify/n_gaussian", "densify_ema/n_gaussian"):
                v = rec.get(key)
                if v is None:
                    continue
                n = int(v)
                if n > peak_n:
                    peak_n = n
                    peak_step = int(rec.get("global_step", 0))
    return peak_n, peak_step


def _estimate_init_n(cfg) -> int | None:
    try:
        from model.build import build_ict

        device = torch.device("cpu")
        ict = build_ict(cfg, device)
        return estimate_init_gaussian_count(cfg, ict)
    except Exception as e:
        print(f"init count estimate skipped: {e}")
        return None


def _baseline_n(rows) -> int | None:
    for r in rows:
        if r["stage"] in ("0_bootstrap_identity", "1_bootstrap_template"):
            return int(r["n_gaussians"])
    return int(rows[0]["n_gaussians"]) if rows else None


def _row_by_stage(rows, stage_name: str):
    matches = [r for r in rows if r["stage"] == stage_name]
    return matches[-1] if matches else None


def main():
    p = argparse.ArgumentParser(description="Audit checkpoint Gaussian counts")
    p.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args()

    output_root = Path(args.output_root)
    ckpt_dir = output_root / "checkpoints"
    if not ckpt_dir.is_dir():
        raise SystemExit(f"no checkpoints dir: {ckpt_dir}")

    ckpts = sorted(ckpt_dir.glob("stage_*_end_step_*.pt"), key=lambda p: _ckpt_stats(p)["step"])
    if not ckpts:
        raise SystemExit(f"no stage-end checkpoints in {ckpt_dir}")

    rows = [_ckpt_stats(p) for p in ckpts]
    baseline = _baseline_n(rows)
    init_n = None
    try:
        sample = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
        init_n = _estimate_init_n(sample["cfg"])
    except Exception:
        pass

    log_path = output_root / "analysis" / "loss_log.jsonl"
    peak_n, peak_step = _loss_log_densify_peak(log_path)

    print(f"output_root: {output_root}")
    if baseline is not None:
        print(f"bootstrap layout (stage 0/1 ckpt): {baseline} Gaussians")
    if init_n is not None and baseline is not None and abs(init_n - baseline) > baseline * 0.02:
        print(f"  (code estimate from cfg: {init_n} — may differ if cfg/ict changed)")
    if peak_n > 0:
        print(f"loss_log densify peak: {peak_n} at global_step={peak_step}")
    print()

    hdr = (
        f"{'checkpoint':<48} {'step':>6} {'stage':<22} {'n':>8} "
        f"{'sh':>3} {'log_s':>8} {'dc_sig':>8}  mtime"
    )
    print(hdr)
    print("-" * len(hdr))
    prev_n = None
    for r in rows:
        flags = []
        if baseline is not None and r["n_gaussians"] <= baseline * 1.01 and r["stage"] == "3_expression_detail":
            flags.append("STALE? (back to bootstrap count)")
        if prev_n is not None and r["n_gaussians"] < prev_n * 0.95:
            flags.append(f"DROP from prev stage ({prev_n} -> {r['n_gaussians']})")
        flag = f"  <-- {'; '.join(flags)}" if flags else ""
        print(
            f"{r['path']:<48} {r['step']:>6} {r['stage']:<22} {r['n_gaussians']:>8} "
            f"{r['sh_dim']:>3} {r['log_scale_mean']:>8.4f} {r['color_dc_sigmoid_mean']:>8.4f}  "
            f"{_fmt_mtime(r['mtime'])}{flag}"
        )
        if r["n_meta"] is not None and r["n_meta"] != r["n_gaussians"]:
            print(f"  WARNING: payload n_gaussians={r['n_meta']} != face_idx={r['n_gaussians']}")
        prev_n = r["n_gaussians"]

    print()
    s2 = _row_by_stage(rows, "2_coarse_mesh")
    s3 = _row_by_stage(rows, "3_expression_detail")
    if s2 and s3 and s3["n_gaussians"] < s2["n_gaussians"] * 0.95:
        print("DIAGNOSIS: stage_3 checkpoint has FEWER Gaussians than stage_2 end.")
        if s3["mtime"] < s2["mtime"]:
            print(
                f"  stage_3 mtime ({_fmt_mtime(s3['mtime'])}) is OLDER than stage_2 "
                f"({_fmt_mtime(s2['mtime'])})."
            )
            print("  => stage_3 file is almost certainly a STALE leftover from an earlier run.")
            print("  => stage_2 was re-trained; stage_3 was never re-completed.")
        else:
            print(
                f"  stage_3 mtime ({_fmt_mtime(s3['mtime'])}) is NEWER than stage_2 — "
                "investigate in-run avatar reset (resume from wrong ckpt?)."
            )
        print()
        print("Render / verify currently pick stage_3 (final). Use stage_2 until stage_3 is re-run:")
        print(f"  --checkpoint checkpoints/{s2['path']}")
        print()
        print("Re-run stage 3 only:")
        print(f"  python train.py --output-root {output_root} --resume checkpoints/{s2['path']}")
    elif s2 and peak_n > s2["n_gaussians"] * 1.05:
        print(
            f"Note: loss_log densify peak ({peak_n}) > stage_2 end ({s2['n_gaussians']}) "
            "(normal after prune at stage end)."
        )


if __name__ == "__main__":
    main()
