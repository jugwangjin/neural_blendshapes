#!/usr/bin/env python
"""
Render gsplat control sequences (synthetic MediaPipe AU ramps) from trained run(s).

Single identity:
  python render_control_video.py \\
    --output-root /Bean/log/gwangjin/2026/neural_blendshapes_10/bala \\
    --sequence smile_wink --device cuda

Identity sweep (same layout as ``scripts/render_tracking_sweep.py``):
  python render_control_video.py --ablation all --device cuda
  python render_control_video.py --log-root .../neural_blendshapes_10 --run-name bala --device cuda

Frame PNGs default to straight-alpha RGBA (``--opaque`` for black background).
Encoded mp4 composites RGBA frames onto white (see ``images_to_mp4``).
  python render_control_video.py --list-sequences
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
_SCRIPTS = ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from dataset import collate_batch, move_batch_to_device
from dataset.image_dataset import ImageDataset
from rendering.control_sequences import (
    apply_mp_aus,
    build_control_sequence,
    emotion_catalog,
    list_control_sequences,
    list_emotion_sequence_names,
)
from run_status import ABLATION_LOG_ROOTS, is_run_complete, log_root_for_ablation
from training.inference_render import render_gsplat_from_tracker_out
from training.inference_timing import InferenceTimer
from training.render_load import find_final_checkpoint, load_run_for_render
from utils.mediapipe_blendshapes import load_mediapipe_mapping

from processing.process_video.frame_sequence_io import images_to_mp4
from tracking_render_video import save_gsplat_frame


def _device_from_cli(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_neutral_batch(cfg, device, *, frame_index: int = 0):
    ds = ImageDataset(
        cfg,
        train=True,
        synthetic_if_empty=False,
        distribution_boost=False,
    )
    if len(ds) == 0:
        raise RuntimeError(f"no frames under {cfg.input_dir}")
    j = int(frame_index) % len(ds)
    batch = collate_batch([ds[j]])
    return move_batch_to_device(batch, device), j


@torch.no_grad()
def render_control_sequence(
    *,
    output_root: Path,
    checkpoint: Path,
    sequence_name: str,
    out_dir: Path,
    device: torch.device,
    neutral_frame_index: int = 0,
    fps: int = 25,
    sequence_kwargs: dict | None = None,
    skip_existing: bool = False,
    transparent_rgb: bool = True,
):
    mp4_path = out_dir / f"control_{sequence_name}.mp4"
    if skip_existing and mp4_path.is_file():
        print(f"  skip (exists): {mp4_path.name}")
        return mp4_path

    stack = load_run_for_render(output_root, checkpoint=checkpoint, device=device)
    cfg = stack.cfg
    spec = stack.spec
    tracker = stack.tracker
    avatar = stack.avatar
    renderer = stack.renderer
    camera = stack.camera

    mp_map = load_mediapipe_mapping(cfg.mediapipe_name_to_ict)
    name_to_idx = mp_map.name_to_idx
    au_frames = build_control_sequence(sequence_name, **(sequence_kwargs or {}))

    neutral_batch, neutral_j = load_neutral_batch(cfg, device, frame_index=neutral_frame_index)
    base_mp = neutral_batch["mp_blendshape"][0].clone()

    frame_dir = out_dir / f"control_{sequence_name}" / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[Path] = []

    render_timer = InferenceTimer(device)
    desc = f"{output_root.name}/{sequence_name}"
    for i, aus in enumerate(tqdm(au_frames, desc=desc)):
        mp = apply_mp_aus(base_mp, name_to_idx, aus)
        mp_b = mp.unsqueeze(0)

        corr = tracker(
            mp_blendshape=mp_b,
            mp_landmarks_2d=neutral_batch.get("mp_landmarks_2d"),
            mp_landmarks_3d=neutral_batch.get("mp_landmarks_3d"),
            world_to_cam=neutral_batch.get("world_to_cam"),
            mp_pose_raw=neutral_batch.get("mp_pose_raw"),
            mp_transform_matrix=neutral_batch.get("mp_transform_matrix"),
            force_gamma_one=spec.fix_gamma_at_one,
            use_global_translation_param=getattr(spec, "use_global_translation_param", False),
            additive_gamma_correction=getattr(spec, "additive_gamma_correction", False),
        )
        t0 = render_timer.start()
        composite = not transparent_rgb
        gs = render_gsplat_from_tracker_out(
            avatar, renderer, camera, corr, spec, composite=composite
        )
        render_timer.stop(t0)
        path = frame_dir / f"{i:05d}.png"
        save_gsplat_frame(
            path,
            gs["rgb"],
            gs.get("alpha"),
            transparent=transparent_rgb,
            composited_on_black=composite,
        )
        frame_paths.append(path)

    images_to_mp4(frame_paths, mp4_path, fps=fps)

    timing = render_timer.summary()
    meta = {
        "checkpoint": str(checkpoint),
        "sequence": sequence_name,
        "n_frames": len(au_frames),
        "fps": fps,
        "neutral_frame_index": neutral_j,
        "video": str(mp4_path),
        "frames_dir": str(frame_dir),
        "stage": spec.name,
        "deformer_cache": True,
        "rgb_transparent": transparent_rgb,
        "video_background": "white",
        "timing": {
            "gsplat": timing,
        },
    }
    meta_path = out_dir / f"control_{sequence_name}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  wrote {mp4_path} ({len(frame_paths)} frames @ {fps} fps)")
    if timing["n"] > 0:
        print(
            f"  gsplat render: {timing['fps']:.2f} fps "
            f"({timing['ms_per_frame']:.1f} ms/frame, n={timing['n']})"
        )
    return mp4_path


def render_control_run(
    output_root: Path,
    *,
    device: torch.device,
    checkpoint: Path | str | None = None,
    sequence_names: list[str],
    out_dir: Path | None = None,
    neutral_frame_index: int = 0,
    fps: int = 25,
    sequence_kwargs: dict | None = None,
    skip_existing: bool = False,
    dry_run: bool = False,
    transparent_rgb: bool = True,
) -> int:
    if checkpoint is None and not is_run_complete(output_root):
        print(f"skip (incomplete): {output_root}")
        return 0

    ckpt_path = find_final_checkpoint(output_root, checkpoint)
    run_out = out_dir or (output_root / "control_video")
    print(f"\n=== {output_root.name} ===")
    print(f"  ckpt: {ckpt_path.name}")
    print(f"  out: {run_out}")

    if dry_run:
        print(f"  sequences: {sequence_names}")
        return len(sequence_names)

    run_out.mkdir(parents=True, exist_ok=True)
    n_done = 0
    for name in sequence_names:
        render_control_sequence(
            output_root=output_root,
            checkpoint=ckpt_path,
            sequence_name=name,
            out_dir=run_out,
            device=device,
            neutral_frame_index=neutral_frame_index,
            fps=fps,
            sequence_kwargs=sequence_kwargs,
            skip_existing=skip_existing,
            transparent_rgb=transparent_rgb,
        )
        n_done += 1

    sweep_meta = {
        "checkpoint": str(ckpt_path),
        "sequences": sequence_names,
        "n_sequences": n_done,
        "out_dir": str(run_out),
        "fps": fps,
        "rgb_transparent": transparent_rgb,
        "video_background": "white",
    }
    (run_out / "control_render_meta.json").write_text(
        json.dumps(sweep_meta, indent=2), encoding="utf-8"
    )
    return n_done


def iter_log_roots(ablation: str):
    if ablation == "all":
        for root in ABLATION_LOG_ROOTS.values():
            yield root
        return
    yield log_root_for_ablation(ablation)


def main():
    p = argparse.ArgumentParser(description="Render FACS control sequences (gsplat).")
    p.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Single run directory (contains checkpoints/). Omit for identity sweep.",
    )
    p.add_argument(
        "--ablation",
        default="all",
        choices=["all", *sorted(ABLATION_LOG_ROOTS.keys())],
        help="Log root(s) for sweep when --output-root is omitted (default: all).",
    )
    p.add_argument("--run-name", default=None, help="Glob filter on run directory name (sweep).")
    p.add_argument("--log-root", default=None, help="Override log root (single-root sweep).")
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint .pt for each run (under run output_root).",
    )
    p.add_argument(
        "--sequence",
        action="append",
        default=None,
        help="Sequence name (repeatable). Default: emotions_all. Use --list-sequences / --list-emotions.",
    )
    p.add_argument("--list-sequences", action="store_true", help="Print all sequence registry keys.")
    p.add_argument(
        "--list-emotions",
        action="store_true",
        help="Print iMotions emotion → FACS AU order (sequential activation).",
    )
    p.add_argument("--au-peak", type=float, default=0.75, help="Peak AU activation per emotion step (0–1).")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output dir (default: each run's output-root/control_video).",
    )
    p.add_argument("--neutral-frame-index", type=int, default=0, help="Dataset frame for rest pose / landmarks.")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--device", default="auto")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-existing", action="store_true", help="Skip sequence if control_*.mp4 exists.")
    p.add_argument(
        "--transparent",
        dest="transparent_rgb",
        action="store_true",
        default=True,
        help="RGBA PNG frames with straight alpha (default: on).",
    )
    p.add_argument(
        "--opaque",
        dest="transparent_rgb",
        action="store_false",
        help="Black-background RGB frames (matches eval_render pred PNG).",
    )
    args = p.parse_args()

    if args.list_sequences:
        for name in list_control_sequences():
            print(name)
        return

    if args.list_emotions:
        for row in emotion_catalog():
            print(f"{row['sequence']:22s}  {row['label']:20s}  AUs: {row['aus']}")
        print(f"\n{len(list_emotion_sequence_names())} emotions; concat: emotions_all")
        return

    device = _device_from_cli(args.device)
    names = args.sequence if args.sequence else ["emotions_all"]
    seq_kw = dict(peak=args.au_peak)

    if args.output_root is not None:
        n = render_control_run(
            Path(args.output_root),
            device=device,
            checkpoint=args.checkpoint,
            sequence_names=names,
            out_dir=args.out_dir,
            neutral_frame_index=args.neutral_frame_index,
            fps=args.fps,
            sequence_kwargs=seq_kw,
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
            transparent_rgb=args.transparent_rgb,
        )
        print(f"\nDone: 1 run, {n} sequence(s) {'(dry-run)' if args.dry_run else 'rendered'}.")
        return

    if args.out_dir is not None:
        print("WARNING: --out-dir ignored in sweep mode (each run uses output-root/control_video)")

    if args.log_root:
        roots = [Path(args.log_root)]
    else:
        roots = list(iter_log_roots(args.ablation))

    pattern = args.run_name
    total = 0
    n_runs = 0
    for log_root in roots:
        if not log_root.is_dir():
            print(f"log root missing: {log_root}")
            continue
        for run_dir in sorted(log_root.iterdir()):
            if not run_dir.is_dir():
                continue
            if pattern and not fnmatch.fnmatch(run_dir.name, pattern):
                continue
            n_runs += 1
            total += render_control_run(
                run_dir,
                device=device,
                checkpoint=args.checkpoint,
                sequence_names=names,
                out_dir=None,
                neutral_frame_index=args.neutral_frame_index,
                fps=args.fps,
                sequence_kwargs=seq_kw,
                skip_existing=args.skip_existing,
                dry_run=args.dry_run,
                transparent_rgb=args.transparent_rgb,
            )

    print(f"\nDone: {n_runs} run(s), {total} sequence(s) {'(dry-run)' if args.dry_run else 'rendered'}.")


if __name__ == "__main__":
    main()
