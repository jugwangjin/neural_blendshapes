#!/usr/bin/env python
"""
Render raw vs personalized tracker mesh (gray + shadow) for completed sweep runs.

For each frame in train ∪ eval splits, writes:
  ``{output_root}/tracking_render_{scene}_gt/{frame}.png``
  ``{output_root}/tracking_render_{scene}_rgb/{frame}.png``
  ``{output_root}/tracking_render_{scene}_raw/{frame}.png``
  ``{output_root}/tracking_render_{scene}_personalized/{frame}.png``

After rendering, encodes each folder to ``{output_root}/tracking_video_{scene}_{gt,rgb,raw,personalized}.mp4``
(RGBA frame PNGs are composited onto white in the mp4).

Mesh is **tracker-only** (no template_mlp / expr_mlp / pose_weight_net).
Mesh ``raw`` / ``personalized`` PNGs: gray Lambert + green MP landmarks (478, iris 468–477)
projected from posed mesh vertices.

Usage (repo root, WSL/docker):
  python scripts/render_tracking_sweep.py --ablation all --dry-run
  python scripts/render_tracking_sweep.py --ablation default --device cuda
  python scripts/render_tracking_sweep.py --run-name justin --skip-existing
  python scripts/render_tracking_sweep.py --log-root .../neural_blendshapes_10 \\
      --run-name bala --device cuda
  # default ckpt: stage_3_expression_detail_end_step_037500.pt (latest final stage)
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_SCRIPTS = ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import imageio
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import collate_batch, move_batch_to_device
from dataset.dataset_util import normalize_split_names, scene_tag_from_image
from dataset.image_dataset import ImageDataset
from eval.tracking_eval_common import mesh_from_tracker_out_pure
from losses.mediapipe_landmark_478 import build_mp_lmk_embedding
from rendering.mesh_tracking_viz import render_mesh_gray_shadow_with_mp_landmarks
from run_status import (
    ABLATION_LOG_ROOTS,
    is_run_complete,
    log_root_for_ablation,
)
from training.landmark_debug_viz import eyelash_exclude_vertex_ids
from training.inference_render import render_gsplat_from_tracker_out
from training.inference_timing import InferenceTimer
from training.render_load import find_final_checkpoint, load_run_for_render
from training.stage_spec_io import resolve_render_stage_spec

from tracking_render_video import (
    assemble_tracking_videos,
    save_gsplat_frame,
    save_gt_raw_frame,
    tracking_frame_dir,
)


def _device_from_cli(device: str | None) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def infer_ablation_from_output_root(output_root: Path) -> str:
    resolved = output_root.parent.resolve()
    for name, root in ABLATION_LOG_ROOTS.items():
        if resolved == root.resolve():
            return name
    return "default"


def combined_all_splits(cfg):
    cfg2 = deepcopy(cfg)
    names = []
    for s in normalize_split_names(cfg.train_split) + normalize_split_names(cfg.eval_split):
        if s not in names:
            names.append(s)
    cfg2.train_split = names
    return cfg2


def build_all_frames_loader(cfg, *, num_workers: int = 0):
    cfg_all = combined_all_splits(cfg)
    dataset = ImageDataset(
        cfg_all,
        train=True,
        synthetic_if_empty=False,
        distribution_boost=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=False,
    )
    return loader, len(dataset)


@torch.no_grad()
def build_raw_tracker_out(tracker, batch, spec):
    """MediaPipe-only tracking: ICT raw coeffs + MP rotation, no MLP residuals."""
    corr = tracker(
        mp_blendshape=batch["mp_blendshape"],
        mp_landmarks_2d=batch.get("mp_landmarks_2d"),
        mp_landmarks_3d=batch.get("mp_landmarks_3d"),
        world_to_cam=batch.get("world_to_cam"),
        mp_pose_raw=batch.get("mp_pose_raw"),
        mp_transform_matrix=batch.get("mp_transform_matrix"),
        force_gamma_one=True,
        use_global_translation_param=False,
        additive_gamma_correction=getattr(spec, "additive_gamma_correction", False),
    )
    b = batch["mp_blendshape"].shape[0]
    device = batch["mp_blendshape"].device
    dtype = batch["mp_blendshape"].dtype
    z3 = torch.zeros(b, 3, device=device, dtype=dtype)
    ones = torch.ones(b, device=device, dtype=dtype)
    return {
        **corr,
        "coeffs": corr["coeffs_raw"],
        "ict_expression_weights": corr["coeffs_raw"],
        "pose_residual": corr["mp_rotation_6d"],
        "translation_residual": z3,
        "translation_global": z3,
        "pose_scale": ones,
    }


@torch.no_grad()
def build_personalized_tracker_out(tracker, batch, spec):
    return tracker(
        mp_blendshape=batch["mp_blendshape"],
        mp_landmarks_2d=batch.get("mp_landmarks_2d"),
        mp_landmarks_3d=batch.get("mp_landmarks_3d"),
        world_to_cam=batch.get("world_to_cam"),
        mp_pose_raw=batch.get("mp_pose_raw"),
        mp_transform_matrix=batch.get("mp_transform_matrix"),
        force_gamma_one=spec.fix_gamma_at_one,
        use_global_translation_param=getattr(spec, "use_global_translation_param", False),
        additive_gamma_correction=getattr(spec, "additive_gamma_correction", False),
    )


def _save_rgb(path: Path, rgb_bhw3: torch.Tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (rgb_bhw3[0].detach().cpu().numpy().clip(0, 1) * 255.0).round().astype("uint8")
    imageio.imwrite(str(path), arr)


def render_run(
    output_root: Path,
    *,
    device: torch.device,
    checkpoint: Path | str | None = None,
    skip_existing: bool,
    dry_run: bool,
    num_workers: int = 0,
    skip_videos: bool = False,
    videos_only: bool = False,
    video_fps: int = 25,
    video_codec: str | None = None,
    transparent_rgb: bool = True,
    mesh_landmarks: bool = True,
):
    infer_ablation = infer_ablation_from_output_root
    if checkpoint is None and not is_run_complete(output_root):
        print(f"skip (incomplete): {output_root}")
        return 0

    ckpt_path = find_final_checkpoint(output_root, checkpoint)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    spec, spec_source = resolve_render_stage_spec(payload, output_root, infer_ablation=infer_ablation)
    print(f"\n=== {output_root.name} ===")
    print(f"  ckpt: {ckpt_path.name}")
    print(f"  stage: {spec.name} ({spec_source})")

    if dry_run:
        cfg = payload["cfg"]
        cfg_all = combined_all_splits(cfg)
        n = sum(
            len(
                list(
                    (Path(cfg.input_dir) / scene / "image").glob("*.png")
                )
            )
            for scene in normalize_split_names(cfg_all.train_split)
            if (Path(cfg.input_dir) / scene / "image").is_dir()
        )
        print(f"  frames (approx): {n}")
        return n

    if videos_only:
        print("  videos-only (skip frame render)")
        videos = assemble_tracking_videos(
            output_root, fps=video_fps, video_codec=video_codec
        )
        print(f"  encoded {len(videos)} video(s)")
        return len(videos)

    stack = load_run_for_render(
        output_root,
        checkpoint=ckpt_path,
        device=device,
        infer_ablation=infer_ablation_from_output_root,
    )
    cfg = stack.cfg
    spec = stack.spec
    ict = stack.ict
    tracker = stack.tracker
    deformer = stack.deformer
    avatar = stack.avatar
    renderer = stack.renderer
    camera = stack.camera
    faces = ict.faces
    eyelash_exclude = eyelash_exclude_vertex_ids(ict)
    mp_lmk_emb = build_mp_lmk_embedding(cfg.mp_embedding, device)
    loader, n_frames = build_all_frames_loader(cfg, num_workers=num_workers)
    subject_root = Path(cfg.input_dir)
    scenes_seen: set[str] = set()
    gsplat_timer = InferenceTimer(device)
    mesh_timer = InferenceTimer(device)

    saved = 0
    for batch in tqdm(loader, total=n_frames, desc=output_root.name):
        batch = move_batch_to_device(batch, device)
        img_path = Path(batch["img_path"][0])
        frame_name = batch["frame_name"][0]
        scene = scene_tag_from_image(subject_root, img_path)
        scenes_seen.add(scene)

        out_gt = tracking_frame_dir(output_root, scene, "gt") / f"{frame_name}.png"
        out_rgb = tracking_frame_dir(output_root, scene, "rgb") / f"{frame_name}.png"
        out_raw = tracking_frame_dir(output_root, scene, "raw") / f"{frame_name}.png"
        out_pers = tracking_frame_dir(output_root, scene, "personalized") / f"{frame_name}.png"

        need_gt = not (skip_existing and out_gt.is_file())
        need_rgb = not (skip_existing and out_rgb.is_file())
        need_track = not (skip_existing and out_raw.is_file() and out_pers.is_file())
        if not (need_gt or need_rgb or need_track):
            continue

        pers_out = build_personalized_tracker_out(tracker, batch, spec)

        if need_gt:
            save_gt_raw_frame(out_gt, Path(batch["img_path"][0]), image_size=cfg.image_size)

        if need_rgb:
            composite = not transparent_rgb
            t0 = gsplat_timer.start()
            gs = render_gsplat_from_tracker_out(
                avatar, renderer, camera, pers_out, spec, composite=composite
            )
            gsplat_timer.stop(t0)
            save_gsplat_frame(
                out_rgb,
                gs["rgb"],
                gs.get("alpha"),
                transparent=transparent_rgb,
                composited_on_black=composite,
            )

        if need_track:
            raw_out = build_raw_tracker_out(tracker, batch, spec)
            t0 = mesh_timer.start()
            mesh_raw = mesh_from_tracker_out_pure(deformer, raw_out, spec)
            mesh_pers = mesh_from_tracker_out_pure(deformer, pers_out, spec)
            mesh_kw = dict(
                image_size=cfg.image_size,
                exclude_vertex_ids=eyelash_exclude,
                draw_landmarks=mesh_landmarks,
            )
            rgb_raw = render_mesh_gray_shadow_with_mp_landmarks(
                mesh_raw, faces, camera, mp_lmk_emb, **mesh_kw
            )
            rgb_pers = render_mesh_gray_shadow_with_mp_landmarks(
                mesh_pers, faces, camera, mp_lmk_emb, **mesh_kw
            )
            mesh_timer.stop(t0)
            _save_rgb(out_raw, rgb_raw)
            _save_rgb(out_pers, rgb_pers)

        saved += 1

    videos: list[Path] = []
    if not skip_videos:
        print("  encoding videos...")
        videos = assemble_tracking_videos(
            output_root,
            fps=video_fps,
            video_codec=video_codec,
            scenes=sorted(scenes_seen) if scenes_seen else None,
        )

    meta = {
        "checkpoint": str(ckpt_path),
        "stage": spec.name,
        "ablation": infer_ablation_from_output_root(output_root),
        "mesh_mode": "tracker_only",
        "mesh_excludes": ["template_mlp", "expr_mlp", "pose_weight_net"],
        "face_exclude_vertex_groups": ["eyelashes_left", "eyelashes_right"],
        "mesh_landmarks": mesh_landmarks,
        "mesh_landmark_style": "green_mp478_postprocess" if mesh_landmarks else None,
        "n_frames": n_frames,
        "n_rendered": saved,
        "video_fps": video_fps,
        "videos": [str(p) for p in videos],
        "rgb_transparent": transparent_rgb,
        "video_background": "white",
        "deformer_cache": True,
        "timing": {
            "gsplat": gsplat_timer.summary(),
            "mesh": mesh_timer.summary(),
        },
    }
    (output_root / "tracking_render_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"  rendered {saved}/{n_frames} frame sets -> tracking_render_*")
    gs = gsplat_timer.summary()
    ms = mesh_timer.summary()
    if gs["n"] > 0:
        print(f"  gsplat: {gs['fps']:.2f} fps ({gs['ms_per_frame']:.1f} ms/frame, n={gs['n']})")
    if ms["n"] > 0:
        print(f"  mesh: {ms['fps']:.2f} fps ({ms['ms_per_frame']:.1f} ms/frame, n={ms['n']})")
    if videos:
        print(f"  videos: {len(videos)} mp4 under {output_root}")
    return saved


def iter_log_roots(ablation: str):
    if ablation == "all":
        for root in ABLATION_LOG_ROOTS.values():
            yield root
        return
    yield log_root_for_ablation(ablation)


def main():
    p = argparse.ArgumentParser(description="Render raw vs personalized tracking for sweep runs.")
    p.add_argument(
        "--ablation",
        default="all",
        choices=["all", *sorted(ABLATION_LOG_ROOTS.keys())],
        help="Which log root(s) to scan (default: all).",
    )
    p.add_argument("--run-name", default=None, help="Glob filter on run directory name.")
    p.add_argument("--log-root", default=None, help="Override log root (single-root mode).")
    p.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Checkpoint .pt for each run (under run output_root). "
            "Examples: stage_2_coarse_mesh_end_step_022500.pt, "
            "checkpoints/stage_2_coarse_mesh_end_step_022500.pt"
        ),
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--skip-videos", action="store_true", help="Do not encode mp4 after rendering.")
    p.add_argument(
        "--videos-only",
        action="store_true",
        help="Skip frame render; encode mp4 from existing tracking_render_* folders.",
    )
    p.add_argument("--video-fps", type=int, default=25)
    p.add_argument("--video-codec", default=None, help="ffmpeg -c:v (default: auto libx264/libopenh264).")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--transparent",
        dest="transparent_rgb",
        action="store_true",
        default=True,
        help="RGBA PNG for gsplat rgb folders (default: on).",
    )
    p.add_argument(
        "--opaque",
        dest="transparent_rgb",
        action="store_false",
        help="Black-background RGB for gsplat (matches eval_render pred PNG).",
    )
    p.add_argument(
        "--no-landmarks",
        dest="mesh_landmarks",
        action="store_false",
        default=True,
        help="Mesh raw/personalized: gray only, no green MP landmark overlay.",
    )
    args = p.parse_args()

    device = _device_from_cli(args.device)
    pattern = args.run_name

    if args.log_root:
        roots = [Path(args.log_root)]
    else:
        roots = list(iter_log_roots(args.ablation))

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
            total += render_run(
                run_dir,
                device=device,
                checkpoint=args.checkpoint,
                skip_existing=args.skip_existing,
                dry_run=args.dry_run,
                num_workers=args.num_workers,
                skip_videos=args.skip_videos,
                videos_only=args.videos_only,
                video_fps=args.video_fps,
                video_codec=args.video_codec,
                transparent_rgb=args.transparent_rgb,
                mesh_landmarks=args.mesh_landmarks,
            )

    print(f"\nDone: {n_runs} run(s), {total} frame pair(s) {'(dry-run est.)' if args.dry_run else 'rendered'}.")


if __name__ == "__main__":
    main()
