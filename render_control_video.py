"""
MediaPipe → tracker MLP → ICT deformer → surface/eye Gaussians → gsplat.

Dataset: ``dataset.image_dataset.ImageDataset`` (image split layout under ``Config.input_dir``).

Stages: 0 bootstrap pose → 1 mesh+tracker → 2A expression → 2B GS detail → 3 view appearance

Run from repo root:
  python train.py
  python train.py --input-dir /path/to/subject --train-split MVI_1814 MVI_1810 MVI_1811 --eval-split MVI_1812
"""

import argparse
import sys
import os
from datetime import datetime
from pathlib import Path

# Force UTF-8 encoding for standard I/O and JIT compiler subprocesses
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["LANG"] = "C.UTF-8"
os.environ["LC_ALL"] = "C.UTF-8"

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import Config
from dataset import build_train_dataset, collate_batch, move_batch_to_device
from rendering import GaussianRenderer
from losses.train_losses import compute_losses
from model.expr_regions import build_expr_region_weight
from model.gaussian_avatar import GaussianAvatar
from model.ict_deformer import ICTDeformer
from model.ict_model import ICTFaceKitTorch
from model.tracker_mlp import TrackerCorrectionMLP
from training.apply import (
    apply_stage_requires_grad,
    build_optimizers,
    init_training_state,
    stage_loss_cfg,
    stage_needs_rasterization,
    stage_needs_surface_forward,
)
from training.checkpoint_io import save_checkpoint
from training.code_dump import dump_training_code
from training.eval_render import render_eval_set, save_deformed_template_obj
from training.stages import STAGE_SCHEDULE, iter_stages, total_training_steps
from training.densify import BarycentricDensificationStrategy
from training.triangle_walking import TriangleWalker
from utils.camera import load_training_camera, training_camera_status
from losses.mediapipe_landmark_478 import build_mp_lmk_embedding
from losses.pie68_jaw_landmark import build_pie68_jaw_vertex_indices
from utils.sampling import count_surface_gaussians


def _code_dump_dir(cfg: Config) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return cfg.codes_dir / stamp


def _tqdm_postfix(losses, global_step: int, n_show: int = 6) -> dict:
    items = [(k, losses[k].item()) for k in losses if k != "total"]
    items.sort(key=lambda x: -abs(x[1]))
    out = {"step": global_step, "loss": f"{losses['total'].item():.4f}"}
    for k, v in items[:n_show]:
        out[k] = f"{v:.4f}"
    return out


@torch.no_grad()
def save_landmark_debug_image(path, vertices, faces, mp_landmarks_2d, mp_lmk_emb, camera, image_size, gt_image):
    import cv2
    import numpy as np
    from losses.mediapipe_landmark_478 import vertices2landmarks_barycentric

    # 1. Get predicted and target UVs
    mp_ids = mp_lmk_emb["mp_ids"]
    face_idx = mp_lmk_emb["face_idx"]
    bary = mp_lmk_emb["bary"]

    lmk_xyz = vertices2landmarks_barycentric(vertices, faces, face_idx, bary)
    proj = camera.project_world_points(lmk_xyz.reshape(-1, 3)).reshape(vertices.shape[0], -1, 2)
    pred_uv = (proj / float(image_size))[0].detach().cpu().numpy() # [N, 2]

    target_uv = mp_landmarks_2d[0, mp_ids].detach().cpu().numpy() # [N, 2]

    # 2. Get GT image and convert to HWC BGR uint8
    img = gt_image.detach().cpu().permute(1, 2, 0).numpy()
    img = (img.clip(0, 1) * 255.0).round().astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 3. Draw dots
    h, w = img_bgr.shape[:2]
    for i in range(len(target_uv)):
        tx, ty = int(target_uv[i, 0] * w), int(target_uv[i, 1] * h)
        px, py = int(pred_uv[i, 0] * w), int(pred_uv[i, 1] * h)
        cv2.circle(img_bgr, (tx, ty), 2, (0, 0, 255), -1) # Red for GT
        cv2.circle(img_bgr, (px, py), 2, (0, 255, 0), -1) # Green for Rendered

    cv2.imwrite(str(path), img_bgr)


def _parse_split_cli(values):
    """
    argparse ``nargs='*'`` → ``SplitNames``.

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


def parse_train_cli():
    p = argparse.ArgumentParser(description="Train MediaPipe → ICT → 3DGS avatar")
    p.add_argument("--input-dir", type=Path, default=None, help="Subject root (scene folders underneath)")
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument(
        "--train-split", "--flare-train-split",
        dest="train_split",
        nargs="*",
        metavar="SCENE",
        help="Train scene folder name(s), space-separated or comma in one arg",
    )
    p.add_argument(
        "--eval-split", "--flare-eval-split",
        dest="eval_split",
        nargs="*",
        metavar="SCENE",
        help="Eval scene folder name(s), space-separated or comma in one arg",
    )
    p.add_argument("--rebuild-mp-cache", action="store_true")
    return p.parse_args()


def apply_train_cli(cfg: Config, args) -> Config:
    if args.input_dir is not None:
        cfg.input_dir = args.input_dir
    if args.output_root is not None:
        cfg.output_root = args.output_root
    train_split = _parse_split_cli(args.train_split)
    if train_split is not None:
        cfg.train_split = train_split
    eval_split = _parse_split_cli(args.eval_split)
    if eval_split is not None:
        cfg.eval_split = eval_split
    if args.rebuild_mp_cache:
        cfg.rebuild_mp_cache = True
    return cfg


def resolve_existing_input_dir(input_dir) -> Path:
    p = Path(input_dir)
    curr = p
    while curr != curr.parent:
        if curr.is_dir():
            if curr != p:
                print(f"Auto-resolved nonexistent input_dir '{p}' to existing path '{curr}'")
            return curr
        curr = curr.parent
    return p


def main():
    cfg = apply_train_cli(Config(), parse_train_cli())
    cfg.input_dir = resolve_existing_input_dir(cfg.input_dir)
    from processing.ict_mediapipe_lmk.embedding_io import resolve_embedding_path

    cfg.mp_embedding = resolve_embedding_path(cfg.mp_embedding)
    assert cfg.batch_size == 1, "avatar/render path is single-mesh; set batch_size=1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.eval_render_dir.mkdir(parents=True, exist_ok=True)

    schedule = STAGE_SCHEDULE
    cfg.iterations = total_training_steps(schedule)

    dump_training_code(ROOT, _code_dump_dir(cfg), cfg, schedule)

    dataset = build_train_dataset(cfg, train=True)
    from dataset.dataset_util import format_splits_label

    split_label = format_splits_label(cfg.train_split)
    n_frames = len(dataset)
    print(
        f"dataset: ImageDataset ({cfg.dataset_type}) "
        f"{cfg.input_dir}/{{{split_label}}}/image — {n_frames} frames, image_size={cfg.image_size}"
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    ict = ICTFaceKitTorch(npy_dir=str(cfg.ict_npy)).to(device)

    tracker = TrackerCorrectionMLP(
        mediapipe_to_ict=ict.mediapipe_to_ict,
        num_ict_expression=ict.num_expression,
        n_blendshapes=cfg.num_mp_blendshapes,
        gamma_min=cfg.gamma_min,
        gamma_max=cfg.gamma_max,
        use_landmarks=True,
    ).to(device)

    expr_region_weight = build_expr_region_weight(ict).to(device)
    deformer = ICTDeformer(
        ict,
        expr_region_weight,
        mediapipe_name_to_ict=str(cfg.mediapipe_name_to_ict),
        n_coeffs=cfg.num_ict_expressions,
    ).to(device)

    avatar = GaussianAvatar.from_ict(
        ict,
        deformer=deformer,
        k_face=cfg.n_surface_gaussians_per_face,
        k_head=cfg.n_surface_gaussians_per_head,
        k_mouth_socket=cfg.n_surface_gaussians_per_mouth_socket,
        k_mouth_interior=cfg.n_surface_gaussians_mouth_interior,
        k_eye_socket=cfg.n_surface_gaussians_per_eye_socket,
        k_eyeball_sclera=cfg.n_surface_gaussians_per_eyeball_sclera,
        k_eye_occlusion=cfg.n_surface_gaussians_per_eye_occlusion,
        n_semantic_classes=cfg.n_semantic_classes,
        gum_h_sigma_scale=cfg.gum_h_sigma_scale,
        gaussian_scale_knn_k=cfg.gaussian_scale_knn_k,
        gaussian_scale_knn_factor=cfg.gaussian_scale_knn_factor,
        face_center_init=cfg.gaussian_face_center_init,
        max_scale=cfg.geometry_max_scale,
    ).to(device)

    n_surface = count_surface_gaussians(
        ict,
        ict.faces,
        k_face=cfg.n_surface_gaussians_per_face,
        k_head=cfg.n_surface_gaussians_per_head,
        k_mouth_socket=cfg.n_surface_gaussians_per_mouth_socket,
        k_mouth_interior=cfg.n_surface_gaussians_mouth_interior,
        k_eye_socket=cfg.n_surface_gaussians_per_eye_socket,
        k_eyeball_sclera=cfg.n_surface_gaussians_per_eyeball_sclera,
        k_eye_occlusion=cfg.n_surface_gaussians_per_eye_occlusion,
        face_center_init=cfg.gaussian_face_center_init,
    )
    print(
        f"surface Gaussians: {n_surface} "
        f"(face/head={cfg.n_surface_gaussians_per_face}/{cfg.n_surface_gaussians_per_head} "
        f"mouth_socket={cfg.n_surface_gaussians_per_mouth_socket} "
        f"mouth_interior={cfg.n_surface_gaussians_mouth_interior} "
        f"eye_socket={cfg.n_surface_gaussians_per_eye_socket} "
        f"sclera/occlusion={cfg.n_surface_gaussians_per_eyeball_sclera}/"
        f"{cfg.n_surface_gaussians_per_eye_occlusion} per face; "
        f"gum h_sigma×{cfg.gum_h_sigma_scale}; eye sclera h=0)"
    )

    init_training_state(avatar)

    # Initialize layout colors for Gaussians from region_colors.py
    try:
        from debug.sanity.region_colors import surface_gaussian_rgb, rgb_to_logit
        colors = surface_gaussian_rgb(avatar, ict, ict.canonical[0], device, mp_embedding_path=cfg.mp_embedding)
        avatar.color.data.copy_(rgb_to_logit(colors))
        print("Initialized Gaussian surface colors with layout region colors.")
    except Exception as e:
        print(f"Warning: Could not initialize surface layout colors ({e})")

    renderer = GaussianRenderer(cfg, image_size=cfg.image_size, sh_degree=None).to(device)
    camera = load_training_camera(
        ict.canonical[0],
        path=cfg.camera_npz,
        width=cfg.image_size,
        height=cfg.image_size,
        device=device,
    )
    print(f"camera: {training_camera_status(cfg.camera_npz)}")
    if not cfg.camera_npz.is_file():
        print(
            "  bake metrical crop: python processing/compute_camera_for_metrical_crop.py "
            "--apply-train-view --write-npz"
        )

    mp_lmk_emb = build_mp_lmk_embedding(cfg.mp_embedding, device)
    print(f"MP→ICT landmark embedding: {cfg.mp_embedding} ({mp_lmk_emb['mp_ids'].numel()} landmarks)")
    pie68_jaw_vertex_idx = build_pie68_jaw_vertex_indices(ict, device)
    print(
        f"PIE-68 jawline: {pie68_jaw_vertex_idx.numel()} ICT verts "
        f"(protocol 0:{ict.landmark_start}, FA batch['landmark'][:, :{ict.landmark_start}])"
    )
    ict_faces = ict.faces.to(device)
    triangle_walker = TriangleWalker(ict_faces, ict.canonical[0], max_iterations=3)

    densify_strategy = BarycentricDensificationStrategy(cfg)
    global_step = 0
    mesh_optim = None
    gaussian_optim = None
    current_spec = None

    eval_ds = build_train_dataset(cfg, train=False)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=0,
    ) if len(eval_ds) > 0 else None

    for stage_idx, spec, stage_start, stage_end in iter_stages(schedule):
        if spec.steps <= 0:
            continue

        if stage_idx == 1 and global_step == 0:
            
            render_eval_set(
                cfg,
                spec,
                tracker,
                avatar,
                renderer,
                camera,
                device,
                out_dir=cfg.eval_render_dir,
                global_step=global_step,
                max_frames=cfg.eval_max_frames,
                eval_loader=eval_loader,
            )

        print(f"\n=== Stage {stage_idx}: {spec.name} ({spec.steps} steps) ===")
        print(spec.description)

        renderer.set_sh_degree(spec.sh_degree)
        cfg.sh_degree = spec.sh_degree

        apply_stage_requires_grad(spec, tracker, deformer, avatar)
        mesh_optim, gaussian_optim = build_optimizers(spec, tracker, deformer, avatar)
        if spec.name in cfg.gaussian_densify_stages:
            densify_strategy.reset_state(
                len(avatar.surface.h), avatar.surface.h.device
            )
        loss_cfg = stage_loss_cfg(spec)
        loss_cfg.image_size = cfg.image_size
        loss_cfg.mp_lmk_iris_weight = cfg.mp_lmk_iris_weight
        loss_cfg.silhouette_use_edt = cfg.silhouette_use_edt
        loss_cfg.silhouette_edt_w_ext = cfg.silhouette_edt_w_ext
        loss_cfg.silhouette_edt_w_int = cfg.silhouette_edt_w_int
        loss_cfg.silhouette_edt_max_dist_px = cfg.silhouette_edt_max_dist_px
        stage_local = 0

        pbar = tqdm(
            total=spec.steps,
            desc=f"stage {stage_idx} {spec.name}",
            unit="step",
            dynamic_ncols=True,
        )
        loader_iter = iter(loader)
        for _ in range(spec.steps):
            if stage_local >= spec.steps:
                break
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)
            stage_local += 1
            global_step += 1

            batch = move_batch_to_device(batch, device)

            corr = tracker(
                mp_blendshape=batch["mp_blendshape"],
                mp_landmarks_2d=batch.get("mp_landmarks_2d"),
                mp_landmarks_3d=batch.get("mp_landmarks_3d"),
                world_to_cam=batch.get("world_to_cam"),
                mp_pose_raw=batch.get("mp_pose_raw"),
                mp_transform_matrix=batch.get("mp_transform_matrix"),
                force_gamma_one=spec.fix_gamma_at_one,
            )

            pose_weight_fixed = 1.0 if spec.pose_weight_one else None
            need_render = stage_needs_rasterization(loss_cfg)
            need_surface = stage_needs_surface_forward(loss_cfg)
            avatar_out = avatar(
                tracker_out=corr,
                apply_expression_deform=spec.train_expression_deform,
                use_pose_scale=spec.apply_pose_scale,
                pose_weight_fixed=pose_weight_fixed,
                rotate_about_centroid=spec.pose_rotate_about_centroid,
                pose_zero_tz=spec.pose_zero_tz,
                skip_surface=not need_surface,
            )
            expr_delta = avatar_out.get("expr_delta")

            render = None
            if need_render:
                render_semantic = loss_cfg.w_seg > 0
                render = renderer(
                    avatar_out,
                    camera,
                    render_semantic=render_semantic,
                )
            losses = compute_losses(
                loss_cfg,
                batch,
                render,
                avatar_out,
                camera,
                mp_lmk_emb,
                ict_faces,
                pie68_jaw_vertex_idx=pie68_jaw_vertex_idx,
                corr=corr,
                deformer=deformer,
                expr_delta=expr_delta,
                avatar=avatar,
                renderer=renderer,
            )

            if mesh_optim is not None:
                mesh_optim.zero_grad(set_to_none=True)
            if gaussian_optim is not None:
                gaussian_optim.zero_grad(set_to_none=True)

            if spec.name in cfg.gaussian_densify_stages and render is not None:
                densify_strategy.pre_backward(global_step, render, avatar=avatar)

            losses["total"].backward()

            if (
                spec.name in cfg.gaussian_densify_stages
                and gaussian_optim is not None
                and render is not None
            ):
                densify_strategy.post_backward(global_step, avatar, render)

            if spec.name == "0_bootstrap_pose" and global_step % 100 == 0:
                dbg_dir = cfg.eval_render_dir / "bootstrap_debug"
                dbg_dir.mkdir(parents=True, exist_ok=True)
                dbg_path = dbg_dir / f"step_{global_step:06d}.png"
                save_landmark_debug_image(
                    dbg_path,
                    avatar_out["mesh_xyz"],
                    ict_faces,
                    batch["mp_landmarks_2d"],
                    mp_lmk_emb,
                    camera,
                    cfg.image_size,
                    batch["image"][0],
                )
                save_deformed_template_obj(
                    dbg_dir / f"step_{global_step:06d}_mesh.obj",
                    avatar_out["mesh_xyz"],
                    ict_faces,
                )

            if mesh_optim is not None:
                mesh_optim.step()
            if gaussian_optim is not None:
                gaussian_optim.step()
                if (
                    spec.train_gaussian_geometry
                    and global_step % max(1, cfg.gaussian_triangle_walk_every) == 0
                ):
                    triangle_walker.step(avatar.surface, gaussian_optim)
                if spec.name in cfg.gaussian_densify_stages:
                    densify_strategy.post_optimizer_step(
                        global_step, avatar, gaussian_optim, ict_faces, ict
                    )

            pbar.update(1)
            if global_step % cfg.log_every == 0 or stage_local >= spec.steps:
                pbar.set_postfix(_tqdm_postfix(losses, global_step), refresh=False)

            if global_step % 500 == 0:
                active_losses = [f"total: {losses['total'].item():.4f}"]
                for k, v in sorted(losses.items()):
                    if k != "total" and abs(v.item()) > 1e-6:
                        active_losses.append(f"{k}: {v.item():.4f}")
                tqdm.write(f"[Step {global_step:06d}] stage {stage_idx} ({spec.name}): " + ", ".join(active_losses))

        pbar.close()
        current_spec = spec
        stage_ckpt = cfg.checkpoint_dir / f"stage_{spec.name}_end_step_{global_step:06d}.pt"
        save_checkpoint(
            stage_ckpt,
            global_step=global_step,
            stage_name=spec.name,
            tracker=tracker,
            deformer=deformer,
            avatar=avatar,
            cfg=cfg,
            extra={"stage_end": True, "stage_steps": spec.steps},
        )
        render_eval_set(
            cfg,
            spec,
            tracker,
            avatar,
            renderer,
            camera,
            device,
            out_dir=cfg.eval_render_dir,
            global_step=global_step,
            max_frames=cfg.eval_max_frames,
            eval_loader=eval_loader,
        )

    print(f"\nDone. Total steps: {global_step}. Last stage: {current_spec.name if current_spec else 'n/a'}")


if __name__ == "__main__":
    main()
