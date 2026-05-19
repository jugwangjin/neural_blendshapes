"""
MediaPipe → tracker MLP → ICT deformer → surface/eye Gaussians → gsplat.

Stages: 0 bootstrap pose → 1 mesh+tracker → 2A expression → 2B GS detail → 3 view appearance

Run from repo root:
  python train.py
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import Config
from dataset.video_dataset import VideoDataset, collate_batch
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
)
from training.stages import STAGE_SCHEDULE, iter_stages, total_training_steps
from utils.accessory_detect import segmentation_has_accessory
from utils.camera import FixedCamera
from losses.mediapipe_landmark_478 import embedding_tensors, load_mediapipe_ict_embedding
from utils.sampling import count_surface_gaussians


def load_mp_embedding(path, device):
    emb = load_mediapipe_ict_embedding(path)
    mp_ids, face_idx, bary = embedding_tensors(emb, device)
    emb["_mp_ids"] = mp_ids
    emb["_face_idx"] = face_idx
    emb["_bary"] = bary
    n = len(emb["mp_landmark_indices"])
    print(f"MP→ICT landmark embedding: {path} ({n} landmarks)")
    return emb


def main():
    cfg = Config()
    assert cfg.batch_size == 1, "avatar/render path is single-mesh; set batch_size=1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    schedule = STAGE_SCHEDULE
    cfg.iterations = total_training_steps(schedule)

    dataset = VideoDataset(cfg, train=True, au_active_boost=True)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=0,
    )

    ict = ICTFaceKitTorch(npy_dir=str(cfg.ict_npy)).to(device)

    tracker = TrackerCorrectionMLP(
        n_blendshapes=cfg.num_mp_blendshapes,
        gamma_min=cfg.gamma_min,
        gamma_max=cfg.gamma_max,
        gaze_uv_range=cfg.gaze_uv_range,
        use_landmarks=True,
    ).to(device)

    expr_region_weight = build_expr_region_weight(ict).to(device)
    deformer = ICTDeformer(
        ict,
        expr_region_weight,
        n_coeffs=cfg.num_mp_blendshapes,
    ).to(device)

    n_acc = cfg.n_accessory_gaussians
    if cfg.auto_detect_accessory and n_acc == 0:
        if segmentation_has_accessory(
            cfg.segmentation_dir,
            cfg.train_scenes,
            min_pixel_ratio=cfg.accessory_min_pixel_ratio,
        ):
            n_acc = 512
            print(f"accessory detected in segmentation — using n_accessory_gaussians={n_acc}")

    n_surface = count_surface_gaussians(
        ict,
        ict.faces,
        k_face=cfg.n_surface_gaussians_per_face,
        k_head=cfg.n_surface_gaussians_per_head,
        k_mouth_socket=cfg.n_surface_gaussians_per_mouth_socket,
        k_mouth_interior=cfg.n_surface_gaussians_mouth_interior,
        k_eye_socket=cfg.n_surface_gaussians_per_eye_socket,
    )
    print(
        f"surface Gaussians: {n_surface} "
        f"(face/head={cfg.n_surface_gaussians_per_face}/{cfg.n_surface_gaussians_per_head} "
        f"mouth_socket={cfg.n_surface_gaussians_per_mouth_socket} "
        f"mouth_interior={cfg.n_surface_gaussians_mouth_interior} "
        f"eye_socket={cfg.n_surface_gaussians_per_eye_socket}; "
        f"gum/k={cfg.n_surface_gaussians_mouth_interior} h_sigma×{cfg.gum_h_sigma_scale}; "
        f"teeth/eyeball mesh skipped)"
    )

    avatar = GaussianAvatar.from_ict(
        ict,
        deformer=deformer,
        k_face=cfg.n_surface_gaussians_per_face,
        k_head=cfg.n_surface_gaussians_per_head,
        k_mouth_socket=cfg.n_surface_gaussians_per_mouth_socket,
        k_mouth_interior=cfg.n_surface_gaussians_mouth_interior,
        k_eye_socket=cfg.n_surface_gaussians_per_eye_socket,
        n_eye_per_side=cfg.n_eye_gaussians_per_side,
        n_accessory_gaussians=n_acc,
        gaze_uv_range=cfg.gaze_uv_range,
        learn_gaze_refine=cfg.learn_gaze_refine,
        n_semantic_classes=cfg.n_semantic_classes,
        gum_h_sigma_scale=cfg.gum_h_sigma_scale,
        eye_uv_sample_mode=cfg.eye_uv_sample_mode,
        eye_sclera_min_front_dot=cfg.eye_sclera_min_front_dot,
        eye_sclera_hemisphere_only=cfg.eye_sclera_hemisphere_only,
        gaussian_scale_knn_k=cfg.gaussian_scale_knn_k,
        gaussian_scale_knn_factor=cfg.gaussian_scale_knn_factor,
    ).to(device)

    init_training_state(avatar)

    renderer = GaussianRenderer(cfg, image_size=cfg.image_size, sh_degree=None).to(device)
    camera = FixedCamera.from_default_or_mesh(
        ict.canonical[0],
        path=cfg.camera_npz,
        width=cfg.image_size,
        height=cfg.image_size,
        device=device,
    )
    pivot = ict.canonical[0].mean(dim=0)
    camera = camera.with_view_correction(pivot)
    if not cfg.camera_npz.is_file():
        print(
            f"note: {cfg.camera_npz} missing — using mesh-bounds camera; "
            "bake with: python scripts/bake_default_camera.py"
        )

    mp_embedding = load_mp_embedding(cfg.mp_embedding, device)
    ict_faces = ict.faces

    global_step = 0
    mesh_optim = None
    gaussian_optim = None
    current_spec = None

    for stage_idx, spec, stage_start, stage_end in iter_stages(schedule):
        if spec.steps <= 0:
            continue

        print(f"\n=== Stage {stage_idx}: {spec.name} ({spec.steps} steps) ===")
        print(spec.description)

        renderer.set_sh_degree(spec.sh_degree)
        cfg.sh_degree = spec.sh_degree

        apply_stage_requires_grad(spec, tracker, deformer, avatar)
        mesh_optim, gaussian_optim = build_optimizers(spec, tracker, deformer, avatar)
        loss_cfg = stage_loss_cfg(spec)
        stage_local = 0

        while stage_local < spec.steps:
            for batch in loader:
                stage_local += 1
                global_step += 1

                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                corr = tracker(
                    mp_blendshape=batch["mp_blendshape"],
                    mp_landmarks_2d=batch.get("mp_landmarks_2d"),
                    mp_pose_raw=batch.get("mp_pose_raw"),
                    force_gamma_one=spec.fix_gamma_at_one,
                )

                pose_weight_fixed = 1.0 if spec.pose_weight_one else None
                avatar_out = avatar(
                    tracker_out=corr,
                    apply_expression_deform=spec.train_expression_deform,
                    use_pose_scale=spec.apply_pose_scale,
                    pose_weight_fixed=pose_weight_fixed,
                    rotate_about_centroid=spec.pose_rotate_about_centroid,
                    pose_zero_tz=spec.pose_zero_tz,
                )
                expr_delta = avatar_out.get("expr_delta")

                render_semantic = spec.w_seg > 0
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
                    mp_embedding,
                    ict_faces,
                    corr=corr,
                    deformer=deformer,
                    expr_delta=expr_delta,
                    avatar=avatar,
                )

                if mesh_optim is not None:
                    mesh_optim.zero_grad(set_to_none=True)
                if gaussian_optim is not None:
                    gaussian_optim.zero_grad(set_to_none=True)

                losses["total"].backward()

                if mesh_optim is not None:
                    mesh_optim.step()
                if gaussian_optim is not None:
                    gaussian_optim.step()

                if global_step % cfg.log_every == 0:
                    parts = " ".join(
                        f"{k}={v.item():.4f}" for k, v in losses.items() if k != "total"
                    )
                    print(
                        f"step {global_step} [{spec.name}:{stage_local}] "
                        f"loss={losses['total'].item():.4f} {parts}"
                    )

                if global_step % cfg.save_every == 0:
                    ckpt = cfg.checkpoint_dir / f"step_{global_step:06d}_{spec.name}.pt"
                    torch.save(
                        {
                            "step": global_step,
                            "stage": spec.name,
                            "tracker": tracker.state_dict(),
                            "deformer": deformer.state_dict(),
                            "avatar": avatar.state_dict(),
                            "cfg": cfg,
                        },
                        ckpt,
                    )
                    print(f"saved {ckpt}")

                if stage_local >= spec.steps:
                    break

        current_spec = spec

    print(f"\nDone. Total steps: {global_step}. Last stage: {current_spec.name if current_spec else 'n/a'}")


if __name__ == "__main__":
    main()
