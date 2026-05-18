"""
MediaPipe → tracker MLP → ICT deformer → UVH/eye Gaussians → 3DGS.

2-stage schedule:
  1 coarse geometry (tracker + template deformer)
  2A expression warmup → 2B Gaussian detail

Run from repo root:
  python train.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import Config
from dataset.video_dataset import VideoDataset, collate_batch
from gaussian_splatting.renderer import GaussianRenderer
from losses.train_losses import compute_losses
from model.expr_regions import build_expr_region_weight
from model.expression_deform_mlp import SupportGatedExpressionDeformer
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
from utils.camera import FixedCamera


def load_mp_embedding(path):
    d = np.load(path, allow_pickle=True)
    return {
        "mp_landmark_indices": d["mp_landmark_indices"],
        "ict_lmk_face_idx": d["ict_lmk_face_idx"],
        "ict_lmk_b_coords": d["ict_lmk_b_coords"],
    }


def main():
    cfg = Config()
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

    ict = ICTFaceKitTorch(
        npy_dir=str(cfg.ict_npy),
        canonical=str(cfg.ict_canonical),
    ).to(device)

    tracker = TrackerCorrectionMLP(
        n_blendshapes=cfg.num_mp_blendshapes,
        gamma_min=cfg.gamma_min,
        gamma_max=cfg.gamma_max,
    ).to(device)

    deformer = ICTDeformer(ict).to(device)
    expr_region_weight = build_expr_region_weight(ict).to(device)
    expr_deform = SupportGatedExpressionDeformer(
        ict,
        expr_region_weight,
        n_coeffs=cfg.num_mp_blendshapes,
    ).to(device)

    avatar = GaussianAvatar.from_ict(
        ict,
        n_face_gaussians=cfg.n_face_gaussians,
        n_eye_per_side=cfg.n_eye_gaussians_per_side,
        gaze_uv_range=cfg.gaze_uv_range,
        learn_gaze_refine=cfg.learn_gaze_refine,
        n_semantic_classes=cfg.n_semantic_classes,
    ).to(device)

    init_training_state(avatar, expr_deform)

    renderer = GaussianRenderer(cfg, image_size=cfg.image_size).to(device)
    camera = FixedCamera.from_default_npz(cfg.camera_npz, width=cfg.image_size, height=cfg.image_size)

    mp_embedding = load_mp_embedding(cfg.mp_embedding)
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

        apply_stage_requires_grad(spec, tracker, deformer, avatar, expr_deform)
        mesh_optim, gaussian_optim = build_optimizers(spec, tracker, deformer, avatar, expr_deform)
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

                expr_delta = None
                if spec.train_expression_deform:
                    expr_delta = expr_deform(corr["coeffs"])

                verts_out = deformer(
                    mp_coeffs_corr=corr["coeffs"],
                    pose_rotation_6d=corr["pose_residual"],
                    pose_translation=corr["translation_residual"],
                    expr_delta=expr_delta,
                )

                avatar.eyes.set_gaze_from_tracker(
                    corr["gaze_uv_left"][0], corr["gaze_uv_right"][0]
                )
                avatar_out = avatar(
                    verts_out["verts_posed"][0],
                    ict.faces,
                    expression_weights=verts_out["expression_weights"],
                    expression_names=ict.expression_names,
                )
                avatar_out["mesh_xyz"] = verts_out["verts_posed"]

                render_semantic = spec.w_seg > 0 and getattr(renderer, "uses_gsplat", False)
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
                    expr_deform=expr_deform if spec.train_expression_deform else None,
                    expr_delta=expr_delta,
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
                            "expr_deform": expr_deform.state_dict(),
                            "avatar": avatar.state_dict(),
                            "cfg": cfg,
                        },
                        ckpt,
                    )
                    print(f"saved {ckpt}")

                if stage_local >= spec.steps:
                    break

        current_spec = spec

    print(f"\nDone. Total steps: {global_step}. Last stage: {current_spec.name if current_spec else 'n/