"""
Check gradient flow of stage-1 mesh silhouette loss.

Run on server GPU:
  python debug/check_mesh_silhouette_grad.py
  python debug/check_mesh_silhouette_grad.py --image-size 256 --batch-index 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Config
from dataset import build_train_dataset, collate_batch, move_batch_to_device
from losses.train_losses import compute_losses
from model.expr_regions import build_expr_region_weight
from model.gaussian_avatar import GaussianAvatar
from model.ict_deformer import ICTDeformer
from model.ict_model import ICTFaceKitTorch
from model.tracker_mlp import TrackerCorrectionMLP
from training.apply import stage_loss_cfg
from training.stages import STAGE_SCHEDULE
from utils.camera import load_training_camera


def grad_norm(module: torch.nn.Module) -> float:
    vals = []
    for p in module.parameters():
        if p.grad is not None:
            vals.append(float(p.grad.detach().norm().item()))
    return float(sum(vals)) if vals else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-index", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    cfg = Config()
    cfg.image_size = int(args.image_size)
    cfg.batch_size = 1
    cfg.num_workers = 0

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    dataset = build_train_dataset(cfg, train=True)
    if len(dataset) == 0:
        raise RuntimeError("empty train dataset")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_batch, num_workers=0)
    batch = None
    for i, b in enumerate(loader):
        if i == args.batch_index:
            batch = b
            break
    if batch is None:
        raise RuntimeError(f"batch-index {args.batch_index} out of range")
    batch = move_batch_to_device(batch, device)

    ict = ICTFaceKitTorch(npy_dir=str(cfg.ict_npy)).to(device)
    region_weight = build_expr_region_weight(ict).to(device)
    deformer = ICTDeformer(
        ict,
        region_weight,
        mediapipe_name_to_ict=str(cfg.mediapipe_name_to_ict),
        n_coeffs=cfg.num_ict_expressions,
    ).to(device)
    tracker = TrackerCorrectionMLP(
        mediapipe_to_ict=ict.mediapipe_to_ict,
        num_ict_expression=ict.num_expression,
        n_blendshapes=cfg.num_mp_blendshapes,
        gamma_min=cfg.gamma_min,
        gamma_max=cfg.gamma_max,
        use_landmarks=True,
    ).to(device)
    avatar = GaussianAvatar.from_ict(
        ict,
        deformer=deformer,
        k_face=cfg.n_surface_gaussians_per_face,
        k_head=cfg.n_surface_gaussians_per_head,
        k_mouth_socket=cfg.n_surface_gaussians_per_mouth_socket,
        k_mouth_interior=cfg.n_surface_gaussians_mouth_interior,
        k_teeth=cfg.n_surface_gaussians_per_teeth,
        k_eye_socket=cfg.n_surface_gaussians_per_eye_socket,
        k_eyeball_sclera=cfg.n_surface_gaussians_per_eyeball_sclera,
        k_eye_occlusion=cfg.n_surface_gaussians_per_eye_occlusion,
        n_semantic_classes=cfg.n_semantic_classes,
        gaussian_scale_knn_k=cfg.gaussian_scale_knn_k,
        gaussian_scale_knn_factor=cfg.gaussian_scale_knn_factor,
        face_center_init=cfg.gaussian_face_center_init,
        max_scale=cfg.geometry_max_scale,
    ).to(device)
    camera = load_training_camera(
        ict.expression_reference_verts(),
        path=cfg.camera_npz,
        width=cfg.image_size,
        height=cfg.image_size,
        device=device,
    )
    ict_faces = ict.faces.to(device)

    spec = next(s for s in STAGE_SCHEDULE if s.name == "1_bootstrap_pose")
    loss_cfg = stage_loss_cfg(spec)
    loss_cfg.image_size = cfg.image_size
    # Isolate mesh silhouette gradient path.
    loss_cfg.w_mp_lmk = 0.0
    loss_cfg.w_pie68_jaw = 0.0
    loss_cfg.w_silhouette = 0.0
    loss_cfg.w_rgb = 0.0
    loss_cfg.w_mesh_silhouette = max(float(getattr(loss_cfg, "w_mesh_silhouette", 0.0)), 1.0)

    tracker.zero_grad(set_to_none=True)
    deformer.zero_grad(set_to_none=True)
    avatar.zero_grad(set_to_none=True)

    corr = tracker(
        mp_blendshape=batch["mp_blendshape"],
        mp_landmarks_2d=batch.get("mp_landmarks_2d"),
        mp_landmarks_3d=batch.get("mp_landmarks_3d"),
        world_to_cam=batch.get("world_to_cam"),
        mp_pose_raw=batch.get("mp_pose_raw"),
        mp_transform_matrix=batch.get("mp_transform_matrix"),
        force_gamma_one=True,
    )
    avatar_out = avatar(
        tracker_out=corr,
        apply_expression_deform=False,
        pose_weight_fixed=1.0,
        skip_surface=False,
    )
    avatar_out["mesh_xyz"].retain_grad()

    losses = compute_losses(
        loss_cfg,
        batch,
        render=None,
        avatar_out=avatar_out,
        camera=camera,
        mp_lmk_emb=None,
        ict_faces=ict_faces,
        pie68_jaw_vertex_idx=None,
        corr=corr,
        deformer=deformer,
        expr_delta=None,
        avatar=avatar,
        renderer=None,
    )
    if "mesh_silhouette" not in losses:
        raise RuntimeError("mesh_silhouette term missing from losses")
    losses["total"].backward()

    mesh_grad = float(avatar_out["mesh_xyz"].grad.detach().norm().item())
    tpl_grad = grad_norm(deformer.template_mlp)
    tracker_grad = grad_norm(tracker.pose_trunk) + grad_norm(tracker.head_pose)
    print(
        f"mesh_sil={float(losses['mesh_silhouette'].item()):.6f} "
        f"mesh_xyz_grad={mesh_grad:.6f} "
        f"template_mlp_grad={tpl_grad:.6f} "
        f"tracker_pose_grad={tracker_grad:.6f}"
    )
    if mesh_grad <= 0.0 or tpl_grad <= 0.0:
        raise RuntimeError(
            "mesh silhouette grad path inactive: "
            f"mesh_xyz_grad={mesh_grad:.6f}, template_mlp_grad={tpl_grad:.6f}"
        )
    print("OK: mesh silhouette gradient path is active.")


if __name__ == "__main__":
    main()

