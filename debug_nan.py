"""
Minimal NaN isolator for coarse-mesh raster losses.

Goal: identify which raster-derived loss first produces NaN gradients.

We run ONE forward/render/loss pass with the stage-2(=coarse mesh) settings and then
probe per-loss gradients. We then repeat with:
  1) w_h = 0
  2) if still NaN: w_rgb = 0
  3) if still NaN: w_silhouette = 0

We also force gsplat antialiasing OFF by setting:
  cfg.gsplat_rasterize_mode = "classic"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["LANG"] = "C.UTF-8"
os.environ["LC_ALL"] = "C.UTF-8"

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import Config
from dataset import build_train_dataset, collate_batch, move_batch_to_device
from losses.mediapipe_landmark_478 import build_mp_lmk_embedding
from losses.pie68_jaw_landmark import build_pie68_jaw_vertex_indices
from losses.train_losses import compute_losses
from model.expr_regions import build_expr_region_weight
from model.gaussian_avatar import GaussianAvatar
from model.ict_deformer import ICTDeformer
from model.ict_model import ICTFaceKitTorch
from model.tracker_mlp import TrackerCorrectionMLP
from rendering import GaussianRenderer
from training.apply import apply_stage_requires_grad, init_training_state, stage_loss_cfg
from training.loss_debug import format_loss_report, probe_raster_loss_grads
from training.stages import STAGE_SCHEDULE
from utils.camera import load_training_camera


def _parse_split_cli(values):
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


def _tensor_is_finite(t):
    return (t is None) or (not isinstance(t, torch.Tensor)) or torch.isfinite(t).all().item()


def _resolve_existing_input_dir(input_dir) -> Path:
    p = Path(input_dir)
    curr = p
    while curr != curr.parent:
        if curr.is_dir():
            return curr
        curr = curr.parent
    return p


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--train-split", nargs="*", default=None)
    p.add_argument("--eval-split", nargs="*", default=None)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    cfg = Config()
    cfg.input_dir = _resolve_existing_input_dir(args.input_dir)
    if args.train_split is not None:
        cfg.train_split = _parse_split_cli(args.train_split)
    if args.eval_split is not None:
        cfg.eval_split = _parse_split_cli(args.eval_split)
    cfg.batch_size = 1

    # Force antialiasing OFF.
    cfg.gsplat_rasterize_mode = "classic"
    # Force packed OFF (suspected NaN grads in gsplat packed backward).
    cfg.gsplat_packed = False

    ds = build_train_dataset(cfg, train=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_batch, num_workers=0)
    batch = next(iter(loader))
    batch = move_batch_to_device(batch, device)

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
    init_training_state(avatar)

    renderer = GaussianRenderer(cfg, image_size=cfg.image_size, sh_degree=None).to(device)
    camera = load_training_camera(
        ict.canonical[0], path=cfg.camera_npz, width=cfg.image_size, height=cfg.image_size, device=device
    )

    mp_lmk_emb = build_mp_lmk_embedding(cfg.mp_embedding, device)
    pie68_jaw_vertex_idx = build_pie68_jaw_vertex_indices(ict, device)
    ict_faces = ict.faces.to(device)

    # Pick the coarse-mesh spec (name contains "coarse_mesh").
    coarse_spec = None
    for s in STAGE_SCHEDULE:
        if "coarse_mesh" in s.name:
            coarse_spec = s
            break
    if coarse_spec is None:
        raise ValueError("could not find coarse_mesh stage in STAGE_SCHEDULE")

    apply_stage_requires_grad(coarse_spec, tracker, deformer, avatar)

    def run_case(tag: str, *, w_h=None, w_rgb=None, w_silhouette=None, gsplat_packed=None, gsplat_rasterize_mode=None):
        if gsplat_packed is not None:
            cfg.gsplat_packed = bool(gsplat_packed)
            renderer.packed = bool(gsplat_packed)
        if gsplat_rasterize_mode is not None:
            cfg.gsplat_rasterize_mode = str(gsplat_rasterize_mode)
            renderer.rasterize_mode = str(gsplat_rasterize_mode)
        loss_cfg = stage_loss_cfg(coarse_spec)
        loss_cfg.image_size = cfg.image_size
        loss_cfg.mp_lmk_iris_weight = cfg.mp_lmk_iris_weight
        loss_cfg.silhouette_use_edt = cfg.silhouette_use_edt
        loss_cfg.silhouette_edt_w_ext = cfg.silhouette_edt_w_ext
        loss_cfg.silhouette_edt_w_int = cfg.silhouette_edt_w_int
        loss_cfg.silhouette_edt_max_dist_px = cfg.silhouette_edt_max_dist_px

        if w_h is not None:
            loss_cfg.w_h = float(w_h)
        if w_rgb is not None:
            loss_cfg.w_rgb = float(w_rgb)
        if w_silhouette is not None:
            loss_cfg.w_silhouette = float(w_silhouette)
            loss_cfg.w_mask = float(w_silhouette)
            loss_cfg.w_mp_mask = float(w_silhouette)

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
            force_gamma_one=coarse_spec.fix_gamma_at_one,
        )
        pose_weight_fixed = 1.0 if coarse_spec.pose_weight_one else None
        avatar_out = avatar(
            tracker_out=corr,
            apply_expression_deform=coarse_spec.train_expression_deform,
            use_pose_scale=coarse_spec.apply_pose_scale,
            pose_weight_fixed=pose_weight_fixed,
            rotate_about_centroid=coarse_spec.pose_rotate_about_centroid,
            pose_zero_tz=coarse_spec.pose_zero_tz,
            skip_surface=False,
            enable_color_pose=getattr(coarse_spec, "train_color_pose", False),
            enable_color_expression=getattr(coarse_spec, "train_color_expression", False),
        )
        render = renderer(avatar_out, camera, render_semantic=False)
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
            expr_delta=avatar_out.get("expr_delta"),
            avatar=avatar,
            renderer=renderer,
        )

        # Basic sanity: forward tensors should be finite.
        ok_forward = _tensor_is_finite(losses.get("total")) and _tensor_is_finite(avatar_out.get("mesh_xyz"))

        bad_losses = probe_raster_loss_grads(losses, loss_cfg, tracker, deformer, avatar)
        print(
            f"[case {tag}] gsplat_rasterize_mode={cfg.gsplat_rasterize_mode} "
            f"gsplat_packed={cfg.gsplat_packed} "
            f"w_h={loss_cfg.w_h} w_rgb={loss_cfg.w_rgb} w_sil={loss_cfg.w_silhouette} "
            f"forward_ok={ok_forward} bad_loss_grads={bad_losses}"
        )
        if not ok_forward:
            print(format_loss_report(losses, loss_cfg))
        return bad_losses

    # Immediate isolation sequence (packed=False, w_h=0 — original debug path).
    bad = run_case("A_w_h0", w_h=0.0)
    if bad:
        bad = run_case("B_w_h0_w_rgb0", w_h=0.0, w_rgb=0.0)
    if bad:
        bad = run_case("C_w_h0_w_rgb0_w_sil0", w_h=0.0, w_rgb=0.0, w_silhouette=0.0)

    # Match train.py defaults: full coarse loss weights + gsplat_packed=True.
    run_case("D_train_match_packed", gsplat_packed=True)
    run_case("E_train_match_packed_w_h0", w_h=0.0, gsplat_packed=True)

    # Antialiased, packed=False
    run_case("AA_aa_w_h0", w_h=0.0, gsplat_packed=False, gsplat_rasterize_mode="antialiased")
    run_case("BB_aa_w_h0_w_rgb0", w_h=0.0, w_rgb=0.0, gsplat_packed=False, gsplat_rasterize_mode="antialiased")
    run_case("CC_aa_w_h0_w_rgb0_w_sil0", w_h=0.0, w_rgb=0.0, w_silhouette=0.0, gsplat_packed=False, gsplat_rasterize_mode="antialiased")
    
    # Antialiased, packed=True
    run_case("DD_aa_train_match_packed", gsplat_packed=True, gsplat_rasterize_mode="antialiased")
    run_case("EE_aa_train_match_packed_w_h0", w_h=0.0, gsplat_packed=True, gsplat_rasterize_mode="antialiased")

    print("done")


if __name__ == "__main__":
    main()
