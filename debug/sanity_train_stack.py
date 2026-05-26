"""
Quick training-stack checks (run from repo root on GPU server).

  python debug/sanity_train_stack.py --check compile
  python debug/sanity_train_stack.py --check eye
  python debug/sanity_train_stack.py --check avatar
  python debug/sanity_train_stack.py --check render
  python debug/sanity_train_stack.py --check loss
  python debug/sanity_train_stack.py --check all
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_ict(cfg, device):
    from model.ict_model import ICTFaceKitTorch

    return ICTFaceKitTorch(npy_dir=str(cfg.ict_npy)).to(device)


def _make_deformer(cfg, ict, device):
    from model.expr_regions import build_expr_region_weight
    from model.ict_deformer import ICTDeformer

    region_weight = build_expr_region_weight(ict).to(device)
    return ICTDeformer(
        ict,
        region_weight,
        mediapipe_name_to_ict=str(cfg.mediapipe_name_to_ict),
        n_coeffs=cfg.num_ict_expressions,
    ).to(device)


def _dummy_mp_landmarks(device, batch=1):
    """Neutral 478×2 UV — satisfies TrackerCorrectionMLP when use_landmarks=True."""
    import torch

    return torch.zeros(batch, 478, 2, device=device)


def _compile():
    files = [
        "config.py",
        "train.py",
        "model/gaussian_avatar.py",
        "legacy/eye_uv_slide/eye_texture_gaussians.py",
        "model/ict_deformer.py",
        "model/tracker_mlp.py",
        "training/apply.py",
        "training/stages.py",
        "losses/train_losses.py",
        "rendering/avatar_renderer.py",
    ]
    for rel in files:
        p = ROOT / rel
        subprocess.check_call([sys.executable, "-m", "py_compile", str(p)])
    print("OK: py_compile")


def _surface_eye_pin():
    from config import Config
    from model.gaussian_avatar import GaussianAvatar
    from model.ict_model import ICTFaceKitTorch
    from utils.ict_texture_maps import all_texture_materials

    cfg = Config()
    ict = ICTFaceKitTorch(npy_dir=str(cfg.ict_npy))
    assert ict.has_texture_maps()
    mats = all_texture_materials(ict)
    assert "M_EyeOcclusion" in mats and "M_Face" in mats
    if ict.vertex_count >= 26718:
        assert int(ict.n_texture_maps) == 12, f"expected 12 usemtl maps, got {ict.n_texture_maps}"
    avatar = GaussianAvatar.from_ict(ict, k_face=2, k_eyeball_sclera=4, k_eye_occlusion=4)
    assert avatar.is_h_pin.any()
    assert (avatar.h_sigma_scale[avatar.is_h_pin] == 0).all()
    assert hasattr(avatar, "face_texture_map_id")
    print(f"OK: sclera+occlusion on surface, n_pin={int(avatar.is_h_pin.sum())}, texture_maps K={ict.n_texture_maps}")


def _avatar_forward(device):
    import torch
    from config import Config
    from model.gaussian_avatar import GaussianAvatar
    from model.tracker_mlp import TrackerCorrectionMLP

    cfg = Config()
    ict = _load_ict(cfg, device)
    deformer = _make_deformer(cfg, ict, device)
    avatar = GaussianAvatar.from_ict(
        ict, deformer=deformer, k_face=4, k_eyeball_sclera=4, k_eye_occlusion=4,
        n_semantic_classes=cfg.n_semantic_classes,
    ).to(device)
    tracker = TrackerCorrectionMLP(
        n_blendshapes=cfg.num_mp_blendshapes,
        num_ict_expression=ict.num_expression,
        mediapipe_to_ict=ict.mediapipe_to_ict,
    ).to(device)
    corr = tracker(
        mp_blendshape=torch.zeros(1, cfg.num_mp_blendshapes, device=device),
        mp_landmarks_2d=_dummy_mp_landmarks(device),
        mp_pose_raw=torch.zeros(1, 6, device=device),
        force_gamma_one=True,
    )
    out = avatar(tracker_out=corr, apply_expression_deform=False)
    assert torch.isfinite(out["xyz"]).all()
    assert out["is_eyeball_surface"].any()
    assert out["mesh_xyz"] is not None
    print(f"OK: avatar forward xyz={tuple(out['xyz'].shape)}")


def _render_forward(device):
    import torch
    from config import Config
    from model.gaussian_avatar import GaussianAvatar
    from model.tracker_mlp import TrackerCorrectionMLP
    from rendering import GaussianRenderer
    from utils.camera import load_training_camera

    cfg = Config()
    cfg.image_size = 128
    ict = _load_ict(cfg, device)
    deformer = _make_deformer(cfg, ict, device)
    avatar = GaussianAvatar.from_ict(ict, deformer=deformer, k_face=4).to(device)
    renderer = GaussianRenderer(cfg, image_size=cfg.image_size).to(device)
    camera = load_training_camera(ict.canonical[0], path=cfg.camera_npz, width=cfg.image_size, height=cfg.image_size)
    corr = TrackerCorrectionMLP(
        n_blendshapes=cfg.num_mp_blendshapes,
        num_ict_expression=ict.num_expression,
        mediapipe_to_ict=ict.mediapipe_to_ict,
    ).to(device)(
        mp_blendshape=torch.zeros(1, 52, device=device),
        mp_landmarks_2d=_dummy_mp_landmarks(device),
        mp_pose_raw=torch.zeros(1, 6, device=device),
        force_gamma_one=True,
    )
    out = avatar(tracker_out=corr)
    render = renderer(out, camera, render_semantic=False)
    assert torch.isfinite(render["rgb"]).all()
    print(f"OK: render rgb={tuple(render['rgb'].shape)}")


def _loss_backward(device):
    import torch
    from torch.utils.data import DataLoader

    from config import Config
    from dataset import build_train_dataset, collate_batch, move_batch_to_device
    from losses.mediapipe_landmark_478 import build_mp_lmk_embedding
    from losses.train_losses import compute_losses
    from model.gaussian_avatar import GaussianAvatar
    from model.tracker_mlp import TrackerCorrectionMLP
    from rendering import GaussianRenderer
    from training.apply import stage_loss_cfg
    from training.stages import STAGE_SCHEDULE
    from utils.camera import load_training_camera

    cfg = Config()
    cfg.image_size = 128
    spec = STAGE_SCHEDULE[1]
    loss_cfg = stage_loss_cfg(spec)
    loss_cfg.image_size = cfg.image_size
    for k in ("w_seg", "w_gamma_prior", "w_pose_prior", "w_gaze_residual", "w_expr_deform_reg",
              "w_expr_neutral", "w_expr_leak", "w_expr_amp", "w_sem_anchor", "w_template_smooth"):
        if hasattr(loss_cfg, k):
            setattr(loss_cfg, k, 0.0)

    dataset = build_train_dataset(cfg, train=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_batch, num_workers=0)
    batch = move_batch_to_device(next(iter(loader)), device)

    ict = _load_ict(cfg, device)
    deformer = _make_deformer(cfg, ict, device)
    avatar = GaussianAvatar.from_ict(ict, deformer=deformer, k_face=4).to(device)
    tracker = TrackerCorrectionMLP(
        n_blendshapes=cfg.num_mp_blendshapes,
        num_ict_expression=ict.num_expression,
        mediapipe_to_ict=ict.mediapipe_to_ict,
    ).to(device)
    renderer = GaussianRenderer(cfg, image_size=cfg.image_size).to(device)
    camera = load_training_camera(ict.canonical[0], path=cfg.camera_npz, width=cfg.image_size, height=cfg.image_size, device=device)
    mp_lmk_emb = build_mp_lmk_embedding(cfg.mp_embedding, device)

    corr = tracker(
        mp_blendshape=batch["mp_blendshape"],
        mp_landmarks_2d=batch.get("mp_landmarks_2d"),
        mp_pose_raw=batch.get("mp_pose_raw"),
        force_gamma_one=True,
    )
    avatar_out = avatar(tracker_out=corr, apply_expression_deform=False)
    render = renderer(avatar_out, camera, render_semantic=spec.w_seg > 0)
    losses = compute_losses(
        loss_cfg, batch, render, avatar_out, camera, mp_lmk_emb, ict.faces.to(device),
        corr=corr, deformer=deformer, avatar=avatar, renderer=renderer,
    )
    losses["total"].backward()
    keys = [k for k in losses if k != "total"]
    print("OK: loss backward", {k: float(losses[k].detach().cpu()) for k in keys})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", choices=("compile", "eye", "avatar", "render", "loss", "all"), default="all")
    args = p.parse_args()
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    if args.check in ("compile", "all"):
        _compile()
    if args.check in ("eye", "all"):
        _surface_eye_pin()
    if args.check in ("avatar", "all"):
        _avatar_forward(device)
    if args.check in ("render", "all"):
        _render_forward(device)
    if args.check in ("loss", "all"):
        _loss_backward(device)


if __name__ == "__main__":
    main()
