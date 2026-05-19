"""
Quick training-stack checks (run from repo root on GPU server).

  python scripts/sanity_train_stack.py --check compile
  python scripts/sanity_train_stack.py --check eye
  python scripts/sanity_train_stack.py --check avatar
  python scripts/sanity_train_stack.py --check render
  python scripts/sanity_train_stack.py --check loss
  python scripts/sanity_train_stack.py --check all
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_ict(cfg, device):
    """Same constructor as ``train.py`` (identity is inside ``ict_facekit_torch.npy``)."""
    from model.ict_model import ICTFaceKitTorch

    return ICTFaceKitTorch(npy_dir=str(cfg.ict_npy)).to(device)


def _compile():
    files = [
        "config.py",
        "train.py",
        "model/gaussian_avatar.py",
        "model/eye_texture_gaussians.py",
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


def _eye_params():
    from model.eye_texture_gaussians import EyeTextureGaussians

    m = EyeTextureGaussians(n_per_eye=16)
    for name, p in m.named_parameters():
        print(f"  param {name} {tuple(p.shape)} grad={p.requires_grad}")
    for name, b in m.named_buffers():
        if "uv" in name or name == "h":
            print(f"  buffer {name} {tuple(b.shape)}")
    bad = [n for n, _ in m.named_parameters() if n.endswith(".uv") or n.endswith(".h")]
    assert not bad, bad
    assert not hasattr(m, "left") and not hasattr(m, "right")
    assert m.gaze_refine_left is not None
    print("OK: shared eye bank, uv/h buffers only")


def _avatar_forward(device):
    import torch
    from config import Config
    from model.gaussian_avatar import GaussianAvatar
    from model.ict_deformer import ICTDeformer
    from model.tracker_mlp import TrackerCorrectionMLP

    cfg = Config()
    ict = _load_ict(cfg, device)
    deformer = ICTDeformer(ict, n_coeffs=cfg.num_mp_blendshapes).to(device)
    avatar = GaussianAvatar.from_ict(
        ict,
        deformer=deformer,
        n_eye_per_side=16,
        gaze_uv_range=cfg.gaze_uv_range,
        learn_gaze_refine=cfg.learn_gaze_refine,
        n_semantic_classes=cfg.n_semantic_classes,
    ).to(device)
    tracker = TrackerCorrectionMLP(
        n_blendshapes=cfg.num_mp_blendshapes,
        gaze_uv_range=cfg.gaze_uv_range,
    ).to(device)

    corr = tracker(
        mp_blendshape=torch.zeros(1, cfg.num_mp_blendshapes, device=device),
        force_gamma_one=True,
    )
    out = avatar(tracker_out=corr, apply_expression_deform=False)
    assert torch.isfinite(out["xyz"]).all()
    assert out["eyes"]["left"]["uv"].shape[0] == 16
    assert out["mesh_xyz"] is not None
    print(f"OK: avatar forward xyz={tuple(out['xyz'].shape)}")


def _render_forward(device):
    import torch
    from config import Config
    from model.gaussian_avatar import GaussianAvatar
    from model.ict_deformer import ICTDeformer
    from model.tracker_mlp import TrackerCorrectionMLP
    from rendering import GaussianRenderer
    from utils.camera import FixedCamera

    cfg = Config()
    cfg.image_size = 128
    ict = _load_ict(cfg, device)
    deformer = ICTDeformer(ict, n_coeffs=cfg.num_mp_blendshapes).to(device)
    avatar = GaussianAvatar.from_ict(ict, deformer=deformer, n_eye_per_side=16).to(device)
    renderer = GaussianRenderer(cfg, image_size=cfg.image_size).to(device)
    camera = FixedCamera.from_default_npz(cfg.camera_npz, width=cfg.image_size, height=cfg.image_size)
    corr = TrackerCorrectionMLP(n_blendshapes=cfg.num_mp_blendshapes).to(device)(
        mp_blendshape=torch.zeros(1, 52, device=device),
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
    from dataset.video_dataset import VideoDataset, collate_batch
    from losses.train_losses import compute_losses
    from model.gaussian_avatar import GaussianAvatar
    from model.ict_deformer import ICTDeformer
    from model.tracker_mlp import TrackerCorrectionMLP
    from rendering import GaussianRenderer
    from training.apply import stage_loss_cfg
    from training.stages import STAGE_SCHEDULE
    from train import load_mp_embedding
    from utils.camera import FixedCamera

    cfg = Config()
    cfg.image_size = 128
    cfg.n_eye_gaussians_per_side = 16
    spec = STAGE_SCHEDULE[1]
    loss_cfg = stage_loss_cfg(spec)
    assert getattr(loss_cfg, "w_silhouette", 0) > 0 or getattr(loss_cfg, "w_mask", 0) > 0
    for k in (
        "w_seg",
        "w_gamma_prior",
        "w_pose_prior",
        "w_gaze_residual",
        "w_expr_deform_reg",
        "w_expr_neutral",
        "w_expr_leak",
        "w_expr_amp",
        "w_sem_anchor",
        "w_template_smooth",
    ):
        if hasattr(loss_cfg, k):
            setattr(loss_cfg, k, 0.0)

    dataset = VideoDataset(cfg, train=True, au_active_boost=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_batch, num_workers=0)
    batch = next(iter(loader))
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

    ict = _load_ict(cfg, device)
    deformer = ICTDeformer(ict, n_coeffs=cfg.num_mp_blendshapes).to(device)
    avatar = GaussianAvatar.from_ict(ict, deformer=deformer, n_eye_per_side=16).to(device)
    tracker = TrackerCorrectionMLP(n_blendshapes=cfg.num_mp_blendshapes, gaze_uv_range=cfg.gaze_uv_range).to(device)
    renderer = GaussianRenderer(cfg, image_size=cfg.image_size).to(device)
    camera = FixedCamera.from_default_npz(cfg.camera_npz, width=cfg.image_size, height=cfg.image_size)
    mp_emb = load_mp_embedding(cfg.mp_embedding, device)

    corr = tracker(
        mp_blendshape=batch["mp_blendshape"],
        mp_landmarks_2d=batch.get("mp_landmarks_2d"),
        mp_pose_raw=batch.get("mp_pose_raw"),
        force_gamma_one=True,
    )
    avatar_out = avatar(tracker_out=corr, apply_expression_deform=False)
    render = renderer(avatar_out, camera, render_semantic=spec.w_seg > 0)
    losses = compute_losses(
        loss_cfg,
        batch,
        render,
        avatar_out,
        camera,
        mp_emb,
        ict.faces,
        corr=corr,
        deformer=deformer,
        avatar=avatar,
    )
    losses["total"].backward()
    keys = [k for k in losses if k != "total"]
    print("OK: loss backward", {k: float(losses[k].detach().cpu()) for k in keys})


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--check",
        choices=("compile", "eye", "avatar", "render", "loss", "all"),
        default="all",
    )
    args = p.parse_args()
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    if args.check in ("compile", "all"):
        _compile()
    if args.check in ("eye", "all"):
        _eye_params()
    if args.check in ("avatar", "all"):
        _avatar_forward(device)
    if args.check in ("render", "all"):
        _render_forward(device)
    if args.check in ("loss", "all"):
        _loss_backward(device)


if __name__ == "__main__":
    main()
