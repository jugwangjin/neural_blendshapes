"""
Sanity-check pose → mesh → Gaussian position / rotation chain.

Run on server (WSL):
  python debug/verify_pose_gaussian_chain.py
  python debug/verify_pose_gaussian_chain.py --yaw-deg 15
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Config
from model.gaussian_avatar import GaussianAvatar
from model.ict_deformer import ICTDeformer
from model.tracker_mlp import TrackerCorrectionMLP
from utils.mesh_ops import rotation_6d_to_matrix, rotation_matrix_to_6d


def _yaw_r6(deg: float, device):
    c = torch.cos(torch.tensor(deg * 3.14159265 / 180.0, device=device))
    s = torch.sin(torch.tensor(deg * 3.14159265 / 180.0, device=device))
    R = torch.tensor(
        [[c, 0, s], [0, 1, 0], [-s, 0, c]],
        device=device,
        dtype=torch.float32,
    )
    return rotation_matrix_to_6d(R.unsqueeze(0)).squeeze(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--yaw-deg", type=float, default=12.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)
    cfg = Config()

    from model.ict_model import ICTFaceKitTorch

    ict = ICTFaceKitTorch(npy_dir=str(cfg.ict_npy))
    tracker = TrackerCorrectionMLP(
        mediapipe_to_ict=ict.mediapipe_to_ict,
        num_ict_expression=ict.num_expression,
    ).to(device)
    deformer = ICTDeformer(ict).to(device)
    avatar = GaussianAvatar.from_ict(
        ict,
        deformer=deformer,
        n_face_gaussians=512,
        n_eye_per_side=32,
        with_mesh_scaling=cfg.gaussian_with_mesh_scaling,
    ).to(device)

    mp = torch.zeros(1, tracker.n_blendshapes, device=device)
    corr0 = tracker(mp_blendshape=mp)
    r6 = _yaw_r6(args.yaw_deg, device).unsqueeze(0)
    corr1 = dict(corr0)
    corr1["pose_residual"] = r6
    corr1["translation_residual"] = torch.zeros(1, 3, device=device)
    corr1["translation_global"] = torch.zeros(1, 3, device=device)

    def run(corr, tag):
        out = avatar(
            tracker_out=corr,
            use_pose_scale=True,
            apply_expression_deform=False,
            enable_color_pose=False,
            enable_color_expression=False,
        )
        surf = out["surface"]
        return {
            "tag": tag,
            "mesh": out["mesh_xyz"][0],
            "xyz": surf["xyz"],
            "rot": surf["rotation"],
            "pose_r6": corr["pose_residual"],
        }

    a = run(corr0, "neutral_pose")
    b = run(corr1, "yaw_pose")

    dm = (a["mesh"] - b["mesh"]).norm(dim=-1).mean().item()
    dx = (a["xyz"] - b["xyz"]).norm(dim=-1).mean().item()
    drot = (a["rot"] - b["rot"]).norm(dim=-1).mean().item()
    qdot = (a["rot"] * b["rot"]).sum(dim=-1).abs().mean().item()

    R = rotation_6d_to_matrix(a["pose_r6"])
    centroid = a["mesh"].mean(dim=0)
    rigid = (a["mesh"] - centroid) @ R[0].T + centroid
    rigid_err = (rigid - b["mesh"]).norm(dim=-1).mean().item()

    print("=== pose → Gaussian chain check ===")
    print(f"device={device} yaw_deg={args.yaw_deg} mesh_scaling={cfg.gaussian_with_mesh_scaling}")
    print(f"mesh centroid displacement (neutral→yaw): {dm:.6f}")
    print(f"Gaussian xyz mean displacement:          {dx:.6f}")
    print(f"Gaussian quat L2 mean change:              {drot:.6f}")
    print(f"|q_neutral·q_yaw| mean (1=same hemisphere): {qdot:.4f}")
    print(f"global rigid replay error on mesh:         {rigid_err:.6f}")
    print()
    print("Expected: mesh & xyz move (dm,dx > 0), quats change (drot > 0), rigid_err small if deformer R matches.")

    if dm < 1e-4:
        print("WARN: mesh barely moved — check pose_residual / deformer.apply_weighted_pose")
    if dx < 1e-4:
        print("WARN: Gaussian xyz barely moved — check mesh_pose / bary embedding")
    if rigid_err > 0.05:
        print("WARN: large rigid replay error — pose_weight blend or R convention mismatch")


if __name__ == "__main__":
    main()
