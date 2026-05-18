"""
UVH + per-eye texture-space Gaussians (h=0 on eyes).
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import Config
from losses import loss_h_anchor_surface, loss_iris_landmarks_2d, soft_uv_box_barrier
from model.gaussian_avatar import GaussianAvatar
from model.ict_model import ICTFaceKitTorch
from utils.camera import FixedCamera


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ict = ICTFaceKitTorch(
        npy_dir=str(cfg.ict_npy),
        canonical=str(cfg.ict_canonical),
    ).to(device)

    avatar = GaussianAvatar.from_ict(
        ict,
        n_face_gaussians=4096,
        n_eye_per_side=cfg.n_eye_gaussians_per_side,
        gaze_uv_range=cfg.gaze_uv_range,
        learn_gaze_refine=cfg.learn_gaze_refine,
    ).to(device)

    camera = FixedCamera.from_default_npz(cfg.default_camera, width=cfg.image_size)

    exp = torch.zeros(1, ict.num_expression, device=device)
    exp[0, ict.expression_names.tolist().index("eyeLookIn_L")] = 0.3
    exp[0, ict.expression_names.tolist().index("eyeLookOut_R")] = 0.2

    verts = ict.forward(expression_weights=exp, to_canonical=False)
    out = avatar(
        verts[0],
        ict.faces,
        expression_weights=exp,
        expression_names=ict.expression_names.tolist(),
    )

    n_face = out["face"]["xyz"].shape[0]
    eyeball_mask = torch.zeros(out["h"].shape[0], dtype=torch.bool, device=device)
    eyeball_mask[n_face:] = True

    loss_h = loss_h_anchor_surface(
        out["h"], out["is_anchor_surface"], eyeball_mask=eyeball_mask
    )

    dummy_mp = torch.rand(478, 2, device=device)
    loss_iris = loss_iris_landmarks_2d(
        out["iris_control_xyz"], dummy_mp, camera, image_size=cfg.image_size
    )
    loss_barrier = soft_uv_box_barrier(out["eyes"]["left"]["uv"]) + soft_uv_box_barrier(
        out["eyes"]["right"]["uv"]
    )

    print("texture spaces:")
    tm = out["texture_meshes"]
    print(f"  face tris: {tm.face_face_idx.numel()}")
    print(f"  left_eye tris: {tm.left_eye_face_idx.numel()}")
    print(f"  right_eye tris: {tm.right_eye_face_idx.numel()}")
    print(f"gaze_uv L={out['gaze_uv_left']}  R={out['gaze_uv_right']}")
    print(f"eye h max={out['h'][n_face:].abs().max().item():.6f} (expect 0)")
    print(f"loss_h={loss_h.item():.6f}  loss_iris={loss_iris.item():.6f}")


if __name__ == "__main__":
    main()
