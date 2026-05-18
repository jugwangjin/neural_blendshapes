"""One training step smoke test (synthetic data)."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Config


def test_train_step_no_nan():
    cfg = Config()
    cfg.iterations = 1
    cfg.batch_size = 1
    device = torch.device("cpu")

    from model.tracker_mlp import TrackerCorrectionMLP
    from model.ict_model import ICTFaceKitTorch
    from model.ict_deformer import ICTDeformer
    from model.gaussian_avatar import GaussianAvatar
    from gaussian_splatting.renderer import GaussianRenderer
    from utils.camera import FixedCamera
    from utils.so3 import rotation_6d_to_matrix
    from dataset.mediapipe_cache import default_frame_dict

    ict = ICTFaceKitTorch(npy_dir=str(cfg.ict_npy), canonical=str(cfg.ict_canonical))
    tracker = TrackerCorrectionMLP()
    deformer = ICTDeformer(ict)
    avatar = GaussianAvatar.from_ict(ict, n_face_gaussians=32, n_eye_per_side=8)
    renderer = GaussianRenderer(cfg, image_size=64)
    camera = FixedCamera(width=64, height=64)

    batch = default_frame_dict(device, 64)
    batch["mp_blendshape"] = batch["mp_blendshape"].unsqueeze(0)
    batch["mp_landmarks_2d"] = batch["mp_landmarks_2d"].unsqueeze(0)
    batch["mp_pose_raw"] = batch["mp_pose_raw"].unsqueeze(0)
    batch["image"] = batch["image"].unsqueeze(0)
    batch["mask"] = batch["mask"].unsqueeze(0)

    corr = tracker(batch["mp_blendshape"])
    verts = deformer(
        mp_coeffs_corr=corr["coeffs"],
        pose_rotation_6d=corr["pose_residual"],
        pose_translation=corr["translation_residual"],
    )
    out = avatar(verts["verts_posed"][0], ict.faces)
    render = renderer(out, camera)
    loss = render["rgb"].mean()
    loss.backward()
    assert torch.isfinite(loss)
