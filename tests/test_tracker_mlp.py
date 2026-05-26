import torch

from config import Config
from model.ict_model import ICTFaceKitTorch
from model.tracker_mlp import TrackerCorrectionMLP


def test_tracker_gamma_preserves_near_zero():
    cfg = Config()
    ict = ICTFaceKitTorch(npy_dir=str(cfg.ict_npy))
    m = TrackerCorrectionMLP(
        n_blendshapes=52,
        mediapipe_to_ict=ict.mediapipe_to_ict,
        num_ict_expression=ict.num_expression,
    )
    mp = torch.zeros(1, 52)
    col = int(ict.mediapipe_to_ict[3].item())
    mp[0, col] = 1e-5
    out = m(mp)
    assert out["coeffs"].min() >= 0
    assert out["coeffs"][0, 3] < 0.01
    assert out["coeffs"].max() <= 1.0
