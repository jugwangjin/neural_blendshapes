import torch

from model.tracker_mlp import TrackerCorrectionMLP


def test_tracker_gamma_preserves_near_zero():
    m = TrackerCorrectionMLP(n_blendshapes=52)
    mp = torch.zeros(1, 52)
    mp[0, 3] = 1e-5
    out = m(mp)
    assert out["coeffs"].min() >= 0
    assert out["coeffs"][0, 3] < 0.01
    assert out["coeffs"].max() <= 1.0
