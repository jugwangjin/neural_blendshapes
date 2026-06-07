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


def test_tracker_gamma_log_symmetric_reciprocal():
    m = TrackerCorrectionMLP(n_blendshapes=52, num_ict_expression=53, gamma_max=2.0)
    assert m.gamma_log_span is not None
    lo = float(m.gamma_min)
    hi = float(m.gamma_max)
    assert abs(lo * hi - 1.0) < 1e-5
    mp = torch.zeros(1, 52)
    h = torch.zeros(1, m.expr_trunk[0].in_features)
    m.expr_trunk[0].weight.zero_()
    m.expr_trunk[0].bias.zero_()
    m.head_gamma.weight.zero_()
    m.head_gamma.bias.fill_(-10.0)
    g_lo = m(mp)["gamma"]
    m.head_gamma.bias.fill_(10.0)
    g_hi = m(mp)["gamma"]
    assert g_lo.mean() < 0.6
    assert g_hi.mean() > 1.4


def test_additive_gamma_zero_delta_matches_raw():
    m = TrackerCorrectionMLP(
        n_blendshapes=52, num_ict_expression=53, additive_gamma_correction=True
    )
    mp = torch.zeros(1, 52)
    out = m(mp, additive_gamma_correction=True)
    assert out["additive_gamma"] is True
    assert torch.allclose(out["coeffs"], out["coeffs_raw"])


def test_load_tracker_legacy_missing_global_translation():
    from training.checkpoint_io import load_tracker_state_dict

    trained = TrackerCorrectionMLP(n_blendshapes=52, num_ict_expression=53)
    trained.global_translation.data.fill_(0.5)
    legacy = {k: v for k, v in trained.state_dict().items() if k != "global_translation"}

    fresh = TrackerCorrectionMLP(n_blendshapes=52, num_ict_expression=53)
    load_tracker_state_dict(fresh, legacy)
    assert torch.allclose(fresh.global_translation, torch.zeros(3))

    load_tracker_state_dict(fresh, trained.state_dict())
    assert torch.allclose(fresh.global_translation, torch.full((3,), 0.5))
