"""color_expression region gate (mouth interior + eye)."""

import torch

from model.color_expression_region_gate import (
    COLOR_EXPRESSION_EXCLUDED_REGION_CODES,
    color_expression_region_allow,
)


def test_excluded_codes():
    assert COLOR_EXPRESSION_EXCLUDED_REGION_CODES == frozenset({0, 1, 2, 5, 6, 7})


def test_gate_masks_mouth_and_eye_not_face():
    codes = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long)
    gate = color_expression_region_allow(codes, enabled=True)
    assert gate.tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]


def test_gate_disabled_returns_none():
    codes = torch.tensor([4], dtype=torch.long)
    assert color_expression_region_allow(codes, enabled=False) is None


def test_avatar_color_expression_support_gated():
    from config import Config
    from model.ict_model import ICTFaceKitTorch
    from model.gaussian_avatar import GaussianAvatar

    cfg = Config()
    npy = cfg.ict_npy
    if not npy.is_file():
        return

    ict = ICTFaceKitTorch(npy_dir=str(npy))
    av = GaussianAvatar.from_ict(
        ict,
        k_face=2,
        k_head=2,
        k_mouth_interior=2,
        k_mouth_socket=2,
        k_teeth=1,
        k_eye_socket=1,
        k_eyeball_sclera=1,
        k_eye_occlusion=1,
        color_expression_exclude_mouth_eye=True,
    )
    gate = av._color_expression_region_allow()
    assert gate is not None
    assert (gate[av.face_region_code == 4] == 1.0).all()
    for code in COLOR_EXPRESSION_EXCLUDED_REGION_CODES:
        on = av.face_region_code == code
        if on.any():
            assert (gate[on] == 0.0).all()
            assert (av.color_expression[on].abs().max().item() == 0.0)

    support = av._color_expression_support()
    raw = av._color_expression_support_raw()
    assert (support[gate == 0] == 0).all()
    assert torch.equal(raw, av.face_expression_support[av.face_idx.long()])

    av.color_expression.data.fill_(1.0)
    av._apply_color_expression_region_exclusion()
    assert (av.color_expression[gate == 0].abs().max().item() == 0.0)

    exp = torch.zeros(ict.num_expression, device=ict.neutral_mesh.device)
    exp[ict.expression_names.tolist().index("jawOpen")] = 1.0
    av.color_expression.data[gate == 1] = 1.0
    out = av._forward_surface(
        ict.neutral_mesh[0],
        ict.faces,
        expr_coeff=exp,
        enable_color_expression=True,
    )
    color = out["color"]
    if color.ndim == 3:
        color = color[:, 0, :]
    delta = color - av.color
    assert (delta[gate == 0].abs().max().item() == 0.0)
