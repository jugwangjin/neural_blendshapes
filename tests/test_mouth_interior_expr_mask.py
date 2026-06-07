"""Mouth interior jaw-only expression masking (ICT linear + expr_mlp)."""

import torch

from config import Config
from model.ict_model import ICTFaceKitTorch
from model.ict_deformer import ICTDeformer
from model.mouth_interior_expr_mask import (
    JAW_ICT_EXPRESSION_NAMES,
    JAW_MP_CHANNEL_NAMES,
    collect_mouth_interior_vertex_indices,
    jaw_ict_expression_indices,
    jaw_mp_channel_indices,
    pick_non_jaw_ict_expression_index,
)
from utils.mediapipe_blendshapes import ICT_GATHER_MP_NAMES


def test_jaw_channel_names():
    for name in JAW_ICT_EXPRESSION_NAMES:
        assert name in JAW_MP_CHANNEL_NAMES
    mp_jaw = {ICT_GATHER_MP_NAMES[j] for j in jaw_mp_channel_indices()}
    assert mp_jaw == JAW_MP_CHANNEL_NAMES


def test_mouth_interior_mask_blocks_smile_not_jaw():
    cfg = Config()
    npy = cfg.ict_npy
    if not npy.is_file():
        return

    ict = ICTFaceKitTorch(npy_dir=str(npy), mouth_interior_jaw_only_expression=True)
    deformer = ICTDeformer(ict, mouth_interior_jaw_only_expression=True)

    interior = collect_mouth_interior_vertex_indices(ict)
    assert len(interior) > 0
    assert hasattr(ict, "ict_expression_vertex_allow_mask")
    assert ict.ict_expression_vertex_allow_mask.shape == (
        ict.num_expression,
        ict.neutral_mesh.shape[1],
    )

    names = ict.expression_names.tolist()
    jaw_e = set(jaw_ict_expression_indices(ict.expression_names))
    smile_e = pick_non_jaw_ict_expression_index(ict.expression_names)
    assert smile_e not in jaw_e

    interior_t = torch.tensor(interior, dtype=torch.long)
    v_smile = int(interior[0])
    assert ict.ict_expression_vertex_allow_mask[smile_e, v_smile].item() == 0.0
    assert ict.ict_expression_vertex_allow_mask[jaw_e.pop(), v_smile].item() == 1.0

    j_smile = ICT_GATHER_MP_NAMES.index("mouthSmileLeft")
    j_jaw = ICT_GATHER_MP_NAMES.index("jawOpen")
    assert deformer.expr_gate[j_smile, v_smile].item() == 0.0
    assert deformer.expr_gate[j_jaw, v_smile].item() > 0.0

    exp = torch.zeros(1, ict.num_expression)
    exp[0, smile_e] = 1.0
    mesh_smile = ict.forward(expression_weights=exp, apply_flame_similarity=False)
    exp_jaw = torch.zeros(1, ict.num_expression)
    exp_jaw[0, names.index("jawOpen")] = 1.0
    mesh_jaw = ict.forward(expression_weights=exp_jaw, apply_flame_similarity=False)
    neutral = ict.neutral_mesh

    d_smile = (mesh_smile[0, interior_t] - neutral[0, interior_t]).norm(dim=-1).max()
    d_jaw = (mesh_jaw[0, interior_t] - neutral[0, interior_t]).norm(dim=-1).max()
    assert d_smile.item() == 0.0
    assert d_jaw.item() > 0.0

    skin_v = int(ict.skin_face_indices[0])
    d_smile_skin = (mesh_smile[0, skin_v] - neutral[0, skin_v]).norm()
    assert d_smile_skin.item() > 0.0


def test_mask_disabled():
    cfg = Config()
    npy = cfg.ict_npy
    if not npy.is_file():
        return

    ict = ICTFaceKitTorch(npy_dir=str(npy), mouth_interior_jaw_only_expression=False)
    assert not hasattr(ict, "ict_expression_vertex_allow_mask")

    interior = collect_mouth_interior_vertex_indices(ict)
    smile_e = pick_non_jaw_ict_expression_index(ict.expression_names)
    exp = torch.zeros(1, ict.num_expression)
    exp[0, smile_e] = 1.0
    mesh = ict.forward(expression_weights=exp, apply_flame_similarity=False)
    interior_t = torch.tensor(interior, dtype=torch.long)
    d = (mesh[0, interior_t] - ict.neutral_mesh[0, interior_t]).norm(dim=-1).max()
    assert d.item() > 0.0
