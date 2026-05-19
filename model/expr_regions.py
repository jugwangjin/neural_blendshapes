"""Per-vertex weights for ICT deformer (expression gate + deform regularization)."""

import torch

EXPR_WEIGHT_BY_PART_ID = {
    0: 1.0,   # face skin
    1: 0.2,   # head/neck
    2: 1.0,   # mouth socket
    3: 1.0,   # eye socket L (same MLP field; socket reg limits delta)
    4: 1.0,   # eye socket R
    5: 1.0,   # gums/tongue (surface Gaussians substitute teeth)
    6: 0.0,   # teeth — no Gaussians, no deformer
    7: 1.0,   # eyeball L
    8: 1.0,   # eyeball R
}

# L2 penalty weight on template / expression deltas (not a forward gate).
DEFORM_REG_BY_PART_ID = {
    0: 0.05,
    1: 0.08,
    2: 0.05,
    3: 1.0,   # eye socket — penalize large deltas (orbit behind eye)
    4: 1.0,
    5: 0.01,  # gums — allow template/expr field to bulge over teeth volume
    6: 2.0,   # teeth (masked in forward; unused)
    7: 0.02,  # eyeball — light reg, field moves with lids
    8: 0.02,
}


def build_expr_region_weight(ict_facekit) -> torch.Tensor:
    """[V] soft gate for support-gated expression (multiply blendshape support)."""
    n_verts = ict_facekit.neutral_mesh.shape[1]
    w = torch.zeros(n_verts, dtype=torch.float32)
    parts = ict_facekit.vertex_parts
    for i, pid in enumerate(parts):
        w[i] = EXPR_WEIGHT_BY_PART_ID.get(int(pid), 0.0)

    if hasattr(ict_facekit, "eyeball_indices"):
        w[ict_facekit.eyeball_indices] = 1.0
    if hasattr(ict_facekit, "eye_socket_left_indices"):
        w[ict_facekit.eye_socket_left_indices] = 1.0
    if hasattr(ict_facekit, "eye_socket_right_indices"):
        w[ict_facekit.eye_socket_right_indices] = 1.0
    if hasattr(ict_facekit, "not_face_indices"):
        w[ict_facekit.not_face_indices] = torch.minimum(
            w[ict_facekit.not_face_indices], torch.tensor(0.2)
        )
    if hasattr(ict_facekit, "skin_face_indices"):
        w[ict_facekit.skin_face_indices] = 1.0
    if hasattr(ict_facekit, "mouth_interior_vertex_indices"):
        w[ict_facekit.mouth_interior_vertex_indices] = 1.0
    if hasattr(ict_facekit, "teeth_indices"):
        w[ict_facekit.teeth_indices] = 0.0
    return w


def build_deform_reg_weight(ict_facekit) -> torch.Tensor:
    """[V] per-vertex L2 penalty on learned deltas (eye socket high, eyeball low)."""
    n_verts = ict_facekit.neutral_mesh.shape[1]
    w = torch.zeros(n_verts, dtype=torch.float32)
    parts = ict_facekit.vertex_parts
    for i, pid in enumerate(parts):
        w[i] = DEFORM_REG_BY_PART_ID.get(int(pid), 0.05)

    if hasattr(ict_facekit, "eye_socket_left_indices"):
        w[ict_facekit.eye_socket_left_indices] = 1.0
    if hasattr(ict_facekit, "eye_socket_right_indices"):
        w[ict_facekit.eye_socket_right_indices] = 1.0
    if hasattr(ict_facekit, "eyeball_indices"):
        w[ict_facekit.eyeball_indices] = 0.02
    if hasattr(ict_facekit, "teeth_indices"):
        w[ict_facekit.teeth_indices] = 2.0
    return w


def build_teeth_mask(ict_facekit) -> torch.Tensor:
    """[V] bool — teeth vertices excluded from template/expression forward."""
    n_verts = ict_facekit.neutral_mesh.shape[1]
    m = torch.zeros(n_verts, dtype=torch.bool)
    if hasattr(ict_facekit, "teeth_indices"):
        m[ict_facekit.teeth_indices] = True
    else:
        parts = ict_facekit.vertex_parts
        for i, pid in enumerate(parts):
            if int(pid) == 6:
                m[i] = True
    return m
