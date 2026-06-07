"""Per-vertex weights for ICT deformer (expression gate + deform regularization)."""

import torch

EXPR_WEIGHT_BY_PART_ID = {
    0: 1.0,   # face skin
    1: 0.15,  # head/neck — low gate; mesh/template carries outer silhouette
    2: 1.0,   # mouth socket
    3: 1.0,   # eye socket L (same MLP field; socket reg limits delta)
    4: 1.0,   # eye socket R
    5: 1.0,   # gums/tongue (surface Gaussians substitute teeth)
    6: 0.0,   # teeth — no Gaussians, no deformer
    7: 1.0,   # eyeball L
    8: 1.0,   # eyeball R
    9: 0.0,   # lacrimal — kept in npy, ignored by deformer
    10: 0.0,  # eye blend
    11: 0.0,
    12: 0.0,
    13: 1.0,  # eye occlusion L — ICT blendshapes + surface Gaussians
    14: 1.0,  # eye occlusion R
    15: 0.0,  # eyelashes
    16: 0.0,
}

# L2 penalty weight on template / expression deltas (not a forward gate).
DEFORM_REG_BY_PART_ID = {
    0: 0.5,
    1: 2.0,   # head/neck — strong L2 on deltas (anti shrink / collapse)
    2: 0.5,
    3: 0.5,   # eye socket — penalize large deltas (orbit behind eye)
    4: 0.5,
    5: 0.5,  # gums — allow template/expr field to bulge over teeth volume
    6: 0.5,   # teeth (masked in forward; unused)
    7: 0.1,  # eyeball — light reg, field moves with lids   # unused
    8: 0.1,
    9: 0.1,
    10: 0.1,
    11: 0.1,
    12: 0.1,
    13: 0.1,  # eye occlusion — allow lid motion
    14: 0.1,
    15: 0.1,
    16: 0.1,
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
            w[ict_facekit.not_face_indices], torch.tensor(0.15)
        )
    if hasattr(ict_facekit, "skin_face_indices"):
        w[ict_facekit.skin_face_indices] = 1.0
    if hasattr(ict_facekit, "mouth_interior_vertex_indices"):
        w[ict_facekit.mouth_interior_vertex_indices] = 1.0
    for key in (
        "lacrimal_indices",
        "eye_blend_indices",
        "eyelashes_left_indices",
        "eyelashes_right_indices",
    ):
        ids = getattr(ict_facekit, key, None)
        if ids is not None and len(ids) > 0:
            w[ids] = 0.0
    for key in ("left_eye_occlusion_indices", "right_eye_occlusion_indices"):
        ids = getattr(ict_facekit, key, None)
        if ids is not None and len(ids) > 0:
            w[ids] = 1.0
    return w


def build_deform_reg_weight(ict_facekit) -> torch.Tensor:
    """[V] per-vertex L2 penalty on learned deltas (eye socket high, eyeball low)."""
    n_verts = ict_facekit.neutral_mesh.shape[1]
    w = torch.zeros(n_verts, dtype=torch.float32)
    parts = ict_facekit.vertex_parts
    for i, pid in enumerate(parts):
        w[i] = DEFORM_REG_BY_PART_ID.get(int(pid), 0.05)

    if hasattr(ict_facekit, "not_face_indices"):
        w[ict_facekit.not_face_indices] = torch.maximum(
            w[ict_facekit.not_face_indices], torch.tensor(2.5)
        )
    if hasattr(ict_facekit, "eye_socket_left_indices"):
        w[ict_facekit.eye_socket_left_indices] = DEFORM_REG_BY_PART_ID.get(3, 0.1)
    if hasattr(ict_facekit, "eye_socket_right_indices"):
        w[ict_facekit.eye_socket_right_indices] = DEFORM_REG_BY_PART_ID.get(4, 0.1)
    if hasattr(ict_facekit, "eyeball_indices"):
        w[ict_facekit.eyeball_indices] = DEFORM_REG_BY_PART_ID.get(7, 0.1)
    for key in (
        "lacrimal_indices",
        "eye_blend_indices",
        "eyelashes_left_indices",
        "eyelashes_right_indices",
        "teeth_indices",
    ):
        ids = getattr(ict_facekit, key, None)
        if ids is not None and len(ids) > 0:
            w[ids] = DEFORM_REG_BY_PART_ID.get(6, 2.0)
    for key in ("left_eye_occlusion_indices", "right_eye_occlusion_indices"):
        ids = getattr(ict_facekit, key, None)
        if ids is not None and len(ids) > 0:
            w[ids] = DEFORM_REG_BY_PART_ID.get(13, 0.1)
    return w
