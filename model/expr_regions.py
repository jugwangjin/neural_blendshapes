"""Per-vertex weights for expression deformer (suppress hair/accessory/eye motion)."""

import torch

from gaussian_splatting.gaussian_semantics import ICT_PART_TO_SEMANTIC
from gaussian_splatting.semantic import SEMANTIC_CLASS_INDEX

EXPR_ALLOW_BY_SEMANTIC = {
    "skin": 1.0,
    "lip": 1.0,
    "eye": 0.0,
    "iris": 0.0,
    "hair": 0.0,
    "accessory": 0.0,
    "bg": 0.0,
}

# ICT vertex_parts id -> expression allow (neck weak)
EXPR_WEIGHT_BY_PART_ID = {
    0: 1.0,
    1: 0.2,
    2: 1.0,
    3: 0.0,
    4: 0.0,
    5: 0.0,
    6: 0.0,
    7: 0.0,
}


def build_expr_region_weight(ict_facekit) -> torch.Tensor:
    """
    [V] float weights in [0, 1] for expression delta masking.
    Uses ICT vertex_parts + face_indices; eyeball/hair/accessory suppressed.
    """
    n_verts = ict_facekit.neutral_mesh.shape[1]
    w = torch.zeros(n_verts, dtype=torch.float32)
    parts = ict_facekit.vertex_parts
    for i, pid in enumerate(parts):
        sem = ICT_PART_TO_SEMANTIC.get(int(pid), "skin")
        w[i] = EXPR_ALLOW_BY_SEMANTIC.get(sem, EXPR_WEIGHT_BY_PART_ID.get(int(pid), 0.0))
    if hasattr(ict_facekit, "face_indices"):
        w[ict_facekit.face_indices] = torch.maximum(
            w[ict_facekit.face_indices],
            torch.tensor(1.0),
        )
    if hasattr(ict_facekit, "eyeball_indices"):
        w[ict_facekit.eyeball_indices] = 0.0
    if hasattr(ict_facekit, "not_face_indices"):
        w[ict_facekit.not_face_indices] = torch.minimum(
            w[ict_facekit.not_face_indices], torch.tensor(0.2)
        )
    return w
