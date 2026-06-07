"""GB-style mouth interior: jaw-only expression on socket / teeth / gum / tongue vertices."""

import torch

from utils.mediapipe_blendshapes import ICT_GATHER_MP_NAMES

JAW_ICT_EXPRESSION_NAMES = frozenset({"jawOpen", "jawForward", "jawLeft", "jawRight"})
JAW_MP_CHANNEL_NAMES = frozenset({"jawOpen", "jawForward", "jawLeft", "jawRight"})

# ICT FaceKit naming (``mouthSmile_L``), not MediaPipe ``mouthSmileLeft``.
NON_JAW_ICT_EXPRESSION_CANDIDATES = (
    "mouthSmile_L",
    "mouthSmile_R",
    "mouthPucker",
    "mouthFunnel",
)

MOUTH_INTERIOR_VERTEX_KEYS = (
    "mouth_interior_vertex_indices",
    "gums_tongue_indices",
    "teeth_indices",
    "mouth_socket_indices",
)


def collect_mouth_interior_vertex_indices(ict):
    seen = set()
    out = []
    for key in MOUTH_INTERIOR_VERTEX_KEYS:
        ids = getattr(ict, key, None)
        if ids is None:
            continue
        for i in ids:
            i = int(i)
            if i not in seen:
                seen.add(i)
                out.append(i)
    return out


def jaw_ict_expression_indices(expression_names):
    if hasattr(expression_names, "tolist"):
        names = expression_names.tolist()
    else:
        names = list(expression_names)
    return sorted(i for i, n in enumerate(names) if n in JAW_ICT_EXPRESSION_NAMES)


def jaw_mp_channel_indices():
    return sorted(j for j, n in enumerate(ICT_GATHER_MP_NAMES) if n in JAW_MP_CHANNEL_NAMES)


def pick_non_jaw_ict_expression_index(expression_names):
    """First available lip/smile ICT mode index (not jaw)."""
    if hasattr(expression_names, "tolist"):
        names = expression_names.tolist()
    else:
        names = list(expression_names)
    for candidate in NON_JAW_ICT_EXPRESSION_CANDIDATES:
        if candidate in names:
            return names.index(candidate)
    for i, name in enumerate(names):
        if name not in JAW_ICT_EXPRESSION_NAMES:
            return i
    raise ValueError("no non-jaw ICT expression found")


def _as_long_tensor(ids, device):
    if torch.is_tensor(ids):
        return ids.to(device=device, dtype=torch.long)
    return torch.tensor(list(ids), device=device, dtype=torch.long)


def build_ict_expression_vertex_allow_mask(
    num_expression,
    num_vertices,
    interior_vertex_indices,
    jaw_expression_indices,
    device,
):
    """
    [E, V] float gate for ICT linear blendshapes.

    Non-jaw modes are zeroed on mouth-interior vertices; all modes stay active elsewhere.
    """
    mask = torch.ones(num_expression, num_vertices, dtype=torch.float32, device=device)
    interior = _as_long_tensor(interior_vertex_indices, device)
    if interior.numel() == 0:
        return mask
    jaw = set(int(i) for i in jaw_expression_indices)
    non_jaw = [e for e in range(num_expression) if e not in jaw]
    if not non_jaw:
        return mask
    e_idx = torch.tensor(non_jaw, device=device, dtype=torch.long)
    mask[e_idx[:, None], interior] = 0.0
    return mask


def build_mp_expr_vertex_allow_mask(
    num_channels,
    num_vertices,
    interior_vertex_indices,
    jaw_channel_indices,
    device,
):
    """
    [J, V] float gate for expr_mlp (MP gather channel order).

    Non-jaw AU channels are zeroed on mouth-interior vertices.
    """
    mask = torch.ones(num_channels, num_vertices, dtype=torch.float32, device=device)
    interior = _as_long_tensor(interior_vertex_indices, device)
    if interior.numel() == 0:
        return mask
    jaw = set(int(j) for j in jaw_channel_indices)
    non_jaw = [j for j in range(num_channels) if j not in jaw]
    if not non_jaw:
        return mask
    j_idx = torch.tensor(non_jaw, device=device, dtype=torch.long)
    mask[j_idx[:, None], interior] = 0.0
    return mask


def register_mouth_interior_jaw_masks(ict, *, enabled=True):
    """
    Register buffers on ``ict`` for ICT linear blendshape vertex masking.

    Returns interior vertex count (0 if disabled or empty).
    """
    device = ict.neutral_mesh.device
    n_verts = int(ict.neutral_mesh.shape[1])
    n_expr = int(ict.num_expression)
    interior_ids = collect_mouth_interior_vertex_indices(ict)
    if not enabled or not interior_ids:
        for key in (
            "mouth_interior_vertex_idx",
            "ict_expression_vertex_allow_mask",
        ):
            if hasattr(ict, key):
                delattr(ict, key)
        return 0

    interior_idx = _as_long_tensor(interior_ids, device)
    jaw_e = jaw_ict_expression_indices(ict.expression_names)
    allow = build_ict_expression_vertex_allow_mask(
        n_expr, n_verts, interior_idx, jaw_e, device
    )
    ict.register_buffer("mouth_interior_vertex_idx", interior_idx)
    ict.register_buffer("ict_expression_vertex_allow_mask", allow)
    return int(interior_idx.numel())


def mouth_interior_jaw_gate_for_deformer(ict, n_coeffs, *, enabled=True):
    """[J, V] multiplier for ``ICTDeformer.expr_gate`` (1 = allow)."""
    device = ict.neutral_mesh.device
    n_verts = int(ict.neutral_mesh.shape[1])
    interior_ids = collect_mouth_interior_vertex_indices(ict)
    if not enabled or not interior_ids:
        return None
    return build_mp_expr_vertex_allow_mask(
        n_coeffs,
        n_verts,
        interior_ids,
        jaw_mp_channel_indices(),
        device,
    )
