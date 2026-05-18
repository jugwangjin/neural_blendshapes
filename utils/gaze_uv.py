"""Map eye gaze to UV offset on per-eye texture space (base range + optional refine)."""

import torch


def gaze_uv_from_expression(expression_weights, expression_names, side, uv_range):
    """
    side: 'L' or 'R'
    Returns [2] tensor (du, dv) in [-uv_range, uv_range].
    """
    names = (
        expression_names.tolist()
        if hasattr(expression_names, "tolist")
        else list(expression_names)
    )
    w = expression_weights[0] if expression_weights.ndim > 1 else expression_weights

    du = w[names.index(f"eyeLookOut_{side}")] - w[names.index(f"eyeLookIn_{side}")]
    dv = w[names.index(f"eyeLookUp_{side}")] - w[names.index(f"eyeLookDown_{side}")]

    offset = torch.stack([du, dv]) * uv_range
    return offset.clamp(-uv_range, uv_range)


def apply_gaze_refine(base_offset, refine_offset):
    """base_offset, refine_offset: [2]. Sum then caller may clamp again."""
    if refine_offset is None:
        return base_offset
    return base_offset + refine_offset
