"""Per-Gaussian gate: disable color_expression on mouth interior + eye regions."""

import torch

# ``utils.ict_regions.classify_surface_triangles_batch`` codes.
# 0 gums/tongue, 1 mouth_socket, 2 eye_socket, 5 sclera, 6 eye_occlusion, 7 teeth
COLOR_EXPRESSION_EXCLUDED_REGION_CODES = frozenset({0, 1, 2, 5, 6, 7})


def color_expression_region_allow(face_region_code, *, enabled=True, dtype=torch.float32):
    """
    [N] gate in {0, 1}: 1 = color_expression allowed, 0 = disabled.

    Face skin (4) and head/neck (3) remain enabled when ``enabled``.
    """
    if not enabled:
        return None
    codes = face_region_code.long()
    excluded = torch.tensor(
        sorted(COLOR_EXPRESSION_EXCLUDED_REGION_CODES),
        device=codes.device,
        dtype=codes.dtype,
    )
    allow = (~torch.isin(codes, excluded)).to(dtype=dtype)
    return allow
