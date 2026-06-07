"""Region-split densify grad2d thresholds (GB-aligned)."""

import torch

from utils.ict_regions import (
    GB_FACE_SURFACE_REGION_CODE,
    grow_grad2d_threshold_per_gaussian,
)


def test_face_vs_mouth_threshold():
    codes = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long)
    thr = grow_grad2d_threshold_per_gaussian(codes, 2e-4, 5.0)
    assert thr.tolist() == [
        2e-4,
        2e-4,
        2e-4,
        2e-4,
        1e-3,
        2e-4,
        2e-4,
        2e-4,
    ]


def test_face_region_code_constant():
    assert GB_FACE_SURFACE_REGION_CODE == 4
