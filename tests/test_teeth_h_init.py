"""Teeth surface Gaussian h init."""

import torch

from model.gaussian_h_init import init_teeth_h, _teeth_gaussian_mask


def test_init_teeth_h_uniform():
    n = 4
    h = torch.zeros(n, 1)
    face_idx = torch.zeros(n, dtype=torch.long)
    bary = torch.full((n, 3), 1.0 / 3.0)
    face_region_code = torch.tensor([7, 7, 4, 4], dtype=torch.long)
    is_teeth = face_region_code == 7

    class Surf:
        pass

    class Ict:
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
        def template_reference_verts(self):
            return torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
            )

    surf = Surf()
    surf.h = h
    surf.face_idx = face_idx
    surf.bary = bary
    surf.face_region_code = face_region_code
    surf.is_teeth = is_teeth

    init_teeth_h(surf, Ict(), 0.02)
    assert _teeth_gaussian_mask(surf).tolist() == [True, True, False, False]
    assert (surf.h[:2].abs() <= 0.02 + 1e-6).all()
    assert (surf.h[2:].abs() < 1e-8).all()
