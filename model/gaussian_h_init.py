"""Initial offset ``h`` for surface Gaussians (teeth volume)."""

import torch


def _teeth_gaussian_mask(surface):
    mask = getattr(surface, "is_teeth", None)
    if mask is not None and mask.any():
        return mask.bool()
    return surface.face_region_code == 7


def init_teeth_h(surface, ict, radius: float):
    """
    ICT teeth (region code 7 / ``is_teeth``): ``h ~ Uniform(-radius, radius)`` along mesh normal.

    Scatters Gaussians through ~2×radius tooth thickness; semantic / prune cull bad samples.
    """
    if radius <= 0:
        return
    mask = _teeth_gaussian_mask(surface)
    if not mask.any():
        return

    n = int(mask.sum().item())
    u = torch.rand(n, 1, device=surface.h.device, dtype=surface.h.dtype) * 2.0 - 1.0
    surface.h.data[mask] = u * float(radius)
