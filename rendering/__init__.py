"""Avatar Gaussian rasterization via gsplat (submodule/pip). Does not modify gsplat source."""

from rendering.avatar_renderer import AvatarRenderer

# Backward-compatible alias
GaussianRenderer = AvatarRenderer

__all__ = ["AvatarRenderer", "GaussianRenderer"]
