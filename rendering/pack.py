"""Pack GaussianAvatar output for gsplat rasterization."""

import torch


def normalize_quaternion_wxyz(q):
    n = q.norm(dim=-1, keepdim=True)
    q = q / n.clamp(min=1e-8)
    near_zero = n.squeeze(-1) < 1e-6
    if near_zero.any():
        identity = torch.zeros_like(q)
        identity[:, 0] = 1.0
        q = torch.where(near_zero.unsqueeze(-1), identity, q)
    return q


def surface_avatar_out(avatar_out):
    """Mesh surface Gaussians (same as full avatar output today)."""
    surf = avatar_out["surface"]
    return {
        "xyz": surf["xyz"],
        "scale": surf["scale"],
        "rotation": surf["rotation"],
        "opacity": surf["opacity"],
        "color": surf["color"],
        "h": surf["h"],
    }


def pack_gaussians(avatar_out, rgb_activation=torch.sigmoid, sh_degree=None):
    xyz = avatar_out["xyz"].reshape(-1, 3)
    scale = avatar_out["scale"].reshape(-1, 3)
    rotation = normalize_quaternion_wxyz(avatar_out["rotation"].reshape(-1, 4))
    opacity = avatar_out["opacity"].reshape(-1)
    color = avatar_out["color"]
    if color.ndim == 3:
        # [N, K, 3] SH coefficients — never flatten K into N (breaks gsplat layout).
        color = color.reshape(color.shape[0], color.shape[1], 3)
        if sh_degree is None:
            color = color[:, 0, :]
            if rgb_activation is not None:
                color = rgb_activation(color)
        elif rgb_activation is not None:
            color = color.clone()
            color[:, 0, :] = rgb_activation(color[:, 0, :])
    else:
        color = color.reshape(-1, color.shape[-1])
        if color.shape[-1] != 3:
            color = color[..., :3]
        if color.shape[-1] < 3:
            pad = 3 - color.shape[-1]
            color = torch.cat([color, color.new_zeros(color.shape[0], pad)], dim=-1)
        if rgb_activation is not None:
            color = rgb_activation(color)

    sem_features = avatar_out.get("sem_features")
    if sem_features is not None:
        sem_features = sem_features.reshape(-1, sem_features.shape[-1])

    return {
        "means": xyz,
        "quats": rotation,
        "scales": scale,
        "opacities": opacity,
        "colors": color,
        "sem_features": sem_features,
    }


def pack_gaussians_silhouette(avatar_out, rgb_activation=None, detach_covariance=True):
    """
    Pack for silhouette / mask loss: stop grads through covariance (scale + rotation) only.

    ``means`` and ``opacities`` stay attached so mask loss can move surface position and
  fill the interior (detached opacity would leave mask loss with no in-region signal).
    """
    packed = pack_gaussians(avatar_out, rgb_activation=rgb_activation, sh_degree=None)
    if detach_covariance:
        packed["scales"] = packed["scales"].detach()
        packed["quats"] = packed["quats"].detach()
    packed["colors"] = packed["colors"].detach()
    return packed
