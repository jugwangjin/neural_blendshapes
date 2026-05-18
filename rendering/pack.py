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


def pack_gaussians(avatar_out, rgb_activation=torch.sigmoid):
    xyz = avatar_out["xyz"].reshape(-1, 3)
    scale = avatar_out["scale"].reshape(-1, 3)
    rotation = normalize_quaternion_wxyz(avatar_out["rotation"].reshape(-1, 4))
    opacity = avatar_out["opacity"].reshape(-1)
    color = avatar_out["color"].reshape(-1, avatar_out["color"].shape[-1])
    if color.shape[-1] != 3:
        color = color[..., :3]
    if color.shape[-1] < 3:
        pad = 3 - color.shape[-1]
        color = torch.cat([color, color.new_zeros(color.shape[0], pad)], dim=-1)
    if rgb_activation is not None:
        color = rgb_activation(color)

    sem_prob = avatar_out.get("sem_prob")
    if sem_prob is not None:
        sem_prob = sem_prob.reshape(-1, sem_prob.shape[-1])

    return {
        "means": xyz,
        "quats": rotation,
        "scales": scale,
        "opacities": opacity,
        "colors": color,
        "sem_prob": sem_prob,
    }
