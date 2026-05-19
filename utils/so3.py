"""Small rotation utilities (no PyTorch3D)."""

import torch


def rotation_6d_to_matrix(r6):
    """r6: [..., 6] -> [..., 3, 3] (Zhou et al.)."""
    a1 = r6[..., 0:3]
    a2 = r6[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def apply_rigid(verts, R, t, scale=None):
    """verts [B, V, 3], R [B, 3, 3], t [B, 3], optional uniform ``scale`` (scalar or [B])."""
    if scale is not None:
        if scale.ndim == 0:
            verts = verts * scale
        else:
            verts = verts * scale.view(-1, 1, 1)
    return verts @ R.transpose(-1, -2) + t[:, None, :]


def compose_pose_delta(R_base, t_base, R_delta, t_delta):
    """Apply delta in camera/head frame after base pose."""
    R = R_delta @ R_base
    t = (R_delta @ t_base.unsqueeze(-1)).squeeze(-1) + t_delta
    return R, t
