"""Local tangent frame on ICT eyeball region (no sphere rotation)."""

from dataclasses import dataclass

import torch

from utils.mesh_ops import compute_vertex_normals


@dataclass
class EyeFrame:
    origin: torch.Tensor     # [3]
    tangent: torch.Tensor    # [3]  T
    bitangent: torch.Tensor  # [3]  B
    normal: torch.Tensor     # [3]  N


def build_eye_frame(verts, faces, eyeball_vertex_indices, device=None):
    """
    Build orthonormal frame from eyeball vertex cluster.
    tangent ≈ horizontal in image, bitangent ≈ vertical, normal ≈ outward.
    """
    device = device or verts.device
    idx = torch.tensor(list(eyeball_vertex_indices), dtype=torch.long, device=device)
    pts = verts[idx]
    origin = pts.mean(dim=0)

    v_normals = compute_vertex_normals(verts, faces)
    normal = torch.nn.functional.normalize(v_normals[idx].mean(dim=0), dim=0)

    centered = pts - origin
    cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
    _, evecs = torch.linalg.eigh(cov)
    tangent = torch.nn.functional.normalize(evecs[:, -1], dim=0)

    bitangent = torch.cross(normal, tangent)
    bitangent = torch.nn.functional.normalize(bitangent, dim=0)
    tangent = torch.cross(bitangent, normal)
    tangent = torch.nn.functional.normalize(tangent, dim=0)

    return EyeFrame(origin=origin, tangent=tangent, bitangent=bitangent, normal=normal)


def points_on_eye_plane(local_uv, h, eye_frame: EyeFrame, gaze_offset=None):
    """
    local_uv: [G, 2], h: [G, 1]
    gaze_offset: [G, 2] or [1, 2] added to uv before placement
    X = O + (u+du)*T + (v+dv)*B + h*N
    """
    uv = local_uv
    if gaze_offset is not None:
        if gaze_offset.ndim == 1:
            gaze_offset = gaze_offset.unsqueeze(0)
        uv = uv + gaze_offset

    o = eye_frame.origin
    t = eye_frame.tangent
    b = eye_frame.bitangent
    n = eye_frame.normal

    xyz = (
        o
        + uv[:, 0:1] * t
        + uv[:, 1:2] * b
        + h * n
    )
    return xyz, uv
