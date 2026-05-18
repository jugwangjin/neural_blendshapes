"""
Mesh-embedded Gaussians in UV + normal displacement (SplattingAvatar-style).

  X(u, v, h) = S(u, v) + h * N(u, v)
"""

import torch
import torch.nn as nn

from utils.uv_mesh import UVMesh, surface_points_from_uvh


class UVHGaussians(nn.Module):
    def __init__(self, n_gaussians, sh_dim=3, uv_init=None, fixed_h=None):
        super().__init__()
        if uv_init is None:
            uv_init = torch.rand(n_gaussians, 2)
        self.uv = nn.Parameter(uv_init.clone())
        self.fixed_h = fixed_h
        if fixed_h is None:
            self.h = nn.Parameter(torch.zeros(n_gaussians, 1))
        else:
            self.register_buffer("h", torch.full((n_gaussians, 1), float(fixed_h)))

        self.log_scale = nn.Parameter(torch.zeros(n_gaussians, 3))
        self.rotation = nn.Parameter(torch.zeros(n_gaussians, 4))
        self.opacity = nn.Parameter(torch.zeros(n_gaussians, 1))
        self.color = nn.Parameter(torch.zeros(n_gaussians, sh_dim))

    def get_h(self):
        return self.h if self.fixed_h is None else self.h.detach()

    def forward(self, uv_mesh: UVMesh, verts=None, faces=None, uv_offset=None):
        mesh = UVMesh(
            verts=verts if verts is not None else uv_mesh.verts,
            faces=faces if faces is not None else uv_mesh.faces,
            verts_uvs=uv_mesh.verts_uvs,
            faces_uvs=uv_mesh.faces_uvs,
            active_face_idx=uv_mesh.active_face_idx,
        )
        uv = self.uv
        if uv_offset is not None:
            uv = uv + uv_offset.unsqueeze(0)

        h = self.get_h()
        xyz, face_idx, bary, normals = surface_points_from_uvh(uv, h, mesh)
        scale = torch.exp(self.log_scale).clamp(max=0.05)
        opacity = torch.sigmoid(self.opacity)
        return {
            "xyz": xyz,
            "scale": scale,
            "rotation": self.rotation,
            "opacity": opacity,
            "color": self.color,
            "h": h,
            "face_idx": face_idx,
            "bary": bary,
            "normals": normals,
            "group": "face_uvh",
        }
