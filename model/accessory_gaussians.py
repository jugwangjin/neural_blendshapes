"""
Optional free accessory Gaussians (not on ICT surface).

Created only when n_gaussians > 0 (dataset has accessory segmentation).
Learnable tangent slide (2D) + normal distance from ray-anchored init.
"""

import torch
import torch.nn as nn

from rendering.semantic import SEMANTIC_CLASS_INDEX


class AccessoryGaussians(nn.Module):
    def __init__(
        self,
        n_gaussians,
        anchor_xyz,
        tangent_basis,
        sh_dim=3,
        n_semantic_classes=7,
    ):
        super().__init__()
        self.n_gaussians = n_gaussians
        if n_gaussians == 0:
            self.register_buffer("anchor_xyz", torch.zeros(0, 3))
            self.register_buffer("tangent_basis", torch.zeros(0, 3, 2))
            self.slide_uv = None
            self.distance = None
            self.log_scale = None
            self.rotation = None
            self.opacity = None
            self.color = None
            self.register_buffer("sem_prob_fixed", torch.zeros(0, n_semantic_classes))
            return

        self.register_buffer("anchor_xyz", anchor_xyz.float())
        self.register_buffer("tangent_basis", tangent_basis.float())
        self.slide_uv = nn.Parameter(torch.zeros(n_gaussians, 2))
        self.distance = nn.Parameter(torch.zeros(n_gaussians, 1))
        self.log_scale = nn.Parameter(torch.full((n_gaussians, 3), -3.0))
        self.rotation = nn.Parameter(torch.zeros(n_gaussians, 4))
        self.rotation.data[:, 0] = 1.0
        self.opacity = nn.Parameter(torch.zeros(n_gaussians, 1))
        self.color = nn.Parameter(torch.zeros(n_gaussians, sh_dim))

        acc_i = SEMANTIC_CLASS_INDEX["accessory"]
        sem = torch.zeros(n_gaussians, n_semantic_classes)
        sem[:, acc_i] = 1.0
        self.register_buffer("sem_prob_fixed", sem)

    def forward(self):
        if self.n_gaussians == 0:
            device = self.anchor_xyz.device
            return {
                "xyz": torch.zeros(0, 3, device=device),
                "scale": torch.zeros(0, 3, device=device),
                "rotation": torch.zeros(0, 4, device=device),
                "opacity": torch.zeros(0, 1, device=device),
                "color": torch.zeros(0, 3, device=device),
                "h": torch.zeros(0, 1, device=device),
                "sem_prob": self.sem_prob_fixed,
                "group": "accessory",
            }

        offset = self.slide_uv[:, 0:1] * self.tangent_basis[:, :, 0] + self.slide_uv[:, 1:2] * self.tangent_basis[:, :, 1]
        xyz = self.anchor_xyz + offset + self.distance * self._normals()
        scale = torch.exp(self.log_scale).clamp(max=0.08)
        opacity = torch.sigmoid(self.opacity)
        return {
            "xyz": xyz,
            "scale": scale,
            "rotation": self.rotation,
            "opacity": opacity,
            "color": self.color,
            "h": self.distance,
            "sem_prob": self.sem_prob_fixed,
            "group": "accessory",
        }

    def _normals(self):
        t0 = self.tangent_basis[:, :, 0]
        t1 = self.tangent_basis[:, :, 1]
        n = torch.cross(t0, t1, dim=-1)
        return n / n.norm(dim=-1, keepdim=True).clamp(min=1e-8)
