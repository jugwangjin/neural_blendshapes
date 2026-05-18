"""
Fixed face_idx + bary surface Gaussians (no train-time UV lookup).

  xyz = bary_sample(verts, face_idx, bary) + h * normal
"""

import torch
import torch.nn as nn

from utils.barycentric import sample_normals, sample_surface


class SurfaceGaussians(nn.Module):
    def __init__(
        self,
        face_idx,
        bary,
        sh_dim=3,
        n_semantic_classes=0,
        fixed_semantic_probs=None,
    ):
        super().__init__()
        n = face_idx.shape[0]
        self.register_buffer("face_idx", face_idx.long())
        self.register_buffer("bary", bary.float())
        self.n_semantic_classes = n_semantic_classes
        self.sem_logits = None
        self.register_buffer("sem_prob_fixed", None)
        self.register_buffer("sem_anchor", None)
        self.register_buffer("sem_frozen_dims", None)
        if fixed_semantic_probs is not None:
            self.register_buffer("sem_prob_fixed", fixed_semantic_probs.clone())
        elif n_semantic_classes > 0:
            self.sem_logits = nn.Parameter(torch.zeros(n, n_semantic_classes))

        self.h = nn.Parameter(torch.zeros(n, 1))
        self.log_scale = nn.Parameter(torch.zeros(n, 3))
        self.rotation = nn.Parameter(torch.zeros(n, 4))
        self.rotation.data[:, 0] = 1.0
        self.opacity = nn.Parameter(torch.zeros(n, 1))
        self.color = nn.Parameter(torch.zeros(n, sh_dim))

    @property
    def n_gaussians(self):
        return self.face_idx.shape[0]

    def forward(self, verts, faces):
        if verts.ndim == 3:
            verts = verts[0]
        xyz_base = sample_surface(verts, faces, self.face_idx, self.bary)
        vn = sample_normals(
            self._vertex_normals(verts, faces), faces, self.face_idx, self.bary
        )
        xyz = xyz_base + self.h * vn
        scale = torch.exp(self.log_scale).clamp(max=0.05)
        opacity = torch.sigmoid(self.opacity)
        out = {
            "xyz": xyz,
            "scale": scale,
            "rotation": self.rotation,
            "opacity": opacity,
            "color": self.color,
            "h": self.h,
            "face_idx": self.face_idx,
            "bary": self.bary,
            "normals": vn,
            "group": "surface",
        }
        if self.sem_prob_fixed is not None:
            out["sem_prob"] = self.sem_prob_fixed
        elif self.sem_logits is not None:
            from rendering.gaussian_semantics import semantic_probs_with_anchor

            out["sem_prob"] = semantic_probs_with_anchor(
                self.sem_logits, self.sem_anchor, self.sem_frozen_dims
            )
        return out

    @staticmethod
    def _vertex_normals(verts, faces):
        from utils.mesh_ops import vertex_normals

        return vertex_normals(verts, faces)
