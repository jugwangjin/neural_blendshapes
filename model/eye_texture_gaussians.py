"""
Per-eye ICT texture-space Gaussians: h=0, UV slide via gaze_uv from tracker.
"""

import torch
import torch.nn as nn

from model.texture_space_gaussians import TextureSpaceGaussians
from rendering.gaussian_semantics import eye_fixed_semantic_probs
from utils.gaze_uv import combine_gaze
from utils.uv_mesh import UVMesh, surface_points_from_uvh

IRIS_TEMPLATE_UV = torch.tensor(
    [
        [0.0, 0.0],
        [0.0, 0.35],
        [0.35, 0.0],
        [0.0, -0.35],
        [-0.35, 0.0],
    ],
    dtype=torch.float32,
)


class EyeTextureGaussians(nn.Module):
    def __init__(
        self,
        n_per_eye=1024,
        sh_dim=3,
        n_iris_control=5,
        gaze_uv_range=0.12,
        learn_gaze_refine=True,
        n_semantic_classes=7,
    ):
        super().__init__()
        self.n_per_eye = n_per_eye
        self.n_iris_control = min(n_iris_control, n_per_eye)
        self.gaze_uv_range = gaze_uv_range
        self.n_semantic_classes = n_semantic_classes

        eye_sem = None
        if n_semantic_classes > 0:
            eye_sem = eye_fixed_semantic_probs(
                n_per_eye, self.n_iris_control, n_semantic_classes, device="cpu"
            )
        self.left = TextureSpaceGaussians(
            n_per_eye,
            sh_dim=sh_dim,
            fixed_h=0.0,
            n_semantic_classes=n_semantic_classes,
            fixed_semantic_probs=eye_sem,
        )
        self.right = TextureSpaceGaussians(
            n_per_eye,
            sh_dim=sh_dim,
            fixed_h=0.0,
            n_semantic_classes=n_semantic_classes,
            fixed_semantic_probs=eye_sem,
        )

        with torch.no_grad():
            self.left.uv[: self.n_iris_control] = IRIS_TEMPLATE_UV[: self.n_iris_control]
            self.right.uv[: self.n_iris_control] = IRIS_TEMPLATE_UV[: self.n_iris_control]

        if learn_gaze_refine:
            self.gaze_refine_left = nn.Parameter(torch.zeros(2))
            self.gaze_refine_right = nn.Parameter(torch.zeros(2))
        else:
            self.gaze_refine_left = None
            self.gaze_refine_right = None

    def _apply_gaze_refine(self, gaze_uv, side):
        refine = self.gaze_refine_left if side == "L" else self.gaze_refine_right
        if refine is None:
            return gaze_uv
        if gaze_uv.ndim == 1:
            gaze_uv = gaze_uv.unsqueeze(0)
        out = combine_gaze(gaze_uv, refine.unsqueeze(0), self.gaze_uv_range)
        return out.squeeze(0) if out.shape[0] == 1 else out

    def _forward_one(self, module: TextureSpaceGaussians, uv_mesh: UVMesh, verts, faces, gaze_offset):
        uv_eff = module.uv + gaze_offset.unsqueeze(0)
        h = torch.zeros_like(module.h)
        xyz, face_idx, bary, normals = surface_points_from_uvh(uv_eff, h, uv_mesh, module)
        scale = torch.exp(module.log_scale).clamp(max=0.02)
        opacity = torch.sigmoid(module.opacity)
        out = {
            "xyz": xyz,
            "scale": scale,
            "rotation": module.rotation,
            "opacity": opacity,
            "color": module.color,
            "h": h,
            "uv": uv_eff,
            "gaze_offset": gaze_offset,
            "face_idx": face_idx,
            "bary": bary,
        }
        if module.sem_prob_fixed is not None:
            out["sem_prob"] = module.sem_prob_fixed
        elif module.sem_logits is not None:
            out["sem_prob"] = torch.softmax(module.sem_logits, dim=-1)
        return out

    def forward(
        self,
        left_uv_mesh: UVMesh,
        right_uv_mesh: UVMesh,
        verts,
        faces,
        gaze_uv_left=None,
        gaze_uv_right=None,
    ):
        device = self.left.uv.device
        if gaze_uv_left is None:
            gaze_uv_left = torch.zeros(2, device=device)
        if gaze_uv_right is None:
            gaze_uv_right = torch.zeros(2, device=device)

        if gaze_uv_left.ndim == 2:
            gaze_l = self._apply_gaze_refine(gaze_uv_left[0], "L")
            gaze_r = self._apply_gaze_refine(gaze_uv_right[0], "R")
        else:
            gaze_l = self._apply_gaze_refine(gaze_uv_left, "L")
            gaze_r = self._apply_gaze_refine(gaze_uv_right, "R")

        out_l = self._forward_one(self.left, left_uv_mesh, verts, faces, gaze_l)
        out_r = self._forward_one(self.right, right_uv_mesh, verts, faces, gaze_r)

        n = self.n_per_eye
        iris_idx = torch.cat(
            [
                torch.arange(self.n_iris_control, device=device),
                torch.arange(n, n + self.n_iris_control, device=device),
            ]
        )

        xyz = torch.cat([out_l["xyz"], out_r["xyz"]], dim=0)
        out = {
            "left": out_l,
            "right": out_r,
            "xyz": xyz,
            "scale": torch.cat([out_l["scale"], out_r["scale"]], dim=0),
            "rotation": torch.cat([out_l["rotation"], out_r["rotation"]], dim=0),
            "opacity": torch.cat([out_l["opacity"], out_r["opacity"]], dim=0),
            "color": torch.cat([out_l["color"], out_r["color"]], dim=0),
            "h": torch.cat([out_l["h"], out_r["h"]], dim=0),
            "iris_control_xyz": xyz[iris_idx],
            "is_eyeball_surface": torch.ones(xyz.shape[0], dtype=torch.bool, device=device),
        }
        if out_l.get("sem_prob") is not None:
            out["sem_prob"] = torch.cat([out_l["sem_prob"], out_r["sem_prob"]], dim=0)
        return out
