"""
Per-eye ICT texture-space Gaussians: UV slide on separate atlases, h=0 on eyeball.

Gaze: base offset from eyeLook* (clamped to gaze_uv_range) + learnable refine.
"""

import torch
import torch.nn as nn

from gaussian_splatting.gaussian_semantics import eye_fixed_semantic_probs
from model.uvh_gaussians import UVHGaussians
from utils.gaze_uv import apply_gaze_refine, gaze_uv_from_expression
from utils.uv_mesh import UVMesh, surface_points_from_uvh

LEFT_IRIS_MP = [468, 469, 470, 471, 472]
RIGHT_IRIS_MP = [473, 474, 475, 476, 477]

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
        n_per_eye=64,
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
        self.left = UVHGaussians(
            n_per_eye,
            sh_dim=sh_dim,
            fixed_h=0.0,
            n_semantic_classes=n_semantic_classes,
            fixed_semantic_probs=eye_sem,
        )
        self.right = UVHGaussians(
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

        self._gaze_base_left = None
        self._gaze_base_right = None

    def set_gaze_from_tracker(self, gaze_uv_left, gaze_uv_right):
        """
        Primary gaze path: tracker MLP outputs (MediaPipe iris / eyeLook corrected).
        Separate from mesh expression_deformer.
        """
        self._gaze_base_left = gaze_uv_left.reshape(-1)[:2]
        self._gaze_base_right = gaze_uv_right.reshape(-1)[:2]

    def set_gaze_base(self, left, right):
        """Alias for tracker gaze UV."""
        self.set_gaze_from_tracker(left, right)

    def gaze_from_expression(self, expression_weights, expression_names):
        self._gaze_base_left = gaze_uv_from_expression(
            expression_weights, expression_names, "L", self.gaze_uv_range
        )
        self._gaze_base_right = gaze_uv_from_expression(
            expression_weights, expression_names, "R", self.gaze_uv_range
        )

    def _effective_gaze(self, side):
        if side == "L":
            base = self._gaze_base_left
            refine = self.gaze_refine_left
        else:
            base = self._gaze_base_right
            refine = self.gaze_refine_right
        if base is None:
            base = torch.zeros(2, device=self.left.uv.device)
        offset = apply_gaze_refine(base, refine)
        return offset.clamp(-self.gaze_uv_range, self.gaze_uv_range)

    def _forward_one(self, module: UVHGaussians, uv_mesh: UVMesh, verts, faces, gaze_offset):
        uv_eff = module.uv + gaze_offset.unsqueeze(0)
        h = torch.zeros_like(module.h)
        xyz, face_idx, bary, normals = surface_points_from_uvh(uv_eff, h, uv_mesh)
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

    def forward(self, left_uv_mesh: UVMesh, right_uv_mesh: UVMesh, verts, faces):
        gaze_l = self._effective_gaze("L")
        gaze_r = self._effective_gaze("R")

        out_l = self._forward_one(self.left, left_uv_mesh, verts, faces, gaze_l)
        out_r = self._forward_one(self.right, right_uv_mesh, verts, faces, gaze_r)

        n = self.n_per_eye
        iris_idx = torch.cat(
            [
                torch.arange(self.n_iris_control, device=out_l["xyz"].device),
                torch.arange(n, n + self.n_iris_control, device=out_l["xyz"].device),
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