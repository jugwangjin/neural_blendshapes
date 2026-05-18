"""ICT FACS deformer: MP coeffs → mesh + pose weight (no DECA / feature slicing)."""

import torch
import torch.nn as nn

from model.pose_weight import PoseWeightMLP
from utils.so3 import apply_rigid, rotation_6d_to_matrix


class ICTDeformer(nn.Module):
    def __init__(self, ict_facekit, template_offset_scale=1e-2):
        super().__init__()
        self.ict = ict_facekit
        self.template_offset = nn.Parameter(
            torch.zeros_like(ict_facekit.neutral_mesh[0]) * template_offset_scale
        )
        self.pose_weight_net = PoseWeightMLP()
        canonical = ict_facekit.neutral_mesh_canonical[0].detach()
        self.register_buffer("canonical_xyz", canonical)

    def mp_to_ict_expression(self, mp_coeffs):
        """mp_coeffs [B, 52] -> ICT expression [B, num_expression]."""
        B = mp_coeffs.shape[0]
        n_exp = self.ict.num_expression
        out = torch.zeros(B, n_exp, device=mp_coeffs.device, dtype=mp_coeffs.dtype)
        idx = self.ict.mediapipe_to_ict
        n = min(len(idx), mp_coeffs.shape[1])
        out[:, idx[:n]] = mp_coeffs[:, :n]
        return out

    def apply_weighted_pose(self, verts, R, t):
        """verts [B,V,3], R [B,3,3], t [B,3]."""
        w = self.pose_weight_net(self.canonical_xyz)
        rigid = apply_rigid(verts, R, t)
        return (1.0 - w.unsqueeze(0)) * verts + w.unsqueeze(0) * rigid

    def forward(
        self,
        mp_coeffs_corr,
        pose_rotation_6d=None,
        pose_translation=None,
        expr_delta=None,
        return_unposed=False,
    ):
        """
        mp_coeffs_corr: [B, 52]
        pose_rotation_6d: [B, 6] residual (optional)
        pose_translation: [B, 3] residual (optional)
        """
        B = mp_coeffs_corr.shape[0]
        device = mp_coeffs_corr.device
        exp_w = self.mp_to_ict_expression(mp_coeffs_corr)

        neutral = self.ict.neutral_mesh + self.template_offset.unsqueeze(0)
        verts_unposed = self.ict.forward(
            expression_weights=exp_w,
            to_canonical=False,
            apply_eyeball_rotation=False,
        )
        verts_unposed = verts_unposed - self.ict.neutral_mesh + neutral
        if expr_delta is not None:
            verts_unposed = verts_unposed + expr_delta

        if pose_rotation_6d is None:
            pose_rotation_6d = torch.zeros(B, 6, device=device, dtype=mp_coeffs_corr.dtype)
        if pose_translation is None:
            pose_translation = torch.zeros(B, 3, device=device, dtype=mp_coeffs_corr.dtype)

        R = rotation_6d_to_matrix(pose_rotation_6d)
        verts_posed = self.apply_weighted_pose(verts_unposed, R, pose_translation)
        pose_w = self.pose_weight_net(self.canonical_xyz)

        out = {
            "verts_posed": verts_posed,
            "expression_weights": exp_w,
            "pose_weight": pose_w,
        }
        if return_unposed:
            out["verts_unposed"] = verts_unposed
        return out
