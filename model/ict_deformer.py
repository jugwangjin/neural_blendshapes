"""
ICT FACS deformer: template MLP + MP coeffs → mesh + region-gated expression + pose weight.

Template / expression MLPs take jaw-open ICT in FLAME space (``ict.canonical`` =
jawOpen + ``flame_alignment_s,R,T`` / ``flame_similarity_s,T`` from npy; NICP bake mesh is
not used as template). ``expr_gate`` limits
which vertices each AU can move (eye occlusion on; teeth / lashes / lacrimal off).
"""

import torch
import torch.nn as nn

from model.blendshape_support import build_mp_gates, precompute_expression_support
from model.expr_regions import build_deform_reg_weight, build_expr_region_weight
from model.pose_weight import PoseWeightMLP
from utils.camera import rotation_matrix_y_deg
from utils.mediapipe_blendshapes import load_mediapipe_mapping, mp_to_ict_expression_weights
from utils.mesh_ops import apply_rigid, apply_rigid_about_centroid, rotation_6d_to_matrix


class ICTDeformer(nn.Module):
    def __init__(
        self,
        ict_facekit,
        region_weight=None,
        *,
        mediapipe_name_to_ict="./assets/mediapipe_name_to_indices.pkl",
        template_hidden=128,
        max_template_delta=1e-2,
        n_coeffs=53,
        expr_hidden=128,
        delta_ratio=0.75,
        delta_floor=1e-4,
        gate_alpha=0.1,         # Normalized Soft Mask alpha threshold (default 0.1)
        dilate_rings=2,
        **kwargs,               # Absorb deprecated support_quantile, support_lo, support_hi
    ):
        super().__init__()
        self.ict = ict_facekit
        self.n_coeffs = n_coeffs
        self.delta_ratio = delta_ratio
        self.delta_floor = delta_floor

        canonical = ict_facekit.canonical[0].detach()
        self.register_buffer("canonical_xyz", canonical)

        if region_weight is None:
            region_weight = build_expr_region_weight(ict_facekit)
        region_weight = region_weight.to(device=ict_facekit.canonical.device)
        self.register_buffer("deform_reg_weight", build_deform_reg_weight(ict_facekit))

        self.template_mlp = nn.Sequential(
            nn.Linear(3, template_hidden),
            nn.Softplus(beta=100),
            nn.Linear(template_hidden, template_hidden),
            nn.Softplus(beta=100),
            nn.Linear(template_hidden, template_hidden),
            nn.Softplus(beta=100),
            nn.Linear(template_hidden, template_hidden),
            nn.Softplus(beta=100),
            nn.Linear(template_hidden, 3),
        )
        nn.init.zeros_(self.template_mlp[-1].weight)
        nn.init.zeros_(self.template_mlp[-1].bias)
        self.log_max_template_delta = nn.Parameter(
            torch.log(torch.tensor(float(max_template_delta)))
        )

        self.pose_weight_net = PoseWeightMLP()

        mag_e, support_e = precompute_expression_support(
            ict_facekit,
            alpha=gate_alpha,
            dilate_rings=dilate_rings,
        )
        if getattr(ict_facekit, "mediapipe_to_ict", None) is not None:
            mp_to_ict = ict_facekit.mediapipe_to_ict
        else:
            mp_to_ict = load_mediapipe_mapping(
                mediapipe_name_to_ict, num_expression=ict_facekit.num_expression
            ).mediapipe_to_ict
        gate, mag_mp = build_mp_gates(ict_facekit, region_weight, mag_e, support_e, mp_to_ict)
        self.register_buffer("mp_to_ict", mp_to_ict)
        self.register_buffer("expr_gate", gate)
        self.register_buffer("expr_mag", mag_mp)

        self.expr_mlp = nn.Sequential(
            nn.Linear(3, expr_hidden),
            nn.Softplus(beta=100),
            nn.Linear(expr_hidden, expr_hidden),
            nn.Softplus(beta=100),
            nn.Linear(expr_hidden, expr_hidden),
            nn.Softplus(beta=100),
            nn.Linear(expr_hidden, expr_hidden),
            nn.Softplus(beta=100),
            nn.Linear(expr_hidden, n_coeffs * 3, bias=False),
        )
        nn.init.zeros_(self.expr_mlp[-1].weight)

    def template_delta(self):
        """[V, 3] smooth identity corrective from canonical coords."""
        raw = self.template_mlp(self.canonical_xyz)
        scale = torch.exp(self.log_max_template_delta)
        return scale * torch.tanh(raw)

    def mp_to_ict_expression(self, mp_coeffs):
        return mp_to_ict_expression_weights(
            mp_coeffs, self.mp_to_ict, self.ict.num_expression
        )

    def expression_raw_tanh(self):
        """[J, V, 3] channel-wise AU basis (before gate × magnitude × c_eff)."""
        j = self.n_coeffs
        raw = torch.tanh(self.expr_mlp(self.canonical_xyz))
        raw = raw.reshape(-1, j, 3).permute(1, 0, 2).contiguous()
        return raw, self.expr_gate

    def expression_delta_basis(self):
        """[J, V, 3] per-AU displacement field (region-gated, unit c_eff=1)."""
        raw, gate = self.expression_raw_tanh()
        max_delta = self.delta_ratio * self.expr_mag + self.delta_floor * gate
        return gate.unsqueeze(-1) * max_delta.unsqueeze(-1) * raw

    def expression_delta(self, c_eff, basis=None, return_aux=False):
        """
        c_eff: [B, J] gamma-corrected ICT expression coefficients
        basis: optional cached [J, V, 3] expression basis
        """
        if basis is None:
            basis = self.expression_delta_basis()
        active = self.expr_gate.amax(dim=1) >= 1e-6
        basis = basis * active.to(basis.dtype).view(-1, 1, 1)
        total = torch.einsum("bj,jvd->bvd", c_eff, basis)
        if not return_aux:
            return total
        raw, gate = self.expression_raw_tanh()
        return total, {"raw_tanh": raw, "gate": gate}

    def apply_weighted_pose(
        self,
        verts,
        R,
        t,
        scale=None,
        pose_weight_fixed=None,
        pose_w=None,
        rotate_about_centroid=False,
    ):
        """verts [B,V,3], R [B,3,3], t [B,3]; optional uniform scale (usually disabled)."""
        if pose_w is None:
            if pose_weight_fixed is not None:
                w = torch.full(
                    (self.canonical_xyz.shape[0], 1),
                    float(pose_weight_fixed),
                    device=verts.device,
                    dtype=verts.dtype,
                )
            else:
                w = self.pose_weight_net(self.canonical_xyz)
        else:
            w = pose_w
        if rotate_about_centroid:
            rigid = apply_rigid_about_centroid(verts, R, t)
        else:
            rigid = apply_rigid(verts, R, t, scale=scale)
        return (1.0 - w.unsqueeze(0)) * verts + w.unsqueeze(0) * rigid, w

    # NOTE: only for debug
    def apply_head_yaw(self, verts, yaw_deg, pivot=None):
        """
        Rotate mesh about world +Y (constant camera → head/mesh orbit).

        Same row-vector convention as ``utils.camera.rotate_points_y``.
        """
        if float(yaw_deg) == 0.0:
            return verts
        if pivot is None:
            pivot = self.canonical_xyz.mean(dim=0)
        pivot = pivot.reshape(3).to(device=verts.device, dtype=verts.dtype)
        B = verts.shape[0]
        R = rotation_matrix_y_deg(float(yaw_deg), device=verts.device, dtype=verts.dtype)
        t = (pivot - pivot @ R.T).unsqueeze(0).expand(B, -1)
        R_b = R.unsqueeze(0).expand(B, -1, -1)
        return apply_rigid(verts, R_b, t)

    def forward(
        self,
        mp_coeffs_corr,
        pose_rotation_6d=None,
        pose_translation=None,
        pose_scale=None,
        c_eff=None,
        expr_delta=None,
        expression_basis=None,
        apply_expression_deform=True,
        return_unposed=False,
        expression_weights=None,
        apply_flame_similarity=True,
        head_yaw_deg=None,
        pose_weight_fixed=None,
        rotate_about_centroid=False,
        pose_zero_tz=False,
    ):
        """
        mp_coeffs_corr: [B, 53] ICT expression coeffs (gamma-corrected); MP input remains [B, 52]
        c_eff: [B, J] per MP channel for region-gated neural delta
        expression_basis: optional cached [J,V,3] basis from expression_delta_basis()
        pose_rotation_6d: [B, 6] residual (optional)
        pose_translation: [B, 3] residual (optional)
        expression_weights: optional [B, num_expression] (bypasses MP gather)
        head_yaw_deg: orbit mesh about world +Y (constant-camera sanity / debug)
        pose_weight_fixed: if set, use this w(x) instead of PoseWeightMLP (e.g. 1.0)
        rotate_about_centroid: rotate about mesh centroid (not world origin)
        pose_zero_tz: if True, zero translation z (deprecated for train — use tz + perspective)
        """
        B = mp_coeffs_corr.shape[0]
        device = mp_coeffs_corr.device
        if expression_weights is not None:
            exp_w = expression_weights
        else:
            exp_w = self.mp_to_ict_expression(mp_coeffs_corr)

        tpl = self.template_delta()
        verts_unposed = self.ict.forward(
            expression_weights=exp_w,
            to_canonical=False,
            apply_eyeball_rotation=False,
            apply_flame_similarity=apply_flame_similarity,
        )
        verts_unposed = verts_unposed + tpl.unsqueeze(0)

        if expr_delta is None and apply_expression_deform and c_eff is not None:
            expr_delta = self.expression_delta(c_eff, basis=expression_basis)
        if expr_delta is not None:
            verts_unposed = verts_unposed + expr_delta

        if pose_rotation_6d is None:
            pose_rotation_6d = torch.zeros(B, 6, device=device, dtype=mp_coeffs_corr.dtype)
        if pose_translation is None:
            pose_translation = torch.zeros(B, 3, device=device, dtype=mp_coeffs_corr.dtype)

        R = rotation_6d_to_matrix(pose_rotation_6d)
        t_pose = pose_translation
        if pose_zero_tz:
            t_pose = t_pose.clone()
            t_pose[..., 2] = 0.0

        if pose_weight_fixed is not None:
            pose_w = torch.full(
                (self.canonical_xyz.shape[0], 1),
                float(pose_weight_fixed),
                device=device,
                dtype=verts_unposed.dtype,
            )
        else:
            pose_w = self.pose_weight_net(self.canonical_xyz)

        verts_posed, _ = self.apply_weighted_pose(
            verts_unposed,
            R,
            t_pose,
            scale=pose_scale,
            pose_weight_fixed=pose_weight_fixed,
            pose_w=pose_w,
            rotate_about_centroid=rotate_about_centroid,
        )
        if head_yaw_deg is not None and float(head_yaw_deg) != 0.0:
            verts_posed = self.apply_head_yaw(verts_posed, head_yaw_deg)

        out = {
            "verts_posed": verts_posed,
            "expression_weights": exp_w,
            "pose_weight": pose_w,
            "template_delta": tpl,
        }
        if expr_delta is not None:
            out["expr_delta"] = expr_delta
        if return_unposed:
            out["verts_unposed"] = verts_unposed
        return out
