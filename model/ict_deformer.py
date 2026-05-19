"""
ICT FACS deformer: template MLP + MP coeffs → mesh + support-gated expression + pose weight.

Template / expression fields are smooth functions of canonical ``xyz`` on all skin +
eyeball vertices (same MLP input). Eye-socket orbit is not hard-gated; large deltas there
are penalized via ``deform_reg_weight``. Teeth stay hard-off (separate rigid part).
"""

import torch
import torch.nn as nn

from model.blendshape_support import build_mp_gates, precompute_expression_support
from model.expr_regions import (
    build_deform_reg_weight,
    build_expr_region_weight,
    build_teeth_mask,
)
from model.pose_weight import PoseWeightMLP
from utils.so3 import apply_rigid, rotation_6d_to_matrix


def charbonnier(x, eps=1e-3):
    return torch.sqrt(x * x + eps * eps)


class ICTDeformer(nn.Module):
    def __init__(
        self,
        ict_facekit,
        region_weight=None,
        *,
        template_hidden=128,
        max_template_delta=1e-2,
        n_coeffs=52,
        expr_hidden=64,
        delta_ratio=0.75,
        delta_floor=1e-4,
        support_quantile=0.95,
        support_lo=0.05,
        support_hi=0.20,
        dilate_rings=2,
    ):
        super().__init__()
        self.ict = ict_facekit
        self.n_coeffs = n_coeffs
        self.delta_ratio = delta_ratio
        self.delta_floor = delta_floor

        canonical = ict_facekit.neutral_mesh_canonical[0].detach()
        self.register_buffer("canonical_xyz", canonical)

        if region_weight is None:
            region_weight = build_expr_region_weight(ict_facekit)
        self.register_buffer("deform_reg_weight", build_deform_reg_weight(ict_facekit))
        self.register_buffer("teeth_mask", build_teeth_mask(ict_facekit))

        self.template_mlp = nn.Sequential(
            nn.Linear(3, template_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(template_hidden, template_hidden),
            nn.ReLU(inplace=True),
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
            quantile=support_quantile,
            support_lo=support_lo,
            support_hi=support_hi,
            dilate_rings=dilate_rings,
        )
        gate, mag_mp, mp_to_ict = build_mp_gates(ict_facekit, region_weight, mag_e, support_e)
        self.register_buffer("expr_gate", gate)
        self.register_buffer("expr_mag", mag_mp)
        self.register_buffer("mp_to_ict", mp_to_ict)

        self.expr_au_embed = nn.Embedding(n_coeffs, expr_hidden)
        self.expr_mlp = nn.Sequential(
            nn.Linear(3 + expr_hidden, expr_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(expr_hidden, 3),
        )
        nn.init.zeros_(self.expr_mlp[-1].weight)
        nn.init.zeros_(self.expr_mlp[-1].bias)
        nn.init.normal_(self.expr_au_embed.weight, std=0.01)

    def _apply_teeth_mask(self, delta):
        if self.teeth_mask.any():
            delta = delta.clone()
            delta[self.teeth_mask] = 0.0
        return delta

    def template_delta(self):
        """[V, 3] smooth identity corrective from canonical coords (all verts except teeth)."""
        raw = self.template_mlp(self.canonical_xyz)
        scale = torch.exp(self.log_max_template_delta)
        delta = scale * torch.tanh(raw)
        return self._apply_teeth_mask(delta)

    def weighted_delta_penalty(self, delta):
        """delta [B,V,3] or [V,3] — region-weighted L2 (eye socket high)."""
        if delta.ndim == 2:
            delta = delta.unsqueeze(0)
        w = self.deform_reg_weight.unsqueeze(0).unsqueeze(-1)
        return (w * delta.pow(2)).mean()

    def template_regularization_loss(self):
        return self.weighted_delta_penalty(self.template_delta())

    def mp_to_ict_expression(self, mp_coeffs):
        """mp_coeffs [B, 52] -> ICT expression [B, num_expression]."""
        B = mp_coeffs.shape[0]
        n_exp = self.ict.num_expression
        out = torch.zeros(B, n_exp, device=mp_coeffs.device, dtype=mp_coeffs.dtype)
        idx = self.ict.mediapipe_to_ict
        n = min(len(idx), mp_coeffs.shape[1])
        out[:, idx[:n]] = mp_coeffs[:, :n]
        return out

    def _expr_delta_one_au(self, j, c_eff_j):
        emb = self.expr_au_embed.weight[j]
        inp = torch.cat(
            [self.canonical_xyz, emb.unsqueeze(0).expand(self.canonical_xyz.shape[0], -1)],
            dim=-1,
        )
        raw = self.expr_mlp(inp)
        gate = self.expr_gate[j]
        max_delta = self.delta_ratio * self.expr_mag[j] + self.delta_floor * gate
        delta_j = gate.unsqueeze(-1) * max_delta.unsqueeze(-1) * torch.tanh(raw)
        return c_eff_j.view(-1, 1, 1) * delta_j.unsqueeze(0)

    def expression_delta(self, c_eff, return_aux=False):
        """
        c_eff: [B, J] gamma-gated effective coefficients (active * C**gamma)
        """
        b = c_eff.shape[0]
        total = c_eff.new_zeros(b, self.canonical_xyz.shape[0], 3)
        raw_list = []
        for j in range(self.n_coeffs):
            if self.expr_gate[j].max() < 1e-6:
                continue
            d_j = self._expr_delta_one_au(j, c_eff[:, j])
            total = total + d_j
            if return_aux:
                emb = self.expr_au_embed.weight[j]
                inp = torch.cat(
                    [
                        self.canonical_xyz,
                        emb.unsqueeze(0).expand(self.canonical_xyz.shape[0], -1),
                    ],
                    dim=-1,
                )
                raw_list.append(torch.tanh(self.expr_mlp(inp)))

        if not return_aux:
            return total

        return total, {"raw_tanh": raw_list, "gate": self.expr_gate}

    def regularization_loss(self, c_eff, c_raw, expr_delta=None):
        """Weak priors: neutral zero, support leakage, amplitude, socket-weighted expr."""
        losses = {}
        neutral_mask = c_raw.abs().sum(dim=-1) < 0.05
        if neutral_mask.any():
            losses["expr_neutral"] = self.expression_delta(c_eff[neutral_mask]).pow(2).mean()
        else:
            losses["expr_neutral"] = c_eff.new_zeros(())

        leak = c_eff.new_zeros(())
        amp = c_eff.new_zeros(())
        n_terms = 0
        for j in range(self.n_coeffs):
            if self.expr_gate[j].max() < 1e-6:
                continue
            emb = self.expr_au_embed.weight[j]
            inp = torch.cat(
                [
                    self.canonical_xyz,
                    emb.unsqueeze(0).expand(self.canonical_xyz.shape[0], -1),
                ],
                dim=-1,
            )
            raw = torch.tanh(self.expr_mlp(inp))
            outside = (1.0 - self.expr_gate[j]).clamp(min=0.0)
            leak = leak + (outside.unsqueeze(-1) * raw).pow(2).mean()
            max_delta = self.delta_ratio * self.expr_mag[j] + self.delta_floor * self.expr_gate[j]
            amp = amp + charbonnier(raw / (max_delta.unsqueeze(-1) + 1e-6)).mean()
            n_terms += 1
        if n_terms > 0:
            leak = leak / n_terms
            amp = amp / n_terms
        losses["expr_leak"] = leak
        losses["expr_amp"] = amp
        if expr_delta is not None:
            losses["expr_socket"] = self.weighted_delta_penalty(expr_delta)
        else:
            losses["expr_socket"] = c_eff.new_zeros(())
        return losses

    def apply_weighted_pose(self, verts, R, t, scale=None):
        """verts [B,V,3], R [B,3,3], t [B,3], optional uniform scale."""
        w = self.pose_weight_net(self.canonical_xyz)
        rigid = apply_rigid(verts, R, t, scale=scale)
        return (1.0 - w.unsqueeze(0)) * verts + w.unsqueeze(0) * rigid

    def forward(
        self,
        mp_coeffs_corr,
        pose_rotation_6d=None,
        pose_translation=None,
        pose_scale=None,
        c_eff=None,
        expr_delta=None,
        apply_expression_deform=True,
        return_unposed=False,
    ):
        """
        mp_coeffs_corr: [B, 52]
        c_eff: [B, J] for support-gated expression (optional if ``expr_delta`` set)
        pose_rotation_6d: [B, 6] residual (optional)
        pose_translation: [B, 3] residual (optional)
        """
        B = mp_coeffs_corr.shape[0]
        device = mp_coeffs_corr.device
        exp_w = self.mp_to_ict_expression(mp_coeffs_corr)

        tpl = self.template_delta()
        neutral = self.ict.neutral_mesh + tpl.unsqueeze(0)
        verts_unposed = self.ict.forward(
            expression_weights=exp_w,
            to_canonical=False,
            apply_eyeball_rotation=False,
        )
        verts_unposed = verts_unposed - self.ict.neutral_mesh + neutral

        if expr_delta is None and apply_expression_deform and c_eff is not None:
            expr_delta = self.expression_delta(c_eff)
        if expr_delta is not None:
            verts_unposed = verts_unposed + expr_delta

        if pose_rotation_6d is None:
            pose_rotation_6d = torch.zeros(B, 6, device=device, dtype=mp_coeffs_corr.dtype)
        if pose_translation is None:
            pose_translation = torch.zeros(B, 3, device=device, dtype=mp_coeffs_corr.dtype)

        R = rotation_6d_to_matrix(pose_rotation_6d)
        verts_posed = self.apply_weighted_pose(
            verts_unposed, R, pose_translation, scale=pose_scale
        )
        pose_w = self.pose_weight_net(self.canonical_xyz)

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
