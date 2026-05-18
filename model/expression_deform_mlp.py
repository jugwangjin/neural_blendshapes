"""
Support-gated per-AU expression correctives.

  V += sum_j C_eff[j] * gate_j * max_delta_j * tanh(raw_delta_j(x, emb_j))

No global MLP(coeffs) -> arbitrary [B,V,3] entanglement.
"""

import torch
import torch.nn as nn

from model.blendshape_support import build_mp_gates, precompute_expression_support


def charbonnier(x, eps=1e-3):
    return torch.sqrt(x * x + eps * eps)


class SupportGatedExpressionDeformer(nn.Module):
    def __init__(
        self,
        ict_facekit,
        region_weight,
        n_coeffs=52,
        hidden=64,
        delta_ratio=0.75,
        delta_floor=1e-4,
        support_quantile=0.95,
        support_lo=0.05,
        support_hi=0.20,
        dilate_rings=2,
    ):
        super().__init__()
        self.n_coeffs = n_coeffs
        self.delta_ratio = delta_ratio
        self.delta_floor = delta_floor

        mag_e, support_e = precompute_expression_support(
            ict_facekit,
            quantile=support_quantile,
            support_lo=support_lo,
            support_hi=support_hi,
            dilate_rings=dilate_rings,
        )
        gate, mag_mp, mp_to_ict = build_mp_gates(ict_facekit, region_weight, mag_e, support_e)

        self.register_buffer("gate", gate)
        self.register_buffer("mag", mag_mp)
        self.register_buffer("mp_to_ict", mp_to_ict)
        self.register_buffer(
            "canonical_xyz",
            ict_facekit.neutral_mesh_canonical[0].detach(),
        )

        self.au_embed = nn.Embedding(n_coeffs, hidden)
        self.mlp = nn.Sequential(
            nn.Linear(3 + hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 3),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.normal_(self.au_embed.weight, std=0.01)

    def _delta_one_au(self, j, c_eff_j):
        """c_eff_j [B]; returns [B, V, 3] corrective for AU j."""
        emb = self.au_embed.weight[j]
        inp = torch.cat(
            [self.canonical_xyz, emb.unsqueeze(0).expand(self.canonical_xyz.shape[0], -1)],
            dim=-1,
        )
        raw = self.mlp(inp)
        gate = self.gate[j]
        max_delta = self.delta_ratio * self.mag[j] + self.delta_floor * gate
        delta_j = gate.unsqueeze(-1) * max_delta.unsqueeze(-1) * torch.tanh(raw)
        return c_eff_j.view(-1, 1, 1) * delta_j.unsqueeze(0)

    def forward(self, c_eff, return_aux=False):
        """
        c_eff: [B, J] gamma-gated effective coefficients (active * C**gamma)
        """
        b = c_eff.shape[0]
        total = c_eff.new_zeros(b, self.canonical_xyz.shape[0], 3)
        raw_list = []
        for j in range(self.n_coeffs):
            if self.gate[j].max() < 1e-6:
                continue
            d_j = self._delta_one_au(j, c_eff[:, j])
            total = total + d_j
            if return_aux:
                emb = self.au_embed.weight[j]
                inp = torch.cat(
                    [
                        self.canonical_xyz,
                        emb.unsqueeze(0).expand(self.canonical_xyz.shape[0], -1),
                    ],
                    dim=-1,
                )
                raw_list.append(torch.tanh(self.mlp(inp)))

        if not return_aux:
            return total

        return total, {"raw_tanh": raw_list, "gate": self.gate}

    def regularization_loss(self, c_eff, c_raw):
        """Weak priors: neutral zero, support leakage, amplitude vs base mag."""
        losses = {}
        neutral_mask = c_raw.abs().sum(dim=-1) < 0.05
        if neutral_mask.any():
            losses["expr_neutral"] = self.forward(c_eff[neutral_mask]).pow(2).mean()
        else:
            losses["expr_neutral"] = c_eff.new_zeros(())

        leak = c_eff.new_zeros(())
        amp = c_eff.new_zeros(())
        n_terms = 0
        for j in range(self.n_coeffs):
            if self.gate[j].max() < 1e-6:
                continue
            emb = self.au_embed.weight[j]
            inp = torch.cat(
                [
                    self.canonical_xyz,
                    emb.unsqueeze(0).expand(self.canonical_xyz.shape[0], -1),
                ],
                dim=-1,
            )
            raw = torch.tanh(self.mlp(inp))
            outside = (1.0 - self.gate[j]).clamp(min=0.0)
            leak = leak + (outside.unsqueeze(-1) * raw).pow(2).mean()
            max_delta = self.delta_ratio * self.mag[j] + self.delta_floor * self.gate[j]
            amp = amp + charbonnier(raw / (max_delta.unsqueeze(-1) + 1e-6)).mean()
            n_terms += 1
        if n_terms > 0:
            leak = leak / n_terms
            amp = amp / n_terms
        losses["expr_leak"] = leak
        losses["expr_amp"] = amp
        return losses


# Backward-compatible alias
ExpressionDeformMLP = SupportGatedExpressionDeformer
