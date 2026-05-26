"""Regularization for ICTDeformer (kept outside the model module)."""

import torch


def charbonnier(x, eps=1e-3):
    return torch.sqrt(x * x + eps * eps)


def weighted_delta_penalty(delta, deform_reg_weight):
    """delta [B,V,3] or [V,3] — per-vertex L2 weighted by region (eye socket high)."""
    if delta.ndim == 2:
        delta = delta.unsqueeze(0)
    w = deform_reg_weight.unsqueeze(0).unsqueeze(-1)
    return (w * delta.pow(2)).mean()


def template_smooth_loss(deformer):
    return weighted_delta_penalty(deformer.template_delta(), deformer.deform_reg_weight)


def deformer_regularization_loss(deformer, c_eff, c_raw, expr_delta=None):
    """Weak priors: neutral zero, support leakage, amplitude, socket-weighted expr."""
    losses = {}
    neutral_mask = c_raw.abs().sum(dim=-1) < 0.05
    if neutral_mask.any():
        losses["expr_neutral"] = deformer.expression_delta(c_eff[neutral_mask]).pow(2).mean()
    else:
        losses["expr_neutral"] = c_eff.new_zeros(())

    raw, gate = deformer.expression_raw_tanh()
    active = gate.amax(dim=1) >= 1e-6
    if not active.any():
        losses["expr_leak"] = c_eff.new_zeros(())
        losses["expr_amp"] = c_eff.new_zeros(())
    else:
        raw_a = raw[active]
        gate_a = gate[active]
        outside = (1.0 - gate_a).clamp(min=0.0)
        losses["expr_leak"] = (outside.unsqueeze(-1) * raw_a).pow(2).mean()
        max_delta = deformer.delta_ratio * deformer.expr_mag[active] + deformer.delta_floor * gate_a
        losses["expr_amp"] = charbonnier(raw_a / (max_delta.unsqueeze(-1) + 1e-6)).mean()

    if expr_delta is not None:
        losses["expr_socket"] = weighted_delta_penalty(expr_delta, deformer.deform_reg_weight)
    else:
        losses["expr_socket"] = c_eff.new_zeros(())
    return losses
