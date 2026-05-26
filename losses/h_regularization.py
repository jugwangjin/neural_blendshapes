"""
Image-space h (distance) regularization — GT part masks + alpha-weighted h render.

No Gaussian ``sem_prob`` or semantic rendering required for ``w_h``.
"""

import torch


def charbonnier(x, eps=1e-3):
    return torch.sqrt(x * x + eps * eps)


def loss_h_image_space(accum_h, alpha, batch, cfg, _get_w):
    """
    Pixel loss on ED-style ``accum_h = Σ w_i h_i`` (surface Gaussians only).

    GT tiers (mutually exclusive part ids in ``flare_semantic``):
      skin / eye (strong) > brow > mouth > misc (weakest)
    """
    accum = accum_h[:, 0]
    a = alpha[:, 0]

    m_skin = batch["h_reg_skin"][:, 0].float()
    m_eye = batch["h_reg_eye"][:, 0].float()
    m_brow = batch["h_reg_brow"][:, 0].float()
    m_misc = batch["h_reg_misc"][:, 0].float()
    m_mouth = batch["h_reg_mouth"][:, 0].float()

    w_skin = _get_w(cfg, "h_w_skin", 1.0)
    w_eye = _get_w(cfg, "h_w_eye", 1.0)
    w_brow = _get_w(cfg, "h_w_brow", 0.45)
    w_misc = _get_w(cfg, "h_w_misc", 0.12)
    w_mouth = _get_w(cfg, "h_w_mouth", 0.28)

    s_skin = _get_w(cfg, "h_skin_sigma", 0.002)
    s_brow = _get_w(cfg, "h_sigma_brow", 0.004)
    s_misc = _get_w(cfg, "h_sigma_misc", 0.010)
    s_mouth = _get_w(cfg, "h_sigma_mouth", 0.008)

    r = accum.abs()
    per_pix = (
        m_skin * w_skin * charbonnier(r / s_skin)
        + m_eye * w_eye * charbonnier(r / s_skin)
        + m_brow * w_brow * charbonnier(r / s_brow)
        + m_misc * w_misc * charbonnier(r / s_misc)
        + m_mouth * w_mouth * charbonnier(r / s_mouth)
    )

    alpha_min = _get_w(cfg, "h_alpha_min", 0.08)
    mask = a * (a > alpha_min).float()
    if batch.get("mask") is not None:
        mask = mask * batch["mask"][:, 0].float()

    region = (m_skin + m_eye + m_brow + m_misc + m_mouth).clamp(max=1.0)
    w = mask * region
    return (per_pix * w).sum() / w.sum().clamp(min=1e-6)
