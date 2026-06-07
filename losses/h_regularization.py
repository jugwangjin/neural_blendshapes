"""
Image-space h regularization (FLARE semantic + legacy 8-ch segmentation).

Each surface Gaussian has scalar ``h`` (meters): world position is
``xyz = barycentric_point(mesh) + h * vertex_normal``.  **h ≈ 0** → splat on mesh.

Penalizes ``accum_h = Σ α_i h_i`` per pixel: tier weight × ``accum_h²`` (no σ).

Sources:
  - ``h_reg_label_eye_occlusion``: FLARE part ids 4/5 only (label path).
  - ``h_reg_seg_face`` / ``h_reg_seg_mouth``: semantic8 ch1 / ch5 (mesh-stick).
  - ``h_reg_seg_neck|hair|glasses|misc``: part ids for loose tiers.

Cloth (part 16) is intentionally omitted — was never supervised (tight matting excluded it).
"""

import torch


def loss_h_image_space(accum_h, alpha, batch, cfg, _get_w):
    """
    Pixel loss on rendered ``accum_h`` (surface Gaussians only).

    Per tier: ``h_w_* * accum_h²`` (masked mean). Mesh-stick vs loose = weight only.
    """
    accum = accum_h[:, 0]
    a = alpha[:, 0]
    r2 = accum * accum

    m_eye = batch["h_reg_label_eye_occlusion"][:, 0].float()
    m_face = batch["h_reg_seg_face"][:, 0].float()
    m_mouth = batch["h_reg_seg_mouth"][:, 0].float()
    m_neck = batch["h_reg_seg_neck"][:, 0].float()
    m_hair = batch["h_reg_seg_hair"][:, 0].float()
    m_glasses = batch["h_reg_seg_glasses"][:, 0].float()
    m_misc = batch["h_reg_seg_misc"][:, 0].float()

    w_skin = _get_w(cfg, "h_w_skin", 1.0)
    w_nose = _get_w(cfg, "h_w_nose", 1.0)
    w_eye = _get_w(cfg, "h_w_eye", 1.0)
    w_brow = _get_w(cfg, "h_w_brow", 1.0)
    w_mouth = _get_w(cfg, "h_w_mouth", 1.0)
    w_neck = _get_w(cfg, "h_w_neck", 1.0)
    w_misc = _get_w(cfg, "h_w_misc", 1.0)
    w_hair = _get_w(cfg, "h_w_hair", 0.015)
    w_glasses = _get_w(cfg, "h_w_glasses", 0.006)

    w_face_mesh = w_skin + w_nose + w_brow

    per_pix = (
        m_face * w_face_mesh * r2
        + m_eye * w_eye * r2
        + m_mouth * w_mouth * r2
        + m_neck * w_neck * r2
        + m_misc * w_misc * r2
        + m_hair * w_hair * r2
        + m_glasses * w_glasses * r2
    )

    alpha_min = _get_w(cfg, "h_alpha_min", 0.08)
    mask = a * (a > alpha_min).float()
    if batch.get("mask") is not None:
        mask = mask * batch["mask"][:, 0].float()

    region = (
        m_face
        + m_eye
        + m_mouth
        + m_neck
        + m_misc
        + m_hair
        + m_glasses
    ).clamp(max=1.0)
    w = mask * region
    return (per_pix * w).sum() / w.sum().clamp(min=1e-6)
