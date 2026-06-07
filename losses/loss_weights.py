"""Loss term table and weighted sum over loss dict."""

import torch

from losses.cfg_access import get_loss_weight

__all__ = [
    "LOSS_TERM_WEIGHTS",
    "aggregate_weighted_losses",
    "get_loss_weight",
    "silhouette_weight",
]


def silhouette_weight(cfg):
    w = get_loss_weight(cfg, "w_silhouette", 0.0)
    if w <= 0.0:
        w = get_loss_weight(cfg, "w_mask", 0.0)
    if w <= 0.0:
        w = get_loss_weight(cfg, "w_mp_mask", 0.0)
    return w


LOSS_TERM_WEIGHTS = [
    ("rgb", "w_rgb"),
    ("normal", "w_normal"),
    ("silhouette", None),
    ("mesh_silhouette", "w_mesh_silhouette"),
    ("mesh_semantic", "w_mesh_seg"),
    ("mp_lmk", "w_mp_lmk"),
    ("pie68_jaw", "w_pie68_jaw"),
    ("h", "w_h"),
    ("geometry", "w_geometry"),
    ("scaling", "w_scaling"),
    ("color_expr_sparse", "w_color_expr_sparse"),
    ("color_expr_per_coeff", "w_color_expr_group_sparse"),
    ("color_expr_per_gaussian", "w_color_expr_per_gaussian"),
    ("opacity", "w_opacity"),
    ("opacity_loose", "w_opacity_loose"),
    ("opacity_headneck", "w_opacity_headneck"),
    ("opacity_decay", "w_opacity_decay"),
    ("gamma_prior", "w_gamma_prior"),
    ("pose_prior", "w_pose_prior"),
    ("pose_tz", "w_pose_tz"),
    ("expr_deform_reg", "w_expr_deform_reg"),
    ("seg", "w_seg"),
    ("lip_mouth_leak", "w_lip_mouth_leak"),
    ("face_region", "w_face_region"),
    ("sparsity", "lambda_sparsity"),
    ("lpips", "w_lpips"),
    ("template_smooth", None),
    ("template_laplacian", "w_template_laplacian"),
    ("template_scale", "w_template_scale_prior"),
    ("identity_prior", "w_identity_prior"),
    ("identity_smooth", "w_identity_smooth"),
]


def aggregate_weighted_losses(losses, cfg, *, ref_tensor, w_silhouette=None):
    """
    Sum weighted loss dict entries into ``losses['total']``.

    ``w_silhouette`` overrides silhouette weight when provided (alias w_mask / w_mp_mask).
    """
    w_tpl = get_loss_weight(cfg, "w_template_smooth", get_loss_weight(cfg, "w_identity_smooth", 0.0))
    w_tpl_lap = get_loss_weight(cfg, "w_template_laplacian", 0.0)
    w_tpl_scale = get_loss_weight(cfg, "w_template_scale_prior", 0.0)
    w_sil = silhouette_weight(cfg) if w_silhouette is None else float(w_silhouette)

    total = torch.zeros((), device=ref_tensor.device, dtype=ref_tensor.dtype)
    for key, wkey in LOSS_TERM_WEIGHTS:
        if key not in losses:
            continue
        if key == "silhouette":
            w = w_sil
        elif key == "template_smooth":
            w = w_tpl
        elif key == "template_laplacian":
            w = w_tpl_lap
        elif key == "template_scale":
            w = w_tpl_scale
        elif key == "identity_smooth":
            w = get_loss_weight(cfg, "w_identity_smooth", 0.0)
        else:
            w = get_loss_weight(cfg, wkey, 0.0)
        total = total + w * losses[key]
    losses["total"] = total
    return losses
