"""Hydrate per-stage loss config from StageSpec + global Config."""


def hydrate_stage_loss_cfg(loss_cfg, spec, cfg):
    """Copy shared loss fields from ``cfg`` / ``spec`` into ``loss_cfg`` (mutates in place)."""
    loss_cfg.image_size = cfg.image_size
    loss_cfg.mp_lmk_iris_weight = cfg.mp_lmk_iris_weight
    loss_cfg.mp_lmk_mouth_weight = cfg.mp_lmk_mouth_weight
    loss_cfg.silhouette_l1 = getattr(spec, "silhouette_l1", getattr(cfg, "silhouette_l1", False))
    loss_cfg.seg_l1 = getattr(spec, "seg_l1", getattr(cfg, "seg_l1", False))
    loss_cfg.silhouette_use_edt = cfg.silhouette_use_edt
    loss_cfg.mesh_silhouette_use_edt = cfg.mesh_silhouette_use_edt
    loss_cfg.silhouette_edt_w_ext = cfg.silhouette_edt_w_ext
    loss_cfg.silhouette_edt_w_int = cfg.silhouette_edt_w_int
    loss_cfg.silhouette_edt_max_dist_px = cfg.silhouette_edt_max_dist_px
    loss_cfg.mesh_silhouette_cull_backfaces = cfg.mesh_silhouette_cull_backfaces
    loss_cfg.mesh_silhouette_cull_flip = cfg.mesh_silhouette_cull_flip
    loss_cfg.rgb_ssim_lambda = getattr(spec, "rgb_ssim_lambda", cfg.rgb_ssim_lambda)
    loss_cfg.lmk_distance_metric = getattr(spec, "lmk_distance_metric", cfg.lmk_distance_metric)
    loss_cfg.lmk_charbonnier_eps = getattr(spec, "lmk_charbonnier_eps", cfg.lmk_charbonnier_eps)
    loss_cfg.lmk_wing_w_px = getattr(spec, "lmk_wing_w_px", cfg.lmk_wing_w_px)
    loss_cfg.lmk_wing_eps_px = getattr(spec, "lmk_wing_eps_px", cfg.lmk_wing_eps_px)
    return loss_cfg


def update_stage_local_loss_weights(loss_cfg, spec, stage_local: int):
    """Per-step loss weight tweaks (LPIPS schedule, mesh seg cutoff)."""
    if hasattr(spec, "w_lpips"):
        lpips_start = int(getattr(spec, "lpips_start_local", 0))
        loss_cfg.w_lpips = float(spec.w_lpips) if stage_local >= lpips_start else 0.0
    mesh_seg_stop = int(getattr(spec, "mesh_seg_stop_local", 0))
    w_mesh_seg = float(getattr(spec, "w_mesh_seg", 0.0))
    if mesh_seg_stop > 0 and stage_local > mesh_seg_stop:
        w_mesh_seg = 0.0
    loss_cfg.w_mesh_seg = w_mesh_seg
    return loss_cfg
