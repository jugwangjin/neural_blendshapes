"""Non-finite loss / raster-grad diagnostics (used by train.py and debug_nan.py)."""

from __future__ import annotations

import torch


def loss_weight(loss_cfg, key: str) -> float:
    if key == "silhouette":
        w = float(getattr(loss_cfg, "w_silhouette", 0.0))
        if w <= 0.0:
            w = float(getattr(loss_cfg, "w_mask", 0.0))
        if w <= 0.0:
            w = float(getattr(loss_cfg, "w_mp_mask", 0.0))
        return w
    if key == "template_smooth":
        return float(
            getattr(loss_cfg, "w_template_smooth", getattr(loss_cfg, "w_identity_smooth", 0.0))
        )
    if key == "identity_smooth":
        return float(getattr(loss_cfg, "w_identity_smooth", 0.0))
    mapping = {
        "rgb": "w_rgb",
        "mp_lmk": "w_mp_lmk",
        "pie68_jaw": "w_pie68_jaw",
        "h": "w_h",
        "geometry": "w_geometry",
        "scaling": "w_scaling",
        "opacity": "w_opacity",
        "opacity_decay": "w_opacity_decay",
        "gamma_prior": "w_gamma_prior",
        "pose_prior": "w_pose_prior",
        "pose_tz": "w_pose_tz",
        "expr_deform_reg": "w_expr_deform_reg",
        "sem_anchor": "w_sem_anchor",
        "seg": "w_seg",
    }
    wkey = mapping.get(key)
    return float(getattr(loss_cfg, wkey, 0.0)) if wkey else 0.0


def nonfinite_loss_keys(losses) -> list[str]:
    bad = []
    for key, val in losses.items():
        if key == "total" or not isinstance(val, torch.Tensor):
            continue
        if not torch.isfinite(val).all():
            bad.append(key)
    return bad


def format_loss_report(losses, loss_cfg) -> str:
    lines = []
    total = losses.get("total")
    total_s = "nan" if total is None else f"{total.item():.6e}"
    lines.append(f"total={total_s} finite={torch.isfinite(total).all().item() if isinstance(total, torch.Tensor) else 'n/a'}")
    for key in sorted(losses.keys()):
        if key == "total":
            continue
        val = losses[key]
        if not isinstance(val, torch.Tensor):
            continue
        w = loss_weight(loss_cfg, key)
        weighted = val * w
        lines.append(
            f"  {key}: raw={val.item():.6e} w={w:g} weighted={weighted.item():.6e} "
            f"finite={torch.isfinite(val).all().item()}"
        )
    return "\n".join(lines)


def probe_raster_loss_grads(losses, loss_cfg, tracker, deformer, avatar):
    """Per-loss backward probe; returns loss names whose weighted term yields non-finite grads."""
    probes = []
    if tracker.head_gamma.weight.requires_grad:
        probes.append(("tracker.head_gamma.weight", tracker.head_gamma.weight))
    if tracker.head_pose.weight.requires_grad:
        probes.append(("tracker.head_pose.weight", tracker.head_pose.weight))
    if tracker.log_pose_scale.requires_grad:
        probes.append(("tracker.log_pose_scale", tracker.log_pose_scale))
    if deformer.log_max_template_delta.requires_grad:
        probes.append(("deformer.log_max_template_delta", deformer.log_max_template_delta))
    if deformer.template_mlp[-1].weight.requires_grad:
        probes.append(("deformer.template_mlp[-1].weight", deformer.template_mlp[-1].weight))
    if avatar.surface.log_scale.requires_grad:
        probes.append(("avatar.surface.log_scale", avatar.surface.log_scale))
    if avatar.surface.opacity.requires_grad:
        probes.append(("avatar.surface.opacity", avatar.surface.opacity))
    if avatar.surface.color.requires_grad:
        probes.append(("avatar.surface.color", avatar.surface.color))

    probe_params = [p for _, p in probes]
    if not probe_params:
        return []

    bad_losses = []
    for key, val in sorted(losses.items()):
        if key == "total" or not isinstance(val, torch.Tensor):
            continue
        w = loss_weight(loss_cfg, key)
        if w == 0.0:
            continue
        weighted = val * w
        if (not weighted.requires_grad) or weighted.grad_fn is None:
            continue
        grads = torch.autograd.grad(
            weighted,
            probe_params,
            retain_graph=True,
            allow_unused=True,
        )
        any_bad = False
        bad_param_names = []
        for (name, p), g in zip(probes, grads):
            if g is None:
                continue
            if not torch.isfinite(g).all():
                any_bad = True
                bad_param_names.append(name)
        if any_bad:
            bad_losses.append(key)
            print(f"  [probe] loss {key} yields NaN/Inf grads on parameters: {bad_param_names}")
    return bad_losses


def report_nonfinite_training_step(
    *,
    global_step: int,
    stage_name: str,
    losses,
    loss_cfg,
    tracker,
    deformer,
    avatar,
    render,
    cfg,
    avatar_out,
    run_grad_probe: bool = True,
):
    bad_forward = nonfinite_loss_keys(losses)
    total = losses.get("total")
    total_bad = total is not None and isinstance(total, torch.Tensor) and not torch.isfinite(total).all()

    print(f"\n[train nonfinite] step={global_step} stage={stage_name}")
    print(
        f"  gsplat: rasterize_mode={getattr(cfg, 'gsplat_rasterize_mode', '?')} "
        f"packed={getattr(cfg, 'gsplat_packed', '?')}"
    )
    print(format_loss_report(losses, loss_cfg))

    mesh = avatar_out.get("mesh_xyz")
    if mesh is not None:
        print(f"  mesh_xyz finite={torch.isfinite(mesh).all().item()}")
    if render is not None:
        rgb = render.get("rgb")
        alpha = render.get("alpha")
        if rgb is not None:
            print(f"  render rgb finite={torch.isfinite(rgb).all().item()}")
        if alpha is not None:
            print(f"  render alpha finite={torch.isfinite(alpha).all().item()}")

    for name, p in [
        ("color", avatar.surface.color),
        ("opacity", avatar.surface.opacity),
        ("log_scale", avatar.surface.log_scale),
        ("h", avatar.surface.h),
    ]:
        print(f"  param {name} finite={torch.isfinite(p).all().item()} shape={tuple(p.shape)}")

    if run_grad_probe and total_bad and total is not None and total.requires_grad:
        bad_grad = probe_raster_loss_grads(losses, loss_cfg, tracker, deformer, avatar)
        print(f"  bad_loss_grads (per-term probe)={bad_grad}")

    return bad_forward, total_bad
