"""
Barycentric / UV+H space Gaussian densification and pruning strategy.
"""

import math
import torch


GAUSSIAN_PARAMETER_NAMES = (
    "h",
    "log_scale",
    "rotation",
    "opacity",
    "color",
    "color_pose",
    "color_expression",
    "bary_uv",
)
SPLIT_COPY_PARAMETER_NAMES = ("rotation", "opacity", "color", "color_pose", "color_expression")
SURFACE_BUFFER_NAMES = ("uv", "is_gum", "is_h_pin", "is_teeth", "face_region_code")


def _replace_surface_param(surf, p_name, new_param):
    """Replace a surface Parameter so ``avatar.state_dict()`` sees densified tensors."""
    if isinstance(getattr(surf, p_name, None), torch.nn.Parameter):
        surf.register_parameter(p_name, new_param)
    else:
        setattr(surf, p_name, new_param)


def _assert_surface_layout(surf):
    n = len(surf.h)
    assert surf.face_idx.shape[0] == n, f"face_idx {surf.face_idx.shape[0]} != h {n}"
    assert surf.color.shape[0] == n, f"color {surf.color.shape[0]} != h {n}"
    assert surf.n_gaussians == n, f"n_gaussians {surf.n_gaussians} != h {n}"


def _execute_prune(surf, optimizer, is_prune, ict_faces, ict, device, prune_stats):
    n_candidates = int(is_prune.sum().item())
    n_removed = 0
    if is_prune.any():
        n_removed = int(prune_surf(surf, optimizer, is_prune, ict_faces, ict))
    out = dict(prune_stats)
    out["candidates"] = n_candidates
    out["removed"] = n_removed
    return n_removed, out


def _format_densify_location(step, stage_local):
    parts = []
    if step is not None:
        parts.append(f"step={step}")
    if stage_local is not None:
        parts.append(f"local={stage_local}")
    return " ".join(parts)


def _print_densify_summary(
    tag,
    *,
    step,
    stage_local,
    n_start,
    n_after_grow,
    n_end,
    n_clone,
    n_split,
    prune_stats,
    grad_thr=None,
    extra="",
):
    n_added = n_after_grow - n_start
    n_removed = n_after_grow - n_end
    net = n_end - n_start
    loc = _format_densify_location(step, stage_local)
    grow_line = f"clone={n_clone} split={n_split} added=+{n_added}"
    if n_clone + n_split != n_added:
        grow_line += f" (expect +{n_clone + n_split})"
    ps = prune_stats
    prune_line = (
        f"removed={ps.get('removed', n_removed)} "
        f"candidates={ps.get('candidates', ps.get('total', 0))} "
        f"[opa={ps['opacity']} world={ps['world_scale']} screen={ps['screen_radius']} "
        f"face={ps['face_cap']}]"
    )
    tail = f"gaussians {n_start}->{n_after_grow}->{n_end} (net {net:+d})"
    if grad_thr is not None:
        if torch.is_tensor(grad_thr):
            tail += (
                f" grad_thr=[{float(grad_thr.min().item()):.6f},"
                f"{float(grad_thr.max().item()):.6f}]"
            )
        else:
            tail += f" grad_thr={float(grad_thr):.6f}"
    if extra:
        tail += f" | {extra}"
    print(f"[{tag}] {loc} | grow {grow_line} | prune {prune_line} | {tail}", flush=True)


def _radii_per_gaussian(radii, n_gaussian, device):
    """Per-Gaussian screen-space radius [N] (max over cameras if batched)."""
    if radii is None:
        return None
    r = radii.float()
    if r.ndim == 1:
        out = r[:n_gaussian]
    elif r.shape[-1] == n_gaussian:
        out = r.reshape(-1, n_gaussian).amax(dim=0)
    else:
        flat = r.reshape(-1)
        if flat.numel() < n_gaussian:
            return None
        out = flat[:n_gaussian]
    if out.shape[0] < n_gaussian:
        pad = torch.zeros(n_gaussian - out.shape[0], device=device, dtype=out.dtype)
        out = torch.cat([out, pad])
    return out[:n_gaussian]


def _per_face_gaussian_counts(face_idx, n_faces=None):
    """``scatter_add`` + index: ``counts[F]``, ``per_gaussian[N]`` (O(N))."""
    n = face_idx.shape[0]
    device = face_idx.device
    fidx = face_idx.long()
    if n_faces is None:
        n_faces = int(fidx.max().item()) + 1
    counts = torch.zeros(n_faces, device=device, dtype=torch.int32)
    counts.scatter_add_(0, fidx, torch.ones(n, device=device, dtype=torch.int32))
    per_gaussian = counts[fidx]
    return counts, per_gaussian


@torch.no_grad()
def _face_cap_prune_mask(face_idx, opacity, k):
    """
    On faces with more than ``k`` Gaussians, prune lowest-opacity extras.
    O(N log N) sort; no Python loop over faces.
    """
    n = face_idx.numel()
    device = face_idx.device
    if k <= 0 or n == 0:
        return torch.zeros(n, dtype=torch.bool, device=device)

    counts = torch.bincount(face_idx.long())
    if not (counts > k).any():
        return torch.zeros(n, dtype=torch.bool, device=device)

    scale = opacity.max().item() + 1e-6
    perm = torch.argsort(face_idx.float() * scale + opacity)
    sorted_faces = face_idx[perm]
    sorted_orig = perm

    face_change = torch.ones(n, dtype=torch.bool, device=device)
    face_change[1:] = sorted_faces[1:] != sorted_faces[:-1]
    seg_starts = torch.where(face_change)[0]
    seg_ids = face_change.cumsum(0) - 1
    rank = torch.arange(n, device=device) - seg_starts[seg_ids]

    excess = counts[sorted_faces.long()] - k
    to_prune_sorted = rank < excess

    prune = torch.zeros(n, dtype=torch.bool, device=device)
    prune[sorted_orig[to_prune_sorted]] = True
    return prune


def _visible_mask_per_gaussian(radii, n_gaussian, device):
    """Reduce gsplat ``radii`` ([N], [C, N], [1, N], …) to a bool mask [N]."""
    if radii is None:
        return torch.ones(n_gaussian, dtype=torch.bool, device=device)
    r = radii
    if r.ndim == 1:
        return r[:n_gaussian] > 0
    if r.shape[-1] == n_gaussian:
        return (r > 0).any(dim=tuple(range(r.ndim - 1)))
    flat = r.reshape(-1)
    if flat.numel() >= n_gaussian:
        return flat[:n_gaussian] > 0
    return torch.ones(n_gaussian, dtype=torch.bool, device=device)


def _viewspace_grad_norm_gb(grad):
    """
    GaussianBlendshapes / 3DGS ``add_densification_stats``:
    ``torch.norm(viewspace_point_tensor.grad[..., :2], dim=-1)``.
    """
    g = grad.float()
    if g.ndim == 3:
        if g.shape[0] == 1:
            g = g[0]
        else:
            return torch.norm(g[..., :2], dim=-1).sum(dim=0)
    if g.ndim == 2:
        return torch.norm(g[:, :2], dim=-1)
    raise ValueError(f"unexpected viewspace grad shape {tuple(g.shape)}")


def _tensor_grad_norm_per_gaussian(grad, n_gaussian: int):
    """L2 norm of a parameter grad per Gaussian ([N])."""
    if grad is None or grad.shape[0] != n_gaussian:
        return None
    g = grad.float()
    if g.ndim >= 2:
        return g.reshape(n_gaussian, -1).norm(dim=-1).flatten()
    if g.ndim == 1:
        return g.abs().flatten()
    return None


def _per_gaussian_color_grad_norm(color_grad, n_gaussian: int):
    """L2 norm of ``surface.color`` grad per Gaussian ([N] flat). Supports SH [N,S,3] and RGB [N,3]."""
    return _tensor_grad_norm_per_gaussian(color_grad, n_gaussian)


def _surface_color_grad_norm_for_densify(surf, n_gaussian: int):
    """
    Combined per-Gaussian color grow signal from all appearance params that feed render RGB.

    ``color`` (DC / RGB), ``color_pose``, ``color_expression`` — LPIPS/RGB/SSIM grads on the
    packed color path backprop to each leaf; we take the L2 norm per Gaussian per tensor,
    then ``sqrt(sum_i norm_i^2)`` so any branch can trigger grow.
    """
    parts = []
    base = _per_gaussian_color_grad_norm(surf.color.grad, n_gaussian)
    if base is not None:
        parts.append(base)
    if hasattr(surf, "color_pose") and surf.color_pose.requires_grad:
        pose = _tensor_grad_norm_per_gaussian(surf.color_pose.grad, n_gaussian)
        if pose is not None:
            parts.append(pose)
    if hasattr(surf, "color_expression") and surf.color_expression.requires_grad:
        expr = _tensor_grad_norm_per_gaussian(surf.color_expression.grad, n_gaussian)
        if expr is not None:
            parts.append(expr)
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    stacked = torch.stack(parts, dim=0)
    return torch.linalg.vector_norm(stacked, dim=0)


def _grow_grad_pixel_scale(cfg):
    """gsplat ``means2d`` grads are ~O(1/W) vs classic 3DGS pixel-space thresholds."""
    if not bool(getattr(cfg, "gaussian_grow_grad2d_pixel_scale", True)):
        return 1.0
    sz = float(getattr(cfg, "image_size", 512))
    return 0.5 * sz


def update_optimizer_param(optimizer, old_param, new_param, optimizer_fn):
    """
    Updates the optimizer's param_group and state when a parameter is resized/re-allocated.
    """
    if optimizer is None:
        return False

    for group in optimizer.param_groups:
        for idx, param in enumerate(group["params"]):
            if param is not old_param:
                continue

            param_state = optimizer.state.pop(old_param, None)
            if param_state is not None:
                for key in list(param_state.keys()):
                    if key != "step" and torch.is_tensor(param_state[key]):
                        param_state[key] = optimizer_fn(key, param_state[key])
                optimizer.state[new_param] = param_state
            group["params"][idx] = new_param
            return True
    return False


def _repeat_selected(value, index, repeats=2):
    repeat_shape = (repeats, *([1] * (value.ndim - 1)))
    return value[index].repeat(*repeat_shape)


def template_face_areas(ict, device):
    from model.mesh_gaussian_pose import calc_face_areas

    verts = ict.template_reference_verts().to(device=device)
    faces = ict.faces.to(device=device)
    return calc_face_areas(verts, faces).reshape(-1)


def resolve_bary_noise_base_std(face_areas, cfg, parent_scale_ref=None) -> float:
    """
    Reference bary std at median face and ``parent_scale_ref`` (default percent_dense×extent).
    Used for logging / manual-mode calibration only when gb_match splits use per-parent scale.
    """
    if parent_scale_ref is None:
        extent = float(getattr(cfg, "gaussian_scene_extent", 1.0))
        pct = float(getattr(cfg, "gaussian_percent_dense", 0.0025))
        parent_scale_ref = pct * extent
    a_med = float(face_areas.median().item())
    char_len = math.sqrt(2.0 * max(a_med, 1e-12))
    return float(parent_scale_ref) / (char_len * math.sqrt(2.0) + 1e-12)


def per_gaussian_bary_noise_std(face_idx, face_areas, cfg, parent_scale, base_std=None):
    """
    GB split: world displacement std ~ parent max scale (rotation frame).
    Mesh: map to bary_uv via per-face (or median) characteristic length ~ sqrt(2A).

      std_bary_g = target_world_g / (char_len_g * sqrt(2))
      target_world_g = parent_scale_g            (gb_match)
                    or parent_scale_g * calib     (manual base_std)
    """
    fi = face_idx.long()
    device = fi.device
    scale = parent_scale.reshape(-1).to(device=device, dtype=torch.float32).clamp(min=1e-12)

    eps = float(getattr(cfg, "gaussian_split_bary_noise_area_eps", 0.0))
    if eps <= 0.0:
        eps = float(face_areas.median().item()) * 1e-3
    a = face_areas[fi].clamp(min=eps)

    if getattr(cfg, "gaussian_split_bary_noise_area_normalize", True):
        char_len = torch.sqrt(2.0 * a)
    else:
        char_len_med = math.sqrt(2.0 * max(float(face_areas.median().item()), eps))
        char_len = scale.new_full(scale.shape, char_len_med)

    if getattr(cfg, "gaussian_split_bary_noise_gb_match", True):
        target_world = scale
    else:
        manual = float(base_std if base_std is not None else getattr(cfg, "gaussian_split_bary_noise", 0.12))
        extent = float(getattr(cfg, "gaussian_scene_extent", 1.0))
        pct = float(getattr(cfg, "gaussian_percent_dense", 0.0025))
        ref_scale = pct * extent
        char_len_med = math.sqrt(2.0 * max(float(face_areas.median().item()), eps))
        ref_target = manual * char_len_med * math.sqrt(2.0)
        target_world = scale * (ref_target / (ref_scale + 1e-12))

    std = target_world / (char_len * math.sqrt(2.0) + 1e-12)
    return std.to(dtype=torch.float32)


def _perturb_bary_uv_children(bary_uv_sel, std_per_gaussian, device):
    """
    Two children per parent: perturb (u, v) only. No in-simplex clamp — triangle walk fixes OOB.
    ``std_per_gaussian``: scalar or [N_sel] per-parent std in bary space.
    """
    n_sel = bary_uv_sel.shape[0]
    if n_sel == 0:
        return bary_uv_sel.new_zeros(0, 2)
    if isinstance(std_per_gaussian, (int, float)):
        if float(std_per_gaussian) <= 0:
            return bary_uv_sel.unsqueeze(0).repeat(2, 1, 1).reshape(-1, 2)
        sc = bary_uv_sel.new_full((1, n_sel, 1), float(std_per_gaussian))
    else:
        sc = std_per_gaussian.reshape(1, n_sel, 1).to(device=device, dtype=bary_uv_sel.dtype)
        if not (sc > 0).any():
            return bary_uv_sel.unsqueeze(0).repeat(2, 1, 1).reshape(-1, 2)
    noise = torch.randn(2, n_sel, 2, device=device, dtype=bary_uv_sel.dtype) * sc
    return (bary_uv_sel.unsqueeze(0) + noise).reshape(-1, 2)


def _perturb_h_children(h_sel, std, device):
    """Small normal offset for split children (keep h perturbation mild)."""
    n_sel = h_sel.shape[0]
    if n_sel == 0:
        return h_sel.new_zeros(0, 1)
    if std <= 0:
        return h_sel.unsqueeze(0).repeat(2, 1, 1).reshape(-1, 1)
    noise = torch.randn(2, n_sel, 1, device=device, dtype=h_sel.dtype) * float(std)
    return (h_sel.unsqueeze(0) + noise).reshape(-1, 1)


def duplicate_surf(surf, optimizer, mask, *, bary_noise=0.0, h_noise=0.0, opacity_correction=False):
    """
    Duplicates selected Gaussians according to mask.
    """
    device = mask.device
    sel = torch.where(mask)[0]
    n_new = len(sel)
    if n_new == 0:
        return

    # Update parameters
    for p_name in GAUSSIAN_PARAMETER_NAMES:
        p = getattr(surf, p_name)
        new_val = torch.cat([p.data, p.data[sel]])
        if p_name == "opacity" and opacity_correction:
            alpha_parent = torch.sigmoid(p.data[sel])
            alpha_hat = 1.0 - torch.sqrt((1.0 - alpha_parent).clamp(min=1e-8))
            alpha_hat = alpha_hat.clamp(min=1e-6, max=1.0 - 1e-6)
            new_val[-n_new:] = torch.logit(alpha_hat)
        new_param = torch.nn.Parameter(new_val, requires_grad=p.requires_grad)
        _replace_surface_param(surf, p_name, new_param)
        
        # Update optimizer
        def optimizer_fn(key, v):
            return torch.cat([v, torch.zeros((n_new, *v.shape[1:]), device=device, dtype=v.dtype)])
        update_optimizer_param(optimizer, p, new_param, optimizer_fn)

    # Update buffers
    face_idx = surf.face_idx
    n_old = face_idx.shape[0]
    surf.register_buffer("face_idx", torch.cat([face_idx, face_idx[sel]]))
    for name in SURFACE_BUFFER_NAMES:
        value = getattr(surf, name)
        surf.register_buffer(name, torch.cat([value, value[sel]]))

    if hasattr(surf, "face_texture_map_id") and surf.face_texture_map_id is not None:
        face_texture_map_id = surf.face_texture_map_id
        surf.register_buffer("face_texture_map_id", torch.cat([face_texture_map_id, face_texture_map_id[sel]]))
    if bary_noise > 0:
        dup = slice(n_old, n_old + n_new)
        surf.bary_uv.data[dup] = surf.bary_uv.data[dup] + torch.randn(
            n_new, 2, device=device, dtype=surf.bary_uv.dtype
        ) * float(bary_noise)
    if h_noise > 0:
        dup = slice(n_old, n_old + n_new)
        surf.h.data[dup] = surf.h.data[dup] + torch.randn(
            n_new, 1, device=device, dtype=surf.h.dtype
        ) * float(h_noise)


def split_surf(
    surf,
    optimizer,
    mask,
    *,
    bary_noise=0.12,
    h_noise=0.001,
    scale_divisor=1.6,
    face_areas=None,
    cfg=None,
):
    """
    Splits selected Gaussians according to mask.
    """
    device = mask.device
    sel = torch.where(mask)[0]
    rest = torch.where(~mask)[0]
    n_sel = len(sel)
    if n_sel == 0:
        return

    # Special rules for log_scale, bary, h:
    # 1. log_scale is reduced by log(0.8 * N), N=2 (GB densify_and_split)
    scales_split = (surf.log_scale.data[sel] - math.log(float(scale_divisor))).repeat(2, 1)
    new_log_scale = torch.cat([surf.log_scale.data[rest], scales_split])
    
    # 2. bary_uv: tangent-plane jitter; triangle walk after grow repairs OOB
    if face_areas is not None and cfg is not None:
        parent_scale = torch.exp(surf.log_scale.data[sel]).amax(dim=-1)
        bary_std = per_gaussian_bary_noise_std(
            surf.face_idx[sel],
            face_areas,
            cfg,
            parent_scale,
            base_std=bary_noise,
        )
    else:
        bary_std = bary_noise
    bary_uv_split = _perturb_bary_uv_children(surf.bary_uv.data[sel], bary_std, device)
    new_bary_uv = torch.cat([surf.bary_uv.data[rest], bary_uv_split])

    # 3. h: small offset only
    h_split = _perturb_h_children(surf.h.data[sel], h_noise, device)
    new_h = torch.cat([surf.h.data[rest], h_split])

    # Update other parameters: rotation, opacity, color
    for p_name in SPLIT_COPY_PARAMETER_NAMES:
        p = getattr(surf, p_name)
        p_split = _repeat_selected(p.data, sel)
        new_val = torch.cat([p.data[rest], p_split])
        new_param = torch.nn.Parameter(new_val, requires_grad=p.requires_grad)
        _replace_surface_param(surf, p_name, new_param)
        
        # Update optimizer
        def optimizer_fn(key, v):
            v_split = torch.zeros((2 * n_sel, *v.shape[1:]), device=device, dtype=v.dtype)
            return torch.cat([v[rest], v_split])
        update_optimizer_param(optimizer, p, new_param, optimizer_fn)

    # For log_scale, h, and bary_uv, we update optimizer too
    old_log_scale = surf.log_scale
    new_log_scale_param = torch.nn.Parameter(new_log_scale, requires_grad=old_log_scale.requires_grad)
    _replace_surface_param(surf, "log_scale", new_log_scale_param)
    def optimizer_fn_scale(key, v):
        v_split = torch.zeros((2 * n_sel, *v.shape[1:]), device=device, dtype=v.dtype)
        return torch.cat([v[rest], v_split])
    update_optimizer_param(optimizer, old_log_scale, new_log_scale_param, optimizer_fn_scale)

    old_h = surf.h
    new_h_param = torch.nn.Parameter(new_h, requires_grad=old_h.requires_grad)
    _replace_surface_param(surf, "h", new_h_param)
    def optimizer_fn_h(key, v):
        v_split = torch.zeros((2 * n_sel, *v.shape[1:]), device=device, dtype=v.dtype)
        return torch.cat([v[rest], v_split])
    update_optimizer_param(optimizer, old_h, new_h_param, optimizer_fn_h)

    old_bary_uv = surf.bary_uv
    new_bary_uv_param = torch.nn.Parameter(new_bary_uv, requires_grad=old_bary_uv.requires_grad)
    _replace_surface_param(surf, "bary_uv", new_bary_uv_param)
    def optimizer_fn_bary_uv(key, v):
        v_split = torch.zeros((2 * n_sel, *v.shape[1:]), device=device, dtype=v.dtype)
        return torch.cat([v[rest], v_split])
    update_optimizer_param(optimizer, old_bary_uv, new_bary_uv_param, optimizer_fn_bary_uv)

    # Update buffers
    face_idx = surf.face_idx
    surf.register_buffer("face_idx", torch.cat([face_idx[rest], face_idx[sel].repeat(2)]))
    for name in SURFACE_BUFFER_NAMES:
        value = getattr(surf, name)
        surf.register_buffer(name, torch.cat([value[rest], _repeat_selected(value, sel)]))

    if hasattr(surf, "face_texture_map_id") and surf.face_texture_map_id is not None:
        face_texture_map_id = surf.face_texture_map_id
        surf.register_buffer("face_texture_map_id", torch.cat([face_texture_map_id[rest], face_texture_map_id[sel].repeat(2)]))

def prune_surf(surf, optimizer, mask, ict_faces, ict):
    """Drop Gaussians where ``mask`` is True (opacity / scale / screen / cap)."""
    keep = ~mask
    if mask.any():
        # Update parameters
        for p_name in GAUSSIAN_PARAMETER_NAMES:
            p = getattr(surf, p_name)
            new_val = p.data[keep]
            new_param = torch.nn.Parameter(new_val, requires_grad=p.requires_grad)
            _replace_surface_param(surf, p_name, new_param)
            
            # Update optimizer
            def optimizer_fn(key, v):
                return v[keep]
            update_optimizer_param(optimizer, p, new_param, optimizer_fn)

        # Update buffers
        surf.register_buffer("face_idx", surf.face_idx[keep])
        for name in SURFACE_BUFFER_NAMES:
            surf.register_buffer(name, getattr(surf, name)[keep])

        if hasattr(surf, "face_texture_map_id") and surf.face_texture_map_id is not None:
            surf.register_buffer("face_texture_map_id", surf.face_texture_map_id[keep])
    return int(mask.sum().item())


def reset_opacity_surf(surf, optimizer, value):
    """
    Resets Gaussian opacities to logit of `value`.
    """
    logit_val = float(torch.logit(torch.tensor(value)))
    p = surf.opacity
    new_val = torch.full_like(p.data, logit_val)
    new_param = torch.nn.Parameter(new_val, requires_grad=p.requires_grad)
    _replace_surface_param(surf, "opacity", new_param)
    
    def optimizer_fn(key, v):
        return torch.zeros_like(v)
    update_optimizer_param(optimizer, p, new_param, optimizer_fn)


def decay_opacity_surf(surf, optimizer, decay_factor, min_value=1e-4):
    """
    Decays Gaussian opacities in probability space:
      sigma_new = clamp(sigma_old * decay_factor, min_value, 1-min_value)
    """
    p = surf.opacity
    op = torch.sigmoid(p.data)
    decayed = (op * float(decay_factor)).clamp(min=float(min_value), max=1.0 - float(min_value))
    new_val = torch.logit(decayed)
    new_param = torch.nn.Parameter(new_val, requires_grad=p.requires_grad)
    _replace_surface_param(surf, "opacity", new_param)

    def optimizer_fn(key, v):
        return torch.zeros_like(v)

    update_optimizer_param(optimizer, p, new_param, optimizer_fn)


def subtract_opacity_surf(surf, optimizer, subtract, min_value=1e-4):
    """
    Opacity regularization (arxiv:2404.06109 Sec. 3.5): in probability space,
      sigma_new = clamp(sigma_old - subtract, min_value, 1-min_value)
    """
    p = surf.opacity
    op = torch.sigmoid(p.data)
    new_op = (op - float(subtract)).clamp(min=float(min_value), max=1.0 - float(min_value))
    new_val = torch.logit(new_op)
    new_param = torch.nn.Parameter(new_val, requires_grad=p.requires_grad)
    _replace_surface_param(surf, "opacity", new_param)

    def optimizer_fn(key, v):
        return torch.zeros_like(v)

    update_optimizer_param(optimizer, p, new_param, optimizer_fn)


class BarycentricDensificationStrategy:
    def __init__(self, cfg):
        self.cfg = cfg
        self.state = {
            "grad_signal": None,
            "grad_signal_2d": None,
            "grad_signal_rgb": None,
            "count": None,
            "max_radii2D": None,
        }
        self._adj_faces = None
        self._last_densify_analysis = None
        self._template_face_areas = None

    def _face_areas_for_ict(self, ict, device):
        if self._template_face_areas is None or self._template_face_areas.device != device:
            self._template_face_areas = template_face_areas(ict, device)
        return self._template_face_areas

    def _grow_option(self):
        opt = str(getattr(self.cfg, "gaussian_grow_option", "grad2d")).lower()
        if opt not in ("grad2d", "gradrgb"):
            raise ValueError(f"gaussian_grow_option must be 'grad2d' or 'gradrgb', got {opt!r}")
        return opt

    def _grow_threshold(self):
        if self._grow_option() == "gradrgb":
            return float(self.cfg.gaussian_grow_gradrgb)
        base = float(self.cfg.gaussian_grow_grad2d)
        scale = float(getattr(self.cfg, "gaussian_grow_grad2d_face_scale", 1.0))
        return base * scale

    def _grow_threshold_per_gaussian(self, surf):
        if self._grow_option() == "gradrgb":
            t = float(self.cfg.gaussian_grow_gradrgb)
            return torch.full((surf.n_gaussians,), t, device=surf.h.device, dtype=torch.float32)
        base = float(self.cfg.gaussian_grow_grad2d)
        scale = float(getattr(self.cfg, "gaussian_grow_grad2d_face_scale", 1.0))
        if not getattr(self.cfg, "gaussian_grow_grad2d_region_split", True):
            t = base * scale
            return torch.full((surf.n_gaussians,), t, device=surf.h.device, dtype=torch.float32)
        codes = getattr(surf, "face_region_code", None)
        if codes is None or codes.numel() != surf.n_gaussians:
            t = base * scale
            return torch.full((surf.n_gaussians,), t, device=surf.h.device, dtype=torch.float32)
        from utils.ict_regions import grow_grad2d_threshold_per_gaussian

        return grow_grad2d_threshold_per_gaussian(codes, base, scale)

    def _scale_viewspace_grad_norm(self, grads_norm):
        return grads_norm * _grow_grad_pixel_scale(self.cfg)

    def _viewspace_grads_from_render(self, render_out):
        if render_out is None:
            return None
        viewspace_points = render_out.get("viewspace_points")
        if viewspace_points is None:
            meta = render_out.get("meta")
            if isinstance(meta, dict):
                viewspace_points = meta.get("means2d")
        if viewspace_points is None or viewspace_points.grad is None:
            return None
        return self._scale_viewspace_grad_norm(_viewspace_grad_norm_gb(viewspace_points.grad))

    def _densify_warmup_local(self):
        return int(getattr(self.cfg, "gaussian_densify_warmup_local", 500))

    def _screen_prune_threshold(self, stage_local=None):
        """GB: ``size_threshold = 20 if iteration > opacity_reset_interval else None``."""
        after = int(getattr(self.cfg, "gaussian_prune_screen_after_local", 3000))
        if stage_local is None or stage_local <= after:
            return None
        return float(getattr(self.cfg, "gaussian_prune_screen_radius", 20.0))

    def _should_run_densify_pass(self, stage_local=None):
        if stage_local is None:
            return False
        if stage_local <= self._densify_warmup_local():
            return False
        every = max(1, int(self.cfg.gaussian_densify_every))
        return stage_local % every == 0

    def _stage_local_range(self, stage_name, key):
        ranges = getattr(self.cfg, key, None) or {}
        if stage_name and stage_name in ranges:
            lo, hi = ranges[stage_name]
            return int(lo), int(hi)
        return None

    def _in_densify_window(self, global_step, stage_name=None, stage_local=None):
        local = self._stage_local_range(stage_name, "gaussian_densify_stage_local")
        if local is not None and stage_local is not None:
            lo, hi = local
            return lo <= stage_local <= hi
        return (
            self.cfg.gaussian_densify_start_iter
            <= global_step
            <= self.cfg.gaussian_densify_stop_iter
        )

    def _in_cleanup_window(self, stage_name=None, stage_local=None):
        local = self._stage_local_range(stage_name, "gaussian_cleanup_stage_local")
        if local is None or stage_local is None:
            return False
        lo, hi = local
        return lo <= stage_local <= hi

    def _should_track_densify_stats(self, global_step, surf, stage_name=None, stage_local=None):
        if not self.cfg.gaussian_densify:
            return False
        if surf is not None and len(surf.h) >= self.cfg.gaussian_densify_max:
            return False
        if self._in_densify_window(global_step, stage_name, stage_local):
            accum_every = max(1, int(getattr(self.cfg, "gaussian_densify_accum_every", 1)))
            return global_step % accum_every == 0
        if self._in_cleanup_window(stage_name, stage_local):
            return True
        return False

    def _should_accumulate(self, step, surf=None, stage_name=None, stage_local=None):
        return self._should_track_densify_stats(step, surf, stage_name, stage_local)

    def reset_state(self, n_gaussian, device):
        z = torch.zeros((n_gaussian, 1), device=device)
        self.state["grad_signal"] = torch.zeros_like(z)
        self.state["grad_signal_2d"] = torch.zeros_like(z)
        self.state["grad_signal_rgb"] = torch.zeros_like(z)
        self.state["count"] = torch.zeros_like(z)
        self.state["max_radii2D"] = torch.zeros(n_gaussian, device=device)

    def _world_scale_prune_threshold(self):
        ratio = float(getattr(self.cfg, "gaussian_prune_world_scale_ratio", 0.1))
        extent = float(getattr(self.cfg, "gaussian_scene_extent", 1.0))
        return ratio * extent

    def _global_mesh_scale(self, tracker=None, apply_pose_scale=True):
        if not apply_pose_scale or tracker is None or not hasattr(tracker, "log_pose_scale"):
            return 1.0
        return torch.exp(tracker.log_pose_scale)

    def _densify_scale_max(self, surf, tracker=None, apply_pose_scale=True):
        """GB ``max(exp(log_scale))``, times tracker global mesh scale when enabled."""
        if hasattr(surf, "densify_scale_max"):
            scale = surf.densify_scale_max()
        else:
            scale = torch.exp(surf.log_scale).amax(dim=-1)
        g = self._global_mesh_scale(tracker, apply_pose_scale)
        if isinstance(g, torch.Tensor):
            g = g.to(device=scale.device, dtype=scale.dtype)
        return scale * g

    def _build_prune_mask(
        self, surf, device, stage_local=None, mesh_verts=None, tracker=None, apply_pose_scale=True
    ):
        """
        Opacity + world-scale + screen-radius prune (GaussianBlendshapes / 3DGS).

        GB ``densify_and_prune``: opacity always; screen + world scale only when
        ``max_screen_size`` is set (after ``opacity_reset_interval`` = 3000).
        Scale uses ``exp(log_scale) * global_mesh_scale`` (not face area ratio).
        Returns (mask [N], stats dict).
        """
        n = len(surf.h)
        op_sig = torch.sigmoid(surf.opacity.flatten())
        op_prune = op_sig < self.cfg.gaussian_prune_opa

        scale_max = self._densify_scale_max(surf, tracker=tracker, apply_pose_scale=apply_pose_scale)
        world_thr = self._world_scale_prune_threshold()

        screen_thr = self._screen_prune_threshold(stage_local)
        screen_prune = torch.zeros(n, dtype=torch.bool, device=device)
        scale_prune = torch.zeros(n, dtype=torch.bool, device=device)
        max_radii = self.state.get("max_radii2D")
        if screen_thr is not None and max_radii is not None and len(max_radii) == n:
            screen_prune = max_radii > screen_thr
            scale_prune = scale_max > world_thr

        face_cap = int(getattr(self.cfg, "gaussian_max_per_face", -1))
        face_prune = torch.zeros(n, dtype=torch.bool, device=device)
        if face_cap > 0:
            face_prune = _face_cap_prune_mask(surf.face_idx, op_sig, face_cap)

        is_prune = op_prune | scale_prune | screen_prune | face_prune
        stats = {
            "opacity": int(op_prune.sum().item()),
            "world_scale": int(scale_prune.sum().item()),
            "screen_radius": int(screen_prune.sum().item()),
            "face_cap": int(face_prune.sum().item()),
            "total": int(is_prune.sum().item()),
            "world_thr": world_thr,
            "screen_thr": screen_thr if screen_thr is not None else -1.0,
            "scale_max_mean": scale_max.mean().item(),
        }
        return is_prune, stats

    def _accumulate_max_radii2d(self, surf, render_out):
        n_gaussian = len(surf.h)
        device = surf.h.device
        if self.state["max_radii2D"] is None or len(self.state["max_radii2D"]) != n_gaussian:
            self.state["max_radii2D"] = torch.zeros(n_gaussian, device=device)

        radii = _radii_per_gaussian(render_out.get("radii"), n_gaussian, device)
        if radii is None:
            return

        visible = _visible_mask_per_gaussian(render_out.get("radii"), n_gaussian, device)
        idx = torch.where(visible)[0]
        if idx.numel() == 0:
            return
        self.state["max_radii2D"][idx] = torch.maximum(
            self.state["max_radii2D"][idx], radii[idx]
        )

    def _visible_indices(self, surf, render_out):
        n_gaussian = len(surf.h)
        device = surf.h.device
        visible_mask = _visible_mask_per_gaussian(render_out.get("radii"), n_gaussian, device)
        return torch.where(visible_mask)[0]

    def _bump_visible_count(self, surf, render_out):
        device = surf.h.device
        visible_indices = self._visible_indices(surf, render_out)
        if visible_indices.numel() == 0:
            return
        self.state["count"].index_add_(
            0,
            visible_indices,
            torch.ones(
                (visible_indices.numel(), 1), device=device, dtype=self.state["count"].dtype
            ),
        )

    def _accumulate_grad_signal(self, surf, render_out, grads_norm, *, target: str = "grad_signal"):
        """GB ``add_densification_stats``: accum += norm(grad); denom in ``count``."""
        n_gaussian = len(surf.h)
        if grads_norm.shape[0] != n_gaussian:
            return

        buf = self.state.get(target)
        if buf is None or len(buf) != n_gaussian:
            return

        visible_indices = self._visible_indices(surf, render_out)
        if visible_indices.numel() == 0:
            return
        visible_mask = _visible_mask_per_gaussian(
            render_out.get("radii"), n_gaussian, surf.h.device
        )
        contrib = grads_norm[visible_mask].reshape(-1, 1)
        buf.index_add_(0, visible_indices, contrib)

    def pre_backward(self, step, render_out, avatar=None, stage_name=None, stage_local=None):
        surf = avatar.surface if avatar is not None else None
        if not self._should_accumulate(step, surf, stage_name, stage_local):
            return
        if surf is not None:
            if surf.color.requires_grad:
                surf.color.retain_grad()
            if hasattr(surf, "color_pose") and surf.color_pose.requires_grad:
                surf.color_pose.retain_grad()
            if hasattr(surf, "color_expression") and surf.color_expression.requires_grad:
                surf.color_expression.retain_grad()
        if render_out is None:
            return
        viewspace_points = render_out.get("viewspace_points")
        if viewspace_points is None:
            meta = render_out.get("meta")
            if isinstance(meta, dict):
                viewspace_points = meta.get("means2d")
        if viewspace_points is not None:
            viewspace_points.retain_grad()

    def post_backward(self, step, avatar, render_out, stage_name=None, stage_local=None):
        """
        Accumulate per-Gaussian grow signal (``grad2d`` viewspace or ``gradrgb`` from
        ``color`` + ``color_pose`` + ``color_expression`` grads).

        Topology changes run in ``post_optimizer_step`` after ``optimizer.step()``.
        """
        surf = avatar.surface
        if not self._should_accumulate(step, surf, stage_name, stage_local) or render_out is None:
            return

        device = surf.h.device
        n_gaussian = len(surf.h)
        if self.state["grad_signal"] is None or len(self.state["grad_signal"]) != n_gaussian:
            self.reset_state(n_gaussian, device)

        self._accumulate_max_radii2d(surf, render_out)

        grow_opt = self._grow_option()
        has_signal = False
        grads_rgb = _surface_color_grad_norm_for_densify(surf, n_gaussian)
        if grads_rgb is not None:
            has_signal = True
            self._accumulate_grad_signal(surf, render_out, grads_rgb, target="grad_signal_rgb")
            if grow_opt == "gradrgb":
                self._accumulate_grad_signal(surf, render_out, grads_rgb, target="grad_signal")

        grads_norm = self._viewspace_grads_from_render(render_out)
        if grads_norm is not None:
            has_signal = True
            self._accumulate_grad_signal(surf, render_out, grads_norm, target="grad_signal_2d")
            if grow_opt == "grad2d":
                self._accumulate_grad_signal(surf, render_out, grads_norm, target="grad_signal")

        if has_signal:
            self._bump_visible_count(surf, render_out)

        if (
            self.state.get("count") is not None
            and self.state["count"].sum() > 0
            and self.state.get("grad_signal_2d") is not None
        ):
            snap = self._build_analysis_snapshot()
            if snap is not None:
                self._last_densify_analysis = snap

    @torch.no_grad()
    def _build_analysis_snapshot(self):
        count = self.state.get("count")
        g2d = self.state.get("grad_signal_2d")
        grgb = self.state.get("grad_signal_rgb")
        if count is None or g2d is None or grgb is None or count.sum() <= 0:
            return None

        def _buf_stats(buf, thr):
            grads = buf / count.clamp_min(1)
            grads[grads.isnan()] = 0.0
            g = grads.flatten()
            obs = count.flatten() > 0
            if not obs.any():
                return {}
            g_obs = g[obs]
            return {
                "n_obs": int(obs.sum().item()),
                "mean": float(g_obs.mean().item()),
                "p50": float(torch.quantile(g_obs, 0.5).item()),
                "p90": float(torch.quantile(g_obs, 0.9).item()),
                "max": float(g_obs.max().item()),
                "above_thr": int((g_obs >= thr).sum().item()),
                "thr": float(thr),
            }

        thr2d = float(self.cfg.gaussian_grow_grad2d) * float(
            getattr(self.cfg, "gaussian_grow_grad2d_face_scale", 1.0)
        )
        thr_rgb = float(self.cfg.gaussian_grow_gradrgb)
        s2d = _buf_stats(g2d, thr2d)
        srgb = _buf_stats(grgb, thr_rgb)
        out = {
            "densify/grow_option": self._grow_option(),
            "densify/pixel_scale": float(_grow_grad_pixel_scale(self.cfg)),
            "densify/n_gaussian": int(count.shape[0]),
        }
        for prefix, stats in (("grad2d", s2d), ("gradrgb", srgb)):
            for k, v in stats.items():
                out[f"densify/{prefix}_{k}"] = v
        return out

    @torch.no_grad()
    def analysis_snapshot(self, global_step, stage_name=None, stage_local=None, surf=None):
        """Running-mean grad norms for grad2d vs gradrgb (loss_log.jsonl)."""
        if not self.cfg.gaussian_densify:
            return {}
        if stage_name not in getattr(self.cfg, "gaussian_densify_stages", []):
            return {}
        if not self._should_track_densify_stats(global_step, surf, stage_name, stage_local):
            return {"densify/tracking": 0}

        count = self.state.get("count")
        snap = None
        if count is not None and count.sum() > 0:
            snap = self._build_analysis_snapshot()
            if snap is not None:
                self._last_densify_analysis = snap

        cached = snap is None and self._last_densify_analysis is not None
        snap = snap or self._last_densify_analysis
        if snap is None:
            return {"densify/tracking": 1, "densify/n_obs": 0}

        out = {"densify/tracking": 1, **snap}
        if cached:
            out["densify/snapshot_cached"] = 1
        return out

    def _should_reset_opacity(self, step, stage_name=None, stage_local=None) -> bool:
        stage_local_iters = getattr(self.cfg, "gaussian_reset_stage_local", None) or {}
        if stage_name and stage_local is not None:
            local_list = stage_local_iters.get(stage_name, [])
            if stage_local in local_list:
                return True
        reset_iters = getattr(self.cfg, "gaussian_reset_iters", None) or []
        if step in reset_iters:
            return True
        every = int(getattr(self.cfg, "gaussian_reset_every", 0))
        return every > 0 and step > 0 and step % every == 0

    def _should_cleanup_reset(self, stage_name=None, stage_local=None) -> bool:
        reset_map = getattr(self.cfg, "gaussian_cleanup_reset_stage_local", None) or {}
        if stage_name and stage_local is not None:
            return stage_local in reset_map.get(stage_name, [])
        return False

    def _should_decay_opacity(self, step, stage_name=None, stage_local=None) -> bool:
        if getattr(self.cfg, "gaussian_opacity_decay_during_cleanup", False):
            if self._in_cleanup_window(stage_name, stage_local):
                every = max(1, int(getattr(self.cfg, "gaussian_cleanup_prune_every", 100)))
                if stage_local is not None and stage_local % every == 0:
                    return True
        stage_local_iters = getattr(self.cfg, "gaussian_opacity_decay_stage_local", None) or {}
        if stage_name and stage_local is not None:
            local_list = stage_local_iters.get(stage_name, [])
            if stage_local in local_list:
                return True
        every = int(getattr(self.cfg, "gaussian_opacity_decay_every", 0))
        return every > 0 and step > 0 and step % every == 0

    def _opacity_reset_value(self):
        return float(
            getattr(
                self.cfg,
                "gaussian_reset_opacity_value",
                self.cfg.gaussian_prune_opa * 2.0,
            )
        )

    def _run_prune_pass(
        self,
        surf,
        optimizer,
        ict_faces,
        ict,
        device,
        label,
        step,
        stage_local=None,
        mesh_verts=None,
        tracker=None,
        apply_pose_scale=True,
    ):
        n_start = len(surf.h)
        is_prune, prune_stats = self._build_prune_mask(
            surf,
            device,
            stage_local=stage_local,
            mesh_verts=mesh_verts,
            tracker=tracker,
            apply_pose_scale=apply_pose_scale,
        )
        n_removed, prune_stats = _execute_prune(
            surf, optimizer, is_prune, ict_faces, ict, device, prune_stats
        )
        n_end = len(surf.h)
        _print_densify_summary(
            label,
            step=step,
            stage_local=stage_local,
            n_start=n_start,
            n_after_grow=n_start,
            n_end=n_end,
            n_clone=0,
            n_split=0,
            prune_stats=prune_stats,
        )
        if n_end != n_start:
            self.reset_state(n_end, device)
        return n_removed

    @torch.no_grad()
    def pre_optimizer_step(
        self,
        step,
        avatar,
        optimizer,
        ict_faces,
        ict,
        stage_name=None,
        stage_local=None,
        mesh_verts=None,
        tracker=None,
        apply_pose_scale=True,
    ):
        """
        GB order: backward → accumulate stats → densify_and_prune → optimizer.step().
        """
        if optimizer is None or not self.cfg.gaussian_densify:
            return
        if stage_name not in getattr(self.cfg, "gaussian_densify_stages", []):
            return
        if self._in_densify_window(step, stage_name, stage_local) and self._should_run_densify_pass(
            stage_local
        ):
            self._grow_and_prune(
                avatar,
                optimizer,
                ict_faces,
                ict,
                step=step,
                stage_local=stage_local,
                mesh_verts=mesh_verts,
                tracker=tracker,
                apply_pose_scale=apply_pose_scale,
            )
            subtract = float(getattr(self.cfg, "gaussian_opacity_subtract_after_densify", 0.0))
            if subtract > 0.0:
                surf = avatar.surface
                min_val = float(getattr(self.cfg, "gaussian_opacity_decay_min", 1e-4))
                subtract_opacity_surf(surf, optimizer, subtract, min_value=min_val)
                op_after = torch.sigmoid(surf.opacity.flatten())
                print(
                    f"[Opacity subtract] stage={stage_name} local={stage_local} "
                    f"opacity -= {subtract:.4f} (mean={op_after.mean().item():.4f})"
                )

        # 3DGS / SA opacity reset: independent of densify-pass cadence. Runs last so the
        # reset value (0.01 > prune_opa) is not immediately pruned in the same step.
        if self._should_reset_opacity(step, stage_name, stage_local):
            surf = avatar.surface
            reset_val = self._opacity_reset_value()
            reset_opacity_surf(surf, optimizer, reset_val)
            op_after = torch.sigmoid(surf.opacity.flatten())
            print(
                f"[Opacity reset] stage={stage_name} local={stage_local} "
                f"all opacity -> {reset_val:.4f} (mean={op_after.mean().item():.4f})"
            )

    @torch.no_grad()
    def post_optimizer_step(
        self,
        step,
        avatar,
        optimizer,
        ict_faces,
        ict,
        stage_name=None,
        stage_local=None,
        mesh_verts=None,
        tracker=None,
        apply_pose_scale=True,
    ):
        """Cleanup-window prune / opacity policy only (GB main loop has no post-step densify)."""
        if optimizer is None:
            return
        if not self.cfg.gaussian_densify:
            return
        if stage_name not in getattr(self.cfg, "gaussian_densify_stages", []):
            return

        surf = avatar.surface
        device = surf.h.device

        in_cleanup = self._in_cleanup_window(stage_name, stage_local)

        if in_cleanup:
            cleanup_every = max(1, int(getattr(self.cfg, "gaussian_cleanup_prune_every", 100)))
            if stage_local is not None and stage_local % cleanup_every == 0:
                self._run_prune_pass(
                    surf,
                    optimizer,
                    ict_faces,
                    ict,
                    device,
                    "Cleanup Prune",
                    step,
                    stage_local,
                    mesh_verts=mesh_verts,
                    tracker=tracker,
                    apply_pose_scale=apply_pose_scale,
                )

            if self._should_cleanup_reset(stage_name, stage_local):
                reset_val = self._opacity_reset_value()
                reset_opacity_surf(surf, optimizer, reset_val)
                op_after = torch.sigmoid(surf.opacity.flatten())
                print(
                    f"[Cleanup reset] stage={stage_name} local={stage_local} "
                    f"all opacity -> {reset_val:.4f} (mean={op_after.mean().item():.4f})"
                )
            elif self._should_decay_opacity(step, stage_name, stage_local):
                decay = float(getattr(self.cfg, "gaussian_opacity_decay_factor", 1.0))
                min_val = float(getattr(self.cfg, "gaussian_opacity_decay_min", 1e-4))
                if 0.0 < decay < 1.0:
                    decay_opacity_surf(surf, optimizer, decay, min_value=min_val)
                    op_after = torch.sigmoid(surf.opacity.flatten())
                    print(
                        f"[Cleanup decay] stage={stage_name} local={stage_local} "
                        f"opacity *= {decay:.4f} (mean={op_after.mean().item():.4f})"
                    )

    def _walk_after_grow(self, surf, optimizer, ict_faces, ict):
        from training.triangle_walking import build_face_adjacency, walk_barycentric_surface

        device = surf.face_idx.device
        faces = ict_faces.to(device)
        if self._adj_faces is None or self._adj_faces.device != device:
            self._adj_faces = build_face_adjacency(faces)
        verts = ict.template_reference_verts().to(device=device, dtype=surf.bary_uv.dtype)
        max_iter = int(getattr(self.cfg, "gaussian_triangle_walk_max_iter", 3))
        return walk_barycentric_surface(
            surf,
            faces,
            verts,
            self._adj_faces,
            optimizer=optimizer,
            max_iterations=max_iter,
        )

    @torch.no_grad()
    def _grow_and_prune(
        self,
        avatar,
        optimizer,
        ict_faces,
        ict,
        step=None,
        stage_local=None,
        mesh_verts=None,
        tracker=None,
        apply_pose_scale=True,
    ):
        """
        GB ``densify_and_prune``: clone → split → opacity/world/screen prune; reset accum.
        """
        surf = avatar.surface
        device = surf.h.device
        n_start = len(surf.h)

        can_grow = (
            n_start < self.cfg.gaussian_densify_max
            and self.state["grad_signal"] is not None
            and self.state["count"] is not None
            and self.state["count"].sum() > 0
        )

        is_dupli = torch.zeros(n_start, dtype=torch.bool, device=device)
        is_split = torch.zeros(n_start, dtype=torch.bool, device=device)
        grad_thr = self._grow_threshold_per_gaussian(surf)
        opacity_correction = bool(getattr(self.cfg, "gaussian_clone_opacity_correction", False))
        n_clone = 0
        n_split = 0
        grow_extra = ""

        if can_grow:
            grads = self.state["grad_signal"] / self.state["count"].clamp_min(1)
            grads[grads.isnan()] = 0.0
            is_grad_high = torch.norm(grads, dim=-1) >= grad_thr

            scale_max = self._densify_scale_max(
                surf, tracker=tracker, apply_pose_scale=apply_pose_scale
            )
            percent_dense = float(getattr(self.cfg, "gaussian_percent_dense", 0.01))
            extent = float(getattr(self.cfg, "gaussian_scene_extent", 1.0))
            small_thr = percent_dense * extent
            is_small = scale_max <= small_thr
            is_dupli = is_grad_high & is_small
            is_split = is_grad_high & ~is_small

            face_cap = int(getattr(self.cfg, "gaussian_max_per_face", 0))
            if face_cap > 0 and is_split.any():
                _, per_face = _per_face_gaussian_counts(surf.face_idx)
                is_split = is_split & (per_face < face_cap)

            if n_start + int(is_dupli.sum()) + int(is_split.sum()) > self.cfg.gaussian_densify_max:
                is_dupli.zero_()
                is_split.zero_()
                grow_extra = "cap=max_gaussians"

            n_clone = int(is_dupli.sum().item())
            n_split = int(is_split.sum().item())

            if n_clone + n_split == 0:
                g = grads.detach().flatten()
                obs = self.state["count"].flatten() > 0
                if obs.any():
                    g_obs = g[obs]
                    thr_obs = grad_thr[obs] if grad_thr.numel() == g.numel() else grad_thr
                    above = int((g_obs >= thr_obs).sum().item())
                    p50 = torch.quantile(g_obs, 0.5).item()
                    p90 = torch.quantile(g_obs, 0.9).item()
                    gmax = g_obs.max().item()
                else:
                    above, p50, p90, gmax = 0, 0.0, 0.0, 0.0
                radii = self.state.get("max_radii2D")
                if radii is not None and radii.numel() == n_start:
                    r_obs = radii[obs] if obs.any() else radii
                    rp50 = torch.quantile(r_obs.float(), 0.5).item()
                    rp90 = torch.quantile(r_obs.float(), 0.9).item()
                    rmax = r_obs.max().item()
                    screen_thr = self._screen_prune_threshold(stage_local)
                    over_r = int(
                        (radii > screen_thr).sum().item()
                        if screen_thr is not None
                        else 0
                    )
                else:
                    rp50, rp90, rmax, over_r = 0.0, 0.0, 0.0, 0
                px = _grow_grad_pixel_scale(self.cfg)
                grow_extra = (
                    f"no grow candidates above_thr={above}/{int(obs.sum().item())} "
                    f"grad p50={p50:.3e} p90={p90:.3e} max={gmax:.3e} pixel_scale={px:.1f} "
                    f"radii2d p90={rp90:.1f} max={rmax:.1f} over_screen={over_r}"
                )

            face_areas = self._face_areas_for_ict(ict, device)
            bary_noise = float(getattr(self.cfg, "gaussian_split_bary_noise", 0.12))
            h_noise = float(getattr(self.cfg, "gaussian_split_h_noise", 0.001))
            split_scale_divisor = float(getattr(self.cfg, "gaussian_split_scale_divisor", 1.6))

            if is_dupli.any():
                # GB densify_and_clone: exact copy, no position noise.
                duplicate_surf(
                    surf,
                    optimizer,
                    is_dupli,
                    bary_noise=0.0,
                    h_noise=0.0,
                    opacity_correction=opacity_correction,
                )

            is_split_cat = torch.cat(
                [
                    is_split,
                    torch.zeros(int(is_dupli.sum().item()), dtype=torch.bool, device=device),
                ]
            )
            if is_split_cat.any():
                split_surf(
                    surf,
                    optimizer,
                    is_split_cat,
                    bary_noise=bary_noise,
                    h_noise=h_noise,
                    scale_divisor=split_scale_divisor,
                    face_areas=face_areas,
                    cfg=self.cfg,
                )

            if (n_clone + n_split) > 0 and getattr(self.cfg, "gaussian_densify_walk_after_grow", True):
                n_walked = self._walk_after_grow(surf, optimizer, ict_faces, ict)
                if n_walked > 0:
                    grow_extra = (grow_extra + " " if grow_extra else "") + f"walk={n_walked}"

        n_after_grow = len(surf.h)

        is_prune, prune_stats = self._build_prune_mask(
            surf,
            device,
            stage_local=stage_local,
            mesh_verts=mesh_verts,
            tracker=tracker,
            apply_pose_scale=apply_pose_scale,
        )
        _, prune_stats = _execute_prune(
            surf, optimizer, is_prune, ict_faces, ict, device, prune_stats
        )
        n_end = len(surf.h)
        _assert_surface_layout(surf)

        _print_densify_summary(
            "Densify",
            step=step,
            stage_local=stage_local,
            n_start=n_start,
            n_after_grow=n_after_grow,
            n_end=n_end,
            n_clone=n_clone,
            n_split=n_split,
            prune_stats=prune_stats,
            grad_thr=grad_thr,
            extra=grow_extra,
        )

        self.reset_state(n_end, device)
