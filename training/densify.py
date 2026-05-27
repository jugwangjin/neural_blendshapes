"""
Barycentric / UV+H space Gaussian densification and pruning strategy.
Protects mouth interior, mouth socket, and eye socket from pruning.
"""

import math
import torch
from utils.ict_regions import classify_surface_triangles_batch


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
SURFACE_BUFFER_NAMES = ("uv", "is_gum", "is_h_pin", "h_sigma_scale", "face_region_code")


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


def _print_tensor_grad_stats(values, label):
    """Print min/max/mean/std and percentiles for a 1D grad signal tensor."""
    t = values.detach().float().reshape(-1)
    t = t[torch.isfinite(t)]
    if t.numel() == 0:
        print(f"[Densify gradrgb] {label}: empty")
        return
    q = torch.tensor(
        [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0],
        device=t.device,
        dtype=t.dtype,
    )
    p = torch.quantile(t, q)
    print(
        f"[Densify gradrgb] {label}: n={t.numel()} "
        f"min={t.min().item():.6e} max={t.max().item():.6e} "
        f"mean={t.mean().item():.6e} std={t.std(unbiased=False).item():.6e}"
    )
    print(
        f"[Densify gradrgb] {label} percentiles: "
        f"p00={p[0].item():.6e} p01={p[1].item():.6e} p05={p[2].item():.6e} "
        f"p10={p[3].item():.6e} p25={p[4].item():.6e} p50={p[5].item():.6e} "
        f"p75={p[6].item():.6e} p90={p[7].item():.6e} p95={p[8].item():.6e} "
        f"p99={p[9].item():.6e} p100={p[10].item():.6e}"
    )


def _grads_norm_per_gaussian(grads, width, height):
    """
    Image-plane gradient norm per Gaussian [N].

    Supports ``grads`` shaped [N, 2], [C, N, 2], or [1, N, 2] (gsplat means2d).
    """
    if grads.ndim == 2:
        g = grads.clone()
        g[..., 0] *= width / 2.0
        g[..., 1] *= height / 2.0
        return g.norm(dim=-1)
    if grads.ndim == 3:
        n_cameras = grads.shape[0]
        g = grads.clone()
        g[..., 0] *= width / 2.0 * n_cameras
        g[..., 1] *= height / 2.0 * n_cameras
        return g.norm(dim=-1).sum(dim=0)
    raise ValueError(f"unexpected viewspace grad shape {tuple(grads.shape)}")


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


def _resize_optional_gaussian_buffers(surf, index):
    for name in ("sem_anchor", "sem_prob_fixed"):
        value = getattr(surf, name, None)
        if value is not None:
            surf.register_buffer(name, value[index])


def _repeat_selected(value, index, repeats=2):
    repeat_shape = (repeats, *([1] * (value.ndim - 1)))
    return value[index].repeat(*repeat_shape)


def duplicate_surf(surf, optimizer, mask):
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
        new_param = torch.nn.Parameter(new_val, requires_grad=p.requires_grad)
        setattr(surf, p_name, new_param)
        
        # Update optimizer
        def optimizer_fn(key, v):
            return torch.cat([v, torch.zeros((n_new, *v.shape[1:]), device=device, dtype=v.dtype)])
        update_optimizer_param(optimizer, p, new_param, optimizer_fn)

    if surf.sem_logits is not None:
        p = surf.sem_logits
        new_val = torch.cat([p.data, p.data[sel]])
        new_param = torch.nn.Parameter(new_val, requires_grad=p.requires_grad)
        surf.sem_logits = new_param
        def optimizer_fn(key, v):
            return torch.cat([v, torch.zeros((n_new, *v.shape[1:]), device=device, dtype=v.dtype)])
        update_optimizer_param(optimizer, p, new_param, optimizer_fn)

    # Update buffers
    face_idx = surf.face_idx
    surf.register_buffer("face_idx", torch.cat([face_idx, face_idx[sel]]))
    for name in SURFACE_BUFFER_NAMES:
        value = getattr(surf, name)
        surf.register_buffer(name, torch.cat([value, value[sel]]))

    if hasattr(surf, "face_texture_map_id") and surf.face_texture_map_id is not None:
        face_texture_map_id = surf.face_texture_map_id
        surf.register_buffer("face_texture_map_id", torch.cat([face_texture_map_id, face_texture_map_id[sel]]))
    _resize_optional_gaussian_buffers(surf, torch.cat([torch.arange(face_idx.shape[0], device=device), sel]))


def split_surf(surf, optimizer, mask):
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
    # 1. log_scale is reduced by log(1.6)
    scales_split = (surf.log_scale.data[sel] - math.log(1.6)).repeat(2, 1)
    new_log_scale = torch.cat([surf.log_scale.data[rest], scales_split])
    
    # 2. barycentric coordinates are perturbed in triangle tangent plane
    bary_sel = surf.bary[sel] # [N_sel, 3] from dynamic property
    noise = torch.randn(2, n_sel, 3, device=device) * 0.03
    # Mean-subtract to ensure sum to 0
    noise = noise - noise.mean(dim=-1, keepdim=True)
    bary_split = bary_sel.unsqueeze(0) + noise
    bary_split = torch.clamp(bary_split, min=1e-5)
    bary_split = bary_split / bary_split.sum(dim=-1, keepdim=True)
    bary_split = bary_split.reshape(-1, 3)

    bary_uv_split = bary_split[:, 1:3]
    new_bary_uv = torch.cat([surf.bary_uv.data[rest], bary_uv_split])

    # 3. h height is perturbed slightly
    h_sel = surf.h.data[sel]
    noise_h = torch.randn(2, n_sel, 1, device=device) * 0.002
    h_split = (h_sel.unsqueeze(0) + noise_h).reshape(-1, 1)
    new_h = torch.cat([surf.h.data[rest], h_split])

    # Update other parameters: rotation, opacity, color, sem_logits
    for p_name in SPLIT_COPY_PARAMETER_NAMES:
        p = getattr(surf, p_name)
        p_split = _repeat_selected(p.data, sel)
        new_val = torch.cat([p.data[rest], p_split])
        new_param = torch.nn.Parameter(new_val, requires_grad=p.requires_grad)
        setattr(surf, p_name, new_param)
        
        # Update optimizer
        def optimizer_fn(key, v):
            v_split = torch.zeros((2 * n_sel, *v.shape[1:]), device=device, dtype=v.dtype)
            return torch.cat([v[rest], v_split])
        update_optimizer_param(optimizer, p, new_param, optimizer_fn)

    # For log_scale, h, and bary_uv, we update optimizer too
    old_log_scale = surf.log_scale
    new_log_scale_param = torch.nn.Parameter(new_log_scale, requires_grad=old_log_scale.requires_grad)
    surf.log_scale = new_log_scale_param
    def optimizer_fn_scale(key, v):
        v_split = torch.zeros((2 * n_sel, *v.shape[1:]), device=device, dtype=v.dtype)
        return torch.cat([v[rest], v_split])
    update_optimizer_param(optimizer, old_log_scale, new_log_scale_param, optimizer_fn_scale)

    old_h = surf.h
    new_h_param = torch.nn.Parameter(new_h, requires_grad=old_h.requires_grad)
    surf.h = new_h_param
    def optimizer_fn_h(key, v):
        v_split = torch.zeros((2 * n_sel, *v.shape[1:]), device=device, dtype=v.dtype)
        return torch.cat([v[rest], v_split])
    update_optimizer_param(optimizer, old_h, new_h_param, optimizer_fn_h)

    old_bary_uv = surf.bary_uv
    new_bary_uv_param = torch.nn.Parameter(new_bary_uv, requires_grad=old_bary_uv.requires_grad)
    surf.bary_uv = new_bary_uv_param
    def optimizer_fn_bary_uv(key, v):
        v_split = torch.zeros((2 * n_sel, *v.shape[1:]), device=device, dtype=v.dtype)
        return torch.cat([v[rest], v_split])
    update_optimizer_param(optimizer, old_bary_uv, new_bary_uv_param, optimizer_fn_bary_uv)

    if surf.sem_logits is not None:
        p = surf.sem_logits
        p_split = _repeat_selected(p.data, sel)
        new_val = torch.cat([p.data[rest], p_split])
        new_param = torch.nn.Parameter(new_val, requires_grad=p.requires_grad)
        surf.sem_logits = new_param
        def optimizer_fn(key, v):
            v_split = torch.zeros((2 * n_sel, *v.shape[1:]), device=device, dtype=v.dtype)
            return torch.cat([v[rest], v_split])
        update_optimizer_param(optimizer, p, new_param, optimizer_fn)

    # Update buffers
    face_idx = surf.face_idx
    surf.register_buffer("face_idx", torch.cat([face_idx[rest], face_idx[sel].repeat(2)]))
    for name in SURFACE_BUFFER_NAMES:
        value = getattr(surf, name)
        surf.register_buffer(name, torch.cat([value[rest], _repeat_selected(value, sel)]))

    if hasattr(surf, "face_texture_map_id") and surf.face_texture_map_id is not None:
        face_texture_map_id = surf.face_texture_map_id
        surf.register_buffer("face_texture_map_id", torch.cat([face_texture_map_id[rest], face_texture_map_id[sel].repeat(2)]))
    _resize_optional_gaussian_buffers(surf, torch.cat([rest, sel.repeat(2)]))


def prune_surf(surf, optimizer, mask, ict_faces, ict):
    """
    Prunes Gaussians according to mask, while strictly protecting
    mouth interior (gums), mouth socket, and eye socket regions.
    """
    device = mask.device
    
    # 1. Identify which region/chart each Gaussian is in
    codes = getattr(surf, "face_region_code", None)
    if codes is None:
        codes = classify_surface_triangles_batch(surf.face_idx, ict_faces, ict, device)
    
    # Protect Region 0 (mouth_interior/gums), Region 1 (mouth_socket), and Region 2 (eye_socket)
    protected = (codes == 0) | (codes == 1) | (codes == 2)
    
    # Final prune mask: mask but not protected
    final_prune = mask & ~protected
    keep = ~final_prune
    
    if final_prune.any():
        # Update parameters
        for p_name in GAUSSIAN_PARAMETER_NAMES:
            p = getattr(surf, p_name)
            new_val = p.data[keep]
            new_param = torch.nn.Parameter(new_val, requires_grad=p.requires_grad)
            setattr(surf, p_name, new_param)
            
            # Update optimizer
            def optimizer_fn(key, v):
                return v[keep]
            update_optimizer_param(optimizer, p, new_param, optimizer_fn)

        if surf.sem_logits is not None:
            p = surf.sem_logits
            new_val = p.data[keep]
            new_param = torch.nn.Parameter(new_val, requires_grad=p.requires_grad)
            surf.sem_logits = new_param
            def optimizer_fn(key, v):
                return v[keep]
            update_optimizer_param(optimizer, p, new_param, optimizer_fn)

        # Update buffers
        surf.register_buffer("face_idx", surf.face_idx[keep])
        for name in SURFACE_BUFFER_NAMES:
            surf.register_buffer(name, getattr(surf, name)[keep])

        if hasattr(surf, "face_texture_map_id") and surf.face_texture_map_id is not None:
            surf.register_buffer("face_texture_map_id", surf.face_texture_map_id[keep])
        _resize_optional_gaussian_buffers(surf, keep)
            
    return final_prune.sum().item()


def reset_opacity_surf(surf, optimizer, value):
    """
    Resets Gaussian opacities to logit of `value`.
    """
    logit_val = float(torch.logit(torch.tensor(value)))
    p = surf.opacity
    new_val = torch.full_like(p.data, logit_val)
    new_param = torch.nn.Parameter(new_val, requires_grad=p.requires_grad)
    surf.opacity = new_param
    
    def optimizer_fn(key, v):
        return torch.zeros_like(v)
    update_optimizer_param(optimizer, p, new_param, optimizer_fn)


class BarycentricDensificationStrategy:
    def __init__(self, cfg):
        self.cfg = cfg
        self.state = {"grad_signal": None, "count": None}

    def _grow_option(self):
        opt = str(getattr(self.cfg, "gaussian_grow_option", "grad2d")).lower()
        if opt not in ("grad2d", "gradrgb"):
            raise ValueError(f"gaussian_grow_option must be 'grad2d' or 'gradrgb', got {opt!r}")
        return opt

    def _grow_threshold(self):
        if self._grow_option() == "gradrgb":
            return float(self.cfg.gaussian_grow_gradrgb)
        return float(self.cfg.gaussian_grow_grad2d)

    def _should_accumulate(self, step, surf=None):
        if not self.cfg.gaussian_densify:
            return False
        if step < self.cfg.gaussian_densify_start_iter or step > self.cfg.gaussian_densify_stop_iter:
            return False
        if surf is not None and len(surf.h) >= self.cfg.gaussian_densify_max:
            return False

        accum_every = max(1, int(getattr(self.cfg, "gaussian_densify_accum_every", 1)))
        return step % accum_every == 0

    def reset_state(self, n_gaussian, device):
        self.state["grad_signal"] = torch.zeros(n_gaussian, device=device)
        self.state["count"] = torch.zeros(n_gaussian, device=device)

    def _accumulate_grad_signal(self, surf, render_out, grads_norm):
        n_gaussian = len(surf.h)
        device = surf.h.device
        meta = render_out.get("meta")
        gaussian_ids = meta.get("gaussian_ids") if isinstance(meta, dict) else None

        if gaussian_ids is not None and grads_norm.shape[0] == gaussian_ids.shape[0]:
            self.state["grad_signal"].index_add_(0, gaussian_ids, grads_norm)
            self.state["count"].index_add_(0, gaussian_ids, torch.ones_like(grads_norm))
            return

        if grads_norm.shape[0] != n_gaussian:
            return

        visible_mask = _visible_mask_per_gaussian(render_out.get("radii"), n_gaussian, device)
        visible_indices = torch.where(visible_mask)[0]
        self.state["grad_signal"].index_add_(0, visible_indices, grads_norm[visible_mask])
        self.state["count"].index_add_(
            0, visible_indices, torch.ones_like(visible_indices, dtype=torch.float32)
        )

    def pre_backward(self, step, render_out, avatar=None):
        surf = avatar.surface if avatar is not None else None
        if not self._should_accumulate(step, surf) or render_out is None:
            return
        if self._grow_option() == "gradrgb":
            if surf is not None and surf.color.requires_grad:
                surf.color.retain_grad()
            return
        viewspace_points = render_out.get("viewspace_points")
        if viewspace_points is None:
            meta = render_out.get("meta")
            if isinstance(meta, dict):
                viewspace_points = meta.get("means2d")
        if viewspace_points is not None:
            viewspace_points.retain_grad()

    def post_backward(self, step, avatar, render_out):
        """
        Accumulate per-Gaussian grow signal (``grad2d`` viewspace or ``gradrgb`` color).

        Topology changes run in ``post_optimizer_step`` after ``optimizer.step()``.
        """
        surf = avatar.surface
        if not self._should_accumulate(step, surf) or render_out is None:
            return

        device = surf.h.device
        n_gaussian = len(surf.h)
        if self.state["grad_signal"] is None or len(self.state["grad_signal"]) != n_gaussian:
            self.reset_state(n_gaussian, device)

        grow_opt = self._grow_option()
        if grow_opt == "gradrgb":
            if surf.color.grad is None:
                return
            grads_norm = surf.color.grad.float().norm(dim=-1)
            self._accumulate_grad_signal(surf, render_out, grads_norm)
            return

        viewspace_points = render_out.get("viewspace_points")
        if viewspace_points is None:
            meta = render_out.get("meta")
            if isinstance(meta, dict):
                viewspace_points = meta.get("means2d")
        if viewspace_points is None or viewspace_points.grad is None:
            return

        rgb = render_out.get("rgb")
        if rgb is None:
            return

        width = int(rgb.shape[-1])
        height = int(rgb.shape[-2])
        grads_norm = _grads_norm_per_gaussian(viewspace_points.grad, width, height)
        self._accumulate_grad_signal(surf, render_out, grads_norm)

    @torch.no_grad()
    def post_optimizer_step(self, step, avatar, optimizer, ict_faces, ict):
        """
        Applies grow/prune/reset after optimizer.step().
        """
        if not self.cfg.gaussian_densify or optimizer is None:
            return

        surf = avatar.surface
        device = surf.h.device
        n_gaussian = len(surf.h)

        if (
            step >= self.cfg.gaussian_densify_start_iter
            and step <= self.cfg.gaussian_densify_stop_iter
        ):
            if step % self.cfg.gaussian_densify_every == 0:
                self._grow_and_prune(avatar, optimizer, ict_faces, ict)
            elif step % self.cfg.gaussian_prune_every == 0:
                is_prune = torch.sigmoid(surf.opacity.flatten()) < self.cfg.gaussian_prune_opa
                if is_prune.any():
                    n_pruned = prune_surf(surf, optimizer, is_prune, ict_faces, ict)
                    if self.cfg.gaussian_densify_max and len(surf.h) != n_gaussian:
                        n_gaussian = len(surf.h)
                        self.reset_state(n_gaussian, device)

            reset_iters = getattr(self.cfg, "gaussian_reset_iters", None) or []
            if step in reset_iters:
                reset_opacity_surf(surf, optimizer, self.cfg.gaussian_prune_opa * 2.0)
            elif (
                self.cfg.gaussian_reset_every > 0
                and step % self.cfg.gaussian_reset_every == 0
                and step > 0
            ):
                reset_opacity_surf(surf, optimizer, self.cfg.gaussian_prune_opa * 2.0)

    @torch.no_grad()
    def _grow_and_prune(self, avatar, optimizer, ict_faces, ict):
        """
        Executes standard 3DGS-style Densification (Duplicate & Split) and Pruning.
        """
        surf = avatar.surface
        device = surf.h.device
        n_current = len(surf.h)

        # 1. Identify which parts are ready to grow, if we are below max and have accumulated grads
        can_grow = (
            n_current < self.cfg.gaussian_densify_max
            and self.state["grad_signal"] is not None
            and self.state["count"] is not None
            and self.state["count"].sum() > 0
        )

        is_dupli = torch.zeros(n_current, dtype=torch.bool, device=device)
        is_split = torch.zeros(n_current, dtype=torch.bool, device=device)

        if (
            self._grow_option() == "gradrgb"
            and self.state["grad_signal"] is not None
            and self.state["count"] is not None
            and self.state["count"].sum() > 0
        ):
            count = self.state["count"]
            grads_avg = self.state["grad_signal"] / count.clamp_min(1)
            observed = count > 0
            grow_thr = self._grow_threshold()
            _print_tensor_grad_stats(grads_avg, "grad_avg_all")
            _print_tensor_grad_stats(grads_avg[observed], "grad_avg_observed")
            _print_tensor_grad_stats(self.state["grad_signal"], "grad_signal_sum")
            _print_tensor_grad_stats(count, "accum_count")
            n_above = int((grads_avg[observed] > grow_thr).sum().item())
            print(
                f"[Densify gradrgb] grow_threshold={grow_thr:.6e} "
                f"observed={int(observed.sum().item())}/{n_current} "
                f"above_threshold={n_above}"
            )

        if can_grow:
            grads = self.state["grad_signal"] / self.state["count"].clamp_min(1)
            is_grad_high = grads > self._grow_threshold()
            
            scale = torch.exp(surf.log_scale).max(dim=-1).values
            split_thresh = float(getattr(self.cfg, "geometry_max_scale", 0.008))
            is_small = scale <= split_thresh
            
            is_dupli = is_grad_high & is_small
            is_split = is_grad_high & ~is_small

            n_dupli = is_dupli.sum().item()
            n_split = is_split.sum().item()
            total_growth = n_dupli + n_split

            # Enforce budget / max gaussians
            if n_current + total_growth > self.cfg.gaussian_densify_max:
                budget = max(0, self.cfg.gaussian_densify_max - n_current)
                if budget == 0:
                    is_dupli.zero_()
                    is_split.zero_()
                else:
                    candidates_mask = is_dupli | is_split
                    candidate_grads = grads[candidates_mask]
                    if len(candidate_grads) > budget:
                        candidate_indices = torch.where(candidates_mask)[0]
                        selected = candidate_indices[torch.topk(candidate_grads, budget).indices]
                        selected_mask = torch.zeros_like(candidates_mask)
                        selected_mask[selected] = True
                        is_dupli = is_dupli & selected_mask
                        is_split = is_split & selected_mask

            # 1. Execute Duplication
            if is_dupli.any():
                duplicate_surf(surf, optimizer, is_dupli)

            # 2. Execute Splitting (with concatenated mask for duplicated parts ignored)
            is_split_cat = torch.cat([
                is_split,
                torch.zeros(is_dupli.sum().item(), dtype=torch.bool, device=device)
            ])
            if is_split_cat.any():
                split_surf(surf, optimizer, is_split_cat)

        # 3. Execute Opacity-based Pruning (always run on densify step!)
        is_prune = torch.sigmoid(surf.opacity.flatten()) < self.cfg.gaussian_prune_opa
        n_pruned = 0
        if is_prune.any():
            n_pruned = prune_surf(surf, optimizer, is_prune, ict_faces, ict)

        new_total = len(surf.h)
        if can_grow or n_pruned > 0:
            print(
                f"[Densify Step] {is_dupli.sum().item()} duplicated, {is_split.sum().item()} split, "
                f"{n_pruned} pruned. Total Gaussians: {n_current} -> {new_total}"
            )

        # Reset state with the new size
        self.reset_state(new_total, device)
