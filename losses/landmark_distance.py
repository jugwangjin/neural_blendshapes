"""2D landmark distances in normalized UV [0,1] (per-point Euclidean)."""

import torch

# Feng et al. CVPR 2018 Wing loss (Eq. 5): w=10, epsilon=2 in pixel coordinates.
WING_W_PX_DEFAULT = 10.0
WING_EPS_PX_DEFAULT = 2.0


def wing_uv_from_pixels(wing_w_px, wing_eps_px, image_size):
    """Paper params (pixels) → UV for distance d = ||pred-target||_2 in [0,1] frame."""
    s = float(image_size)
    return float(wing_w_px) / s, float(wing_eps_px) / s


def wing_loss_uv(d, wing_w, wing_eps):
    """
    Eq. (5): wing(x)=w*ln(1+|x|/ε) if |x|<w else |x|-C, C=w-w*ln(1+w/ε).
    d: non-negative scalar distance (UV).
    """
    w = d.new_tensor(float(wing_w))
    e = d.new_tensor(float(wing_eps))
    c = w - w * torch.log1p(w / e)
    return torch.where(d < w, w * torch.log1p(d / e), d - c)


def point_uv_distance(
    pred,
    target,
    metric="wing",
    *,
    eps=1e-4,
    wing_w_px=WING_W_PX_DEFAULT,
    wing_eps_px=WING_EPS_PX_DEFAULT,
    image_size=None,
    wing_w=None,
    wing_eps=None,
):
    """
    pred, target: [B, N, 2] UV. Returns [B, N] scalar distance per point.

    metric:
      - ``l2``: ||pred-target||_2^2 in UV (MSE on x,y; small errors → small loss)
      - ``l1``: ||pred-target||_2 (Euclidean UV distance)
      - ``charbonnier`` / ``smooth_l1``: sqrt(d^2 + eps^2) (differentiable L1)
      - ``wing``: Feng et al. Wing loss on d; use ``wing_*_px`` + ``image_size``
        (default w=10px, ε=2px from arXiv:1711.06753)

    Do not set ε too small in pixels (paper: unstable / exploding grads).
    """
    diff2 = (pred - target).pow(2).sum(dim=-1)
    if metric == "l2":
        return diff2
    d = diff2.add(1e-12).sqrt()
    if metric == "l1":
        return d
    if metric in ("charbonnier", "smooth_l1"):
        e = float(eps)
        return torch.sqrt(d * d + e * e)
    if metric == "wing":
        if wing_w is None or wing_eps is None:
            if image_size is None:
                raise ValueError("wing metric requires image_size with wing_w_px / wing_eps_px")
            wing_w, wing_eps = wing_uv_from_pixels(wing_w_px, wing_eps_px, image_size)
        return wing_loss_uv(d, wing_w, wing_eps)
    raise ValueError(f"unknown landmark metric {metric!r}")


def weighted_landmark_loss(
    pred,
    target,
    *,
    valid=None,
    point_weight=None,
    metric="smooth_l1",
    eps=1e-4,
    wing_w_px=WING_W_PX_DEFAULT,
    wing_eps_px=WING_EPS_PX_DEFAULT,
    image_size=None,
    wing_w=None,
    wing_eps=None,
):
    """Weighted mean of per-point UV distances."""
    dist = point_uv_distance(
        pred,
        target,
        metric,
        eps=eps,
        wing_w_px=wing_w_px,
        wing_eps_px=wing_eps_px,
        image_size=image_size,
        wing_w=wing_w,
        wing_eps=wing_eps,
    )
    w = torch.ones_like(dist)
    if point_weight is not None:
        w = w * point_weight.view(1, -1).to(device=dist.device, dtype=dist.dtype)
    if valid is not None:
        w = w * valid.to(device=dist.device, dtype=dist.dtype)
    finite = torch.isfinite(pred).all(dim=-1) & torch.isfinite(target).all(dim=-1)
    w = w * finite.to(dtype=dist.dtype)
    denom = w.sum().clamp(min=1.0)
    return (dist * w).sum() / denom


def robust_l1(pred, target, valid=None, point_weight=None, eps=1e-4, metric=None, **kwargs):
    """Backward-compatible alias; default metric is ``smooth_l1`` (Charbonnier on UV distance)."""
    m = metric if metric is not None else "smooth_l1"
    return weighted_landmark_loss(
        pred,
        target,
        valid=valid,
        point_weight=point_weight,
        metric=m,
        eps=eps,
        wing_w_px=kwargs.get("wing_w_px", WING_W_PX_DEFAULT),
        wing_eps_px=kwargs.get("wing_eps_px", WING_EPS_PX_DEFAULT),
        image_size=kwargs.get("image_size"),
        wing_w=kwargs.get("wing_w"),
        wing_eps=kwargs.get("wing_eps"),
    )
