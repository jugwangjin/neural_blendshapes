"""Image-space segmentation losses on gsplat-rendered semantic features."""

import torch
import torch.nn.functional as F


def exclude_flare_parts_from_mask(part_label, mask, exclude_parts=None):
    """
    Drop FLARE part ids from a boolean ``[B,H,W]`` supervision mask.

    Default: eyeglasses (part 6) — no mesh/GS class; must not contribute to seg loss.
    """
    from dataset.flare_semantic import SEG_EXCLUDE_FLARE_PARTS

    if exclude_parts is None:
        exclude_parts = SEG_EXCLUDE_FLARE_PARTS
    pl = part_label.long()
    if pl.ndim == 2:
        pl = pl.unsqueeze(0)
    keep = torch.ones_like(pl, dtype=torch.bool)
    for pid in exclude_parts:
        keep = keep & (pl != int(pid))
    if mask is None:
        return keep
    m = mask
    if m.ndim == 2:
        m = m.unsqueeze(0)
    return m & keep


def _nchw_mask(mask, pred_shape, dtype, device):
    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 3:
        mask = mask[:, None] if mask.shape[0] != 1 else mask[None]
    if mask.shape[1] != 1:
        mask = mask[:, :1]
    if mask.shape[-2:] != pred_shape[-2:]:
        mask = F.interpolate(mask.float(), size=pred_shape[-2:], mode="nearest")
    return mask.to(device=device, dtype=dtype)


def loss_segmentation_l2(render_accum, target_label, valid_mask=None, ignore_index=None):
    """
    L2 on alpha-blended one-hot **accum** (``render["semantic"]``, single-pass with RGB), not ``accum / alpha``.

    ``render_accum``: [B, K, H, W] = Σ w_i f_i (zero where no splats — unlike normalized CE).
    ``target_label``: [B, H, W] class ids; ``ignore_index`` and ``valid_mask==False`` pixels skipped.

    If every pixel is invalid (e.g. frame is all background / excluded parts), returns zero
    without dividing by zero.
    """
    if ignore_index is None:
        from rendering.semantic import SEMANTIC_IGNORE_INDEX

        ignore_index = SEMANTIC_IGNORE_INDEX
    k = int(render_accum.shape[1])
    if target_label.ndim == 2:
        target = target_label.long().unsqueeze(0)
    else:
        target = target_label.long()
    target_oh = F.one_hot(target.clamp(min=0), num_classes=k).permute(0, 3, 1, 2).float()
    diff = (render_accum - target_oh).pow(2)
    valid = torch.ones(
        target.shape[0],
        1,
        target.shape[-2],
        target.shape[-1],
        dtype=torch.bool,
        device=render_accum.device,
    )
    if valid_mask is not None:
        vm = valid_mask
        if vm.ndim == 2:
            vm = vm.unsqueeze(0)
        valid = valid & vm.unsqueeze(1)
    valid = valid & (target != int(ignore_index)).unsqueeze(1)
    n_valid = valid.sum()
    if n_valid.item() == 0:
        return render_accum.sum() * 0.0
    denom = valid.float().sum()
    return (diff * valid.float()).sum() / denom


def loss_segmentation_l1(render_accum, target_label, valid_mask=None, ignore_index=None):
    """L1 on alpha-blended one-hot accum (same layout as ``loss_segmentation_l2``)."""
    if ignore_index is None:
        from rendering.semantic import SEMANTIC_IGNORE_INDEX

        ignore_index = SEMANTIC_IGNORE_INDEX
    k = int(render_accum.shape[1])
    if target_label.ndim == 2:
        target = target_label.long().unsqueeze(0)
    else:
        target = target_label.long()
    target_oh = F.one_hot(target.clamp(min=0), num_classes=k).permute(0, 3, 1, 2).float()
    diff = (render_accum - target_oh).abs()
    valid = torch.ones(
        target.shape[0],
        1,
        target.shape[-2],
        target.shape[-1],
        dtype=torch.bool,
        device=render_accum.device,
    )
    if valid_mask is not None:
        vm = valid_mask
        if vm.ndim == 2:
            vm = vm.unsqueeze(0)
        valid = valid & vm.unsqueeze(1)
    valid = valid & (target != int(ignore_index)).unsqueeze(1)
    if valid.sum().item() == 0:
        return render_accum.sum() * 0.0
    denom = valid.float().sum()
    return (diff * valid.float()).sum() / denom


def loss_full_face_region(render_region, target_mask, *, alpha_min: float = 0.02):
    """
    One-sided full-face attribution (anti side-cheating).

    ``render_region`` channel 0 = skin-face + eye-occlusion GS only (ICT codes 4, 6).
    Target = ``full_face_region_mask`` (FLARE incl. mouth interior 11, lips 12/13).

    Penalize only where the image is in the face mask but in-region Gaussian contribution is low.
    Do not penalize in-region Gaussians outside the mask (e.g. bangs).
    Mouth/socket/gum Gaussians (0, 1) do not satisfy this loss on interior/lip pixels.
    """
    pred_face = render_region["expected"][:, :1].clamp(0.0, 1.0)
    alpha = render_region["alpha"].detach()
    target = _nchw_mask(target_mask, pred_face.shape, pred_face.dtype, pred_face.device)
    visible = (alpha > float(alpha_min)).to(dtype=pred_face.dtype)
    face_px = (target > 0.5).to(dtype=pred_face.dtype)
    valid = visible * face_px
    denom = valid.sum().clamp(min=1.0)
    return ((1.0 - pred_face) * valid).sum() / denom


def loss_lip_mouth_leak(
    render_accum,
    part_label,
    valid_mask=None,
    *,
    mouth_class_idx: int = 1,
):
    """
    Penalize mouth-interior semantic contribution on FLARE lip pixels (parts 12/13).

    This directly targets the common failure where teeth / mouth-socket Gaussians
    explain lip pixels and bake white teeth into the lip color.
    """
    if part_label is None:
        return render_accum.sum() * 0.0
    pl = part_label
    if pl.ndim == 2:
        pl = pl.unsqueeze(0)
    if pl.shape[-2:] != render_accum.shape[-2:]:
        pl = F.interpolate(
            pl.unsqueeze(1).float(),
            size=render_accum.shape[-2:],
            mode="nearest",
        ).squeeze(1).long()
    lip = (pl == 12) | (pl == 13)
    if valid_mask is not None:
        vm = valid_mask
        if vm.ndim == 2:
            vm = vm.unsqueeze(0)
        if vm.shape[-2:] != render_accum.shape[-2:]:
            vm = F.interpolate(
                vm.unsqueeze(1).float(),
                size=render_accum.shape[-2:],
                mode="nearest",
            ).squeeze(1) > 0.5
        lip = lip & vm
    denom = lip.float().sum().clamp(min=1.0)
    mouth = render_accum[:, int(mouth_class_idx)].clamp(min=0.0)
    return (mouth * lip.float()).sum() / denom
