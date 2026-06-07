"""Checkpoint save/load for ``train.py`` (``out/checkpoints/*.pt``)."""

from pathlib import Path

import torch

# New tracker fields that older checkpoints omit; keep module init (usually zeros).
_TRACKER_OPTIONAL_KEYS = frozenset({"global_translation"})


def avatar_n_from_state_dict(state_dict) -> int:
    return int(state_dict["face_idx"].shape[0])


def avatar_color_stats(state_dict) -> dict:
    """Summarize stored SH / DC color (logit space in checkpoint)."""
    color = state_dict["color"]
    if color.ndim == 3:
        dc = color[:, 0, :]
        rest = color[:, 1:, :]
        sh_dim = int(color.shape[1])
        rest_rms = float(rest.pow(2).mean().sqrt().item()) if rest.numel() else 0.0
    else:
        dc = color
        sh_dim = 1
        rest_rms = 0.0
    dc_sig = torch.sigmoid(dc)
    return {
        "sh_dim": sh_dim,
        "color_shape": tuple(color.shape),
        "dc_sigmoid_mean": float(dc_sig.mean().item()),
        "dc_sigmoid_max": float(dc_sig.max().item()),
        "sh_rest_rms": rest_rms,
        "color_logit_std": float(color.float().std().item()),
    }


def format_avatar_color_stats(state_dict) -> str:
    s = avatar_color_stats(state_dict)
    return (
        f"dc_sigmoid_mean={s['dc_sigmoid_mean']:.4f} dc_sigmoid_max={s['dc_sigmoid_max']:.4f} "
        f"sh_rest_rms={s['sh_rest_rms']:.4f}"
    )


def warn_if_avatar_color_init_like(state_dict, *, tag: str = "") -> bool:
    """
    GB random init: U(0,1/255) DC only, sh_rest=0.
    SH stage-2+ can keep low DC mean while sh_rest is trained — do not flag those.
    """
    s = avatar_color_stats(state_dict)
    init_like = (
        s["dc_sigmoid_mean"] < 0.02
        and s["dc_sigmoid_max"] < 0.08
        and s["sh_rest_rms"] < 0.05
    )
    if init_like:
        prefix = f"{tag}: " if tag else ""
        print(
            f"WARNING {prefix}avatar color looks init-like ({format_avatar_color_stats(state_dict)}) "
            f"— RGB training likely did not update DC, or checkpoint is from pre-RGB stage"
        )
    return init_like


def load_tracker_state_dict(tracker, state_dict, *, tag: str = ""):
    """
    Backward-compatible tracker restore.

    Older checkpoints lack ``global_translation``; those keys stay at init (zeros).
    """
    incompatible = tracker.load_state_dict(state_dict, strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    optional_missing = missing & _TRACKER_OPTIONAL_KEYS
    other_missing = missing - _TRACKER_OPTIONAL_KEYS
    prefix = f"tracker load{tag}: "
    if optional_missing:
        print(f"{prefix}optional missing (init): {sorted(optional_missing)}")
    if other_missing:
        print(f"{prefix}missing (init): {sorted(other_missing)}")
    if unexpected:
        print(f"{prefix}ignored unexpected: {sorted(unexpected)}")


def load_avatar_state_dict(avatar, state_dict, *, tag: str = ""):
    from model.gaussian_avatar import GaussianAvatar

    if isinstance(avatar, GaussianAvatar):
        avatar.load_avatar_state_dict(state_dict)
        return
    avatar.load_state_dict(state_dict, strict=False)


def load_checkpoint(
    path,
    *,
    tracker=None,
    deformer=None,
    avatar=None,
    map_location=None,
    payload=None,
    load_avatar: bool = True,
):
    """
    Load ``save_checkpoint`` payload. Pass modules to restore ``state_dict``; returns the full dict.

    Set ``load_avatar=False`` when avatar was already loaded via ``from_checkpoint_state``.
    """
    path = Path(path)
    if payload is None:
        payload = torch.load(path, map_location=map_location or "cpu", weights_only=False)
    if tracker is not None and "tracker" in payload:
        load_tracker_state_dict(tracker, payload["tracker"], tag=f" ({path.name})")
    if deformer is not None and "deformer" in payload:
        deformer.load_state_dict(payload["deformer"])
    if load_avatar and avatar is not None and "avatar" in payload:
        load_avatar_state_dict(avatar, payload["avatar"], tag=f" ({path.name})")
    return payload


def save_checkpoint(
    path: Path,
    *,
    global_step: int,
    stage_name: str,
    tracker,
    deformer,
    avatar,
    cfg,
    spec=None,
    extra=None,
):
    from training.stage_spec_io import stage_spec_to_dict

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    av_sd = avatar.state_dict()
    n_gaussians = avatar_n_from_state_dict(av_sd)
    color = av_sd.get("color")
    color_shape = tuple(color.shape) if color is not None else None
    color_note = format_avatar_color_stats(av_sd)
    warn_if_avatar_color_init_like(av_sd, tag=f"save {path.name}")
    payload = {
        "step": global_step,
        "stage": stage_name,
        "n_gaussians": n_gaussians,
        "tracker": tracker.state_dict(),
        "deformer": deformer.state_dict(),
        "avatar": av_sd,
        "cfg": cfg,
    }
    if spec is not None:
        payload["stage_spec"] = stage_spec_to_dict(spec)
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    sh_note = getattr(spec, "sh_degree", None) if spec is not None else None
    print(
        f"saved {path} (n_gaussians={n_gaussians}, color={color_shape}, "
        f"{color_note}, stage_sh_degree={sh_note})"
    )
    return path
