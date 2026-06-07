"""Stage-2 debug: log MP / tracker ICT coeffs when jawOpen & mouthClose are small (closed/neutral)."""

from __future__ import annotations

from utils.mediapipe_blendshapes import default_mediapipe_mapping


def build_mouth_debug_indices(ict):
    mp = default_mediapipe_mapping()
    names = list(ict.expression_names)
    return {
        "jaw_mp": int(mp.name_to_idx["jawOpen"]),
        "mouth_close_mp": int(mp.name_to_idx["mouthClose"]),
        "jaw_ict": int(names.index("jawOpen")),
        "mouth_close_ict": int(names.index("mouthClose")),
    }


def mouth_debug_active(spec, stage_local, cfg) -> bool:
    if spec.name != "2_coarse_mesh":
        return False
    if not bool(getattr(cfg, "mouth_debug_enabled", False)):
        return False
    return int(stage_local) <= int(getattr(cfg, "mouth_debug_stage_local_max", 3000))


def log_small_mouth_coeff_batch(
    *,
    cfg,
    spec,
    stage_local,
    global_step,
    batch,
    corr,
    indices,
    stem=None,
):
    """
    Print one line when all samples in batch have small jawOpen & mouthClose (MP).

    Thresholds: ``mouth_debug_jaw_open_max``, ``mouth_debug_mouth_close_max``.
    """
    mp = batch["mp_blendshape"].detach().float()
    jaw_max = float(mp[:, indices["jaw_mp"]].max().item())
    close_max = float(mp[:, indices["mouth_close_mp"]].max().item())
    t_jaw = float(getattr(cfg, "mouth_debug_jaw_open_max", 0.15))
    t_close = float(getattr(cfg, "mouth_debug_mouth_close_max", 0.35))
    if jaw_max > t_jaw or close_max > t_close:
        return

    coeffs = corr["coeffs"].detach().float()
    gamma = corr["gamma"].detach().float()
    raw = corr.get("coeffs_raw")
    if raw is not None:
        raw = raw.detach().float()

    ji = indices["jaw_ict"]
    mi = indices["mouth_close_ict"]
    parts = [
        f"[mouth_debug] stage={spec.name} g={global_step} local={stage_local}",
        f"stem={stem!r}" if stem is not None else None,
        f"MP jawOpen_max={jaw_max:.4f} mouthClose_max={close_max:.4f}",
        f"ICT jaw={coeffs[:, ji].mean().item():.4f}",
        f"mouthClose={coeffs[:, mi].mean().item():.4f}",
        f"gamma_jaw={gamma[:, ji].mean().item():.4f}",
        f"gamma_close={gamma[:, mi].mean().item():.4f}",
    ]
    if raw is not None:
        parts.append(f"raw_jaw={raw[:, ji].mean().item():.4f}")
        parts.append(f"raw_close={raw[:, mi].mean().item():.4f}")
    print(" ".join(p for p in parts if p), flush=True)
