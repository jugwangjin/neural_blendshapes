"""Checkpoint resume helpers."""

from pathlib import Path

import torch

from model.build import avatar_checkpoint_layout_kwargs, sh_dim_from_avatar_state
from model.gaussian_avatar import GaussianAvatar
from training.checkpoint_io import avatar_n_from_state_dict, load_checkpoint


def resume_from_checkpoint(path, *, ict, deformer, tracker, device, cfg, payload=None):
    """
    Load ``save_checkpoint`` payload; avatar rebuilt from checkpoint tensors only
    (never ``from_ict``). Layout kwargs come from the **checkpoint cfg**, not the CLI cfg.

    Returns ``(avatar, global_step, resume_meta)`` where ``resume_meta`` holds
    ``path``, ``n_gaussians``, ``stage`` for stage-start audit.
    """
    path = Path(path)
    if payload is None:
        payload = torch.load(path, map_location=device, weights_only=False)

    ckpt_av = payload["avatar"]
    n_ckpt = avatar_n_from_state_dict(ckpt_av)
    ckpt_sh = sh_dim_from_avatar_state(ckpt_av)
    ckpt_cfg = payload["cfg"]
    layout_kw = avatar_checkpoint_layout_kwargs(ckpt_cfg)

    avatar = GaussianAvatar.from_checkpoint_state(
        ict,
        deformer,
        ckpt_av,
        **layout_kw,
    ).to(device)

    if int(avatar.n_gaussians) != n_ckpt:
        raise RuntimeError(
            f"resume: from_checkpoint_state n={avatar.n_gaussians} != ckpt face_idx={n_ckpt} "
            f"({path.name})"
        )
    if int(avatar.sh_dim) != ckpt_sh:
        raise RuntimeError(
            f"resume: sh_dim={avatar.sh_dim} != ckpt sh_dim={ckpt_sh} ({path.name})"
        )

    load_checkpoint(
        path,
        tracker=tracker,
        deformer=deformer,
        avatar=avatar,
        map_location=device,
        payload=payload,
        load_avatar=False,
    )

    if int(avatar.n_gaussians) != n_ckpt:
        raise RuntimeError(
            f"resume: after load_checkpoint n={avatar.n_gaussians} != ckpt face_idx={n_ckpt} "
            f"({path.name})"
        )

    global_step = int(payload["step"])
    stage_name = payload.get("stage", "?")
    meta = {
        "path": path.resolve(),
        "n_gaussians": n_ckpt,
        "stage": stage_name,
        "global_step": global_step,
    }
    print(
        f"resume: loaded {path.name} at global_step={global_step} (after stage {stage_name}) "
        f"[avatar n={n_ckpt} sh_dim={ckpt_sh}]"
    )
    return avatar, global_step, meta


def resolve_existing_input_dir(input_dir) -> Path:
    p = Path(input_dir)
    curr = p
    while curr != curr.parent:
        if curr.is_dir():
            if curr != p:
                print(f"Auto-resolved nonexistent input_dir '{p}' to existing path '{curr}'")
            return curr
        curr = curr.parent
    return p
