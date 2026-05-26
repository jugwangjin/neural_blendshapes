"""Checkpoint save/load for ``train.py`` (``out/checkpoints/*.pt``)."""

from pathlib import Path

import torch


def load_checkpoint(path, *, tracker=None, deformer=None, avatar=None, map_location=None):
    """
    Load ``save_checkpoint`` payload. Pass modules to restore ``state_dict``; returns the full dict.
    """
    path = Path(path)
    payload = torch.load(path, map_location=map_location or "cpu", weights_only=False)
    if tracker is not None and "tracker" in payload:
        tracker.load_state_dict(payload["tracker"])
    if deformer is not None and "deformer" in payload:
        deformer.load_state_dict(payload["deformer"])
    if avatar is not None and "avatar" in payload:
        avatar.load_state_dict(payload["avatar"])
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
    extra=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": global_step,
        "stage": stage_name,
        "tracker": tracker.state_dict(),
        "deformer": deformer.state_dict(),
        "avatar": avatar.state_dict(),
        "cfg": cfg,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"saved {path}")
    return path
