"""Load a trained run for inference render (tracking / control video)."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch

from model.build import avatar_checkpoint_layout_kwargs, build_deformer, build_ict, build_tracker
from model.gaussian_avatar import GaussianAvatar
from rendering import GaussianRenderer
from run_status import FINAL_STAGE_NAME
from training.apply import apply_inference_forward_flags
from training.checkpoint_io import load_checkpoint
from training.deformer_inference_cache import install_deformer_inference_cache
from training.stage_spec_io import resolve_render_stage_spec
from utils.camera import load_training_camera, training_camera_status


@dataclass
class RenderStack:
    cfg: object
    spec: object
    ict: object
    tracker: object
    deformer: object
    avatar: GaussianAvatar
    renderer: GaussianRenderer
    camera: object
    checkpoint: Path
    global_step: int


def _ckpt_step(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload.get("step", 0))


def _ckpt_n_gaussians(path: Path, payload=None) -> int:
    if payload is None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload["avatar"]["face_idx"].shape[0])


def resolve_checkpoint_path(output_root: Path, ckpt: Path | str) -> Path:
    """Resolve ``checkpoints/foo.pt``, ``foo.pt``, or absolute path under ``output_root``."""
    output_root = Path(output_root)
    ckpt = Path(ckpt)
    if ckpt.is_file():
        return ckpt.resolve()
    for candidate in (
        output_root / ckpt,
        output_root / "checkpoints" / ckpt.name,
        output_root / "checkpoints" / ckpt,
    ):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"checkpoint not found: {ckpt} (tried {output_root / ckpt}, "
        f"{output_root / 'checkpoints' / ckpt.name})"
    )


def final_stage_spec(output_root: Path, *, infer_ablation=None):
    from training.stage_spec_io import final_stage_spec as _final_stage_spec

    return _final_stage_spec(output_root, infer_ablation=infer_ablation)


def load_run_loss_overrides(output_root: Path, *, infer_ablation=None) -> dict:
    from training.stage_spec_io import load_run_loss_overrides as _load

    return _load(output_root, infer_ablation=infer_ablation)


def find_final_checkpoint(
    output_root: Path,
    ckpt: Path | str | None = None,
) -> Path:
    """Latest ``stage_{FINAL_STAGE_NAME}_end_step_*.pt`` under ``checkpoints/``."""
    output_root = Path(output_root)
    if ckpt is not None:
        return resolve_checkpoint_path(output_root, ckpt)

    ckpt_dir = output_root / "checkpoints"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"no checkpoints dir: {ckpt_dir}")

    candidates = sorted(ckpt_dir.glob("stage_*_end_step_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"no stage-end checkpoints under {ckpt_dir}")

    final_pattern = f"stage_{FINAL_STAGE_NAME}_end_step_"
    final_ckpts = [p for p in candidates if final_pattern in p.name]
    pool = final_ckpts if final_ckpts else candidates
    chosen = sorted(pool, key=_ckpt_step)[-1]

    n = _ckpt_n_gaussians(chosen)
    print(f"render checkpoint: {chosen.name} (step={_ckpt_step(chosen)}, n_gaussians={n})")
    return chosen


def load_run_for_render(
    output_root: Path,
    *,
    checkpoint: Path | None = None,
    device: torch.device | None = None,
    infer_ablation=None,
    cache_deformer: bool = True,
) -> RenderStack:
    output_root = Path(output_root)
    ckpt_path = find_final_checkpoint(output_root, checkpoint)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = deepcopy(payload["cfg"])
    cfg.output_root = output_root
    spec, spec_source = resolve_render_stage_spec(payload, output_root, infer_ablation=infer_ablation)
    print(
        f"render spec: {spec.name} sh_degree={spec.sh_degree}, "
        f"expr_deform={spec.train_expression_deform}, "
        f"color_pose={getattr(spec, 'train_color_pose', False)}, "
        f"color_expr={getattr(spec, 'train_color_expression', False)} "
        f"({spec_source})"
    )

    ict = build_ict(cfg, device)
    tracker = build_tracker(cfg, ict, device)
    deformer = build_deformer(cfg, ict, device)
    ckpt_av = payload["avatar"]
    ckpt_cfg = payload["cfg"]
    avatar = GaussianAvatar.from_checkpoint_state(
        ict,
        deformer,
        ckpt_av,
        **avatar_checkpoint_layout_kwargs(ckpt_cfg),
    ).to(device)
    load_checkpoint(
        ckpt_path,
        tracker=tracker,
        deformer=deformer,
        avatar=avatar,
        map_location=device,
        payload=payload,
        load_avatar=False,
    )
    fwd = apply_inference_forward_flags(avatar, spec)
    print(f"render forward flags: h_trainable={fwd['h_trainable']} (from spec.train_gaussian_h)")

    if cache_deformer:
        install_deformer_inference_cache(deformer, spec)

    renderer = GaussianRenderer(cfg, image_size=cfg.image_size, sh_degree=None).to(device)
    if spec.sh_degree is not None:
        renderer.set_sh_degree(spec.sh_degree)
        cfg.sh_degree = spec.sh_degree

    camera = load_training_camera(
        ict.expression_reference_verts(),
        path=cfg.camera_npz,
        width=cfg.image_size,
        height=cfg.image_size,
        device=device,
    )
    print(f"render load: avatar from checkpoint layout (n={avatar.n_gaussians}, sh_dim={avatar.sh_dim})")
    print(f"camera: {training_camera_status(cfg.camera_npz)}")
    if payload.get("stage_spec") is None:
        ckpt_stage = payload.get("stage")
        if ckpt_stage and spec.name != str(ckpt_stage):
            print(
                f"WARNING render spec name={spec.name!r} != checkpoint stage={ckpt_stage!r} "
                f"({spec_source})"
            )

    return RenderStack(
        cfg=cfg,
        spec=spec,
        ict=ict,
        tracker=tracker,
        deformer=deformer,
        avatar=avatar,
        renderer=renderer,
        camera=camera,
        checkpoint=ckpt_path,
        global_step=int(payload.get("step", 0)),
    )
