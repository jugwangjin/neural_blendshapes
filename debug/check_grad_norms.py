"""
Single-stage backward pass profiler for RGB / grad2d densification thresholds.

Runs one training stage for ``--steps`` iterations (default 200), accumulates gradient
norm statistics, prints percentiles, and exits. No optimizer.step(), no multi-stage loop.

Run from repo root:
  python debug/check_grad_norms.py --input-dir /path/to/subject
  python debug/check_grad_norms.py --input-dir /path/to/subject --checkpoint out/checkpoints/stage_1_coarse_mesh_end_step_020000.pt
"""

import argparse
import sys
import os
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["LANG"] = "C.UTF-8"
os.environ["LC_ALL"] = "C.UTF-8"

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import Config
from dataset import build_train_dataset, collate_batch, move_batch_to_device
from rendering import GaussianRenderer
from losses.train_losses import compute_losses
from model.build import avatar_checkpoint_layout_kwargs
from model.expr_regions import build_expr_region_weight
from model.gaussian_avatar import GaussianAvatar
from model.ict_deformer import ICTDeformer
from model.ict_model import ICTFaceKitTorch
from model.tracker_mlp import TrackerCorrectionMLP
from training.apply import (
    apply_stage_requires_grad,
    build_optimizers,
    init_training_state,
    stage_loss_cfg,
    stage_needs_rasterization,
    stage_needs_surface_forward,
)
from training.checkpoint_io import load_checkpoint, load_tracker_state_dict
from training.stages import STAGE_SCHEDULE
from utils.camera import load_training_camera, training_camera_status
from losses.mediapipe_landmark_478 import build_mp_lmk_embedding
from losses.pie68_jaw_landmark import build_pie68_jaw_vertex_indices
from utils.sampling import count_surface_gaussians


def _tqdm_postfix(losses, step: int, n_show: int = 6) -> dict:
    items = [(k, losses[k].item()) for k in losses if k != "total"]
    items.sort(key=lambda x: -abs(x[1]))
    out = {"step": step, "loss": f"{losses['total'].item():.4f}"}
    for k, v in items[:n_show]:
        out[k] = f"{v:.4f}"
    return out


def _flatten_concat(xs):
    if len(xs) == 0:
        return None
    return torch.cat([x.reshape(-1).float() for x in xs], dim=0)


# torch.quantile() fails on very large tensors (e.g. 200×512×512 pixel grads).
_MAX_QUANTILE_SAMPLES = 2_000_000
_NONZERO_CHUNK = 10_000_000


def _nonzero_ratio(flat: torch.Tensor) -> float:
    n = flat.numel()
    if n == 0:
        return 0.0
    pos = 0
    for i in range(0, n, _NONZERO_CHUNK):
        pos += (flat[i : i + _NONZERO_CHUNK] > 0).sum().item()
    return pos / n


def _quantiles(flat: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    n = flat.numel()
    if n <= _MAX_QUANTILE_SAMPLES:
        return torch.quantile(flat, probs)
    g = torch.Generator(device=flat.device)
    g.manual_seed(0)
    idx = torch.randint(0, n, (_MAX_QUANTILE_SAMPLES,), generator=g, device=flat.device)
    return torch.quantile(flat[idx], probs)


def _describe(name: str, xs):
    flat = _flatten_concat(xs)
    if flat is None or flat.numel() == 0:
        print(f"[{name}] no samples")
        return
    n = flat.numel()
    probs = torch.tensor(
        [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99], device=flat.device
    )
    q = _quantiles(flat, probs)
    sampled = n > _MAX_QUANTILE_SAMPLES
    pct_note = f" (percentiles from {_MAX_QUANTILE_SAMPLES} random samples)" if sampled else ""
    nonzero = _nonzero_ratio(flat)
    print(
        f"[{name}] count={n} nonzero={nonzero:.4f}{pct_note} "
        f"min={flat.min().item():.6e} max={flat.max().item():.6e} "
        f"mean={flat.mean().item():.6e} std={flat.std(unbiased=False).item():.6e} "
        f"p01={q[0].item():.6e} p05={q[1].item():.6e} p10={q[2].item():.6e} "
        f"p25={q[3].item():.6e} p50={q[4].item():.6e} p75={q[5].item():.6e} "
        f"p90={q[6].item():.6e} p95={q[7].item():.6e} p99={q[8].item():.6e}"
    )


def _grad2d_norm_per_entry(means2d_grad: torch.Tensor, width: int, height: int) -> torch.Tensor:
    from training.densify import _viewspace_grad_norm_gb

    del width, height
    return _viewspace_grad_norm_gb(means2d_grad)


def _find_stage(name: str):
    for i, spec in enumerate(STAGE_SCHEDULE):
        if spec.name == name:
            return i, spec
    choices = [s.name for s in STAGE_SCHEDULE if s.steps > 0]
    raise ValueError(f"unknown stage {name!r}; choices: {choices}")


def _parse_split_cli(values):
    if values is None:
        return None
    if len(values) == 0:
        return None
    if len(values) == 1 and "," in values[0]:
        parts = [s.strip() for s in values[0].split(",") if s.strip()]
        return parts[0] if len(parts) == 1 else parts
    if len(values) == 1:
        return values[0]
    return list(values)


def parse_cli():
    p = argparse.ArgumentParser(description="Profile gradient norms (single stage, no training)")
    p.add_argument("--input-dir", type=Path, default=None)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument(
        "--train-split",
        nargs="*",
        metavar="SCENE",
        help="Train scene folder name(s)",
    )
    p.add_argument("--rebuild-mp-cache", action="store_true")
    p.add_argument(
        "--stage",
        default="1_coarse_mesh",
        help=f"Stage name (default: 1_coarse_mesh). Choices: {[s.name for s in STAGE_SCHEDULE if s.steps > 0]}",
    )
    p.add_argument("--steps", type=int, default=200, help="Number of backward passes (default: 200)")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint .pt (tracker/deformer/avatar state_dict)",
    )
    return p.parse_args()


def apply_cli(cfg: Config, args) -> Config:
    if args.input_dir is not None:
        cfg.input_dir = args.input_dir
    if args.output_root is not None:
        cfg.output_root = args.output_root
    train_split = _parse_split_cli(args.train_split)
    if train_split is not None:
        cfg.train_split = train_split
    if args.rebuild_mp_cache:
        cfg.rebuild_mp_cache = True
    return cfg


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


def main():
    args = parse_cli()
    cfg = apply_cli(Config(), args)
    cfg.input_dir = resolve_existing_input_dir(cfg.input_dir)
    from processing.ict_mediapipe_lmk.embedding_io import resolve_embedding_path


    cfg.mp_embedding = resolve_embedding_path(cfg.mp_embedding)
    assert cfg.batch_size == 1, "avatar/render path is single-mesh; set batch_size=1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stage_idx, spec = _find_stage(args.stage)
    n_steps = args.steps
    print(f"grad norm probe: stage {stage_idx} ({spec.name}), {n_steps} backward steps")

    dataset = build_train_dataset(cfg, train=True)
    from dataset.dataset_util import format_splits_label

    split_label = format_splits_label(cfg.train_split)
    print(
        f"dataset: {cfg.input_dir}/{{{split_label}}}/image — {len(dataset)} frames, "
        f"image_size={cfg.image_size}"
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    ict = ICTFaceKitTorch(npy_dir=str(cfg.ict_npy)).to(device)

    tracker = TrackerCorrectionMLP(
        mediapipe_to_ict=ict.mediapipe_to_ict,
        num_ict_expression=ict.num_expression,
        n_blendshapes=cfg.num_mp_blendshapes,
        gamma_min=cfg.gamma_min,
        gamma_max=cfg.gamma_max,
        use_landmarks=True,
    ).to(device)

    expr_region_weight = build_expr_region_weight(ict).to(device)
    deformer = ICTDeformer(
        ict,
        expr_region_weight,
        mediapipe_name_to_ict=str(cfg.mediapipe_name_to_ict),
        n_coeffs=cfg.num_ict_expressions,
    ).to(device)

    if args.checkpoint is not None:
        ckpt_path = args.checkpoint
        if not ckpt_path.is_file():
            for base in (cfg.checkpoint_dir, cfg.output_root / "checkpoints"):
                candidate = base / args.checkpoint
                if candidate.is_file():
                    ckpt_path = candidate
                    break
        payload = load_checkpoint(ckpt_path, map_location=device)
        if "tracker" in payload:
            load_tracker_state_dict(tracker, payload["tracker"])
        if "deformer" in payload:
            deformer.load_state_dict(payload["deformer"])
        avatar = GaussianAvatar.from_checkpoint_state(
            ict,
            deformer,
            payload["avatar"],
            **avatar_checkpoint_layout_kwargs(cfg),
        ).to(device)
        print(
            f"loaded checkpoint: {ckpt_path} "
            f"(step={payload.get('step')}, stage={payload.get('stage')}, "
            f"n_gaussians={avatar.n_gaussians})"
        )
    else:
        avatar = GaussianAvatar.from_ict(
            ict,
            deformer=deformer,
            k_face=cfg.n_surface_gaussians_per_face,
            k_head=cfg.n_surface_gaussians_per_head,
            k_mouth_socket=cfg.n_surface_gaussians_per_mouth_socket,
            k_mouth_interior=cfg.n_surface_gaussians_mouth_interior,
            k_teeth=cfg.n_surface_gaussians_per_teeth,
            k_eye_socket=cfg.n_surface_gaussians_per_eye_socket,
            k_eyeball_sclera=cfg.n_surface_gaussians_per_eyeball_sclera,
            k_eye_occlusion=cfg.n_surface_gaussians_per_eye_occlusion,
            n_semantic_classes=cfg.n_semantic_classes,
            gaussian_scale_knn_k=cfg.gaussian_scale_knn_k,
            gaussian_scale_knn_factor=cfg.gaussian_scale_knn_factor,
            face_center_init=cfg.gaussian_face_center_init,
            max_scale=cfg.geometry_max_scale,
        ).to(device)
        n_surface = count_surface_gaussians(
            ict,
            ict.faces,
            k_face=cfg.n_surface_gaussians_per_face,
            k_head=cfg.n_surface_gaussians_per_head,
            k_mouth_socket=cfg.n_surface_gaussians_per_mouth_socket,
            k_mouth_interior=cfg.n_surface_gaussians_mouth_interior,
            k_teeth=cfg.n_surface_gaussians_per_teeth,
            k_eye_socket=cfg.n_surface_gaussians_per_eye_socket,
            k_eyeball_sclera=cfg.n_surface_gaussians_per_eyeball_sclera,
            k_eye_occlusion=cfg.n_surface_gaussians_per_eye_occlusion,
            face_center_init=cfg.gaussian_face_center_init,
        )
        print(f"surface Gaussians: {n_surface}")
        init_training_state(avatar)
        try:
            from debug.sanity.region_colors import surface_gaussian_rgb, rgb_to_logit
            colors = surface_gaussian_rgb(
                avatar, ict, ict.template_reference_verts(), device, mp_embedding_path=cfg.mp_embedding
            )
            avatar.color.data.copy_(rgb_to_logit(colors))
            print("initialized surface colors from layout (no checkpoint)")
        except Exception as e:
            print(f"warning: layout color init skipped ({e})")

    renderer = GaussianRenderer(cfg, image_size=cfg.image_size, sh_degree=None).to(device)
    camera = load_training_camera(
        ict.expression_reference_verts(),
        path=cfg.camera_npz,
        width=cfg.image_size,
        height=cfg.image_size,
        device=device,
    )
    print(f"camera: {training_camera_status(cfg.camera_npz)}")

    mp_lmk_emb = build_mp_lmk_embedding(cfg.mp_embedding, device)
    pie68_jaw_vertex_idx = build_pie68_jaw_vertex_indices(ict, device)
    ict_faces = ict.faces.to(device)

    print(f"\n=== Stage {stage_idx}: {spec.name} ===")
    print(spec.description)

    renderer.set_sh_degree(spec.sh_degree)
    cfg.sh_degree = spec.sh_degree

    apply_stage_requires_grad(spec, tracker, deformer, avatar)
    mesh_optim, gaussian_optim = build_optimizers(spec, tracker, deformer, avatar)

    loss_cfg = stage_loss_cfg(spec)
    loss_cfg.image_size = cfg.image_size
    loss_cfg.mp_lmk_iris_weight = cfg.mp_lmk_iris_weight
    loss_cfg.silhouette_use_edt = cfg.silhouette_use_edt
    loss_cfg.silhouette_edt_w_ext = cfg.silhouette_edt_w_ext
    loss_cfg.silhouette_edt_w_int = cfg.silhouette_edt_w_int
    loss_cfg.silhouette_edt_max_dist_px = cfg.silhouette_edt_max_dist_px

    rgb_pixel_grad_norms = []
    rgb_color_param_grad_norms = []
    grad2d_entry_norms = []
    grad2d_gaussian_norms = []

    pbar = tqdm(total=n_steps, desc=f"grad probe {spec.name}", unit="step", dynamic_ncols=True)
    loader_iter = iter(loader)

    for step in range(1, n_steps + 1):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        batch = move_batch_to_device(batch, device)

        corr = tracker(
            mp_blendshape=batch["mp_blendshape"],
            mp_landmarks_2d=batch.get("mp_landmarks_2d"),
            mp_landmarks_3d=batch.get("mp_landmarks_3d"),
            world_to_cam=batch.get("world_to_cam"),
            mp_pose_raw=batch.get("mp_pose_raw"),
            mp_transform_matrix=batch.get("mp_transform_matrix"),
            force_gamma_one=spec.fix_gamma_at_one,
            additive_gamma_correction=getattr(spec, "additive_gamma_correction", False),
        )

        pose_weight_fixed = 1.0 if spec.pose_weight_one else None
        need_render = stage_needs_rasterization(loss_cfg)
        need_surface = stage_needs_surface_forward(loss_cfg)
        avatar_out = avatar(
            tracker_out=corr,
            apply_expression_deform=spec.train_expression_deform,
            use_pose_scale=spec.apply_pose_scale,
            pose_weight_fixed=pose_weight_fixed,
            rotate_about_centroid=spec.pose_rotate_about_centroid,
            pose_zero_tz=spec.pose_zero_tz,
            skip_surface=not need_surface,
        )
        expr_delta = avatar_out.get("expr_delta")

        render = None
        if need_render:
            render_semantic = loss_cfg.w_seg > 0
            render = renderer(avatar_out, camera, render_semantic=render_semantic)
            if render.get("rgb") is not None:
                render["rgb"].retain_grad()
            means2d = render.get("viewspace_points")
            if means2d is None and isinstance(render.get("meta"), dict):
                means2d = render["meta"].get("means2d")
            if means2d is not None:
                means2d.retain_grad()

        losses = compute_losses(
            loss_cfg,
            batch,
            render,
            avatar_out,
            camera,
            mp_lmk_emb,
            ict_faces,
            pie68_jaw_vertex_idx=pie68_jaw_vertex_idx,
            corr=corr,
            deformer=deformer,
            expr_delta=expr_delta,
            avatar=avatar,
            renderer=renderer,
        )

        if mesh_optim is not None:
            mesh_optim.zero_grad(set_to_none=True)
        if gaussian_optim is not None:
            gaussian_optim.zero_grad(set_to_none=True)

        losses["total"].backward()

        if render is not None:
            rgb = render.get("rgb")
            if rgb is not None and rgb.grad is not None:
                pix = rgb.grad.float().norm(dim=1).reshape(-1).detach().cpu()
                rgb_pixel_grad_norms.append(pix)

            means2d = render.get("viewspace_points")
            if means2d is None and isinstance(render.get("meta"), dict):
                means2d = render["meta"].get("means2d")
            if means2d is not None and means2d.grad is not None and rgb is not None:
                g2d_entry = _grad2d_norm_per_entry(
                    means2d.grad, width=int(rgb.shape[-1]), height=int(rgb.shape[-2])
                )
                if g2d_entry.numel() > 0:
                    g2d_e = g2d_entry.detach().cpu()
                    grad2d_entry_norms.append(g2d_e)

                    meta = render.get("meta")
                    gaussian_ids = meta.get("gaussian_ids") if isinstance(meta, dict) else None
                    if gaussian_ids is not None and gaussian_ids.numel() == g2d_entry.numel():
                        n = avatar.surface.n_gaussians
                        per_g = torch.zeros(n, device=gaussian_ids.device, dtype=g2d_entry.dtype)
                        per_g.index_add_(0, gaussian_ids.long(), g2d_entry)
                        grad2d_gaussian_norms.append(per_g.detach().cpu())

        color_grad = avatar.surface.color.grad
        if color_grad is not None:
            rgb_color_param_grad_norms.append(color_grad.float().norm(dim=-1).reshape(-1).detach().cpu())

        pbar.update(1)
        if step % cfg.log_every == 0 or step == n_steps:
            pbar.set_postfix(_tqdm_postfix(losses, step), refresh=False)

    pbar.close()

    print(f"\n=== Gradient statistics ({n_steps} steps, stage {spec.name}) ===")
    _describe("rgb_pixel_grad_norm", rgb_pixel_grad_norms)
    _describe("rgb_color_param_grad_norm", rgb_color_param_grad_norms)
    _describe("grad2d_entry_norm", grad2d_entry_norms)
    _describe("grad2d_gaussian_norm", grad2d_gaussian_norms)
    print("done.")


if __name__ == "__main__":
    main()
