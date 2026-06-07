"""Build datasets, models, renderer, and landmark assets for training."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import build_train_dataset, collate_batch
from dataset.collate import batch_has_gt_normal
from losses.mediapipe_landmark_478 import build_mp_lmk_embedding
from losses.pie68_jaw_landmark import (
    build_pie68_jaw_vertex_indices,
    build_pie68_train_landmark_indices,
)
from model.build import build_avatar, build_deformer, build_ict, build_tracker, print_surface_gaussian_count
from rendering import GaussianRenderer
from training.apply import init_training_state
from training.mouth_coeff_debug import build_mouth_debug_indices
from training.resume import resume_from_checkpoint
from training.triangle_walking import TriangleWalker
from training.densify import BarycentricDensificationStrategy
from utils.camera import load_training_camera, training_camera_status
from utils.seed import dataloader_generator, worker_init_fn


@dataclass
class TrainingStack:
    cfg: object
    device: torch.device
    loader: DataLoader
    eval_loader: object
    ict: object
    tracker: object
    deformer: object
    avatar: object
    renderer: object
    camera: object
    mp_lmk_emb: dict
    pie68_jaw_vertex_idx: torch.Tensor
    pie68_vertex_idx: torch.Tensor
    pie68_protocol_idx: torch.Tensor
    ict_faces: torch.Tensor
    mouth_debug_idx: object
    triangle_walker: TriangleWalker
    densify_strategy: BarycentricDensificationStrategy
    eval_render_viz: dict
    resume_meta: dict | None = None


def build_train_loader(cfg):
    dataset = build_train_dataset(cfg, train=True)
    from dataset.dataset_util import format_splits_label

    split_label = format_splits_label(cfg.train_split)
    n_frames = len(dataset)
    print(
        f"dataset: ImageDataset ({cfg.dataset_type}) "
        f"{cfg.input_dir}/{{{split_label}}}/image — {n_frames} frames, image_size={cfg.image_size}"
    )
    train_gen = dataloader_generator(cfg.seed, train=True)
    nw = int(cfg.num_workers)
    persistent = nw > 0 and bool(getattr(cfg, "dataloader_persistent_workers", True))
    loader_kw = dict(
        dataset=dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=nw,
        pin_memory=cfg.pin_memory,
        generator=train_gen,
        worker_init_fn=worker_init_fn if nw > 0 else None,
        persistent_workers=persistent,
    )
    if nw > 0:
        loader_kw["prefetch_factor"] = int(getattr(cfg, "dataloader_prefetch_factor", 2))
    loader = DataLoader(**loader_kw)
    return loader


def build_eval_loader(cfg):
    eval_ds = build_train_dataset(cfg, train=False)
    if len(eval_ds) == 0:
        return None
    return DataLoader(
        eval_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=0,
    )


def init_avatar_colors_random(avatar, seed=0):
    """GB ``face_gs_model.create_from_face``: ``rand/255`` albedo on SH DC (logit storage)."""
    from debug.sanity.region_colors import rgb_to_logit

    n = avatar.n_gaussians
    dev = avatar.color.device
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(seed))
    albedos = torch.rand(n, 3, device=dev, generator=gen) / 255.0
    with torch.no_grad():
        if avatar.sh_dim > 1:
            avatar.color.data.zero_()
            avatar.color.data[:, 0, :] = rgb_to_logit(albedos)
        else:
            avatar.color.data.copy_(rgb_to_logit(albedos))
    print(f"Initialized Gaussian colors: random U(0,1/255) (GB-style), n={n}, seed={seed}.")


def init_avatar_colors(avatar, ict, device, mp_embedding_path: Path):
    from debug.sanity.region_colors import surface_gaussian_rgb, rgb_to_logit

    colors = surface_gaussian_rgb(
        avatar,
        ict,
        ict.template_reference_verts(),
        device,
        mp_embedding_path=mp_embedding_path,
    )
    avatar.color.data.copy_(rgb_to_logit(colors))
    print("Initialized Gaussian surface colors with layout region colors.")


def build_training_stack(cfg, device, *, resume_path=None):
    loader = build_train_loader(cfg)
    eval_loader = build_eval_loader(cfg)

    ict = build_ict(cfg, device)
    tracker = build_tracker(cfg, ict, device)
    deformer = build_deformer(cfg, ict, device)
    global_step = 0
    resume_meta = None
    if resume_path is not None:
        avatar, global_step, resume_meta = resume_from_checkpoint(
            resume_path,
            ict=ict,
            deformer=deformer,
            tracker=tracker,
            device=device,
            cfg=cfg,
        )
    else:
        avatar = build_avatar(cfg, ict, deformer, device)
        print_surface_gaussian_count(cfg, ict)
        init_training_state(avatar, cfg)
        if getattr(cfg, "gaussian_color_random_init", True):
            init_avatar_colors_random(avatar, seed=cfg.seed)
        else:
            init_avatar_colors(avatar, ict, device, cfg.mp_embedding)

    renderer = GaussianRenderer(cfg, image_size=cfg.image_size, sh_degree=None).to(device)
    print(
        f"gsplat: rasterize_mode={renderer.rasterize_mode} "
        f"(classic = 3DGS/GB, no mip antialiasing), packed={renderer.packed}"
    )
    camera = load_training_camera(
        ict.expression_reference_verts(),
        path=cfg.camera_npz,
        width=cfg.image_size,
        height=cfg.image_size,
        device=device,
    )
    print(f"camera: {training_camera_status(cfg.camera_npz)}")
    if not cfg.camera_npz.is_file():
        print(
            "  bake metrical crop: python processing/compute_camera_for_metrical_crop.py "
            "--apply-train-view --write-npz"
        )

    mp_lmk_emb = build_mp_lmk_embedding(cfg.mp_embedding, device)
    print(f"MP→ICT landmark embedding: {cfg.mp_embedding} ({mp_lmk_emb['mp_ids'].numel()} landmarks)")
    pie68_jaw_vertex_idx = build_pie68_jaw_vertex_indices(ict, device)
    pie68_vertex_idx, pie68_protocol_idx = build_pie68_train_landmark_indices(ict, device)
    mouth_debug_idx = build_mouth_debug_indices(ict)
    print(
        f"PIE-68 train landmarks: {pie68_vertex_idx.numel()} verts "
        f"(jaw 0:{ict.landmark_start} only; w_pie68_jaw)"
    )
    print(
        f"mouth_debug: stage 2 local<={cfg.mouth_debug_stage_local_max}, "
        f"MP jawOpen<={cfg.mouth_debug_jaw_open_max}, mouthClose<={cfg.mouth_debug_mouth_close_max}"
    )
    ict_faces = ict.faces.to(device)
    eval_render_viz = dict(
        mp_lmk_emb=mp_lmk_emb,
        pie68_jaw_vertex_idx=pie68_jaw_vertex_idx,
        ict_faces=ict_faces,
        ict=ict,
    )
    triangle_walker = TriangleWalker(ict_faces, ict.template_reference_verts(), max_iterations=3)
    densify_strategy = BarycentricDensificationStrategy(cfg)

    return TrainingStack(
        cfg=cfg,
        device=device,
        loader=loader,
        eval_loader=eval_loader,
        ict=ict,
        tracker=tracker,
        deformer=deformer,
        avatar=avatar,
        renderer=renderer,
        camera=camera,
        mp_lmk_emb=mp_lmk_emb,
        pie68_jaw_vertex_idx=pie68_jaw_vertex_idx,
        pie68_vertex_idx=pie68_vertex_idx,
        pie68_protocol_idx=pie68_protocol_idx,
        ict_faces=ict_faces,
        mouth_debug_idx=mouth_debug_idx,
        triangle_walker=triangle_walker,
        densify_strategy=densify_strategy,
        eval_render_viz=eval_render_viz,
        resume_meta=resume_meta,
    ), global_step


# batch_has_gt_normal used by train_step render branch
__all__ = ["TrainingStack", "batch_has_gt_normal", "build_training_stack"]
