"""Render eval scenes after each training stage (no grad)."""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import build_train_dataset, collate_batch, move_batch_to_device
from dataset.dataset_util import rgb_to_srgb
from training.apply import stage_loss_cfg, stage_needs_rasterization
from training.checkpoint_io import save_checkpoint
from training.inference_render import render_gsplat_from_tracker_out, tracker_out_from_batch


def _write_obj(path, verts, faces):
    """Minimal OBJ export (1-based face indices)."""
    v = verts.detach().float().cpu().numpy()
    f = faces.detach().long().cpu().numpy()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for x, y, z in v:
            out.write(f"v {x} {y} {z}\n")
        for a, b, c in f:
            out.write(f"f {a + 1} {b + 1} {c + 1}\n")


@torch.no_grad()
def extract_template_mesh(deformer, ict):
    """
    Neutral template mesh (no tracker pose / expression MLP / per-frame coeffs).

    - ``ict_base``: jaw-closed ICT mesh (identity fixed zero)
    - ``template_full``: ``ict_base`` + ``template_mlp`` delta (GS embed target)
    """
    dtype = deformer.canonical_xyz_template.dtype
    device = deformer.canonical_xyz_template.device
    tpl_delta = deformer.template_delta()
    ict_base = ict.template_reference_verts().to(device=device, dtype=dtype)
    template_full = ict_base + tpl_delta
    return {
        "ict_base": ict_base,
        "template_delta": tpl_delta,
        "template_full": template_full,
    }


@torch.no_grad()
def save_eval_template_mesh(stage_dir, deformer, ict):
    """Write template OBJ once per eval step (not per frame)."""
    mesh = extract_template_mesh(deformer, ict)
    mesh_dir = Path(stage_dir) / "mesh"
    faces = ict.faces
    _write_obj(mesh_dir / "template_full.obj", mesh["template_full"], faces)
    _write_obj(mesh_dir / "template_ict_base.obj", mesh["ict_base"], faces)
    meta = (
        f"template_delta_l2={float(mesh['template_delta'].pow(2).mean().sqrt()):.6f}\n"
    )
    (mesh_dir / "template_meta.txt").write_text(meta, encoding="utf-8")
    return mesh_dir


def _resolve_deformer(deformer, avatar):
    if deformer is not None:
        return deformer
    return getattr(avatar, "deformer", None)


def _chw_to_bgra_uint8(t, alpha_t=None):
    """CHW linear RGB [0,1] → BGRA uint8 for display (sRGB gamma before *255)."""
    x = t.detach().float().clamp(0, 1).cpu()
    if x.ndim == 4:
        x = x[0]
    hwc = rgb_to_srgb(x.permute(1, 2, 0)).numpy()
    hwc = (np.clip(hwc, 0, 1) * 255.0).round().astype(np.uint8)
    bgr = cv2.cvtColor(hwc, cv2.COLOR_RGB2BGR)
    if alpha_t is not None:
        a = alpha_t.detach().float().clamp(0, 1).cpu()
        if a.ndim == 4:
            a = a[0]
        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]
        a_hw = (a.numpy() * 255.0).round().astype(np.uint8)
        if a_hw.shape[:2] == bgr.shape[:2]:
            return np.concatenate([bgr, a_hw[..., None]], axis=-1)
    a_hw = np.full(bgr.shape[:2] + (1,), 255, dtype=np.uint8)
    return np.concatenate([bgr, a_hw], axis=-1)


@torch.no_grad()
def render_eval_set(
    cfg,
    spec,
    tracker,
    avatar,
    renderer,
    camera,
    device,
    *,
    out_dir: Path,
    global_step: int,
    max_frames: int = 0,
    eval_loader=None,
    deformer=None,
    save_checkpoint_pt: bool | None = None,
    mp_lmk_emb=None,
    pie68_jaw_vertex_idx=None,
    ict_faces=None,
    ict=None,
):
    """
    Save eval PNGs under ``out_dir / {stage_name} / step_{global_step} /``:

    - ``{stem}_compare.png`` — GT | pred side-by-side (sRGB for display)
    - ``render/{stem}.png`` — pred only, no GT
    - ``mesh/template_full.obj`` — once per eval (not stage 1): ICT + template_mlp
    - ``{stem}_bootstrap_debug.jpg`` — stage 1 only: landmarks + mesh/GT mask overlay

    Dataset GT and renderer output are linear RGB; both are converted with
    ``rgb_to_srgb`` before writing PNGs.

    ``max_frames``: 0 = all eval frames; else cap count.

    ``1_bootstrap_template`` exports only ``*_bootstrap_debug.jpg`` (no gsplat/mesh).
    Other bootstrap stages (e.g. identity) export template mesh + checkpoint only.
    """
    loss_cfg = stage_loss_cfg(spec)
    needs_rgb = stage_needs_rasterization(loss_cfg)

    if eval_loader is None:
        eval_ds = build_train_dataset(cfg, train=False)
        n = len(eval_ds)
        if n == 0:
            from dataset.dataset_util import format_splits_label

            split_label = format_splits_label(getattr(cfg, "eval_split", "test"))
            print(
                f"eval render skipped [{spec.name}]: no frames in "
                f"{cfg.input_dir}/{{{split_label}}}/image"
            )
            return

        if max_frames > 0:
            n = min(n, max_frames)

        loader = DataLoader(
            eval_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_batch,
            num_workers=0,
        )
    else:
        loader = eval_loader
        eval_ds = loader.dataset
        n = len(eval_ds)
        if n == 0:
            return
        if max_frames > 0:
            n = min(n, max_frames)

    stage_dir = Path(out_dir) / spec.name / f"step_{global_step:06d}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    deformer = _resolve_deformer(deformer, avatar)
    mesh_dir = None
    if deformer is not None and spec.name != "1_bootstrap_template":
        mesh_dir = save_eval_template_mesh(stage_dir, deformer, deformer.ict)

    do_save_pt = save_checkpoint_pt
    if do_save_pt is None:
        do_save_pt = bool(getattr(cfg, "save_eval_checkpoint", False))
    if do_save_pt and deformer is not None:
        ckpt_path = stage_dir / f"eval_step_{global_step:06d}.pt"
        save_checkpoint(
            ckpt_path,
            global_step=global_step,
            stage_name=spec.name,
            tracker=tracker,
            deformer=deformer,
            avatar=avatar,
            cfg=cfg,
            spec=spec,
            extra={"eval_render": True},
        )

    if spec.name == "1_bootstrap_template":
        if mp_lmk_emb is None or ict_faces is None:
            print(
                f"eval render skipped [{spec.name}]: mp_lmk_emb and ict_faces required"
            )
            return

        from training.landmark_debug_viz import (
            eyelash_exclude_vertex_ids,
            save_landmark_debug_image,
        )

        saved = 0
        for batch in loader:
            if saved >= n:
                break
            batch = move_batch_to_device(batch, device)
            corr = tracker(
                mp_blendshape=batch["mp_blendshape"],
                mp_landmarks_2d=batch.get("mp_landmarks_2d"),
                mp_landmarks_3d=batch.get("mp_landmarks_3d"),
                world_to_cam=batch.get("world_to_cam"),
                mp_pose_raw=batch.get("mp_pose_raw"),
                mp_transform_matrix=batch.get("mp_transform_matrix"),
                force_gamma_one=spec.fix_gamma_at_one,
                use_global_translation_param=getattr(spec, "use_global_translation_param", False),
                additive_gamma_correction=getattr(spec, "additive_gamma_correction", False),
            )
            pose_weight_fixed = 1.0 if spec.pose_weight_one else None
            avatar_out = avatar(
                tracker_out=corr,
                apply_expression_deform=spec.train_expression_deform,
                use_pose_scale=spec.apply_pose_scale,
                pose_weight_fixed=pose_weight_fixed,
                rotate_about_centroid=spec.pose_rotate_about_centroid,
                pose_zero_tz=spec.pose_zero_tz,
                enable_color_pose=getattr(spec, "train_color_pose", False),
                enable_color_expression=getattr(spec, "train_color_expression", False),
                skip_surface=True,
            )
            paths = batch.get("path", [f"frame_{saved:05d}"])
            stem = Path(paths[0]).stem if paths else f"frame_{saved:05d}"
            save_landmark_debug_image(
                stage_dir / f"{stem}_bootstrap_debug.jpg",
                avatar_out["mesh_xyz"],
                ict_faces,
                batch["mp_landmarks_2d"],
                mp_lmk_emb,
                camera,
                cfg.image_size,
                batch["image"][0],
                batch.get("mask")[0] if batch.get("mask") is not None else None,
                exclude_vertex_ids=eyelash_exclude_vertex_ids(ict) if ict is not None else None,
                jaw_vertex_idx=pie68_jaw_vertex_idx,
                landmark_fa=batch.get("landmark"),
                jaw_score_thresh=getattr(cfg, "pie68_jaw_score_thresh", 0.3),
            )
            saved += 1

        extra = (
            f", checkpoint {stage_dir / f'eval_step_{global_step:06d}.pt'}"
            if do_save_pt and deformer is not None
            else ""
        )
        print(
            f"eval render [{spec.name}]: {saved} frames -> {stage_dir} "
            f"(bootstrap_debug only){extra}"
        )
        return

    render_dir = stage_dir / "render"
    render_dir.mkdir(parents=True, exist_ok=True)

    if not needs_rgb:
        extra = f", checkpoint {stage_dir / f'eval_step_{global_step:06d}.pt'}" if do_save_pt and deformer is not None else ""
        mesh_note = f", mesh -> {mesh_dir}" if mesh_dir is not None else ""
        print(
            f"eval render [{spec.name}]: mesh/checkpoint only (no gsplat){extra}{mesh_note}"
        )
        return

    saved = 0
    for batch in loader:
        if saved >= n:
            break

        batch = move_batch_to_device(batch, device)

        corr = tracker_out_from_batch(tracker, batch, spec)
        render = render_gsplat_from_tracker_out(
            avatar, renderer, camera, corr, spec, composite=True
        )
        pred_bgra = _chw_to_bgra_uint8(render["rgb"], render.get("alpha"))
        gt_bgra = _chw_to_bgra_uint8(batch["image"], batch.get("mask"))

        paths = batch.get("path", [f"frame_{saved:05d}"])
        stem = Path(paths[0]).stem if paths else f"frame_{saved:05d}"
        cv2.imwrite(str(stage_dir / f"{stem}_compare.png"), np.concatenate([gt_bgra, pred_bgra], axis=1))
        cv2.imwrite(str(render_dir / f"{stem}.png"), pred_bgra)
        saved += 1

    extra = f", checkpoint {stage_dir / f'eval_step_{global_step:06d}.pt'}" if do_save_pt and deformer is not None else ""
    mesh_note = f", mesh -> {mesh_dir}" if mesh_dir is not None else ""
    print(
        f"eval render [{spec.name}]: {saved} frames -> {stage_dir} "
        f"(compare + render/){extra}{mesh_note}"
    )


def training_render_stage_dir(output_root: Path, stage_name: str, global_step: int) -> Path:
    """``{output_root}/renders/{stage}/step_{global_step:06d}`` — same as ``Config.eval_render_dir`` layout."""
    return Path(output_root) / "renders" / stage_name / f"step_{int(global_step):06d}"


def training_render_pred_dir(output_root: Path, stage_name: str, global_step: int) -> Path:
    """Gsplat pred PNGs: ``.../render/{stem}.png`` (right half of ``{stem}_compare.png``)."""
    return training_render_stage_dir(output_root, stage_name, global_step) / "render"


def list_training_render_pred_stems(output_root: Path, stage_name: str, global_step: int) -> list[str]:
    d = training_render_pred_dir(output_root, stage_name, global_step)
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.png"))
