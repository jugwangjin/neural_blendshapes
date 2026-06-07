"""Periodic landmark / silhouette debug images during bootstrap stages."""

from training.landmark_debug_viz import (
    eyelash_exclude_vertex_ids,
    save_landmark_debug_image,
)


def maybe_save_bootstrap_debug(state, avatar_out, corr, batch):
    spec = state.spec
    cfg = state.cfg
    stack = state.stack
    global_step = state.global_step

    if spec.name not in ("0_bootstrap_identity", "1_bootstrap_template"):
        return
    if global_step % 100 != 0:
        return

    dbg_dir = cfg.eval_render_dir / "bootstrap_debug"
    dbg_dir.mkdir(parents=True, exist_ok=True)
    exclude = eyelash_exclude_vertex_ids(stack.ict)
    dbg_path = dbg_dir / f"step_{global_step:06d}_{spec.name}.jpg"

    save_landmark_debug_image(
        dbg_path,
        avatar_out["mesh_xyz"],
        stack.ict_faces,
        batch["mp_landmarks_2d"],
        stack.mp_lmk_emb,
        stack.camera,
        cfg.image_size,
        batch["image"][0],
        batch.get("mask")[0] if batch.get("mask") is not None else None,
        exclude_vertex_ids=exclude,
        jaw_vertex_idx=stack.pie68_jaw_vertex_idx,
        landmark_fa=batch.get("landmark"),
        jaw_score_thresh=cfg.pie68_jaw_score_thresh,
    )
