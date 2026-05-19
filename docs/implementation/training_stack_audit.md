# Training stack audit (2026-03)

Active path: **MP → `TrackerCorrectionMLP` → `ICTDeformer` → `GaussianAvatar` (surface + shared `EyeTextureGaussians`) → `AvatarRenderer` (gsplat).**

Entry: `python train.py` (stage schedule in `training/stages.py`).

## Resolved / current state

| Area | Status |
|------|--------|
| `GaussianAvatar.forward` | Accepts `tracker_out=corr` **or** legacy `verts` + `gaze_uv_*`; integrated deformer path sets `mesh_xyz`. |
| Shared eye | Single `EyeTextureGaussians`; `uv`/`h` **buffers** (frozen); per-side output via `gaze_offset` + optional `gaze_refine_*`. |
| `training/apply.py` | **Fixed**: no `eyes.left/right`; no `uv.requires_grad`; optimizes color/opacity/log_scale/rotation + `gaze_refine_*`. |
| `ICTDeformer` | Replaces old `SupportGatedExpressionDeformer`; template MLP + per-AU expr MLP inside deformer. |
| `train.py` | Wires `avatar(tracker_out=corr)` — no separate `verts_out` + manual gaze pass. |

## Loss focus (sanity / early training)

Primary terms to watch in logs:

- `rgb` — rendering L1
- **`silhouette`** — render `alpha` vs dataset foreground mask (`*_mask.png`); **always weighted** via `w_silhouette` (never drop from schedule)
- `mp_lmk` — 3D MP landmarks → 2D
- `iris` — iris control pentagon → MP 468–477
- `h` — distance / semantic prior on surface (eye `h` fixed 0)
- `scale`, `opacity` — Gaussian reg
- `eye_uv` — soft box on **effective** per-side UV (barrier only)

Optional / stage-gated (set weight 0 for sanity): `seg`, `gamma_prior`, `pose_prior`, `gaze_residual`, `expr_*`, `template_smooth`, `sem_anchor`.

## Stale code (not on `train.py` path)

- `draw_mediapipe.py` — old `expression_deformer` / `template_deformer` API
- `model/expression_deform_mlp.py` — superseded by `ICTDeformer` (file may remain for reference)
- `processing/optimize_ict_expression_to_flame.py` — separate experiment

## Sanity script

```bash
python scripts/sanity_train_stack.py --check all
```

Steps: `compile` → `eye` (param/buffer) → `avatar` → `render` → `loss` (one batch backward with expr/seg priors zeroed).

## Assets

Required for training:

- `assets/ict_facekit_torch.npy` — `left/right_eyeball_indices`, `triangle_uv_local`, `face_material_name`
- `assets/ict_mediapipe_landmark_indices.npz` (or legacy long name in `config.mp_embedding`)
- `assets/default_camera.npz`
- `assets/ict_identity.npy`

Bake: `processing/ict_facekit_to_npy_full_head.py`, `processing/ict_mediapipe_lmk/bake_mediapipe_to_ict.py`.

## Eye training design (locked)

- **Do not** train sclera base `uv` or `h`.
- **Do** train `gaze_refine_left/right` when `train_eye_gaze=True` (and tracker gaze heads).
- Shared appearance: one `color`/`opacity`/`log_scale`/`rotation` bank for both eyes; right side uses `mirror_right_u` + separate `UVMesh` chart.
