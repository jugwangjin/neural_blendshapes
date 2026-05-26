# Training stages

## Schedule (stages 0–3)

| Stage | Steps | Trainable | Frozen |
|-------|-------|-----------|--------|
| **0_bootstrap_pose** | 4000 | pose R+t (centroid pivot, w=1, tz=0) | expr, template, GS |
| **1_coarse_mesh** | 23000 | tracker, gamma, pose weight, template MLP, surface color/opacity/h | expression |
| **2_expression_detail** | 38000 | expression MLP, surface GS h/scale/color/opacity | tracker, template, pose |
| **3_view_appearance** | 8000 | color/opacity only | everything else |

**Total:** 73000 steps (+ `0_precompute` with 0 steps).

## Loss stack (all stages 0–3)

Same weights everywhere (`BASIC_LOSS` in `training/stages.py`):

| Key | Weight |
|-----|--------|
| `w_rgb` | 1.0 |
| `w_silhouette` | 10.0 |
| `w_mp_lmk` | 50.0 |
| `w_pie68_jaw` | 25.0 |
| `w_h` | 0.25 |
| `w_geometry` | 0.01 |
| `w_opacity` | 0.03 |
| `w_gamma_prior` | 0.5 |
| `w_pose_prior` | 0.1 |
| `w_template_smooth` | 0.05 |

Not used: `w_seg`, `w_sem_anchor`, `w_expr_neutral` / `leak` / `amp`.

Stage 0 only adds `w_pose_tz=2.0` for bootstrap.

Details: [basic_train_losses.md](basic_train_losses.md).

## Flags

- Stage 1: `geometry_lr_scale=0.1`, `lr_tracker=1e-4`
- Stage 2: `train_expression_deform=True`, `geometry_lr_scale=0.15`, no `train_gaussian_semantic`
- Stage 3: `train_gaussian_appearance` only; set `sh_degree` when renderer supports SH

Eyes: surface Gaussians on sclera + eye-occlusion (`GaussianAvatar.eyes is None`).

## Checkpoints & eval (`train.py`)

| When | Output |
|------|--------|
| Run start | `{output_root}/codes/<timestamp>/` — core `.py` + `config.json` + `STAGE_SCHEDULE.json` |
| Every `save_every` | `{output_root}/checkpoints/step_XXXXXX_<stage>.pt` |
| Stage end | `{output_root}/checkpoints/stage_<name>_end_step_XXXXXX.pt` + eval PNGs |

Eval: `{output_root}/renders/<stage>/step_XXXXXX/<frame>_compare.png`. Set `Config.output_root` only.
