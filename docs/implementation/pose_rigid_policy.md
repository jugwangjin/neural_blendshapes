# Head pose: scale fixed, R+t about centroid

## Motivation

Learning a global uniform scale on the ICT mesh interacted badly with a perspective camera:
depth and scale were ambiguous, and rotation about the world origin (with partial pose weights)
made the head appear to drift because the rotation center was unclear.

## Policy

| Item | Setting |
|------|---------|
| Mesh uniform scale | **Not applied** (`apply_pose_scale=False`, default all stages) |
| Pose residual | MLP `head_pose` → 6D rotation + 3D translation (jointly) |
| Rigid pivot | **Per-batch vertex centroid** (`apply_rigid_about_centroid`) |
| Optional `log_pose_scale` | Frozen unless `train_pose_scale=True` and `apply_pose_scale=True` |

## Bootstrap stage (`0_bootstrap_pose`)

- 4000 steps before `1_coarse_mesh`
- `fix_gamma_at_one`, no `train_expression_deform` / template / GS
- `pose_weight_one` → full rigid on all vertices (`w=1`)
- `pose_zero_tz` + `w_pose_tz` → limit depth sliding during landmark alignment
- Losses: high `w_mp_lmk`, moderate `w_iris` / `w_silhouette`, `w_rgb=0`

## Code paths

- `utils/so3.py`: `apply_rigid_about_centroid`
- `model/ict_deformer.py`: `rotate_about_centroid`, `pose_zero_tz` in `forward`
- `model/gaussian_avatar.py`: `use_pose_scale`, passes pose flags to deformer
- `training/stages.py`: `StageSpec` flags + `0_bootstrap_pose`
- `train.py`: reads `spec.*` into `avatar(...)`

## Re-enabling scale (not recommended)

Set on a `StageSpec`: `apply_pose_scale=True`, `train_pose_scale=True`, and add scale term via `w_pose_prior` with `apply_pose_scale` in loss cfg.
