# Training stages

## Rationale

| Stage | Trainable | Frozen | Role |
|-------|-----------|--------|------|
| **1_coarse_mesh** | tracker, template MLP, pose weight, gaze, color/opacity, small `h` | expression | MP→ICT identity; `w_h`+template reg guide mesh; eye UV slide |
| **2A_expression** | expression MLP, `h`/scale, gaze refine, appearance | tracker, template, pose | AU residual for blendshape gap; distance detail continues |
| **2B_geometry_detail** | Gaussian `h`, scale, color, opacity, accessory | all mesh + gaze | View-independent splat detail (diffuse `sh_degree=None`) |
| **3_view_appearance** | color/opacity only | everything else | View-dependent appearance when SH enabled |

## Flags (see `training/stages.py`)

- Stage 1: `train_gaussian_geometry=True`, `geometry_lr_scale=0.08`, `train_eye_gaze=True`
- Stage 2A: no `train_tracker` / `train_template_deformer` / `train_pose_weight`
- Stage 2B: no `train_expression_deform` / `train_eye_gaze`
- Stage 3: `train_gaussian_appearance` only; set `sh_degree` when renderer supports it

## Step budget (default)

15000 + 8000 + 30000 + 8000 = 61000 steps (excluding stage 0).
