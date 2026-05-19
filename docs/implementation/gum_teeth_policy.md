# Teeth vs gum Gaussians

## Policy

- **Teeth mesh**: no surface Gaussians, no template/expression deformer (`teeth_mask`).
- **Gums** (`mouth_interior` triangles): sparse surface Gaussians substitute teeth in RGB; learn larger `h` and color.

## Sampling

`classify_surface_triangles_batch` skips any triangle touching eyeball or **teeth** vertices.
Gum-only triangles get `code=0` → `k_mouth_interior` samples per tri (default **4**).

## h regularization (distance)

Semantic class init: gum verts → `lip`; teeth verts → `bg` (unused).

Gum Gaussians carry `avatar.h_sigma_scale = gum_h_sigma_scale` (default **4.0**), multiplying
allowed |h| in `loss_h_semantic` — looser than lip/socket (σ≈0.003).

Stage 1 anchor-only h loss: gums are not face anchors, so `h` is already free early.

## Deformer

`build_deform_reg_weight`: gums **0.01**, teeth masked in forward.

Config: `n_surface_gaussians_mouth_interior`, `gum_h_sigma_scale`.
