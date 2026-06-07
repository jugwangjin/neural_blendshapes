# Teeth vs gum Gaussians

## Teeth surface Gaussians

- Triangles on ``teeth_indices`` → ``face_region_code=7``, ``is_teeth=True``.
- Initial ``h ~ Uniform(-teeth_h_radius, teeth_h_radius)`` along tooth mesh normal
  (``model/gaussian_h_init.init_teeth_h``; default **0.01 m ≈ 1 cm**).
- ``apply_h_constraint(train_h=False)`` keeps mouth socket + teeth ``h`` (others pinned to 0).
- Bad samples culled by semantic / opacity prune during early training.

Config: ``teeth_h_radius``, ``n_surface_gaussians_per_teeth``.

## Policy (legacy gum note)

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
