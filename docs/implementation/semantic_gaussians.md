# Gaussian semantics + eye gaze

## Head face Gaussians

- No per-expression Gaussian delta; follow mesh deformation.
- Semantic init from ICT `vertex_parts` via `face_idx` + barycentric mix (`rendering/gaussian_semantics.py`).
- Hybrid labels: **skin/lip/eye/iris anchored**; hair/accessory learnable.
- `w_sem_anchor` weak CE toward init labels on frozen dims.

## Eye texture Gaussians

- **Fixed** semantic one-hot: iris control points → `iris`, rest → `eye` (sclera).
- Texture mesh = `vertex_parts` **6 / 7 only** (eyeball), not eye sockets 3/4.
- **Gaze**: `base_gaze_from_mediapipe(eyeLook*)` + small MLP residual → `GaussianAvatar.forward(..., gaze_uv_left, gaze_uv_right)`.
- ICT `apply_eyeball_rotation=False` by default.

```text
MP eyeLook → GazeCalibrator → base gaze
tracker MLP → gaze residual
→ uv_eff = uv + gaze_uv
```

## gsplat semantic pass

Two-pass RGB + feature render. Foreground semantic map:

```text
semantic_prob = semantic_composite / (alpha + eps)
```

Segmentation loss uses `render["semantic_prob"]`.

## Expression deformer mask

`build_expr_region_weight(ict)` uses `ICT_PART_TO_SEMANTIC` + `EXPR_ALLOW_BY_SEMANTIC` (hair/accessory/iris = 0).

Combined with AU support gate in `SupportGatedExpressionDeformer`.
