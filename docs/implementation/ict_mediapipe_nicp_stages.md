# MediaPipe bake — staged face NICP

## Pipeline (`nicp.py`)

When `ict_npy_dict` is passed (bake always does):

1. **Stage 1** — `jawOpen` + per-iter `s,R,T`; inner 68 `[17:]` L1 + **PIE jawline `[0:16]` KNN** + weak chamfer.
2. **Stage 2** — `jawOpen` + identity + per-iter `s,R,T`; same losses + L2 on identity.
3. **Stage 3** — Large Steps vertex residual + inner L1 + jawline KNN (weak surf/chamfer).

Face patch only (`0:9409`); eyeball+ from pre-NICP `v_ict` (npy jaw+`flame_alignment`).

Stage 1–2 rebuild the face from **ICT neutral + blendshapes** into FLAME space (does not preserve per-vertex npy `flame_alignment` on the face).

## CLI (`bake_mediapipe_to_ict.py`)

| Flag | Default |
|------|---------|
| `--nicp_stage1_iters` | 150 |
| `--nicp_stage2_iters` | 400 |
| `--nicp_iterations` / `--nicp_stage3_iters` | 300 (`-1` → use iterations; `0` → no vertex stage) |

## UV debug textures

`ict_nicp_fit_68lmk_textured.obj` uses **seam/local `uvs`** wireframe background, not `triangle_uv_atlas` chart tiles (`use_atlas_chart=False` in `export_ict_68_texture`).

Atlas layout remains available via `ict_uv_layout_texture(..., use_atlas_chart=True)` for UV debug only.
