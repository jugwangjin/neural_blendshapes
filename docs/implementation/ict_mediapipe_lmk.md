# ICT MediaPipe landmark baker

## Purpose

Bake metrical-tracker MediaPipe landmarks (478 points including iris) onto ICT-FaceKit via NICP + barycentric transfer, with debug exports for visual QA.

## Fix: FLAME `shape_params` shape

`FLAME.__init__` expects `shape_params` with shape `(1, n_shape)` for `einsum('bl,mkl->bmk', ...)`.  
`bake_mediapipe_to_ict.py` previously passed `shape.squeeze(0)` (1D), which caused:

```
RuntimeError: einsum(): the number of subscripts in the equation (2) does not match the number of dimensions (1)
```

**Fix:** pass `shape` without squeezing (same as `train.py` / `dataset_train.shape_params`).

## NICP: FLAME vs ICT vertex count

FLAME canonical has ~5023 vertices; ICT face patch has 9409. Do **not** slice `v_flame[:9409]`.  
`nicp.py` uses paired FLAME/ICT landmarks for rigid init, KNN surface term to the full FLAME mesh, and vertex-normal loss via nearest FLAME points.

## Texture visualization

On `--export_debug` (default), outputs under `processing/ict_mediapipe_lmk/debug/`:

| File | Description |
|------|-------------|
| `flame_mediapipe_texture.png` | UV texture: MP contours + indices on FLAME canonical |
| `flame_mediapipe_textured.obj` (+ `.mtl`, `.png`) | Textured FLAME mesh |
| `ict_mediapipe_texture.png` | UV texture on NICP-fitted ICT mesh |
| `ict_mediapipe_textured.obj` | Textured ICT mesh |
| `flame_canonical.obj`, `ict_fit_to_flame.obj` | Untextured meshes |
| `mp_points_on_flame.ply`, `mp_points_on_ict.ply` | 3D landmark point clouds |

Drawing style follows [metrical-tracker mediapipe.jpg](https://github.com/Zielon/metrical-tracker/blob/master/mediapipe.jpg): green contours (lips, eyes, brows, face oval), red face points, cyan iris rings, white index labels.

- FLAME UV: `assets/canonical_eye_smpl.obj` (same topology as processed FLAME faces).
- ICT UV: `ict_facekit_torch.npy` (`uvs`, `uv_faces`).

## Layout

```
processing/
  ict_mediapipe_lmk/     # this baker
  flame/                 # FLAME decoder (FLAME2020 under flame/)
  metrical-tracker/      # FLAME MediaPipe embedding source
  large-steps-pytorch/   # NICP parameterization
model/                   # ICTFaceKitTorch
assets/                  # outputs (npz, canonical meshes)
```

Imports: `model.ict_model.ICTFaceKitTorch`, `flame.FLAME` (with `processing/` on `sys.path`). Shared paths: `processing/paths.py`.

## Run

From repo root (defaults point at `processing/metrical-tracker` and `processing/large-steps-pytorch`):

```bash
python processing/ict_mediapipe_lmk/bake_mediapipe_to_ict.py
```

Optional: `--skip_nicp`, `--texture_size 4096`, explicit `--metrical_root` / `--large_steps_root`.

## Modules

- `bake_mediapipe_to_ict.py` — CLI
- `metrical.py` — load MP embedding + iris vertices on FLAME
- `nicp.py` — ICT face patch → FLAME (Large Steps)
- `transfer.py` — project FLAME MP points onto ICT (barycentric)
- `texture_viz.py` — UV landmark bake + textured OBJ export
- `mediapipe_connections.py` — FACEMESH contour edges + iris rings
