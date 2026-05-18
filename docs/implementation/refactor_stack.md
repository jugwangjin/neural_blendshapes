# MediaPipe-only + UVH/3DGS refactor

## Active pipeline

```
MP cache (52 bs + 478 lmk + pose)
  → TrackerCorrectionMLP (gamma coeffs, pose/gaze residual)
  → ICTDeformer (FACS + template offset + pose weight)
  → GaussianAvatar (UVH face + eye texture Gaussians)
  → GaussianRenderer (wrapper; vendor = gaussian-splatting submodule)
  → losses: RGB, MP lmk, iris, mask, h, scale/opacity, eye UV barrier
```

## Key modules

| Module | Role |
|--------|------|
| `model/tracker_mlp.py` | `C' = C^gamma`, no image encoder |
| `model/ict_deformer.py` | MP→ICT expression, pose weight, no feature slicing |
| `model/pose_weight.py` | Per-vertex w∈[0,1] |
| `model/gaussian_avatar.py` | UVH + eye texture spaces |
| `dataset/video_dataset.py` | Precomputed MP cache |
| `losses/train_losses.py` | `compute_losses()` |
| `gaussian_splatting/renderer.py` | `diff_gaussian_rasterization` wrapper (no submodule edits) |
| `gaussian_splatting/camera.py` | FixedCamera → 3DGS matrices (OpenCV↔OpenGL axis bridge) |

## Config (`config.py`)

Removed `flame_model`. Added `mp_cache_dir`, `stage`, `lr_tracker`, `camera_npz`, loss weights.

## Training stages

- **A**: tracker + deformer (avatar frozen)
- **B**: avatar/Gaussians (tracker frozen)
- **C**: joint (default all trainable)

## Precompute (TODO scripts)

- `scripts/precompute_mediapipe.py`
- `scripts/precompute_segmentation.py`

## 3DGS renderer

Install only the CUDA extension (submodule optional):

```bash
pip install ./gaussian-splatting/submodules/diff-gaussian-rasterization
# optional: simple-knn for init scales
```

`GaussianRenderer` uses `colors_precomp` (view-independent RGB from UVH). Camera:
world/ICT +Y up → GL bridge in `gaussian_splatting/camera.py`; MP 2D projection stays in `utils/camera.py`.

## Tests

```bash
python -m pytest tests/ -q
```
