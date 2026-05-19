# Eye Gaussian UV — sclera chart only (hemisphere)

## Policy (aligned with `bake_mediapipe_to_ict` / `eye_transplant`)

| Include | Exclude |
|---------|---------|
| `M_ScleraLeft` / `M_ScleraRight` (filled disk chart) | `M_IrisLeft` / `M_IrisRight` annulus |
| Eyeball vertices | Non-eyeball tris |
| Front sclera hemisphere+ (default) | Back of eyeball on sclera chart |

Bake iris MP projection uses `sclera_eyeball_face_mask` (sclera material ∩ eyeball).  
Eye Gaussians use **`sclera_sampling_face_indices(..., hemisphere_only=True)`** — same material rule + normal-based front cap.

## Sampling (`utils/eye_uv_sampling.py`)

`mode="hemisphere"` (default):

1. Triangle set = `sclera_sampling_face_indices` (not iris, not back cap unless `hemisphere_only=False`).
2. Solid-angle uniform directions on visible hemisphere; snap to closest sclera tri → UV.

`mode="triangle"`: uniform bary per sclera sampling tri (legacy; still sclera-only).

## Config

```python
eye_uv_sample_mode: str = "hemisphere"
eye_sclera_min_front_dot: float = -0.15
eye_sclera_hemisphere_only: bool = True
```

## Runtime UVMesh

`TextureSpaceMeshes.from_ict` → `build_sclera_uv_mesh(..., hemisphere_only=True)` so `surface_points_from_uvh` never resolves onto `M_Iris*`.

Re-init requires rebuilding the avatar (new training run or sanity); frozen `EyeTextureGaussians.uv` buffer is set at construction.

## Verify

```bash
python scripts/sanity_gaussian_layout.py --pcd-mode both --single
# inspect *_gaussians.ply — white eye Gaussians should wrap the sclera cap, not a tight pupil disk
```
