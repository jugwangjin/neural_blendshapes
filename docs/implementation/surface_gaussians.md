# Surface Gaussians + eye texture + accessory

## Architecture

| Module | Path | Role |
|--------|------|------|
| `SurfaceGaussians` | `model/surface_gaussians.py` | Fixed `face_idx` + `bary`; learn `h`, scale, rot, opacity, color |
| `TextureSpaceGaussians` | `model/texture_space_gaussians.py` | Eyeball only: learnable UV + gaze slide, `h=0` |
| `EyeTextureGaussians` | `model/eye_texture_gaussians.py` | Left/right texture spaces |
| `AccessoryGaussians` | `model/accessory_gaussians.py` | Optional free group (0 if no accessory in data) |

`UVHGaussians` is deprecated (`model/uvh_gaussians.py` re-exports `TextureSpaceGaussians`).

## Sampling

`utils/sampling.build_surface_gaussian_layout(faces, vertex_parts, k_per_face)`:

- ICT triangles with all verts in `PART_FACE` = (0,1,2,5) — not eyeball (6,7)
- `K` barycentric samples per triangle (default `K=8` → ~74k Gaussians for ~9.2k faces)

No train-time `uv_to_face_bary()` lookup.

## Eye / ICT

- `ICTDeformer` calls `apply_eyeball_rotation=False` always.
- Gaze: `tracker` → `gaze_uv_left/right` → `EyeTextureGaussians.forward(...)` (stateless).
- `ict_model.py` eyeball centers: verts `21451:23021` (L), `23021:24591` (R).

## Accessory

- `auto_detect_accessory=True`: scans `segmentation_dir` for accessory class pixels.
- If none: `n_accessory_gaussians=0` → no `AccessoryGaussians` params.
- If present: default spawn `512` free Gaussians with learnable tangent slide + normal distance.
- Seg loss can mask accessory class when group is disabled (`mask_accessory_in_seg` in stage cfg).

## Training stages

See `training/stages.py`: 1 → 2A → 2B → 3 (optional appearance-only).

Stage 3 `sh_degree` hook exists on `AvatarRenderer.set_sh_degree()`; expanding `color` to SH coeffs is TODO.

## Config

```python
n_surface_gaussians_per_face: int = 8
n_eye_gaussians_per_side: int = 1024
n_accessory_gaussians: int = 0
auto_detect_accessory: bool = True
```
