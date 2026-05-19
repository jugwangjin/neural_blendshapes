# Shared eye texture-space Gaussians

## Design

```text
self.eye  (single TextureSpaceGaussians, n_per_eye)
  uv, h     → buffers (frozen)
  color, opacity, log_scale, rotation, sem → shared, optimized

left instance:  uv_eff = uv + gaze_left  → lift on left_uv_mesh
right instance: uv_eff = uv + gaze_right → lift on right_uv_mesh
```

Render batch: `2 * n_per_eye` Gaussians (same appearance tensors referenced twice).

## Optimized vs frozen

| Parameter | Optimized |
|-----------|-----------|
| `eye.color`, `opacity`, `log_scale`, `rotation`, `sem_logits` | yes |
| `gaze_refine_left`, `gaze_refine_right` | yes (if `learn_gaze_refine`) |
| `eye.uv`, `eye.h` | no (`_freeze_attr_as_buffer`) |

## Right-eye u mirror

Bake (`ict_facekit_to_npy_full_head.py`) runs `analyze_eye_uv_symmetry` and stores `eye_uv_mirror_right_u` in npy.

`GaussianAvatar.from_ict` passes it to `EyeTextureGaussians(mirror_right_u=...)`. When true, right instance uses `u' = 1 - u` on shared `uv_eff` before lifting on `right_uv_mesh`.

## API

`EyeTextureGaussians.forward(left_uv_mesh, right_uv_mesh, gaze_uv_*)` — `verts`/`faces` optional (compat only).

Gaze tensors are cast to `self.eye.uv` device/dtype inside `forward`.

`GaussianAvatar.forward` also casts gaze to `verts.device` before calling eyes.
