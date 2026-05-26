# gsplat avatar renderer + semantic segmentation

## Layout

Avatar rasterization lives in **`rendering/`** (repo root). **gsplat is required** — module top-level `from gsplat import rasterization`. No INRIA / `gaussian_splatting` fallback. The **`gsplat/` git submodule is never modified**.

| Module | Role |
|--------|------|
| `rendering/avatar_renderer.py` | `AvatarRenderer` / `GaussianRenderer` — RGB + semantic two-pass |
| `rendering/pack.py` | Pack `GaussianAvatar` tensors for gsplat |
| `rendering/gsplat_camera.py` | `FixedCamera` → OpenCV `viewmats` + `K` |
| `rendering/semantic.py` | Class names + per-class h priors |
| `rendering/gaussian_semantics.py` | ICT `vertex_parts` → Gaussian `sem_logits` init |

Install (WSL/docker):

```bash
pip install gsplat
# or: git submodule update --init gsplat && pip install -e gsplat
```

## Two-pass rendering

| Pass | `colors` | `sh_degree` | Output |
|------|----------|-------------|--------|
| RGB | sigmoid(rgb) `[G,3]` | `None` | `render["rgb"]` `[1,3,H,W]` |
| Semantic | `sem_prob` `[G,K]` | `None` | `render["semantic"]` `[1,K,H,W]` |

Entry points in `rendering/avatar_renderer.py`:

- `render_rgb()`
- `render_features()` — N-D features, `backgrounds=[1,K]`
- `render_depth()` — `render_mode="ED"` etc.
- `forward()` — RGB + optional semantic pass

Camera: `rendering/gsplat_camera.fixed_camera_to_gsplat()` — OpenCV w2c + `K`.

## Gaussian semantic logits

Per-Gaussian `sem_logits [G,K]` → `softmax` → composited in pass 2.

Classes (`rendering/semantic.py`):

`skin, lip, eye, iris, hair, accessory, bg`

Eye Gaussians init biased toward `eye` / `iris`.

## Losses

1. **Image-space**: `losses/segmentation.py` — CE on `render["semantic"]` vs `batch["seg_label"]`, or L1 vs `seg_onehot`.
2. **h prior**: `loss_h_semantic(h, sem_prob, class_sigma)` — expected Charbonnier over class-specific σ.

Dataset: `{frame}_seg.png` next to cache, or under `cfg.segmentation_dir`.

## Stage weights

`w_seg` is **0** for all stages 0–3 (`training/stages.py`); semantic render not used in the default schedule.
