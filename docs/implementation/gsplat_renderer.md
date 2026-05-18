# gsplat renderer + semantic segmentation

## Backend

`GaussianRenderer` prefers **gsplat** (`config.renderer_backend = "gsplat"`).
Falls back to `diff_gaussian_rasterization` if gsplat is not installed.

Install (WSL/docker):

```bash
pip install gsplat
```

## Two-pass rendering

| Pass | `colors` | `sh_degree` | Output |
|------|----------|-------------|--------|
| RGB | sigmoid(rgb) `[G,3]` | `None` | `render["rgb"]` `[1,3,H,W]` |
| Semantic | `sem_prob` `[G,K]` | `None` | `render["semantic"]` `[1,K,H,W]` |

Entry points in `gaussian_splatting/gsplat_renderer.py`:

- `render_rgb()`
- `render_features()` — N-D features, `backgrounds=[1,K]`
- `render_depth()` — `render_mode="ED"` etc.
- `forward()` — RGB + optional semantic pass

Camera: `gsplat_camera.fixed_camera_to_gsplat()` — OpenCV w2c + `K`, **no GL bridge** (unlike INRIA path).

## Gaussian semantic logits

Per-Gaussian `sem_logits [G,K]` → `softmax` → composited in pass 2.

Classes (`gaussian_splatting/semantic.py`):

`skin, lip, eye, iris, hair, accessory, bg`

Eye Gaussians init biased toward `eye` / `iris`.

## Losses

1. **Image-space**: `losses/segmentation.py` — CE on `render["semantic"]` vs `batch["seg_label"]`, or L1 vs `seg_onehot`.
2. **h prior**: `loss_h_semantic(h, sem_prob, class_sigma)` — expected Charbonnier over class-specific σ.

Dataset: `{frame}_seg.png` next to cache, or under `cfg.segmentation_dir`.

## Stage weights

`w_seg` active from stage **2_gaussian_uvh** onward (`training/stages.py`).
