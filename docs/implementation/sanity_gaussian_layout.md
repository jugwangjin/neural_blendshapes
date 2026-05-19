# Sanity: Gaussian layout + fixed camera render

Script: `scripts/sanity_gaussian_layout.py`

Before `train.py`, run on the server (same env as training):

```bash
python scripts/sanity_gaussian_layout.py
python scripts/sanity_gaussian_layout.py --out out/sanity_gaussians --image-size 512
```

## Forced appearance

| Region | RGB (display) |
|--------|----------------|
| face | peach `(1.0, 0.78, 0.62)` |
| head/neck | black |
| mouth_interior (gum) | red |
| eyeball (texture Gaussians) | white |
| mouth_socket | dark red (debug) |
| eye_socket | slate (debug) |
| accessory | cyan (if enabled) |

Opacity logit `12` → σ(opacity) ≈ 1.

Mesh: ICT neutral (no deformer). Camera: `assets/default_camera.npz`.

## Outputs

- `sanity_rgb.png` — gsplat RGB
- `sanity_depth_ed.png` — expected depth (normalized preview)
- `sanity_alpha.png` — coverage
- `README.txt` — legend

Check: face peach on skin, black collar/head back, red oral cavity, white sclera disks, no obvious holes at eye rim.
