# Surface Gaussians (mesh-embedded only)

All trainable Gaussians live on the ICT surface mesh (`GaussianAvatar`).

| Region | Layout key | Notes |
|--------|------------|-------|
| Face / head / neck | `k_face`, `k_head` | Skin triangles |
| Mouth socket / interior | `k_mouth_socket`, `k_mouth_interior` | Gum `h` scaled |
| Eye socket | `k_eye_socket` | |
| Sclera / occlusion | `k_eyeball_sclera`, `k_eye_occlusion` | `is_h_pin` on sclera+occlusion |

Sampling: `utils/sampling.build_surface_gaussian_layout`.

## Removed: free `AccessoryGaussians`

Glasses / hat / hair outside the ICT mesh are **not** modeled with separate free Gaussians.
Pixel-wise **image-space h regularization** (`losses/h_regularization.py`) pulls surface `h`
toward GT depth on skin / eye / brow / mouth tiers.

FLARE semantic labels for glasses/hat may still appear in **seg loss** (`rendering/semantic.py`
class `accessory`) — that is GT supervision only, not extra splats.

Legacy implementation: `legacy/accessory_gaussians.py`.

## Config (excerpt)

```python
n_surface_gaussians_per_face: int = 8
n_surface_gaussians_per_eyeball_sclera: int = 1
n_surface_gaussians_per_eye_occlusion: int = 8
```
