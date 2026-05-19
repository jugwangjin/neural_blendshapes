# Pipeline fixes (import / ICT parts / surface Gaussians)

## ICT `vertex_parts` (authoritative: `processing/ict_facekit_to_npy.py`)

`parts_split = [9409, 11248, 13294, 13678, 14062, 17039, 21451, 23021, 24591]` → **9 parts (ids 0–8)**.

| id | vertex range | region |
|----|----------------|--------|
| 0 | 0:9409 | face |
| 1 | 9409:11248 | head/neck (not_face) |
| 2 | 11248:13294 | mouth socket |
| 3 | 13294:13678 | eye socket L |
| 4 | 13678:14062 | eye socket R |
| 5 | 14062:17039 | gums/tongue |
| 6 | 17039:21451 | teeth |
| 7 | 21451:23021 | eyeball L |
| 8 | 23021:24591 | eyeball R |

**Critical:** eye texture must use parts **7, 8** — not 6, 7 (6 is teeth).

Hair/accessory are **not** in ICT; use segmentation + `AccessoryGaussians`.

## Code updates

- `rendering/gaussian_semantics.py` — corrected `ICT_PART_TO_SEMANTIC`
- `utils/texture_spaces.py` — `PART_EYEBALL_L/R = (7,)`, `(8,)`, surface `PART_SURFACE = (0,1,2,5,6)`
- `model/expr_regions.py` — `rendering.*` imports, part weights 0–8
- `model/surface_gaussians.py` + `GaussianAvatar` — fixed bary, no face UV optimize
- `model/eye_texture_gaussians.py` — stateless `gaze_uv_left/right`
- `train.py` — `assert batch_size == 1`
- Legacy moved under `legacy/`; active `model/deformer.py` removed (restore from GitHub if needed)

## gsplat

- `rendering/avatar_renderer.py` only; no `gaussian_splatting` package in active path.
