# `model/` — 3DGS avatar training stack

Import submodules directly (e.g. `from model.ict_model import ICTFaceKitTorch`).  
Do not rely on `from model import …` package re-exports.

## Active modules (`train.py`)

| Module | Role |
|--------|------|
| `ict_model.py` | ICT-FaceKit mesh + blendshapes (loads `assets/ict_facekit_torch.npy`) |
| `ict_deformer.py` | MP coeffs → ICT verts (+ optional expr delta) |
| `tracker_mlp.py` | MP correction, gamma, pose, gaze UV |
| `expression_deform_mlp.py` | Support-gated per-AU vertex deltas |
| `expr_regions.py` | Per-vertex expr mask from npy regions |
| `blendshape_support.py` | AU support gates (used by expr deformer) |
| `pose_weight.py` | Pose residual MLP (used by deformer) |
| `gaussian_avatar.py` | Surface + eye + optional accessory Gaussians |
| `surface_gaussians.py` | Fixed bary surface 3DGS |
| `eye_texture_gaussians.py` | Eyeball UV-slide 3DGS |
| `texture_space_gaussians.py` | UV + h texture-space Gaussians |
| `accessory_gaussians.py` | Optional free accessory Gaussians |

## Also used outside train

- `ict_model.py` — `processing/optimize_ict_expression_to_flame.py`, `processing/nicp_from_ict_to_flame.py`, etc.

## Moved to `legacy/model/`

`fc.py`, `math_np.py`, `uvh_gaussians.py` (unused by current pipeline).
