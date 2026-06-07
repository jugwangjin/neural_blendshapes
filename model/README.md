# `model/` — 3DGS avatar training stack

| Module | Role |
|--------|------|
| `build.py` | Factory: `build_ict` / `build_tracker` / `build_deformer` / `build_avatar` from `Config` |
| `ict_model.py` | ICT-FaceKit + npy texture maps (`material_names`, `triangle_uv_local`) |
| `ict_deformer.py` | MP coeffs → ICT verts (template 5×128, expr channel-wise 5×128 Softplus MLP; `docs/implementation/ict_deformer_mlps.md`) |
| `tracker_mlp.py` | MP correction, gamma, pose |
| `expression_deform_mlp.py` | Support-gated expr deltas |
| `expr_regions.py` | Per-vertex expr / deform weights |
| `blendshape_support.py` | AU support gates |
| `pose_weight.py` | Pose residual MLP |
| `gaussian_avatar.py` | Surface Gaussians (sclera + eye occlusion) |

Archived eye UV slide: `legacy/eye_uv_slide/` (not used by `train.py`).

Docs: `docs/implementation/eye_uv_slide.md`
