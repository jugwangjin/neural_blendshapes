# Legacy mesh-texture / FLARE inventory

Active stack (`train.py`): **SurfaceGaussians + EyeTextureGaussians + gsplat** — no neural mesh shader, no trainable face UV.

## Safe to remove or move to `legacy/`

| Path | Role | Used by active `train.py`? |
|------|------|---------------------------|
| `model/embedder.py` | NeRF positional encoding | **Deleted** |
| `model/encoder.py` | DECA/ResNet image encoder | No (marked LEGACY) |
| `model/resnet.py` | ResNet backbone for encoder | Only `encoder.py` |
| `model/deformer.py` | `NeuralBlendshapes`, tcnn HashGrid, MLP deform | No — **imports deleted `embedder`** |
| `model/eye_plane_gaussians.py` | Eye tangent plane (pre texture-space) | No — replaced by `eye_texture_gaussians.py` |
| `utils/eye_frame.py` | `EyeFrame` for eye plane | Only `eye_plane_gaussians.py` |
| `losses/mediapipe_landmark.py` | 68-pt + gbuffer clip-space loss | No — use `train_losses.loss_mediapipe_landmarks_2d` |
| `visualize_texture.py` | Debug texture map PNG crop | No |
| `arguments.py` | FLARE CLI (`finetune_color`, `material_mlp`, hashgrid shader) | No (`config.py` instead) |
| `configs/`, `configs_tmp/` | FLARE experiment configs | No |
| Root FLARE scripts | `test.py`, `load_nbshapes.py`, `draw_mediapipe.py`, `track_video*.py`, `gui_by_facs*.py`, `run_trains.py` | No |

## Keep (still needed)

| Path | Role |
|------|------|
| `utils/uv_mesh.py` | `UVMesh`, `surface_points_from_uvh`, `uv_to_face_bary` — **eyeball texture only** |
| `utils/texture_spaces.py` | ICT `vertex_parts` → eye/face UV submeshes |
| `model/texture_space_gaussians.py` | Eye UV slide |
| `model/uvh_gaussians.py` | Deprecated alias → `TextureSpaceGaussians` |

Note: `uv_mesh._lookup_face_bary` + reproject cache was for **trainable face UV**; surface path no longer calls it. Could trim to eye-only API later.

## Submodule / preprocessing (not mesh texture, but FLARE-era)

| Path | Role |
|------|------|
| `face_normals/` (git submodule) | ResNet UNet normals — `processing/prepare_normals.py`, `__init__.py` |
| `processing/save_canonical_pose.py`, `optimize_ict_expression_to_flame.py` | FLARE imports |

## Recommended cleanup order

1. Move `model/{encoder,resnet,deformer,eye_plane_gaussians}.py` + `utils/eye_frame.py` + `losses/mediapipe_landmark.py` → `legacy/flare/`
2. Move root FLARE entry scripts → `legacy/scripts/`
3. Delete or archive `visualize_texture.py`, unused `configs_tmp/`
4. Slim `uv_mesh.py` (drop face UV cache path if unused)
