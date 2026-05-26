# MediaPipe landmark loss (active path)

## Source of truth

```text
assets/ict_mediapipe_landmark_embedding_from_metrical_tracker.npz
```

Built by `processing/ict_mediapipe_lmk/bake_mediapipe_to_ict.py` (FLAME metrical-tracker → NICP ICT face patch only → component-aware barycentric transfer).

### Bake routing (component-aware)

| Landmark group | Projection candidates | 3D query source |
|----------------|----------------------|-----------------|
| face / jaw / mouth | `surface_sample` (no eyeball) | FLAME metrical embedding |
| eyelid / eye contour | `eye_socket` L/R (no eyeball) | FLAME metrical embedding |
| iris 468–477 | `M_Sclera*` ∩ eyeball verts (not `M_Iris*` annulus) | **FLAME eye NICP → transplant** (`eye_transplant.py`) |

### Iris bake (eye-only NICP)

```text
ICT full mesh in FLAME space: npy jawOpen + flame_alignment_s,R,T (facekit / load_ict_asset)
Face NICP (does not move eyeball verts)
Per-eye: rigid prescale FLAME eyeball → ICT eyeball (pytorch3d s,R,T)
eye-only NICP (Chamfer + edge + iris anchor on ICT iris verts)
fitted FLAME iris seeds → project to ICT sclera-chart triangles
```

Bake uses ``v_ict_fit`` for eye transplant projection (eyeball = post-global-align ICT).
Log ``iris seed spread`` / ``fitted iris spread`` — if min≈max≈0, anchors collapsed (check npy align).

Training: surface Gaussians on sclera + `M_EyeOcclusion`; iris 468–477 via mesh bary (`w_mp_lmk`).

Sanity: `check_projected_faces_are_in_eye()` — iris faces on eyeball verts only.

NICP deforms **face patch only** (`0:9409`); eyeball drift should be ~0.

Output also includes `geometry_chart_id` (0=face, 1=left eye, 2=right eye) for texture/chart separation.

## Bake texture QA (`bake_mediapipe_to_ict.py`)

- Combined PNG: face-focused overview (`VIZ_SKIP_MATERIALS` — teeth, eyes, occlusion, etc. omitted from one atlas).
- Per-map under `processing/ict_mediapipe_lmk/debug/texture_maps/` — **every** `usemtl` in npy `material_names` (from `ICT_FaceKit/.../generic_neutral_mesh.obj`):
  - `ict_mediapipe_M_Face_texture.png`, `ict_mediapipe_M_Teeth_texture.png`, `ict_mediapipe_M_EyeOcclusion_texture.png`, `ict_mediapipe_M_EyeBlend_texture.png`, … (all catalog materials present in mesh)
  - Landmark count may be 0 on unused charts (layout wireframe only).
  - Iris pentagon extras: `ict_eye_occlusion_*_iris5_texture.png` (ray_occ bake) or `ict_eyeball_*_iris5_texture.png` (legacy sclera bake).
- Uses `triangle_uv_local` (per texture map 0–1), not seam VT.

Npy build (`ict_facekit_to_npy_full_head.py`) also writes layout charts under `debugs/ict_facekit_uv/texture_charts/` by default (`--no_export_uv_debug` to skip).

Fields used at train time:

- `mp_landmark_indices` — which of 478 MP points to supervise
- `ict_lmk_face_idx` — ICT triangle index per landmark
- `ict_lmk_b_coords` — barycentric weights on that triangle

## Dual landmarks (PIE 68 + MP)

- **Train MP**: `mp_landmark_indices` → `w_mp_lmk` (no chin contour in MP).
- **Train jaw**: PIE protocol `0..16` on ICT verts → `w_pie68_jaw` vs FA 68 detections (`losses/pie68_jaw_landmark.py`). See `pie68_jaw_train_loss.md`.
- **Bake / NICP / npy**: inner `[17:]` + jawline KNN (`pie68_jaw_and_mediapipe_landmarks.md`).

## Not used for training

- `legacy/flare/losses/legacy_landmark_68.py` — FLARE gbuffer clip-space (all 68)

## Implementation

`losses/mediapipe_landmark_478.py`:

1. `vertices2landmarks_barycentric(verts, faces, face_idx, bary)` → 3D points on deformed mesh
2. `camera.project_world_points` → pixels, divide by `image_size` → normalized UV
3. Target: `batch["mp_landmarks_2d"][:, mp_landmark_indices]`
4. `robust_l1` with `batch["mp_valid"][:, mp_landmark_indices]`

Wired from `losses/train_losses.py` on `avatar_out["mesh_xyz"]` (`verts_posed` from `ICTDeformer`).

## Config

`config.py`: `mp_embedding = Path("assets/ict_mediapipe_landmark_embedding_from_metrical_tracker.npz")`

Loss weight: `w_mp_lmk` (stage overrides in `training/stages.py`).
