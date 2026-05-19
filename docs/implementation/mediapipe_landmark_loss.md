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

Matches training: `EyeTextureGaussians` samples on `M_Sclera*` filled disk; `M_Iris*` is UV annulus (empty center).

Sanity: `check_projected_faces_are_in_eye()` — iris faces on eyeball verts only.

NICP deforms **face patch only** (`0:9409`); eyeball drift should be ~0.

Output also includes `geometry_chart_id` (0=face, 1=left eye, 2=right eye) for texture/chart separation.

## Bake texture QA (`bake_mediapipe_to_ict.py`)

- Combined PNG: **M_Face** (+ skin) only — teeth / `M_GumsTongue` / iris annulus omitted.
- Per-map under `debug/texture_maps/`:
  - `ict_mediapipe_M_Face_texture.png` — face MP (no teeth/gums)
  - `ict_eyeball_left_iris5_texture.png` — **only** MP 468–472 on eyeball `triangle_uv_local` (sclera∩eyeball tris)
  - `ict_eyeball_right_iris5_texture.png` — **only** MP 473–477
  - `mediapipe_eyeball_iris5_left_vs_right.png` — side-by-side iris QA
- Uses `triangle_uv_local` (per texture map 0–1), not seam VT — iris pentagon + labels visible on chart.

Fields used at train time:

- `mp_landmark_indices` — which of 478 MP points to supervise
- `ict_lmk_face_idx` — ICT triangle index per landmark
- `ict_lmk_b_coords` — barycentric weights on that triangle

## Dual landmarks (PIE 68 + MP)

- **Train**: MP bary embedding only (`mp_landmark_indices` → 2D loss).
- **Bake / NICP / npy align**: Multi-PIE **inner** `[17:]` ↔ FLAME embedding + **jawline** `[0:16]` KNN to FLAME mesh (MP has no chin contour). See `pie68_jaw_and_mediapipe_landmarks.md`.

## Not used for training

- `ict_facekit.landmark_indices` — 68 vertex picks (jawline + inner); not wired to `train_losses` MP path
- `losses/legacy_landmark_68.py` — old gbuffer + 68-vertex path (FLARE)

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
