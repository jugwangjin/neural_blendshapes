# Eye UV symmetry ↔ npy ↔ MediaPipe bake

## npy build (`ict_facekit_to_npy_full_head.py`)

`analyze_eye_uv_symmetry` compares normalized `uv_neutral_mesh` on part #7/#8 eyeballs:

- **direct** score low → L/R charts already co-oriented (shared UV bank, no `u'=1-u` on right).
- **mirror_u** score high → would need `eye_uv_mirror_right_u=True` for `EyeTextureGaussians`.

Typical good run:

```
left/right VT=1570/1570
score direct≈0.001  mirror_u≈0.36
eye_uv_mirror_right_u = False
```

Stored in `ict_facekit_torch.npy`; read by `ICTFaceKitTorch` and `GaussianAvatar.from_ict(mirror_right_u=...)`.

## MediaPipe bake (`bake_mediapipe_to_ict.py`)

| Stage | Eyeballs (21451:24591) |
|-------|-------------------------|
| `load_ict_asset` + `flame_similarity_s/T` | Scaled/translated with full mesh |
| Face NICP (`fit_ict_face_to_flame`) | **Not** deformed (`v_ict_fit[:9409]` only) |
| Face MP transfer | Uses `v_ict_fit` |
| Eye iris transplant (`run_eye_transplant`) | Uses `v_ict` (eye verts unchanged by face NICP) |
| Iris bary projection | `M_ScleraLeft` / `M_ScleraRight` tris only (`sclera_eyeball_face_mask`) |

`eye_uv_mirror_right_u` does **not** affect landmark bake (3D barycentrics only). Iris UV chart choice is material-based, not mirror flag.

## After your npy build

Proceed with bake (defaults match schema 6 / official_24591):

```bash
python processing/ict_mediapipe_lmk/bake_mediapipe_to_ict.py
```

Expect log: `eyeball NICP drift ... ~0`, `eye_uv_mirror_right_u=False`, FLAME `use_processed_faces=False`.

After bake, by default in `processing/ict_mediapipe_lmk/debug/`:

- `ict_mediapipe_texture.png` — baked ICT embedding on UV layout
- `flame_mediapipe_texture.png` — metrical MP on FLAME UV (`processing/flame`)
- `mediapipe_landmarks_ict_vs_flame.png` — side-by-side panel

Skip textures: `--no_texture_viz`. Skip OBJ/PLY only: `--no_export_debug`.
