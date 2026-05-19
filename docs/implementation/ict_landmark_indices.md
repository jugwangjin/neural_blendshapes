# ICT-FaceKit 68 landmark indices

## Source

[USC-ICT/ICT-FaceKit README](https://github.com/USC-ICT/ICT-FaceKit) — **Multi-PIE 68 point facial landmarks**.

Canonical list in code: `processing/ict_landmarks.py` → `LANDMARK_INDICES_MULTIPIE_68_JAWLINE`.

## Jawline substitution (recommended for FLAME alignment)

| Multi-PIE index | Official contour | This repo |
|-----------------|------------------|-----------|
| 0–7 | Multi-PIE brow/cheek | **Right jawline** `RIGHT_JAWLINE_68` |
| 8 | 966 | 966 |
| 9–16 | Multi-PIE brow/cheek | **Left jawline** `LEFT_JAWLINE_68` |
| 17–67 | inner features | `MULTIPIE_68_OFFICIAL[17:]` (unchanged) |

Right jawline: `1278, 1272, 12, 1834, 243, 781, 2199, 1447`  
Left jawline: `3661, 4390, 3022, 2484, 4036, 2253, 3490, 3496`

## Verification

`ict_facekit_to_npy_full_head.py` calls `validate_landmark_indices(..., n_verts=24591)`.

All indices must be `< 24591` (official parts #0–#8 only).

## Pairing with FLAME 68

`flame_static_embedding.pkl` uses the same **index order** (iBUG / Multi-PIE).  
Alignment uses `[landmark_start:]` default **17** (skip contour for NICP landmark loss).

## Bake debug outputs (`debug/`)

| File | Content |
|------|---------|
| `ict_canonical_flame_space.obj` | ICT neutral + optimized jawOpen + `flame_alignment_s,R,T` |
| `flame_canonical.obj` | FLAME canonical pose mesh |
| `ict_68_landmarks.ply` / `flame_68_landmarks.ply` | 68 points in 3D |
| `ict_canonical_68lmk_texture.png` | 68 pts on ICT UV layout |
| `flame_canonical_68lmk_texture.png` | 68 pts on FLAME UV layout |
| `canonical_68lmk_ict_vs_flame.png` | Side-by-side 68-point panel |

MediaPipe (478) textures are separate: `*_mediapipe_texture.png`.
