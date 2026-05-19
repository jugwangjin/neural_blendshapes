# FLAME UV for MediaPipe texture QA

## Role of `canonical_eye_smpl.obj` vs `uvs_for_flame.py`

- `assets/canonical_eye_smpl.obj` is the **processed FLAME face topology** (≈8090 triangles) used by `processing/flame/FLAME.py` when `use_processed_faces=True`. It has **no `vt` lines** — not a UV atlas.
- `processing/flame/FLAME2020/uvs_for_flame.py` (CLI) reports **face correspondence** between that OBJ and `generic_model.pkl` faces (≈9976 triangles). Same logic as the old nested-loop script, now in `processing/flame/flame_face_uv.py`.

## Which FLAME topology to use

Vertices are the same (5023); only **triangle lists** differ:

| Mode | `faces_tensor` source | F (typical) |
|------|------------------------|-------------|
| `use_processed_faces=True` | `canonical_eye_smpl.obj` (metrical / FLARE style) | ~8090 |
| `use_processed_faces=False` | `generic_model.pkl` (`flame_model.f`) | ~9976 |

**Must match** the mesh used when `mediapipe_landmark_embedding.npz` was built: `lmk_face_idx` indexes into that face array. Wrong F → wrong 3D landmark positions before ICT projection (silent bug).

**NICP (ICT face → FLAME)** deforms **vertices** on the face patch; it does not depend on triangulation for the fit itself. So NICP and landmark sampling can disagree if you mix topologies.

**This repo today**

- `bake_mediapipe_to_ict.py` default: **pkl** (`use_processed_faces=False`) — comment says align with `flame_static_embedding.pkl`.
- `nicp_from_ict_to_flame.py`, `optimize_ict_expression_to_flame.py`: also `use_processed_faces=False`.
- `processing/flame/FLAME.py` class default is `use_processed_faces=True` (metrical-style decoder if you instantiate without overriding).

If metrical-tracker baked MP embedding on **processed** faces (common for FLARE/metrical), prefer:

```bash
python processing/ict_mediapipe_lmk/bake_mediapipe_to_ict.py --use_processed_faces
```

Then FLAME UV from metrical `head_template_mesh.obj` usually needs **no** flare→pkl remap (same F as canonical).

**Quick check on server** (replace path):

```python
import numpy as np
emb = np.load("processing/metrical-tracker/flame/mediapipe/mediapipe_landmark_embedding.npz")
print("max lmk_face_idx", emb["lmk_face_idx"].max())  # must be < F-1
# F=8090 → processed; F=9976 → pkl
```

Re-bake ICT npz after changing topology; training only sees the baked ICT embedding, not FLAME `f` at runtime.

## UV loading in bake debug

`processing/ict_mediapipe_lmk/texture_viz.py` → `resolve_flame_uv_for_topology()`:

1. Optional cache: `assets/flame_uv_for_pkl_faces.npz` (remapped UV on pkl topology).
2. Try OBJ candidates with `vt` (metrical `head_template_mesh.obj`, `--flame_uv_mesh`, etc.).
3. If UV OBJ matches **flare** face count: remap to pkl faces via vertex-identical triangle lookup (`build_flare_to_flame_face_map`).
4. If UV OBJ already matches `f_flame` count: use as-is.

ICT side uses UV from `ict_facekit_torch.npy` (no change).

## Commands

```bash
# Face correspondence stats only
python processing/flame/FLAME2020/uvs_for_flame.py

# Bake (needs metrical head OBJ with UV for FLAME texture png)
python processing/ict_mediapipe_lmk/bake_mediapipe_to_ict.py --export_debug
```

Outputs under `processing/ict_mediapipe_lmk/debug/`:

- `flame_mediapipe_texture.png` — requires metrical UV source or remapped cache
- `ict_mediapipe_texture.png` — from npy UV

## Note

`FLAME_UV_MESH` in `processing/paths.py` remains the **topology reference** for flare/processed faces, not a UV file.
