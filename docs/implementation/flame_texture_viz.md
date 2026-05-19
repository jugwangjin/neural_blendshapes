# FLAME side of landmark texture QA

Uses **repo** `processing/flame` only (not metrical head mesh/UV for geometry).

| Piece | Module |
|-------|--------|
| Neutral mesh + `faces_tensor` | `processing/flame/FLAME.py` via `flame_viz.load_flame_canonical_mesh` |
| UV on pkl topology | `processing/flame/flame_face_uv.py` — flare OBJ with `vt` + remap (`uvs_for_flame` logic) |
| MP landmark indices | metrical `mediapipe_landmark_embedding.npz` (barycentric only) |

## UV OBJ

`canonical_eye_smpl.obj` has **no** `vt`. One-time setup:

```bash
python processing/flame/prepare_flame_head_uv.py --build_cache
```

See **`docs/implementation/flame_head_uv_setup.md`** (metrical `install.sh` or FLAME TextureSpace).

Then bake uses `assets/flame_head_uv.obj` or `--flame_uv_mesh`.

## CLI

```bash
python processing/flame/FLAME2020/uvs_for_flame.py   # face correspondence stats

python processing/ict_mediapipe_lmk/viz_landmark_textures.py \
  --embedding assets/ict_mediapipe_landmark_indices.npz \
  --flame_uv_mesh assets/flame_head_uv.obj
```
