# MediaPipe landmark texture QA (visual check)

Landmarks are drawn on **UV texture-map layouts** (triangle charts + wireframe), not on photo albedo.

ICT (baked embedding) vs FLAME (repo `processing/flame`), side by side. MP barycentrics for FLAME still come from metrical `mediapipe_landmark_embedding.npz`.

## Bake (automatic)

`bake_mediapipe_to_ict.py` exports ICT + FLAME landmark UV textures by default into `--debug_dir` (skip with `--no_texture_viz`).

## Standalone command

```bash
python processing/ict_mediapipe_lmk/viz_landmark_textures.py \
  --embedding assets/ict_mediapipe_landmark_indices.npz \
  --ict_npy assets/ict_facekit_torch.npy \
  --out_dir processing/ict_mediapipe_lmk/debug \
  --metrical_root processing/metrical-tracker \
  --flame_uv_mesh assets/flame_head_uv.obj
```

See also `docs/implementation/flame_texture_viz.md`.

## Outputs (`debug/`)

| File | UV base | Landmarks |
|------|---------|-----------|
| `ict_mediapipe_texture.png` | ICT `triangle_uv_atlas` tile layout (or local UV wireframe) | Baked ICT embedding |
| `flame_mediapipe_texture.png` | FLAME UV wireframe (`flame_face_uv`) | Metrical MP embedding |
| `mediapipe_landmarks_ict_vs_flame.png` | Side-by-side panel | |

## Landmark overlay colors

- Green: MediaPipe connections  
- Red: face/eyelid  
- Cyan: iris 468–477  
