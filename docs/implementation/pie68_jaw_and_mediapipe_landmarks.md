# Multi-PIE 68 + MediaPipe landmarks (dual use)

## Roles

| Signal | Indices | Used for |
|--------|---------|----------|
| **Multi-PIE inner** | protocol `17..67` (51 pts) | FLAME static embedding pairing, NICP inner L1, npy `jawOpen`+`s,R,T` |
| **Multi-PIE jawline** | protocol `0..16` (17 pts) | NICP / npy jaw KNN to FLAME mesh (no FLAME jaw embedding) |
| **MediaPipe 478** | baked bary on ICT | Train 2D `w_mp_lmk` (face/eyes/iris; **no jaw contour**) |
| **PIE jawline 2D** | FA 68 pts 0..16 | Train `w_pie68_jaw` on ICT jaw verts (`pie68_jaw_train_loss.md`) |

MediaPipe face routing (`landmark_routing.py`) has no dedicated jaw **contour**; jaw blendshape names map to interior verts, not the PIE chin arc. Jaw shape during NICP/alignment is anchored by **PIE jawline KNN**.

## Implementation

- `processing/ict_landmarks.py` — `landmark_jawline_vertex_indices`, `landmark_inner_vertex_indices`
- `processing/ict_flame_similarity.py` — `jawline_knn_mean`, `w_jaw_knn` in `optimize_ict_jaw_open` / `compute_ict_flame_similarity`
- `processing/ict_mediapipe_lmk/nicp.py` — `_jaw_knn_loss` in stages 1–3 + Large Steps stage 3
- `model/ict_model.py` — `landmark_vertices(mesh, region='inner'|'jawline'|'all')`

## CLI

| Script | Flag | Default |
|--------|------|---------|
| `ict_facekit_to_npy_full_head.py` | `--w_jaw_knn` | 25 |
| `bake_mediapipe_to_ict.py` | `--nicp_w_jaw` | 30 |

## Npy fields

- `flame_similarity_jaw_knn_mean` — mean min-distance jawline → FLAME mesh after alignment
