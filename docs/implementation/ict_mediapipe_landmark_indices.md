# `ict_mediapipe_landmark_indices.npz`

Minimal train asset produced by `bake_mediapipe_to_ict.py`.

## Arrays (only these three)

| Key | Shape | Role |
|-----|-------|------|
| `mp_landmark_indices` | `[N]` | MediaPipe index per row (e.g. 0–477 subset) |
| `ict_lmk_face_idx` | `[N]` | ICT triangle id |
| `ict_lmk_b_coords` | `[N, 3]` | Barycentric weights on that triangle |

Loss: `losses/mediapipe_landmark_478.py` → sample deformed `mesh_xyz` and compare to 2D MP targets.

## Debug (optional)

`processing/ict_mediapipe_lmk/debug/ict_mediapipe_bake_aux.npz` — `transfer_error`, `v_ict_fit`, chart ids, etc.

## Legacy filename

`ict_mediapipe_landmark_embedding_from_metrical_tracker.npz` — same 3 keys if re-saved; loader accepts both paths via `resolve_embedding_path()`.

## Migrate existing long npz

```python
import numpy as np
from pathlib import Path
from processing.ict_mediapipe_lmk.embedding_io import save_ict_mediapipe_embedding

z = np.load("assets/ict_mediapipe_landmark_embedding_from_metrical_tracker.npz", allow_pickle=True)
save_ict_mediapipe_embedding("assets/ict_mediapipe_landmark_indices.npz", z)
```
