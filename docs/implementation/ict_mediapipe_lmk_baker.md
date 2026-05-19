# ICT MediaPipe landmark baker (`processing/ict_mediapipe_lmk/`)

FLAME은 offline baker로만 쓰고, 학습 시에는 ICT topology 위 `vertices2landmarks`로 MediaPipe supervision을 건다.

## 선행 조건

1. ICT npy (공식 24591 topology):

```bash
python processing/ict_facekit_to_npy_full_head.py
# → assets/ict_facekit_torch.npy  (schema v2, official_24591)
```

2. Submodule / clone (under `processing/`):

- `processing/metrical-tracker/flame/mediapipe/mediapipe_landmark_embedding.npz`
- `processing/large-steps-pytorch`
- `assets/flame_static_embedding.pkl`

## 실행

프로젝트 루트:

```bash
python processing/ict_mediapipe_lmk/bake_mediapipe_to_ict.py
python processing/ict_mediapipe_lmk/bake_mediapipe_to_ict.py \
  --metrical_root processing/metrical-tracker \
  --large_steps_root processing/large-steps-pytorch
```

출력: `assets/ict_mediapipe_landmark_embedding_from_metrical_tracker.npz`

## 파이프라인

1. **NICP** (`nicp.py`): ICT skin patch verts `0:9409`, faces with all corners `< 9409` → FLAME canonical (Large Steps).
   - FLAME default: `use_processed_faces=False` (must match `flame_static_embedding.pkl`).
2. **FLAME MP 샘플** (`metrical.py`): metrical npz bary + iris vertex indices (`constants.py`).
3. **Transfer** (`transfer.py`): FLAME MP 3D → fitted ICT. Iris는 `left_eyeball_indices` / `right_eyeball_indices` (npy) 삼각형으로 제한.

## NPZ keys (embedding)

| Key | 설명 |
|-----|------|
| `mp_landmark_indices` | MediaPipe index |
| `ict_lmk_face_idx` | ICT triangle id |
| `ict_lmk_b_coords` | barycentric |
| `transfer_error` | projection distance |
| `ict_lmk_target_type` | face / left_iris / right_iris |
| `source` | metrical-tracker / iris_hardcoded |
| `ict_asset_variant` | npy variant string |
| `ict_asset_schema_version` | npy schema |
| `ict_vertex_count` | V |

## 학습에서 사용

```python
from utils.mediapipe_ict import load_ict_mediapipe_embedding, ict_vertices_to_mediapipe

emb = load_ict_mediapipe_embedding("assets/ict_mediapipe_landmark_embedding_from_metrical_tracker.npz")
ict_mp = ict_vertices_to_mediapipe(verts, ict.faces, emb)  # [B, L, 3]
```

또는:

```python
from processing.ict_mediapipe_lmk.landmarks import vertices2landmarks
```

## 디버그

`processing/ict_mediapipe_lmk/debug/`: `flame_canonical.obj`, `ict_fit_to_flame.obj`, MP point clouds, UV texture QA.
