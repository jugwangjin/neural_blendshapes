# ICT MediaPipe landmark baker (`ict_mediapipe_lmk/`)

FLAME은 offline baker로만 쓰고, 학습 시에는 ICT topology 위 `vertices2landmarks`로 MediaPipe supervision을 건다.

## 의존성

- `metrical-tracker/flame/mediapipe/mediapipe_landmark_embedding.npz`
- `metrical-tracker/tracker.py` iris hardcoded indices (코드에 복제: `constants.py`)
- `large-steps-pytorch` (`pip install largesteps` 또는 repo clone)
- `assets/flame_static_embedding.pkl` (NICP 68-point term)
- `assets/ict_facekit_torch.npy`

## 실행

프로젝트 루트:

```bash
python ict_mediapipe_lmk/bake_mediapipe_to_ict.py
python ict_mediapipe_lmk/bake_mediapipe_to_ict.py --metrical_root ./metrical-tracker --large_steps_root ./large-steps-pytorch
```

출력: `assets/ict_mediapipe_landmark_embedding_from_metrical_tracker.npz`

## 파이프라인

1. **NICP** (`nicp.py`): ICT face verts `0:9409`, faces `0:9230`을 FLAME canonical에 Large Steps로 fit.
2. **FLAME MP 샘플** (`metrical.py`): metrical npz bary + iris vertex indices.
3. **Transfer** (`transfer.py`): FLAME MP 3D 점을 fitted ICT에 closest+bary projection. Iris는 face 전체가 아닌 iris vertex range로 제한.

## NPZ keys

| Key | Shape | 설명 |
|-----|-------|------|
| `mp_landmark_indices` | [N] | MediaPipe index |
| `ict_lmk_face_idx` | [N] | ICT triangle id |
| `ict_lmk_b_coords` | [N, 3] | barycentric |
| `transfer_error` | [N] | projection distance |
| `ict_lmk_target_type` | [N] | face / left_iris / right_iris |
| `source` | [N] | metrical-tracker / iris_hardcoded |

## 학습에서 사용

```python
from ict_mediapipe_lmk.landmarks import vertices2landmarks

emb = np.load("assets/ict_mediapipe_landmark_embedding_from_metrical_tracker.npz")
ict_mp = vertices2landmarks(
    ict_deformed_vertices,  # [B,V,3]
    ict_faces,
    emb["ict_lmk_face_idx"],
    emb["ict_lmk_b_coords"],
)
```

## 디버그

`ict_mediapipe_lmk/debug/`: `flame_canonical.obj`, `ict_fit_to_flame.obj`, MP point clouds, `transfer_error` 통계.
