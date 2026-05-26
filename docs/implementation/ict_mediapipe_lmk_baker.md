# ICT MediaPipe landmark baker (`processing/ict_mediapipe_lmk/`)

FLAME은 offline baker로만 쓰고, 학습 시에는 ICT topology 위 `vertices2landmarks`로 MediaPipe supervision을 건다.

## 선행 조건

1. ICT npy (full head 26719):

```bash
python processing/ict_facekit_to_npy_full_head.py
# → assets/ict_facekit_torch.npy
```

2. Bake (writes NPZ + optional `nicp_canonical_mesh` reference in npy; **train template stays rigid-only**):

```bash
python processing/ict_mediapipe_lmk/bake_mediapipe_to_ict.py
```

See `docs/implementation/nicp_template_runtime.md`.

3. Submodule / clone (under `processing/`):

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

1. **NICP** (`nicp.py`): ICT skin patch verts `0:9409` → FLAME canonical (Large Steps).
   - **Extension** (`nicp_template.py`): mouth socket, eye sockets, gums, eye occlusion — displacement propagated from face NICP (eyeballs untouched).
   - Merge `nicp_canonical_mesh` into npy (unless `--skip_nicp_npy_merge`).
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

`debug/verify_mp_embedding_mesh_consistency.py`: npy `faces`와 embedding `ict_lmk_face_idx` 범위·런타임 canonical vs bake `v_ict_fit` 3D 거리.

## 인덱스 / mesh 일관성 (3단계 파이프라인)

| 단계 | mesh vertices | triangle `faces` | landmark 정의 |
|------|---------------|------------------|---------------|
| `ict_facekit_to_npy_full_head.py` | 원본 OBJ `vertices` (26719), **seam split `new_vertices` 아님** | quad→tri `faces` 저장 (`uv_faces`=`new_faces`는 UV 전용) | `landmark_indices`: Multi-PIE **68 vertex id** (README) |
| `bake_mediapipe_to_ict.py` | bake 시 `apply_ict_to_flame_space(neutral+jaw)` → NICP → **`v_ict_fit`** | npy와 동일 `f_ict` = `model_dict['faces']` | NPZ: `ict_lmk_face_idx` + `ict_lmk_b_coords` (FLAME MP→ICT 투영) |
| `ict_model.py` / train | `forward()` + **`nicp_vertex_offset`** if npy has `nicp_canonical_mesh` | `self.faces` = npy `faces` | MP loss: NPZ bary on **`mesh_xyz`**; PIE-68: `landmark_indices` |

**`vmapping`**: npy에 저장되지만 `landmark_indices` / `faces` / `neutral_mesh`에는 **적용되지 않음**. `update_vmapping()`은 학습 경로에서 호출되지 않음.

**UV texture (`ict_mediapipe_texture.png`)가 맞아 보이는 이유**: `triangle_uv_local[face_idx]` + 동일 bary — **face index 공간은 train과 동일**. UV는 2D chart라 NICP 3D 변형과 무관하게 “맞아 보임”.

**sanity 초록 / MP misalignment (index 범위는 맞아도)**:
1. embedding은 **NICP 후 `v_ict_fit`** 에서 투영됨.
2. **의도된 차이**: train template은 rigid alignment만; embedding은 NICP `v_ict_fit`에서 bake (`nicp_template_runtime.md`).
3. npy를 재생성했는데 **bake/NPZ 미갱신** → `face_idx` 불일치 또는 stale template (치명적).

**확인**:
```bash
python debug/verify_mp_embedding_mesh_consistency.py \
  --aux processing/ict_mediapipe_lmk/debug/ict_mediapipe_bake_aux.npz
```
canonical vs `v_ict_fit` mean dist가 크면 (1)(2)가 원인. `max face_idx >= F`면 (3).
