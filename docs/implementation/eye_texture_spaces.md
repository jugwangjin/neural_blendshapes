# Eye texture spaces (좌/우 분리 atlas)

ICT `vertex_parts`로 **texture space index**를 나누고, 좌/우 눈은 각각 별도 UV atlas에서 Gaussian을 둔다.  
eyeball은 normal displacement **`h = 0`** (표면에 고정).

## 배경

- 예전 `EyePlaneGaussians` + tangent frame (`eye_frame.py`) 방식은 **deprecated**
- ICT는 이미 part별 UV island가 있음 → `vertex_parts`로 삼각형 필터 후 multi-texture mesh처럼 사용

## `vertex_parts` (ict_facekit_to_npy.py)

`parts_split = [9409, 11248, 13294, 13678, 14062, 17039, 21451, 23021, 24591]`

| part id | vertex range (대략) | texture space |
|--------|---------------------|---------------|
| 0 | 0 – 9408 | face skin |
| 1 | 9409 – 11247 | mouth socket |
| 2 | 11248 – 13293 | … |
| 3 | 13294 – 13677 | **left** eye socket |
| 4 | 13678 – 14061 | **right** eye socket |
| 5 | 14062 – 17038 | … |
| 6 | 21451 – 23020 | **left** eyeball mesh |
| 7 | 23021 – 24590 | **right** eyeball mesh |

### 코드 상수 (`utils/texture_spaces.py`)

```python
PART_FACE = (0, 1, 2, 5)
PART_LEFT_EYE = (3, 6)    # socket + eyeball
PART_RIGHT_EYE = (4, 7)
```

삼각형은 **세 꼭짓점 모두** allowed part에 속할 때만 해당 texture space에 포함.

## 파이프라인

```
ICT npy (vertex_parts, uvs, uv_faces)
    → TextureSpaceMeshes.from_ict(ict)
         face / left_eye / right_eye  (각 UVMesh + active_face_idx)
    → GaussianAvatar
         face: UVHGaussians (h learnable)
         eyes: EyeTextureGaussians ×2 (fixed_h=0)
```

### `UVMesh.active_face_idx`

`uv_to_face_bary`가 **해당 space 삼각형만** 검색. 다른 atlas island와 섞이지 않음.

## Gaze: texture space UV slide

눈 시선은 3D eyeball rotation 대신 **UV offset** `(du, dv)` 로 표현.

| 단계 | 설명 |
|------|------|
| **Base** | `eyeLookIn/Out/Up/Down_{L,R}` expression → `(du,dv)` × `gaze_uv_range`, clamp |
| **Refine** | `gaze_refine_left`, `gaze_refine_right` (`nn.Parameter`, shape [2]) — 학습으로 소량 보정 |

기본 범위: `config.gaze_uv_range = 0.12` (나중에 deformer가 refine을 덮어써도 됨).

### 수식 (base)

```text
du = (eyeLookOut - eyeLookIn) * gaze_uv_range
dv = (eyeLookUp   - eyeLookDown) * gaze_uv_range
clamp to [-gaze_uv_range, gaze_uv_range]
effective = clamp(base + refine, same range)
uv_eff = uv_param + effective   # per Gaussian, broadcast
```

### API

```python
# expression 기반 base
avatar(..., expression_weights=exp, expression_names=ict.expression_names.tolist())

# 수동 override
avatar.eyes.set_gaze_base(left=torch.tensor([du, dv]), right=...)
```

## Eyeball h = 0

`UVHGaussians(..., fixed_h=0.0)` — `h`는 buffer, 학습 안 함.

```python
X = S(u, v) + 0 * N(u, v)  ==  S(u, v)
```

## h regularization (face만)

| 영역 | h | loss |
|------|---|------|
| Face skin (`ict.face_indices`) | learnable | `loss_h_anchor_surface` → 0 |
| Eye texture Gaussians | **0 고정** | loss 제외 (`eyeball_mask`) |

```python
n_face = out["face"]["xyz"].shape[0]
eyeball_mask = torch.zeros(out["h"].shape[0], dtype=torch.bool)
eyeball_mask[n_face:] = True

loss_h = loss_h_anchor_surface(
    out["h"], out["is_anchor_surface"], eyeball_mask=eyeball_mask
)
```

## Iris supervision

- MediaPipe 468–477 (좌 5 + 우 5)
- 앞 `n_iris_control=5`개 Gaussian UV를 `IRIS_TEMPLATE_UV`로 초기화
- `out["iris_control_xyz"]` → `losses/iris_landmark.py`

## 관련 파일

| 파일 | 역할 |
|------|------|
| `utils/texture_spaces.py` | `TextureSpaceMeshes`, part 필터 |
| `utils/gaze_uv.py` | expression → gaze UV |
| `utils/uv_mesh.py` | `UVMesh`, `active_face_idx`, bary lookup |
| `model/eye_texture_gaussians.py` | 좌/우 eye Gaussians |
| `model/gaussian_avatar.py` | face + eyes 합성 |
| `model/uvh_gaussians.py` | `fixed_h` 옵션 |
| `losses/h_regularization.py` | skin anchor h prior |
| `losses/eye_uv_barrier.py` | UV [0,1] soft barrier |
| `config.py` | `gaze_uv_range`, `learn_gaze_refine` |

## Deprecated

- `model/eye_plane_gaussians.py`
- `utils/eye_frame.py`

## Smoke test

```bash
python train.py
```

출력 예: texture space별 triangle 수, `gaze_uv L/R`, eye `h max == 0`.
