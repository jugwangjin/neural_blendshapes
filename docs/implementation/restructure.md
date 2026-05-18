# 프로젝트 재구성 (FLARE → UVH 3DGS)

## 목표

- FLAME / nvdiffrast / `NeuralShader` / `flare/core/renderer.py` 제거
- ICT + unit-wise deformer + **UV+h** mesh-embedded Gaussians + 3DGS
- SplattingAvatar 스타일: `X = S(u,v) + h·N(u,v)`

## 새 디렉터리

```text
model/           ict_model, deformer, uvh_gaussians
losses/          rgb, mediapipe_* (setup 후)
dataset/         dataset_util, collate (setup 후)
utils/           mesh_ops, uv_mesh, barycentric, camera, default_camera
gaussian_splatting/vendor/   gaussian-splatting 복사본
assets/          ict, embeddings, default_camera.npz
config.py
train.py
scripts/setup_project_layout.py
```

## 서버에서 한 번 실행

```bash
python scripts/setup_project_layout.py
```

`old/flare/` → 위 layout으로 복사 + import 경로 패치 (`flare.*` → `model.*` / `utils.*`).

`old/` 디렉터리는 참고용으로 두고 **새 코드는 import 하지 않음**.

## UVH + Avatar (구현됨)

| 파일 | 역할 |
|------|------|
| `utils/uv_mesh.py` | `UVMesh`, `active_face_idx`, bary lookup |
| `utils/texture_spaces.py` | ICT `vertex_parts` → face / left_eye / right_eye |
| `utils/gaze_uv.py` | eyeLook* → gaze UV offset |
| `model/uvh_gaussians.py` | UV+h Gaussians (`fixed_h` for eyes) |
| `model/eye_texture_gaussians.py` | per-eye atlas, h=0, gaze slide |
| `model/gaussian_avatar.py` | face + eyes |
| `losses/h_regularization.py` | skin h→0 |
| `losses/iris_landmark.py` | MP iris |
| `losses/eye_uv_barrier.py` | UV box barrier |

상세: [uvh_gaussian_avatar.md](uvh_gaussian_avatar.md), [eye_texture_spaces.md](eye_texture_spaces.md)

## 버릴 것 (`old/` 안에만 유지)

```text
old/flare/core/renderer.py      # nvdiffrast
old/flare/modules/neuralshader.py
old/nvdiffrec/
old/manipulate_*.py             # FLAME 실험 스크립트
```

## 다음 작업

1. `dataset/dataset.py` — FLAME 이름 제거, MP cache, fixed camera
2. `gaussian_splatting/renderer.py` — vendor `diff_gaussian_rasterization` 래핑
3. `losses/` — rgb + mp_lmk2d + lip_iris
4. `uv_to_face_bary` grid 가속 (현재 per-point loop)

## 카메라

```bash
python average_dataset_cameras.py --all-subjects --save-defaults
```

→ `assets/default_camera.npz` → `FixedCamera.from_default_npz()`
