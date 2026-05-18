# 문서 인덱스

구현·설계 문서는 **`docs/implementation/`** 에 둔다.  
(`internal/` 은 사용하지 않음.)

## 아키텍처

| 문서 | 내용 |
|------|------|
| [implementation/restructure.md](implementation/restructure.md) | FLARE → UVH 3DGS 마이그레이션, 디렉터리 |
| [implementation/uvh_gaussian_avatar.md](implementation/uvh_gaussian_avatar.md) | UVH + `GaussianAvatar` 전체 스택 |
| [implementation/eye_texture_spaces.md](implementation/eye_texture_spaces.md) | 좌/우 eye texture space, gaze UV, h=0 |

## 에셋 / 전처리

| 문서 | 내용 |
|------|------|
| [implementation/default_camera.md](implementation/default_camera.md) | `default_camera.npz` bake·로드 |
| [implementation/ict_mediapipe_lmk_baker.md](implementation/ict_mediapipe_lmk_baker.md) | MediaPipe → ICT landmark embedding |
| [implementation/save_ict_blendshapes.md](implementation/save_ict_blendshapes.md) | ICT blendshape npz export |
| [implementation/ict_mediapipe_lmk.md](implementation/ict_mediapipe_lmk.md) | (요약) MP landmark on ICT |

## 기타

- [loss_memo.txt](loss_memo.txt) — loss 실험 메모 (비정형)

## 빠른 참조 (eye)

```python
from model.gaussian_avatar import GaussianAvatar
from model.ict_model import ICTFaceKitTorch

ict = ICTFaceKitTorch(npy_dir="assets/ict_facekit_torch.npy")
avatar = GaussianAvatar.from_ict(ict, n_face_gaussians=65536, gaze_uv_range=0.12)
verts = ict.forward(expression_weights=exp, to_canonical=False)
out = avatar(verts[0], ict.faces, expression_weights=exp, expression_names=ict.expression_names)
```
