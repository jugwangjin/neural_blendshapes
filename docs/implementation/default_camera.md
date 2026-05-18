# default_camera (assets/)

`average_dataset_cameras.py --all-subjects --save-defaults` 로 flare_2 데이터셋 전체 frame 평균 카메라를 구워 저장한다.

## 파일

- `assets/default_camera.npz`
- `assets/default_camera.txt`

## NPZ keys

| Key | Shape | 설명 |
|-----|-------|------|
| `intrinsics_norm` | (4,) | json `intrinsics` [fx, fy, cx, cy] |
| `K` | (3,3) | 픽셀 intrinsics (512 기준 평균) |
| `R` | (3,3) | OpenGL convention (`world_mat` 파싱 후 R*=−1) |
| `t` | (3,) | translation |
| `s` | (1,) | scale (고정 1.0, ICT canonical `s` 자리 호환) |
| `center` | (3,) | −Rᵀt |
| `resolution` | (2,) | [H, W] |
| `subjects_used` | object array | bake에 쓴 subject 이름 |

## 로드

```python
from flare.utils.default_camera import load_default_camera

cam = load_default_camera()
K, R, t = cam["K"], cam["R"], cam["t"]
intr = cam["intrinsics_norm"]  # fx, fy, cx, cy
```

`Camera` 클래스 없이 numpy dict만 사용.

## scene 탐색

`train_dir` / `eval_dir` 이름은 `input_dir/<scene>/flame_params.json` 직접 경로를 먼저 본다 (MVI_* 등).
