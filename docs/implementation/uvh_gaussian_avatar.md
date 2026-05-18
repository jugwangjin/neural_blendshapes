# UVH mesh-embedded Gaussians + GaussianAvatar

SplattingAvatar 스타일: mesh 표면 위 UV + normal displacement로 3D Gaussian 위치를 정의한다.

## 파라미터화

```text
X(u, v, h) = S(u, v) + h · N(u, v)
```

- `S`: barycentric interpolation on deformed ICT mesh
- `N`: per-vertex normals (area-weighted)
- Face: `u, v, h` 모두 learnable
- Eye (별도 texture space): **`h ≡ 0`** — [eye_texture_spaces.md](eye_texture_spaces.md)

## 모듈 구조

```text
model/uvh_gaussians.py       UV, h, log_scale, rotation, opacity, color
model/eye_texture_gaussians.py   좌/우 eye (fixed_h=0, gaze UV slide)
model/gaussian_avatar.py     face + eyes → 단일 xyz/attrs 텐서
utils/uv_mesh.py               UVMesh, uv_to_face_bary, surface_points_from_uvh
utils/texture_spaces.py        ICT vertex_parts → 3× UVMesh
utils/barycentric.py           surface / normal sampling
utils/mesh_ops.py              vertex normals, laplacian
```

## `GaussianAvatar`

### 생성

```python
avatar = GaussianAvatar.from_ict(
    ict,
    n_face_gaussians=65536,
    n_eye_per_side=64,
    gaze_uv_range=0.12,
    learn_gaze_refine=True,
)
```

`from_ict`가 `TextureSpaceMeshes.from_ict`로 face / left_eye / right_eye mesh를 붙인다.

### Forward

```python
verts = ict.forward(expression_weights=exp, to_canonical=False)  # [1,V,3]
out = avatar(
    verts[0],
    ict.faces,
    expression_weights=exp,
    expression_names=ict.expression_names.tolist(),
)
```

### 출력 dict (주요 키)

| Key | 설명 |
|-----|------|
| `xyz`, `scale`, `rotation`, `opacity`, `color`, `h` | 렌더러용 concatenated |
| `face` / `eyes` | 서브모듈 출력 |
| `texture_meshes` | `TextureSpaceMeshes` |
| `is_anchor_surface` | face skin anchor mask + eye (eye는 h=0) |
| `is_eyeball_surface` | eye Gaussians만 |
| `iris_control_xyz` | MP iris loss용 10점 |
| `gaze_uv_left`, `gaze_uv_right` | effective gaze offset [2] |

## Loss (현재 skeleton)

| Loss | 파일 | 비고 |
|------|------|------|
| h anchor | `losses/h_regularization.py` | skin만, eye 제외 |
| iris 2D | `losses/iris_landmark.py` | MP 468–477 |
| eye UV box | `losses/eye_uv_barrier.py` | [0,1] 밖 soft penalty |

가중치: `config.py` — `w_h_anchor`, `w_iris`, `w_eye_uv_barrier`, …

## 카메라

고정 카메라: [default_camera.md](default_camera.md)  
`utils/camera.py` — `FixedCamera.from_default_npz()`

## 알려진 제한

- `uv_to_face_bary`: Python loop (나중에 grid / BVH)
- 3DGS rasterizer / RGB loss: `train.py`는 smoke test 수준
- ICT `forward` eyeball rotation: rendering path에서는 gaze UV로 대체 (mesh rot 무시 권장)

## Config (`config.py`)

```python
n_gaussians: int = 65536
n_eye_gaussians_per_side: int = 64
gaze_uv_range: float = 0.12
learn_gaze_refine: bool = True
w_h_anchor: float = 0.01
w_iris: float = 1.0
w_eye_uv_barrier: float = 0.001
```
