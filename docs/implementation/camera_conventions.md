# Camera conventions: FLARE vs FixedCamera vs gsplat

## FLARE (`old/flare/core/camera.py`)

Documented as **OpenCV pinhole**. Projection:

```text
X_cam = X_world @ R.T + t          (row vectors)
u = fx * X_cam_x / X_cam_z + cx
v = fy * X_cam_y / X_cam_z + cy
```

Dataset build (`dataset_real._parse_frame_single`):

```text
world_mat = pose from cv2.decomposeProjectionMatrix (4×4, c2w-like)
R = world_mat[:3, :3];  R *= -1
t = world_mat[:3, 3]
camera = Camera(K, R, t)
```

`assets/default_camera.npz` (`R_mean`, `t_mean`, `K_mean`) should be the **mean of these** `Camera` objects (same convention).

## FLARE mesh renderer (`old/flare/core/renderer.py`)

Adds **OpenGL-only** Z handling (not used in `Camera.project`):

```text
gl_transform = diag(1, 1, -1, 1)
Rt_gl = gl_transform @ Rt
P = projection_matrix @ Rt_gl   # nvdiffrast clip space
```

MP landmarks, ICT projection losses, and `FixedCamera` follow **`Camera.project`**, not the GL path.

## This repo (`utils/camera.py` + `rendering/gsplat_camera.py`)

`FixedCamera` copies the same algebra as FLARE `Camera.project`:

```python
world_to_camera:  points @ R.T + t
project_points:   fx * x/z + cx,  fy * y/z + cy
```

gsplat pack:

```python
w2v[:3, :3] = R
w2v[:3, 3] = t
viewmats = w2v.unsqueeze(0)   # world-to-camera, OpenCV-style
Ks = cam.K
```

gsplat docs: `viewmats` = **world-to-camera**; pinhole `Ks`; **+Z forward** in camera space (OpenCV).

## Consistency verdict

| Path | Matches FLARE `Camera.project`? | Notes |
|------|----------------------------------|-------|
| MP / iris losses (`camera.project_world_points`) | **Yes** | Same R, t, K from `default_camera.npz` |
| `AvatarRenderer` → gsplat | **Yes** (intended) | Same w2c as projection; no extra GL Z flip |
| FLARE `Renderer` (nvdiffrast) | **No** (by design) | Extra `diag(1,1,-1,1)` for GL clip space only |

If RGB render looks mirrored or inside-out but MP landmark loss is low, suspect **R/t in npz** vs **mesh coordinate frame**, not gsplat K layout.

## Checks on the server

```python
import torch
from utils.camera import FixedCamera
from old.flare.core.camera import Camera

cam_f = FixedCamera.from_default_npz("assets/default_camera.npz", 512, 512)
R, t, K = cam_f.R, cam_f.t, cam_f.K
cam_flare = Camera(K, R, t)

verts = ict.neutral_mesh[0][:100]  # sample
p1 = cam_f.project_world_points(verts)
p2 = cam_flare.project(verts)[:, :2]
print((p1 - p2).abs().max())  # expect ~0
```

Render sanity: project a known vertex; check pixel is inside image and positive depth in `world_to_camera(...)[2]`.

## Resolution

`FixedCamera.from_default_npz(..., width=cfg.image_size, height=cfg.image_size)` — `K` in npz must be for that resolution (FLARE used 512×512). If `image_size != 512`, scale `fx, fy, cx, cy` proportionally (not done automatically today).
