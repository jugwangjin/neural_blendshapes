"""Gray mesh tracking preview + MediaPipe landmark overlay (478 incl. iris)."""

from __future__ import annotations

import torch

from losses.mediapipe_landmark_478 import MP_IRIS_INDEX_LO, vertices2landmarks_barycentric
from rendering.mesh_gray_shadow import render_mesh_gray_shadow


@torch.no_grad()
def overlay_mp_landmarks_on_mesh_rgb(
    rgb_bhw3: torch.Tensor,
    mesh_xyz: torch.Tensor,
    faces: torch.Tensor,
    mp_lmk_emb: dict,
    camera,
    image_size: int,
    *,
    color_bgr=(0, 255, 0),
    radius: int = 2,
    iris_radius: int = 3,
) -> torch.Tensor:
    """
    Draw mesh-projected MediaPipe landmarks (468–477 iris included) on ``rgb`` [B,H,W,3] float.

    Landmarks come from posed ICT vertices via baked barycentric embedding (same as training loss).
    """
    import cv2

    lmk_xyz = vertices2landmarks_barycentric(
        mesh_xyz, faces, mp_lmk_emb["face_idx"], mp_lmk_emb["bary"]
    )
    proj = camera.project_world_points(lmk_xyz.reshape(-1, 3)).reshape(
        mesh_xyz.shape[0], lmk_xyz.shape[1], 2
    )
    uv = (proj / float(image_size))[0].detach().cpu().numpy()

    arr = (rgb_bhw3[0].detach().cpu().numpy().clip(0, 1) * 255.0).round().astype("uint8")
    img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    for i in range(uv.shape[0]):
        px, py = int(round(uv[i, 0] * w)), int(round(uv[i, 1] * h))
        if not (0 <= px < w and 0 <= py < h):
            continue
        r = int(iris_radius if i >= MP_IRIS_INDEX_LO else radius)
        cv2.circle(img_bgr, (px, py), r, color_bgr, -1, lineType=cv2.LINE_AA)

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.astype("float32") / 255.0).unsqueeze(0)


@torch.no_grad()
def render_mesh_gray_shadow_with_mp_landmarks(
    mesh_xyz,
    faces,
    camera,
    mp_lmk_emb,
    *,
    image_size: int | None = None,
    exclude_vertex_ids=None,
    draw_landmarks: bool = True,
    **render_kw,
) -> torch.Tensor:
    """Flat gray Lambert mesh + optional green MP landmark dots (post-process)."""
    rgb = render_mesh_gray_shadow(
        mesh_xyz,
        faces,
        camera,
        image_size=image_size,
        exclude_vertex_ids=exclude_vertex_ids,
        **render_kw,
    )
    if not draw_landmarks:
        return rgb
    sz = int(image_size or getattr(camera, "width", 512))
    return overlay_mp_landmarks_on_mesh_rgb(
        rgb, mesh_xyz, faces, mp_lmk_emb, camera, sz
    )
