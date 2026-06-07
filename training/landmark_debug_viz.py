"""MP / PIE jaw landmark overlays on GT images (bootstrap debug + stage-end eval)."""

import torch


def eyelash_exclude_vertex_ids(ict):
    vidx = []
    vidx.extend(list(getattr(ict, "eyelashes_left_indices", []) or []))
    vidx.extend(list(getattr(ict, "eyelashes_right_indices", []) or []))
    return vidx


def _annotate_landmarks_bgr(
    img_bgr,
    vertices,
    faces,
    mp_landmarks_2d,
    mp_lmk_emb,
    camera,
    image_size,
    jaw_vertex_idx=None,
    landmark_fa=None,
    jaw_score_thresh=0.3,
):
    """Draw MP (red=GT, green=pred) and optional PIE jaw on a BGR uint8 image."""
    import cv2
    from losses.mediapipe_landmark_478 import vertices2landmarks_barycentric

    mp_ids = mp_lmk_emb["mp_ids"]
    face_idx = mp_lmk_emb["face_idx"]
    bary = mp_lmk_emb["bary"]

    lmk_xyz = vertices2landmarks_barycentric(vertices, faces, face_idx, bary)
    proj = camera.project_world_points(lmk_xyz.reshape(-1, 3)).reshape(vertices.shape[0], -1, 2)
    pred_uv = (proj / float(image_size))[0].detach().cpu().numpy()
    target_uv = mp_landmarks_2d[0, mp_ids].detach().cpu().numpy()

    h, w = img_bgr.shape[:2]
    for i in range(len(target_uv)):
        tx, ty = int(target_uv[i, 0] * w), int(target_uv[i, 1] * h)
        px, py = int(pred_uv[i, 0] * w), int(pred_uv[i, 1] * h)
        cv2.circle(img_bgr, (tx, ty), 2, (0, 0, 255), -1)
        cv2.circle(img_bgr, (px, py), 2, (0, 255, 0), -1)

    if jaw_vertex_idx is not None and landmark_fa is not None:
        J = int(jaw_vertex_idx.numel())
        idx = jaw_vertex_idx.to(device=vertices.device, dtype=torch.long)
        jaw_xyz = vertices[:, idx]
        jaw_proj = camera.project_world_points(jaw_xyz.reshape(-1, 3)).reshape(
            vertices.shape[0], J, 2
        )
        pred_jaw_uv = (jaw_proj / float(image_size))[0].detach().cpu().numpy()
        fa = landmark_fa[0] if landmark_fa.ndim == 3 else landmark_fa
        target_jaw_uv = fa[:J, :2].detach().cpu().numpy()
        jaw_score = fa[:J, 3].detach().cpu().numpy()
        for j in range(J):
            px, py = int(pred_jaw_uv[j, 0] * w), int(pred_jaw_uv[j, 1] * h)
            cv2.circle(img_bgr, (px, py), 3, (0, 255, 255), -1)
            if float(jaw_score[j]) >= float(jaw_score_thresh):
                tx, ty = int(target_jaw_uv[j, 0] * w), int(target_jaw_uv[j, 1] * h)
                cv2.circle(img_bgr, (tx, ty), 3, (255, 0, 255), -1)

    return img_bgr


@torch.no_grad()
def save_landmark_debug_image(
    path,
    vertices,
    faces,
    mp_landmarks_2d,
    mp_lmk_emb,
    camera,
    image_size,
    gt_image,
    gt_mask=None,
    exclude_vertex_ids=None,
    jaw_vertex_idx=None,
    landmark_fa=None,
    jaw_score_thresh=0.3,
):
    import cv2
    import numpy as np
    from losses.mesh_silhouette import render_mesh_silhouette_alpha

    img = gt_image.detach().cpu().permute(1, 2, 0).numpy()
    img = (img.clip(0, 1) * 255.0).round().astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _annotate_landmarks_bgr(
        img_bgr,
        vertices,
        faces,
        mp_landmarks_2d,
        mp_lmk_emb,
        camera,
        image_size,
        jaw_vertex_idx=jaw_vertex_idx,
        landmark_fa=landmark_fa,
        jaw_score_thresh=jaw_score_thresh,
    )
    cv2.imwrite(str(path), img_bgr)

    base = path.with_suffix("")
    if gt_mask is not None:
        m = gt_mask.detach().float().cpu()
        if m.ndim == 3 and m.shape[0] == 1:
            m = m[0]
        m = (m.clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
        cv2.imwrite(str(base) + "_gt_mask.jpg", m)

    mesh_alpha = render_mesh_silhouette_alpha(
        vertices,
        faces,
        camera,
        image_size=image_size,
        downsample=1,
        exclude_vertex_ids=exclude_vertex_ids,
    )
    rm = mesh_alpha[0, 0].detach().float().cpu()
    rm = (rm.clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
    cv2.imwrite(str(base) + "_mesh_mask.jpg", rm)

    if gt_mask is not None:
        overlay = np.zeros((rm.shape[0], rm.shape[1], 3), dtype=np.uint8)
        overlay[..., 1] = rm
        overlay[..., 2] = m
        cv2.imwrite(str(base) + "_sil_overlay.jpg", overlay)
