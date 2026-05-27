"""
Multi-PIE 68 jawline (protocol 0..16) vs deformed ICT mesh vertices.

MediaPipe 478 has no chin contour; jaw shape is anchored at bake via PIE KNN (``nicp.py``).
At train time this loss complements ``w_mp_lmk`` using face_alignment 68-pt detections.
"""

import torch

from losses.mediapipe_landmark_478 import robust_l1


def build_pie68_jaw_vertex_indices(ict, device):
    """ICT vertex ids for PIE jawline ``[0 : landmark_start)``."""
    n = int(ict.landmark_start)
    idx = list(ict.landmark_indices)[:n]
    return torch.tensor(idx, dtype=torch.long, device=device)


def loss_pie68_jawline(
    vertices,
    jaw_vertex_idx,
    landmark_fa,
    camera,
    image_size,
    *,
    score_thresh=0.3,
):
    """
    vertices: [B, V, 3] posed ICT mesh
    jaw_vertex_idx: [J] long, J = landmark_start (17)
    landmark_fa: [B, 68, 4+] — x,y in normalized [0,1] (see ``ImageDataset``), score in [..., 3]
    """
    if not torch.isfinite(vertices).all():
        return vertices.new_zeros(())

    B = vertices.shape[0]
    J = jaw_vertex_idx.shape[0]
    idx = jaw_vertex_idx.to(device=vertices.device)
    jaw_xyz = vertices[:, idx]
    from utils.camera import world_to_camera

    jaw_cam = world_to_camera(jaw_xyz, camera)
    in_front = jaw_cam[..., 2] > 1e-3
    proj = camera.project_world_points(jaw_xyz.reshape(-1, 3)).reshape(B, J, 2)
    pred_uv = proj / float(image_size)

    target_uv = landmark_fa[:, :J, :2].to(device=vertices.device, dtype=vertices.dtype)
    valid = landmark_fa[:, :J, 3].to(device=vertices.device, dtype=vertices.dtype) >= score_thresh
    valid = valid & in_front
    return robust_l1(pred_uv, target_uv, valid=valid)
