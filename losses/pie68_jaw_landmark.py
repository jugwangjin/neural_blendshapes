"""
Multi-PIE 68 landmarks vs deformed ICT mesh vertices (FA detections).

Train loss ``w_pie68_jaw``: PIE jawline protocol ``0:landmark_start`` only (no mouth 48:67; lips via ``w_mp_lmk``).
MediaPipe 478 has no chin contour; jaw is anchored at bake via PIE KNN (``nicp.py``).
"""

import torch

from losses.mediapipe_landmark_478 import robust_l1


def build_pie68_jaw_vertex_indices(ict, device):
    """ICT vertex ids for PIE jawline only ``[0 : landmark_start)`` (bootstrap debug viz)."""
    n = int(ict.landmark_start)
    idx = list(ict.landmark_indices)[:n]
    return torch.tensor(idx, dtype=torch.long, device=device)


def build_pie68_train_landmark_indices(ict, device):
    """
    (vertex_idx, protocol_idx) for ``w_pie68_jaw`` training.

    Jawline only ``0:landmark_start``; FA ``batch['landmark']`` indexed by protocol id.
    """
    ls = int(ict.landmark_start)
    verts = torch.tensor(list(ict.landmark_indices)[:ls], dtype=torch.long, device=device)
    protocol = torch.arange(ls, dtype=torch.long, device=device)
    return verts, protocol


def loss_pie68_landmarks(
    vertices,
    vertex_idx,
    protocol_idx,
    landmark_fa,
    camera,
    image_size,
    *,
    score_thresh=0.3,
    lmk_metric="smooth_l1",
    lmk_eps=1e-4,
    lmk_wing_w_px=10.0,
    lmk_wing_eps_px=2.0,
):
    """
    vertices: [B, V, 3] posed ICT mesh
    vertex_idx: [P] ICT vertex ids
    protocol_idx: [P] FA 68 protocol indices (0..67)
    landmark_fa: [B, 68, 4+] — x,y normalized [0,1], score in [..., 3]
    """
    if not torch.isfinite(vertices).all():
        return vertices.new_zeros(())

    B = vertices.shape[0]
    P = int(protocol_idx.numel())
    vidx = vertex_idx.to(device=vertices.device, dtype=torch.long)
    pidx = protocol_idx.to(device=vertices.device, dtype=torch.long)
    lmk_xyz = vertices[:, vidx]
    from utils.camera import world_to_camera

    lmk_cam = world_to_camera(lmk_xyz, camera)
    in_front = lmk_cam[..., 2] > 1e-3
    proj = camera.project_world_points(lmk_xyz.reshape(-1, 3)).reshape(B, P, 2)
    pred_uv = proj / float(image_size)

    fa = landmark_fa.to(device=vertices.device, dtype=vertices.dtype)
    target_uv = fa[:, pidx, :2]
    valid = fa[:, pidx, 3] >= score_thresh
    valid = valid & in_front
    return robust_l1(
        pred_uv,
        target_uv,
        valid=valid,
        metric=lmk_metric,
        eps=lmk_eps,
        wing_w_px=lmk_wing_w_px,
        wing_eps_px=lmk_wing_eps_px,
        image_size=image_size,
    )


def loss_pie68_jawline(
    vertices,
    jaw_vertex_idx,
    landmark_fa,
    camera,
    image_size,
    *,
    protocol_idx=None,
    score_thresh=0.3,
    lmk_metric="smooth_l1",
    lmk_eps=1e-4,
    lmk_wing_w_px=10.0,
    lmk_wing_eps_px=2.0,
):
    """PIE jawline; ``protocol_idx`` defaults to ``0..J-1`` when omitted."""
    if protocol_idx is None:
        J = jaw_vertex_idx.shape[0]
        protocol_idx = torch.arange(J, device=jaw_vertex_idx.device, dtype=torch.long)
    return loss_pie68_landmarks(
        vertices,
        jaw_vertex_idx,
        protocol_idx,
        landmark_fa,
        camera,
        image_size,
        score_thresh=score_thresh,
        lmk_metric=lmk_metric,
        lmk_eps=lmk_eps,
        lmk_wing_w_px=lmk_wing_w_px,
        lmk_wing_eps_px=lmk_wing_eps_px,
    )
