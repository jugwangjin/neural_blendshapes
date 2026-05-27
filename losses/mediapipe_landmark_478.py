"""
MediaPipe landmark loss on ICT mesh via baked barycentric embedding.

Source of truth: assets/ict_mediapipe_landmark_indices.npz
(mp_landmark_indices, ict_lmk_face_idx, ict_lmk_b_coords).

Do NOT use ict_facekit.landmark_indices (legacy 68 ICT vertex list).
"""

import numpy as np
import torch

from utils.barycentric import vertices2landmarks


def load_mediapipe_ict_embedding(path):
    from processing.ict_mediapipe_lmk.embedding_io import load_ict_mediapipe_embedding

    emb = load_ict_mediapipe_embedding(path)
    d = np.load(path, allow_pickle=True)
    for k in ("ict_lmk_target_type", "transfer_error", "source"):
        if k in d:
            emb[k] = d[k]
    return emb


def vertices2landmarks_barycentric(vertices, faces, face_idx, bary):
    """
    vertices: [B, V, 3]
    faces: [F, 3]
    face_idx: [N]
    bary: [N, 3]
    -> [B, N, 3]
    """
    return vertices2landmarks(vertices, faces, face_idx, bary)


MP_IRIS_INDEX_LO = 468  # MP 468–472 (L), 473–477 (R)


def robust_l1(pred, target, valid=None, point_weight=None, eps=1e-3):
    """Charbonnier / robust L1 on 2D points. pred/target: [B, N, 2]. valid: [B, N]."""
    diff = torch.sqrt((pred - target).pow(2).sum(dim=-1) + eps * eps)
    w = torch.ones_like(diff)
    if point_weight is not None:
        w = w * point_weight.view(1, -1).to(device=diff.device, dtype=diff.dtype)
    if valid is not None:
        w = w * valid.to(device=diff.device, dtype=diff.dtype)
    finite = torch.isfinite(pred).all(dim=-1) & torch.isfinite(target).all(dim=-1)
    w = w * finite.to(dtype=diff.dtype)
    denom = w.sum().clamp(min=1.0)
    return (diff * w).sum() / denom


def build_mp_lmk_embedding(path, device):
    """
    Load NPZ once and return fixed GPU tensors for training.

    Keys: ``mp_ids``, ``face_idx``, ``bary`` (all on ``device``).
    Call from ``train.py`` at startup; pass the dict into ``loss_mediapipe_landmarks_478``.
    """
    emb = load_mediapipe_ict_embedding(path)
    return {
        "mp_ids": torch.tensor(emb["mp_landmark_indices"], dtype=torch.long, device=device),
        "face_idx": torch.tensor(emb["ict_lmk_face_idx"], dtype=torch.long, device=device),
        "bary": torch.tensor(emb["ict_lmk_b_coords"], dtype=torch.float32, device=device),
    }


def loss_mediapipe_landmarks_478(
    vertices,
    faces,
    mp_landmarks_2d,
    mp_lmk_emb,
    camera,
    image_size,
    mp_valid=None,
    iris_weight=2.5,
):
    """
    vertices: [B, V, 3] deformed ICT mesh (world)
    mp_landmarks_2d: [B, 478, 2] normalized UV in [0, 1]
    mp_lmk_emb: dict from ``build_mp_lmk_embedding`` (``mp_ids``, ``face_idx``, ``bary``)
    iris_weight: per-point multiplier for MP iris indices 468–477 (face/eyelid use 1.0)
    """
    # Since mp_lmk_emb and faces are already fully initialized on the target device
    # during build_mp_lmk_embedding and main training script, we avoid calling
    # .to(device) repeatedly inside the training loop unless the devices are different.
    mp_ids = mp_lmk_emb["mp_ids"]
    face_idx = mp_lmk_emb["face_idx"]
    bary = mp_lmk_emb["bary"]

    # Quick check for device/dtype compatibility
    if mp_ids.device != vertices.device:
        mp_ids = mp_ids.to(device=vertices.device)
        face_idx = face_idx.to(device=vertices.device)
    if bary.device != vertices.device or bary.dtype != vertices.dtype:
        bary = bary.to(device=vertices.device, dtype=vertices.dtype)
    if faces.device != vertices.device:
        faces = faces.to(device=vertices.device)

    if not torch.isfinite(vertices).all():
        return vertices.new_zeros(())

    lmk_xyz = vertices2landmarks_barycentric(vertices, faces, face_idx, bary)
    from utils.camera import world_to_camera

    lmk_cam = world_to_camera(lmk_xyz, camera)
    in_front = lmk_cam[..., 2] > 1e-3
    proj = camera.project_world_points(lmk_xyz.reshape(-1, 3)).reshape(vertices.shape[0], -1, 2)
    pred_uv = proj / float(image_size)

    target_uv = mp_landmarks_2d[:, mp_ids].to(device=vertices.device, dtype=vertices.dtype)
    valid = in_front.to(device=vertices.device, dtype=vertices.dtype)
    if mp_valid is not None:
        valid = valid * mp_valid[:, mp_ids].to(device=vertices.device, dtype=vertices.dtype)

    point_weight = torch.ones(mp_ids.shape[0], device=vertices.device, dtype=vertices.dtype)
    if iris_weight != 1.0:
        point_weight = torch.where(
            mp_ids >= MP_IRIS_INDEX_LO,
            point_weight.new_tensor(float(iris_weight)),
            point_weight,
        )
    return robust_l1(pred_uv, target_uv, valid=valid, point_weight=point_weight)
