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


def robust_l1(pred, target, valid=None, eps=1e-3):
    """Charbonnier / robust L1 on 2D points. pred/target: [B, N, 2]. valid: [B, N]."""
    diff = torch.sqrt((pred - target).pow(2).sum(dim=-1) + eps * eps)
    if valid is None:
        return diff.mean()
    w = valid.float()
    return (diff * w).sum() / w.sum().clamp(min=1.0)


def embedding_tensors(embedding, device):
    mp_ids = torch.tensor(embedding["mp_landmark_indices"], dtype=torch.long, device=device)
    face_idx = torch.tensor(embedding["ict_lmk_face_idx"], dtype=torch.long, device=device)
    bary = torch.tensor(embedding["ict_lmk_b_coords"], dtype=torch.float32, device=device)
    return mp_ids, face_idx, bary


def loss_mediapipe_landmarks_478(
    vertices,
    faces,
    mp_landmarks_2d,
    embedding,
    camera,
    image_size,
    mp_valid=None,
    mp_ids=None,
    face_idx=None,
    bary=None,
):
    """
    vertices: [B, V, 3] deformed ICT mesh (world)
    mp_landmarks_2d: [B, 478, 2] normalized UV in [0, 1]
    embedding: dict from load_mediapipe_ict_embedding
    """
    device = vertices.device
    if mp_ids is None:
        mp_ids, face_idx, bary = embedding_tensors(embedding, device)

    lmk_xyz = vertices2landmarks_barycentric(vertices, faces, face_idx, bary)
    proj = camera.project_world_points(lmk_xyz.reshape(-1, 3)).reshape(vertices.shape[0], -1, 2)
    pred_uv = proj / float(image_size)

    target_uv = mp_landmarks_2d[:, mp_ids]
    valid = None
    if mp_valid is not None:
        valid = mp_valid[:, mp_ids]
    return robust_l1(pred_uv, target_uv, valid=valid)
