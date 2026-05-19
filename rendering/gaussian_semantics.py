"""Gaussian semantic init from ICT npy region index arrays (not hardcoded part→hair/accessory)."""

import torch
import torch.nn.functional as F

from rendering.semantic import SEMANTIC_CLASS_INDEX

FIXED_SEMANTIC_CLASSES = frozenset({"skin", "lip", "eye", "iris"})
LEARNABLE_SEMANTIC_CLASSES = frozenset({"hair", "accessory", "bg"})


def _list_to_set(ids):
    if torch.is_tensor(ids):
        return set(ids.cpu().tolist())
    return set(ids)


def vertex_semantic_name(ict, vertex_id: int) -> str:
    """Assign semantic class from npy region membership."""
    v = int(vertex_id)
    if v in _list_to_set(ict.eyeball_indices):
        if hasattr(ict, "left_eyeball_indices") and v in _list_to_set(ict.left_eyeball_indices):
            return "eye"
        if hasattr(ict, "right_eyeball_indices") and v in _list_to_set(ict.right_eyeball_indices):
            return "eye"
        return "eye"
    if hasattr(ict, "mouth_interior_vertex_indices") and v in _list_to_set(ict.mouth_interior_vertex_indices):
        return "lip"
    if hasattr(ict, "mouth_socket_indices") and v in _list_to_set(ict.mouth_socket_indices):
        return "lip"
    if hasattr(ict, "teeth_indices") and v in _list_to_set(ict.teeth_indices):
        return "bg"
    if hasattr(ict, "eye_socket_left_indices") and v in _list_to_set(ict.eye_socket_left_indices):
        return "eye"
    if hasattr(ict, "eye_socket_right_indices") and v in _list_to_set(ict.eye_socket_right_indices):
        return "eye"
    if v in _list_to_set(ict.not_face_indices):
        return "skin"
    if v in _list_to_set(ict.face_indices):
        return "skin"
    return "skin"


def ict_vertex_semantic_ids(ict_facekit, device=None):
    n = ict_facekit.neutral_mesh.shape[1]
    out = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        name = vertex_semantic_name(ict_facekit, i)
        out[i] = SEMANTIC_CLASS_INDEX[name]
    if device is not None:
        out = out.to(device)
    return out


def gaussian_semantic_probs_bary(face_ids, bary, faces, vertex_semantic_ids, num_classes):
    tri = faces[face_ids.long()]
    tri_labels = vertex_semantic_ids[tri]
    tri_onehot = F.one_hot(tri_labels, num_classes).float()
    w = bary.float().unsqueeze(-1)
    return (tri_onehot * w).sum(dim=1)


def init_face_gaussian_semantics(face_module, face_idx, bary, ict, faces, learnable_seg_classes=True):
    k = face_module.n_semantic_classes
    if k == 0 or face_module.sem_logits is None:
        return
    v_sem = ict_vertex_semantic_ids(ict, device=face_idx.device)
    prob = gaussian_semantic_probs_bary(face_idx, bary, faces, v_sem, k)
    with torch.no_grad():
        face_module.sem_logits.copy_(torch.log(prob.clamp(1e-4, 1.0)))
        face_module.register_buffer("sem_anchor", prob.clone())
    frozen = torch.zeros(k, dtype=torch.bool)
    for name in FIXED_SEMANTIC_CLASSES:
        if name in SEMANTIC_CLASS_INDEX:
            frozen[SEMANTIC_CLASS_INDEX[name]] = True
    if learnable_seg_classes:
        for name in LEARNABLE_SEMANTIC_CLASSES:
            if name in SEMANTIC_CLASS_INDEX:
                frozen[SEMANTIC_CLASS_INDEX[name]] = False
    face_module.register_buffer("sem_frozen_dims", frozen)


def eye_fixed_semantic_probs(n_per_eye, n_classes, device):
    """Sclera-only eye Gaussians → semantic class ``eye`` (no iris Gaussians)."""
    g = 2 * n_per_eye
    prob = torch.zeros(g, n_classes, device=device)
    eye_i = SEMANTIC_CLASS_INDEX["eye"]
    prob[:, eye_i] = 1.0
    return prob


def semantic_probs_with_anchor(sem_logits, sem_anchor, sem_frozen_dims):
    prob = torch.softmax(sem_logits, dim=-1)
    if sem_anchor is None or sem_frozen_dims is None:
        return prob
    frozen = sem_frozen_dims.view(1, -1)
    return torch.where(frozen, sem_anchor, prob)


def loss_semantic_anchor(sem_logits, sem_anchor, sem_frozen_dims):
    if sem_anchor is None:
        return sem_logits.new_zeros(())
    prob = torch.softmax(sem_logits, dim=-1)
    if sem_frozen_dims is None:
        return (prob - sem_anchor).pow(2).mean()
    frozen = sem_frozen_dims.view(1, -1).float()
    return ((prob - sem_anchor).pow(2) * frozen).mean()
