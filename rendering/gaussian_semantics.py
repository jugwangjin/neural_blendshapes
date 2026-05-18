"""ICT vertex_parts → Gaussian semantic labels."""

import torch
import torch.nn.functional as F

from rendering.semantic import SEMANTIC_CLASS_INDEX, SEMANTIC_CLASSES

ICT_PART_TO_SEMANTIC = {
    0: "skin",
    1: "skin",
    2: "lip",
    3: "eye",
    4: "eye",
    5: "hair",
    6: "hair",
    7: "accessory",
}

FIXED_SEMANTIC_CLASSES = frozenset({"skin", "lip", "eye", "iris"})
LEARNABLE_SEMANTIC_CLASSES = frozenset({"hair", "accessory", "bg"})


def ict_vertex_semantic_ids(ict_facekit, device=None):
    parts = ict_facekit.vertex_parts
    out = torch.zeros(len(parts), dtype=torch.long)
    for i, pid in enumerate(parts):
        name = ICT_PART_TO_SEMANTIC.get(int(pid), "skin")
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


def init_face_gaussian_semantics(face_module, face_idx, bary, ict, faces, learnable_hair=True):
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
    if learnable_hair:
        for name in LEARNABLE_SEMANTIC_CLASSES:
            if name in SEMANTIC_CLASS_INDEX:
                frozen[SEMANTIC_CLASS_INDEX[name]] = False
    face_module.register_buffer("sem_frozen_dims", frozen)


def eye_fixed_semantic_probs(n_per_eye, n_iris_control, n_classes, device):
    g = 2 * n_per_eye
    prob = torch.zeros(g, n_classes, device=device)
    eye_i = SEMANTIC_CLASS_INDEX["eye"]
    iris_i = SEMANTIC_CLASS_INDEX["iris"]
    for base in (0, n_per_eye):
        prob[base : base + n_iris_control, iris_i] = 1.0
        prob[base + n_iris_control : base + n_per_eye, eye_i] = 1.0
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
