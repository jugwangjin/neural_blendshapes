"""Mesh-embedded Gaussian semantics: class from ICT face region, not learnable per-Gaussian logits."""

import torch
import torch.nn.functional as F

from rendering.semantic import SEMANTIC_CLASS_INDEX


def ict_face_semantic_class_table(faces, ict, device):
    """
    Per-mesh-face semantic class [F] from ``surface_triangle_code_table``.

    - 0 gums/tongue, 1 mouth socket, 7 teeth → ``mouth_interior``
    - 2 eye socket, 5 sclera, 6 eye occlusion → ``eye_occlusion``
    - 3 head, 4 face → ``others``
    """
    from utils.ict_regions import surface_triangle_code_table

    codes = surface_triangle_code_table(faces, ict, device)
    others = SEMANTIC_CLASS_INDEX["others"]
    out = torch.full((faces.shape[0],), others, dtype=torch.long, device=device)
    out[(codes == 0) | (codes == 1) | (codes == 7)] = SEMANTIC_CLASS_INDEX["mouth_interior"]
    out[(codes == 2) | (codes == 5) | (codes == 6)] = SEMANTIC_CLASS_INDEX["eye_occlusion"]
    return out


def gaussian_semantic_onehot(face_idx, face_semantic_class, n_classes):
    """[N, K] one-hot features for gsplat semantic raster (fixed, not learned)."""
    cls = face_semantic_class[face_idx.long()]
    return F.one_hot(cls, num_classes=int(n_classes)).float()
