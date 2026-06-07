"""Per-ICT-triangle semantic class for mesh raster (same 3-class as surface Gaussians)."""

import torch
import torch.nn.functional as F

from rendering.semantic import SEMANTIC_CLASS_INDEX, SEMANTIC_IGNORE_INDEX
from utils.ict_regions import surface_triangle_code_table


def mesh_triangle_semantic_class(faces, ict, device):
    """
    [F] long class id: others | mouth_interior | eye_occlusion | IGNORE (-1).

    ICT codes 0 gums/tongue, 1 mouth socket, 7 teeth → mouth_interior; 2/5/6 → eye_occlusion; else others.
    """
    codes = surface_triangle_code_table(faces, ict, device)
    others = SEMANTIC_CLASS_INDEX["others"]
    out = torch.full((faces.shape[0],), others, dtype=torch.long, device=device)
    out[(codes == 0) | (codes == 1) | (codes == 7)] = SEMANTIC_CLASS_INDEX["mouth_interior"]
    out[(codes == 2) | (codes == 5) | (codes == 6)] = SEMANTIC_CLASS_INDEX["eye_occlusion"]
    out[codes == -1] = SEMANTIC_IGNORE_INDEX
    return out


def vertex_semantic_onehot(faces, face_cls, n_verts, n_classes):
    """[V, K] one-hot; vertex class = max-priority incident face (mouth > eye > others)."""
    device = faces.device
    pri = torch.zeros(face_cls.shape[0], device=device, dtype=torch.long)
    pri[face_cls == SEMANTIC_CLASS_INDEX["others"]] = 0
    pri[face_cls == SEMANTIC_CLASS_INDEX["eye_occlusion"]] = 1
    pri[face_cls == SEMANTIC_CLASS_INDEX["mouth_interior"]] = 2
    valid = face_cls != SEMANTIC_IGNORE_INDEX
    vert_pri = torch.zeros(int(n_verts), device=device, dtype=torch.long)
    if valid.any():
        f = faces[valid].long()
        p = pri[valid]
        idx = f.reshape(-1)
        src = p.unsqueeze(1).expand(-1, 3).reshape(-1)
        vert_pri.scatter_reduce_(0, idx, src, reduce="amax", include_self=False)
    vert_cls = torch.full((int(n_verts),), SEMANTIC_CLASS_INDEX["others"], device=device, dtype=torch.long)
    vert_cls[vert_pri == 1] = SEMANTIC_CLASS_INDEX["eye_occlusion"]
    vert_cls[vert_pri == 2] = SEMANTIC_CLASS_INDEX["mouth_interior"]
    return F.one_hot(vert_cls, num_classes=int(n_classes)).float()
