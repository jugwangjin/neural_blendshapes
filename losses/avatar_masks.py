"""Gaussian-level masks for opacity regularization (tight vs loose face)."""

import torch

# ICT ``face_region_code``: never ``w_opacity_loose`` tight-face targets.
# Teeth (7) excluded so opacity is not forced onto lip pixels.
_LOOSE_OPACITY_REGION_CODES = (0, 1, 2, 3, 5, 6)


def face_region_target_mask(batch):
    """Image-space full-face target (skin + eyes + …); see ``FULL_FACE_REGION_PARTS``."""
    if batch.get("full_face_region_mask") is not None:
        return batch["full_face_region_mask"]
    if batch.get("skin_mask") is not None:
        return batch["skin_mask"]
    return None


def non_tight_face_mask(avatar):
    """
    True for Gaussians outside tight-face skin triangles.

    Tight-face: all three triangle vertices in ``ict.skin_face_indices``.
    Also includes mouth interior, sockets, head, sclera, eye occ via ``face_region_code``.
    """
    surf = avatar.surface
    face_idx = getattr(surf, "face_idx", None)
    if face_idx is None:
        return None
    ict = avatar.ict
    codes = getattr(surf, "face_region_code", None)
    if codes is not None and codes.shape[0] == face_idx.shape[0]:
        region_loose = torch.zeros(face_idx.shape[0], dtype=torch.bool, device=face_idx.device)
        for c in _LOOSE_OPACITY_REGION_CODES:
            region_loose = region_loose | (codes == c)
    else:
        region_loose = None

    skin_vidx = getattr(ict, "skin_face_indices", None)
    if skin_vidx is None:
        return region_loose
    if not torch.is_tensor(skin_vidx):
        skin_vidx = torch.as_tensor(skin_vidx, device=face_idx.device, dtype=torch.long)
    else:
        skin_vidx = skin_vidx.to(device=face_idx.device, dtype=torch.long)
    tri = ict.faces.to(device=face_idx.device, dtype=torch.long)[face_idx.long()]
    n_verts = int(ict.vertex_count)
    skin_mask = torch.zeros(n_verts, dtype=torch.bool, device=face_idx.device)
    skin_mask[skin_vidx] = True
    geom_loose = ~skin_mask[tri].all(dim=1)
    if region_loose is None:
        return geom_loose
    return geom_loose | region_loose


def tight_face_mask(avatar):
    """True when the Gaussian triangle is fully inside ICT tight-face (skin) vertices."""
    surf = avatar.surface
    face_idx = getattr(surf, "face_idx", None)
    if face_idx is None:
        return None
    ict = avatar.ict
    skin_vidx = getattr(ict, "skin_face_indices", None)
    if skin_vidx is None:
        return None
    if not torch.is_tensor(skin_vidx):
        skin_vidx = torch.as_tensor(skin_vidx, device=face_idx.device, dtype=torch.long)
    else:
        skin_vidx = skin_vidx.to(device=face_idx.device, dtype=torch.long)
    if skin_vidx.numel() == 0:
        return None
    tri = ict.faces.to(device=face_idx.device, dtype=torch.long)[face_idx.long()]
    vmask = torch.zeros(int(ict.vertex_count), dtype=torch.bool, device=face_idx.device)
    vmask[skin_vidx] = True
    return vmask[tri].all(dim=1)


def eyelash_vertex_ids(avatar):
    eyelash_vidx = []
    if avatar is not None and getattr(avatar, "ict", None) is not None:
        ict = avatar.ict
        eyelash_vidx.extend(list(getattr(ict, "eyelashes_left_indices", []) or []))
        eyelash_vidx.extend(list(getattr(ict, "eyelashes_right_indices", []) or []))
    return eyelash_vidx
