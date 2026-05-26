"""
ICT ``face_texture_map_id`` / ``material_names`` helpers (``generic_neutral_mesh.obj`` usemtl).

Runtime texture **images** follow ``material_names[tex_id]``; UV coords for sampling use
``triangle_uv_local`` (per-map 0–1) when present, else seam ``uvs`` / ``uv_faces``.
"""

from __future__ import annotations

import numpy as np
import torch

from processing.ict_obj_materials import TEXTURE_MATERIAL_CATALOG, normalize_material_name
from utils.uv_mesh import UVMesh


def material_names_from_ict(ict):
    if hasattr(ict, "material_names") and ict.material_names is not None:
        names = ict.material_names
        if torch.is_tensor(names):
            names = names.detach().cpu().tolist()
        return [normalize_material_name(m) for m in names]
    if hasattr(ict, "face_material_name"):
        raw = ict.face_material_name
        if torch.is_tensor(raw):
            raw = raw.detach().cpu().numpy()
        return sorted({normalize_material_name(m) for m in raw})
    if isinstance(ict, dict):
        if "material_names" in ict:
            return [normalize_material_name(m) for m in ict["material_names"]]
        if "face_material_name" in ict:
            return sorted({normalize_material_name(m) for m in ict["face_material_name"]})
    return []


def all_texture_materials(ict):
    """Stable catalog order; only materials present in the asset."""
    names = material_names_from_ict(ict)
    if not names:
        return tuple(TEXTURE_MATERIAL_CATALOG)
    in_set = set(names)
    ordered = [m for m in TEXTURE_MATERIAL_CATALOG if m in in_set]
    for m in names:
        if m not in ordered:
            ordered.append(m)
    return tuple(ordered)


def _face_texture_map_id_np(ict):
    if hasattr(ict, "face_texture_map_id"):
        t = ict.face_texture_map_id
        if torch.is_tensor(t):
            return t.detach().cpu().numpy().astype(np.int64)
        return np.asarray(t, dtype=np.int64)
    if isinstance(ict, dict) and "face_texture_map_id" in ict:
        return np.asarray(ict["face_texture_map_id"], dtype=np.int64)
    return None


def face_indices_for_material(ict, material_name):
    names = material_names_from_ict(ict)
    mat = normalize_material_name(material_name)
    if mat not in names:
        return np.array([], dtype=np.int64)
    tid = names.index(mat)
    ftmi = _face_texture_map_id_np(ict)
    if ftmi is None:
        return np.array([], dtype=np.int64)
    return np.where(ftmi == tid)[0].astype(np.int64)


def material_name_for_face(ict, face_idx):
    names = material_names_from_ict(ict)
    ftmi = _face_texture_map_id_np(ict)
    if ftmi is None or not names:
        return None
    fi = int(np.asarray(face_idx).reshape(-1)[0])
    tid = int(ftmi[fi])
    if tid < 0 or tid >= len(names):
        return None
    return names[tid]


def bary_to_texture_chart_uv(face_idx, bary, ict):
    """
    Map mesh (face_idx, bary) → chart-local UV [N, 2].

    Uses ``triangle_uv_local`` when in npy; else seam VT bary interpolation.
    """
    fi = face_idx.long() if torch.is_tensor(face_idx) else torch.tensor(face_idx, dtype=torch.long)
    bary_t = bary.float() if torch.is_tensor(bary) else torch.tensor(bary, dtype=torch.float32)
    device = bary_t.device

    if hasattr(ict, "triangle_uv_local") and ict.triangle_uv_local is not None:
        tuv = ict.triangle_uv_local
        if not torch.is_tensor(tuv):
            tuv = torch.tensor(tuv, dtype=torch.float32, device=device)
        else:
            tuv = tuv.to(device=device, dtype=torch.float32)
        tri = tuv[fi]
        return (tri * bary_t[:, :, None]).sum(dim=1)

    uvs = ict.uvs.to(device)
    uf = ict.uv_faces.to(device)
    from utils.barycentric import bary_to_uv_coords

    return bary_to_uv_coords(fi, bary_t, uf, uvs)


def build_material_uv_mesh(ict, material_name, device=None):
    """
    ``UVMesh`` restricted to triangles on one ``usemtl`` material (full 3D topology).

    UV lookup still uses seam ``uvs``; chart-local coords come from ``bary_to_texture_chart_uv``.
    """
    device = device or ict.neutral_mesh.device
    fi = face_indices_for_material(ict, material_name)
    if fi.size == 0:
        return None
    fi_t = torch.tensor(fi, dtype=torch.long, device=device)
    return UVMesh(
        verts=ict.neutral_mesh[0].to(device),
        faces=ict.faces.to(device),
        verts_uvs=ict.uvs.to(device),
        faces_uvs=ict.uv_faces.to(device),
        active_face_idx=fi_t,
    )


def build_all_material_uv_meshes(ict, device=None):
    """Every ``material_names[]`` entry → ``UVMesh`` (or skip if F=0)."""
    device = device or ict.neutral_mesh.device
    out = {}
    for mat in all_texture_materials(ict):
        mesh = build_material_uv_mesh(ict, mat, device=device)
        if mesh is not None:
            out[mat] = mesh
    return out
