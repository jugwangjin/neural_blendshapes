"""
ICT region helpers — source of truth is assets/ict_facekit_torch.npy index arrays.

Do not hardcode official ICT README vertex ranges at runtime; use ict.face_indices,
ict.surface_sample_vertex_indices, ict.eyeball_indices, etc. from the loaded npy.
"""

import numpy as np
import torch

# Surface mesh Gaussians: skin/head/mouth + sparse sclera / eye-occlusion (ICT blendshapes).
# Sclera & occlusion are sampled here (h pinned to 0); iris/lashes charts stay excluded.
SURFACE_EXCLUDED_MATERIALS = frozenset(
    {
        "M_IrisLeft",
        "M_IrisRight",
        "M_EyeballLeft",
        "M_EyeballRight",
        "M_LacrimalFluid",
        "M_EyeBlend",
        "M_EyeLashes",
    }
)

SCLERA_MATERIALS = frozenset({"M_ScleraLeft", "M_ScleraRight"})
EYE_OCCLUSION_MATERIAL = "M_EyeOcclusion"


def _as_long_tensor(ids, device):
    if torch.is_tensor(ids):
        return ids.to(device=device, dtype=torch.long)
    return torch.tensor(list(ids), device=device, dtype=torch.long)


def _vertex_mask(n_verts, ids, device):
    m = torch.zeros(n_verts, dtype=torch.bool, device=device)
    m[_as_long_tensor(ids, device)] = True
    return m


def filter_triangles_all_vertices_in(faces, allowed_vertex_ids, device=None):
    """Triangles whose three vertices are all in allowed_vertex_ids."""
    device = device or faces.device
    if torch.is_tensor(allowed_vertex_ids) and allowed_vertex_ids.dtype == torch.bool:
        allowed = allowed_vertex_ids.to(device)
    else:
        n_verts = int(faces.max().item()) + 1
        allowed = _vertex_mask(n_verts, allowed_vertex_ids, device)
    tri = faces.long()
    return torch.where(allowed[tri].all(dim=1))[0]


def filter_triangles_exclude_vertices(faces, exclude_vertex_ids, device=None):
    """Drop triangles that touch any vertex in ``exclude_vertex_ids``."""
    device = device or faces.device
    if exclude_vertex_ids is None or len(exclude_vertex_ids) == 0:
        return torch.arange(faces.shape[0], device=device, dtype=torch.long)
    n_verts = int(faces.max().item()) + 1
    excluded = _vertex_mask(n_verts, exclude_vertex_ids, device)
    tri = faces.long()
    keep = ~excluded[tri].any(dim=1)
    return torch.where(keep)[0]


def surface_excluded_vertex_ids(ict):
    """
    Vertices excluded from skin/head surface layout (not ``M_EyeOcclusion`` / sclera tris).

    Teeth + lacrimal / eye-blend / eyelashes aux parts; eyeball tris handled via material mask.
    """
    out = list(getattr(ict, "teeth_indices", []) or [])
    for key in (
        "lacrimal_indices",
        "eye_blend_indices",
        "eyelashes_left_indices",
        "eyelashes_right_indices",
    ):
        out.extend(list(getattr(ict, key, []) or []))
    return out


def _material_face_mask(ict, material_names):
    names = _face_material_names_np(ict)
    if names is None:
        return None
    from processing.ict_obj_materials import normalize_material_name

    want = {normalize_material_name(m) for m in material_names}
    out = np.zeros(len(names), dtype=bool)
    for i, raw in enumerate(names):
        if normalize_material_name(str(raw)) in want:
            out[i] = True
    return out


def sclera_layout_face_indices(ict):
    """``M_ScleraLeft`` / ``M_ScleraRight`` faces (forward hemisphere sampling set)."""
    from utils.eye_chart import sclera_face_indices

    fi_l = sclera_face_indices(ict, "L")
    fi_r = sclera_face_indices(ict, "R")
    if fi_l.size == 0 and fi_r.size == 0:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate([fi_l, fi_r]).astype(np.int64)


def eye_occlusion_layout_face_indices(ict):
    """``M_EyeOcclusion`` — deformed by ICT eye-occlusion blendshapes."""
    mask = _material_face_mask(ict, {EYE_OCCLUSION_MATERIAL})
    if mask is None:
        return np.zeros(0, dtype=np.int64)
    return np.where(mask)[0].astype(np.int64)


def eye_gaussian_layout_face_indices(ict):
    """Sclera + eye-occlusion triangles for sparse white surface Gaussians."""
    parts = [sclera_layout_face_indices(ict), eye_occlusion_layout_face_indices(ict)]
    parts = [p for p in parts if p.size > 0]
    if not parts:
        return np.zeros(0, dtype=np.int64)
    return np.unique(np.concatenate(parts)).astype(np.int64)


def _face_material_names_np(ict):
    if hasattr(ict, "face_material_name"):
        names = ict.face_material_name
    elif isinstance(ict, dict) and "face_material_name" in ict:
        names = ict["face_material_name"]
    else:
        return None
    if torch.is_tensor(names):
        names = names.detach().cpu().numpy()
    return np.asarray(names, dtype=object)


def surface_excluded_material_mask(ict):
    """Boolean [F] — eye / sclera / iris OBJ charts (not skin surface)."""
    names = _face_material_names_np(ict)
    if names is None:
        return None
    from processing.ict_obj_materials import normalize_material_name

    out = np.zeros(len(names), dtype=bool)
    for i, raw in enumerate(names):
        if normalize_material_name(str(raw)) in SURFACE_EXCLUDED_MATERIALS:
            out[i] = True
    return out


def surface_excluded_material_mask_torch(ict, device):
    mask = surface_excluded_material_mask(ict)
    if mask is None:
        return None
    return torch.tensor(mask, dtype=torch.bool, device=device)


def surface_layout_triangle_ids(ict, faces, device=None):
    """
    Triangle indices for ``build_surface_gaussian_layout``.

    Skin/head/mouth/eye-socket (``surface_allowed_vertices``) plus sclera +
    ``M_EyeOcclusion`` faces. Teeth and iris/lash eye charts excluded.
    """
    device = device or faces.device
    allowed = surface_allowed_vertices(ict)
    tri_ids = filter_triangles_all_vertices_in(faces, allowed, device=device)
    if tri_ids.numel() == 0:
        tri_skin = tri_ids
    else:
        exclude_verts = surface_excluded_vertex_ids(ict)
        if exclude_verts:
            keep = filter_triangles_exclude_vertices(faces, exclude_verts, device=device)
            tri_ids = tri_ids[torch.isin(tri_ids, keep)]
        mat_mask = surface_excluded_material_mask_torch(ict, device)
        if mat_mask is not None and tri_ids.numel() > 0:
            tri_ids = tri_ids[~mat_mask[tri_ids]]
        tri_skin = tri_ids

    eye_fi = eye_gaussian_layout_face_indices(ict)
    if eye_fi.size == 0:
        return tri_skin
    tri_eye = torch.tensor(eye_fi, dtype=torch.long, device=device)
    if tri_skin.numel() == 0:
        return tri_eye
    return torch.unique(torch.cat([tri_skin, tri_eye], dim=0))


def surface_allowed_vertices(ict):
    exclude = set(surface_excluded_vertex_ids(ict))
    if hasattr(ict, "surface_sample_vertex_indices"):
        base = list(ict.surface_sample_vertex_indices)
    else:
        face = list(ict.face_indices)
        head = list(ict.not_face_indices)
        base = [i for i in face + head if i not in exclude]
    # Older npy may omit eye-socket from ``surface_sample_vertex_indices``.
    extra = []
    for key in ("eye_socket_left_indices", "eye_socket_right_indices"):
        ids = getattr(ict, key, None)
        if ids:
            extra.extend(list(ids))
    seen = set(base)
    for i in extra:
        if i not in exclude and i not in seen:
            base.append(i)
            seen.add(i)
    return [i for i in base if i not in exclude]


def classify_surface_triangles_batch(tri_ids, faces, ict, device):
    """
    Vectorized region tags for mesh triangle indices.

    Returns int64 codes: 0 mouth_interior, 1 mouth_socket, 2 eye_socket,
    3 head, 4 face, 5 eyeball_sclera, 6 eye_occlusion, -1 skip.
    """
    code_by_face = surface_triangle_code_table(faces, ict, device)
    return code_by_face[tri_ids.long()]


def surface_triangle_code_table(faces, ict, device):
    """Cached [F] table for ``classify_surface_triangles_batch``."""
    cache = getattr(ict, "_surface_triangle_code_cache", None)
    key = str(device)
    if isinstance(cache, dict) and key in cache:
        table = cache[key]
        if table.device == torch.device(device) and table.shape[0] == faces.shape[0]:
            return table

    table = _build_surface_triangle_code_table(faces, ict, device)
    if not isinstance(cache, dict):
        cache = {}
        setattr(ict, "_surface_triangle_code_cache", cache)
    cache[key] = table
    return table


def _build_surface_triangle_code_table(faces, ict, device):
    faces = faces.long().to(device)
    n_faces = faces.shape[0]
    n_verts = int(faces.max().item()) + 1
    tri = faces

    def any_vertex_in(ids):
        if ids is None or len(ids) == 0:
            return torch.zeros(n_faces, dtype=torch.bool, device=device)
        return _vertex_mask(n_verts, ids, device)[tri].any(dim=1)

    def all_vertices_in(ids):
        return _vertex_mask(n_verts, ids, device)[tri].all(dim=1)

    on_sclera = _material_face_mask_torch_cached(ict, SCLERA_MATERIALS, device, "sclera")
    on_occ = _material_face_mask_torch_cached(ict, {EYE_OCCLUSION_MATERIAL}, device, "eye_occ")
    if on_sclera is None:
        on_sclera = torch.zeros(n_faces, dtype=torch.bool, device=device)
    if on_occ is None:
        on_occ = torch.zeros(n_faces, dtype=torch.bool, device=device)

    skip = any_vertex_in(getattr(ict, "teeth_indices", []))
    skip = skip | (any_vertex_in(ict.eyeball_indices) & ~(on_sclera | on_occ))
    mat_mask = surface_excluded_material_mask_torch(ict, device)
    if mat_mask is not None:
        skip = skip | (mat_mask & ~(on_sclera | on_occ))

    gums = any_vertex_in(
        getattr(ict, "mouth_interior_vertex_indices", getattr(ict, "gums_tongue_indices", []))
    )
    mouth_sock = any_vertex_in(getattr(ict, "mouth_socket_indices", []))
    eye_sock = any_vertex_in(getattr(ict, "eye_socket_left_indices", [])) | any_vertex_in(
        getattr(ict, "eye_socket_right_indices", [])
    )
    head = all_vertices_in(ict.not_face_indices)

    code = torch.full((n_faces,), 4, dtype=torch.long, device=device)
    code[head] = 3
    code[eye_sock] = 2
    code[mouth_sock] = 1
    code[gums] = 0
    code[on_sclera] = 5
    code[on_occ] = 6
    code[skip] = -1
    return code


def _material_face_mask_torch_cached(ict, material_names, device, cache_name):
    cache = getattr(ict, "_material_face_mask_cache", None)
    key = (cache_name, str(device))
    if isinstance(cache, dict) and key in cache:
        return cache[key]

    mask = _material_face_mask(ict, material_names)
    out = None if mask is None else torch.tensor(mask, dtype=torch.bool, device=device)
    if not isinstance(cache, dict):
        cache = {}
        setattr(ict, "_material_face_mask_cache", cache)
    cache[key] = out
    return out


def classify_surface_triangle(fi, faces, ict, device):
    """Return 'skip' | 'mouth_interior' | 'mouth_socket' | 'eye_socket' | 'head' | 'face'."""
    tri = faces[fi].tolist()
    teeth = set(getattr(ict, "teeth_indices", []))
    gums = set(getattr(ict, "mouth_interior_vertex_indices", getattr(ict, "gums_tongue_indices", [])))
    mouth_sock = set(getattr(ict, "mouth_socket_indices", []))
    eye_sock = set(getattr(ict, "eye_socket_left_indices", [])) | set(
        getattr(ict, "eye_socket_right_indices", [])
    )
    head = set(ict.not_face_indices)
    eye = set(ict.eyeball_indices)

    if any(v in eye for v in tri):
        return "skip"
    if any(v in teeth for v in tri):
        return "skip"
    if any(v in gums for v in tri):
        return "mouth_interior"
    if any(v in mouth_sock for v in tri):
        return "mouth_socket"
    if any(v in eye_sock for v in tri):
        return "eye_socket"
    if all(v in head for v in tri):
        return "head"
    return "face"


def eyeball_left_vertices(ict):
    if hasattr(ict, "left_eyeball_indices"):
        return ict.left_eyeball_indices
    return list(range(21451, 23021))


def eyeball_right_vertices(ict):
    if hasattr(ict, "right_eyeball_indices"):
        return ict.right_eyeball_indices
    return list(range(23021, 24591))


def iris_vertices(ict, side):
    if side == "L":
        return list(getattr(ict, "left_iris_indices", []))
    return list(getattr(ict, "right_iris_indices", []))


def sclera_vertices(ict, side):
    """
    Vertices touched by ``M_Sclera*`` triangles (preferred over eyeball−iris heuristic).

    Falls back to eyeball minus iris vertex ids when ``face_material_name`` is missing.
    """
    if hasattr(ict, "face_material_name") or (
        isinstance(ict, dict) and "face_material_name" in ict
    ):
        from utils.eye_chart import _ict_faces_np, sclera_face_indices

        faces = _ict_faces_np(ict)
        fi = sclera_face_indices(ict, side)
        if fi.size > 0:
            return np.unique(faces[fi].reshape(-1)).tolist()

    if side == "L":
        eye = list(eyeball_left_vertices(ict))
    else:
        eye = list(eyeball_right_vertices(ict))
    iris = set(iris_vertices(ict, side))
    return [v for v in eye if v not in iris]
