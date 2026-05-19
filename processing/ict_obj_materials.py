"""
UV: face-level atlas tile for local UV / seam split.
Texture image selection: OBJ usemtl (material_names).
"""

from __future__ import annotations

import numpy as np

# face_texture_map_id catalog order (known materials first).
TEXTURE_MATERIAL_CATALOG = [
    "M_Face",
    "M_BackHead",
    "M_GumsTongue",
    "M_ScleraLeft",
    "M_ScleraRight",
    "M_Teeth",
    "M_LacrimalFluid",
    "M_EyeLashes",
    "M_IrisLeft",
    "M_IrisRight",
    "M_EyeballLeft",
    "M_EyeballRight",
    "M_EyeOcclusion",
    "M_EyeBlend",
]

# Textures used at runtime (others mapped for completeness).
PRIMARY_TEXTURE_MATERIALS = [
    "M_Face",
    "M_BackHead",
    "M_GumsTongue",
    "M_ScleraLeft",
]

_MATERIAL_CANON = {m.lower(): m for m in TEXTURE_MATERIAL_CATALOG}


def normalize_material_name(name):
    key = str(name).strip()
    low = key.lower()
    if low in _MATERIAL_CANON:
        return _MATERIAL_CANON[low]
    return key


def parse_obj_face_materials(obj_path):
    """One usemtl per OBJ `f` line (tri or quad), file order — matches openmesh face order."""
    materials = []
    current_mtl = "UNKNOWN"
    with open(obj_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("usemtl "):
                current_mtl = line.split()[1]
            elif line.startswith("f "):
                corners = line.split()[1:]
                if len(corners) >= 3:
                    materials.append(current_mtl)
    return materials


parse_obj_quad_materials = parse_obj_face_materials


def build_texture_map_index_from_materials(tri_material_names):
    """
    face_texture_map_id from usemtl name per triangle.
    Unknown materials in OBJ are appended after TEXTURE_MATERIAL_CATALOG.
    """
    names = [normalize_material_name(n) for n in tri_material_names]
    found = set(names)

    catalog = [m for m in TEXTURE_MATERIAL_CATALOG if m in found]
    for m in sorted(found):
        if m not in catalog:
            catalog.append(m)

    mtl_to_id = {m: i for i, m in enumerate(catalog)}
    face_texture_map_id = np.array([mtl_to_id[n] for n in names], dtype=np.int32)

    return {
        "face_texture_map_id": face_texture_map_id,
        "face_material_name": np.asarray(names, dtype=object),
        "material_names": np.asarray(catalog, dtype=object),
        "n_texture_maps": int(len(catalog)),
        "primary_texture_materials": np.asarray(PRIMARY_TEXTURE_MATERIALS, dtype=object),
    }


def _is_tri_face(face):
    face = np.asarray(face, dtype=np.int32).ravel()
    if face.size == 3:
        return True
    if face.size >= 4 and face[3] < 0:
        return True
    return False


def convert_quad_mesh_to_triangle_mesh(faces, uvs, face_materials=None, n_verts=None):
    """
    Tri or quad faces → triangle list (1 tri per tri face, 2 per quad).
    Drops tris with vertex index < 0 or (if n_verts set) index >= n_verts.
    """
    faces = np.asarray(faces)
    uvs = np.asarray(uvs)
    if face_materials is not None:
        face_materials = list(face_materials)

    tri_faces_list = []
    tri_uv_list = []
    tri_mtl_list = []

    for i in range(len(faces)):
        face = np.asarray(faces[i], dtype=np.int32).ravel()
        uv = np.asarray(uvs[i], dtype=np.float64)
        mtl = face_materials[i] if face_materials is not None else None

        if _is_tri_face(face):
            tri_faces_list.append(face[:3])
            tri_uv_list.append(uv[:3])
            if mtl is not None:
                tri_mtl_list.append(mtl)
        else:
            f4 = face[:4]
            u4 = uv[:4]
            tri_faces_list.append(f4[[0, 1, 2]])
            tri_faces_list.append(f4[[2, 3, 0]])
            tri_uv_list.append(u4[[0, 1, 2]])
            tri_uv_list.append(u4[[2, 3, 0]])
            if mtl is not None:
                tri_mtl_list.append(mtl)
                tri_mtl_list.append(mtl)

    triangle_mesh = np.asarray(tri_faces_list, dtype=np.int32)
    triangle_uv = np.asarray(tri_uv_list, dtype=np.float64)
    keep = np.all(triangle_mesh >= 0, axis=1)
    if n_verts is not None:
        keep &= np.all(triangle_mesh < n_verts, axis=1)
    out_faces = triangle_mesh[keep]
    out_uv = triangle_uv[keep]
    if face_materials is None:
        return out_faces, out_uv
    tri_mtl = np.array(tri_mtl_list, dtype=object)[keep]
    return out_faces, out_uv, tri_mtl


def infer_face_uv_tiles(triangle_uv_atlas, eps=1e-6, strict=True):
    """
    triangle_uv_atlas: [F, 3, 2] atlas-space UV.

    Returns:
        face_uv_tile: [F, 2] int32
        bad_faces: face indices with no valid single-tile fit
    """
    uv = np.asarray(triangle_uv_atlas, dtype=np.float64)
    f = uv.shape[0]

    face_uv_tile = np.zeros((f, 2), dtype=np.int32)
    bad_faces = []

    for fi in range(f):
        tri = uv[fi]

        candidates = set()
        candidates.add(tuple(np.floor(np.mean(tri, axis=0)).astype(np.int32)))
        candidates.add(tuple(np.floor(np.min(tri, axis=0) + eps).astype(np.int32)))
        candidates.add(tuple(np.floor(np.max(tri, axis=0) - eps).astype(np.int32)))

        for p in tri:
            candidates.add(tuple(np.floor(p).astype(np.int32)))
            candidates.add(tuple(np.floor(p - eps).astype(np.int32)))
            candidates.add(tuple(np.floor(p + eps).astype(np.int32)))

        valid = []
        for cand in candidates:
            tile = np.asarray(cand, dtype=np.float64)
            local = tri - tile[None, :]
            ok = (
                np.all(local[:, 0] >= -eps)
                and np.all(local[:, 0] <= 1.0 + eps)
                and np.all(local[:, 1] >= -eps)
                and np.all(local[:, 1] <= 1.0 + eps)
            )
            if ok:
                mean_local = np.mean(local, axis=0)
                score = float(np.sum((mean_local - 0.5) ** 2))
                valid.append((score, cand))

        if len(valid) == 0:
            bad_faces.append(fi)
            fallback = tuple(np.floor(np.mean(tri, axis=0)).astype(np.int32))
            face_uv_tile[fi] = fallback
        else:
            valid.sort(key=lambda x: x[0])
            face_uv_tile[fi] = np.asarray(valid[0][1], dtype=np.int32)

    bad_faces = np.asarray(bad_faces, dtype=np.int32)
    if strict and len(bad_faces) > 0:
        raise ValueError(
            f"{len(bad_faces)} faces do not fit in a single UV tile. "
            f"Examples: {bad_faces[:10].tolist()}"
        )

    return face_uv_tile, bad_faces


def localize_triangle_uv_by_face_tile(triangle_uv_atlas, face_uv_tile, eps=1e-6):
    tri_local = (
        np.asarray(triangle_uv_atlas, dtype=np.float64)
        - np.asarray(face_uv_tile, dtype=np.float64)[:, None, :]
    )
    tri_local[np.abs(tri_local) < eps] = 0.0
    tri_local[np.abs(tri_local - 1.0) < eps] = 1.0

    if np.any(tri_local < -eps) or np.any(tri_local > 1.0 + eps):
        mn = float(tri_local.min())
        mx = float(tri_local.max())
        raise ValueError(f"Localized UV out of [0,1]: min={mn}, max={mx}")

    return np.clip(tri_local, 0.0, 1.0).astype(np.float32)


def build_texture_map_index_from_tiles(face_uv_tile):
    face_uv_tile = np.asarray(face_uv_tile, dtype=np.int32)
    unique_tiles = np.unique(face_uv_tile, axis=0)
    tile_to_id = {(int(t[0]), int(t[1])): i for i, t in enumerate(unique_tiles)}
    face_texture_map_id = np.array(
        [tile_to_id[(int(t[0]), int(t[1]))] for t in face_uv_tile],
        dtype=np.int32,
    )
    return {
        "face_texture_map_id": face_texture_map_id,
        "face_uv_tile_u": face_uv_tile[:, 0].astype(np.int32),
        "face_uv_tile_v": face_uv_tile[:, 1].astype(np.int32),
        "texture_map_tile": unique_tiles.astype(np.int32),
        "n_texture_maps": int(len(unique_tiles)),
    }


def build_texture_map_index_from_uv(triangle_uv_atlas, eps=1e-6, strict=True):
    triangle_uv_atlas = np.asarray(triangle_uv_atlas, dtype=np.float64)
    face_uv_tile, bad_faces = infer_face_uv_tiles(triangle_uv_atlas, eps=eps, strict=strict)
    tex_pack = build_texture_map_index_from_tiles(face_uv_tile)
    triangle_uv_local = localize_triangle_uv_by_face_tile(
        triangle_uv_atlas, face_uv_tile, eps=eps
    )
    tex_pack["triangle_uv_local"] = triangle_uv_local
    tex_pack["face_uv_tile"] = face_uv_tile
    tex_pack["bad_faces"] = bad_faces
    return tex_pack


def face_part_id_from_vertices(faces, vertex_parts):
    """Per-triangle topology part (vertex_parts on corners; first corner)."""
    parts = np.asarray(vertex_parts, dtype=np.int32)[np.asarray(faces, dtype=np.int32)]
    return parts[:, 0].astype(np.int32)


def build_geometry_chart_index(face_part_id):
    face_part_id = np.asarray(face_part_id, dtype=np.int32)
    unique_parts = np.unique(face_part_id)
    part_to_chart = {int(p): i for i, p in enumerate(unique_parts)}
    face_geometry_chart_id = np.array(
        [part_to_chart[int(p)] for p in face_part_id],
        dtype=np.int32,
    )
    return {
        "face_geometry_chart_id": face_geometry_chart_id,
        "geometry_chart_part": unique_parts.astype(np.int32),
        "n_geometry_charts": int(len(unique_parts)),
    }


def build_uv_seam_mesh(
    faces,
    vertices,
    triangle_uv_local,
    face_uv_tile,
    n_3d_verts,
    uv_round=8,
):
    """
    Seam mesh from face-local UV + face tile (not vertex-wise floor).

    Returns:
        new_vertices, new_uvs, new_faces, vmapping, uv_tile_index_vt, vertex_uvs, uv_tile_index_v
    """
    faces = np.asarray(faces, dtype=np.int32)
    triangle_uv_local = np.asarray(triangle_uv_local, dtype=np.float32)
    face_uv_tile = np.asarray(face_uv_tile, dtype=np.int32)
    vertices = np.asarray(vertices)

    new_vertices = []
    new_uvs = []
    new_faces = []
    vmapping = []
    uv_tile_index_vt = []
    vertex_uvs = np.zeros((n_3d_verts, 2), dtype=np.float32) - 1.0
    uv_tile_index_v = np.zeros((n_3d_verts, 2), dtype=np.int32) - 1

    key_to_new_vid = {}

    for fi, face in enumerate(faces):
        new_face = []
        tile = face_uv_tile[fi]

        for ci in range(3):
            vertex_idx = int(face[ci])
            local_uv = triangle_uv_local[fi, ci]
            key = (
                vertex_idx,
                int(tile[0]),
                int(tile[1]),
                round(float(local_uv[0]), uv_round),
                round(float(local_uv[1]), uv_round),
            )

            if key not in key_to_new_vid:
                new_id = len(new_vertices)
                key_to_new_vid[key] = new_id
                vmapping.append(vertex_idx)
                new_vertices.append(vertices[vertex_idx])
                new_uvs.append(local_uv)
                uv_tile_index_vt.append(tile)

                if 0 <= vertex_idx < n_3d_verts and vertex_uvs[vertex_idx, 0] < 0:
                    vertex_uvs[vertex_idx] = local_uv
                    uv_tile_index_v[vertex_idx] = tile

            new_face.append(key_to_new_vid[key])

        new_faces.append(new_face)

    return {
        "new_vertices": np.asarray(new_vertices, dtype=np.float32),
        "new_uvs": np.asarray(new_uvs, dtype=np.float32),
        "new_faces": np.asarray(new_faces, dtype=np.int32),
        "vmapping": np.asarray(vmapping, dtype=np.int32),
        "uv_tile_index_vt": np.asarray(uv_tile_index_vt, dtype=np.int32),
        "vertex_uvs": vertex_uvs,
        "uv_tile_index_v": uv_tile_index_v,
    }


def print_texture_map_statistics(mtl_pack, uv_pack=None, geom_pack=None):
    f_tid = mtl_pack["face_texture_map_id"]
    names = list(mtl_pack["material_names"])

    print("\n========== Texture map index (usemtl) ==========")
    print(f"  tri faces F={len(f_tid)}  materials K={mtl_pack['n_texture_maps']}")
    print(f"  primary (runtime): {list(mtl_pack['primary_texture_materials'])}")
    print(f"  catalog: {names}")

    for kid, name in enumerate(names):
        mask = f_tid == kid
        primary = " *" if name in PRIMARY_TEXTURE_MATERIALS else ""
        print(f"    tex[{kid}] {name}{primary}: F={mask.sum()}")

    if uv_pack is not None:
        bad = uv_pack.get("bad_faces", np.array([], dtype=np.int32))
        catalog = uv_pack["texture_map_tile"]
        if len(bad):
            print(f"  UV tile bad_faces: {len(bad)}")
        print(f"  UV atlas tiles (tu,tv): {[tuple(t) for t in catalog.tolist()]}")

    if geom_pack is not None:
        print(f"  geometry charts G={geom_pack['n_geometry_charts']}  parts={geom_pack['geometry_chart_part'].tolist()}")
        g_id = geom_pack["face_geometry_chart_id"]
        for gid in range(geom_pack["n_geometry_charts"]):
            part = int(geom_pack["geometry_chart_part"][gid])
            mask = g_id == gid
            print(f"    geom[{gid}] part={part}  F={mask.sum()}")
    print("========================================================\n")
