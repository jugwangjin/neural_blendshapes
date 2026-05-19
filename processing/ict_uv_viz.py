"""
UV layout visualization for ICT facekit npy bake.
"""

from pathlib import Path

import numpy as np

PART_NAMES = (
    "0_face_skin",
    "1_head_neck",
    "2_mouth_socket",
    "3_eye_socket_L",
    "4_eye_socket_R",
    "5_gums_tongue",
    "6_teeth",
    "7_eyeball_L",
    "8_eyeball_R",
)

PART_COLORS_BGR = np.array(
    [
        [200, 200, 200],
        [180, 140, 100],
        [100, 200, 255],
        [255, 180, 100],
        [100, 180, 255],
        [120, 255, 120],
        [80, 80, 220],
        [255, 100, 255],
        [255, 255, 100],
    ],
    dtype=np.uint8,
)

CHART_COLORS_BGR = [
    (220, 180, 120),
    (120, 200, 255),
    (120, 255, 180),
    (180, 120, 255),
    (255, 200, 100),
    (100, 255, 255),
    (255, 120, 120),
    (200, 255, 120),
    (120, 120, 255),
    (255, 180, 220),
    (180, 255, 255),
    (255, 255, 120),
    (200, 200, 200),
    (160, 160, 160),
    (100, 100, 100),
]


def _uv_stats_block(name, uv, mask=None):
    if mask is not None:
        uv = uv[mask]
    if uv.size == 0:
        return [f"  {name}: (empty)"]
    uv = np.asarray(uv, dtype=np.float64)
    tile_u = np.floor(uv[:, 0])
    tile_v = np.floor(uv[:, 1])
    frac_u = uv[:, 0] - tile_u
    frac_v = uv[:, 1] - tile_v
    lines = [
        f"  {name}: N={len(uv)}",
        f"    raw u: min={uv[:, 0].min():.6f} max={uv[:, 0].max():.6f} mean={uv[:, 0].mean():.6f}",
        f"    raw v: min={uv[:, 1].min():.6f} max={uv[:, 1].max():.6f} mean={uv[:, 1].mean():.6f}",
        f"    tile index u: unique={np.unique(tile_u).astype(int).tolist()}",
        f"    tile index v: unique={np.unique(tile_v).astype(int).tolist()}",
        f"    frac u (after floor): min={frac_u.min():.6f} max={frac_u.max():.6f}",
        f"    frac v (after floor): min={frac_v.min():.6f} max={frac_v.max():.6f}",
    ]
    in_unit = (frac_u >= -1e-4) & (frac_u <= 1.0 + 1e-4) & (frac_v >= -1e-4) & (frac_v <= 1.0 + 1e-4)
    lines.append(f"    in [0,1]^2 (frac): {in_unit.sum()}/{len(uv)} ({100.0 * in_unit.mean():.1f}%)")
    return lines


def _tile_index_stats(name, tile_uv):
    tile_uv = np.asarray(tile_uv, dtype=np.int64)
    if tile_uv.size == 0:
        return [f"  {name}: (empty)"]
    tu = np.unique(tile_uv[:, 0])
    tv = np.unique(tile_uv[:, 1])
    pairs = sorted({(int(t[0]), int(t[1])) for t in tile_uv})
    return [
        f"  {name}: N={len(tile_uv)}",
        f"    tile_u unique: {tu.astype(int).tolist()}",
        f"    tile_v unique: {tv.astype(int).tolist()}",
        f"    (tu,tv) pairs: {pairs}",
    ]


def print_uv_statistics(
    triangle_uv_atlas,
    triangle_uv_local,
    new_uvs,
    vertex_uvs,
    vertex_parts,
    vmapping,
    n_3d_verts,
    uv_quad_corners=None,
    uv_tile_index_vt=None,
    uv_tile_index_v=None,
    face_uv_tile=None,
):
    vp = np.asarray(vertex_parts, dtype=np.int64)
    vm = np.asarray(vmapping, dtype=np.int64)

    print("\n========== UV coordinate statistics ==========")
    if uv_quad_corners is not None:
        corners = np.asarray(uv_quad_corners).reshape(-1, 2)
        for line in _uv_stats_block("OBJ quad corners (all quads)", corners):
            print(line)

    tri_flat = np.asarray(triangle_uv_atlas).reshape(-1, 2)
    for line in _uv_stats_block("triangle_uv_atlas (pre-seam)", tri_flat):
        print(line)

    tri_loc = np.asarray(triangle_uv_local).reshape(-1, 2)
    for line in _uv_stats_block("triangle_uv_local (face tile)", tri_loc):
        print(line)

    for line in _uv_stats_block("new_uvs (seam VT, face-local)", new_uvs):
        print(line)

    if face_uv_tile is not None:
        for line in _tile_index_stats("face_uv_tile (per tri face)", face_uv_tile):
            print(line)

    if uv_tile_index_vt is not None:
        for line in _tile_index_stats("uv_tile_index_vt (atlas tile per seam VT)", uv_tile_index_vt):
            print(line)

    valid_v = vertex_uvs[:, 0] >= 0
    for line in _uv_stats_block(
        "uv_neutral_mesh (per 3D vertex, local UV only)", vertex_uvs, mask=valid_v
    ):
        print(line)

    if uv_tile_index_v is not None:
        for line in _tile_index_stats(
            "uv_tile_index_v (atlas tile per 3D vertex; one chart per vert)",
            uv_tile_index_v[valid_v],
        ):
            print(line)

    n_seam_extra = len(new_uvs) - n_3d_verts
    print(
        f"\n  Topology: V_3d={n_3d_verts}  VT_seam={len(new_uvs)}  "
        f"seam duplicates={n_seam_extra}  vmapping len={len(vm)}"
    )

    print("\n  Per vertex_part (3D verts, uv_neutral_mesh local):")
    for pid in sorted(np.unique(vp)):
        mask = vp == pid
        pname = PART_NAMES[pid] if pid < len(PART_NAMES) else f"part_{pid}"
        sub = vertex_uvs[mask]
        sub_valid = sub[sub[:, 0] >= 0]
        if sub_valid.size == 0:
            print(f"    {pname}: no valid UV")
            continue
        print(
            f"    {pname}: n={mask.sum()}  "
            f"u=[{sub_valid[:, 0].min():.4f},{sub_valid[:, 0].max():.4f}]  "
            f"v=[{sub_valid[:, 1].min():.4f},{sub_valid[:, 1].max():.4f}]"
        )

    print("\n  Per vertex_part (seam VT via vmapping, local):")
    for pid in sorted(np.unique(vp)):
        vt_mask = vp[vm] == pid
        pname = PART_NAMES[pid] if pid < len(PART_NAMES) else f"part_{pid}"
        sub = new_uvs[vt_mask]
        if sub.size == 0:
            print(f"    {pname}: no VT")
            continue
        print(
            f"    {pname}: VT={vt_mask.sum()}  "
            f"u=[{sub[:, 0].min():.4f},{sub[:, 0].max():.4f}]  "
            f"v=[{sub[:, 1].min():.4f},{sub[:, 1].max():.4f}]"
        )
    print("==============================================\n")


def uv_to_pixel(uv, size):
    u, v = float(uv[0]), float(uv[1])
    x = int(round(u * (size - 1)))
    y = int(round((1.0 - v) * (size - 1)))
    return np.clip(x, 0, size - 1), np.clip(y, 0, size - 1)


def uv_to_pixel_atlas(uv_atlas, size, margin=8):
    """Map atlas UV (with tile offset) into a horizontal strip canvas."""
    u, v = float(uv_atlas[0]), float(uv_atlas[1])
    tu = int(np.floor(u))
    tv = int(np.floor(v))
    fu = u - tu
    fv = v - tv
    x = margin + tu * (size + margin) + int(round(fu * (size - 1)))
    y = margin + tv * (size + margin) + int(round((1.0 - fv) * (size - 1)))
    return x, y


def bake_faces_local_uv(uvs_local, uv_faces, face_indices, size, color_bgr):
    import cv2

    img = np.zeros((size, size, 3), dtype=np.uint8)
    for fi in face_indices:
        tri = uvs_local[uv_faces[fi]]
        pts = np.array([uv_to_pixel(uv, size) for uv in tri], dtype=np.int32)
        cv2.fillConvexPoly(img, pts, color_bgr, lineType=cv2.LINE_AA)
        cv2.polylines(img, [pts], isClosed=True, color=(40, 40, 40), thickness=1, lineType=cv2.LINE_AA)
    return img


def bake_faces_atlas_uv(triangle_uv_atlas, face_indices, size, color_bgr, canvas_tiles_u=8):
    import cv2

    margin = 8
    canvas_w = margin + canvas_tiles_u * (size + margin)
    canvas_h = margin + 4 * (size + margin)
    img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for fi in face_indices:
        tri = triangle_uv_atlas[fi]
        pts = []
        for uv in tri:
            u, v = float(uv[0]), float(uv[1])
            tu = int(np.floor(u))
            tv = int(np.floor(v))
            fu = u - tu
            fv = v - tv
            x = margin + tu * (size + margin) + int(round(fu * (size - 1)))
            y = margin + tv * (size + margin) + int(round((1.0 - fv) * (size - 1)))
            pts.append([x, y])
        pts = np.array(pts, dtype=np.int32)
        cv2.fillConvexPoly(img, pts, color_bgr, lineType=cv2.LINE_AA)
    return img


def _safe_chart_label(name):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name))


def export_texture_map_charts(
    out_dir,
    uvs_local,
    uv_faces,
    face_texture_map_id,
    triangle_uv_atlas,
    texture_size=512,
    material_names=None,
    texture_map_tile=None,
):
    """
    One PNG per face_texture_map_id.
    Labels from material_names (usemtl) if given, else UV tile.
    """
    import cv2

    out_dir = Path(out_dir)
    charts_dir = out_dir / "texture_charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    face_texture_map_id = np.asarray(face_texture_map_id, dtype=np.int32)
    n_charts = int(face_texture_map_id.max()) + 1 if face_texture_map_id.size else 0
    if material_names is not None:
        chart_labels = [_safe_chart_label(m) for m in material_names]
        n_charts = max(n_charts, len(chart_labels))
    elif texture_map_tile is not None:
        texture_map_tile = np.asarray(texture_map_tile, dtype=np.int32)
        chart_labels = [
            f"tile{int(texture_map_tile[i, 0])}_{int(texture_map_tile[i, 1])}"
            for i in range(len(texture_map_tile))
        ]
        n_charts = max(n_charts, len(chart_labels))
    else:
        chart_labels = [f"chart{i}" for i in range(n_charts)]

    for tid in range(n_charts):
        mask = face_texture_map_id == tid
        fi = np.where(mask)[0]
        if fi.size == 0:
            continue
        label = chart_labels[tid] if tid < len(chart_labels) else f"chart{tid}"
        color = CHART_COLORS_BGR[tid % len(CHART_COLORS_BGR)]
        local_img = bake_faces_local_uv(uvs_local, uv_faces, fi, texture_size, color)
        cv2.imwrite(str(charts_dir / f"chart_{tid:02d}_{label}_local.png"), local_img)

        atlas_img = bake_faces_atlas_uv(triangle_uv_atlas, fi, texture_size, color)
        cv2.imwrite(str(charts_dir / f"chart_{tid:02d}_{label}_atlas.png"), atlas_img)

        print(f"  chart [{tid}] {label}: F={fi.size}  -> chart_{tid:02d}_{label}_local.png")

    overview = np.zeros((texture_size, texture_size * max(n_charts, 1), 3), dtype=np.uint8)
    for tid in range(n_charts):
        mask = face_texture_map_id == tid
        fi = np.where(mask)[0]
        if fi.size == 0:
            continue
        color = CHART_COLORS_BGR[tid % len(CHART_COLORS_BGR)]
        patch = bake_faces_local_uv(uvs_local, uv_faces, fi, texture_size, color)
        overview[:, tid * texture_size : (tid + 1) * texture_size] = patch

    cv2.imwrite(str(out_dir / "texture_charts_overview_local.png"), overview)
    print(f"Texture charts written to {charts_dir}/")


def bake_part_id_texture(vertices_3d, faces_3d, uvs, uv_faces, vertex_parts, size=2048):
    import cv2

    img = np.zeros((size, size, 3), dtype=np.uint8)
    vertex_parts = list(vertex_parts)

    for fi in range(faces_3d.shape[0]):
        tri_v = faces_3d[fi]
        tri_uv = uvs[uv_faces[fi]]
        part = int(vertex_parts[tri_v[0]])
        color = PART_COLORS_BGR[part % len(PART_COLORS_BGR)]
        pts = np.array([uv_to_pixel(uv, size) for uv in tri_uv], dtype=np.int32)
        cv2.fillConvexPoly(img, pts, color.tolist(), lineType=cv2.LINE_AA)

    return img


def bake_uv_wireframe(uvs, uv_faces, size=2048, color=(80, 80, 80)):
    import cv2

    img = np.zeros((size, size, 3), dtype=np.uint8)
    for ufi in uv_faces:
        tri = uvs[ufi]
        pts = np.array([uv_to_pixel(uv, size) for uv in tri], dtype=np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)
    return img


def export_seam_mesh_obj(path, positions_seam, uvs, uv_faces):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in positions_seam:
            f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        for uv in uvs:
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        for ufi in uv_faces:
            f.write(
                f"f {ufi[0] + 1}/{ufi[0] + 1} "
                f"{ufi[1] + 1}/{ufi[1] + 1} "
                f"{ufi[2] + 1}/{ufi[2] + 1}\n"
            )
    return path


def export_uv_debug(
    out_dir,
    vertices_3d,
    faces_3d,
    uvs,
    uv_faces,
    vertex_uvs_per_3d,
    vertex_parts,
    vmapping,
    face_texture_map_id=None,
    material_names=None,
    texture_map_tile=None,
    triangle_uv_atlas=None,
    texture_size=2048,
):
    import cv2

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    positions_seam = np.asarray(vertices_3d)[np.asarray(vmapping)]
    export_seam_mesh_obj(out_dir / "ict_uv_seam.obj", positions_seam, uvs, uv_faces)

    part_tex = bake_part_id_texture(
        vertices_3d, faces_3d, uvs, uv_faces, vertex_parts, size=texture_size
    )
    cv2.imwrite(str(out_dir / "ict_part_atlas.png"), part_tex)

    wire = bake_uv_wireframe(uvs, uv_faces, size=texture_size)
    cv2.imwrite(str(out_dir / "ict_uv_wireframe.png"), wire)

    if face_texture_map_id is not None and triangle_uv_atlas is not None:
        export_texture_map_charts(
            out_dir,
            uvs,
            uv_faces,
            face_texture_map_id,
            triangle_uv_atlas,
            texture_size=texture_size,
            material_names=material_names,
            texture_map_tile=texture_map_tile,
        )

    scatter = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
    for vid, uv in enumerate(vertex_uvs_per_3d):
        if uv[0] < 0:
            continue
        px = uv_to_pixel(uv, texture_size)
        part = int(vertex_parts[vid])
        scatter[px[1], px[0]] = PART_COLORS_BGR[part % len(PART_COLORS_BGR)]
    cv2.imwrite(str(out_dir / "ict_per_vertex_uv.png"), scatter)

    np.savez(
        out_dir / "uv_indices.npz",
        uvs=uvs.astype(np.float32),
        uv_faces=uv_faces.astype(np.int64),
        uv_neutral_mesh=vertex_uvs_per_3d.astype(np.float32),
        vmapping=np.asarray(vmapping, dtype=np.int64),
        vertex_parts=np.asarray(vertex_parts, dtype=np.int64),
        faces_3d=faces_3d.astype(np.int64),
        face_texture_map_id=(
            np.asarray(face_texture_map_id, dtype=np.int32)
            if face_texture_map_id is not None
            else np.array([], dtype=np.int32)
        ),
        material_names=(
            np.asarray(material_names, dtype=object)
            if material_names is not None
            else np.array([], dtype=object)
        ),
        texture_map_tile=(
            np.asarray(texture_map_tile, dtype=np.int32)
            if texture_map_tile is not None
            else np.array([], dtype=np.int32)
        ),
    )

    print(f"UV debug exported to {out_dir}")
    return out_dir
