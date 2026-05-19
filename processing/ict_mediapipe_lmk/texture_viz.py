"""Bake MediaPipe landmarks and contours into UV textures for visual QA."""

from pathlib import Path

import cv2
import numpy as np
import trimesh

from processing.ict_mediapipe_lmk.constants import (
    LEFT_IRIS_FLAME,
    LEFT_IRIS_MP,
    RIGHT_IRIS_FLAME,
    RIGHT_IRIS_MP,
)
from processing.ict_mediapipe_lmk.landmark_routing import CHART_LEFT_EYE, CHART_RIGHT_EYE
from processing.ict_mediapipe_lmk.mediapipe_connections import connections_for_landmarks
from processing.ict_mediapipe_lmk.metrical import load_metrical_mediapipe_embedding
from processing.ict_obj_materials import normalize_material_name

# Omit from combined UV QA (separate per-chart exports for eyes).
VIZ_SKIP_MATERIALS = frozenset(
    {
        "M_Teeth",
        "M_GumsTongue",
        "M_IrisLeft",
        "M_IrisRight",
        "M_EyeballLeft",
        "M_EyeballRight",
        "M_LacrimalFluid",
        "M_EyeOcclusion",
        "M_EyeBlend",
        "M_EyeLashes",
    }
)

# Face skin chart; eye iris QA uses dedicated eyeball exports (see export_eyeball_iris5_texture).
VIZ_PER_CHART_MATERIALS = ("M_Face",)


def parse_obj_uv(path):
    """Load vt and per-face UV indices from OBJ (``f v/vt``)."""
    path = Path(path)
    uvs = []
    uv_faces = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("vt "):
                parts = line.split()
                uvs.append([float(parts[1]), float(parts[2])])
            elif line.startswith("f "):
                corners = []
                for tok in line.split()[1:]:
                    slash = tok.split("/")
                    vt_i = int(slash[1]) - 1 if len(slash) > 1 and slash[1] else -1
                    corners.append(vt_i)
                if any(c < 0 for c in corners):
                    raise ValueError(f"Face without vt in {path}: {line.strip()}")
                uv_faces.append(corners)
    if not uvs or not uv_faces:
        raise ValueError(f"No vt/uv-faces in {path}")
    return np.asarray(uvs, dtype=np.float64), np.asarray(uv_faces, dtype=np.int64)


def load_mesh_uv_trimesh(path):
    """Load OBJ with vt; returns uvs [Vt,2], uv_faces [F,3]."""
    mesh = trimesh.load(str(path), process=False, maintain_order=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    faces = np.asarray(mesh.faces, dtype=np.int64)
    visual = mesh.visual
    if hasattr(visual, "uv") and visual.uv is not None and len(visual.uv) > 0:
        uvs = np.asarray(visual.uv, dtype=np.float64)
        if uvs.shape[0] == len(mesh.vertices):
            return uvs, faces.copy()
    raise ValueError(f"No per-vertex UV in {path}")


def load_mesh_uv(path):
    """Load UV atlas; tries trimesh per-vertex UV, OBJ vt parse, pytorch3d."""
    path = Path(path)
    for loader in (load_mesh_uv_trimesh, parse_obj_uv, _load_flame_uv_pytorch3d):
        try:
            return loader(path)
        except Exception:
            continue
    raise ValueError(f"Could not load UV from {path}")


def load_flame_uv_mesh(path, n_faces_expected=None):
    path = Path(path)
    uvs, uv_faces = load_mesh_uv(path)
    if n_faces_expected is not None and len(uv_faces) != n_faces_expected:
        raise ValueError(
            f"UV face count {len(uv_faces)} != expected F={n_faces_expected} ({path})"
        )
    return uvs, uv_faces


def resolve_flame_uv_for_topology(
    f_flame,
    flame_model_path=None,
    flare_topo_path=None,
    flame_uv_mesh=None,
):
    from processing.flame.flame_viz import load_flame_uv_for_viz

    return load_flame_uv_for_viz(
        f_flame,
        flame_model_path=flame_model_path,
        flare_topo_path=flare_topo_path,
        flare_uv_obj=flame_uv_mesh,
    )


def _load_flame_uv_pytorch3d(path):
    from pytorch3d.io import load_obj

    _, faces, aux = load_obj(str(path), load_textures=False)
    if aux.verts_uvs is None or aux.faces_uvs is None:
        raise ValueError("pytorch3d load_obj: verts_uvs is None")
    uvs = aux.verts_uvs.cpu().numpy()
    uv_faces = aux.faces_uvs.cpu().numpy()
    return uvs, uv_faces


def resolve_flame_uv_mesh_path(user_path, metrical_root=None):
    """Return ``(path_or_none, tried_paths)`` for a raw UV OBJ (no topology remap)."""
    tried = []
    paths = []
    if user_path:
        paths.append(Path(user_path))
    paths.extend(flame_uv_mesh_candidates(metrical_root))
    for p in paths:
        if not p.is_file():
            continue
        tried.append(str(p))
        try:
            load_mesh_uv(p)
            return p, tried
        except Exception:
            continue
    return None, tried


def landmark_uv_from_bary(face_idx, bary, faces, uv_faces, uvs):
    tri = faces[face_idx]
    tri_uv_idx = uv_faces[face_idx]
    tri_uv = uvs[tri_uv_idx]
    return (tri_uv * bary[:, :, None]).sum(axis=1)


def landmark_uv_from_embedding(embedding, f_ict, uv_faces, uvs, ict_npy_dict=None):
    """Per-face ``triangle_uv_local`` when available (texture-map 0–1), else seam VT bary."""
    face_idx = np.asarray(embedding["ict_lmk_face_idx"], dtype=np.int64)
    bary = np.asarray(embedding["ict_lmk_b_coords"], dtype=np.float64)
    if ict_npy_dict is not None and "triangle_uv_local" in ict_npy_dict:
        tuv = np.asarray(ict_npy_dict["triangle_uv_local"], dtype=np.float64)
        tri = tuv[face_idx]
        return (tri * bary[:, :, None]).sum(axis=1)
    return landmark_uv_from_bary(face_idx, bary, f_ict, uv_faces, uvs)


def bake_triangle_uv_local_chart(triangle_uv_local, face_indices, size, fill_bgr=(40, 44, 52)):
    """Rasterize ``triangle_uv_local[fi]`` (chart-local 0–1) for one texture map."""
    triangle_uv_local = np.asarray(triangle_uv_local, dtype=np.float64)
    face_indices = np.asarray(face_indices, dtype=np.int64)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for fi in face_indices:
        tri_uv = triangle_uv_local[fi]
        pts = np.array([uv_to_pixel(uv, size) for uv in tri_uv], dtype=np.int32)
        cv2.fillConvexPoly(img, pts, fill_bgr, lineType=cv2.LINE_AA)
        cv2.polylines(img, [pts], isClosed=True, color=(90, 90, 90), thickness=1, lineType=cv2.LINE_AA)
    return img


def eyeball_sclera_face_indices(ict_npy_dict, side, f_ict, n_verts):
    """``M_Sclera*`` triangles on the eyeball — same faces iris bake projects onto."""
    from utils.eye_chart import sclera_eyeball_face_mask

    key = "left_eyeball_indices" if side == "left" else "right_eyeball_indices"
    eye_ids = ict_npy_dict[key]
    ch = "L" if side == "left" else "R"
    mask = sclera_eyeball_face_mask(ict_npy_dict, ch, f_ict, eye_ids, n_verts)
    return np.where(mask)[0].astype(np.int64)


def landmark_uv_from_vertex_corners(vertex_idx, faces, uv_faces, uvs):
    out = []
    for vid in np.asarray(vertex_idx, dtype=np.int64):
        fi = int(np.where(np.any(faces == vid, axis=1))[0][0])
        corner = int(np.where(faces[fi] == vid)[0][0])
        out.append(uvs[uv_faces[fi, corner]])
    return np.stack(out, axis=0)


def uv_to_pixel(uv, size):
    u, v = float(uv[0]), float(uv[1])
    x = int(round(u * (size - 1)))
    y = int(round((1.0 - v) * (size - 1)))
    return x, y


def material_names_from_npy(ict_npy_dict):
    if ict_npy_dict is None:
        return []
    if "material_names" in ict_npy_dict:
        return [normalize_material_name(m) for m in ict_npy_dict["material_names"]]
    if "face_material_name" in ict_npy_dict:
        uniq = sorted({normalize_material_name(m) for m in ict_npy_dict["face_material_name"]})
        return uniq
    return []


def face_indices_for_material(ict_npy_dict, material_name):
    names = material_names_from_npy(ict_npy_dict)
    mat = normalize_material_name(material_name)
    if mat not in names:
        return np.array([], dtype=np.int64)
    tid = names.index(mat)
    ftmi = np.asarray(ict_npy_dict["face_texture_map_id"], dtype=np.int64)
    return np.where(ftmi == tid)[0].astype(np.int64)


def face_indices_excluding_materials(ict_npy_dict, skip_materials):
    names = material_names_from_npy(ict_npy_dict)
    if not names:
        return np.arange(len(ict_npy_dict["face_texture_map_id"]), dtype=np.int64)
    ftmi = np.asarray(ict_npy_dict["face_texture_map_id"], dtype=np.int64)
    skip = {normalize_material_name(m) for m in skip_materials}
    keep = [fi for fi in range(len(ftmi)) if names[int(ftmi[fi])] not in skip]
    return np.asarray(keep, dtype=np.int64)


def landmark_mask_on_faces(embedding, face_indices):
    fi_set = set(np.asarray(face_indices, dtype=np.int64).tolist())
    fi = np.asarray(embedding["ict_lmk_face_idx"], dtype=np.int64)
    return np.array([int(f) in fi_set for f in fi], dtype=bool)


def landmark_mask_for_eye_chart(embedding, side, ict_npy_dict):
    """Landmarks for M_Sclera* chart (iris + eyelid), not M_Iris* annulus."""
    types = embedding.get("ict_lmk_target_type")
    if types is not None:
        types = np.asarray(types, dtype=object)
        if side == "L":
            return np.isin(types, ["left_iris", "left_eyelid"])
        return np.isin(types, ["right_iris", "right_eyelid"])
    geo = embedding.get("geometry_chart_id")
    if geo is not None:
        geo = np.asarray(geo, dtype=np.int64)
        want = CHART_LEFT_EYE if side == "L" else CHART_RIGHT_EYE
        return geo == want
    mat = "M_ScleraLeft" if side == "L" else "M_ScleraRight"
    return landmark_mask_on_faces(embedding, face_indices_for_material(ict_npy_dict, mat))


def rasterize_uv_layout_map(uvs, uv_faces, size, fill_bgr=(36, 36, 36), wire_bgr=(100, 100, 100)):
    """UV texture-map layout (triangle fill + wireframe). No photo albedo."""
    from processing.ict_uv_viz import bake_uv_wireframe

    uvs = np.asarray(uvs, dtype=np.float64)
    uv_faces = np.asarray(uv_faces, dtype=np.int64)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for fi in range(len(uv_faces)):
        tri_uv = uvs[uv_faces[fi]]
        pts = np.array([uv_to_pixel(uv, size) for uv in tri_uv], dtype=np.int32)
        cv2.fillConvexPoly(img, pts, fill_bgr, lineType=cv2.LINE_AA)
    wire = bake_uv_wireframe(uvs, uv_faces, size=size, color=wire_bgr)
    mask = wire.any(axis=2)
    img[mask] = wire[mask]
    return img


def ict_uv_layout_texture(
    ict_npy_dict,
    uvs,
    uv_faces,
    faces,
    size=2048,
    *,
    use_atlas_chart=False,
    face_indices=None,
    exclude_materials=None,
):
    """
    UV background for landmark QA textures.

    Default: seam ``uvs`` / ``uv_faces`` wireframe (material-local 0–1), not per-chart atlas tiles.
    ``face_indices`` / ``exclude_materials``: limit rasterized triangles (e.g. skip teeth).
  """
    if use_atlas_chart and ict_npy_dict is not None and "triangle_uv_atlas" in ict_npy_dict:
        from processing.ict_uv_viz import bake_faces_atlas_uv

        atlas = np.asarray(ict_npy_dict["triangle_uv_atlas"], dtype=np.float64)
        fi = face_indices if face_indices is not None else np.arange(len(atlas))
        return bake_faces_atlas_uv(atlas, fi, size, (50, 55, 60))
    uvs = np.asarray(uvs, dtype=np.float64)
    uv_faces = np.asarray(uv_faces, dtype=np.int64)
    if face_indices is None and exclude_materials and ict_npy_dict is not None:
        face_indices = face_indices_excluding_materials(ict_npy_dict, exclude_materials)
    if face_indices is not None:
        from processing.ict_uv_viz import bake_faces_local_uv

        face_indices = np.asarray(face_indices, dtype=np.int64)
        return bake_faces_local_uv(uvs, uv_faces, face_indices, size, (36, 36, 36))
    return rasterize_uv_layout_map(uvs, uv_faces, size)


def flame_uv_layout_texture(uvs, uv_faces, size=2048):
    """FLAME UV chart layout (wireframe on gray)."""
    return rasterize_uv_layout_map(uvs, uv_faces, size, fill_bgr=(45, 50, 55), wire_bgr=(110, 110, 110))


def bake_mediapipe_texture(
    landmark_uv,
    mp_indices,
    size=2048,
    base_bgr=None,
    atlas_tiled=False,
    line_color=(0, 255, 0),
    point_color=(255, 0, 0),
    iris_color=(0, 200, 255),
    draw_indices=True,
):
    if atlas_tiled:
        from processing.ict_uv_viz import uv_to_pixel_atlas

        def to_px(uv):
            return uv_to_pixel_atlas(uv, size)
    else:

        def to_px(uv):
            return uv_to_pixel(uv, size)

    if base_bgr is None:
        img = np.zeros((size, size, 3), dtype=np.uint8)
    else:
        img = np.asarray(base_bgr, dtype=np.uint8).copy()
        if not atlas_tiled and (img.shape[0] != size or img.shape[1] != size):
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    mp_to_row = {int(mp): i for i, mp in enumerate(mp_indices)}

    for a, b in connections_for_landmarks(mp_indices):
        pa = to_px(landmark_uv[mp_to_row[a]])
        pb = to_px(landmark_uv[mp_to_row[b]])
        cv2.line(img, pa, pb, line_color, 2, cv2.LINE_AA)

    for mp_id, uv in zip(mp_indices, landmark_uv):
        px = to_px(uv)
        color = iris_color if mp_id >= 468 else point_color
        cv2.circle(img, px, 4, color, -1, cv2.LINE_AA)
        if draw_indices:
            cv2.putText(
                img,
                str(int(mp_id)),
                (px[0] + 3, px[1] - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.28,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return img


def export_textured_mesh(path, vertices, faces, uvs, uv_faces, texture_bgr):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tex_path = path.with_suffix(".png")
    cv2.imwrite(str(tex_path), texture_bgr)
    mtl_path = path.with_suffix(".mtl")

    with open(mtl_path, "w", encoding="utf-8") as mf:
        mf.write("newmtl mediapipe\n")
        mf.write("Kd 1.0 1.0 1.0\n")
        mf.write(f"map_Kd {tex_path.name}\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"mtllib {mtl_path.name}\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for uv in uvs:
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        f.write("usemtl mediapipe\n")
        for face, ufi in zip(faces, uv_faces):
            f.write(
                f"f {face[0] + 1}/{ufi[0] + 1} "
                f"{face[1] + 1}/{ufi[1] + 1} "
                f"{face[2] + 1}/{ufi[2] + 1}\n"
            )
    return tex_path


def flame_mediapipe_landmark_uv(v_flame, f_flame, uv_faces, uvs, mp_embedding_path):
    mp_ids, flame_face_idx, flame_bary = load_metrical_mediapipe_embedding(mp_embedding_path)
    skin_uv = landmark_uv_from_bary(flame_face_idx, flame_bary, f_flame, uv_faces, uvs)

    iris_mp = np.concatenate([LEFT_IRIS_MP, RIGHT_IRIS_MP])
    iris_verts = np.concatenate([LEFT_IRIS_FLAME, RIGHT_IRIS_FLAME])
    iris_uv = landmark_uv_from_vertex_corners(iris_verts, f_flame, uv_faces, uvs)

    all_mp = np.concatenate([mp_ids, iris_mp])
    all_uv = np.concatenate([skin_uv, iris_uv], axis=0)
    return all_mp, all_uv


def ict_mediapipe_landmark_uv(embedding, f_ict, uv_faces, uvs, triangle_uv_atlas=None, ict_npy_dict=None):
    face_idx = embedding["ict_lmk_face_idx"]
    bary = embedding["ict_lmk_b_coords"]
    if triangle_uv_atlas is not None:
        tri = np.asarray(triangle_uv_atlas, dtype=np.float64)[face_idx]
        lmk_uv = (tri * bary[:, :, None]).sum(axis=1)
        return embedding["mp_landmark_indices"], lmk_uv
    if ict_npy_dict is not None and "triangle_uv_local" in ict_npy_dict:
        lmk_uv = landmark_uv_from_embedding(embedding, f_ict, uv_faces, uvs, ict_npy_dict)
        return embedding["mp_landmark_indices"], lmk_uv
    return embedding["mp_landmark_indices"], landmark_uv_from_bary(
        face_idx, bary, f_ict, uv_faces, uvs
    )


def bake_iris_pentagon_texture_zoomed(landmark_uv, mp_indices, size, base_bgr, margin_frac=0.35):
    """
    Magnified iris QA: map a UV window around the pentagon centroid onto the full image.

    On ``M_Sclera*``, iris landmarks sit near chart center (0.5, 0.5); full-disk PNGs look
    like a single point — this export makes the pentagon visible.
    """
    landmark_uv = np.asarray(landmark_uv, dtype=np.float64)
    mp_indices = [int(m) for m in mp_indices]
    if len(mp_indices) != len(landmark_uv):
        raise ValueError(f"iris mp/uv length mismatch: {len(mp_indices)} vs {len(landmark_uv)}")

    cx, cy = landmark_uv.mean(axis=0)
    half = float(max(landmark_uv.ptp(axis=0).max() * 0.55 + 0.012, 0.025))
    half *= 1.0 + margin_frac

    def uv_to_zoom_px(uv):
        u, v = float(uv[0]), float(uv[1])
        zu = (u - cx) / (2.0 * half) + 0.5
        zv = (v - cy) / (2.0 * half) + 0.5
        return uv_to_pixel(np.array([zu, zv], dtype=np.float64), size)

    img = np.asarray(base_bgr, dtype=np.uint8).copy()
    if img.shape[0] != size or img.shape[1] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)

    y0 = int(round((cy - half) * (size - 1)))
    y1 = int(round((cy + half) * (size - 1)))
    x0 = int(round((cx - half) * (size - 1)))
    x1 = int(round((cx + half) * (size - 1)))
    y0, y1 = max(0, y0), min(size - 1, y1)
    x0, x1 = max(0, x0), min(size - 1, x1)
    if y1 > y0 and x1 > x0:
        crop = img[y0 : y1 + 1, x0 : x1 + 1]
        img = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)

    pts = [uv_to_zoom_px(uv) for uv in landmark_uv]
    for j in range(len(pts)):
        cv2.line(img, pts[j], pts[(j + 1) % len(pts)], (0, 255, 0), 3, cv2.LINE_AA)
    for mp_id, uv in zip(mp_indices, landmark_uv):
        px = uv_to_zoom_px(uv)
        cv2.circle(img, px, 18, (0, 200, 255), -1, cv2.LINE_AA)
        cv2.circle(img, px, 18, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(
            img,
            str(mp_id),
            (px[0] + 8, px[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        img,
        f"zoom half={half:.4f} center=({cx:.3f},{cy:.3f})",
        (12, size - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return img


def bake_iris_pentagon_texture(landmark_uv, mp_indices, size, base_bgr):
    """Draw exactly 5 iris MP indices on a chart-local texture (pentagon + labels)."""
    img = np.asarray(base_bgr, dtype=np.uint8).copy()
    if img.shape[0] != size or img.shape[1] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    mp_indices = [int(m) for m in mp_indices]
    landmark_uv = np.asarray(landmark_uv, dtype=np.float64)
    if len(mp_indices) != len(landmark_uv):
        raise ValueError(f"iris mp/uv length mismatch: {len(mp_indices)} vs {len(landmark_uv)}")
    pts = [uv_to_pixel(uv, size) for uv in landmark_uv]
    for j in range(len(pts)):
        cv2.line(img, pts[j], pts[(j + 1) % len(pts)], (0, 255, 0), 2, cv2.LINE_AA)
    for mp_id, px_uv in zip(mp_indices, landmark_uv):
        px = uv_to_pixel(px_uv, size)
        cv2.circle(img, px, 14, (0, 200, 255), -1, cv2.LINE_AA)
        cv2.circle(img, px, 14, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(
            img,
            str(mp_id),
            (px[0] + 6, px[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return img


def export_eyeball_iris5_texture(
    debug_dir,
    v_ict,
    f_ict,
    uvs,
    uv_faces,
    embedding,
    ict_npy_dict,
    side,
    *,
    size=2048,
):
    """
    Eyeball texture map QA: **only** MP iris 468–472 (L) or 473–477 (R) on ``triangle_uv_local``.

    Background = ``M_Sclera*`` ∩ eyeball tris (same as bake / EyeTextureGaussians chart).
    """
    debug_dir = Path(debug_dir)
    side = str(side).lower()
    if side not in ("left", "right"):
        raise ValueError(f"side must be left|right, got {side!r}")
    mp_order = np.array(LEFT_IRIS_MP if side == "left" else RIGHT_IRIS_MP, dtype=np.int64)
    if "triangle_uv_local" not in ict_npy_dict:
        raise ValueError("ict_npy_dict missing triangle_uv_local — re-run ict_facekit_to_npy_full_head.py")

    tuv = np.asarray(ict_npy_dict["triangle_uv_local"], dtype=np.float64)
    fi_eye = eyeball_sclera_face_indices(ict_npy_dict, side, f_ict, len(v_ict))
    if fi_eye.size == 0:
        print(f"  WARNING: no sclera∩eyeball faces for {side} eye — skip iris texture")
        return None

    base = bake_triangle_uv_local_chart(tuv, fi_eye, size, fill_bgr=(35, 40, 48))
    mp_all, lmk_uv_all = ict_mediapipe_landmark_uv(
        embedding, f_ict, uv_faces, uvs, ict_npy_dict=ict_npy_dict
    )
    mp_all = np.asarray(mp_all, dtype=np.int64)
    rows = []
    for mp in mp_order:
        hit = np.where(mp_all == mp)[0]
        if hit.size == 0:
            print(f"  WARNING: missing iris MP {mp} in embedding for {side} eye")
            continue
        rows.append(int(hit[0]))
    if len(rows) != 5:
        print(f"  WARNING: {side} eye iris: expected 5 landmarks, got {len(rows)}")
    mp_ids = mp_all[rows]
    lmk_uv = lmk_uv_all[rows]
    tex = bake_iris_pentagon_texture(lmk_uv, mp_order[: len(rows)], size, base)
    tex_zoom = bake_iris_pentagon_texture_zoomed(lmk_uv, mp_order[: len(rows)], size, base)

    label = f"eyeball_{side}_iris5"
    out = debug_dir / f"ict_{label}_textured.obj"
    tex_path = export_textured_mesh(out, v_ict, f_ict, uvs, uv_faces, tex)
    png_path = debug_dir / f"ict_{label}_texture.png"
    png_zoom_path = debug_dir / f"ict_{label}_texture_zoom.png"
    cv2.imwrite(str(png_path), tex)
    cv2.imwrite(str(png_zoom_path), tex_zoom)
    u_rng = lmk_uv[:, 0].min(), lmk_uv[:, 0].max()
    v_rng = lmk_uv[:, 1].min(), lmk_uv[:, 1].max()
    print(
        f"  eyeball [{side}] iris×5: {png_path.name} + {png_zoom_path.name}  "
        f"faces={fi_eye.size}  uv_u=[{u_rng[0]:.3f},{u_rng[1]:.3f}] "
        f"uv_v=[{v_rng[0]:.3f},{v_rng[1]:.3f}]"
    )
    return {
        "side": side,
        "obj": out,
        "texture": png_path,
        "texture_zoom": png_zoom_path,
        "mp_indices": mp_ids,
        "landmark_uv": lmk_uv,
        "n_eyeball_faces": int(fi_eye.size),
    }


def export_flame_mediapipe_texture(
    debug_dir,
    v_flame,
    f_flame,
    mp_embedding_path,
    flame_uv_mesh=None,
    flame_model_path=None,
    size=2048,
    basename="flame_mediapipe",
):
    uvs, uv_faces, uv_src = resolve_flame_uv_for_topology(
        f_flame,
        flame_model_path=flame_model_path,
        flame_uv_mesh=flame_uv_mesh,
    )
    print(f"FLAME UV (processing/flame): {uv_src} (F={len(uv_faces)})")
    mp_ids, landmark_uv = flame_mediapipe_landmark_uv(
        v_flame, f_flame, uv_faces, uvs, mp_embedding_path
    )
    base = flame_uv_layout_texture(uvs, uv_faces, size=size)
    tex = bake_mediapipe_texture(landmark_uv, mp_ids, size=size, base_bgr=base)
    out = Path(debug_dir) / f"{basename}_textured.obj"
    tex_path = export_textured_mesh(out, v_flame, f_flame, uvs, uv_faces, tex)
    cv2.imwrite(str(Path(debug_dir) / f"{basename}_texture.png"), tex)
    return out, tex_path


def _safe_chart_basename(material_name):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in normalize_material_name(material_name))


def export_ict_mediapipe_texture_chart(
    debug_dir,
    v_ict,
    f_ict,
    uvs,
    uv_faces,
    embedding,
    ict_npy_dict,
    material_name,
    landmark_mask,
    *,
    size=2048,
):
    """Single texture-map PNG/OBJ (local UV 0–1) with landmarks whose faces lie on that map."""
    debug_dir = Path(debug_dir)
    fi = face_indices_for_material(ict_npy_dict, material_name)
    if fi.size == 0:
        return None
    if "triangle_uv_local" in ict_npy_dict:
        tuv = np.asarray(ict_npy_dict["triangle_uv_local"], dtype=np.float64)
        base = bake_triangle_uv_local_chart(tuv, fi, size)
    else:
        base = ict_uv_layout_texture(
            ict_npy_dict, uvs, uv_faces, f_ict, size=size, face_indices=fi
        )
    mp_ids_all, landmark_uv_all = ict_mediapipe_landmark_uv(
        embedding, f_ict, uv_faces, uvs, ict_npy_dict=ict_npy_dict
    )
    lmk_mask = np.asarray(landmark_mask, dtype=bool)
    if lmk_mask.shape[0] != len(mp_ids_all):
        lmk_mask = landmark_mask_on_faces(embedding, fi)
    mp_ids = mp_ids_all[lmk_mask]
    landmark_uv = landmark_uv_all[lmk_mask]
    if len(mp_ids) == 0:
        tex = base
    else:
        tex = bake_mediapipe_texture(
            landmark_uv, mp_ids, size=size, base_bgr=base, atlas_tiled=False
        )
    label = _safe_chart_basename(material_name)
    out = debug_dir / f"ict_mediapipe_{label}_textured.obj"
    tex_path = export_textured_mesh(out, v_ict, f_ict, uvs, uv_faces, tex)
    png_path = debug_dir / f"ict_mediapipe_{label}_texture.png"
    cv2.imwrite(str(png_path), tex)
    return {"material": material_name, "obj": out, "texture": png_path, "n_landmarks": int(len(mp_ids))}


def export_ict_mediapipe_per_texture_map(
    debug_dir,
    v_ict,
    f_ict,
    uvs,
    uv_faces,
    embedding,
    ict_npy_dict,
    *,
    size=2048,
):
    """
    Per texture-map QA: ``M_Face`` + eyeball iris×5 (``triangle_uv_local`` on sclera∩eyeball).
    """
    debug_dir = Path(debug_dir) / "texture_maps"
    debug_dir.mkdir(parents=True, exist_ok=True)
    charts = {}

    for mat in VIZ_PER_CHART_MATERIALS:
        if mat not in material_names_from_npy(ict_npy_dict):
            continue
        lmk_mask = landmark_mask_on_faces(embedding, face_indices_for_material(ict_npy_dict, mat))
        out = export_ict_mediapipe_texture_chart(
            debug_dir,
            v_ict,
            f_ict,
            uvs,
            uv_faces,
            embedding,
            ict_npy_dict,
            mat,
            lmk_mask,
            size=size,
        )
        if out is not None:
            charts[mat] = out
            print(
                f"  texture map [{mat}]: {out['texture'].name} "
                f"({out['n_landmarks']} landmarks)"
            )

    for side in ("left", "right"):
        iris_out = export_eyeball_iris5_texture(
            debug_dir,
            v_ict,
            f_ict,
            uvs,
            uv_faces,
            embedding,
            ict_npy_dict,
            side,
            size=size,
        )
        if iris_out is not None:
            key = f"eyeball_{side}_iris5"
            charts[key] = iris_out
    return charts


def export_ict_mediapipe_texture(
    debug_dir,
    v_ict,
    f_ict,
    uvs,
    uv_faces,
    embedding,
    size=2048,
    basename="ict_mediapipe",
    ict_npy_dict=None,
):
    """Face-focused overview (no teeth / mouth interior); see ``texture_maps/`` for eyes."""
    fi = None
    if ict_npy_dict is not None and "face_texture_map_id" in ict_npy_dict:
        fi = face_indices_excluding_materials(ict_npy_dict, VIZ_SKIP_MATERIALS)
    base = ict_uv_layout_texture(
        ict_npy_dict,
        uvs,
        uv_faces,
        f_ict,
        size=size,
        use_atlas_chart=False,
        face_indices=fi,
        exclude_materials=VIZ_SKIP_MATERIALS if fi is None else None,
    )
    mp_ids, landmark_uv = ict_mediapipe_landmark_uv(
        embedding, f_ict, uv_faces, uvs, ict_npy_dict=ict_npy_dict
    )
    if fi is not None:
        lmk_mask = landmark_mask_on_faces(embedding, fi)
        mp_ids = mp_ids[lmk_mask]
        landmark_uv = landmark_uv[lmk_mask]
    tex = bake_mediapipe_texture(
        landmark_uv,
        mp_ids,
        size=size,
        base_bgr=base,
        atlas_tiled=False,
    )
    out = Path(debug_dir) / f"{basename}_textured.obj"
    tex_path = export_textured_mesh(out, v_ict, f_ict, uvs, uv_faces, tex)
    cv2.imwrite(str(Path(debug_dir) / f"{basename}_texture.png"), tex)
    per_chart = {}
    if ict_npy_dict is not None and "face_texture_map_id" in ict_npy_dict:
        per_chart = export_ict_mediapipe_per_texture_map(
            Path(debug_dir),
            v_ict,
            f_ict,
            uvs,
            uv_faces,
            embedding,
            ict_npy_dict,
            size=size,
        )
    return out, tex_path, per_chart


def embedding_dict_from_npz(path):
    from processing.ict_mediapipe_lmk.embedding_io import load_ict_mediapipe_embedding

    return load_ict_mediapipe_embedding(path)


def _label_panel(img, title, bar_height=56):
    out = np.zeros((img.shape[0] + bar_height, img.shape[1], 3), dtype=np.uint8)
    out[bar_height:, :] = img
    cv2.putText(
        out,
        title,
        (12, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def write_comparison_panel(
    out_path,
    ict_tex_bgr,
    flame_tex_bgr,
    gap=16,
    *,
    left_title="ICT (baked embedding)",
    right_title="FLAME (repo FLAME.py + flame_face_uv)",
):
    ict_tex_bgr = np.asarray(ict_tex_bgr, dtype=np.uint8)
    flame_tex_bgr = np.asarray(flame_tex_bgr, dtype=np.uint8)
    h = max(ict_tex_bgr.shape[0], flame_tex_bgr.shape[0])
    if ict_tex_bgr.shape[0] != h:
        ict_tex_bgr = cv2.resize(ict_tex_bgr, (ict_tex_bgr.shape[1], h))
    if flame_tex_bgr.shape[0] != h:
        flame_tex_bgr = cv2.resize(flame_tex_bgr, (flame_tex_bgr.shape[1], h))
    panel = np.hstack(
        [
            _label_panel(ict_tex_bgr, left_title),
            np.full((h + 56, gap, 3), 32, dtype=np.uint8),
            _label_panel(flame_tex_bgr, right_title),
        ]
    )
    out_path = Path(out_path)
    cv2.imwrite(str(out_path), panel)
    return out_path


def export_landmark_texture_comparison(
    debug_dir,
    v_ict,
    f_ict,
    ict_uvs,
    ict_uv_faces,
    ict_embedding,
    *,
    v_flame,
    f_flame,
    mp_embedding_path,
    flame_model_path=None,
    flame_uv_mesh=None,
    ict_npy_dict=None,
    size=2048,
    export_flame=True,
):
    """ICT + FLAME textured landmark OBJ/PNG and side-by-side comparison."""
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    ict_obj, ict_tex_path, per_chart = export_ict_mediapipe_texture(
        debug_dir,
        v_ict,
        f_ict,
        ict_uvs,
        ict_uv_faces,
        ict_embedding,
        size=size,
        ict_npy_dict=ict_npy_dict,
    )
    ict_tex = cv2.imread(str(ict_tex_path))

    result = {"ict_obj": ict_obj, "ict_texture": ict_tex_path, "ict_texture_charts": per_chart}

    if not export_flame:
        return result

    flame_obj, flame_tex_path = export_flame_mediapipe_texture(
        debug_dir,
        v_flame,
        f_flame,
        mp_embedding_path,
        flame_uv_mesh=flame_uv_mesh,
        flame_model_path=flame_model_path,
        size=size,
    )
    flame_tex = cv2.imread(str(flame_tex_path))
    compare_path = write_comparison_panel(
        debug_dir / "mediapipe_landmarks_ict_vs_flame.png",
        ict_tex,
        flame_tex,
    )
    result["flame_obj"] = flame_obj
    result["flame_texture"] = flame_tex_path
    result["comparison_png"] = compare_path

    iris_l = per_chart.get("eyeball_left_iris5")
    iris_r = per_chart.get("eyeball_right_iris5")
    if iris_l is not None and iris_r is not None:
        sl = cv2.imread(str(iris_l["texture"]))
        sr = cv2.imread(str(iris_r["texture"]))
        eye_panel = write_comparison_panel(
            debug_dir / "mediapipe_eyeball_iris5_left_vs_right.png",
            sl,
            sr,
            left_title="eyeball L — iris MP 468-472",
            right_title="eyeball R — iris MP 473-477",
        )
        result["eyeball_iris_comparison_png"] = eye_panel
    return result
