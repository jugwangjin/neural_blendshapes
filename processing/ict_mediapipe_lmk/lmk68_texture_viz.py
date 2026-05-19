"""68-point (Multi-PIE / ICT-FaceKit) landmark UV texture QA — not MediaPipe."""

from pathlib import Path

import cv2
import numpy as np

from processing.ict_landmarks import FACIAL_68_CONNECTIONS, LANDMARK_START_FLAME_PAIRING
from processing.ict_mediapipe_lmk.texture_viz import (
    export_textured_mesh,
    flame_uv_layout_texture,
    ict_uv_layout_texture,
    landmark_uv_from_vertex_corners,
    resolve_flame_uv_for_topology,
    uv_to_pixel,
)


def flame_embedding_protocol_offset(n_landmarks):
    """``flame_static_embedding.pkl`` is usually 51 pts (= protocol indices 17..67)."""
    if n_landmarks >= 68:
        return 0
    if n_landmarks == 51:
        return LANDMARK_START_FLAME_PAIRING
    return max(0, 68 - n_landmarks)


def _uv_to_pixel_fn(size, atlas_tiled=False):
    if atlas_tiled:
        from processing.ict_uv_viz import uv_to_pixel_atlas

        return lambda uv: uv_to_pixel_atlas(uv, size)
    return lambda uv: uv_to_pixel(uv, size)


def bake_68_landmark_texture(
    landmark_uv,
    size,
    base_bgr,
    *,
    protocol_offset=0,
    atlas_tiled=False,
):
    """
    Draw Multi-PIE topology on UV layout.

    ``landmark_uv[i]`` ↔ protocol index ``protocol_offset + i`` (e.g. FLAME 51 → 17..67).
    """
    to_px = _uv_to_pixel_fn(size, atlas_tiled)
    img = np.asarray(base_bgr, dtype=np.float64).copy()
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if not atlas_tiled and (img.shape[0] != size or img.shape[1] != size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)

    n = len(landmark_uv)
    p0, p1 = protocol_offset, protocol_offset + n

    for seg in FACIAL_68_CONNECTIONS:
        local = [int(i) - protocol_offset for i in seg if p0 <= int(i) < p1]
        for j in range(len(local) - 1):
            a, b = local[j], local[j + 1]
            pa = to_px(landmark_uv[a])
            pb = to_px(landmark_uv[b])
            cv2.line(img, pa, pb, (0, 255, 0), 2, cv2.LINE_AA)

    for i in range(n):
        px = to_px(landmark_uv[i])
        label = str(protocol_offset + i)
        cv2.circle(img, px, 4, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            img,
            label,
            (px[0] + 3, px[1] - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.28,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return img


def ict_68_landmark_uv(v_ict, f_ict, uv_faces, uvs, landmark_indices):
    return landmark_uv_from_vertex_corners(landmark_indices, f_ict, uv_faces, uvs)


def export_ict_68_texture(
    debug_dir,
    v_ict,
    f_ict,
    uvs,
    uv_faces,
    landmark_indices,
    *,
    size=2048,
    ict_npy_dict=None,
    basename="ict_68lmk",
    protocol_offset=0,
):
    from processing.ict_mediapipe_lmk.texture_viz import (
        VIZ_SKIP_MATERIALS,
        face_indices_excluding_materials,
    )

    landmark_uv = ict_68_landmark_uv(v_ict, f_ict, uv_faces, uvs, landmark_indices)
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
    )
    tex = bake_68_landmark_texture(
        landmark_uv, size, base, protocol_offset=protocol_offset, atlas_tiled=False
    )
    out = Path(debug_dir) / f"{basename}_textured.obj"
    tex_path = export_textured_mesh(out, v_ict, f_ict, uvs, uv_faces, tex)
    png_path = Path(debug_dir) / f"{basename}_texture.png"
    cv2.imwrite(str(png_path), tex)
    return out, png_path


def export_flame_68_texture(
    debug_dir,
    v_flame,
    f_flame,
    flame_lmk_face_idx,
    flame_lmk_bary,
    *,
    flame_model_path=None,
    flame_uv_mesh=None,
    size=2048,
    basename="flame_68lmk",
):
    uvs, uv_faces, _ = resolve_flame_uv_for_topology(
        f_flame,
        flame_model_path=flame_model_path,
        flame_uv_mesh=flame_uv_mesh,
    )
    landmark_uv = landmark_uv_from_bary_flame(
        v_flame, f_flame, flame_lmk_face_idx, flame_lmk_bary, uv_faces, uvs
    )
    proto_off = flame_embedding_protocol_offset(len(flame_lmk_face_idx))
    base = flame_uv_layout_texture(uvs, uv_faces, size=size)
    tex = bake_68_landmark_texture(landmark_uv, size, base, protocol_offset=proto_off, atlas_tiled=False)
    out = Path(debug_dir) / f"{basename}_textured.obj"
    export_textured_mesh(out, v_flame, f_flame, uvs, uv_faces, tex)
    png_path = Path(debug_dir) / f"{basename}_texture.png"
    cv2.imwrite(str(png_path), tex)
    return out, png_path, proto_off


def landmark_uv_from_bary_flame(v_flame, f_flame, face_idx, bary, uv_faces, uvs):
    from processing.ict_mediapipe_lmk.texture_viz import landmark_uv_from_bary

    return landmark_uv_from_bary(face_idx, bary, f_flame, uv_faces, uvs)


def export_68_landmark_texture_comparison(
    debug_dir,
    v_ict,
    f_ict,
    ict_uvs,
    ict_uv_faces,
    landmark_indices,
    *,
    v_flame,
    f_flame,
    flame_lmk_face_idx,
    flame_lmk_bary,
    flame_model_path=None,
    flame_uv_mesh=None,
    ict_npy_dict=None,
    size=2048,
    ict_basename="ict_68lmk",
    flame_basename="flame_68lmk",
    panel_name="68lmk_ict_vs_flame.png",
):
    from processing.ict_mediapipe_lmk.texture_viz import write_comparison_panel

    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    ict_obj, ict_tex = export_ict_68_texture(
        debug_dir,
        v_ict,
        f_ict,
        ict_uvs,
        ict_uv_faces,
        landmark_indices,
        size=size,
        ict_npy_dict=ict_npy_dict,
        basename=ict_basename,
        protocol_offset=0,
    )
    flame_obj, flame_tex, proto_off = export_flame_68_texture(
        debug_dir,
        v_flame,
        f_flame,
        flame_lmk_face_idx,
        flame_lmk_bary,
        flame_model_path=flame_model_path,
        flame_uv_mesh=flame_uv_mesh,
        size=size,
        basename=flame_basename,
    )
    print(f"  FLAME embedding: n={len(flame_lmk_face_idx)} protocol_offset={proto_off}")
    ict_bgr = cv2.imread(str(ict_tex))
    flame_bgr = cv2.imread(str(flame_tex))
    panel = write_comparison_panel(
        debug_dir / panel_name,
        ict_bgr,
        flame_bgr,
    )
    return {
        "ict_obj": ict_obj,
        "ict_texture": ict_tex,
        "flame_obj": flame_obj,
        "flame_texture": flame_tex,
        "comparison_png": panel,
        "flame_protocol_offset": proto_off,
    }

