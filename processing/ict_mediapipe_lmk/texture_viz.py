"""Bake MediaPipe landmarks and contours into UV textures for visual QA."""

from pathlib import Path

import cv2
import numpy as np
from pytorch3d.io import load_obj

from ict_mediapipe_lmk.constants import (
    LEFT_IRIS_FLAME,
    LEFT_IRIS_MP,
    RIGHT_IRIS_FLAME,
    RIGHT_IRIS_MP,
)
from ict_mediapipe_lmk.mediapipe_connections import connections_for_landmarks
from ict_mediapipe_lmk.metrical import load_metrical_mediapipe_embedding


def load_flame_uv_mesh(path):
    _, faces, aux = load_obj(str(path), load_textures=False)
    faces = faces.verts_idx.cpu().numpy()
    uvs = aux.verts_uvs.cpu().numpy()
    uv_faces = aux.faces_uvs.cpu().numpy()
    return faces, uvs, uv_faces


def landmark_uv_from_bary(face_idx, bary, faces, uv_faces, uvs):
    tri = faces[face_idx]
    tri_uv_idx = uv_faces[face_idx]
    tri_uv = uvs[tri_uv_idx]
    return (tri_uv * bary[:, :, None]).sum(axis=1)


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


def bake_mediapipe_texture(
    landmark_uv,
    mp_indices,
    size=2048,
    line_color=(0, 255, 0),
    point_color=(255, 0, 0),
    iris_color=(0, 200, 255),
    draw_indices=True,
):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    mp_to_row = {int(mp): i for i, mp in enumerate(mp_indices)}

    for a, b in connections_for_landmarks(mp_indices):
        pa = uv_to_pixel(landmark_uv[mp_to_row[a]], size)
        pb = uv_to_pixel(landmark_uv[mp_to_row[b]], size)
        cv2.line(img, pa, pb, line_color, 2, cv2.LINE_AA)

    for mp_id, uv in zip(mp_indices, landmark_uv):
        px = uv_to_pixel(uv, size)
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


def ict_mediapipe_landmark_uv(embedding, f_ict, uv_faces, uvs):
    face_idx = embedding["ict_lmk_face_idx"]
    bary = embedding["ict_lmk_b_coords"]
    return embedding["mp_landmark_indices"], landmark_uv_from_bary(
        face_idx, bary, f_ict, uv_faces, uvs
    )


def export_flame_mediapipe_texture(
    debug_dir,
    v_flame,
    f_flame,
    flame_uv_mesh_path,
    mp_embedding_path,
    size=2048,
):
    _, uvs, uv_faces = load_flame_uv_mesh(flame_uv_mesh_path)
    mp_ids, landmark_uv = flame_mediapipe_landmark_uv(
        v_flame, f_flame, uv_faces, uvs, mp_embedding_path
    )
    tex = bake_mediapipe_texture(landmark_uv, mp_ids, size=size)
    out = Path(debug_dir) / "flame_mediapipe_textured.obj"
    tex_path = export_textured_mesh(out, v_flame, f_flame, uvs, uv_faces, tex)
    cv2.imwrite(str(Path(debug_dir) / "flame_mediapipe_texture.png"), tex)
    return out, tex_path


def export_ict_mediapipe_texture(
    debug_dir,
    v_ict,
    f_ict,
    uvs,
    uv_faces,
    embedding,
    size=2048,
):
    mp_ids, landmark_uv = ict_mediapipe_landmark_uv(embedding, f_ict, uv_faces, uvs)
    tex = bake_mediapipe_texture(landmark_uv, mp_ids, size=size)
    out = Path(debug_dir) / "ict_mediapipe_textured.obj"
    tex_path = export_textured_mesh(out, v_ict, f_ict, uvs, uv_faces, tex)
    cv2.imwrite(str(Path(debug_dir) / "ict_mediapipe_texture.png"), tex)
    return out, tex_path
