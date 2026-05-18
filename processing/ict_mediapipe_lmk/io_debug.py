"""Debug exports for the ICT MediaPipe baker."""

from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

from ict_mediapipe_lmk.landmarks import vertices2landmarks
from ict_mediapipe_lmk.texture_viz import (
    export_flame_mediapipe_texture,
    export_ict_mediapipe_texture,
)
import torch


def write_mesh(path, vertices, faces):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)


def write_point_cloud(path, points, color=(0.0, 1.0, 0.0)):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.paint_uniform_color(color)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), pcd)


def export_debug(
    debug_dir,
    v_flame,
    f_flame,
    v_ict_fit,
    f_ict,
    mp_pack,
    embedding,
    mp_embedding_path=None,
    flame_uv_mesh_path=None,
    ict_uvs=None,
    ict_uv_faces=None,
    texture_size=2048,
):
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    write_mesh(debug_dir / "flame_canonical.obj", v_flame, f_flame)
    write_mesh(debug_dir / "ict_fit_to_flame.obj", v_ict_fit, f_ict)
    write_point_cloud(debug_dir / "mp_points_on_flame.ply", mp_pack["points_flame"], (0.0, 1.0, 0.0))

    v_t = torch.tensor(v_ict_fit, dtype=torch.float32)[None]
    f_t = torch.tensor(f_ict, dtype=torch.long)
    ict_mp = vertices2landmarks(
        v_t,
        f_t,
        torch.tensor(embedding["ict_lmk_face_idx"], dtype=torch.long),
        torch.tensor(embedding["ict_lmk_b_coords"], dtype=torch.float32),
    )[0].numpy()
    write_point_cloud(debug_dir / "mp_points_on_ict.ply", ict_mp, (1.0, 0.2, 0.0))

    err = embedding["transfer_error"]
    print(
        f"transfer_error: mean={err.mean():.6f}, max={err.max():.6f}, "
        f"p95={np.percentile(err, 95):.6f}"
    )

    if mp_embedding_path and flame_uv_mesh_path:
        flame_obj, flame_tex = export_flame_mediapipe_texture(
            debug_dir,
            v_flame,
            f_flame,
            flame_uv_mesh_path,
            mp_embedding_path,
            size=texture_size,
        )
        print(f"FLAME textured mesh: {flame_obj} (texture: {flame_tex})")

    if ict_uvs is not None and ict_uv_faces is not None:
        ict_obj, ict_tex = export_ict_mediapipe_texture(
            debug_dir,
            v_ict_fit,
            f_ict,
            ict_uvs,
            ict_uv_faces,
            embedding,
            size=texture_size,
        )
        print(f"ICT textured mesh: {ict_obj} (texture: {ict_tex})")
