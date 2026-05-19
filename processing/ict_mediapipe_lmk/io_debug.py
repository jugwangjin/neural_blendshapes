"""Debug exports for the ICT MediaPipe baker."""

from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
import torch

from processing.ict_mediapipe_lmk.landmarks import vertices2landmarks
from processing.ict_mediapipe_lmk.lmk68_texture_viz import export_68_landmark_texture_comparison
from processing.ict_mediapipe_lmk.texture_viz import export_landmark_texture_comparison


def export_landmark_texture_qa(
    debug_dir,
    v_ict_fit,
    f_ict,
    ict_uvs,
    ict_uv_faces,
    embedding,
    *,
    v_flame,
    f_flame,
    mp_embedding_path,
    flame_model_path=None,
    flame_uv_mesh=None,
    ict_npy_dict=None,
    texture_size=2048,
):
    """ICT + FLAME MediaPipe landmarks on UV texture maps + side-by-side PNG."""
    return export_landmark_texture_comparison(
        debug_dir,
        v_ict_fit,
        f_ict,
        ict_uvs,
        ict_uv_faces,
        embedding,
        v_flame=v_flame,
        f_flame=f_flame,
        mp_embedding_path=mp_embedding_path,
        flame_model_path=flame_model_path,
        flame_uv_mesh=flame_uv_mesh,
        ict_npy_dict=ict_npy_dict,
        size=texture_size,
        export_flame=True,
    )


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


def export_initial_alignment_debug(
    debug_dir,
    v_ict_initial,
    f_ict,
    v_flame,
    f_flame,
    ict_lmk,
    flame_lmk_face_idx,
    flame_lmk_bary,
    *,
    ict_uvs=None,
    ict_uv_faces=None,
    ict_npy_dict=None,
    flame_model_path=None,
    flame_uv_mesh=None,
    texture_size=2048,
):
    """
    **Initial** ICT in FLAME space (jawOpen + coarse s,T; optional flame_alignment s,R,T), before face NICP.
    Canonical FLAME mesh + 68-point (ICT full / FLAME [17:]) UV textures.
    """
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    write_mesh(debug_dir / "ict_initial_flame_space.obj", v_ict_initial, f_ict)
    write_mesh(debug_dir / "flame_canonical.obj", v_flame, f_flame)

    lmk_ict = np.asarray(v_ict_initial, dtype=np.float64)[np.asarray(ict_lmk, dtype=np.int64)]
    write_point_cloud(debug_dir / "ict_initial_68_landmarks.ply", lmk_ict, (1.0, 0.2, 0.0))

    from processing.ict_flame_similarity import sample_landmarks_bary

    lmk_flame = sample_landmarks_bary(v_flame, f_flame, flame_lmk_face_idx, flame_lmk_bary)
    write_point_cloud(debug_dir / "flame_68_landmarks.ply", lmk_flame, (0.0, 1.0, 0.0))

    if ict_uvs is None or ict_uv_faces is None:
        return None

    out = export_68_landmark_texture_comparison(
        debug_dir,
        v_ict_initial,
        f_ict,
        ict_uvs,
        ict_uv_faces,
        ict_lmk,
        v_flame=v_flame,
        f_flame=f_flame,
        flame_lmk_face_idx=flame_lmk_face_idx,
        flame_lmk_bary=flame_lmk_bary,
        flame_model_path=flame_model_path,
        flame_uv_mesh=flame_uv_mesh,
        ict_npy_dict=ict_npy_dict,
        size=texture_size,
        ict_basename="ict_initial_68lmk",
        flame_basename="flame_canonical_68lmk",
        panel_name="initial_68lmk_ict_vs_flame.png",
    )
    print(f"Initial ICT (pre-NICP) 68pt texture: {out['ict_texture']}")
    print(f"FLAME canonical 68pt texture: {out['flame_texture']}")
    print(f"Initial 68pt panel: {out['comparison_png']}")
    return out


def export_nicp_fit_68_debug(
    debug_dir,
    v_ict_fit,
    f_ict,
    ict_lmk,
    *,
    ict_uvs,
    ict_uv_faces,
    ict_npy_dict=None,
    texture_size=2048,
):
    """ICT after face NICP — 68-point UV texture only."""
    from processing.ict_mediapipe_lmk.lmk68_texture_viz import export_ict_68_texture

    debug_dir = Path(debug_dir)
    write_mesh(debug_dir / "ict_nicp_fit_to_flame.obj", v_ict_fit, f_ict)
    lmk = np.asarray(v_ict_fit, dtype=np.float64)[np.asarray(ict_lmk, dtype=np.int64)]
    write_point_cloud(debug_dir / "ict_nicp_fit_68_landmarks.ply", lmk, (1.0, 0.5, 0.0))
    if ict_uvs is None or ict_uv_faces is None:
        return None
    obj, tex = export_ict_68_texture(
        debug_dir,
        v_ict_fit,
        f_ict,
        ict_uvs,
        ict_uv_faces,
        ict_lmk,
        size=texture_size,
        ict_npy_dict=ict_npy_dict,
        basename="ict_nicp_fit_68lmk",
        protocol_offset=0,
    )
    print(f"Post-NICP ICT 68pt texture: {tex}")
    return {"obj": obj, "texture": tex}


# Back-compat alias
export_canonical_alignment_debug = export_initial_alignment_debug


def export_eye_fitting_debug(debug_dir, eye_embedding):
    """
    Per-eye meshes after FLAME→ICT eye NICP (FLAME space).

    ``debug_dir/eyes/``:
      - ``flame_eye_{left,right}_fitted.obj`` — NICP-deformed FLAME eyeball
      - ``ict_eyeball_{left,right}.obj`` — ICT-FaceKit eyeball submesh (target)
      - ``*_iris_*.ply`` — MP iris center / sclera UV pole / fitted pentagon
      - ``ict_eye_fitting.npz`` — numeric bundle
    """
    debug_dir = Path(debug_dir) / "eyes"
    debug_dir.mkdir(parents=True, exist_ok=True)
    npz_parts = {}

    for side_key, label in (("left", "left"), ("right", "right")):
        pack = eye_embedding.get(side_key)
        if pack is None or "mesh_debug" not in pack:
            continue
        md = pack["mesh_debug"]
        fl_v = md["flame_eye_fit_vertices"]
        fl_f = md["flame_eye_fit_faces"]
        ic_v = md["ict_eyeball_vertices"]
        ic_f = md["ict_eyeball_faces"]

        write_mesh(debug_dir / f"flame_eye_{label}_fitted.obj", fl_v, fl_f)
        write_mesh(debug_dir / f"flame_eye_{label}_canonical.obj", md["flame_eye_canonical_vertices"], fl_f)
        write_mesh(debug_dir / f"ict_eyeball_{label}.obj", ic_v, ic_f)

        write_point_cloud(
            debug_dir / f"ict_eyeball_{label}_sclera_uv_center.ply",
            md["ict_sclera_uv_center"][None],
            (1.0, 0.8, 0.0),
        )
        write_point_cloud(
            debug_dir / f"ict_eyeball_{label}_back.ply",
            md["ict_eyeball_back"][None],
            (0.6, 0.4, 1.0),
        )
        write_point_cloud(
            debug_dir / f"flame_eye_{label}_mp_iris_center.ply",
            md["flame_iris_center"][None],
            (1.0, 0.2, 0.2),
        )
        write_point_cloud(
            debug_dir / f"flame_eye_{label}_back.ply",
            md["flame_eyeball_back"][None],
            (1.0, 0.5, 0.0),
        )
        write_point_cloud(
            debug_dir / f"flame_eye_{label}_iris_fitted.ply",
            md["fitted_iris_points"],
            (0.0, 1.0, 0.4),
        )

        prefix = f"{label}"
        npz_parts[f"{prefix}_flame_eye_fit_v"] = fl_v.astype(np.float32)
        npz_parts[f"{prefix}_flame_eye_f"] = fl_f.astype(np.int64)
        npz_parts[f"{prefix}_ict_eyeball_v"] = ic_v.astype(np.float32)
        npz_parts[f"{prefix}_ict_eyeball_f"] = ic_f.astype(np.int64)
        npz_parts[f"{prefix}_ict_eyeball_vids"] = md["ict_eyeball_vertex_ids_global"]
        npz_parts[f"{prefix}_flame_eye_vids"] = md["flame_eye_vertex_ids_global"]
        npz_parts[f"{prefix}_fitted_iris"] = md["fitted_iris_points"].astype(np.float32)
        npz_parts[f"{prefix}_flame_iris_center"] = md["flame_iris_center"].astype(np.float32)
        npz_parts[f"{prefix}_ict_sclera_uv_center"] = md["ict_sclera_uv_center"].astype(np.float32)
        npz_parts[f"{prefix}_ict_eyeball_back"] = md["ict_eyeball_back"].astype(np.float32)
        npz_parts[f"{prefix}_flame_eyeball_back"] = md["flame_eyeball_back"].astype(np.float32)
        npz_parts[f"{prefix}_eye_s"] = np.float32(md["flame_alignment_s"])
        npz_parts[f"{prefix}_eye_R"] = md["flame_alignment_R"].astype(np.float32)
        npz_parts[f"{prefix}_eye_T"] = md["flame_alignment_T"].astype(np.float32)

        print(
            f"  eye fit [{label}]: flame fitted V={fl_v.shape[0]} | "
            f"ict eyeball V={ic_v.shape[0]} → {debug_dir.name}/"
        )

    if npz_parts:
        npz_path = debug_dir / "ict_eye_fitting.npz"
        np.savez(npz_path, **npz_parts)
        print(f"  eye fit npz: {npz_path}")
    return debug_dir


def export_debug(
    debug_dir,
    v_flame,
    f_flame,
    v_ict_fit,
    f_ict,
    mp_pack,
    embedding,
    mp_embedding_path=None,
    flame_uv_mesh=None,
    flame_model_path=None,
    ict_uvs=None,
    ict_uv_faces=None,
    ict_npy_dict=None,
    texture_size=2048,
    export_flame_texture=True,
):
    """Meshes + PLY point clouds + ICT/FLAME landmark UV textures for side-by-side QA."""
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    write_mesh(debug_dir / "flame_canonical_after_nicp_ref.obj", v_flame, f_flame)
    write_mesh(debug_dir / "ict_nicp_fit_to_flame.obj", v_ict_fit, f_ict)
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

    if "transfer_error" in embedding:
        err = embedding["transfer_error"]
        types = embedding["ict_lmk_target_type"]
        print(
            f"transfer_error: mean={err.mean():.6f}, max={err.max():.6f}, "
            f"p95={np.percentile(err, 95):.6f}"
        )
        for t in np.unique(types):
            m = types == t
            print(f"  [{t}] n={m.sum()} err_mean={err[m].mean():.6f} err_max={err[m].max():.6f}")

    if export_flame_texture and ict_uvs is not None and ict_uv_faces is not None and mp_embedding_path:
        out = export_landmark_texture_qa(
            debug_dir,
            v_ict_fit,
            f_ict,
            ict_uvs,
            ict_uv_faces,
            embedding,
            v_flame=v_flame,
            f_flame=f_flame,
            mp_embedding_path=mp_embedding_path,
            flame_model_path=flame_model_path,
            flame_uv_mesh=flame_uv_mesh,
            ict_npy_dict=ict_npy_dict,
            texture_size=texture_size,
        )
        print(f"ICT landmark texture QA (face): {out['ict_texture']}")
        print(f"FLAME landmark texture QA: {out['flame_texture']}")
        print(f"Comparison panel: {out['comparison_png']}")
        for mat, info in (out.get("ict_texture_charts") or {}).items():
            print(f"  ICT texture map [{mat}]: {info['texture']}")
        if "eyeball_iris_comparison_png" in out:
            print(f"Eyeball iris×5 panel: {out['eyeball_iris_comparison_png']}")
