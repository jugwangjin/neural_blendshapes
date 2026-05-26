"""
Build assets/ict_facekit_torch.npy — runtime topology source of truth.

Full ICT-FaceKit topology parts #0–#16 (vertices 0:26719), including
``M_EyeOcclusion`` (parts #13–#14). Lacrimal / eye-blend / eyelashes are kept in
the asset but gated off in ``model/expr_regions`` (train deformer ignores them).

Regenerate on server:
  python processing/ict_facekit_to_npy_full_head.py

Fits ICT→FLAME initial alignment (``optimize_ict_expression_to_flame``): ``jawOpen`` grid +
pytorch3d ``s,R,T`` on landmarks ``[17:]`` → ``flame_alignment_*``. ``--coarse_st_only`` for legacy ``s,T`` only.
Blendshapes unchanged.
"""

import sys
from pathlib import Path

_PROCESSING_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _PROCESSING_ROOT.parent
for _root in (_REPO_ROOT, _PROCESSING_ROOT):
    _p = str(_root)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from processing.paths import ASSETS_DIR, setup_import_paths, setup_ict_facekit_import
from processing.ict_region_dict import (
    OFFICIAL_FULL_PART_SPLITS,
    VERTEX_COUNT_FULL,
    build_full_head_region_indices,
    build_region_dict,
    vertex_parts_from_splits,
)

setup_import_paths()
_, facex_dir = setup_ict_facekit_import()
from ICT_FaceKit.Scripts import face_model_io

import argparse
import os

import numpy as np
import openmesh as om
from processing.ict_obj_materials import (
    build_geometry_chart_index,
    build_texture_map_index_from_materials,
    build_texture_map_index_from_uv,
    build_uv_seam_mesh,
    convert_quad_mesh_to_triangle_mesh,
    face_part_id_from_vertices,
    parse_obj_face_materials,
    print_texture_map_statistics,
)
from processing.ict_uv_viz import export_uv_debug, print_uv_statistics
from processing.ict_eye_uv_symmetry import analyze_eye_uv_symmetry, print_eye_uv_symmetry_report
from processing.ict_landmarks import (
    LANDMARK_START_FLAME_PAIRING,
    landmark_indices_for_asset,
    validate_landmark_indices,
)
from processing.ict_flame_similarity import (
    compute_ict_flame_alignment_for_npy,
    default_flame_similarity_fields,
    expression_modes_as_vertex_deltas,
    print_flame_alignment_report,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no_export_uv_debug",
        action="store_true",
        help="Skip debug/ict_facekit_uv/ per-usemtl chart PNGs (default: export all materials)",
    )
    parser.add_argument("--uv_debug_dir", type=str, default="debugs/ict_facekit_uv")
    parser.add_argument("--uv_texture_size", type=int, default=512)
    parser.add_argument(
        "--skip_flame_similarity",
        action="store_true",
        help="Do not fit/store FLAME uniform scale+translation (flame_similarity_s/T)",
    )
    parser.add_argument(
        "--flame_similarity_device",
        type=str,
        default="cpu",
        help="torch device for FLAME forward during similarity fit",
    )
    parser.add_argument(
        "--landmark_start",
        type=int,
        default=LANDMARK_START_FLAME_PAIRING,
        help="Skip first N ICT/FLAME landmarks (NICP / optimize_ict use [17:])",
    )
    parser.add_argument(
        "--coarse_st_only",
        action="store_true",
        help="Legacy: jaw + uniform s,T only (no flame_alignment s,R,T)",
    )
    parser.add_argument(
        "--ict_jaw_open",
        type=float,
        default=0.75,
        help="Initial jawOpen for grid search (final stored as flame_similarity_ict_jaw_open)",
    )
    parser.add_argument(
        "--no_optimize_jaw",
        action="store_true",
        help="Use fixed --ict_jaw_open instead of grid-search with coarse s,T",
    )
    parser.add_argument("--jaw_min", type=float, default=0.0)
    parser.add_argument("--jaw_max", type=float, default=1.5)
    parser.add_argument("--n_jaw_steps", type=int, default=25)
    parser.add_argument(
        "--w_jaw_knn",
        type=float,
        default=25.0,
        help="Weight on PIE jawline 0:16 KNN-to-FLAME when grid-searching jawOpen",
    )
    parser.add_argument(
        "--flame_use_processed_faces",
        action="store_true",
        help="FLAME processed faces (default: pkl / use_processed_faces=False)",
    )
    parser.add_argument(
        "--no_flame_canonical_pose",
        action="store_true",
        help="Use zero FLAME pose instead of canonical jaw-open pose",
    )
    args = parser.parse_args()

    file_path = os.fspath(Path(facex_dir) / "generic_neutral_mesh.obj")

    generic_neutral_mesh = om.read_polymesh(file_path, halfedge_tex_coord=True)
    mesh_faces = generic_neutral_mesh.face_vertex_indices()
    vertices = generic_neutral_mesh.points()
    tex_coords = generic_neutral_mesh.halfedge_texcoords2D()
    mesh_uvs = tex_coords[generic_neutral_mesh.face_halfedge_indices()]

    face_materials = parse_obj_face_materials(file_path)
    if len(face_materials) != len(mesh_faces):
        raise ValueError(
            f"OBJ face count {len(face_materials)} != openmesh {len(mesh_faces)}. "
            "Check generic_neutral_mesh.obj / usemtl order."
        )

    faces, triangle_uv, tri_material_names = convert_quad_mesh_to_triangle_mesh(
        mesh_faces, mesh_uvs, face_materials, n_verts=VERTEX_COUNT_FULL
    )
    print(f"  tris (full head verts 0:{VERTEX_COUNT_FULL}): F={len(faces)}")
    triangle_uv_atlas = triangle_uv.copy()

    mtl_pack = build_texture_map_index_from_materials(tri_material_names)
    uv_pack = build_texture_map_index_from_uv(triangle_uv_atlas)
    triangle_uv_local = uv_pack["triangle_uv_local"]
    face_uv_tile = uv_pack["face_uv_tile"]

    seam = build_uv_seam_mesh(
        faces,
        vertices,
        triangle_uv_local,
        face_uv_tile,
        n_3d_verts=VERTEX_COUNT_FULL,
    )
    new_vertices = seam["new_vertices"]
    new_uvs = seam["new_uvs"]
    new_faces = seam["new_faces"]
    vmapping = seam["vmapping"]
    uv_tile_index_vt = seam["uv_tile_index_vt"]
    vertex_uvs = seam["vertex_uvs"]
    uv_tile_index_v = seam["uv_tile_index_v"]

    ict_model = face_model_io.load_face_model(str(facex_dir))
    vertices = vertices[:VERTEX_COUNT_FULL]
    vertex_parts = vertex_parts_from_splits(len(vertices), OFFICIAL_FULL_PART_SPLITS)

    face_part_id = face_part_id_from_vertices(faces, vertex_parts)
    geom_pack = build_geometry_chart_index(face_part_id)
    face_texture_map_id = mtl_pack["face_texture_map_id"]

    print_texture_map_statistics(mtl_pack, uv_pack, geom_pack)

    print_uv_statistics(
        triangle_uv_atlas,
        triangle_uv_local,
        new_uvs,
        vertex_uvs,
        vertex_parts,
        vmapping,
        n_3d_verts=len(vertices),
        uv_quad_corners=mesh_uvs.reshape(-1, 2) if mesh_uvs.size else None,
        uv_tile_index_vt=uv_tile_index_vt,
        uv_tile_index_v=uv_tile_index_v,
        face_uv_tile=face_uv_tile,
    )

    regions = build_full_head_region_indices()

    eye_uv_symmetry = analyze_eye_uv_symmetry(
        vertex_uvs,
        regions["left_eyeball_indices"],
        regions["right_eyeball_indices"],
    )
    print_eye_uv_symmetry_report(eye_uv_symmetry)

    landmark_indices = landmark_indices_for_asset(use_jawline=True)
    validate_landmark_indices(landmark_indices, len(vertices))
    print(f"  landmark_indices: 68 Multi-PIE (ICT-FaceKit README jawline 0-16)")
    expression_shape_modes = expression_modes_as_vertex_deltas(
        ict_model._expression_shape_modes, len(vertices)
    )
    expression_names = ict_model._expression_names

    flame_sim_kw = dict(
        landmark_start=args.landmark_start,
        use_processed_faces=args.flame_use_processed_faces,
        use_canonical_pose=not args.no_flame_canonical_pose,
        ict_jaw_open=args.ict_jaw_open,
        optimize_jaw=not args.no_optimize_jaw,
        jaw_min=args.jaw_min,
        jaw_max=args.jaw_max,
        n_jaw_steps=args.n_jaw_steps,
        w_jaw_knn=args.w_jaw_knn,
    )
    if args.skip_flame_similarity:
        flame_sim = default_flame_similarity_fields(**flame_sim_kw)
        print("  FLAME similarity: skipped (--skip_flame_similarity), using s=1 T=0")
    else:
        flame_sim = compute_ict_flame_alignment_for_npy(
            vertices,
            landmark_indices,
            expression_shape_modes,
            expression_names,
            device=args.flame_similarity_device,
            coarse_st_only=args.coarse_st_only,
            **flame_sim_kw,
        )
        mode = "jaw+s,T" if args.coarse_st_only else "jaw+s,R,T"
        print(
            f"  FLAME align ({mode}): jawOpen={flame_sim['flame_similarity_ict_jaw_open']:.4f} "
            f"n_pairs={flame_sim['flame_similarity_n_pairs']} "
            f"lmk_err_mean={flame_sim['flame_similarity_lmk_err_mean']:.6f} "
            f"jaw_knn={flame_sim.get('flame_similarity_jaw_knn_mean', 0.0):.6f}"
        )
        if not args.coarse_st_only:
            print(
                f"  flame_alignment: s={flame_sim['flame_alignment_s']:.6f} "
                f"lmk_err_mean={flame_sim.get('flame_alignment_lmk_err_mean', flame_sim['flame_similarity_lmk_err_mean']):.6f}"
            )
        partial = {**flame_sim, "neutral_mesh": vertices, "landmark_indices": landmark_indices}
        print_flame_alignment_report(
            vertices,
            landmark_indices,
            expression_shape_modes,
            expression_names,
            partial,
            landmark_start=args.landmark_start,
            device=args.flame_similarity_device,
        )

    ict_model_dict = {
        "neutral_mesh": vertices,
        "uv_neutral_mesh": vertex_uvs,
        "uv_tile_index_v": uv_tile_index_v,
        "vertex_parts": vertex_parts,
        "faces": faces,
        "uv_faces": new_faces,
        "uvs": new_uvs,
        "uv_tile_index_vt": uv_tile_index_vt,
        "vmapping": vmapping,
        "triangle_uv_atlas": triangle_uv_atlas.astype(np.float32),
        "triangle_uv_local": triangle_uv_local,
        "face_texture_map_id": face_texture_map_id,
        "face_material_name": mtl_pack["face_material_name"],
        "material_names": mtl_pack["material_names"],
        "primary_texture_materials": mtl_pack["primary_texture_materials"],
        "n_texture_maps": mtl_pack["n_texture_maps"],
        "face_geometry_chart_id": geom_pack["face_geometry_chart_id"],
        "face_part_id": face_part_id,
        "face_uv_tile_u": uv_pack["face_uv_tile_u"],
        "face_uv_tile_v": uv_pack["face_uv_tile_v"],
        "texture_map_tile": uv_pack["texture_map_tile"],
        "geometry_chart_part": geom_pack["geometry_chart_part"],
        "n_geometry_charts": geom_pack["n_geometry_charts"],
        "quad_faces": generic_neutral_mesh.face_vertex_indices(),
        "num_expression": ict_model._num_expression_shapes,
        "num_identity": ict_model._num_identity_shapes,
        "expression_shape_modes": expression_shape_modes,
        "identity_shape_modes": ict_model._identity_shape_modes[:, : len(vertices)],
        "expression_names": expression_names,
        "identity_names": ict_model._identity_names,
        "model_config": ict_model._model_config,
        "landmark_indices": landmark_indices,
        "eye_uv_mirror_right_u": np.bool_(eye_uv_symmetry["mirror_right_u"]),
        **flame_sim,
    }
    ict_model_dict.update(regions)
    ict_model_dict.update(
        build_region_dict(
            vertices,
            vertex_parts,
            regions["face_indices"],
            regions["not_face_indices"],
            regions["eyeball_indices"],
            OFFICIAL_FULL_PART_SPLITS,
        )
    )
    n_occ = len(regions.get("left_eye_occlusion_indices", [])) + len(
        regions.get("right_eye_occlusion_indices", [])
    )
    print(f"  eye_occlusion verts: {n_occ}  (M_EyeOcclusion surface Gaussians)")

    out_path = ASSETS_DIR / "ict_facekit_torch.npy"
    np.save(str(out_path), ict_model_dict)
    print(f"saved {out_path}")
    print(f"  variant={ict_model_dict['asset_variant']} schema={ict_model_dict['asset_schema_version']}")
    print(f"  verts={len(vertices)} surface_sample={len(regions['surface_sample_vertex_indices'])}")
    print(f"  eyeball L/R={len(regions['left_eyeball_indices'])}/{len(regions['right_eyeball_indices'])}")
    print(
        f"  flame_similarity s={ict_model_dict['flame_similarity_s']:.6f} "
        f"T={ict_model_dict['flame_similarity_T']}"
    )
    print(
        f"  UV: F_tri={len(faces)} VT={len(new_uvs)} vmapping={len(vmapping)} "
        f"materials K={mtl_pack['n_texture_maps']} geometry_charts G={geom_pack['n_geometry_charts']}"
    )

    if not args.no_export_uv_debug:
        export_uv_debug(
            args.uv_debug_dir,
            vertices,
            faces,
            new_uvs,
            new_faces,
            vertex_uvs,
            vertex_parts,
            vmapping,
            face_texture_map_id=face_texture_map_id,
            material_names=mtl_pack["material_names"],
            texture_map_tile=uv_pack["texture_map_tile"],
            triangle_uv_atlas=triangle_uv_atlas,
            texture_size=args.uv_texture_size,
        )


if __name__ == "__main__":
    main()
