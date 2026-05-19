"""
Bake MediaPipe landmarks from metrical-tracker FLAME embedding onto ICT-FaceKit.

Pipeline:
  A. npy ``jawOpen`` + ``flame_alignment_s,R,T`` (or coarse ``s,T`` with --coarse_st_only)
  B. NICP: inner 68[17:]+PIE jawline[0:16] KNN -> FLAME (eyeball untouched)
  C. Sample FLAME MediaPipe face/eyelid points -> project to fitted ICT
  D. Eye s,T (R=I): bidirectional chamfer + front/back anchors
  E. Transplant iris MP 468–477 onto ICT eyeball barycentric coords
  F. Save canonical ICT (FLAME space) + FLAME meshes; 68-point UV QA
  G. ICT + FLAME MediaPipe landmark UV texture maps (``--no_texture_viz`` to skip)

Run from repo root:
  python processing/ict_mediapipe_lmk/bake_mediapipe_to_ict.py
"""

import argparse
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent
_PROCESSING_ROOT = _PKG.parent
_REPO_ROOT = _PROCESSING_ROOT.parent
for _root in (_REPO_ROOT, _PROCESSING_ROOT):
    _p = str(_root)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

from processing.paths import setup_import_paths
from processing.ict_mediapipe_lmk.constants import (
    DEFAULT_DEBUG_DIR,
    DEFAULT_FLAME_LMK_EMBEDDING,
    DEFAULT_FLAME_MODEL,
    DEFAULT_FLAME_UV_MESH,
    DEFAULT_ICT_CANONICAL,
    DEFAULT_ICT_NPY,
    DEFAULT_LARGE_STEPS_ROOT,
    DEFAULT_METRICAL_ROOT,
    DEFAULT_OUTPUT_NPZ,
)
from processing.ict_mediapipe_lmk.embedding_io import (
    assert_flame_mp_embedding,
    save_ict_mediapipe_embedding,
    save_ict_mediapipe_embedding_aux,
)
from processing.ict_mediapipe_lmk.io_debug import (
    export_debug,
    export_eye_fitting_debug,
    export_initial_alignment_debug,
    export_landmark_texture_qa,
    export_nicp_fit_68_debug,
)
from processing.ict_mediapipe_lmk.metrical import build_flame_mp_points, load_flame_static_embedding
from processing.ict_mediapipe_lmk.nicp import fit_ict_face_to_flame
from processing.ict_mediapipe_lmk.eye_transplant import merge_iris_into_embedding, run_eye_transplant
from processing.ict_mediapipe_lmk.transfer import transfer_mediapipe_to_ict, validate_embedding

setup_import_paths()


def flame_flags_from_npy(ict_npy_dict, args):
    """Match FLAME mesh to npy similarity fit (processed faces, canonical pose)."""
    default_proc = args.use_processed_faces and not args.no_processed_faces
    default_pose = not args.no_flame_canonical_pose
    if ict_npy_dict is None:
        return default_proc, default_pose
    use_proc = bool(ict_npy_dict.get("flame_similarity_use_processed_faces", default_proc))
    use_pose = bool(ict_npy_dict.get("flame_similarity_use_canonical_pose", default_pose))
    return use_proc, use_pose


def load_flame_canonical(args, device, ict_npy_dict=None):
    if args.flame_mesh:
        import trimesh

        mesh = trimesh.load_mesh(args.flame_mesh, process=False)
        return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)

    from processing.flame.flame_viz import load_flame_canonical_mesh

    use_proc, use_pose = flame_flags_from_npy(ict_npy_dict, args)
    return load_flame_canonical_mesh(
        args.flame_model,
        use_processed_faces=use_proc,
        use_canonical_pose=use_pose,
        n_shape=args.n_shape,
        n_exp=args.n_exp,
        device=device,
    )


def load_ict_from_npy(args):
    from processing.ict_npy_loader import load_ict_asset

    apply_sim = not args.skip_flame_similarity
    v_ict, f_ict, uvs, uv_faces, ict_lmk, regions, d = load_ict_asset(
        args.ict_npy, apply_flame_similarity_transform=False
    )
    if args.ict_mesh:
        import trimesh

        mesh = trimesh.load_mesh(args.ict_mesh, process=False)
        v_ict = np.asarray(mesh.vertices, dtype=np.float64)
    return v_ict, f_ict, ict_lmk, regions, uvs, uv_faces, apply_sim, d


def ensure_flame_alignment(args, ict_npy_dict, ict_lmk, device):
    from processing.ict_flame_similarity import (
        fit_rigid_alignment_fields,
        has_flame_alignment,
        merge_flame_alignment_into_npy,
    )

    if has_flame_alignment(ict_npy_dict) and not args.recompute_flame_alignment:
        return ict_npy_dict
    if args.coarse_st_only or args.ignore_stored_flame_alignment:
        return ict_npy_dict

    use_proc = bool(
        ict_npy_dict.get(
            "flame_similarity_use_processed_faces",
            args.use_processed_faces and not args.no_processed_faces,
        )
    )
    use_pose = bool(
        ict_npy_dict.get(
            "flame_similarity_use_canonical_pose",
            not args.no_flame_canonical_pose,
        )
    )
    landmark_start = int(
        ict_npy_dict.get("flame_similarity_landmark_start", args.landmark_start)
    )
    fields = fit_rigid_alignment_fields(
        ict_npy_dict,
        flame_model_path=args.flame_model,
        flame_lmk_embedding_path=args.flame_lmk_embedding,
        landmark_indices=ict_lmk,
        landmark_start=landmark_start,
        use_processed_faces=use_proc,
        use_canonical_pose=use_pose,
        device=device,
    )
    ict_npy_dict.update(fields)
    print(
        f"FLAME alignment (jaw + s,R,T): s={fields['flame_alignment_s']:.6f} "
        f"lmk_err_mean={fields['flame_alignment_lmk_err_mean']:.6f}"
    )
    if not args.skip_save_alignment:
        merge_flame_alignment_into_npy(args.ict_npy, fields)
        print(f"  updated {args.ict_npy} with flame_alignment_s/R/T")
    return ict_npy_dict


def main():
    parser = argparse.ArgumentParser(description="Bake metrical-tracker MediaPipe landmarks onto ICT")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_NPZ))
    parser.add_argument("--debug_dir", type=str, default=str(DEFAULT_DEBUG_DIR))
    parser.add_argument("--metrical_root", type=str, default=str(DEFAULT_METRICAL_ROOT))
    parser.add_argument("--mp_embedding", type=str, default="")
    parser.add_argument("--large_steps_root", type=str, default=str(DEFAULT_LARGE_STEPS_ROOT))
    parser.add_argument("--flame_lmk_embedding", type=str, default=str(DEFAULT_FLAME_LMK_EMBEDDING))
    parser.add_argument("--ict_npy", type=str, default=str(DEFAULT_ICT_NPY))
    parser.add_argument("--ict_canonical", type=str, default=str(DEFAULT_ICT_CANONICAL))
    parser.add_argument("--flame_model", type=str, default=str(DEFAULT_FLAME_MODEL))
    parser.add_argument("--flame_mesh", type=str, default="")
    parser.add_argument("--flame_uv_mesh", type=str, default=str(DEFAULT_FLAME_UV_MESH))
    parser.add_argument("--ict_mesh", type=str, default="")
    parser.add_argument("--texture_size", type=int, default=2048)
    parser.add_argument("--n_shape", type=int, default=100)
    parser.add_argument("--n_exp", type=int, default=50)
    parser.add_argument(
        "--use_processed_faces",
        action="store_true",
        help="FLAME processed face list (8090 tris). Default off — matches flame_static_embedding.pkl",
    )
    parser.add_argument(
        "--no_processed_faces",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--nicp_iterations", type=int, default=300, help="Stage3 vertex iters (if staged)")
    parser.add_argument("--nicp_stage1_iters", type=int, default=150, help="jaw + s,R,T blendshape iters")
    parser.add_argument("--nicp_stage2_iters", type=int, default=400, help="jaw + identity + s,R,T iters")
    parser.add_argument("--nicp_stage3_iters", type=int, default=-1, help="-1: use --nicp_iterations; 0: skip vertex")
    parser.add_argument("--nicp_lr", type=float, default=1e-2)
    parser.add_argument("--nicp_stage1_lr", type=float, default=5e-3)
    parser.add_argument("--nicp_stage2_lr", type=float, default=1e-2)
    parser.add_argument("--lambda_large_steps", type=float, default=10.0)
    parser.add_argument("--nicp_w_idt_reg", type=float, default=0.05)
    parser.add_argument("--nicp_w68", type=float, default=100.0, help="NICP inner 68[17:] landmark L1 weight")
    parser.add_argument(
        "--nicp_w_jaw",
        type=float,
        default=30.0,
        help="NICP Multi-PIE jawline 0:16 KNN-to-FLAME-mesh weight (MP has no jaw contour)",
    )
    parser.add_argument(
        "--nicp_w_chamfer",
        type=float,
        default=0.25,
        help="NICP bidirectional chamfer (ICT↔FLAME) weight; weak global shape term",
    )
    parser.add_argument("--nicp_w_surf", type=float, default=1.0, help="NICP one-way surface knn weight")
    parser.add_argument("--nicp_w_normal", type=float, default=0.1)
    parser.add_argument("--nicp_w_edge", type=float, default=1.0)
    parser.add_argument("--landmark_start", type=int, default=17)
    parser.add_argument(
        "--skip_flame_similarity",
        action="store_true",
        help="Skip coarse+jaw and rigid FLAME alignment (no flame_alignment_* write)",
    )
    parser.add_argument(
        "--skip_save_alignment",
        action="store_true",
        help="Apply alignment for bake but do not write flame_alignment_* back to ict_npy",
    )
    parser.add_argument(
        "--coarse_st_only",
        action="store_true",
        help="Use npy coarse jaw+s,T only (ignore flame_alignment_s,R,T)",
    )
    parser.add_argument(
        "--ignore_stored_flame_alignment",
        action="store_true",
        help="Do not use/write flame_alignment_* from npy (recompute only with --recompute_flame_alignment)",
    )
    parser.add_argument(
        "--recompute_flame_alignment",
        action="store_true",
        help="Re-fit flame_alignment_s,R,T even if already in npy (stale/bad npy)",
    )
    parser.add_argument(
        "--no_flame_canonical_pose",
        action="store_true",
        help="FLAME zero pose (default: canonical jaw-open, matches npy similarity fit)",
    )
    parser.add_argument("--skip_nicp", action="store_true", help="Skip face NICP (use ICT neutral as fit)")
    parser.add_argument("--skip_eye_transplant", action="store_true", help="Skip eye rigid iris bake")
    parser.add_argument("--eye_rigid_iters", type=int, default=300)
    parser.add_argument("--eye_rigid_lr", type=float, default=1e-2)
    parser.add_argument("--eye_w_chamfer", type=float, default=1.0)
    parser.add_argument("--eye_w_anchor", type=float, default=200.0)
    parser.add_argument(
        "--no_texture_viz",
        action="store_true",
        help="Skip ICT/FLAME landmark UV texture PNGs (default: export after bake)",
    )
    parser.add_argument(
        "--no_export_debug",
        action="store_true",
        help="Skip OBJ/PLY debug meshes (texture viz still runs unless --no_texture_viz)",
    )
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu")
    if device.type == "cpu":
        print("Warning: Large Steps laplacian uses CUDA internally; use GPU for NICP.")

    v_ict, f_ict, ict_lmk, regions, ict_uvs, ict_uv_faces, applied_sim, ict_npy_dict = load_ict_from_npy(args)
    use_proc, use_pose = flame_flags_from_npy(ict_npy_dict, args)
    v_flame, f_flame = load_flame_canonical(args, device, ict_npy_dict)
    mp_embedding = args.mp_embedding or str(
        Path(args.metrical_root) / "flame" / "mediapipe" / "mediapipe_landmark_embedding.npz"
    )
    assert_flame_mp_embedding(mp_embedding, len(f_flame))

    print(
        f"FLAME F={len(f_flame)} use_processed_faces={use_proc} canonical_pose={use_pose} | "
        f"ICT variant={regions['asset_variant']} schema={regions['asset_schema_version']} "
        f"V={v_ict.shape[0]} F={len(f_ict)}"
    )
    jaw = ict_npy_dict.get("flame_similarity_ict_jaw_open", 0.75)
    use_rigid = False
    if applied_sim:
        ict_npy_dict = ensure_flame_alignment(args, ict_npy_dict, ict_lmk, device)
        from processing.ict_flame_similarity import (
            apply_ict_to_flame_space,
            has_flame_alignment,
            print_flame_alignment_report,
        )

        use_rigid = has_flame_alignment(ict_npy_dict) and not args.coarse_st_only
        v_ict = apply_ict_to_flame_space(
            np.asarray(ict_npy_dict["neutral_mesh"], dtype=np.float64),
            ict_npy_dict,
            use_final_alignment=use_rigid,
        )
        print(
            f"ICT in FLAME space: jawOpen={float(jaw):.4f} "
            f"transform={'jaw+s,R,T' if use_rigid else 'jaw+s,T'}"
        )
        err = float(
            ict_npy_dict.get(
                "flame_alignment_lmk_err_mean",
                ict_npy_dict.get("flame_similarity_lmk_err_mean", 1.0),
            )
        )
        if err > 0.05:
            print(
                f"WARNING: npy inner landmark error {err:.4f} m is high — "
                "re-run ict_facekit_to_npy_full_head.py"
            )
        if "flame_similarity_n_pairs" in ict_npy_dict:
            print_flame_alignment_report(
                np.asarray(ict_npy_dict["neutral_mesh"], dtype=np.float64),
                ict_lmk,
                ict_npy_dict["expression_shape_modes"],
                ict_npy_dict["expression_names"],
                ict_npy_dict,
                flame_model_path=args.flame_model,
                flame_lmk_embedding_path=args.flame_lmk_embedding,
                landmark_start=int(
                    ict_npy_dict.get("flame_similarity_landmark_start", args.landmark_start)
                ),
                device=str(device),
            )
    else:
        print("ICT in local space (flame alignment not applied)")
    eye_mirror = bool(ict_npy_dict.get("eye_uv_mirror_right_u", False))
    print(
        f"eye_uv_mirror_right_u={eye_mirror} "
        f"(EyeTextureGaussians; iris bake is 3D eye NICP, independent of UV mirror)"
    )

    flame_lmk_face_idx, flame_lmk_bary = load_flame_static_embedding(args.flame_lmk_embedding)

    debug_dir = Path(args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    export_initial_alignment_debug(
        debug_dir,
        v_ict,
        f_ict,
        v_flame,
        f_flame,
        ict_lmk,
        flame_lmk_face_idx,
        flame_lmk_bary,
        ict_uvs=ict_uvs,
        ict_uv_faces=ict_uv_faces,
        ict_npy_dict=ict_npy_dict,
        flame_model_path=args.flame_model,
        flame_uv_mesh=args.flame_uv_mesh or None,
        texture_size=args.texture_size,
    )

    if args.skip_nicp:
        v_ict_fit = v_ict.copy()
    else:
        s3 = None if args.nicp_stage3_iters < 0 else args.nicp_stage3_iters
        v_ict_fit, _ = fit_ict_face_to_flame(
            v_ict,
            f_ict,
            v_flame,
            f_flame,
            ict_lmk,
            flame_lmk_face_idx,
            flame_lmk_bary,
            args.large_steps_root,
            device,
            ict_npy_dict=ict_npy_dict,
            iterations=args.nicp_iterations,
            lr=args.nicp_lr,
            lambda_large_steps=args.lambda_large_steps,
            w68=args.nicp_w68,
            w_jaw=args.nicp_w_jaw,
            wsurf=args.nicp_w_surf,
            w_chamfer_bidir=args.nicp_w_chamfer,
            wnormal=args.nicp_w_normal,
            wedge=args.nicp_w_edge,
            landmark_start=int(
                ict_npy_dict.get("flame_similarity_landmark_start", args.landmark_start)
            ),
            skip_rigid_init=applied_sim and use_rigid,
            stage1_iters=args.nicp_stage1_iters,
            stage1_lr=args.nicp_stage1_lr,
            stage2_iters=args.nicp_stage2_iters,
            stage2_lr=args.nicp_stage2_lr,
            stage3_iters=s3,
            w_idt_reg=args.nicp_w_idt_reg,
            jaw_init=float(jaw),
        )

    eye = np.asarray(regions["eyeball_indices"], dtype=np.int64)
    if eye.size > 0:
        drift = np.linalg.norm(v_ict_fit[eye] - v_ict[eye], axis=1).max()
        print(f"eyeball NICP drift (max vertex displacement): {drift:.6f} (expect ~0)")

    export_nicp_fit_68_debug(
        debug_dir,
        v_ict_fit,
        f_ict,
        ict_lmk,
        ict_uvs=ict_uvs,
        ict_uv_faces=ict_uv_faces,
        ict_npy_dict=ict_npy_dict,
        texture_size=args.texture_size,
    )

    mp_pack = build_flame_mp_points(v_flame, f_flame, mp_embedding)
    face_embedding = transfer_mediapipe_to_ict(
        mp_pack,
        v_ict_fit,
        f_ict,
        regions["face_indices"],
        regions["eyeball_indices"],
        left_eyeball_indices=regions["left_eyeball_indices"],
        right_eyeball_indices=regions["right_eyeball_indices"],
        eye_socket_left_indices=regions["eye_socket_left_indices"],
        eye_socket_right_indices=regions["eye_socket_right_indices"],
        surface_sample_vertex_indices=regions["surface_sample_vertex_indices"],
        skin_face_indices=regions["skin_face_indices"],
        head_neck_indices=regions.get("head_neck_indices") or regions.get("not_face_indices"),
        mouth_socket_indices=regions["mouth_socket_indices"],
        gums_tongue_indices=regions["gums_tongue_indices"],
    )

    if args.skip_eye_transplant:
        from processing.ict_mediapipe_lmk.iris_ict import bake_iris_landmarks_ict

        print("WARNING: --skip_eye_transplant uses legacy ict-native iris bake (no FLAME eye NICP)")
        iris_bake = bake_iris_landmarks_ict(
            v_ict_fit,
            f_ict,
            regions["left_eyeball_indices"],
            regions["right_eyeball_indices"],
        )
        eye_embedding = {
            "mp_landmark_indices": np.concatenate([iris_bake["left_iris_mp"], iris_bake["right_iris_mp"]]),
            "ict_lmk_face_idx": np.concatenate([iris_bake["left_face_idx"], iris_bake["right_face_idx"]]),
            "ict_lmk_b_coords": np.concatenate([iris_bake["left_bary"], iris_bake["right_bary"]], axis=0),
            "transfer_error": np.concatenate([iris_bake["left_error"], iris_bake["right_error"]]),
            "ict_lmk_target_type": np.array(["left_iris"] * 5 + ["right_iris"] * 5, dtype=object),
            "source": np.array(["ict_eyeball_native"] * 10, dtype=object),
            "geometry_chart_id": np.array([1] * 5 + [2] * 5, dtype=np.int32),
        }
        embedding = merge_iris_into_embedding(face_embedding, eye_embedding)
    else:
        eye_embedding = run_eye_transplant(
            v_flame,
            f_flame,
            v_ict_fit,
            f_ict,
            regions,
            device,
            ict_npy_dict=ict_npy_dict,
            eye_rigid_iters=args.eye_rigid_iters,
            eye_rigid_lr=args.eye_rigid_lr,
            eye_w_chamfer=args.eye_w_chamfer,
            eye_w_anchor=args.eye_w_anchor,
        )
        embedding = merge_iris_into_embedding(face_embedding, eye_embedding)
        if not args.no_export_debug:
            export_eye_fitting_debug(debug_dir, eye_embedding)

    validate_embedding(embedding, regions)

    out_path = Path(args.output)
    save_ict_mediapipe_embedding(out_path, embedding)
    print(f"Saved {out_path} ({len(embedding['mp_landmark_indices'])} landmarks, 3 arrays)")

    aux_path = Path(args.debug_dir) / "ict_mediapipe_bake_aux.npz"
    save_ict_mediapipe_embedding_aux(aux_path, embedding, v_ict_fit, f_ict, regions)
    print(f"Saved aux debug: {aux_path}")

    if not args.no_texture_viz:
        tex_out = export_landmark_texture_qa(
            debug_dir,
            v_ict_fit,
            f_ict,
            ict_uvs,
            ict_uv_faces,
            embedding,
            v_flame=v_flame,
            f_flame=f_flame,
            mp_embedding_path=mp_embedding,
            flame_model_path=args.flame_model,
            flame_uv_mesh=args.flame_uv_mesh or None,
            ict_npy_dict=ict_npy_dict,
            texture_size=args.texture_size,
        )
        print(f"ICT landmark texture (face, no teeth/gums): {tex_out['ict_texture']}")
        print(f"FLAME landmark texture: {tex_out['flame_texture']}")
        print(f"Side-by-side panel:    {tex_out['comparison_png']}")
        charts = tex_out.get("ict_texture_charts") or {}
        for mat, info in charts.items():
            print(f"  per-map [{mat}]: {info['texture']}")
        if "eyeball_iris_comparison_png" in tex_out:
            print(f"Eyeball iris×5 panel:  {tex_out['eyeball_iris_comparison_png']}")

    if not args.no_export_debug:
        export_debug(
            debug_dir,
            v_flame,
            f_flame,
            v_ict_fit,
            f_ict,
            mp_pack,
            embedding,
            mp_embedding_path=mp_embedding,
            flame_uv_mesh=args.flame_uv_mesh or None,
            flame_model_path=args.flame_model,
            ict_uvs=ict_uvs,
            ict_uv_faces=ict_uv_faces,
            ict_npy_dict=ict_npy_dict,
            texture_size=args.texture_size,
            export_flame_texture=False,
        )
        np.savez(
            debug_dir / "ict_nicp_fit.npz",
            v_ict_fit=v_ict_fit.astype(np.float32),
            ict_faces=f_ict.astype(np.int64),
        )


if __name__ == "__main__":
    main()
