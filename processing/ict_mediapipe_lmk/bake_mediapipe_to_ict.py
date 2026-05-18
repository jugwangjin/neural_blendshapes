"""
Bake MediaPipe landmarks from metrical-tracker FLAME embedding onto ICT-FaceKit.

Pipeline:
  A. NICP: ICT face patch -> FLAME canonical (Large Steps)
  B. Sample FLAME MediaPipe points (metrical-tracker npz + iris hardcoded)
  C. Project to fitted ICT mesh -> mp_index / ict_face_idx / bary_coords
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

from flame.FLAME import FLAME
from model.ict_model import ICTFaceKitTorch

from ict_mediapipe_lmk.constants import (
    DEFAULT_DEBUG_DIR,
    DEFAULT_FLAME_LMK_EMBEDDING,
    DEFAULT_FLAME_MODEL,
    DEFAULT_FLAME_UV_MESH,
    DEFAULT_ICT_CANONICAL,
    DEFAULT_ICT_NPY,
    DEFAULT_LARGE_STEPS_ROOT,
    DEFAULT_METRICAL_ROOT,
    DEFAULT_MP_EMBEDDING,
    DEFAULT_OUTPUT_NPZ,
)
from ict_mediapipe_lmk.io_debug import export_debug
from ict_mediapipe_lmk.metrical import build_flame_mp_points, load_flame_static_embedding
from ict_mediapipe_lmk.nicp import fit_ict_face_to_flame
from ict_mediapipe_lmk.transfer import transfer_mediapipe_to_ict


def load_flame_canonical(args, device):
    if args.flame_mesh:
        import trimesh

        mesh = trimesh.load_mesh(args.flame_mesh, process=False)
        return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)

    shape = torch.zeros(1, args.n_shape)
    flame = FLAME(
        args.flame_model,
        n_shape=args.n_shape,
        n_exp=args.n_exp,
        shape_params=shape,
        use_processed_faces=not args.no_processed_faces,
    ).to(device)
    exp = torch.zeros(1, args.n_exp, device=device)
    pose = torch.zeros(1, 15, device=device)
    verts, _, _ = flame(expression_params=exp, full_pose=pose)
    faces = flame.faces_tensor.cpu().numpy()
    return verts[0].cpu().numpy(), faces


def load_ict_canonical(args, device):
    ict = ICTFaceKitTorch(npy_dir=args.ict_npy, canonical=args.ict_canonical).to(device)
    if args.ict_mesh:
        import trimesh

        mesh = trimesh.load_mesh(args.ict_mesh, process=False)
        v_ict = np.asarray(mesh.vertices, dtype=np.float64)
    else:
        v_ict = ict.neutral_mesh[0].cpu().numpy()
    f_ict = ict.faces.cpu().numpy()
    return (
        v_ict,
        f_ict,
        ict.landmark_indices,
        list(ict.face_indices),
        list(ict.eyeball_indices),
        ict.uvs.cpu().numpy(),
        ict.uv_faces.cpu().numpy(),
    )


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
    parser.add_argument("--no_processed_faces", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--nicp_iterations", type=int, default=300)
    parser.add_argument("--nicp_lr", type=float, default=1e-2)
    parser.add_argument("--lambda_large_steps", type=float, default=10.0)
    parser.add_argument("--landmark_start", type=int, default=17)
    parser.add_argument("--skip_nicp", action="store_true", help="Skip NICP (use ICT neutral as fit)")
    parser.add_argument("--export_debug", action="store_true", default=True)
    args = parser.parse_args()

    mp_embedding = args.mp_embedding or str(Path(args.metrical_root) / "flame" / "mediapipe" / "mediapipe_landmark_embedding.npz")

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu")
    if device.type == "cpu":
        print("Warning: Large Steps laplacian uses CUDA internally; use GPU for NICP.")

    v_flame, f_flame = load_flame_canonical(args, device)
    v_ict, f_ict, ict_lmk, face_indices, eyeball_indices, ict_uvs, ict_uv_faces = load_ict_canonical(
        args, device
    )

    flame_lmk_face_idx, flame_lmk_bary = load_flame_static_embedding(args.flame_lmk_embedding)

    if args.skip_nicp:
        v_ict_fit = v_ict.copy()
    else:
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
            iterations=args.nicp_iterations,
            lr=args.nicp_lr,
            lambda_large_steps=args.lambda_large_steps,
            landmark_start=args.landmark_start,
        )

    mp_pack = build_flame_mp_points(v_flame, f_flame, mp_embedding)
    embedding = transfer_mediapipe_to_ict(
        mp_pack,
        v_ict_fit,
        f_ict,
        face_indices,
        eyeball_indices,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        mp_landmark_indices=embedding["mp_landmark_indices"],
        ict_lmk_face_idx=embedding["ict_lmk_face_idx"],
        ict_lmk_b_coords=embedding["ict_lmk_b_coords"],
        transfer_error=embedding["transfer_error"],
        ict_lmk_target_type=embedding["ict_lmk_target_type"],
        source=embedding["source"],
        v_ict_fit=v_ict_fit.astype(np.float32),
        ict_faces=f_ict.astype(np.int64),
    )
    print(f"Saved {out_path} ({len(embedding['mp_landmark_indices'])} landmarks)")

    if args.export_debug:
        export_debug(
            args.debug_dir,
            v_flame,
            f_flame,
            v_ict_fit,
            f_ict,
            mp_pack,
            embedding,
            mp_embedding_path=mp_embedding,
            flame_uv_mesh_path=args.flame_uv_mesh,
            ict_uvs=ict_uvs,
            ict_uv_faces=ict_uv_faces,
            texture_size=args.texture_size,
        )
        np.savez(
            Path(args.debug_dir) / "ict_nicp_fit.npz",
            v_ict_fit=v_ict_fit.astype(np.float32),
            ict_faces=f_ict.astype(np.int64),
        )


if __name__ == "__main__":
    main()
