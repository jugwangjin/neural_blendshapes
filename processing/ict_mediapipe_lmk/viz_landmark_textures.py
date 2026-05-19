"""
Visual QA: MediaPipe landmarks on UV textures — ICT (baked) vs FLAME (metrical), side by side.

  python processing/ict_mediapipe_lmk/viz_landmark_textures.py \\
    --embedding assets/ict_mediapipe_landmark_indices.npz \\
    --ict_npy assets/ict_facekit_torch.npy \\
    --out_dir processing/ict_mediapipe_lmk/debug
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
    DEFAULT_FLAME_MODEL,
    DEFAULT_ICT_NPY,
    DEFAULT_METRICAL_ROOT,
    DEFAULT_OUTPUT_NPZ,
)
from processing.ict_mediapipe_lmk.texture_viz import (
    embedding_dict_from_npz,
    export_landmark_texture_comparison,
)

setup_import_paths()


def load_ict_mesh(ict_npy, embedding_npz, debug_dir, apply_flame_similarity=True):
    from processing.ict_npy_loader import load_ict_asset

    v, f, uvs, uv_faces, _, _, d = load_ict_asset(
        ict_npy, apply_flame_similarity_transform=apply_flame_similarity
    )
    aux = Path(debug_dir) / "ict_mediapipe_bake_aux.npz"
    if aux.is_file():
        z = np.load(aux, allow_pickle=True)
        v = np.asarray(z["v_ict_fit"], dtype=np.float64)
        f = np.asarray(z["ict_faces"], dtype=np.int64)
    else:
        z = np.load(embedding_npz, allow_pickle=True)
        if "v_ict_fit" in z:
            v = np.asarray(z["v_ict_fit"], dtype=np.float64)
        if "ict_faces" in z:
            f = np.asarray(z["ict_faces"], dtype=np.int64)
    return v, f, uvs, uv_faces, d


def load_flame_canonical(flame_model, use_processed_faces, use_canonical_pose, device):
    from processing.flame.flame_viz import load_flame_canonical_mesh

    return load_flame_canonical_mesh(
        flame_model,
        use_processed_faces=use_processed_faces,
        use_canonical_pose=use_canonical_pose,
        device=device,
    )


def main():
    parser = argparse.ArgumentParser(description="ICT vs FLAME landmark UV texture comparison")
    parser.add_argument("--embedding", type=str, default=str(DEFAULT_OUTPUT_NPZ))
    parser.add_argument("--ict_npy", type=str, default=str(DEFAULT_ICT_NPY))
    parser.add_argument("--out_dir", type=str, default=str(_PKG / "debug"))
    parser.add_argument("--texture_size", type=int, default=2048)
    parser.add_argument("--no_flame", action="store_true", help="ICT texture only (no FLAME / comparison)")
    parser.add_argument("--flame_model", type=str, default=str(DEFAULT_FLAME_MODEL))
    parser.add_argument("--metrical_root", type=str, default=str(DEFAULT_METRICAL_ROOT))
    parser.add_argument(
        "--mp_embedding",
        type=str,
        default="",
        help="metrical MP npz (landmarks only); mesh/UV from processing/flame",
    )
    parser.add_argument(
        "--flame_uv_mesh",
        type=str,
        default="",
        help="flare-topology OBJ with vt (else assets/flame_head_uv.obj)",
    )
    parser.add_argument(
        "--use_processed_faces",
        action="store_true",
        help="FLAME processed topology (~8090F)",
    )
    parser.add_argument("--no_flame_canonical_pose", action="store_true")
    parser.add_argument("--skip_flame_similarity", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    embedding = embedding_dict_from_npz(args.embedding)
    v_ict, f_ict, uvs, uv_faces, ict_npy_dict = load_ict_mesh(
        args.ict_npy,
        args.embedding,
        args.out_dir,
        apply_flame_similarity=not args.skip_flame_similarity,
    )

    mp_emb = args.mp_embedding or str(
        Path(args.metrical_root) / "flame" / "mediapipe" / "mediapipe_landmark_embedding.npz"
    )

    if args.no_flame:
        from processing.ict_mediapipe_lmk.texture_viz import export_ict_mediapipe_texture

        ict_obj, ict_tex = export_ict_mediapipe_texture(
            out_dir,
            v_ict,
            f_ict,
            uvs,
            uv_faces,
            embedding,
            size=args.texture_size,
            ict_npy_dict=ict_npy_dict,
        )
        print(f"ICT QA: {ict_obj} | {ict_tex}")
        return

    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu"
    )
    v_flame, f_flame = load_flame_canonical(
        args.flame_model,
        args.use_processed_faces,
        not args.no_flame_canonical_pose,
        device,
    )

    out = export_landmark_texture_comparison(
        out_dir,
        v_ict,
        f_ict,
        uvs,
        uv_faces,
        embedding,
        v_flame=v_flame,
        f_flame=f_flame,
        mp_embedding_path=mp_emb,
        flame_model_path=args.flame_model,
        flame_uv_mesh=args.flame_uv_mesh or None,
        ict_npy_dict=ict_npy_dict,
        size=args.texture_size,
        export_flame=True,
    )
    print(f"ICT:  {out['ict_texture']}")
    print(f"FLAME: {out['flame_texture']}")
    print(f"Compare: {out['comparison_png']}")
    print(f"({len(embedding['mp_landmark_indices'])} ICT landmarks)")


if __name__ == "__main__":
    main()
