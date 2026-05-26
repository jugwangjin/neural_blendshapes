"""
Check MP embedding face indices against current ict_facekit_torch.npy topology.

Run from repo root (GPU cluster):
  python debug/verify_mp_embedding_mesh_consistency.py
  python debug/verify_mp_embedding_mesh_consistency.py --aux processing/ict_mediapipe_lmk/debug/ict_mediapipe_bake_aux.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Config
from processing.ict_mediapipe_lmk.embedding_io import resolve_embedding_path
from losses.mediapipe_landmark_478 import load_mediapipe_ict_embedding
from model.ict_model import ICTFaceKitTorch
from processing.ict_mediapipe_lmk.landmarks import sample_bary
from utils.barycentric import vertices2landmarks


def mesh_canonical_jaw(ict, device):
    exp = torch.zeros(1, ict.num_expression, device=device)
    exp[0, ict.jaw_index] = float(ict.expression[0, ict.jaw_index].item())
    return ict.forward(
        expression_weights=exp,
        apply_flame_similarity=True,
        apply_eyeball_rotation=False,
    )[0]


def sample_mp_points(verts, faces, emb):
    fi = np.asarray(emb["ict_lmk_face_idx"], dtype=np.int64)
    bary = np.asarray(emb["ict_lmk_b_coords"], dtype=np.float64)
    v = verts.detach().cpu().numpy() if torch.is_tensor(verts) else np.asarray(verts)
    f = faces.detach().cpu().numpy() if torch.is_tensor(faces) else np.asarray(faces)
    return sample_bary(v, f, fi, bary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aux", type=Path, default=None, help="ict_mediapipe_bake_aux.npz (v_ict_fit + ict_faces)")
    args = parser.parse_args()

    cfg = Config()
    cfg.mp_embedding = resolve_embedding_path(cfg.mp_embedding)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ict = ICTFaceKitTorch(npy_dir=str(cfg.ict_npy)).to(device)
    faces = ict.faces
    n_f = int(faces.shape[0])
    n_v = int(ict.vertex_count)

    emb = load_mediapipe_ict_embedding(cfg.mp_embedding)
    fi = np.asarray(emb["ict_lmk_face_idx"], dtype=np.int64)
    mx_fi = int(fi.max())
    mn_fi = int(fi.min())
    print("=== NPZ vs npy topology ===")
    has_nicp_bake = bool(getattr(ict, "has_nicp_bake_mesh", False))
    print(f"  mp_embedding: {cfg.mp_embedding}")
    print(f"  npy: V={n_v}  F={n_f}  variant={ict.asset_variant}  schema={ict.asset_schema_version}")
    print(f"  nicp_canonical_mesh in npy (bake reference only): {'yes' if has_nicp_bake else 'no'}")
    print(f"  embedding: L={len(fi)}  face_idx in [{mn_fi}, {mx_fi}]")
    if mx_fi >= n_f:
        print(f"  FAIL: max face_idx {mx_fi} >= F={n_f} — rebake embedding on this npy")
        return
    print("  face_idx range: OK")

    tri = faces[fi].reshape(-1)
    mx_v = int(tri.max().item())
    if mx_v >= n_v:
        print(f"  FAIL: max corner vertex {mx_v} >= V={n_v}")
        return
    print("  corner vertices: OK")

    v_canon = mesh_canonical_jaw(ict, device)
    pts_canon = sample_mp_points(v_canon, faces, emb)
    print(f"\n=== Sample on runtime template (jawOpen + rigid flame_alignment) ===")
    print(f"  points: {pts_canon.shape[0]}  bbox extent {np.ptp(pts_canon, axis=0).max():.4f}")

    if args.aux is not None and args.aux.is_file():
        aux = np.load(args.aux, allow_pickle=True)
        v_fit = aux["v_ict_fit"]
        f_aux = aux["ict_faces"]
        if len(f_aux) != n_f:
            print(f"\n  WARNING: aux F={len(f_aux)} != npy F={n_f}")
        pts_fit = sample_bary(v_fit, f_aux, fi, np.asarray(emb["ict_lmk_b_coords"]))
        dist = np.linalg.norm(pts_canon - pts_fit, axis=1)
        print(f"\n=== Bake mesh (aux v_ict_fit) vs runtime canonical ===")
        print(f"  mean dist: {dist.mean():.6f}  max: {dist.max():.6f}  p95: {np.percentile(dist, 95):.6f}")
        print(
            "  (gap is expected: embedding on NICP bake mesh; train template