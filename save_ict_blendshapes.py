"""
Export ICT-FaceKit expression blendshapes (see flare/utils/ict_model.py).

Writes under ict_bshapes/ by default:
  - ict_blendshapes.npz: neutral, per-expression deltas, faces, names
  - expression_names.txt
  - meshes/*.obj (optional): neutral + each expression at unit weight
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

from flare.utils.ict_model import ICTFaceKitTorch


def write_obj(path, vertices, faces):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    o3d.io.write_triangle_mesh(str(path), mesh)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ict_bshapes",
        help="Output directory (default: ict_bshapes)",
    )
    parser.add_argument(
        "--npy_dir",
        type=str,
        default="./assets/ict_facekit_torch.npy",
        help="ICT-FaceKit torch asset",
    )
    parser.add_argument(
        "--canonical",
        type=str,
        default="./assets/ict_identity.npy",
        help="Canonical identity transform (used only if --to_canonical)",
    )
    parser.add_argument(
        "--weight",
        type=float,
        default=1.0,
        help="Activation weight for per-expression meshes",
    )
    parser.add_argument(
        "--to_canonical",
        action="store_true",
        help="Apply canonical transform in forward() when exporting meshes",
    )
    parser.add_argument(
        "--save_meshes",
        action="store_true",
        help="Export neutral.obj and one OBJ per expression",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ict = ICTFaceKitTorch(npy_dir=args.npy_dir, canonical=args.canonical)
    ict.eval()

    neutral = ict.neutral_mesh[0].cpu().numpy()
    deltas = ict.expression_shape_modes[0].cpu().numpy()
    faces = ict.faces.cpu().numpy()
    names = ict.expression_names.tolist()

    np.savez_compressed(
        output_dir / "ict_blendshapes.npz",
        neutral_mesh=neutral.astype(np.float32),
        expression_shape_modes=deltas.astype(np.float32),
        faces=faces.astype(np.int64),
        expression_names=np.array(names, dtype=object),
        num_expression=np.int32(ict.num_expression),
    )

    with open(output_dir / "expression_names.txt", "w", encoding="utf-8") as f:
        for i, name in enumerate(names):
            f.write(f"{i}\t{name}\n")

    if args.save_meshes:
        mesh_dir = output_dir / "meshes"
        mesh_dir.mkdir(parents=True, exist_ok=True)

        write_obj(mesh_dir / "neutral.obj", neutral, faces)

        num_exp = ict.num_expression
        identity = torch.zeros(1, ict.num_identity)
        for i, name in enumerate(names):
            expression = torch.zeros(1, num_exp)
            expression[0, i] = args.weight
            verts = ict.forward(
                expression_weights=expression,
                identity_weights=identity,
                to_canonical=args.to_canonical,
            )[0].cpu().numpy()
            write_obj(mesh_dir / f"{name}.obj", verts, faces)

    print(f"Saved {len(names)} expression blendshapes to {output_dir.resolve()}")
    print(f"  npz: {output_dir / 'ict_blendshapes.npz'}")
    if args.save_meshes:
        print(f"  meshes: {output_dir / 'meshes'}")


if __name__ == "__main__":
    main()
