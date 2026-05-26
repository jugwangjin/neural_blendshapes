"""
Export ICT-FaceKit ``generic_neutral_mesh.obj`` as triangle mesh per topology part.

Part vertex ranges follow the official ICT-FaceKit README (parts #0–#16):
https://github.com/USC-ICT/ICT-FaceKit

Run from repo root (requires ICT_FaceKit clone + openmesh):
  python processing/ict_save_parts.py
  python processing/ict_save_parts.py --out debugs/ict_parts_full
  python processing/ict_save_parts.py --mesh /path/to/FaceXModel/generic_neutral_mesh.obj
  python processing/ict_save_parts.py --with-blendshapes   # also write .npy per part
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import openmesh as om

_PROCESSING_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _PROCESSING_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from processing.ict_flame_similarity import expression_modes_as_vertex_deltas
from processing.ict_mediapipe_lmk.io_debug import write_mesh
from processing.ict_obj_materials import convert_quad_mesh_to_triangle_mesh
from processing.ict_region_dict import (
    OFFICIAL_FULL_PART_SPLITS,
    VERTEX_COUNT_FULL,
    vertex_parts_from_splits,
)
from processing.paths import setup_ict_facekit_import

# https://github.com/USC-ICT/ICT-FaceKit — Face Model Topology table
FULL_PART_NAMES = (
    "0_face",
    "1_head_neck",
    "2_mouth_socket",
    "3_eye_socket_L",
    "4_eye_socket_R",
    "5_gums_tongue",
    "6_teeth",
    "7_eyeball_L",
    "8_eyeball_R",
    "9_lacrimal_fluid_L",
    "10_lacrimal_fluid_R",
    "11_eye_blend_L",
    "12_eye_blend_R",
    "13_eye_occlusion_L",
    "14_eye_occlusion_R",
    "15_eyelashes_L",
    "16_eyelashes_R",
)

GLOBAL_KEYS_COPY = (
    "num_expression",
    "num_identity",
    "expression_names",
    "identity_names",
    "model_config",
)


def part_name(part_id: int) -> str:
    if 0 <= part_id < len(FULL_PART_NAMES):
        return FULL_PART_NAMES[part_id]
    return f"part_{part_id}"


def load_generic_neutral_triangle_mesh(obj_path: Path):
    """
    Read ICT ``generic_neutral_mesh.obj`` (quad polymesh) → triangle mesh.

    Returns vertices [V,3], faces [F,3], vertex_parts [V], quad_face_count.
    """
    mesh = om.read_polymesh(str(obj_path), halfedge_tex_coord=True)
    quad_faces = mesh.face_vertex_indices()
    vertices = np.asarray(mesh.points(), dtype=np.float64)
    tex_coords = mesh.halfedge_texcoords2D()
    uv_quads = tex_coords[mesh.face_halfedge_indices()]
    tri_faces, _ = convert_quad_mesh_to_triangle_mesh(quad_faces, uv_quads, n_verts=vertices.shape[0])

    n_verts = int(vertices.shape[0])
    splits = OFFICIAL_FULL_PART_SPLITS
    if n_verts != VERTEX_COUNT_FULL:
        splits = [s for s in splits if s <= n_verts]
        if splits[-1] != n_verts:
            splits = splits + [n_verts]
        print(f"  note: V={n_verts} (README full={VERTEX_COUNT_FULL}) — using truncated splits")

    vertex_parts = vertex_parts_from_splits(n_verts, splits)
    return vertices, tri_faces.astype(np.int64), vertex_parts, int(len(quad_faces))


def vertex_ids_for_part(vertex_parts, part_id: int) -> np.ndarray:
    vp = np.asarray(vertex_parts, dtype=np.int64)
    return np.where(vp == int(part_id))[0].astype(np.int64)


def extract_part_triangle_mesh(vertices, faces, keep_vids: np.ndarray):
    keep_vids = np.asarray(keep_vids, dtype=np.int64)
    vid_set = set(int(v) for v in keep_vids.tolist())
    g2l = {int(g): i for i, g in enumerate(keep_vids.tolist())}

    faces = np.asarray(faces, dtype=np.int64)
    keep_mask = np.array([all(int(v) in vid_set for v in tri) for tri in faces], dtype=bool)
    f_sub = faces[keep_mask]
    new_faces = np.vectorize(lambda v: g2l[int(v)], otypes=[np.int64])(f_sub)
    new_verts = np.asarray(vertices, dtype=np.float64)[keep_vids]
    return new_verts, new_faces.astype(np.int64), keep_mask


def subset_vertex_modes(modes, keep_vids: np.ndarray, n_verts_full: int):
    if modes is None:
        return None
    m = expression_modes_as_vertex_deltas(modes, n_verts_full)
    return np.asarray(m[:, keep_vids, :], dtype=np.float32)


def attach_blendshapes(full: dict, facex_dir: Path):
    from ICT_FaceKit.Scripts import face_model_io

    ict_model = face_model_io.load_face_model(str(facex_dir))
    n = int(np.asarray(full["neutral_mesh"]).shape[0])
    full["expression_shape_modes"] = np.asarray(
        ict_model._expression_shape_modes[:, :n], dtype=np.float32
    )
    full["identity_shape_modes"] = np.asarray(
        ict_model._identity_shape_modes[:, :n], dtype=np.float32
    )
    full["expression_names"] = ict_model._expression_names
    full["identity_names"] = ict_model._identity_names
    full["num_expression"] = int(full["expression_shape_modes"].shape[0])
    full["num_identity"] = int(full["identity_shape_modes"].shape[0])
    full["model_config"] = ict_model._model_config


def build_part_record(full: dict, part_id: int) -> dict | None:
    keep_vids = vertex_ids_for_part(full["vertex_parts"], part_id)
    if keep_vids.size == 0:
        return None

    n_verts_full = int(np.asarray(full["neutral_mesh"]).shape[0])
    verts, faces, _ = extract_part_triangle_mesh(full["neutral_mesh"], full["faces"], keep_vids)

    out = {
        "part_id": int(part_id),
        "part_name": part_name(part_id),
        "source_vertex_count": n_verts_full,
        "global_vertex_ids": keep_vids.astype(np.int64),
        "neutral_mesh": verts.astype(np.float32),
        "faces": faces.astype(np.int64),
        "vertex_count": int(verts.shape[0]),
        "face_count": int(faces.shape[0]),
        "vertex_parts": [int(part_id)] * int(verts.shape[0]),
        "parts_split": list(full.get("parts_split", OFFICIAL_FULL_PART_SPLITS)),
    }

    for key in GLOBAL_KEYS_COPY:
        if key in full:
            out[key] = full[key]

    if "expression_shape_modes" in full:
        out["expression_shape_modes"] = subset_vertex_modes(
            full["expression_shape_modes"], keep_vids, n_verts_full
        )
    if "identity_shape_modes" in full:
        out["identity_shape_modes"] = subset_vertex_modes(
            full["identity_shape_modes"], keep_vids, n_verts_full
        )
    return out


def save_part_meshes(
    full: dict,
    out_dir: Path,
    part_ids=None,
    write_npy: bool = False,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vp = np.asarray(full["vertex_parts"], dtype=np.int64)
    available = sorted(int(x) for x in np.unique(vp).tolist())
    targets = available if part_ids is None else [int(p) for p in part_ids if int(p) in available]

    manifest = {
        "source": str(full.get("source_mesh", "")),
        "source_vertex_count": int(np.asarray(full["neutral_mesh"]).shape[0]),
        "source_triangle_count": int(np.asarray(full["faces"]).shape[0]),
        "source_quad_face_count": int(full.get("quad_face_count", 0)),
        "parts_split": list(full.get("parts_split", OFFICIAL_FULL_PART_SPLITS)),
        "reference": "https://github.com/USC-ICT/ICT-FaceKit",
        "parts": [],
    }

    for pid in targets:
        part = build_part_record(full, pid)
        if part is None:
            continue
        stem = f"{pid:02d}_{part['part_name']}"
        obj_path = out_dir / f"{stem}.obj"
        write_mesh(obj_path, part["neutral_mesh"], part["faces"])

        entry = {
            "part_id": pid,
            "part_name": part["part_name"],
            "n_vertices": part["vertex_count"],
            "n_triangles": part["face_count"],
            "global_vertex_range": [
                int(part["global_vertex_ids"].min()),
                int(part["global_vertex_ids"].max()),
            ],
            "obj": obj_path.name,
        }

        if write_npy:
            npy_path = out_dir / f"{stem}.npy"
            np.save(npy_path, part, allow_pickle=True)
            entry["npy"] = npy_path.name

        manifest["parts"].append(entry)
        print(
            f"  {pid:02d} {part['part_name']:22s}  "
            f"V={part['vertex_count']:5d}  F={part['face_count']:5d}  -> {obj_path.name}"
        )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {len(manifest['parts'])} parts -> {out_dir}")
    print(f"manifest: {manifest_path}")
    return manifest


def parse_part_list(text: str):
    if not text:
        return None
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def resolve_mesh_path(mesh_arg: Path | None, facex_dir: Path | None) -> Path:
    if mesh_arg is not None:
        return Path(mesh_arg)
    if facex_dir is not None:
        return Path(facex_dir) / "generic_neutral_mesh.obj"
    _, facex = setup_ict_facekit_import()
    return Path(facex) / "generic_neutral_mesh.obj"


def main():
    parser = argparse.ArgumentParser(
        description="ICT generic_neutral_mesh → triangle .obj per part (#0–#16)"
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        default=None,
        help="Path to generic_neutral_mesh.obj (default: ICT_FaceKit/FaceXModel/...)",
    )
    parser.add_argument(
        "--facex-dir",
        type=Path,
        default=None,
        help="ICT_FaceKit FaceXModel directory (for default mesh path + optional blendshapes)",
    )
    parser.add_argument("--out", type=Path, default=_REPO_ROOT / "debugs" / "ict_parts_full")
    parser.add_argument(
        "--parts",
        type=str,
        default="",
        help="Comma-separated part ids (default: all #0–#16)",
    )
    parser.add_argument(
        "--with-blendshapes",
        action="store_true",
        help="Also save .npy per part (expression/identity modes from face_model_io)",
    )
    args = parser.parse_args()

    facex_dir = args.facex_dir
    if facex_dir is None and args.mesh is None:
        _, facex_dir = setup_ict_facekit_import()
        facex_dir = Path(facex_dir)

    mesh_path = resolve_mesh_path(args.mesh, facex_dir)
    print(f"loading: {mesh_path}")
    vertices, faces, vertex_parts, n_quad = load_generic_neutral_triangle_mesh(mesh_path)
    print(
        f"  V={vertices.shape[0]}  tri_F={faces.shape[0]}  quad_F={n_quad}  "
        f"parts={len(set(vertex_parts))}"
    )

    full = {
        "source_mesh": str(mesh_path),
        "neutral_mesh": vertices,
        "faces": faces,
        "vertex_parts": vertex_parts,
        "parts_split": OFFICIAL_FULL_PART_SPLITS,
        "quad_face_count": n_quad,
    }

    if args.with_blendshapes:
        if facex_dir is None:
            _, facex_dir = setup_ict_facekit_import()
            facex_dir = Path(facex_dir)
        print(f"loading blendshapes: {facex_dir}")
        attach_blendshapes(full, facex_dir)

    part_ids = parse_part_list(args.parts)
    save_part_meshes(full, args.out, part_ids=part_ids, write_npy=args.with_blendshapes)


if __name__ == "__main__":
    main()
