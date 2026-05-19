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
    OFFICIAL_PART_SPLITS,
    VERTEX_COUNT_STANDARD,
    build_official_region_indices,
    build_region_dict,
    vertex_parts_from_splits,
)

setup_import_paths()
_, facex_dir = setup_ict_facekit_import()
from ICT_FaceKit.Scripts import face_model_io

import numpy as np
import pickle
import os
import chumpy as ch
import openmesh as om
# from ICT_FaceKit.ict_model import ICTModel

import tqdm
import trimesh
import open3d as o3d

ICT_LMK_IDX = [36, 39, 42, 45, 30, 48, 54]
FLAME_LMK_IDX = [19, 22, 25, 28, 13, 31, 37]

def load_binary_pickle( filepath ):
    with open(filepath, 'rb') as f:
        data = pickle.load(f, encoding="latin1")
    return data

def convert_quad_mesh_to_triangle_mesh(faces, uvs):
    """Converts a quad mesh represented as a faces array to a triangle mesh.

    Args:
        faces: A NumPy array of shape (F, 4), where each row represents a quad.

    Returns:
        A NumPy array of shape (F * 2, 3), where each row represents a triangle.
    """

    # Create a new triangle mesh.
    triangle_mesh = np.zeros((faces.shape[0] * 2, 3)).astype(np.int32)
    triangle_uv = np.zeros((faces.shape[0] * 2, 3, 2)).astype(np.float64)

    # For each quad in the faces array, create two triangles by splitting the quad diagonally.
    for i in range(faces.shape[0]):
        
        triangle_mesh[i * 2] = faces[i, [0, 1, 2]]
        triangle_mesh[i * 2 + 1] = faces[i, [2, 3, 0]]

        triangle_uv[i * 2] = uvs[i, [0, 1, 2]]
        triangle_uv[i * 2 + 1] = uvs[i, [2, 3, 0]]

    triangle_mesh, triangle_uv = remove_negative_triangles(triangle_mesh, triangle_uv)

    return triangle_mesh, triangle_uv

def remove_negative_triangles(triangle_mesh, triangle_uv):
    """Removes triangles from a triangle mesh that have negative (-1) elements.

    Args:
        triangle_mesh: A NumPy array of shape (F, 3), where each row represents a triangle.

    Returns:
        A NumPy array of shape (F', 3), where each row represents a triangle without negative elements.
    """
    
    positive_triangles = triangle_mesh[np.all(triangle_mesh >= 0, axis=1)]
    positive_uv = triangle_uv[np.all(triangle_mesh >= 0, axis=1)]
    return positive_triangles, positive_uv


def main():
    # read quad mesh using openmesh
    # half edge representation
    file_path = str(facex_dir / "generic_neutral_mesh.obj")
    generic_neutral_mesh = om.read_polymesh(file_path, halfedge_tex_coord=True)
    quad_faces = generic_neutral_mesh.face_vertex_indices()
    vertices = generic_neutral_mesh.points()
    tex_coords = generic_neutral_mesh.halfedge_texcoords2D()
    uv_quads = tex_coords[generic_neutral_mesh.face_halfedge_indices()]

    faces, triangle_uv = convert_quad_mesh_to_triangle_mesh(quad_faces, uv_quads)

    # print(np.min(triangle_uv, axis=0), np.max(triangle_uv, axis=0))

    # print(np.unique(faces), np.unique(faces).shape, vertices.shape)

    # duplicate vertices on uv seams
    # traverse all triangle uv coordinates
    # vmapping is needed to keep the segmentation indices. so reinventing a wheel with a single extra output
    new_vertices = []
    new_uvs = []
    vmapping = []
    new_faces = []
    vertex_uvs = np.zeros((VERTEX_COUNT_STANDARD, 2)) - 1

    for n, face in tqdm.tqdm(enumerate(faces)):
        new_face = []
        for f in range(3):
            vertex_idx = face[f]
            if vertex_uvs[vertex_idx][0] < 0:
                vertex_uvs[vertex_idx] = triangle_uv[n, f]
            if vertex_idx not in vmapping:
                vmapping.append(vertex_idx)
                new_vertices.append(vertices[vertex_idx])
                new_uvs.append(triangle_uv[n, f])
                new_face.append(len(vmapping) - 1)
            else:
                uv_coord = triangle_uv[n, f]
                conflicting_uv_coord = new_uvs[vmapping.index(vertex_idx)]
                if not np.allclose(uv_coord, conflicting_uv_coord):
                    vmapping.append(vertex_idx)
                    new_vertices.append(vertices[vertex_idx])
                    new_uvs.append(triangle_uv[n, f])
                    new_face.append(len(vmapping) - 1)
                else:
                    new_face.append(vmapping.index(vertex_idx))

        new_faces.append(np.array(new_face))

    print(np.all(vertex_uvs >= 0))

    new_vertices = np.vstack(new_vertices)
    new_uvs = np.vstack(new_uvs)
    vmapping = np.array(vmapping)
    new_faces = np.vstack(new_faces)
    
    # subtract all integer parts to keep only decimal parts in new_uvs and vertex_uvs
    new_uvs -= np.floor(new_uvs)
    # vertex_uvs -= np.floor(vertex_uvs)

    vertex_uvs -= np.floor(vertex_uvs)
    # print(np.min(vertex_uvs[21451:], axis=0), np.max(vertex_uvs[21451:], axis=0))
    # vertex_uvs[21451:, 0] /= 2.
    # vertex_uvs[21451:]


    # debug
    # trimesh_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
    # trimesh_mesh.visual = trimesh.visual.TextureVisuals(uv=new_uvs)
    # trimesh_mesh.export('debug/cleaned_uv.obj')

    ict_model = face_model_io.load_face_model(str(facex_dir))

    ict_num_expression = ict_model._num_expression_shapes
    ict_num_identity = ict_model._num_identity_shapes
    ict_expression_shape_modes = ict_model._expression_shape_modes[:, :VERTEX_COUNT_STANDARD]
    ict_identity_shape_modes = ict_model._identity_shape_modes[:, :VERTEX_COUNT_STANDARD]

    landmark_indices = [
        1278, 1272, 12, 1834, 243, 781, 2199, 1447, 966, 3661, 4390,
        3022, 2484, 4036, 2253, 3490, 3496, 268, 493, 1914,
        2044, 1401, 3615, 4240, 4114, 2734, 2509, 978, 4527, 4942,
        4857, 1140, 2075, 1147, 4269, 3360, 1507, 1542, 1537, 1528,
        1518, 1511, 3742, 3751, 3756, 3721, 3725, 3732, 5708, 5695,
        2081, 0, 4275, 6200, 6213, 6346, 6461, 5518, 5957, 5841, 5702,
        5711, 5533, 6216, 6207, 6470, 5517, 5966,
    ]

    vertices = vertices[:VERTEX_COUNT_STANDARD]
    vertex_uvs = vertex_uvs[:VERTEX_COUNT_STANDARD]
    vertex_parts = vertex_parts_from_splits(len(vertices), OFFICIAL_PART_SPLITS)
    regions = build_official_region_indices()

    # fetch above five elements to a single dict
    ict_model_dict = {}
    ict_model_dict['neutral_mesh'] = vertices
    ict_model_dict['uv_neutral_mesh'] = vertex_uvs
    ict_model_dict['vertex_parts'] = vertex_parts
    ict_model_dict['faces'] = faces
    ict_model_dict['uv_faces'] = new_faces
    ict_model_dict['quad_faces'] = generic_neutral_mesh.face_vertex_indices()
    ict_model_dict['uvs'] = new_uvs
    ict_model_dict['vmapping'] = vmapping
    ict_model_dict['quad_faces'] = ict_model._generic_neutral_mesh.face_vertex_indices()
    ict_model_dict['num_expression'] = ict_num_expression
    ict_model_dict['num_identity'] = ict_num_identity
    ict_model_dict['expression_shape_modes'] = ict_expression_shape_modes
    ict_model_dict['identity_shape_modes'] = ict_identity_shape_modes
    # ict_model_dict['generic_neutral_mesh'] = ict_model._generic_neutral_mesh
    ict_model_dict['expression_names'] = ict_model._expression_names
    ict_model_dict['identity_names'] = ict_model._identity_names
    ict_model_dict['model_config'] = ict_model._model_config
    ict_model_dict['landmark_indices'] = landmark_indices
    ict_model_dict.update(regions)
    ict_model_dict.update(
        build_region_dict(
            vertices,
            vertex_parts,
            regions["face_indices"],
            regions["not_face_indices"],
            regions["eyeball_indices"],
            OFFICIAL_PART_SPLITS,
            asset_variant="official_24591",
        )
    )

    out_path = ASSETS_DIR / "ict_facekit_torch.npy"
    np.save(str(out_path), ict_model_dict)
    print(f"saved {out_path}  verts={len(vertices)}  variant={ict_model_dict['asset_variant']}")


if __name__ == "__main__":
    main()
