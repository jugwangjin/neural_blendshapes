from arguments import config_parser
import torch

import numpy as np
import pickle
import os
import chumpy as ch

from ICT_FaceKit.Scripts import face_model_io
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


def main(args):
    # read quad mesh using openmesh
    # half edge representation
    file_path = os.path.join('ICT_FaceKit/FaceXModel/generic_neutral_mesh.obj')
    generic_neutral_mesh = om.read_polymesh(file_path, halfedge_tex_coord = True)
    faces = generic_neutral_mesh.face_vertex_indices()[:24692]
    vertices = generic_neutral_mesh.points()
    tex_coords = generic_neutral_mesh.halfedge_texcoords2D()
    uv_quads = tex_coords[generic_neutral_mesh.face_halfedge_indices()]
    
    # convert it to triangle mesh
    faces, triangle_uv = convert_quad_mesh_to_triangle_mesh(faces, uv_quads)

    # duplicate vertices on uv seams
    # traverse all triangle uv coordinates
    # vmapping is needed to keep the segmentation indices. so reinventing a wheel with a single extra output
    new_vertices = []
    new_uvs = []
    vmapping = []
    new_faces = []
    vertex_uvs = np.zeros((len(vertices), 2))

    for n, face in tqdm.tqdm(enumerate(faces)):
        new_face = []
        for f in range(3):
            vertex_idx = face[f]
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

    new_vertices = np.vstack(new_vertices)
    new_uvs = np.vstack(new_uvs)
    vmapping = np.array(vmapping)
    new_faces = np.vstack(new_faces)
    
    # subtract all integer parts to keep only decimal parts in new_uvs and vertex_uvs
    new_uvs -= np.floor(new_uvs)
    vertex_uvs -= np.floor(vertex_uvs)
    


    # debug
    # trimesh_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
    # trimesh_mesh.visual = trimesh.visual.TextureVisuals(uv=new_uvs)
    # trimesh_mesh.export('debug/cleaned_uv.obj')

    ict_model = face_model_io.load_face_model('ICT_FaceKit/FaceXModel')

    ict_num_expression = ict_model._num_expression_shapes
    ict_num_identity = ict_model._num_identity_shapes
    ict_expression_shape_modes = ict_model._expression_shape_modes[:, 24591]
    ict_identity_shape_modes = ict_model._identity_shape_modes[:, 24591]

    landmark_indices = [1225, 1888, 1052, 367, 1719, 1722, 2199, 1447, 966, 3661, 
                                 4390, 3927, 3924, 2608, 3272, 4088, 3443, 268, 493, 1914, 
                                 2044, 1401, 3615, 4240, 4114, 2734, 2509, 978, 4527, 4942, 
                                 4857, 1140, 2075, 1147, 4269, 3360, 1507, 1542, 1537, 1528, 
                                 1518, 1511, 3742, 3751, 3756, 3721, 3725, 3732, 5708, 5695, 
                                 2081, 0, 4275, 6200, 6213, 6346, 6461, 5518, 5957, 5841, 5702, 
                                 5711, 5533, 6216, 6207, 6470, 5517, 5966]
    face_indices = list(range(0, 9409)) + list(range(11248, 21451)) 
    not_face_indices = list(range(9409, 11248))
    eyeball_indices = list(range(21451, 24591)) 
    head_indices = face_indices + not_face_indices

    vertices = vertices[:24591]
    vertex_uvs = vertex_uvs[:24591]

    parts_split = [9409, 14062, 17039, 21451, 24591]
    vertex_parts = [0] * len(vertices)
    
    for i, part in enumerate(parts_split):
        if i == 0:
            vertex_parts[:part] = [i] * part
        else:
            vertex_parts[parts_split[i-1]:part] = [i] * (part - parts_split[i-1])

    
    # build vmapped indices indices to face / not face / eyeball dict
    vmapping_dict = {v: i for i, v in enumerate(vmapping)}
    new_landmark_indices = []
    for landmark_index in landmark_indices:
        new_landmark_indices.append(vmapping_dict[landmark_index])
    # landmark_indices = new_landmark_indices

        # build original indices to face / not face / eyeball dict
    region_dict = [0] * (len(face_indices) + len(not_face_indices) + len(eyeball_indices))

    for i in range(len(region_dict)):
        region_dict[i] = 0 if i in face_indices else 1 if i in not_face_indices else 2

    new_face_indices = []
    new_not_face_indices = []
    new_eyeball_indices = []

    for face_index in face_indices:
        new_face_indices.append(vmapping_dict[face_index])

    for not_face_index in not_face_indices:
        new_not_face_indices.append(vmapping_dict[not_face_index])

    for eyeball_index in eyeball_indices:
        new_eyeball_indices.append(vmapping_dict[eyeball_index])

    # face_indices = new_face_indices
    # not_face_indices = new_not_face_indices
    # eyeball_indices = new_eyeball_indices
    # head_indices = face_indices + not_face_indices


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
    ict_model_dict['face_indices'] = face_indices
    ict_model_dict['not_face_indices'] = not_face_indices
    ict_model_dict['eyeball_indices'] = eyeball_indices
    ict_model_dict['head_indices'] = head_indices

    # save as a numpy
    np.save('./assets/ict_facekit_torch.npy', ict_model_dict)


if __name__ == '__main__':
    
    parser = config_parser()
    args = parser.parse_args()

    main(args)
