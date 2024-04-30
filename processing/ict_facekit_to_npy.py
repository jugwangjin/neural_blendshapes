from flare.dataset import *
from flame.FLAME import FLAME
from arguments import config_parser
import torch
from flame.FLAME import FLAME
from flare.dataset import *

from flare.dataset import dataset_util
import numpy as np
import pickle
import os
import chumpy as ch

from ICT_FaceKit.Scripts import face_model_io
# from ICT_FaceKit.ict_model import ICTModel

ICT_LMK_IDX = [36, 39, 42, 45, 30, 48, 54]
FLAME_LMK_IDX = [19, 22, 25, 28, 13, 31, 37]

def load_binary_pickle( filepath ):
    with open(filepath, 'rb') as f:
        data = pickle.load(f, encoding="latin1")
    return data

def convert_quad_mesh_to_triangle_mesh(faces):
    """Converts a quad mesh represented as a faces array to a triangle mesh.

    Args:
        faces: A NumPy array of shape (F, 4), where each row represents a quad.

    Returns:
        A NumPy array of shape (F * 2, 3), where each row represents a triangle.
    """

    # Create a new triangle mesh.
    triangle_mesh = np.zeros((faces.shape[0] * 2, 3))

    # For each quad in the faces array, create two triangles by splitting the quad diagonally.
    for i in range(faces.shape[0]):
        
        triangle_mesh[i * 2] = faces[i, [0, 1, 2]]
        triangle_mesh[i * 2 + 1] = faces[i, [2, 3, 0]]

    triangle_mesh = remove_negative_triangles(triangle_mesh)

    return triangle_mesh

def remove_negative_triangles(triangle_mesh):
    """Removes triangles from a triangle mesh that have negative (-1) elements.

    Args:
        triangle_mesh: A NumPy array of shape (F, 3), where each row represents a triangle.

    Returns:
        A NumPy array of shape (F', 3), where each row represents a triangle without negative elements.
    """
    positive_triangles = triangle_mesh[np.all(triangle_mesh >= 0, axis=1)]
    return positive_triangles


def main(args):
    device = torch.device('cpu')
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    print(f"Using device {device}")
    
    ict_model = face_model_io.load_face_model('ICT_FaceKit/FaceXModel')

    ict_canonical_vertex = ict_model._generic_neutral_mesh.points()
    ict_faces = ict_model._generic_neutral_mesh.face_vertex_indices()
    ict_faces = convert_quad_mesh_to_triangle_mesh(ict_faces)
    
    ict_num_expression = ict_model._num_expression_shapes
    ict_num_identity = ict_model._num_identity_shapes
    ict_expression_shape_modes = ict_model._expression_shape_modes
    ict_identity_shape_modes = ict_model._identity_shape_modes

    # fetch above five elements to a single dict
    ict_model_dict = {}
    ict_model_dict['neutral_mesh'] = ict_canonical_vertex
    ict_model_dict['faces'] = ict_faces
    ict_model_dict['quad_faces'] = ict_model._generic_neutral_mesh.face_vertex_indices()
    ict_model_dict['num_expression'] = ict_num_expression
    ict_model_dict['num_identity'] = ict_num_identity
    ict_model_dict['expression_shape_modes'] = ict_expression_shape_modes
    ict_model_dict['identity_shape_modes'] = ict_identity_shape_modes
    # ict_model_dict['generic_neutral_mesh'] = ict_model._generic_neutral_mesh
    ict_model_dict['expression_names'] = ict_model._expression_names
    ict_model_dict['identity_names'] = ict_model._identity_names
    ict_model_dict['model_config'] = ict_model._model_config

    # save as a numpy
    np.save('./assets/ict_facekit_torch.npy', ict_model_dict)


if __name__ == '__main__':
    
    parser = config_parser()
    args = parser.parse_args()

    main(args)
