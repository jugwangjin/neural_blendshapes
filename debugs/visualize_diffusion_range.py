# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de
from arguments import config_parser
import os
os.environ["GLOG_minloglevel"] ="2"
import numpy as np
from pathlib import Path
from gpytoolbox import remesh_botsch
import torch
from tqdm import tqdm
from flame.FLAME import FLAME
from flare.dataset import *
from flare.dataset import dataset_util

from flare.core import (
    Mesh, Renderer
)
from flare.losses import *
from flare.modules import (
    NeuralShader, get_neural_blendshapes
)
from flare.utils import (
    AABB, read_mesh, write_mesh,
    visualize_training,
    make_dirs, set_defaults_finetune, copy_sources
)
import nvdiffrec.render.light as light
from test import run, quantitative_eval

import time

from flare.utils.ict_model import ICTFaceKitTorch
import open3d as o3d


import matplotlib.pyplot as plt
# from flare.modules.optimizer import torch.optim.Adam

def compute_laplacian_uniform_filtered(mesh, head_index=11248):
    """
    Computes the laplacian in packed form.
    The definition of the laplacian is
    L[i, j] =    -1       , if i == j
    L[i, j] = 1 / deg(i)  , if (i, j) is an edge
    L[i, j] =    0        , otherwise
    where deg(i) is the degree of the i-th vertex in the graph
    Returns:
        Sparse FloatTensor of shape (V, V) where V = sum(V_n)
    """

    # This code is adapted from from PyTorch3D 
    # (https://github.com/facebookresearch/pytorch3d/blob/88f5d790886b26efb9f370fb9e1ea2fa17079d19/pytorch3d/structures/meshes.py#L1128)

    verts_packed = mesh.vertices # (sum(V_n), 3)
    edges_packed = mesh.edges    # (sum(E_n), 2)

    # filter out the head vertices
    verts_packed = verts_packed[:head_index]

    # filter out the head edges
    edges_packed = edges_packed[edges_packed[:, 0] < head_index]
    edges_packed = edges_packed[edges_packed[:, 1] < head_index]

    V = head_index

    e0, e1 = edges_packed.unbind(1)

    idx01 = torch.stack([e0, e1], dim=1)  # (sum(E_n), 2)
    idx10 = torch.stack([e1, e0], dim=1)  # (sum(E_n), 2)
    idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*sum(E_n))

    # First, we construct the adjacency matrix,
    # i.e. A[i, j] = 1 if (i,j) is an edge, or
    # A[e0, e1] = 1 &  A[e1, e0] = 1
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=mesh.device)
    A = torch.sparse_coo_tensor(idx, ones, (V, V), dtype=ones.dtype, device=mesh.device)

    # the sum of i-th row of A gives the degree of the i-th vertex
    deg = torch.sparse.sum(A, dim=1).to_dense()

    # We construct the Laplacian matrix by adding the non diagonal values
    # i.e. L[i, j] = 1 ./ deg(i) if (i, j) is an edge
    deg0 = deg[e0]
    deg0 = torch.where(deg0 > 0.0, 1.0 / deg0, deg0)
    deg1 = deg[e1]
    deg1 = torch.where(deg1 > 0.0, 1.0 / deg1, deg1)
    val = torch.cat([deg0, deg1])
    L = torch.sparse_coo_tensor(idx, val, (V, V), dtype=ones.dtype, device=mesh.device)

    # Then we add the diagonal values L[i, i] = -1.
    idx = torch.arange(V, device=mesh.device)
    idx = torch.stack([idx, idx], dim=0)
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=mesh.device)
    L -= torch.sparse_coo_tensor(idx, ones, (V, V), dtype=ones.dtype, device=mesh.device)

    return L












def main(args, device, dataset_train, dataloader_train, debug_views):
    ## ============== Dir ==============================
    ## ============== load ict facekit ==============================
    ict_facekit = ICTFaceKitTorch(npy_dir = './assets/ict_facekit_torch.npy', canonical = Path(args.input_dir) / 'ict_identity.npy')
    ict_facekit = ict_facekit.to(device)

    # ==============================================================================================
    # deformation 
    # ==============================================================================================

    model_path = None
    print("=="*50)
    print("Training Deformer")

    neural_blendshapes = get_neural_blendshapes(model_path=model_path, train=args.train_deformer, vertex_parts=ict_facekit.vertex_parts, ict_facekit=ict_facekit, exp_dir = None, lambda_=args.lambda_, device=device) 

    neural_blendshapes = neural_blendshapes.to(device)

    # ==============================================================================================
    # T R A I N I N G
    # ==============================================================================================

    custom_delta = torch.zeros_like(ict_facekit.canonical[0][:neural_blendshapes.socket_index])

    custom_delta[0, 1] = 4

    custom_delta = neural_blendshapes.solve(custom_delta) # V 3
    

    ict_mesh_w_temp = neural_blendshapes.ict_facekit.neutral_mesh_canonical[0][:neural_blendshapes.socket_index] + custom_delta
    deformation = ict_mesh_w_temp - neural_blendshapes.ict_facekit.neutral_mesh_canonical[0][:neural_blendshapes.socket_index]
    deformation = torch.cat([deformation, deformation[neural_blendshapes.interior_displacement_index]], dim=0)
    ict_mesh_w_temp = neural_blendshapes.ict_facekit.neutral_mesh_canonical[0] + deformation

    deformation = (deformation / torch.amax(torch.abs(deformation))).abs() / 2 + 0.5

    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(ict_facekit.neutral_mesh_canonical[0].cpu().data.numpy())
    mesh.triangles = o3d.utility.Vector3iVector(ict_facekit.faces.cpu().data.numpy())

    vertex_colors = np.zeros_like(ict_facekit.neutral_mesh_canonical[0].cpu().data.numpy()) + 0.5
    vertex_colors = deformation.cpu().data.numpy()
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    o3d.io.write_triangle_mesh(f'./debug/range_on_neutral.obj', mesh)
    print(ict_mesh_w_temp.shape, ict_mesh_w_temp.device, ict_mesh_w_temp.dtype)
    mesh.vertices = o3d.utility.Vector3dVector(ict_mesh_w_temp.float().cpu().data.numpy())

    o3d.io.write_triangle_mesh(f'./debug/range_on_deformed.obj', mesh)
 


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()

    # Select the device
    device = torch.device('cpu')
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    print(f"Using device {device}")

    dataset_train    = None
    dataset_val      = None
    dataloader_train = None

    main(args, device, dataset_train, dataloader_train, None)