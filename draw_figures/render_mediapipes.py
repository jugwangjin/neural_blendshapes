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
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from arguments import config_parser

import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(20202464)

import os
os.environ["GLOG_minloglevel"] ="2"
import numpy as np
from pathlib import Path

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
    visualize_training, save_shading,
    make_dirs, set_defaults_finetune, copy_sources
)
import nvdiffrec.render.light as light
from test import run, quantitative_eval

import time

from flare.utils.ict_model import ICTFaceKitTorch
import open3d as o3d
import cv2

import hashlib

import time
import gc

import matplotlib.pyplot as plt
# from flare.modules.optimizer import torch.optim.Adam
import sys


def hash_file(file_path):
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def hash_directory(directory):
    hashes = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            # if not ends with .py, skip
            if not file.endswith('.py'):
                continue
            file_path = os.path.join(root, file)
            file_hash = hash_file(file_path)
            hashes.append(file_hash)
    combined_hash = hashlib.sha256(''.join(hashes).encode()).hexdigest()
    return combined_hash

def hash_arguments(arguments):
    hasher = hashlib.sha256()
    for arg in arguments:
        hasher.update(arg.encode())
    return hasher.hexdigest()



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


def clip_grad(neural_blendshapes, shader, norm=1.0):
    # torch.nn.utils.clip_grad_norm_(shader.material_mlp.parameters(), norm)
    # torch.nn.utils.clip_grad_norm_(shader.light_mlp.parameters(), norm)
    # torch.nn.utils.clip_grad_norm_(shader.brdf_mlp.parameters(), norm)
    torch.nn.utils.clip_grad_norm_(neural_blendshapes.template_deformer.parameters(), norm)
    torch.nn.utils.clip_grad_norm_(neural_blendshapes.expression_deformer.parameters(), norm)
    torch.nn.utils.clip_grad_norm_(neural_blendshapes.pose_weight.parameters(), norm)
    torch.nn.utils.clip_grad_norm_(neural_blendshapes.encoder.tail.parameters(), norm)
    torch.nn.utils.clip_grad_norm_(neural_blendshapes.encoder.bshape_modulator.parameters(), norm)
    torch.nn.utils.clip_grad_norm_([neural_blendshapes.encoder.bshapes_multiplier], norm)
    return


def main(args, device, dataset_train, dataloader_train, debug_views):


    ## ============== Dir ==============================
    run_name = args.run_name if args.run_name is not None else args.input_dir.parent.name
    images_save_path, images_eval_save_path, meshes_save_path, shaders_save_path, experiment_dir = make_dirs(args, run_name, args.finetune_color)
    copy_sources(args, run_name)

    project_directory = os.getcwd()

    print("Project Directory:", project_directory)
    print("start hashing directory")
    directory_hash = hash_directory(project_directory)
    print("start hashing arguments")
    arguments_hash = hash_arguments(sys.argv)
    print("hashing done")
    final_hash = hashlib.sha256((directory_hash + arguments_hash).encode()).hexdigest()


    ## =================== Load FLAME ==============================
    flame_path = args.working_dir / 'flame/FLAME2020/generic_model.pkl'
    flame_shape = dataset_train.shape_params
    FLAMEServer = FLAME(flame_path, n_shape=100, n_exp=50, shape_params=flame_shape).to(device)

    ## ============== canonical with mouth open (jaw pose 0.4) ==============================
    FLAMEServer.canonical_exp = (dataset_train.get_mean_expression()).to(device)
    FLAMEServer.canonical_pose = FLAMEServer.canonical_pose.to(device)
    FLAMEServer.canonical_verts, FLAMEServer.canonical_pose_feature, FLAMEServer.canonical_transformations = \
        FLAMEServer(expression_params=FLAMEServer.canonical_exp, full_pose=FLAMEServer.canonical_pose)
    if args.ghostbone:
        FLAMEServer.canonical_transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).float().to(device), FLAMEServer.canonical_transformations], 1)
    FLAMEServer.canonical_verts = FLAMEServer.canonical_verts.to(device)

    # flame_canonical_mesh = Mesh(FLAMEServer.canonical_verts[0].cpu().data, FLAMEServer.faces.cpu().data, device=device)
    # flame_canonical_mesh.compute_connectivity()


    ## ============== load ict facekit ==============================
    ict_facekit = ICTFaceKitTorch(npy_dir = './assets/ict_facekit_torch.npy', canonical = Path(args.input_dir) / 'ict_identity.npy')
    ict_facekit = ict_facekit.to(device)

    ict_canonical_mesh = Mesh(ict_facekit.canonical[0].cpu().data, ict_facekit.faces.cpu().data, ict_facekit=ict_facekit, device=device)
    ict_canonical_mesh.compute_connectivity()


    ## ============== renderer ==============================
    aabb = AABB(ict_canonical_mesh.vertices.cpu().numpy())
    ict_mesh_aabb = [torch.min(ict_canonical_mesh.vertices, dim=0).values, torch.max(ict_canonical_mesh.vertices, dim=0).values]

    renderer = Renderer(device=device)
    renderer.set_near_far(dataset_train, torch.from_numpy(aabb.corners).to(device), epsilon=0.5)
    channels_gbuffer = ['mask', 'position', 'normal', "canonical_position"]
    print("Rasterizing:", channels_gbuffer)
    

    # ==============================================================================================
    # shading
    # ==============================================================================================

    lgt = light.create_env_rnd()    
    disentangle_network_params = {
        "material_mlp_ch": args.material_mlp_ch,
        "light_mlp_ch":args.light_mlp_ch,
        "material_mlp_dims":args.material_mlp_dims,
        "light_mlp_dims":args.light_mlp_dims,
        "brdf_mlp_dims": args.brdf_mlp_dims,

    }

    # Create the optimizer for the neural shader
    # shader = NeuralShader(fourier_features=args.fourier_features,
    shader = NeuralShader(fourier_features='positional',
                          activation=args.activation,
                          last_activation=torch.nn.Sigmoid(), 
                          disentangle_network_params=disentangle_network_params,
                          bsdf=args.bsdf,
                          aabb=ict_mesh_aabb,
                          device=device)
    
    for iter_, views_subset in tqdm(enumerate(dataloader_train)):
        

    # get target outpupts. 
    # args.input
    input_meshes = []
    # if args.input is a directory, grep all meshes (obj, ply, ...)
    if os.path.isdir(args.input):
        for root, dirs, files in os.walk(args.input):
            for file in files:
                if file.endswith('.obj') or file.endswith('.ply'):
                    input_meshes.append(os.path.join(root, file))
    else:
        input_meshes.append(args.input)

    # make output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_mesh in input_meshes:
        # read the input mesh
        mesh = read_mesh(input_mesh, device=device)
        try:
            mesh_ = Mesh(mesh.vertices, mesh.indices, ict_facekit = ict_facekit, device=device)
            deformed_vertices = mesh_.vertices[None]
            d_normals = mesh.fetch_all_normals(deformed_vertices, mesh_)
            mesh = mesh_
            # mesh.compute_connectivity()
            # deformed_vertices = mesh.vertices[None]
            # d_normals = mesh.fetch_all_normals(deformed_vertices, mesh)

        except Exception as e:
            print(input_mesh, e)

        mesh.compute_connectivity()
        deformed_vertices = mesh.vertices[None]
        d_normals = mesh.fetch_all_normals(deformed_vertices, mesh)


        
        gbuffer = renderer.render_batch(debug_views['camera'], deformed_vertices=deformed_vertices, deformed_normals=d_normals,
                                        channels=channels_gbuffer, with_antialiasing=True,
                                        canonical_v=mesh.vertices, canonical_idx=mesh.indices, canonical_uv=ict_facekit.uv_neutral_mesh)
        rgb_pred, cbuffers, gbuffer_mask = shader.shade(gbuffer, debug_views, mesh, args.finetune_color, lgt)

        # determine output image name. 
        output_mesh = (input_mesh.split('/')[-1].split('.')[0] + '_shading.png')

        save_shading(rgb_pred, cbuffers, gbuffer, debug_views, output_dir, 0, ict_facekit=ict_facekit, save_name=output_mesh)




if __name__ == '__main__':
    parser = config_parser()
    # add arguments for parser
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()


    device = torch.device('cpu')
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    print(f"Using device {device}")

    # needed arguments. 
    # train_dir
    # eval_dir

    # ==============================================================================================
    # load data
    # ==============================================================================================
    print("loading train views...")
    dataset_train    = DatasetLoader(args, train_dir=args.train_dir, sample_ratio=args.sample_idx_ratio, pre_load=False, train=True)
    dataset_val      = DatasetLoader(args, train_dir=args.eval_dir, sample_ratio=24, pre_load=False)
    # assert dataset_train.len_img == len(dataset_train.importance)
    # dataset_sampler = torch.utils.data.WeightedRandomSampler(dataset_train.importance, dataset_train.len_img, replacement=True)
    # dataloader_train    = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, collate_fn=dataset_train.collate, drop_last=True, sampler=dataset_sampler)
    dataloader_train = None
    view_indices = np.array([0]).astype(int)
    d_l = [dataset_val.__getitem__(idx) for idx in view_indices]
    debug_views = dataset_val.collate(d_l)


    del dataset_val

    main(args, device, dataset_train, dataloader_train, debug_views)
