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
import nvdiffrec.render.light as light

import time

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

@torch.no_grad()
def main(args, device, dataset_train, dataloader_train, debug_views):
    original_dir = os.getcwd()
    # Add the path to the 'flare' directory
    flare_path = os.path.join(args.model_dir, args.model_name, 'sources')
    
    sys.path.insert(0, flare_path)

    from flame.FLAME import FLAME
    from flare.dataset import dataset_util

    from flare.core import (
        Mesh, Renderer
    )
    # from flare.modules import (
    #     NeuralShader, get_neural_blendshapes
    # )
    from flare.utils import (
        AABB, read_mesh, write_mesh,
        visualize_training, save_shading,
        make_dirs, set_defaults_finetune, copy_sources
    )
    from flare.modules import (
        NeuralShader, get_neural_blendshapes
    )

    ## ============== Dir ==============================
    run_name = args.run_name if args.run_name is not None else args.input_dir.parent.name

    project_directory = os.getcwd()

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

        if args.model_name not in input_mesh:
            continue

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

        print("=="*50)
        print(args.model_name, input_mesh)

        mesh.compute_connectivity()
        deformed_vertices = mesh.vertices[None]
        d_normals = mesh.fetch_all_normals(deformed_vertices, mesh)


        
        gbuffer = renderer.render_batch(debug_views['camera'], deformed_vertices=deformed_vertices.contiguous(), deformed_normals=d_normals,
                                        channels=channels_gbuffer, with_antialiasing=True, mesh=mesh,
                                        canonical_v=mesh.vertices, canonical_idx=mesh.indices, canonical_uv=ict_facekit.uv_neutral_mesh)
        rgb_pred, cbuffers, gbuffer_mask = shader.shade(gbuffer, debug_views, mesh, args.finetune_color, lgt)

        # determine output image name. 
        output_mesh = (input_mesh.split('/')[-1].split('.')[0] + '_shading.png')

        save_shading(rgb_pred, cbuffers, gbuffer, debug_views, output_dir, 0, ict_facekit=ict_facekit, save_name=output_mesh, )
        print(output_dir, output_mesh)

        print(deformed_vertices.shape, deformed_vertices.amin(dim=1), deformed_vertices.amax(dim=1))

        print("=="*50)
        print("smile and sadness")

    
        # neural blendshapes
        model_path = os.path.join(args.output_dir, args.run_name, 'stage_1', 'network_weights', 'neural_blendshapes.pt')
        print("=="*50)
        print("Training Deformer")
        face_normals = ict_canonical_mesh.get_vertices_face_normals(ict_facekit.neutral_mesh_canonical[0])[0]
        neural_blendshapes = get_neural_blendshapes(model_path=model_path, train=args.train_deformer, ict_facekit=ict_facekit, aabb = ict_mesh_aabb, face_normals=face_normals,device=device) 

        neural_blendshapes = neural_blendshapes.to(device)

        return_dict = neural_blendshapes

        facs = torch.zeros(1, 53).to(device)
        translation = torch.zeros(1, 3).to(device)
        rotation = torch.zeros(1, 3).to(device)
        global_translation = torch.zeros(1, 3).to(device)

        features = torch.zeros(1, 63).to(device)


        happiness = ['cheekSquint_L', 'cheekSquint_R', 'mouthSmile_L', 'mouthSmile_R']
        sadness = ['browInnerUp_L', 'browInnerUp_R', 'browDown_L', 'browDown_R', 'mouthFrown_L', 'mouthFrown_R']

        facs = torch.zeros(1, 53).to(device)
        facs[:, [ict_facekit.expression_names.tolist().index(name) for name in happiness]] = 0.8


        features[:, :53] = facs
        features[:, 53:56] = rotation
        features[:, 56:59] = translation
        features[:, 59:60] = torch.ones_like(translation[:, -1:]) * (neural_blendshapes.encoder.elu(neural_blendshapes.encoder.scale) + 1)
        features[:, 60:63] = global_translation
        
        
        return_dict = neural_blendshapes(features=features)
        deformed_vertices = return_dict['expression_mesh_posed']

        d_normals = ict_canonical_mesh.fetch_all_normals(deformed_vertices, ict_canonical_mesh)

        gbuffer = renderer.render_batch(debug_views['camera'], deformed_vertices=deformed_vertices.contiguous(), deformed_normals=d_normals,
                                        channels=channels_gbuffer, with_antialiasing=True, mesh=ict_canonical_mesh,
                                        canonical_v=ict_canonical_mesh.vertices, canonical_idx=ict_canonical_mesh.indices, canonical_uv=ict_facekit.uv_neutral_mesh)
        rgb_pred, cbuffers, gbuffer_mask = shader.shade(gbuffer, debug_views, ict_canonical_mesh, args.finetune_color, lgt)

        # determine output image name. 
        print(deformed_vertices.shape, deformed_vertices.amin(dim=1), deformed_vertices.amax(dim=1))
        output_mesh = (input_mesh.split('/')[-1].split('.')[0] + '_shading_smile.png')

        save_shading(rgb_pred, cbuffers, gbuffer, debug_views, output_dir, 0, ict_facekit=ict_facekit, save_name=output_mesh, )
        print(output_dir, output_mesh)
        print('sdada    ')

        facs = torch.zeros(1, 53).to(device)
        facs[:, [ict_facekit.expression_names.tolist().index(name) for name in sadness]] = 0.8


        features[:, :53] = facs
        features[:, 53:56] = rotation
        features[:, 56:59] = translation
        features[:, 59:60] = torch.ones_like(translation[:, -1:]) * (neural_blendshapes.encoder.elu(neural_blendshapes.encoder.scale) + 1)
        features[:, 60:63] = global_translation
        
        return_dict = neural_blendshapes(features=features)
        deformed_vertices = return_dict['expression_mesh_posed']

        d_normals = ict_canonical_mesh.fetch_all_normals(deformed_vertices, ict_canonical_mesh)

        gbuffer = renderer.render_batch(debug_views['camera'], deformed_vertices=deformed_vertices.contiguous(), deformed_normals=d_normals,
                                        channels=channels_gbuffer, with_antialiasing=True, mesh=ict_canonical_mesh,
                                        canonical_v=ict_canonical_mesh.vertices, canonical_idx=ict_canonical_mesh.indices, canonical_uv=ict_facekit.uv_neutral_mesh)
        rgb_pred, cbuffers, gbuffer_mask = shader.shade(gbuffer, debug_views, ict_canonical_mesh, args.finetune_color, lgt)

        # determine output image name. 
        output_mesh = (input_mesh.split('/')[-1].split('.')[0] + '_shading_sadness.png')
        print(deformed_vertices.shape, deformed_vertices.amin(dim=1), deformed_vertices.amax(dim=1))

        save_shading(rgb_pred, cbuffers, gbuffer, debug_views, output_dir, 0, ict_facekit=ict_facekit, save_name=output_mesh, )
        
        print(output_dir, output_mesh)


    
if __name__ == '__main__':
    import configargparse
    parser = configargparse.ArgumentParser()
    # add arguments for parser
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str, required=True)
    
    parser.add_argument('--model_dir', type=str, default='/Bean/log/gwangjin/2024/nbshapes_comparisons/ours_enc_v6', help='Path to the trained model')
    parser.add_argument('--model_name', type=str, default='marcel', help='Name of the run in model_dir')
    args = parser.parse_args()

    config_file = os.path.join(args.model_dir, args.model_name, 'sources', 'config.txt')
    if not os.path.exists(config_file):
        exit()
        config_file = os.path.join('configs_tmp', args.model_name+'.txt')

    # Override the config file argument

    parser = config_parser()
    parser.add_argument('--index', nargs='+', type=str, default='0', help='List of indices (e.g., 1 3-6 8-11)')

    args2 = parser.parse_args(['--config', config_file])
    args2.run_name = args.model_name
    args2.output_dir = args.model_dir
    args2.model_dir = args.model_dir
    args2.model_name = args.model_name
    args2.input = args.input
    args2.output = args.output

    args = args2

    device = torch.device('cpu')
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    print(f"Using device {device}")

    # needed arguments. 
    # train_dir
    # eval_dir
    flare_path = os.path.join(args.model_dir, args.model_name, 'sources')

    sys.path.insert(0, flare_path)

    from flare.utils.ict_model import ICTFaceKitTorch
    from flame.FLAME import FLAME
    from flare.dataset import dataset_util, DatasetLoader

    # ==============================================================================================
    # load data
    # ==============================================================================================
    print("loading train views...")
    dataset_train    = DatasetLoader(args2, train_dir=args.train_dir, sample_ratio=args.sample_idx_ratio, pre_load=False, train=True)
    dataset_val      = DatasetLoader(args2, train_dir=args.eval_dir, sample_ratio=24, pre_load=False)
    # assert dataset_train.len_img == len(dataset_train.importance)
    # dataset_sampler = torch.utils.data.WeightedRandomSampler(dataset_train.importance, dataset_train.len_img, replacement=True)
    # dataloader_train    = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, collate_fn=dataset_train.collate, drop_last=True, sampler=dataset_sampler)
    dataloader_train = None
    view_indices = np.array([0]).astype(int)
    d_l = [dataset_val.__getitem__(idx) for idx in view_indices]
    debug_views = dataset_val.collate(d_l)


    del dataset_val

    main(args2, device, dataset_train, dataloader_train, debug_views)
