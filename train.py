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
    visualize_training, visualize_training_no_lm,
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
    # torch.nn.utils.clip_grad_norm_(neural_blendshapes.parameters(), norm)
    # torch.nn.utils.clip_grad_norm_(shader.parameters(), norm)
    # torch.nn.utils.clip_grad_norm_(neural_blendshapes.expression_deformer.parameters(), norm)
    # torch.nn.utils.clip_grad_norm_(neural_blendshapes.fou
    torch.nn.utils.clip_grad_norm_(neural_blendshapes.pose_weight.parameters(), norm)
    torch.nn.utils.clip_grad_norm_(neural_blendshapes.encoder.parameters(), norm)
    # torch.nn.utils.clip_grad_norm_(neural_blendshapes.encoder.encoder.encoder.parameters(), norm)
    return


def save_blendshape_figure(bshapes, names, title, save_path):

    plt.clf()
    plt.figsize=(8,8)
    plt.bar(np.arange(53), bshapes)
    plt.ylim(0, 1)
    plt.xticks(np.arange(53), names, rotation=90, fontsize=6)
    plt.title(title)
    plt.savefig(str(save_path))


def visualize_specific_traininig(mesh_key_name, return_dict, renderer, shader, ict_facekit, views_subset, mesh, save_name, images_save_path, iteration, lgt):
    debug_gbuffer = renderer.render_batch(views_subset['flame_camera'], return_dict[mesh_key_name+'_posed'].contiguous(), mesh.fetch_all_normals(return_dict[mesh_key_name+'_posed'], mesh),
                            channels=channels_gbuffer + ['segmentation'], with_antialiasing=True, 
                            canonical_v=mesh.vertices, canonical_idx=mesh.indices, canonical_uv=ict_facekit.uv_neutral_mesh,
                            mesh=mesh
                            )
    debug_rgb_pred, debug_cbuffers, _ = shader.shade(debug_gbuffer, views_subset, mesh, args.finetune_color, lgt)
    visualize_training(debug_rgb_pred, debug_cbuffers, debug_gbuffer, views_subset, images_save_path, iteration, ict_facekit=ict_facekit, save_name=save_name)

    return debug_gbuffer



def downsample_upsample(buffer, low_res, high_res):
    if low_res is None:
        return buffer
    if low_res == high_res:
        return buffer
    # Convert from (B, H, W, C) -> (B, C, H, W)
    buffer = buffer.permute(0, 3, 1, 2)
    
    downsampled = F.interpolate(buffer, size=low_res, mode='bilinear', align_corners=False)
    # Perform bilinear upsampling
    upsampled = F.interpolate(downsampled, size=high_res, mode='bilinear', align_corners=False)
    
    # Convert back from (B, C, H, W) -> (B, H, W, C)
    return upsampled.permute(0, 2, 3, 1)
        


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
                       

    # draw histogram - mode


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

    ict_to_flame_indices = np.load('assets/ict_to_flame_closest_indices.npy') # shape of [socket_index] vector, containing ict vertex to flame vertex mapping
    ict_to_flame_indices = torch.from_numpy(ict_to_flame_indices).long().to(device)

    ict_to_flame_pairs = np.load('assets/ict_to_flame_closest_pair.npy')
    ict_pair_indices = np.array(ict_to_flame_pairs[0]).astype(np.int64)
    flame_pair_indices = np.array(ict_to_flame_pairs[1]).astype(np.int64)

    tight_face_index = 6705

    full_face_index = 9409

    # print(ict_pair_indices)

    full_face_ict_flame = np.where(ict_pair_indices < full_face_index)[0]
    tight_face_ict_flame = np.where(ict_pair_indices < tight_face_index)[0]  

    ## ============== load ict facekit ==============================
    ict_facekit = ICTFaceKitTorch(npy_dir = './assets/ict_facekit_torch.npy', canonical = Path(args.input_dir) / 'ict_identity.npy')
    ict_facekit = ict_facekit.to(device)

    ict_canonical_mesh = Mesh(ict_facekit.canonical[0].cpu().data, ict_facekit.faces.cpu().data, ict_facekit=ict_facekit, device=device)
    ict_canonical_mesh.compute_connectivity()

    write_mesh(Path(meshes_save_path / "init_ict_canonical.obj"), ict_canonical_mesh.to('cpu'))


    if args.recompute_mode:
        precomputed_mode = dataset_train.base_dir / 'bshapes_mode.pt'
        if precomputed_mode.exists():
            # remove the file
            os.remove(precomputed_mode)
            
    mode = dataset_train.get_bshapes_mode(args.compute_mode)[ict_facekit.mediapipe_to_ict]

    bshapes = mode.detach().cpu().numpy()
    bshapes = np.round(bshapes, 2)
    
    names = ict_facekit.expression_names.tolist()

    os.makedirs(images_save_path / "grid", exist_ok=True)

    save_blendshape_figure(bshapes, names, f"Blendshapes Modes", images_save_path / "grid" / f"a_blendshape_modes.png")
                     


    tight_face_index = 6705
    face_index = 9409     
    head_index = 14062
    socket_index = 11248
    socket_index = 14062
    head_index=11248

    # filter vertices by head_index
    filtered_vertices = ict_facekit.neutral_mesh_canonical[0].cpu().data[:socket_index]
    filtered_faces = ict_facekit.faces.cpu().data
    # filter: 
    filtered_faces = filtered_faces[filtered_faces[:, 0] < socket_index]
    filtered_faces = filtered_faces[filtered_faces[:, 1] < socket_index]
    filtered_faces = filtered_faces[filtered_faces[:, 2] < socket_index]

    head_mesh = Mesh(filtered_vertices, filtered_faces, ict_facekit=None, device=device)

    ## ============== renderer ==============================
    aabb = AABB(ict_canonical_mesh.vertices.cpu().numpy())
    ict_mesh_aabb = [torch.min(ict_canonical_mesh.vertices, dim=0).values, torch.max(ict_canonical_mesh.vertices, dim=0).values]

    renderer = Renderer(device=device)
    renderer.set_near_far(dataset_train, torch.from_numpy(aabb.corners).to(device), epsilon=0.5)
    channels_gbuffer = ['mask', 'position', 'normal', "canonical_position"]
    print("Rasterizing:", channels_gbuffer)
    
    renderer_visualization = Renderer(device=device)
    renderer_visualization.set_near_far(dataset_train, torch.from_numpy(aabb.corners).to(device), epsilon=0.5)

    # ==============================================================================================
    # deformation 
    # ==============================================================================================

    model_path = None
    print("=="*50)
    print("Training Deformer")

    neural_blendshapes = get_neural_blendshapes(model_path=model_path, train=args.train_deformer, vertex_parts=ict_facekit.vertex_parts, ict_facekit=ict_facekit, exp_dir = experiment_dir, lambda_=args.lambda_, aabb = ict_mesh_aabb, device=device) 
    print(ict_canonical_mesh.vertices.shape, ict_canonical_mesh.vertices.device)

    neural_blendshapes = neural_blendshapes.to(device)

    neural_blendshapes_params = list(neural_blendshapes.parameters())
    neural_blendshapes_expression_params = list(neural_blendshapes.expression_deformer.parameters())
    neural_blendshapes_template_params = list(neural_blendshapes.template_deformer.parameters())
    neural_blendshapes_pe = list(neural_blendshapes.fourier_feature_transform.parameters()) + list(neural_blendshapes.fourier_feature_transform2.parameters())
    neural_blendshapes_pose_weight_params = list(neural_blendshapes.pose_weight.parameters())
    neural_blendshapes_encoder_params = list(neural_blendshapes.encoder.parameters())
    # neural_blendshapes_others_params = list(set(neural_blendshapes_params) - set(neural_blendshapes_expression_params) - set(neural_blendshapes_template_params) - set(neural_blendshapes_pe) - set(neural_blendshapes_pose_weight_params)) 
    
    # print("expression_deformer")
    # for name, param in neural_blendshapes.expression_deformer.named_parameters():
    #     print(name, param.shape)

    # print("template_deformer")
    # for name, param in neural_blendshapes.template_deformer.named_parameters():
    #     print(name, param.shape)

    # print("pose_weight")
    # for name, param in neural_blendshapes.pose_weight.named_parameters():
    #     print(name, param.shape)

    # print("encoder")
    # for name, param in neural_blendshapes.encoder.named_parameters():
    #     print(name, param.shape)

    # print("fourier_feature_transform")
    # for name, param in neural_blendshapes.fourier_feature_transform.named_parameters():
    #     print(name, param.shape)



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

    shader = NeuralShader(fourier_features='positional',
                          activation=args.activation,
                          last_activation=torch.nn.Sigmoid(), 
                          disentangle_network_params=disentangle_network_params,
                          bsdf=args.bsdf,
                          aabb=ict_mesh_aabb,
                          device=device)
    params = list(shader.parameters()) 

    if args.weight_albedo_regularization > 0:
        from robust_loss_pytorch.adaptive import AdaptiveLossFunction
        _adaptive = AdaptiveLossFunction(num_dims=4, float_dtype=np.float32, device=device)
        params += list(_adaptive.parameters()) ## need to train it


    optimizer_shader = torch.optim.Adam(params, lr=args.lr_shader, weight_decay=1e-4)

    # ==============================================================================================
    # Loss Functions
    # ==============================================================================================
    # Initialize the loss weights and losses
    loss_weights = {
        "mask": args.weight_mask,
        "laplacian_regularization": args.weight_laplacian_regularization,
        # "normal_regularization": args.weight_normal_regularization,
        "shading": args.weight_shading,
        "perceptual_loss": args.weight_perceptual_loss,
        "landmark": args.weight_landmark,
        "closure": args.weight_closure,
        "feature_regularization": args.weight_feature_regularization,
        "geometric_regularization": args.weight_geometric_regularization,
        "normal_laplacian": args.weight_normal_laplacian,
        "linearity_regularization": args.weight_linearity_regularization,
        "flame_regularization": args.weight_flame_regularization,
        "white_lgt_regularization": args.weight_white_lgt_regularization,
        "roughness_regularization": args.weight_roughness_regularization,
        "albedo_regularization": args.weight_albedo_regularization,
        "fresnel_coeff": args.weight_fresnel_coeff,
        "temporal_regularization": args.weight_feature_regularization,
    }

    losses = {k: torch.tensor(0.0, device=device) for k in loss_weights}
    print(loss_weights)
    VGGloss = VGGPerceptualLoss().to(device)

    print("=="*50)
    shader.train()
    
    neural_blendshapes.train()
    print("Batch Size:", args.batch_size)
    print("=="*50)

    # ==============================================================================================
    # T R A I N I N G
    # ==============================================================================================

    dataloader_train    = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, collate_fn=dataset_train.collate, drop_last=True, )
    importance = torch.ones(len(dataloader_train), device=device)
    dataset_sampler = torch.utils.data.WeightedRandomSampler(importance, dataset_train.len_img, replacement=True)
    dataloader_train    = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, collate_fn=dataset_train.collate, drop_last=True, sampler=dataset_sampler)

    loss_for_each_index = torch.zeros_like(importance) - 1

    losses = {k: torch.tensor(0.0, device=device) for k in loss_weights}

    epochs = (args.iterations // len(dataloader_train)) + 1
    iteration = 0
    
    progress_bar = tqdm(range(epochs))
    start = time.time()

    acc_losses = []
    acc_total_loss = 0

    weight_decay_rate = 0.1

    filtered_lap = compute_laplacian_uniform_filtered(ict_canonical_mesh, head_index=ict_canonical_mesh.vertices.shape[0])


    bshapes_multipliers = []
    for i in range(53):
        bshapes_multipliers.append([])

    use_jaw=True


    optimizer_neural_blendshapes = torch.optim.Adam([
                                                    {'params': neural_blendshapes_encoder_params, 'lr': args.lr_deformer},
                                                    {'params': neural_blendshapes_template_params, 'lr': args.lr_jacobian},
                                                    {'params': neural_blendshapes_pe, 'lr': args.lr_jacobian},
                                                    {'params': neural_blendshapes_pose_weight_params, 'lr': args.lr_jacobian},
                                                {'params': neural_blendshapes_expression_params, 'lr': args.lr_jacobian},
                                                    ],
                                                    )

    stage_iterations = args.stage_iterations
    # stage_iterations contain the number of iterations for each stage
    milestones = [args.stage_iterations[0]]
    for i in range(1, len(stage_iterations)):
        milestones.append(milestones[-1] + stage_iterations[i])

    
    # milestones should be a list of integers, length of 6
    assert len(milestones) == 6

    args.iterations = milestones[-1]


    import wandb
    if 'debug' not in run_name and not args.skip_wandb:
        wandb_name = args.wandb_name if args.wandb_name is not None else run_name
        wandb.init(project="neural_jacobian_blendshape_full_deform", name=wandb_name, config=args)
    else:
        losses_keys = {}
        for k in loss_weights:
            losses_keys[k] = []
        losses_keys['total_loss'] = []

            
    epochs = (args.iterations // len(dataloader_train)) + 1
    progress_bar = tqdm(range(epochs))
    start = time.time()

    epsilon = 1e-5

    iteration = 0
    for epoch in progress_bar:
        # importance = importance / (importance.amax() + epsilon)
        dataset_sampler = torch.utils.data.WeightedRandomSampler(importance, dataset_train.len_img, replacement=True)
        dataloader_train    = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, collate_fn=dataset_train.collate, drop_last=True, sampler=dataset_sampler)
        for iter_, views_subset in tqdm(enumerate(dataloader_train)):
            iteration += 1
            # Determine the stage based on iteration and milestones
            stage = next((i for i, milestone in enumerate(milestones) if iteration < milestone), len(milestones))

            '''
            training options
            '''
            # stage 0 : low resolution, only flame loss, encoder and template deformer training
            # stage 1 : low resolution, full loss, encoder and template deformer training
            # stage 2 : medium resolution, full loss, template deformer training
            # stage 3 : medium resolution, full loss, expression deformer training
            # stage 4 : high resolution, full loss, expression deformer training
            # stage 5 : high resolution, full loss, shader training
            # if stage < 2 :
            #     target_res = (256, 256)
            # else:
            target_res = None            

            if stage > 0:
                target_range = tight_face_ict_flame
            else:
                target_range = None
            
            if stage < 3 : 
                deformed_vertices_key = 'ict_mesh_w_temp'
            else:
                deformed_vertices_key = 'expression_mesh'

            if iteration == milestones[2]:
                loss_weights["flame_regularization"] = loss_weights["flame_regularization"] * 0.1


            '''
            optimizer updates
            '''
            if iteration == milestones[1]: # on stage 2 -> update the optimizer to only template 
                print("\nUpdating the optimizer to only template\n")
                # now only update the expression parameters
                optimizer_neural_blendshapes.zero_grad(set_to_none=True)
                optimizer_neural_blendshapes = None
                optimizer_neural_blendshapes = torch.optim.Adam([
                                                {'params': neural_blendshapes_template_params, 'lr': args.lr_jacobian},
                                                {'params': neural_blendshapes_pe, 'lr': args.lr_jacobian},
                                                {'params': neural_blendshapes_pose_weight_params, 'lr': args.lr_jacobian},
                                                {'params': neural_blendshapes_expression_params, 'lr': args.lr_jacobian},
                                                ],
                                                )

            if iteration == milestones[2]: # on stage 3 -> update the optimizer to only expression 
                print("\nUpdating the optimizer to only expression\n")
                # now only update the expression parameters
                optimizer_neural_blendshapes.zero_grad(set_to_none=True)
                optimizer_neural_blendshapes = None
                optimizer_neural_blendshapes = torch.optim.Adam([
                                                {'params': neural_blendshapes_expression_params, 'lr': args.lr_jacobian},
                                                {'params': neural_blendshapes_pe, 'lr': args.lr_jacobian},
                                                # {'params': neural_blendshapes_pose_weight_params, 'lr': args.lr_jacobian},
                                                ],
                                                )

            if iteration == milestones[4]:
                print("\nUpdating the optimizer to only shader\n")
                del shader
                del optimizer_shader

                shader = NeuralShader(fourier_features='hashgrid',
                        activation=args.activation,
                        last_activation=torch.nn.Sigmoid(), 
                        disentangle_network_params=disentangle_network_params,
                        bsdf=args.bsdf,
                        aabb=ict_mesh_aabb,
                        device=device)
                params = list(shader.parameters()) 

                if args.weight_albedo_regularization > 0:
                    from robust_loss_pytorch.adaptive import AdaptiveLossFunction
                    _adaptive = AdaptiveLossFunction(num_dims=4, float_dtype=np.float32, device=device)
                    params += list(_adaptive.parameters()) ## need to train it

                optimizer_shader = torch.optim.Adam(params, lr=args.lr_shader, weight_decay=1e-4)

                optimizer_neural_blendshapes = None
                neural_blendshapes.eval()

            progress_bar.set_description(desc=f'Epoch {epoch}, Iter {iteration}, Stage {stage}')
            losses = {k: torch.tensor(0.0, device=device) for k in loss_weights}
                        
            use_jaw = True
            
            input_image = views_subset["img"].permute(0, 3, 1, 2).to(device)

            return_dict = neural_blendshapes(input_image, views_subset)
            mesh = ict_canonical_mesh.with_vertices(ict_canonical_mesh.vertices)

            deformed_vertices = return_dict[deformed_vertices_key+'_posed']
            deformed_vertices_no_pose = return_dict[deformed_vertices_key]

            d_normals = mesh.fetch_all_normals(deformed_vertices, mesh)

            channels = channels_gbuffer + ['segmentation']
            gbuffers = renderer.render_batch(views_subset['flame_camera'], deformed_vertices.contiguous(), d_normals,
                                    channels=channels, with_antialiasing=True, 
                                    canonical_v=mesh.vertices, canonical_idx=mesh.indices, canonical_uv=ict_facekit.uv_neutral_mesh,
                                    mesh=mesh, target_resolution=target_res
                                    )
            pred_color_masked, cbuffers, gbuffer_mask = shader.shade(gbuffers, views_subset, mesh, args.finetune_color, lgt)

            '''
            2D signal losses
            '''
            if stage > 0:

                landmark_loss, closure_loss = landmark_loss_function(ict_facekit, gbuffers, views_subset, use_jaw, device)
                # mask_loss = mask_loss_function(views_subset["mask"], gbuffer_mask)

                segmentation_gt = downsample_upsample(views_subset["skin_mask"][..., :1], None, (512, 512))
                eyes_segmentation_gt = downsample_upsample(views_subset["skin_mask"][..., 3:4], None, (512, 512))
                mouth_segmentation_gt = downsample_upsample(views_subset["skin_mask"][..., 4:5], None, (512, 512))

                mask_loss_segmentation = mask_loss_function(segmentation_gt, gbuffers['segmentation'])
                shading_loss, pred_color, tonemapped_colors = shading_loss_batch(pred_color_masked, views_subset, views_subset['img'].size(0))

                segmentation_loss = segmentation_loss_function(eyes_segmentation_gt, gbuffers['eyes']) + \
                                    segmentation_loss_function(mouth_segmentation_gt, gbuffers['mouth'])

                perceptual_loss = VGGloss(tonemapped_colors[0], tonemapped_colors[1], iteration)

                normal_laplacian_loss = normal_loss(gbuffers, views_subset, gbuffers['segmentation'], device) 
                inverted_normal_loss = inverted_normal_loss_function(gbuffers, views_subset, gbuffer_mask, device)
                eyeball_normal_loss = eyeball_normal_loss_function(gbuffers, views_subset, gbuffer_mask, device)

                losses['mask'] += mask_loss_segmentation + segmentation_loss * 1e1
                losses['landmark'] += landmark_loss 
                # losses['closure'] += closure_loss

                if stage < 2:
                    shading_loss *= 0.1
                    perceptual_loss *= 0.1
                losses['shading'] = shading_loss
                losses['perceptual_loss'] = perceptual_loss 


                ## ======= regularization color ========
                losses['albedo_regularization'] = albedo_regularization(_adaptive, shader, mesh, device, None, iteration)
                losses['white_lgt_regularization'] = white_light(cbuffers)
                losses['roughness_regularization'] = roughness_regularization(cbuffers["roughness"], views_subset["skin_mask"], views_subset["mask"], r_mean=args.r_mean)
                losses["fresnel_coeff"] = spec_intensity_regularization(cbuffers["ko"], views_subset["skin_mask"], views_subset["mask"])
                losses['normal_laplacian'] = normal_laplacian_loss + inverted_normal_loss + eyeball_normal_loss 

            if stage < 5:
                '''
                Regularizations
                '''
                # laplacian regularization
                template_mesh_laplacian_regularization = laplacian_loss_two_meshes(mesh, ict_facekit.neutral_mesh_canonical[0], return_dict['template_mesh'], filtered_lap, head_index =ict_canonical_mesh.vertices.shape[0]) 
                expression_mesh_laplacian_regularization = laplacian_loss_two_meshes(mesh, return_dict['ict_mesh_w_temp'], return_dict['expression_mesh'], filtered_lap, head_index =ict_canonical_mesh.vertices.shape[0]) 

                # normal regularization
                # template_mesh_normal_regularization = normal_reg_loss(mesh, ict_canonical_mesh.with_vertices(ict_facekit.neutral_mesh_canonical[0]), ict_canonical_mesh.with_vertices(return_dict['template_mesh']), head_index =ict_canonical_mesh.vertices.shape[0])
                # expression_mesh_normal_regularization = torch.tensor(0., device=device)
                # for i in range(return_dict['expression_mesh'].shape[0]):
                    # expression_mesh_normal_regularization += normal_reg_loss(mesh, ict_canonical_mesh.with_vertices(return_dict['expression_mesh'][i]), ict_canonical_mesh.with_vertices(return_dict['ict_mesh_w_temp'][i]), head_index =ict_canonical_mesh.vertices.shape[0])  
                # expression_mesh_normal_regularization /= return_dict['expression_mesh'].shape[0] 

                # more regularizations
                feature_regularization = feature_regularization_loss(return_dict['features'], views_subset['mp_blendshape'][..., ict_facekit.mediapipe_to_ict],
                                                                    neural_blendshapes, None, views_subset, dataset_train.bshapes_mode[ict_facekit.mediapipe_to_ict], rot_mult=1, mult=1e1)

                random_blendshapes = torch.rand(views_subset['mp_blendshape'].shape[0], 53, device=device)
                expression_delta_random = neural_blendshapes.get_expression_delta(blendshapes=random_blendshapes)

                l1_regularization = expression_delta_random[53:].pow(2).mean()

                template_geometric_regularization = (ict_facekit.neutral_mesh_canonical[0] - return_dict['template_mesh']).pow(2).mean()
                expression_geometric_regularization = (return_dict['ict_mesh_w_temp'] - return_dict['expression_mesh']).pow(2).mean() 
                

                losses['feature_regularization'] = feature_regularization
                losses['laplacian_regularization'] = template_mesh_laplacian_regularization + expression_mesh_laplacian_regularization 
                # losses['normal_regularization'] = template_mesh_normal_regularization + expression_mesh_normal_regularization
                losses['geometric_regularization'] = template_geometric_regularization + expression_geometric_regularization 
                losses['linearity_regularization'] = l1_regularization

                # adding temporal regularization, which have save weight to feature regularization
                # objective: get items from dataset - for 'idx', randomly add 1 or -1 to the index, clamp it to 0 and len(dataset). 
                # cur_idx = views_subset['idx']
                # next_idx = torch.clamp(cur_idx + 1, 0, len(dataset_train) - 1)
                # next_idx_views = dataset_train.collate([dataset_train[id_] for id_ in next_idx])
                # next_idx_features = neural_blendshapes.encoder(next_idx_views)

                # losses['temporal_regularization'] = (next_idx_features - return_dict['features']).pow(2).mean() * 1e-2

            '''
            FLAME regularization
            '''
            if stage < 4:
                flame_loss = FLAME_loss_function(FLAMEServer, views_subset['flame_expression'], views_subset['flame_pose'], deformed_vertices, flame_pair_indices, ict_pair_indices, target_range=target_range)
                
                zero_pose_w_jaw = torch.zeros_like(views_subset['flame_pose'])
                zero_pose_w_jaw[:, 6:9] = views_subset['flame_pose'][:, 6:9]

                flame_loss_no_pose = FLAME_loss_function(FLAMEServer, views_subset['flame_expression'], zero_pose_w_jaw, deformed_vertices_no_pose, flame_pair_indices, ict_pair_indices, target_range=target_range)

                losses['flame_regularization'] = flame_loss + flame_loss_no_pose


            decay_keys = ['mask', 'landmark', 'flame_regularization']
            
            loss = torch.tensor(0., device=device) 
            for k, v in losses.items():
                if torch.isnan(v).any():
                    print(f'NAN in {k}')
                    print(losses)
                    exit()
                    continue
                    
                loss += v.mean() * loss_weights[k]
                        

            # decay value is  the ten times of summation of loss for mask, segmentation, landmark, closure. 
            with torch.no_grad():
                decay_value = 0
                for k in decay_keys:
                    decay_value += losses[k].mean() * loss_weights[k]
                decay_value = decay_value.clamp(1e-4).pow(0.5)

                for idx in views_subset['idx']:    
                    importance[idx] = ((1 - weight_decay_rate) * importance[idx] + weight_decay_rate * decay_value).clamp(min=1e-2).item()
                
            acc_losses.append(losses)
            acc_total_loss += loss.detach()

            if len(acc_losses) > 9:
                losses_to_log = {}
                for k in acc_losses[0].keys():
                    val = torch.stack([l[k] for l in acc_losses]).mean() * loss_weights[k]
                    # if val > 0:
                    losses_to_log[k] = val
                losses_to_log["total_loss"] = acc_total_loss / len(acc_losses)

                if 'debug' not in run_name and not args.skip_wandb:
                    wandb.log({k: v.item() for k, v in losses_to_log.items() if v > 0}, step=iteration)
                    
                for k, v in losses_to_log.items():
                    del v
                del losses_to_log
                acc_losses = []
                acc_total_loss = 0

            if iteration % 200 == 1:
                print(return_dict['features'][0, 53:53+9])
                print("=="*50)
                for k, v in losses.items():
                    # if k in losses_to_print:
                    v = v.mean()
                # if v > 0:
                    print(f"{k}: {v.item() * loss_weights[k]}")
                print("=="*50)


            # ==============================================================================================
            # Optimizer step
            # ==============================================================================================
            # if not pretrain:
            neural_blendshapes.zero_grad()
            shader.zero_grad()
            optimizer_shader.zero_grad()
            if optimizer_neural_blendshapes is not None:
                optimizer_neural_blendshapes.zero_grad()

            loss.backward()
            torch.cuda.synchronize()

            clip_grad(neural_blendshapes, shader, norm=1.0)

            if optimizer_neural_blendshapes is not None:
                optimizer_neural_blendshapes.step() 

            optimizer_shader.step()

            progress_bar.set_postfix({'loss': loss.detach().cpu().item(), })

            for k, v in losses.items():
                del v
            del loss, losses
            torch.cuda.empty_cache()

            # ==============================================================================================
            # V I S U A L I Z A T I O N S
            # ==============================================================================================
            if (args.visualization_frequency > 0) and (iteration == 1 or iteration % args.visualization_frequency == 0):
            
                with torch.no_grad():
                    neural_blendshapes.eval()
                    shader.eval()

                    return_dict_ = neural_blendshapes(views_subset['img'], views_subset)

                    visualize_specific_traininig('expression_mesh', return_dict_, renderer, shader, ict_facekit, views_subset, mesh, 'training', images_save_path, iteration, lgt)
                    

                    return_dict_ = neural_blendshapes(debug_views['img'], debug_views)

                    bshapes = return_dict_['features'][:, :53].detach().cpu().numpy()
                    bshapes = np.round(bshapes, 2)
                    jawopen = bshapes[:, ict_facekit.expression_names.tolist().index('jawOpen')]
                    eyeblink_l = bshapes[:, ict_facekit.expression_names.tolist().index('eyeBlink_L')]
                    eyeblink_r = bshapes[:, ict_facekit.expression_names.tolist().index('eyeBlink_R')]
                    
                    print(f"JawOpen: {jawopen}, EyeBlink_L: {eyeblink_l}, EyeBlink_R: {eyeblink_r}")

                    visualize_specific_traininig('ict_mesh_w_temp', return_dict_, renderer, shader, ict_facekit, debug_views, mesh, 'ict_w_temp', images_save_path, iteration, lgt)

                    debug_gbuffer = visualize_specific_traininig('expression_mesh', return_dict_, renderer, shader, ict_facekit, debug_views, mesh, 'expression', images_save_path, iteration, lgt)


                    for ith in range(debug_views['img'].shape[0]):
                        seg = debug_gbuffer['segmentation'][ith, ..., 0]
                        seg = seg * 255
                        seg = seg.cpu().numpy().astype(np.uint8)
                        gt_seg = debug_views['skin_mask'][ith, ..., 0]
                        gt_seg = gt_seg * 255
                        gt_seg = gt_seg.cpu().numpy().astype(np.uint8)

                        seg = np.stack([seg, gt_seg, np.zeros(seg.shape)], axis=2)

                        cv2.imwrite(str(images_save_path / "grid" / f'grid_{iteration}_seg_narrow_{ith}.png'), seg)

                        bshapes = return_dict_['features'][ith, :53].detach().cpu().numpy()
                        bshapes = np.round(bshapes, 2)
                        
                        names = ict_facekit.expression_names.tolist()

                        save_blendshape_figure(bshapes, names, f"Blendshapes Activation_{ith}", images_save_path / "grid" / f"grid_{iteration}_blendshapes_activation_{ith}.png")
                                            

                        n = ith                     
                        if n == 0: 
                            write_mesh(meshes_save_path / f"mesh_{iteration:06d}_exp.obj", mesh.with_vertices(return_dict_['expression_mesh'][n]).detach().to('cpu'))                    
                            # save the posed meshes as well
                            write_mesh(meshes_save_path / f"mesh_{iteration:06d}_exp_posed.obj", mesh.with_vertices(return_dict_['expression_mesh_posed'][n]).detach().to('cpu'))

                    write_mesh(meshes_save_path / f"mesh_{iteration:06d}_temp.obj", mesh.with_vertices(return_dict_['template_mesh']).detach().to('cpu'))                                
                    del debug_gbuffer, return_dict_

                    neural_blendshapes.train()
                    shader.train()

            if iteration == 1 or iteration % (args.visualization_frequency * 10) == 0:
                print(images_save_path / "grid" / f'grid_{iteration}.png')
                if 'debug' not in run_name and not args.skip_wandb:
                    wandb.log({"Grid": [wandb.Image(str(images_save_path / "grid" / f'grid_{iteration}_expression.png'))]}, step=iteration)

            ## ============== save intermediate ==============================
            if (args.save_frequency > 0) and (iteration == 1 or iteration % args.save_frequency == 0):
                with torch.no_grad():
                    shader.save(shaders_save_path / f'shader.pt')
                    neural_blendshapes.save(shaders_save_path / f'neural_blendshapes.pt')

            
    end = time.time()
    total_time = ((end - start) % 3600)
    print("TIME TAKEN (mins):", int(total_time // 60))

    if 'debug' not in run_name and not args.skip_wandb:
        wandb.finish()

    # ==============================================================================================
    # s a v e
    # ==============================================================================================
    with open(experiment_dir / "args.txt", "w") as text_file:
        print(f"{args}", file=text_file)
    shader.save(shaders_save_path / f'shader_latest.pt')
    neural_blendshapes.save(shaders_save_path / f'neural_blendshapes_latest.pt')

    # ==============================================================================================
    # FINAL: qualitative and quantitative results
    # ==============================================================================================

    ## ============== free memory before evaluation ==============================
    del dataset_train, dataloader_train, debug_views, views_subset

    if not args.skip_eval:

        print("=="*50)
        print("E V A L U A T I O N")
        print("=="*50)
        dataset_val      = DatasetLoader(args, train_dir=args.eval_dir, sample_ratio=1 if 'debug' not in run_name else 50, pre_load=False)

        dataloader_validate = torch.utils.data.DataLoader(dataset_val, batch_size=4, collate_fn=dataset_val.collate)

        quantitative_eval(args, mesh, dataloader_validate, ict_facekit, neural_blendshapes, shader, renderer, device, channels_gbuffer, experiment_dir
                        , images_eval_save_path / "qualitative_results", lgt=lgt, save_each=True)

if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()

    # Select the device
    device = torch.device('cpu')
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    print(f"Using device {device}")

    # ==============================================================================================
    # load data
    # ==============================================================================================
    print("loading train views...")
    dataset_train    = DatasetLoader(args, train_dir=args.train_dir, sample_ratio=args.sample_idx_ratio, pre_load=False, train=True)
    dataset_val      = DatasetLoader(args, train_dir=args.eval_dir, sample_ratio=24, pre_load=False)
    
    dataloader_train = None
    view_indices = np.array(args.visualization_views).astype(int)
    d_l = [dataset_val.__getitem__(idx) for idx in view_indices[2:]]
    d_l.append(dataset_train.__getitem__(view_indices[0]))
    d_l.append(dataset_train.__getitem__(view_indices[1]))
    debug_views = dataset_val.collate(d_l)

    # print(dataset_val[0]['flame_camera'].t, dataset_val[0]['flame_camera'].R)
    # print(dataset_train[0]['flame_camera'].t, dataset_train[0]['flame_camera'].R)
    # print(dataset_train[-1]['flame_camera'].t, dataset_train[-1]['flame_camera'].R)
    
    # exit()
    # # for i in range(100), save dataset_train[i]['img'] on debug/trainset
    # os.makedirs("debug/trainset", exist_ok=True)
    # for i in range(100):
    #     json_dict = dataset_train.all_img_path[i]
        
    #     print(json_dict["dir"] / Path(json_dict["file_path"] + ".png"))
    #     # shape of image

    # exit()
        


    del dataset_val
    
    # lbs = dataset_train.get_bshapes_lower_bounds()
    # print(lbs)
    
    # exit()

    # ==============================================================================================
    # main run
    # ==============================================================================================
    import time
    while True:
        try:
            main(args, device, dataset_train, dataloader_train, debug_views)
            break  # Exit the loop if main() runs successfully
        except ValueError as e:
            print(e)
            print("--"*50)
            print("Warning: Re-initializing main() because the training of light MLP diverged and all the values are zero. If the training does not restart, please end it and restart. ")
            print("--"*50)
            raise e
            time.sleep(5)

        except Exception as e:
            print(e)
            print('Error: Unexpected error occurred. Aborting the training.')
            raise e

    