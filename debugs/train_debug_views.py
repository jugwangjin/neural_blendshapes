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

    
    # print(tight_face_ict_flame)
    # print(ict_pair_indices[tight_face_ict_flame])
    # print(tight_face_ict_flame.shape)

    # exit()

    ## ============== load ict facekit ==============================
    ict_facekit = ICTFaceKitTorch(npy_dir = './assets/ict_facekit_torch.npy', canonical = Path(args.input_dir) / 'ict_identity.npy')
    ict_facekit = ict_facekit.to(device)

    ict_canonical_mesh = Mesh(ict_facekit.canonical[0].cpu().data, ict_facekit.faces.cpu().data, ict_facekit=ict_facekit, device=device)
    ict_canonical_mesh.compute_connectivity()

    write_mesh(Path(meshes_save_path / "init_ict_canonical.obj"), ict_canonical_mesh.to('cpu'))


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
    neural_blendshapes_pe = list(neural_blendshapes.fourier_feature_transform.parameters())
    neural_blendshapes_pose_weight_params = list(neural_blendshapes.pose_weight.parameters())
    neural_blendshapes_others_params = list(set(neural_blendshapes_params) - set(neural_blendshapes_expression_params) - set(neural_blendshapes_template_params) - set(neural_blendshapes_pe) - set(neural_blendshapes_pose_weight_params)) 
    optimizer_neural_blendshapes = torch.optim.Adam([
                                                    {'params': neural_blendshapes_others_params, 'lr': args.lr_deformer},
                                                    # {'params': neural_blendshapes_expression_params, 'lr': args.lr_jacobian},
                                                    {'params': neural_blendshapes_template_params, 'lr': args.lr_jacobian},
                                                    {'params': neural_blendshapes_pe, 'lr': args.lr_jacobian },
                                                    {'params': neural_blendshapes_pose_weight_params, 'lr': args.lr_jacobian},
                                                    ],
                                                    # ], betas=(0.05, 0.1)
                                                    )
                                                     
    scheduler_milestones = [args.iterations*2]
    scheduler_gamma = 0.25

    scheduler_neural_blendshapes = torch.optim.lr_scheduler.MultiStepLR(optimizer_neural_blendshapes, milestones=scheduler_milestones, gamma=scheduler_gamma)



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
    params = list(shader.parameters()) 

    if args.weight_albedo_regularization > 0:
        from robust_loss_pytorch.adaptive import AdaptiveLossFunction
        _adaptive = AdaptiveLossFunction(num_dims=4, float_dtype=np.float32, device=device)
        params += list(_adaptive.parameters()) ## need to train it

    optimizer_shader = torch.optim.Adam(params, lr=args.lr_shader)

    scheduler_shader = torch.optim.lr_scheduler.MultiStepLR(optimizer_shader, milestones=scheduler_milestones, gamma=scheduler_gamma)
    
    # ==============================================================================================
    # Loss Functions
    # ==============================================================================================
    # Initialize the loss weights and losses
    loss_weights = {
        "mask": args.weight_mask,
        "laplacian_regularization": args.weight_laplacian_regularization,
        "normal_regularization": args.weight_normal_regularization,
        "shading": args.weight_shading,
        "perceptual_loss": args.weight_perceptual_loss,
        "landmark": args.weight_landmark,
        "closure": args.weight_closure,
        "feature_regularization": args.weight_feature_regularization,
        "geometric_regularization": args.weight_geometric_regularization,
        "normal_laplacian": args.weight_normal_laplacian,
        "linearity_regularization": 1e-1,
        "identity_weight_regularization": 1e-3,
        "flame_regularization": args.weight_flame_regularization,
        "white_lgt_regularization": args.weight_white_lgt_regularization,
        "roughness_regularization": args.weight_roughness_regularization,
        "fresnel_coeff": args.weight_fresnel_coeff,
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


    losses = {k: torch.tensor(0.0, device=device) for k in loss_weights}

    epochs = (args.iterations // len(dataloader_train)) + 1
    iteration = 0
    
    progress_bar = tqdm(range(epochs))
    start = time.time()

    acc_losses = []
    acc_total_loss = 0

    weight_decay_rate = 0.05

    filtered_lap = compute_laplacian_uniform_filtered(ict_canonical_mesh, head_index=ict_canonical_mesh.vertices.shape[0])

            
    epochs = 5 if 'debug' not in run_name else 0
    epochs = 3

    epochs = (args.iterations // len(dataloader_train)) + 1
    epochs = epochs // 2
    # epochs = 2
    # epochs = (args.iterations // len(dataloader_train)) + 1 
    epochs = epochs if 'debug' not in run_name else 0

    progress_bar = tqdm(range(epochs))
    start = time.time()

    with torch.no_grad():

        mesh = ict_canonical_mesh.with_vertices(ict_canonical_mesh.vertices)
        return_dict_ = neural_blendshapes(debug_views['img'], debug_views)


        debug_gbuffer = renderer.render_batch(debug_views['flame_camera'], return_dict_['ict_mesh_w_temp_posed'].contiguous(), mesh.fetch_all_normals(return_dict_['ict_mesh_w_temp_posed'], mesh), 
                                channels=channels_gbuffer+['segmentation'], with_antialiasing=True, 
                                canonical_v=mesh.vertices, canonical_idx=mesh.indices, canonical_uv=ict_facekit.uv_neutral_mesh) 
        debug_rgb_pred, debug_cbuffers, _ = shader.shade(debug_gbuffer, debug_views, mesh, args.finetune_color, lgt)
        visualize_training(debug_rgb_pred, debug_cbuffers, debug_gbuffer, debug_views, images_save_path, 0, ict_facekit=ict_facekit, save_name='only_flame')

        filename = os.path.join(images_save_path, "grid", f"grid_0_only_flame.png")
        target_filename = os.path.join(images_save_path, "grid", f"a_ict_init.png")

        os.rename(filename, target_filename)

        for ith in range(return_dict_['features'].shape[0]):
            bshapes = return_dict_['features'][ith, :53].detach().cpu().numpy()
            bshapes = np.round(bshapes, 2)
            
            names = ict_facekit.expression_names.tolist()
            
            plt.clf()
            plt.figsize=(8,8)
            plt.bar(np.arange(53), bshapes)
            plt.xticks(np.arange(53), names, rotation=90, fontsize=6)
            plt.title(f"Blendshapes Activation_{ith}")
            plt.savefig(str(images_save_path / "grid" / f"flame_init_blendshapes_activation_{ith}.png"))

            bshapes = debug_views['mp_blendshape'][ith, ict_facekit.mediapipe_to_ict].detach().cpu().numpy()
            bshapes = np.round(bshapes, 2)

            plt.clf()
            plt.figsize=(8,8)
            plt.bar(np.arange(53), bshapes)
            plt.xticks(np.arange(53), names, rotation=90, fontsize=6)
            plt.title(f"Blendshapes Activation_{ith}")
            plt.savefig(str(images_save_path / "grid" / f"mediapipe_blendshapes_activation_{ith}.png"))
                
        flame_gt, _, _ = FLAMEServer(debug_views['flame_expression'], debug_views['flame_pose'])

        for num_flame in range(flame_gt.shape[0]):
            flame_tmp_mesh = o3d.geometry.TriangleMesh()
            flame_tmp_mesh.vertices = o3d.utility.Vector3dVector(flame_gt[num_flame].cpu().numpy())
            flame_tmp_mesh.triangles = o3d.utility.Vector3iVector(FLAMEServer.faces_tensor.cpu().numpy())
            o3d.io.write_triangle_mesh(f'debug/debug_view_{num_flame}_flame_gt.obj', flame_tmp_mesh)

            ict_tmp_mesh = o3d.geometry.TriangleMesh()
            ict_tmp_mesh.vertices = o3d.utility.Vector3dVector(return_dict_['ict_mesh_w_temp_posed'][num_flame].cpu().numpy())
            ict_tmp_mesh.triangles = o3d.utility.Vector3iVector(ict_facekit.faces.cpu().numpy())
            o3d.io.write_triangle_mesh(f'debug/debug_view_{num_flame}_ict_gt.obj', ict_tmp_mesh)
            

            flame_tmp_gt = flame_gt[num_flame, flame_pair_indices]
            ict_tmp_taget = return_dict_['ict_mesh_w_temp_posed'][num_flame, ict_pair_indices]

            lineset = o3d.geometry.LineSet()
            lineset.points = o3d.utility.Vector3dVector(torch.cat([flame_tmp_gt, ict_tmp_taget], 0).cpu().numpy())
            lineset.lines = o3d.utility.Vector2iVector(np.array([np.arange(flame_tmp_gt.shape[0]), np.arange(flame_tmp_gt.shape[0], flame_tmp_gt.shape[0] + ict_tmp_taget.shape[0])]).T)
            o3d.io.write_line_set(f'debug/debug_view_{num_flame}_pair.ply', lineset)



        for n in range(debug_views['flame_pose'].shape[1]):
            pose = torch.zeros_like(debug_views['flame_pose'])[:1]
            pose[0, n] = 1

            flame_gt, _, _ = FLAMEServer(torch.zeros_like(debug_views['flame_expression'][:1]), pose)
            flame_tmp_mesh = o3d.geometry.TriangleMesh()
            flame_tmp_mesh.vertices = o3d.utility.Vector3dVector(flame_gt[0].cpu().numpy())
            flame_tmp_mesh.triangles = o3d.utility.Vector3iVector(FLAMEServer.faces_tensor.cpu().numpy())
            o3d.io.write_triangle_mesh(f'debug/debug_view_pose_{n}_flame_gt.obj', flame_tmp_mesh)

        # fmesh = Mesh(vertices = FLAMEServer.canonical_verts[0], indices=FLAMEServer.faces_tensor, device=device)

        # fmesh.compute_connectivity()
        # fdeformed_vertices, _, _ = FLAMEServer(debug_views['flame_expression'], debug_views['flame_pose'])
        # fd_normals = fmesh.fetch_all_normals(fdeformed_vertices, fmesh)

        # return_dict_ = neural_blendshapes(debug_views['img'], debug_views)
        

        # debug_gbuffer = renderer.render_batch(debug_views['flame_camera'], fdeformed_vertices.contiguous(), fd_normals, 
        #                         channels=channels_gbuffer+['segmentation'], with_antialiasing=True, 
        #                         canonical_v=fmesh.vertices, canonical_idx=fmesh.indices, canonical_uv=ict_facekit.uv_neutral_mesh) 
        # debug_rgb_pred, debug_cbuffers, _ = shader.shade(debug_gbuffer, debug_views, fmesh, args.finetune_color, lgt)
        # visualize_training_no_lm(debug_rgb_pred, debug_cbuffers, debug_gbuffer, debug_views, images_save_path, 0, ict_facekit=ict_facekit, save_name='only_flame')

        # filename = os.path.join(images_save_path, "grid", f"grid_0_only_flame.png")
        # target_filename = os.path.join(images_save_path, "grid", f"a_flame_canonical_init.png")

        # os.rename(filename, target_filename)


    # exit()

    bshapes_multipliers = []
    for i in range(53):
        bshapes_multipliers.append([])

    use_jaw=True

    iteration = 0
    
    for epoch in progress_bar:
        for iter_, views_subset in tqdm(enumerate(dataloader_train)):
            iteration += 1

            # if iteration == args.iterations // 4:
            #     optimizer_neural_blendshapes = torch.optim.Adam([
            #                                     {'params': neural_blendshapes_others_params, 'lr': args.lr_deformer},
            #                                     # {'params': neural_blendshapes_expression_params, 'lr': args.lr_jacobian},
            #                                     {'params': neural_blendshapes_template_params, 'lr': args.lr_jacobian},
            #                                     {'params': neural_blendshapes_pe, 'lr': args.lr_jacobian },
            #                                     {'params': neural_blendshapes_pose_weight_params, 'lr': args.lr_jacobian},
            #                                     ],
            #                                     # ], betas=(0.05, 0.1)
            #                                     )
            progress_bar.set_description(desc=f'Epoch {epoch}, Iter {iteration}')
            losses = {k: torch.tensor(0.0, device=device) for k in loss_weights}

                # def collate(self, batch):
        # return {
        #     'img' : torch.cat(list([item['img'] for item in batch]), dim=0).to(device),
        #     'mask' : torch.cat(list([item['mask'] for item in batch]), dim=0).to(device),
        #     'skin_mask' : torch.cat(list([item['skin_mask'] for item in batch]), dim=0).to(device),
        #     'camera': list([item['camera'] for item in batch]),
        #     'frame_name': list([item['frame_name'] for item in batch]),
        #     'idx': torch.LongTensor(list([item['idx'] for item in batch])).to(device),
        #     'landmark' : torch.cat(list([item['landmark'] for item in batch]), dim=0).to(device),
        #     'mp_landmark': torch.cat(list([item['mp_landmark'] for item in batch]), dim=0).to(device),
        #     'mp_blendshape' : torch.cat(list([item['mp_blendshape'] for item in batch]), dim=0).to(device),
        #     'mp_transform_matrix' : torch.cat(list([item['mp_transform_matrix'] for item in batch]), dim=0).to(device),
        #     'normal' : torch.cat(list([item['normal'] for item in batch]), dim=0).to(device),
        #     'flame_expression' : torch.cat(list([item['flame_expression'] for item in batch]), dim=0).to(device),
        #     'flame_pose' : torch.cat(list([item['flame_pose'] for item in batch]), dim=0).to(device),
        #     'flame_camera': list([item['flame_camera'] for item in batch]),
        # }

            idx = iteration % len(debug_views['camera'])
            
            views_subset = {}
            for k in debug_views:
                if k in ['camera, frame_name', 'flame_camera']:
                    views_subset[k] = [debug_views[k][idx]]
                else:
                    views_subset[k] = debug_views[k][idx:idx+1]

            input_image = views_subset["img"].permute(0, 3, 1, 2).to(device)
            
            return_dict = neural_blendshapes(input_image, views_subset, pretrain=True)
            mesh = ict_canonical_mesh.with_vertices(ict_canonical_mesh.vertices)

            # template optimization
            ict_mesh_w_temp_posed = return_dict['ict_mesh_w_temp_posed']
            d_normals = mesh.fetch_all_normals(ict_mesh_w_temp_posed, mesh)
            deformed_vertices = ict_mesh_w_temp_posed

            gbuffers = renderer.render_batch(views_subset['flame_camera'], deformed_vertices.contiguous(), d_normals,
                                    channels=channels_gbuffer+['segmentation'], with_antialiasing=True, 
                                    canonical_v=mesh.vertices, canonical_idx=mesh.indices, canonical_uv=ict_facekit.uv_neutral_mesh)
            _, _, gbuffer_mask = shader.shade(gbuffers, views_subset, mesh, args.finetune_color, lgt)
            # _, _, gbuffer_mask = shader.get_mask(gbuffers, views_subset, mesh, args.finetune_color, lgt)
            
            # print(views_subset['camera'][0].K - views_subset['flame_camera'][0].K)
            # print(views_subset['camera'][0].R - views_subset['flame_camera'][0].R)
            # print(views_subset['camera'][0].t - views_subset['flame_camera'][0].t)
            
            # with torch.no_grad():
            flame_gt, _, _ = FLAMEServer(views_subset['flame_expression'], views_subset['flame_pose'])
                
                # flame_gt_clip_space = renderer.get_vertices_clip_space_from_view(views_subset['flame_camera'], flame_gt)

                # ict_gt = flame_gt_clip_space[:, flame_pair_indices]
            ict_gt = flame_gt[:, flame_pair_indices]
                # if pretrain:
            flame_loss = (deformed_vertices[:, ict_pair_indices] * 1 - ict_gt * 1) 

            flame_loss = flame_loss.pow(2).mean()

            flame_loss = flame_loss


            zero_pose_w_jaw = torch.zeros_like(views_subset['flame_pose'])
            zero_pose_w_jaw[:, 6:9] = views_subset['flame_pose'][:, 6:9]
            flame_gt_no_pose, _, _ = FLAMEServer(views_subset['flame_expression'], zero_pose_w_jaw)

            ict_gt = flame_gt_no_pose[:, flame_pair_indices]

            flame_loss_no_pose = (return_dict['ict_mesh_w_temp'][:, ict_pair_indices] * 1 - ict_gt * 1)
            flame_loss_no_pose = flame_loss_no_pose.pow(2).mean()

            template_mesh_laplacian_regularization = laplacian_loss_two_meshes(mesh, ict_facekit.neutral_mesh_canonical[0], return_dict['template_mesh'], filtered_lap, head_index=ict_canonical_mesh.vertices.shape[0]) 
            feature_regularization = feature_regularization_loss(return_dict['features'], views_subset['mp_blendshape'][..., ict_facekit.mediapipe_to_ict],
                                                                neural_blendshapes, return_dict['bshape_modulation'], facs_weight=0)

            template_geometric_regularization = (ict_facekit.neutral_mesh_canonical[0] - return_dict['template_mesh']).pow(2).mean()
            template_mesh_normal_regularization = normal_reg_loss(mesh, ict_canonical_mesh.with_vertices(ict_facekit.neutral_mesh_canonical[0]), ict_canonical_mesh.with_vertices(return_dict['template_mesh']), head_index =ict_canonical_mesh.vertices.shape[0])
            # eyeball_normal_loss = eyeball_normal_loss_function(debug_gbuffer, debug_views, ict_facekit.skin_mask, device)


            inverted_normal_loss = inverted_normal_loss_function(gbuffers, views_subset, gbuffer_mask, device)
            eyeball_normal_loss = eyeball_normal_loss_function(gbuffers, views_subset, gbuffer_mask, device)
            # landmark_loss, closure_loss = landmark_loss_function(ict_facekit, gbuffers, views_subset, use_jaw, device)

        
            losses['laplacian_regularization'] = template_mesh_laplacian_regularization 
            losses['feature_regularization'] = feature_regularization  
            losses['flame_regularization'] = flame_loss + flame_loss_no_pose
            losses['geometric_regularization'] = template_geometric_regularization
            losses['normal_regularization'] = template_mesh_normal_regularization 
            losses['normal_laplacian'] = inverted_normal_loss +  eyeball_normal_loss 
            # losses['landmark'] = landmark_loss
            # losses['closure'] = closure_loss


            loss = torch.tensor(0., device=device) 
            for k, v in losses.items():
                if torch.isnan(v).any():
                    print(f'NAN in {k}')
                    print(losses)
                    exit()
                    continue
                loss += v.mean() * loss_weights[k]
            
            loss += neural_blendshapes.encoder.identity_weights.pow(2).mean() * 1e-3
        
            if iteration % 100 == 1:
                
                print(return_dict['features'][0, 53:])
                print(neural_blendshapes.encoder.transform_origin)
                print("=="*50)
                keys = ['laplacian_regularization', 'feature_regularization', 'flame_regularization', 'geometric_regularization', 'normal_regularization', 'normal_laplacian']
                for k in keys:
                    # if k in losses_to_print:
                    v = losses[k]
                    v = v.mean()
                # if v > 0:
                    print(f"{k}: {v.item() * loss_weights[k]}")

                for nn in range(53):
                    bshapes_multipliers[nn].append(1 + neural_blendshapes.encoder.softplus(neural_blendshapes.encoder.bshapes_multiplier).cpu().data.numpy()[nn])

                    plt.clf()
                    plt.plot(bshapes_multipliers[nn])
                    plt.title(f"BSMult_{nn}_{ict_facekit.expression_names.tolist()[nn]}")
                    plt.savefig(str(images_save_path / f"bsmult_{nn}_{ict_facekit.expression_names.tolist()[nn]}.png"))


                bshapes = return_dict['features'][:, :53].detach().cpu().numpy()
                bshapes = np.round(bshapes, 2)
                jawopen = bshapes[:, ict_facekit.expression_names.tolist().index('jawOpen')]
                eyeblink_l = bshapes[:, ict_facekit.expression_names.tolist().index('eyeBlink_L')]
                eyeblink_r = bshapes[:, ict_facekit.expression_names.tolist().index('eyeBlink_R')]
                
                m_jawopen = neural_blendshapes.encoder.bshapes_multiplier[ict_facekit.expression_names.tolist().index('jawOpen')]
                m_eyeblink_l = neural_blendshapes.encoder.bshapes_multiplier[ict_facekit.expression_names.tolist().index('eyeBlink_L')]
                m_eyeblink_r = neural_blendshapes.encoder.bshapes_multiplier[ict_facekit.expression_names.tolist().index('eyeBlink_R')]
                print(f"MJawOpen: {m_jawopen}, MEyeBlink_L: {m_eyeblink_l}, MEyeBlink_R: {m_eyeblink_r}")
                print(f"JawOpen: {jawopen}, EyeBlink_L: {eyeblink_l}, EyeBlink_R: {eyeblink_r}")


                print("=="*50)

            # ==============================================================================================
            # Optimizer step
            # ==============================================================================================
            # if not pretrain:
            neural_blendshapes.zero_grad()
            shader.zero_grad()
            optimizer_shader.zero_grad()
            optimizer_neural_blendshapes.zero_grad()

            loss.backward()



            torch.cuda.synchronize()
            
            if args.grad_scale and args.fourier_features == "hashgrid":
                shader.fourier_feature_transform.params.grad /= 8.0
                
            clip_grad(neural_blendshapes, shader, norm=5.0)
            
            optimizer_neural_blendshapes.step()
            scheduler_neural_blendshapes.step()
            # else:

            optimizer_shader.step()
            scheduler_shader.step()


            # exit()

            for k, v in losses.items():
                del v
            del loss, losses
            torch.cuda.empty_cache()

        with torch.no_grad():

            return_dict_ = neural_blendshapes(debug_views['img'], debug_views)
            

            debug_gbuffer = renderer.render_batch(debug_views['flame_camera'], return_dict_['ict_mesh_w_temp_posed'].contiguous(), mesh.fetch_all_normals(return_dict_['ict_mesh_w_temp_posed'], mesh), 
                                    channels=channels_gbuffer+['segmentation'], with_antialiasing=True, 
                                    canonical_v=mesh.vertices, canonical_idx=mesh.indices, canonical_uv=ict_facekit.uv_neutral_mesh) 
            debug_rgb_pred, debug_cbuffers, _ = shader.shade(debug_gbuffer, debug_views, mesh, args.finetune_color, lgt)
            visualize_training(debug_rgb_pred, debug_cbuffers, debug_gbuffer, debug_views, images_save_path, epoch, ict_facekit=ict_facekit, save_name='only_flame')

            filename = os.path.join(images_save_path, "grid", f"grid_{epoch}_only_flame.png")
            target_filename = os.path.join(images_save_path, "grid", f"a_flame_epoch_{epoch}.png")

            os.rename(filename, target_filename)



            flame_gt, _, _ = FLAMEServer(debug_views['flame_expression'], debug_views['flame_pose'])

            for num_flame in range(flame_gt.shape[0]):
                ict_tmp_mesh = o3d.geometry.TriangleMesh()
                ict_tmp_mesh.vertices = o3d.utility.Vector3dVector(return_dict_['ict_mesh_w_temp_posed'][num_flame].cpu().numpy())
                ict_tmp_mesh.triangles = o3d.utility.Vector3iVector(ict_facekit.faces.cpu().numpy())
                o3d.io.write_triangle_mesh(f'debug/debug_view_{num_flame}_ict_gt_epoch{epoch}.obj', ict_tmp_mesh)
                

                flame_tmp_gt = flame_gt[num_flame, flame_pair_indices]
                ict_tmp_taget = return_dict_['ict_mesh_w_temp_posed'][num_flame, ict_pair_indices]

                lineset = o3d.geometry.LineSet()
                lineset.points = o3d.utility.Vector3dVector(torch.cat([flame_tmp_gt, ict_tmp_taget], 0).cpu().numpy())
                lineset.lines = o3d.utility.Vector2iVector(np.array([np.arange(flame_tmp_gt.shape[0]), np.arange(flame_tmp_gt.shape[0], flame_tmp_gt.shape[0] + ict_tmp_taget.shape[0])]).T)
                o3d.io.write_line_set(f'debug/debug_view_{num_flame}_pair_epoch{epoch}.ply', lineset)




            for ith in range(debug_views['img'].shape[0]):
                seg = debug_gbuffer['segmentation'][ith, ..., 0]
                # seg = debug_gbuffer['narrow_face'][ith, ..., 0]
                seg = seg * 255
                seg = seg.cpu().numpy().astype(np.uint8)
                gt_seg = debug_views['skin_mask'][ith, ..., 0]
                gt_seg = gt_seg * 255
                gt_seg = gt_seg.cpu().numpy().astype(np.uint8)

                seg = np.stack([seg, gt_seg, np.zeros(seg.shape)], axis=2)

                cv2.imwrite(str(images_save_path / "grid" / f'a_flame_epoch_{epoch}_seg_{ith}.png'), seg)
                # draw bar graph of blendshapes.
                # y for activaiton (value)
                # x for each blendshape index.
                # label for each blendshape name.
                bshapes = return_dict_['features'][ith, :53].detach().cpu().numpy()
                bshapes = np.round(bshapes, 2)
                
                names = ict_facekit.expression_names.tolist()

                plt.clf()
                plt.figsize=(8,8)
                plt.bar(np.arange(53), bshapes)
                plt.xticks(np.arange(53), names, rotation=90, fontsize=5)
                plt.title(f"Blendshapes Activation_{ith}")
                plt.savefig(str(images_save_path / "grid" / f"a_flame_epoch_{epoch}_blendshapes_activation_{ith}.png"))
                
    exit()
    # exit()

    import wandb
    if 'debug' not in run_name and not args.skip_wandb:
        wandb_name = args.wandb_name if args.wandb_name is not None else run_name
        wandb.init(project="neural_jacobian_blendshape_full_deform", name=wandb_name, config=args)
    else:
        # define arrays to store the loss values
        # losses_total = []
        losses_keys = {}
        for k in loss_weights:
            losses_keys[k] = []
        losses_keys['total_loss'] = []

            

    epochs = (args.iterations // len(dataloader_train)) + 1
    progress_bar = tqdm(range(epochs))
    start = time.time()

    optimizer_neural_blendshapes = torch.optim.Adam([
                                                    {'params': neural_blendshapes_others_params, 'lr': args.lr_deformer * 0.5},
                                                    {'params': neural_blendshapes_expression_params, 'lr': args.lr_jacobian},
                                                    {'params': neural_blendshapes_template_params, 'lr': args.lr_jacobian},
                                                    {'params': neural_blendshapes_pe, 'lr': args.lr_jacobian},
                                                    {'params': neural_blendshapes_pose_weight_params, 'lr': args.lr_jacobian},
                                                    ],
                                                    # ], betas=(0.05, 0.1)
                                                    )
                                                     

    iteration = 0
    for epoch in progress_bar:
        
        for iter_, views_subset in tqdm(enumerate(dataloader_train)):
            iteration += 1
            progress_bar.set_description(desc=f'Epoch {epoch}, Iter {iteration}')
            losses = {k: torch.tensor(0.0, device=device) for k in loss_weights}

            super_flame = False
            pretrain = iteration < args.iterations // 3
            if iteration == args.iterations // 3:
                neural_blendshapes_tail_params = list(neural_blendshapes.encoder.tail.parameters())
                optimizer_neural_blendshapes = torch.optim.Adam([
                                                                # {'params': neural_blendshapes_tail_params, 'lr': args.lr_deformer * 0.25}, # fix the blendshapes
                                                                {'params': neural_blendshapes_expression_params, 'lr': args.lr_jacobian},
                                                                {'params': neural_blendshapes_template_params, 'lr': args.lr_jacobian},
                                                                {'params': neural_blendshapes_pe, 'lr': args.lr_jacobian},
                                                                {'params': neural_blendshapes_pose_weight_params, 'lr': args.lr_jacobian},
                                                                ],
                                                                # ], betas=(0.05, 0.1)
                                                                )
                                                     
            # pretrain = False


            if iteration == 3 * (args.iterations // 4) and args.fourier_features != "positional":
                del shader
                del optimizer_shader
                del scheduler_shader

                shader = NeuralShader(fourier_features=args.fourier_features,
                        activation=args.activation,
                        last_activation=torch.nn.Sigmoid(), 
                        disentangle_network_params=disentangle_network_params,
                        bsdf=args.bsdf,
                        aabb=ict_mesh_aabb,
                        # existing_encoder = neural_blendshapes.fourier_feature_transform,
                        device=device)
                params = list(shader.parameters()) 

                if args.weight_albedo_regularization > 0:
                    from robust_loss_pytorch.adaptive import AdaptiveLossFunction
                    _adaptive = AdaptiveLossFunction(num_dims=4, float_dtype=np.float32, device=device)
                    params += list(_adaptive.parameters()) ## need to train it

                optimizer_shader = torch.optim.Adam(params, lr=args.lr_shader)

                scheduler_shader = torch.optim.lr_scheduler.MultiStepLR(optimizer_shader, milestones=scheduler_milestones, gamma=scheduler_gamma)

                # args.lr_deformer *= 0.01
                # args.lr_jacobian *= 0.01

                optimizer_neural_blendshapes = None

                # optimizer_neural_blendshapes = torch.optim.Adam([
                #                                                 {'params': neural_blendshapes_others_params, 'lr': args.lr_deformer * 0.2},
                #                                                 {'params': neural_blendshapes_expression_params, 'lr': args.lr_jacobian},
                #                                                 {'params': neural_blendshapes_template_params, 'lr': args.lr_jacobian},
                #                                                 {'params': neural_blendshapes_pe, 'lr': args.lr_jacobian},
                #                                                 {'params': neural_blendshapes_pose_weight_params, 'lr': args.lr_jacobian},
                #                                                 ],
                                                                # ], betas=(0.05, 0.1)
                                                                # )
                                                     


            # flame_loss_full_head for one-third of args.iterations.
            # no_flame loss after two-third of args.iterations.

            flame_loss_full_head = iteration < 0

            flame_loss_tight_face = iteration > (args.iterations // 4)
            no_flame_loss = iteration > (args.iterations // 2)

            use_jaw = True
            # use_jaw = iteration < args.iterations // 20
            

            input_image = views_subset["img"].permute(0, 3, 1, 2).to(device)
            
            return_dict = neural_blendshapes(input_image, views_subset, pretrain=pretrain)
            mesh = ict_canonical_mesh.with_vertices(ict_canonical_mesh.vertices)
            
            
            if pretrain:
                deformed_vertices = return_dict['ict_mesh_w_temp_posed'] 
                d_normals = mesh.fetch_all_normals(deformed_vertices, mesh)
                channels = ['mask', 'canonical_position'] if not pretrain else channels_gbuffer + ['segmentation']
                gbuffers = renderer.render_batch(views_subset['flame_camera'], deformed_vertices.contiguous(), d_normals,
                                        channels=channels, with_antialiasing=True, 
                                        canonical_v=mesh.vertices, canonical_idx=mesh.indices, canonical_uv=ict_facekit.uv_neutral_mesh)
                if pretrain:
                    pred_color_masked, cbuffers, gbuffer_mask = shader.shade(gbuffers, views_subset, mesh, args.finetune_color, lgt)
                else:
                    _, _, gbuffer_mask = shader.get_mask(gbuffers, views_subset, mesh, args.finetune_color, lgt)

                landmark_loss, closure_loss = landmark_loss_function(ict_facekit, gbuffers, views_subset, use_jaw, device)
                mask_loss = mask_loss_function(views_subset["mask"], gbuffer_mask)
                mask_loss_segmentation = mask_loss_function(views_subset["skin_mask"][..., :1], gbuffers['segmentation'])

                shading_loss, pred_color, tonemapped_colors = shading_loss_batch(pred_color_masked, views_subset, views_subset['img'].size(0))
                perceptual_loss = VGGloss(tonemapped_colors[0], tonemapped_colors[1], iteration)
            
                white_light_loss = white_light_regularization(cbuffers['light'])
                # roughness_loss = material_regularization_function(cbuffers['material'][..., 3], views_subset['skin_mask'], gbuffer_mask, roughness=True)
                specular_loss = material_regularization_function(cbuffers['material'], views_subset['skin_mask'], gbuffer_mask, specular=True)
                
                normal_laplacian_loss = normal_loss(gbuffers, views_subset, gbuffer_mask, device)
                inverted_normal_loss = inverted_normal_loss_function(gbuffers, views_subset, gbuffer_mask, device)
                eyeball_normal_loss = eyeball_normal_loss_function(gbuffers, views_subset, gbuffer_mask, device)

            else: 
                deformed_vertices = return_dict['ict_mesh_w_temp_posed'] 
                d_normals = mesh.fetch_all_normals(deformed_vertices, mesh)
                channels = ['mask', 'canonical_position'] if not pretrain else channels_gbuffer + ['segmentation']
                gbuffers = renderer.render_batch(views_subset['flame_camera'], deformed_vertices.contiguous(), d_normals,
                                        channels=channels, with_antialiasing=True, 
                                        canonical_v=mesh.vertices, canonical_idx=mesh.indices, canonical_uv=ict_facekit.uv_neutral_mesh)
                if pretrain:
                    pred_color_masked, cbuffers, gbuffer_mask = shader.shade(gbuffers, views_subset, mesh, args.finetune_color, lgt)
                else:
                    _, _, gbuffer_mask = shader.get_mask(gbuffers, views_subset, mesh, args.finetune_color, lgt)

                landmark_loss, closure_loss = landmark_loss_function(ict_facekit, gbuffers, views_subset, use_jaw, device)
                mask_loss = mask_loss_function(views_subset["mask"], gbuffer_mask)
            
                # landmark_loss = torch.tensor(0., device=device)
                # closure_loss = torch.tensor(0., device=device)
                # mask_loss = torch.tensor(0., device=device)

                mask_loss_segmentation = torch.tensor(0., device=device)
                shading_loss = torch.tensor(0., device=device)
                perceptual_loss = torch.tensor(0., device=device)
                white_light_loss = torch.tensor(0., device=device)
                specular_loss = torch.tensor(0., device=device)
                normal_laplacian_loss = torch.tensor(0., device=device)
                inverted_normal_loss = torch.tensor(0., device=device)
                eyeball_normal_loss = torch.tensor(0., device=device)

            roughness_loss = torch.tensor(0., device=device)

            ict_landmark_loss = landmark_loss
            ict_closure_loss = closure_loss
            ict_mask_loss = mask_loss
            ict_mask_loss_segmentation = mask_loss_segmentation
            ict_shading_loss = shading_loss
            ict_perceptual_loss = perceptual_loss
            ict_white_light_loss = white_light_loss
            ict_roughness_loss = roughness_loss
            ict_specular_loss = specular_loss
            ict_normal_laplacian_loss = normal_laplacian_loss
            ict_inverted_normal_loss = inverted_normal_loss
            ict_eyeball_normal_loss = eyeball_normal_loss

            # if not pretrain:
                
            #     ict_landmark_loss *= 0.25
            #     ict_closure_loss *= 0.25
            #     ict_mask_loss *= 0.25
            #     ict_mask_loss_segmentation *= 0.25
            #     ict_shading_loss *= 0.25
            #     ict_perceptual_loss *= 0.25
            #     ict_white_light_loss *= 0.25
            #     ict_roughness_loss *= 0.25
            #     ict_specular_loss *= 0.25
            #     ict_normal_laplacian_loss *= 0.25
            #     ict_inverted_normal_loss *= 0.25




            if not pretrain:
                deformed_vertices = return_dict['expression_mesh_posed']
                d_normals = mesh.fetch_all_normals(deformed_vertices, mesh)
                channels = channels_gbuffer + ['segmentation']
                gbuffers = renderer.render_batch(views_subset['flame_camera'], deformed_vertices.contiguous(), d_normals,
                                        channels=channels, with_antialiasing=True, 
                                        canonical_v=mesh.vertices, canonical_idx=mesh.indices, canonical_uv=ict_facekit.uv_neutral_mesh)
                pred_color_masked, cbuffers, gbuffer_mask = shader.shade(gbuffers, views_subset, mesh, args.finetune_color, lgt)

                landmark_loss, closure_loss = landmark_loss_function(ict_facekit, gbuffers, views_subset, use_jaw, device)
                mask_loss = mask_loss_function(views_subset["mask"], gbuffer_mask)
                mask_loss_segmentation = mask_loss_function(views_subset["skin_mask"][..., :1], gbuffers['segmentation'])
                shading_loss, pred_color, tonemapped_colors = shading_loss_batch(pred_color_masked, views_subset, views_subset['img'].size(0))
                perceptual_loss = VGGloss(tonemapped_colors[0], tonemapped_colors[1], iteration)

                white_light_loss = white_light_regularization(cbuffers['light'])
                # roughness_loss = material_regularization_function(cbuffers['material'][..., 3], views_subset['skin_mask'], gbuffer_mask, roughness=True)
                specular_loss = material_regularization_function(cbuffers['material'], views_subset['skin_mask'], gbuffer_mask, specular=True)


                normal_laplacian_loss = normal_loss(gbuffers, views_subset, gbuffer_mask, device)
                inverted_normal_loss = inverted_normal_loss_function(gbuffers, views_subset, gbuffer_mask, device)
                eyeball_normal_loss = eyeball_normal_loss_function(gbuffers, views_subset, gbuffer_mask, device)

            else:
                landmark_loss = torch.tensor(0., device=device)
                closure_loss = torch.tensor(0., device=device)
                mask_loss = torch.tensor(0., device=device)
                mask_loss_segmentation = torch.tensor(0., device=device)
                shading_loss = torch.tensor(0., device=device)
                perceptual_loss = torch.tensor(0., device=device)
                white_light_loss = torch.tensor(0., device=device)
                specular_loss = torch.tensor(0., device=device)
                normal_laplacian_loss = torch.tensor(0., device=device)
                inverted_normal_loss = torch.tensor(0., device=device)
                eyeball_normal_loss = torch.tensor(0., device=device)

            roughness_loss = torch.tensor(0., device=device)

            # regularizations
            # 1. laplacian regularization - every output mesh should have smooth mesh. using laplacian_loss_given_lap
            template_mesh_laplacian_regularization = laplacian_loss_two_meshes(mesh, ict_facekit.neutral_mesh_canonical[0], return_dict['template_mesh'], filtered_lap, head_index =ict_canonical_mesh.vertices.shape[0]) 
            expression_mesh_laplacian_regularization = laplacian_loss_two_meshes(mesh, return_dict['ict_mesh_w_temp'].detach(), return_dict['expression_mesh'], filtered_lap, head_index =ict_canonical_mesh.vertices.shape[0])  if not pretrain else torch.tensor(0., device=device)
            # 2. normal regularization - template mesh should have similar normal with canonical mesh. using normal_reg_loss
            template_mesh_normal_regularization = normal_reg_loss(mesh, ict_canonical_mesh.with_vertices(ict_facekit.neutral_mesh_canonical[0]), ict_canonical_mesh.with_vertices(return_dict['template_mesh']), head_index =ict_canonical_mesh.vertices.shape[0])
            expression_mesh_normal_regularization = torch.tensor(0., device=device)
            if not pretrain:
                for i in range(return_dict['expression_mesh'].shape[0]):
                    expression_mesh_normal_regularization += normal_reg_loss(mesh, ict_canonical_mesh.with_vertices(return_dict['expression_mesh'][i]), ict_canonical_mesh.with_vertices(return_dict['ict_mesh_w_temp'][i].detach()), head_index =ict_canonical_mesh.vertices.shape[0])  
                expression_mesh_normal_regularization /= return_dict['expression_mesh'].shape[0] 
                # expression_mesh_normal_regularization *= 1e1

            # 3. feature regularization - feature should be similar with neural blendshapes. using feature_regularization_loss
            feature_regularization = feature_regularization_loss(return_dict['features'], views_subset['mp_blendshape'][..., ict_facekit.mediapipe_to_ict],
                                                                neural_blendshapes, return_dict['bshape_modulation'], facs_weight=0)

            # if not pretrain:
            random_blendshapes = torch.rand(views_subset['mp_blendshape'].shape[0], 53, device=device)
            # random_blendshapes = torch.cat([random_blendshapes[:, 0:1], torch.mean(random_blendshapes, dim=1, keepdim=True), random_blendshapes[:, 1:2]], dim=1)
            # random_blendshapes = random_blendshapes.reshape(-1, 53)
            
            expression_delta_random = neural_blendshapes.get_expression_delta(blendshapes=random_blendshapes)


            # expression_delta_random = expression_delta_random.reshape(views_subset['mp_blendshape'].shape[0], 3, expression_delta_random.shape[1], expression_delta_random.shape[2], expression_delta_random.shape[3])


            # linearity_regularization = (expression_delta_random[:, 0] + expression_delta_random[:, 2] - 2 * expression_delta_random[:, 1]).abs().mean() * 1e-5

            l1_regularization = expression_delta_random.abs().mean() * 1e-7
            # else:
            #     l1_regularization = torch.tensor(0., device=device)

            template_geometric_regularization = (ict_facekit.neutral_mesh_canonical[0] - return_dict['template_mesh']).pow(2).mean()
            if not pretrain:
                expression_geometric_regularization = (return_dict['ict_mesh_w_temp'].detach() - return_dict['expression_mesh']).pow(2).mean() 
            else:
                expression_geometric_regularization = torch.tensor(0., device=device)

            # if not no_flame_loss:
            # with torch.no_grad():
            flame_gt, _, _ = FLAMEServer(views_subset['flame_expression'], views_subset['flame_pose'])
            # flame_gt_clip_space = renderer.get_vertices_clip_space_from_view(views_subset['flame_camera'], flame_gt)
            ict_gt = flame_gt[:, flame_pair_indices]
            # ict_gt = flame_gt_clip_space[:, flame_pair_indices]
            # if pretrain:
            flame_loss = (deformed_vertices[:, ict_pair_indices] * 1 - ict_gt * 1) * 5
            if not flame_loss_full_head and flame_loss_tight_face:
                flame_loss = flame_loss[:, tight_face_ict_flame]
            elif not flame_loss_full_head and not flame_loss_tight_face:
                flame_loss = flame_loss[:, full_face_ict_flame] 
            flame_loss = flame_loss.pow(2).mean()
                
            # else:
            if no_flame_loss:
                flame_loss = flame_loss * 0.05

            losses['laplacian_regularization'] = template_mesh_laplacian_regularization + expression_mesh_laplacian_regularization 
            losses['normal_regularization'] = template_mesh_normal_regularization + expression_mesh_normal_regularization
            losses['feature_regularization'] = feature_regularization
            losses['geometric_regularization'] = template_geometric_regularization + expression_geometric_regularization 

            losses['mask'] += mask_loss + mask_loss_segmentation + ict_mask_loss + ict_mask_loss_segmentation
            losses['landmark'] += landmark_loss + ict_landmark_loss
            losses['closure'] += closure_loss + ict_closure_loss
            losses['flame_regularization'] = flame_loss 
            losses['normal_laplacian'] = normal_laplacian_loss + inverted_normal_loss + ict_normal_laplacian_loss + ict_inverted_normal_loss + eyeball_normal_loss + ict_eyeball_normal_loss

            losses['shading'] = shading_loss + ict_shading_loss
            losses['perceptual_loss'] = perceptual_loss + ict_perceptual_loss

            losses['linearity_regularization'] = l1_regularization  

            losses['white_lgt_regularization'] = white_light_loss + ict_white_light_loss
            losses['roughness_regularization'] = roughness_loss + ict_roughness_loss
            losses['fresnel_coeff'] = specular_loss + ict_specular_loss

            decay_keys = ['mask', 'landmark', 'closure']
            
            # with torch.no_grad():
            #     shading_decay_value = 0
            #     for k in decay_keys:
            #         shading_decay_value += losses[k].mean() * loss_weights[k]
                
            #     shading_decay_value = torch.exp(-shading_decay_value)

            # losses['shading'] = losses['shading'] * shading_decay_value
            # losses['perceptual_loss'] = losses['perceptual_loss'] * shading_decay_value
            # losses['normal_laplacian'] = losses['normal_laplacian'] * shading_decay_value


            loss = torch.tensor(0., device=device) 
            for k, v in losses.items():
                if torch.isnan(v).any():
                    print(f'NAN in {k}')
                    print(losses)
                    exit()
                    continue
                loss += v.mean() * loss_weights[k]
            
            loss += neural_blendshapes.encoder.identity_weights.pow(2).mean() * 1e-3
            
            # decay value is  the ten times of summation of loss for mask, segmentation, landmark, closure. 
            decay_value = 0
            for k in decay_keys:
                decay_value += losses[k].mean() * loss_weights[k]
            decay_value *= 3

            for idx in views_subset['idx']:            
                importance[idx] = ((1 - weight_decay_rate) * importance[idx] + weight_decay_rate * decay_value).clamp(min=5e-2).item()

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
                    
                    estim_bshape = return_dict['features'][0, ict_facekit.expression_names.tolist().index('jawOpen')]
                    gt_bshape = views_subset['mp_blendshape'][0, ict_facekit.mediapipe_to_ict][ict_facekit.expression_names.tolist().index('jawOpen')]
                    bshape_modulation = estim_bshape - gt_bshape
                    
                    wandb.log({"bshape_modulation": bshape_modulation.item()}, step=iteration)
                else:
                    for k, v in losses_to_log.items():
                        losses_keys[k].append(v.item())
                        plt.clf()
                        plt.plot(losses_keys[k])
                        plt.savefig(str(images_save_path / f"{k}.png"))
                for k, v in losses_to_log.items():
                    del v
                del losses_to_log
                acc_losses = []
                acc_total_loss = 0

            if iteration % 100 == 1:
                
                print(return_dict['features'][0, 53:])
                print(neural_blendshapes.encoder.transform_origin)
                print("=="*50)
                for k, v in losses.items():
                    # if k in losses_to_print:
                    v = v.mean()
                # if v > 0:
                    print(f"{k}: {v.item() * loss_weights[k]}")

                for nn in range(53):
                    bshapes_multipliers[nn].append(1 + neural_blendshapes.encoder.softplus(neural_blendshapes.encoder.bshapes_multiplier).cpu().data.numpy()[nn])

                    plt.clf()
                    plt.plot(bshapes_multipliers[nn])
                    plt.title(f"BSMult_{nn}_{ict_facekit.expression_names.tolist()[nn]}")
                    plt.savefig(str(images_save_path / f"bsmult_{nn}_{ict_facekit.expression_names.tolist()[nn]}.png"))


                print("=="*50)

                # dataset_sampler = torch.utils.data.WeightedRandomSampler(importance, dataset_train.len_img, replacement=True)
                # dataloader_train    = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, collate_fn=dataset_train.collate, drop_last=True, sampler=dataset_sampler)


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
            # print(neural_blendshapes.expression_deformer[-1].weight.grad)
            

            torch.cuda.synchronize()
            
            if args.grad_scale and args.fourier_features == "hashgrid":
                shader.fourier_feature_transform.params.grad /= 8.0

            clip_grad(neural_blendshapes, shader, norm=5.0)
                
            

            # print(neural_blendshapes.expression_deformer[-1].weight.grad)
            # print(neural_blendshapes.expression_deformer[-1].weight)

            if optimizer_neural_blendshapes is not None:
                optimizer_neural_blendshapes.step() 
                scheduler_neural_blendshapes.step()
            # else:
            # print(neural_blendshapes.expression_deformer[-1].weight)
            # exit()
            # if iteration > 10:
            #     exit()

            optimizer_shader.step()
            scheduler_shader.step()


            progress_bar.set_postfix({'loss': loss.detach().cpu().item(), 'decay': decay_value.detach().cpu().item()})

            # torch.cuda.empty_cache()

            # del loss, gbuffers, d_normals, pred_color_masked, cbuffers, gbuffer_mask

            for k, v in losses.items():
                del v
            del loss, losses
            torch.cuda.empty_cache()

            # ==============================================================================================
            # V I S U A L I Z A T I O N S
            # ==============================================================================================
            if (args.visualization_frequency > 0) and (iteration == 1 or iteration % args.visualization_frequency == 0):
            
                with torch.no_grad():

                    return_dict_ = neural_blendshapes(views_subset['img'], views_subset)
                    debug_gbuffer = renderer.render_batch(views_subset['flame_camera'], return_dict_['expression_mesh_posed'].contiguous(), mesh.fetch_all_normals(return_dict_['expression_mesh_posed'], mesh),
                                            channels=channels_gbuffer + ['segmentation'], with_antialiasing=True, 
                                            canonical_v=mesh.vertices, canonical_idx=mesh.indices, canonical_uv=ict_facekit.uv_neutral_mesh)
                    debug_rgb_pred, debug_cbuffers, _ = shader.shade(debug_gbuffer, views_subset, mesh, args.finetune_color, lgt)
                    visualize_training(debug_rgb_pred, debug_cbuffers, debug_gbuffer, views_subset, images_save_path, iteration, ict_facekit=ict_facekit, save_name='training')
                    
                    return_dict_ = neural_blendshapes(debug_views['img'], debug_views)
                    
                    bshapes = return_dict_['features'][:, :53].detach().cpu().numpy()
                    bshapes = np.round(bshapes, 2)
                    jawopen = bshapes[:, ict_facekit.expression_names.tolist().index('jawOpen')]
                    eyeblink_l = bshapes[:, ict_facekit.expression_names.tolist().index('eyeBlink_L')]
                    eyeblink_r = bshapes[:, ict_facekit.expression_names.tolist().index('eyeBlink_R')]
                    
                    m_jawopen = neural_blendshapes.encoder.bshapes_multiplier[ict_facekit.expression_names.tolist().index('jawOpen')]
                    m_eyeblink_l = neural_blendshapes.encoder.bshapes_multiplier[ict_facekit.expression_names.tolist().index('eyeBlink_L')]
                    m_eyeblink_r = neural_blendshapes.encoder.bshapes_multiplier[ict_facekit.expression_names.tolist().index('eyeBlink_R')]
                    print(f"MJawOpen: {m_jawopen}, MEyeBlink_L: {m_eyeblink_l}, MEyeBlink_R: {m_eyeblink_r}")
                    print(f"JawOpen: {jawopen}, EyeBlink_L: {eyeblink_l}, EyeBlink_R: {eyeblink_r}")

                    debug_gbuffer = renderer.render_batch(debug_views['flame_camera'], return_dict_['ict_mesh_w_temp_posed'].contiguous(), mesh.fetch_all_normals(return_dict_['ict_mesh_w_temp_posed'], mesh), 
                                            channels=channels_gbuffer+['segmentation'], with_antialiasing=True, 
                                            canonical_v=mesh.vertices, canonical_idx=mesh.indices, canonical_uv=ict_facekit.uv_neutral_mesh) 
                    debug_rgb_pred, debug_cbuffers, _ = shader.shade(debug_gbuffer, debug_views, mesh, args.finetune_color, lgt)
                    visualize_training(debug_rgb_pred, debug_cbuffers, debug_gbuffer, debug_views, images_save_path, iteration, ict_facekit=ict_facekit, save_name='ict_w_temp')

                    debug_gbuffer = renderer.render_batch(debug_views['flame_camera'], return_dict_['expression_mesh_posed'].contiguous(), mesh.fetch_all_normals(return_dict_['expression_mesh_posed'], mesh),
                                            channels=channels_gbuffer + ['segmentation'], with_antialiasing=True, 
                                            canonical_v=mesh.vertices, canonical_idx=mesh.indices, canonical_uv=ict_facekit.uv_neutral_mesh)
                    debug_rgb_pred, debug_cbuffers, _ = shader.shade(debug_gbuffer, debug_views, mesh, args.finetune_color, lgt)
                    visualize_training(debug_rgb_pred, debug_cbuffers, debug_gbuffer, debug_views, images_save_path, iteration, ict_facekit=ict_facekit, save_name='expression')


                    for ith in range(debug_views['img'].shape[0]):
                        seg = debug_gbuffer['segmentation'][ith, ..., 0]
                        seg = seg * 255
                        seg = seg.cpu().numpy().astype(np.uint8)
                        gt_seg = debug_views['skin_mask'][ith, ..., 0]
                        gt_seg = gt_seg * 255
                        gt_seg = gt_seg.cpu().numpy().astype(np.uint8)

                        seg = np.stack([seg, gt_seg, np.zeros(seg.shape)], axis=2)

                        cv2.imwrite(str(images_save_path / "grid" / f'grid_{iteration}_seg_narrow_{ith}.png'), seg)

                        

                        # draw bar graph of blendshapes.
                        # y for activaiton (value)
                        # x for each blendshape index.
                        # label for each blendshape name.
                        bshapes = return_dict_['features'][ith, :53].detach().cpu().numpy()
                        bshapes = np.round(bshapes, 2)
                        
                        names = ict_facekit.expression_names.tolist()

                        plt.clf()
                        plt.figsize=(8,8)
                        plt.bar(np.arange(53), bshapes)
                        plt.xticks(np.arange(53), names, rotation=90, fontsize=5)
                        plt.title(f"Blendshapes Activation_{ith}")
                        plt.savefig(str(images_save_path / "grid" / f"grid_{iteration}_blendshapes_activation_{ith}.png"))

                    for n in range(debug_views['img'].shape[0]):                            
                        if n != 0:
                            break      
                        write_mesh(meshes_save_path / f"mesh_{iteration:06d}_exp.obj", mesh.with_vertices(return_dict_['expression_mesh'][n]).detach().to('cpu'))                    
                        # save the posed meshes as well
                        write_mesh(meshes_save_path / f"mesh_{iteration:06d}_exp_posed.obj", mesh.with_vertices(return_dict_['expression_mesh_posed'][n]).detach().to('cpu'))

                    write_mesh(meshes_save_path / f"mesh_{iteration:06d}_temp.obj", mesh.with_vertices(return_dict_['template_mesh']).detach().to('cpu'))                                
                    del debug_gbuffer, debug_cbuffers, debug_rgb_pred, return_dict_


            if iteration == 1 or iteration % (args.visualization_frequency * 10) == 0:
                print(images_save_path / "grid" / f'grid_{iteration}.png')
                if 'debug' not in run_name and not args.skip_wandb:
                    wandb.log({"Grid": [wandb.Image(str(images_save_path / "grid" / f'grid_{iteration}_expression.png'))]}, step=iteration)

            ## ============== save intermediate ==============================
            if (args.save_frequency > 0) and (iteration == 1 or iteration % args.save_frequency == 0):
                with torch.no_grad():
                    shader.save(shaders_save_path / f'shader.pt')
                    neural_blendshapes.save(shaders_save_path / f'neural_blendshapes.pt')

            # deformer_or_shader = not deformer_or_shader

            # t = time.time()
            # gc.collect()
            # print("TIME TAKEN secs:", time.time() - t)

            
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
    # assert dataset_train.len_img == len(dataset_train.importance)
    # dataset_sampler = torch.utils.data.WeightedRandomSampler(dataset_train.importance, dataset_train.len_img, replacement=True)
    # dataloader_train    = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, collate_fn=dataset_train.collate, drop_last=True, sampler=dataset_sampler)
    dataloader_train = None
    view_indices = np.array(args.visualization_views).astype(int)
    d_l = [dataset_val.__getitem__(idx) for idx in view_indices[2:]]
    d_l.append(dataset_train.__getitem__(view_indices[0]))
    d_l.append(dataset_train.__getitem__(view_indices[1]))
    debug_views = dataset_val.collate(d_l)


    del dataset_val
    
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

    