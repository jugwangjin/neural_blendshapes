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

set_seed(20241224)

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





def clip_grad(neural_blendshapes, shader, norm=1.0):
    return
    torch.nn.utils.clip_grad_norm_(neural_blendshapes.parameters(), norm)
    # torch.nn.utils.clip_grad_norm_(shader.parameters(), norm)

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
    channels_gbuffer = ['mask', 'position', 'normal', "canonical_position", "segmentation"]
    debug_gbuffer = renderer.render_batch(views_subset['flame_camera'], return_dict[mesh_key_name+'_posed'].contiguous(), mesh.fetch_all_normals(return_dict[mesh_key_name+'_posed'], mesh),
                            channels=channels_gbuffer, with_antialiasing=True, 
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

    full_head_index = 11248

    # flame_except_eyes_conditions : y < 0.03, or y > 0.2 or x < -0.25 or x > 0.25
    full_head_ict_flame = np.where((ict_pair_indices < full_head_index) &
                                    ((FLAMEServer.canonical_verts.cpu().data.numpy()[0, flame_pair_indices][:, 1] < 0.02) |
                                    (FLAMEServer.canonical_verts.cpu().data.numpy()[0, flame_pair_indices][:, 1] > 0.2) |
                                    (FLAMEServer.canonical_verts.cpu().data.numpy()[0, flame_pair_indices][:, 0] < -0.25) |
                                    (FLAMEServer.canonical_verts.cpu().data.numpy()[0, flame_pair_indices][:, 0] > 0.25)))[0]


    tight_face_ict_flame = np.where((ict_pair_indices < tight_face_index) &
                                    ((FLAMEServer.canonical_verts.cpu().data.numpy()[0, flame_pair_indices][:, 1] < 0.02) |
                                    (FLAMEServer.canonical_verts.cpu().data.numpy()[0, flame_pair_indices][:, 1] > 0.2) |
                                    (FLAMEServer.canonical_verts.cpu().data.numpy()[0, flame_pair_indices][:, 0] < -0.25) |
                                    (FLAMEServer.canonical_verts.cpu().data.numpy()[0, flame_pair_indices][:, 0] > 0.25)))[0]

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
            
    mode = dataset_train.get_bshapes_mode()[ict_facekit.mediapipe_to_ict]

    bshapes = mode.detach().cpu().numpy()
    bshapes = np.round(bshapes, 2)
    
    names = ict_facekit.expression_names.tolist()

    os.makedirs(images_save_path / "grid", exist_ok=True)

    save_blendshape_figure(bshapes, names, f"Blendshapes Modes", images_save_path / "grid" / f"a_blendshape_modes.png")
                     
    tight_face_index = 6705

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

    face_normals = ict_canonical_mesh.get_vertices_face_normals(ict_facekit.neutral_mesh_canonical[0])[0]
    neural_blendshapes = get_neural_blendshapes(model_path=model_path, train=args.train_deformer, ict_facekit=ict_facekit, aabb = ict_mesh_aabb, face_normals=face_normals, fix_bshapes=args.fix_bshapes, additive=args.additive, disable_pose=args.disable_pose, device=device) 
    print(ict_canonical_mesh.vertices.shape, ict_canonical_mesh.vertices.device)

    neural_blendshapes = neural_blendshapes.to(device)

    neural_blendshapes_params = list(neural_blendshapes.parameters())
    neural_blendshapes_expression_params = list(neural_blendshapes.expression_deformer.parameters())
    neural_blendshapes_template_params = list(neural_blendshapes.template_deformer.parameters()) 
    neural_blendshapes_detail_params = [neural_blendshapes.face_details]
    neural_blendshapes_pe = list(neural_blendshapes.fourier_feature_transform.parameters()) 
    neural_blendshapes_pose_weight_params = list(neural_blendshapes.pose_weight.parameters())
    neural_blendshapes_encoder_params = list(neural_blendshapes.encoder.parameters())


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


    optimizer_shader = torch.optim.Adam(params, lr=args.lr_shader,)

    # ==============================================================================================
    # Loss Functions
    # ==============================================================================================
    # Initialize the loss weights and losses
    loss_weights = {
        "mask": args.weight_mask,
        "segmentation": args.weight_mask,
        "laplacian_regularization": args.weight_laplacian_regularization,
        "shading": args.weight_shading,
        "perceptual_loss": args.weight_perceptual_loss,
        "landmark": args.weight_landmark,
        "closure": args.weight_closure,
        "feature_regularization": args.weight_feature_regularization,
        "geometric_regularization": args.weight_geometric_regularization,
        "normal_laplacian": args.weight_normal_laplacian,
        "inverted_normal": args.weight_normal_laplacian,
        "eyeball_normal": args.weight_normal_laplacian,
        "linearity_regularization": args.weight_linearity_regularization,
        "flame_regularization": args.weight_flame_regularization,
        "white_lgt_regularization": args.weight_white_lgt_regularization,
        "roughness_regularization": args.weight_roughness_regularization,
        "albedo_regularization": args.weight_albedo_regularization,
        "fresnel_coeff": args.weight_fresnel_coeff,
        "temporal_regularization": args.weight_temporal_regularization,
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

    dataloader_train    = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, collate_fn=dataset_train.collate, drop_last=True, shuffle=True)

    losses = {k: torch.tensor(0.0, device=device) for k in loss_weights}

    epochs = (args.iterations // len(dataloader_train)) + 1
    iteration = 0
    
    progress_bar = tqdm(range(epochs))
    start = time.time()

    acc_losses = []
    acc_total_loss = 0

    bshapes_multipliers = []
    for i in range(53):
        bshapes_multipliers.append([])

    use_jaw=True


    optimizer_neural_blendshapes = torch.optim.Adam([
                                                    {'params': neural_blendshapes_encoder_params, 'lr': args.lr_deformer, },
                                                    {'params': neural_blendshapes_template_params, 'lr': args.lr_jacobian},
                                                    {'params': neural_blendshapes_detail_params, 'lr': args.lr_jacobian * 1e-1},
                                                    {'params': neural_blendshapes_pose_weight_params, 'lr': args.lr_jacobian},
                                                    {'params': neural_blendshapes_expression_params, 'lr': args.lr_jacobian},
                                                    {'params': neural_blendshapes_pe, 'lr': args.lr_jacobian},
                                                    ]
                                                    )
    from torch.optim.lr_scheduler import LinearLR
    num_warmup_steps = 50  # Adjust this value based on your needs
    warmup_scheduler = LinearLR(
        optimizer_neural_blendshapes,
        start_factor=0.01,  # Start with 10% of the base learning rate
        end_factor=1.0,    # End with 100% of the base learning rate
        total_iters=num_warmup_steps
    )

    stage_iterations = args.stage_iterations
    # stage_iterations contain the number of iterations for each stage
    milestones = [args.stage_iterations[0]]
    for i in range(1, len(stage_iterations)):
        milestones.append(milestones[-1] + stage_iterations[i])


    # milestones should be a list of integers, length of 6
    assert len(milestones) == 4

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

    args.finetune_color = False

    # with torch.no_grad():
    #     verts = FLAMEServer.canonical_verts.squeeze(0)
    #     faces = FLAMEServer.faces_tensor

    #     flame_canonical_mesh: Mesh = None
    #     flame_canonical_mesh = Mesh(verts, faces, device=device)
    #     flame_canonical_mesh.compute_connectivity()

    #     channels_gbuffer = ['mask', 'position', 'normal', "canonical_position",]
    #     return_dict = neural_blendshapes(debug_views['img'], debug_views)

    #     vertices, _, _  = FLAMEServer(debug_views['flame_expression'], debug_views['flame_pose'])
        
    #     # print(vertices.min(), vertices.max(), 

    #     debug_gbuffer = renderer.render_batch(debug_views['flame_camera'], vertices.contiguous(), flame_canonical_mesh.fetch_all_normals(vertices, flame_canonical_mesh),
    #                             channels=channels_gbuffer, with_antialiasing=True, 
    #                             canonical_v=flame_canonical_mesh.vertices, canonical_idx=flame_canonical_mesh.indices, canonical_uv=ict_facekit.uv_neutral_mesh,
    #                             mesh=flame_canonical_mesh
    #                             )
    #     debug_rgb_pred, debug_cbuffers, _ = shader.shade(debug_gbuffer, debug_views, flame_canonical_mesh, args.finetune_color, lgt)
    #     visualize_training_no_lm(debug_rgb_pred, debug_cbuffers, debug_gbuffer, debug_views, images_save_path, 0, ict_facekit=ict_facekit, save_name='flame_init')

    iteration = 0
    for epoch in range(epochs):
        progress_bar = tqdm(enumerate(dataloader_train))
        for iter_, views_subset in progress_bar:
            iteration += 1
            # Determine the stage based on iteration and milestones
            stage = next((i for i, milestone in enumerate(milestones) if iteration < milestone), len(milestones))

            '''
            training options
            '''
            # stage 0 : template and encoder
            # stage 1 : only template
            # stage 2 : only expression
            # stage 3 : full loss, shader training

            target_res = None            

            if iteration > args.only_flame_iterations:
                target_range = tight_face_ict_flame
            else:
                target_range = full_head_ict_flame
            
            if stage < 2 : 
                deformed_vertices_key = 'ict_mesh_w_temp'
            else:
                deformed_vertices_key = 'expression_mesh'

            '''
            optimizer updates
            '''

            if iteration == milestones[0]: # on stage 1 -> update the optimizer to fix encoder
                print("\nUpdating the optimizer to only template\n")
                # now only update the expression parameters
                optimizer_neural_blendshapes.zero_grad(set_to_none=True)
                optimizer_neural_blendshapes = None
                optimizer_neural_blendshapes = torch.optim.Adam([
                                                    {'params': neural_blendshapes_template_params, 'lr': args.lr_jacobian * 1e-1},
                                                    {'params': neural_blendshapes_detail_params, 'lr': args.lr_jacobian * 1e-2},
                                                    {'params': neural_blendshapes_pose_weight_params, 'lr': args.lr_jacobian * 1e-1},
                                                    {'params': neural_blendshapes_expression_params, 'lr': args.lr_jacobian},
                                                    {'params': neural_blendshapes_pe, 'lr': args.lr_jacobian},
                                                    ],
                                                )
                # Create a warm-up scheduler
                from torch.optim.lr_scheduler import LinearLR
                num_warmup_steps = 50  # Adjust this value based on your needs
                warmup_scheduler = LinearLR(
                    optimizer_neural_blendshapes,
                    start_factor=0.01,  # Start with 10% of the base learning rate
                    end_factor=1.0,    # End with 100% of the base learning rate
                    total_iters=num_warmup_steps
                )

            if iteration == milestones[1]: # on stage 2 -> update the optimizer to only expression 
                # print("\nUpdating the optimizer to only expression\n")
                # # now only update the expression parameters
                # optimizer_neural_blendshapes.zero_grad(set_to_none=True)
                # optimizer_neural_blendshapes = None
                # optimizer_neural_blendshapes = torch.optim.Adam([
                #                                 {'params': neural_blendshapes_template_params, 'lr': args.lr_jacobian},
                #                                 {'params': neural_blendshapes_pose_weight_params, 'lr': args.lr_jacobian},
                #                                 {'params': neural_blendshapes_expression_params, 'lr': args.lr_jacobian},
                #                                 {'params': neural_blendshapes_pe, 'lr': args.lr_jacobian},
                #                                 ], 
                #                                 )
                # # Create a warm-up scheduler
                from torch.optim.lr_scheduler import LinearLR
                num_warmup_steps = 50  # Adjust this value based on your needs
                warmup_scheduler = LinearLR(
                    optimizer_neural_blendshapes,
                    start_factor=0.01,  # Start with 10% of the base learning rate
                    end_factor=1.0,    # End with 100% of the base learning rate
                    total_iters=num_warmup_steps
                )


            if iteration == milestones[2]:
                print("\nUpdating the optimizer to only shader\n")
                del shader
                del optimizer_shader

                args.material_mlp_dims = [64, 64]
                args.light_mlp_dims = [64, 64]



                lgt = light.create_env_rnd()    
                disentangle_network_params = {
                    "material_mlp_ch": args.material_mlp_ch,
                    "light_mlp_ch":args.light_mlp_ch,
                    "material_mlp_dims":args.material_mlp_dims,
                    "light_mlp_dims":args.light_mlp_dims,
                    "brdf_mlp_dims": args.brdf_mlp_dims,

                }

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

                optimizer_shader = torch.optim.Adam(params, lr=args.lr_shader)
                args.finetune_color = True

                optimizer_neural_blendshapes.zero_grad(set_to_none=True)
                optimizer_neural_blendshapes = None
                optimizer_neural_blendshapes = torch.optim.Adam([
                                                    {'params': neural_blendshapes_template_params, 'lr': args.lr_jacobian * 1e-2},
                                                    {'params': neural_blendshapes_detail_params, 'lr': args.lr_jacobian * 1e-2},
                                                    {'params': neural_blendshapes_pose_weight_params, 'lr': args.lr_jacobian * 1e-2},
                                                    {'params': neural_blendshapes_expression_params, 'lr': args.lr_jacobian * 1e-2},
                                                    {'params': neural_blendshapes_pe, 'lr': args.lr_jacobian * 1e-2},
                                                    ],
                                                )
                # Create a warm-up scheduler
                from torch.optim.lr_scheduler import LinearLR
                num_warmup_steps = 50  # Adjust this value based on your needs
                warmup_scheduler = LinearLR(
                    optimizer_neural_blendshapes,
                    start_factor=0.01,  # Start with 10% of the base learning rate
                    end_factor=1.0,    # End with 100% of the base learning rate
                    total_iters=num_warmup_steps
                )

                neural_blendshapes.train()


            progress_bar.set_description(desc=f'{run_name}, Epoch {epoch}/{epochs}, Iter {iteration}/{len(dataloader_train)}, Stage {stage}')
            losses = {k: torch.tensor(0.0, device=device) for k in loss_weights}
                        
            use_jaw = True
            
            input_image = views_subset["img"].permute(0, 3, 1, 2).to(device)

            # if stage is 3, with no grad. else, with grad
            # with torch.set_grad_enabled(stage != 3):
            return_dict = neural_blendshapes(input_image, views_subset)
                
            mesh = ict_canonical_mesh

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


            if stage == 0:
                indices = views_subset['idx']
                sequential_frames = dataloader_train.collate_fn([dataset_train.get_sequential_frame(idx) for idx in indices])
                sequential_features = neural_blendshapes.encoder(sequential_frames)
                temporal_loss = (return_dict['features'] - sequential_features)
                temporal_loss[:, 53:] *= 1e1
                temporal_loss = temporal_loss.pow(2).mean()
                losses['temporal_regularization'] = temporal_loss

            '''
            2D signal losses
            '''
            if iteration > args.only_flame_iterations:


                # skin_mask_w_mouth: views_subset["skin_mask"][..., :1] + views_subset["skin_mask"][..., 4:5]
                # skin_mask_w_mouth = (views_subset["skin_mask"][..., :1] + views_subset["skin_mask"][..., 4:5] >= 1).float()
                # segmentation_gt = downsample_upsample(skin_mask_w_mouth, None, (512, 512))
                left_eye_segmentation_gt = downsample_upsample(views_subset["skin_mask"][..., 3:4], None, (512, 512))
                right_eye_segmentation_gt = downsample_upsample(views_subset["skin_mask"][..., 4:5], None, (512, 512))
                mouth_segmentation_gt = downsample_upsample(views_subset["skin_mask"][..., 5:6], None, (512, 512))
                face_segmentation_gt = downsample_upsample(views_subset["skin_mask"][..., 6:7], None, (512, 512))

                mask_loss_segmentation = mask_loss_function(views_subset["mask"], gbuffer_mask)
                shading_loss, pred_color, tonemapped_colors = shading_loss_batch(pred_color_masked, views_subset, views_subset['img'].size(0))


                eyes_closed_expression_mesh = neural_blendshapes.get_random_mesh_eyes_closed(deformed_vertices_key)
                eyes_closed_normals = {k: v[:1] for k, v in d_normals.items()}
                eyes_closed_gbuffers = renderer.render_batch(views_subset['flame_camera'][:1], eyes_closed_expression_mesh.contiguous(), eyes_closed_normals,
                                        channels=['segmentation'], with_antialiasing=True, 
                                        canonical_v=mesh.vertices, canonical_idx=mesh.indices, canonical_uv=ict_facekit.uv_neutral_mesh,
                                        mesh=mesh
                                        )
                
                eyes_closed_left_eye_seg_loss = eyes_closed_gbuffers['left_eye'].pow(2).mean()
                eyes_closed_right_eye_seg_loss = eyes_closed_gbuffers['right_eye'].pow(2).mean()

                segmentation_loss = segmentation_loss_function(left_eye_segmentation_gt, gbuffers['left_eye']) * 1e-1 +\
                                    segmentation_loss_function(right_eye_segmentation_gt, gbuffers['right_eye']) * 1e-1 +\
                                    segmentation_loss_function(mouth_segmentation_gt, gbuffers['mouth']) * 1e-1 +\
                                    segmentation_loss_function(face_segmentation_gt, gbuffers['face']) * 1e-1 +\
                                    eyes_closed_left_eye_seg_loss * 1e-1 +\
                                    eyes_closed_right_eye_seg_loss * 1e-1

                # visualize gbuffer_mask, views_subset["mask"], gbuffers['eyes'], eyes_segmentation_gt, mouth_segmentation_gt, gbuffers['mouth'], views_subset["img"]
                # for bn in range(views_subset["img"].size(0)):
                #     os.makedirs('debug/masks_vis', exist_ok=True)
                #     cv2.imwrite(os.path.join('debug', 'masks_vis', f"{iteration}_{bn}_gbuffer_mask.png"), (gbuffer_mask[bn].cpu().data.numpy() * 255).astype(np.uint8))
                #     cv2.imwrite(os.path.join('debug', 'masks_vis', f"{iteration}_{bn}_mask.png"), (views_subset["mask"][bn].cpu().data.numpy() * 255).astype(np.uint8))
                #     cv2.imwrite(os.path.join('debug', 'masks_vis', f"{iteration}_{bn}_lefteye_segmentation_gt.png"), (left_eye_segmentation_gt[bn].cpu().data.numpy() * 255).astype(np.uint8))
                #     cv2.imwrite(os.path.join('debug', 'masks_vis', f"{iteration}_{bn}_righteyes_segmentation_gt.png"), (right_eye_segmentation_gt[bn].cpu().data.numpy() * 255).astype(np.uint8))
                #     cv2.imwrite(os.path.join('debug', 'masks_vis', f"{iteration}_{bn}_mouth_segmentation_gt.png"), (mouth_segmentation_gt[bn].cpu().data.numpy() * 255).astype(np.uint8))
                #     cv2.imwrite(os.path.join('debug', 'masks_vis', f"{iteration}_{bn}_face_segmentation_gt.png"), (face_segmentation_gt[bn].cpu().data.numpy() * 255).astype(np.uint8))
                #     cv2.imwrite(os.path.join('debug', 'masks_vis', f"{iteration}_{bn}_gbuffers_lefteye.png"), (gbuffers['left_eye'][bn].cpu().data.numpy() * 255).astype(np.uint8))
                #     cv2.imwrite(os.path.join('debug', 'masks_vis', f"{iteration}_{bn}_gbuffers_righteye.png"), (gbuffers['right_eye'][bn].cpu().data.numpy() * 255).astype(np.uint8))
                #     cv2.imwrite(os.path.join('debug', 'masks_vis', f"{iteration}_{bn}_gbuffers_mouth.png"), (gbuffers['mouth'][bn].cpu().data.numpy() * 255).astype(np.uint8))
                #     cv2.imwrite(os.path.join('debug', 'masks_vis', f"{iteration}_{bn}_gbuffers_seg.png"), (gbuffers['segmentation'][bn].cpu().data.numpy() * 255).astype(np.uint8))
                #     cv2.imwrite(os.path.join('debug', 'masks_vis', f"{iteration}_{bn}_gbuffers_face.png"), (gbuffers['face'][bn].cpu().data.numpy() * 255).astype(np.uint8))
                #     cv2.imwrite(os.path.join('debug', 'masks_vis', f"{iteration}_{bn}_img.png"), (views_subset["img"][bn].cpu().data.numpy() * 255).astype(np.uint8))    

                

                # if iteration > 10:
                #     exit()
                

                perceptual_loss = VGGloss(tonemapped_colors[0], tonemapped_colors[1], iteration)

                normal_laplacian_loss = normal_loss(gbuffers, views_subset, gbuffers['segmentation'], device) 
                # inverted_normal_loss = inverted_normal_loss_function(gbuffers, views_subset, gbuffer_mask, device)
                eyeball_normal_loss = eyeball_normal_loss_function(gbuffers, views_subset, gbuffer_mask, device)

                losses['mask'] = mask_loss_segmentation 
                losses['segmentation'] = segmentation_loss

                # if stage == 0:
                #     shading_loss *= 1e-1
                #     perceptual_loss *= 1e-1

                losses['shading'] = shading_loss
                losses['perceptual_loss'] = perceptual_loss 

                ## ======= regularization color ========
                if args.weight_albedo_regularization > 0 and stage == 3:
                    # mult: iterations == milestones[3] - 1000 -> 0
                    # mult: iterations == milestones[3] - 500 -> 1
                    to_milestone_3 = milestones[3] - iteration
                    mult = 500 - to_milestone_3
                    mult = max(0, min(1, mult / 500))
                    
                    losses['albedo_regularization'] = albedo_regularization(_adaptive, shader, mesh, device, None, iteration, mult=mult)
                losses['white_lgt_regularization'] = white_light(cbuffers)
                losses['roughness_regularization'] = roughness_regularization(cbuffers["roughness"], views_subset["skin_mask"], views_subset["mask"], r_mean=args.r_mean)
                losses["fresnel_coeff"] = spec_intensity_regularization(cbuffers["ko"], views_subset["skin_mask"], views_subset["mask"])
                # losses['normal_laplacian'] = normal_laplacian_loss
                losses['normal_laplacian'] = normal_laplacian_loss
                # losses['inverted_normal'] = inverted_normal_loss
                
                # if iteration > milestones[0] // 2:
                #     eyeball_normal_loss *= 1e1

                losses['eyeball_normal']    = eyeball_normal_loss


            # if stage < 3:
            '''
            FLAME regularization
            '''
            landmark_loss, closure_loss = landmark_loss_function(ict_facekit, gbuffers, views_subset, use_jaw, device)
            
            losses['landmark'] = landmark_loss 
            losses['closure'] = closure_loss
            
            flame_loss = FLAME_loss_function(FLAMEServer, views_subset['flame_expression'], views_subset['flame_pose'], deformed_vertices, flame_pair_indices, ict_pair_indices, target_range=target_range)
            
            zero_pose_w_jaw = torch.zeros_like(views_subset['flame_pose'])
            zero_pose_w_jaw[:, 6:9] = views_subset['flame_pose'][:, 6:9]

            flame_loss_no_pose = FLAME_loss_function(FLAMEServer, views_subset['flame_expression'], zero_pose_w_jaw, deformed_vertices_no_pose, flame_pair_indices, ict_pair_indices, target_range=target_range)

            flame_loss_template = FLAME_loss_function(FLAMEServer, torch.zeros_like(views_subset['flame_expression'][:1]), torch.zeros_like(views_subset['flame_pose'][:1]), return_dict['template_mesh'][None], flame_pair_indices, ict_pair_indices, target_range=tight_face_ict_flame)

            losses['flame_regularization'] = flame_loss + flame_loss_no_pose

            if stage > 0:
                losses['flame_regularization'] = losses['flame_regularization'] * 5e-1
            
            losses['flame_regularization'] = losses['flame_regularization'] + flame_loss_template

            '''
            Regularizations
            '''
            # laplacian regularization
            template_mesh_laplacian_regularization = laplacian_loss_two_meshes(mesh, ict_facekit.neutral_mesh_canonical[0], return_dict['template_mesh'], ict_canonical_mesh.laplacian, head_index =ict_canonical_mesh.vertices.shape[0]) 
            # expression_mesh_laplacian_regularization = laplacian_loss_two_meshes(mesh, return_dict['ict_mesh_w_temp'], return_dict['expression_mesh'], ict_canonical_mesh.laplacian, head_index =ict_canonical_mesh.vertices.shape[0]) * 1e1 if stage == 2 else 0

            # more regularizations
            feature_regularization = feature_regularization_loss(return_dict['features'], views_subset['mp_blendshape'][..., ict_facekit.mediapipe_to_ict],
                                                                neural_blendshapes, None, views_subset, dataset_train.bshapes_mode[ict_facekit.mediapipe_to_ict], rot_mult=1e-1, mult=1)

            # random_bshapes = torch.rand_like(return_dict['features'][:, :53]) 
            expression_delta_random = neural_blendshapes.get_expression_delta()

            expression_linearity_regularization = expression_delta_random.pow(2).mean()   if stage == 2 else 0
            # expression_linearity_regularization = expression_delta_random.abs().mean() * 1e1  if stage == 2 else 0
            detail_linearity_regularization = neural_blendshapes.face_details.pow(2).mean() * 1e2

            template_geometric_regularization = (ict_facekit.neutral_mesh_canonical[0] - return_dict['template_mesh']).pow(2).mean() 
            # expression_geometric_regularization = (return_dict['ict_mesh_w_temp'] - return_dict['expression_mesh']).pow(2).mean() if stage == 2 else 0


            losses['feature_regularization'] = feature_regularization
            losses['laplacian_regularization'] = template_mesh_laplacian_regularization 
            # losses['laplacian_regularization'] = template_mesh_laplacian_regularization + expression_mesh_laplacian_regularization 
            losses['geometric_regularization'] = template_geometric_regularization
            # losses['geometric_regularization'] = template_geometric_regularization + expression_geometric_regularization
            losses['linearity_regularization'] = expression_linearity_regularization + detail_linearity_regularization 

            loss = torch.tensor(0., device=device) 
            for k, v in losses.items():
                if torch.isnan(v).any():
                    print(f'NAN in {k}')
                    print(losses)
                    exit()
                    continue
                    
                loss += v.mean() * loss_weights[k]
                        
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
                print(return_dict['features'][0, 53:53+7], neural_blendshapes.encoder.global_translation)
                print("=="*50)
                for k, v in losses.items():
                    # if k in losses_to_print:
                    v = v.mean()
                    if v > 0:
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


            if num_warmup_steps >= 0 and warmup_scheduler is not None:
                warmup_scheduler.step()
                num_warmup_steps = num_warmup_steps - 1

            clip_grad(neural_blendshapes, shader, norm=10.0)

            if optimizer_neural_blendshapes is not None:
                optimizer_neural_blendshapes.step() 

            optimizer_shader.step()


            if optimizer_neural_blendshapes is not None:
                progress_bar.set_postfix({'loss': loss.detach().cpu().item(), 'lr': optimizer_neural_blendshapes.param_groups[0]['lr']})
            else:
                progress_bar.set_postfix({'loss': loss.detach().cpu().item(), 'lr': optimizer_shader.param_groups[0]['lr']})

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
                    
                    write_mesh(meshes_save_path / f"mesh_{iteration:06d}_exp_train.obj", mesh.with_vertices(return_dict_[deformed_vertices_key][0]).detach().to('cpu'))                    
                    # save the posed meshes as well
                    write_mesh(meshes_save_path / f"mesh_{iteration:06d}_exp_posed_train.obj", mesh.with_vertices(return_dict_[deformed_vertices_key + '_posed'][0]).detach().to('cpu'))

                    # vertices, _, _  = FLAMEServer(views_subset['flame_expression'], views_subset['flame_pose'])
                    # write_mesh(meshes_save_path / f"mesh_{iteration:06d}_flame_train.obj", flame_canonical_mesh.with_vertices(vertices[0]).detach().to('cpu'))

                    zero_pose_w_jaw = torch.zeros_like(views_subset['flame_pose'])
                    zero_pose_w_jaw[:, 6:9] = views_subset['flame_pose'][:, 6:9]
                    vertices, _, _  = FLAMEServer(views_subset['flame_expression'], zero_pose_w_jaw)
                    # write_mesh(meshes_save_path / f"mesh_{iteration:06d}_flame_zero_pose_train.obj", flame_canonical_mesh.with_vertices(vertices[0]).detach().to('cpu'))



                    visualize_specific_traininig(deformed_vertices_key, return_dict_, renderer, shader, ict_facekit, views_subset, mesh, 'training', images_save_path, iteration, lgt)
                    

                    return_dict_ = neural_blendshapes(debug_views['img'], debug_views)


                    bshapes = return_dict_['features'][:, :53].detach().cpu().numpy()
                    bshapes = np.round(bshapes, 2)
                    jawopen = bshapes[:, ict_facekit.expression_names.tolist().index('jawOpen')]
                    eyeblink_l = bshapes[:, ict_facekit.expression_names.tolist().index('eyeBlink_L')]
                    eyeblink_r = bshapes[:, ict_facekit.expression_names.tolist().index('eyeBlink_R')]
                    
                    print(f"{iteration} JawOpen: {jawopen}, EyeBlink_L: {eyeblink_l}, EyeBlink_R: {eyeblink_r}")

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
                            write_mesh(meshes_save_path / f"mesh_{iteration:06d}_exp.obj", mesh.with_vertices(return_dict_[deformed_vertices_key][n]).detach().to('cpu'))                    
                            # save the posed meshes as well
                            write_mesh(meshes_save_path / f"mesh_{iteration:06d}_exp_posed.obj", mesh.with_vertices(return_dict_[deformed_vertices_key+'_posed'][n]).detach().to('cpu'))

                        # vertices, _, _  = FLAMEServer(debug_views['flame_expression'], debug_views['flame_pose'])
                        # write_mesh(meshes_save_path / f"mesh_{iteration:06d}_flame.obj", flame_canonical_mesh.with_vertices(vertices[0]).detach().to('cpu'))

                        # zero_pose_w_jaw = torch.zeros_like(debug_views['flame_pose'])
                        # zero_pose_w_jaw[:, 6:9] = debug_views['flame_pose'][:, 6:9]
                        # vertices, _, _  = FLAMEServer(debug_views['flame_expression'], zero_pose_w_jaw)
                        # write_mesh(meshes_save_path / f"mesh_{iteration:06d}_flame_zero_pose.obj", flame_canonical_mesh.with_vertices(vertices[0]).detach().to('cpu'))


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

            if iteration > args.iterations:
                break
            
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
    dataset_val      = DatasetLoader(args, train_dir=args.eval_dir, sample_ratio=50, pre_load=False)
    
    dataloader_train = None
    # view_indices = np.array(args.visualization_views).astype(int)
    view_indices = np.array([0, 1, 2, 3, 4])
    d_l = [dataset_val.__getitem__(idx) for idx in view_indices[2:]]
    d_l.append(dataset_train.__getitem__(view_indices[0]))
    d_l.append(dataset_train.__getitem__(view_indices[1]))
    debug_views = dataset_val.collate(d_l)

    # print(debug_views['flame_pose'].shape)
    # exit()
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
            # with torch.autograd.set_detect_anomaly(True):
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

    