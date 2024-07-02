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
    run_name = args.run_name if args.run_name is not None else args.input_dir.parent.name
    images_save_path, images_eval_save_path, meshes_save_path, shaders_save_path, experiment_dir = make_dirs(args, run_name, args.finetune_color)
    copy_sources(args, run_name)

    ## ============== load ict facekit ==============================
    ict_facekit = ICTFaceKitTorch(npy_dir = './assets/ict_facekit_torch.npy', canonical = Path(args.input_dir) / 'ict_identity.npy')
    ict_facekit = ict_facekit.to(device)

    ict_canonical_mesh = Mesh(ict_facekit.canonical[0].cpu().data, ict_facekit.faces.cpu().data, ict_facekit=ict_facekit, device=device)
    ict_canonical_mesh.compute_connectivity()

    write_mesh(Path(meshes_save_path / "init_ict_canonical.obj"), ict_canonical_mesh.to('cpu'))


    head_index=11248

    # filter vertices by head_index
    filtered_vertices = ict_facekit.canonical[0].cpu().data[:head_index]
    filtered_faces = ict_facekit.faces.cpu().data
    # filter: 
    filtered_faces = filtered_faces[filtered_faces[:, 0] < head_index]
    filtered_faces = filtered_faces[filtered_faces[:, 1] < head_index]
    filtered_faces = filtered_faces[filtered_faces[:, 2] < head_index]

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

    neural_blendshapes = get_neural_blendshapes(model_path=model_path, train=args.train_deformer, vertex_parts=ict_facekit.vertex_parts, ict_facekit=ict_facekit, exp_dir = experiment_dir, device=device) 
    print(ict_canonical_mesh.vertices.shape, ict_canonical_mesh.vertices.device)

    neural_blendshapes = neural_blendshapes.to(device)



    lmk_adaptive = None
    facs_adaptive = None
    
    neural_blendshapes_params = list(neural_blendshapes.parameters())
    # neural_blendshapes_encoder_params = list(neural_blendshapes.encoder.parameters())
    # neural_blendshapes_expression_params = list(neural_blendshapes.expression_deformer.parameters())
    neural_blendshapes_expression_params = list(neural_blendshapes.expression_deformer.parameters()) + list(neural_blendshapes.template_deformer.parameters()) 
    neural_blendshapes_others_params = list(set(neural_blendshapes_params) - set(neural_blendshapes_expression_params)) 
    # neural_blendshapes_others_params = list(set(neural_blendshapes_params) - set(neural_blendshapes_encoder_params) - set(neural_blendshapes_expression_params)) 
    # optimizer_neural_blendshapes = torch.optim.Adam([{'params': neural_blendshapes_encoder_params, 'lr': args.lr_encoder},
    optimizer_neural_blendshapes = torch.optim.Adam([
                                                    {'params': neural_blendshapes_others_params, 'lr': args.lr_deformer},
                                                    {'params': neural_blendshapes_expression_params, 'lr': args.lr_jacobian},
                                                    ],
                                                    weight_decay=1e-4)
                                                     
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
        "light_mlp_dims":args.light_mlp_dims
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
        "segmentation": args.weight_segmentation,
        "geometric_regularization": args.weight_geometric_regularization
        # "semantic_stat": args.weight_semantic_stat,
        # "normal_laplacian": args.weight_normal_laplacian,
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

    filtered_lap = compute_laplacian_uniform_filtered(ict_canonical_mesh, head_index=11248)

    import wandb
    if 'debug' not in run_name and not args.skip_wandb:
        wandb_name = args.wandb_name if args.wandb_name is not None else run_name
        wandb.init(project="neural_jacobian_blendshape", name=wandb_name, config=args)
    for epoch in progress_bar:
        
        if epoch == 3 * (epochs // 4) and args.fourier_features != "positional":
            del shader
            del optimizer_shader
            del scheduler_shader

            shader = NeuralShader(fourier_features=args.fourier_features,
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

            # reduce lr of neural_blendshapes_optimizer by 10
            for param_group in optimizer_neural_blendshapes.param_groups:
                param_group['lr'] /= 10

            importance = torch.ones(len(dataloader_train), device=device)

        for iter_, views_subset in tqdm(enumerate(dataloader_train)):
            iteration += 1
            progress_bar.set_description(desc=f'Epoch {epoch}, Iter {iteration}')
            losses = {k: torch.tensor(0.0, device=device) for k in loss_weights}

            use_jaw = True

            input_image = views_subset["img"].permute(0, 3, 1, 2).to(device)

            return_dict = neural_blendshapes(input_image, views_subset)
            mesh = ict_canonical_mesh.with_vertices(ict_canonical_mesh.vertices)

            pretrain = iteration < args.iterations // 8

            # pose optimization
            # ict_mesh_posed = return_dict['ict_mesh_posed']
            # ict_mesh_gbuffers = {}
            # ict_mesh_gbuffers['deformed_verts_clip_space'] = renderer.get_vertices_clip_space_from_view(views_subset, ict_mesh_posed)

            # ict_mesh_landmark_loss, _ = landmark_loss(ict_facekit, ict_mesh_gbuffers, views_subset, use_jaw, device)
            
            # template optimization
            ict_mesh_w_temp_posed = return_dict['ict_mesh_w_temp_posed']
            d_normals = mesh.fetch_all_normals(ict_mesh_w_temp_posed, mesh)
            ict_mesh_w_temp_gbuffers = renderer.render_batch(views_subset['camera'], ict_mesh_w_temp_posed.contiguous(), d_normals,
                                    channels=['mask', 'canonical_position'], with_antialiasing=True, 
                                    canonical_v=mesh.vertices, canonical_idx=mesh.indices, canonical_uv=ict_facekit.uv_neutral_mesh)
            _, _, ict_mesh_w_temp_gbuffers_mask = shader.get_mask(ict_mesh_w_temp_gbuffers, views_subset, mesh, args.finetune_color, lgt)

            ict_mesh_w_temp_landmark_loss, _ = landmark_loss(ict_facekit, ict_mesh_w_temp_gbuffers, views_subset, use_jaw, device)
            ict_mesh_w_temp_mask_loss = mask_loss(views_subset["mask"], ict_mesh_w_temp_gbuffers_mask)

            # expression optimization
            expression_mesh_posed = return_dict['expression_mesh_posed']
            d_normals = mesh.fetch_all_normals(expression_mesh_posed, mesh)
            expression_gbuffers = renderer.render_batch(views_subset['camera'], expression_mesh_posed.contiguous(), d_normals,
                                    channels=channels_gbuffer, with_antialiasing=True, 
                                    canonical_v=mesh.vertices, canonical_idx=mesh.indices, canonical_uv=ict_facekit.uv_neutral_mesh)
            pred_color_masked, expression_cbuffers, expression_gbuffer_mask = shader.shade(expression_gbuffers, views_subset, mesh, args.finetune_color, lgt)

            expression_landmark_loss, expression_closure_loss = landmark_loss(ict_facekit, expression_gbuffers, views_subset, use_jaw, device)
            expression_mask_loss = mask_loss(views_subset["mask"], expression_gbuffer_mask)

            expression_shading_loss, pred_color, tonemapped_colors = shading_loss_batch(pred_color_masked, views_subset, views_subset['img'].size(0))
            expression_perceptual_loss = VGGloss(tonemapped_colors[0], tonemapped_colors[1], iteration)

            '''
            currently I'm suspecting that segmentation loss is not working properly.
            omitting it.
            losses['segmentation'], _ = segmentation_loss(views_subset, gbuffers, ict_facekit.parts_indices, mesh.vertices)
            '''

            # regularizations
            # 1. laplacian regularization - every output mesh should have smooth mesh. using laplacian_loss_given_lap
            
            template_mesh_laplacian_regularization = laplacian_loss_two_meshes(mesh, ict_facekit.canonical[0], return_dict['template_mesh'], filtered_lap, )
            expression_mesh_laplacian_regularization = laplacian_loss_two_meshes(mesh, return_dict['ict_mesh_w_temp'], return_dict['expression_mesh'], filtered_lap, ) 

            # template_mesh_laplacian_regularization = laplacian_loss_given_lap(mesh, filtered_lap, return_dict['template_mesh'])
            # # ict_mesh_laplacian_regularization = laplacian_loss_given_lap(mesh, filtered_lap, return_dict['ict_mesh'])
            # ict_mesh_w_temp_laplacian_regularization = laplacian_loss_given_lap(mesh, filtered_lap, return_dict['ict_mesh_w_temp'])
            # expression_mesh_laplacian_regularization = laplacian_loss_given_lap(mesh, filtered_lap, return_dict['expression_mesh'])

            # 2. normal regularization - template mesh should have similar normal with canonical mesh. using normal_reg_loss
            template_mesh_normal_regularization = normal_reg_loss(mesh, ict_canonical_mesh, ict_canonical_mesh.with_vertices(return_dict['template_mesh']))

            # 3. feature regularization - feature should be similar with neural blendshapes. using feature_regularization_loss
            feature_regularization = feature_regularization_loss(return_dict['features'], views_subset['mp_blendshape'][..., ict_facekit.mediapipe_to_ict],
                                                                neural_blendshapes, facs_weight=0)

            # 4. geometric regularization. 
            #   1) template mesh close to canonical mesh. for face region.
            #   2) expression mesh close to ict mesh. for out of face region. 
            template_geometric_regularization = (ict_facekit.canonical[0, :neural_blendshapes.face_index] - return_dict['template_mesh'][:neural_blendshapes.face_index]).pow(2).mean()
            expression_geometric_regularization = (return_dict['ict_mesh_w_temp'][:, neural_blendshapes.face_index:] - return_dict['expression_mesh'][:, neural_blendshapes.face_index:]).pow(2).mean()

            pose_weight_geometric_regularization = (1 / args.weight_geometric_regularization) * (return_dict['pose_weight'][ict_facekit.landmark_indices] - 1).pow(2).mean()


            losses['mask'] = ict_mesh_w_temp_mask_loss + expression_mask_loss
            losses['landmark'] = ict_mesh_w_temp_landmark_loss + expression_landmark_loss
            # losses['landmark'] = ict_mesh_landmark_loss + ict_mesh_w_temp_landmark_loss + expression_landmark_loss
            losses['closure'] = expression_closure_loss
            losses['shading'] = expression_shading_loss
            losses['perceptual_loss'] = expression_perceptual_loss
            losses['laplacian_regularization'] = template_mesh_laplacian_regularization + expression_mesh_laplacian_regularization
            # losses['laplacian_regularization'] = template_mesh_laplacian_regularization + ict_mesh_w_temp_laplacian_regularization + expression_mesh_laplacian_regularization
            # losses['laplacian_regularization'] = template_mesh_laplacian_regularization + ict_mesh_laplacian_regularization + ict_mesh_w_temp_laplacian_regularization + expression_mesh_laplacian_regularization
            losses['normal_regularization'] = template_mesh_normal_regularization
            losses['feature_regularization'] = feature_regularization
            losses['geometric_regularization'] = template_geometric_regularization + expression_geometric_regularization + pose_weight_geometric_regularization


            decay_keys = ['mask', 'segmentation', 'landmark', 'closure']
            with torch.no_grad():
                shading_decay = 0
                for k in decay_keys:
                    shading_decay += losses[k].mean() * loss_weights[k]
                shading_decay = torch.exp(-(shading_decay)).detach()
            losses['shading'] = losses['shading'] * shading_decay
            losses['perceptual_loss'] = losses['perceptual_loss'] * shading_decay
            torch.cuda.empty_cache()

            loss = torch.tensor(0., device=device) 
            for k, v in losses.items():
                if torch.isnan(v).any():
                    print(f'NAN in {k}')
                    print(losses)
                    exit()
                    continue
                loss += v.mean() * loss_weights[k]
            
            # decay value is  the ten times of summation of loss for mask, segmentation, landmark, closure. 
            decay_value = 0
            for k in decay_keys:
                decay_value += losses[k].mean() * loss_weights[k]
            decay_value *= 2

            for idx in views_subset['idx']:            
                importance[idx] = ((1 - weight_decay_rate) * importance[idx] + weight_decay_rate * decay_value).clamp(min=1e-2).item()

            acc_losses.append(losses)
            acc_total_loss += loss

            if len(acc_losses) > 9:
                losses_to_log = {}
                for k in acc_losses[0].keys():
                    val = torch.stack([l[k] for l in acc_losses]).mean() * loss_weights[k]
                    # if val > 0:
                    losses_to_log[k] = val
                losses_to_log["total_loss"] = acc_total_loss / len(acc_losses)
                if 'debug' not in run_name and not args.skip_wandb:
                    wandb.log({k: v.item() for k, v in losses_to_log.items()}, step=iteration)

                acc_losses = []
                acc_total_loss = 0

            if iteration % 100 == 1:
                # bshapes = return_dict['features'][:, :53].detach().cpu().numpy()
                # bshapes = np.round(bshapes, 2)
                # jawopen = bshapes[:, ict_facekit.expression_names.tolist().index('jawOpen')]
                # eyeblink_l = bshapes[:, ict_facekit.expression_names.tolist().index('eyeBlink_L')]
                # eyeblink_r = bshapes[:, ict_facekit.expression_names.tolist().index('eyeBlink_R')]

                # print(f"JawOpen: {jawopen}, EyeBlink_L: {eyeblink_l}, EyeBlink_R: {eyeblink_r}")

                # print(bshapes)

                print(return_dict['features'][0, 53:])
                print("=="*50)
                for k, v in losses.items():
                    # if k in losses_to_print:
                    v = v.mean()
                # if v > 0:
                    print(f"{k}: {v.item() * loss_weights[k]}")
                print("=="*50)

                dataset_sampler = torch.utils.data.WeightedRandomSampler(importance, dataset_train.len_img, replacement=True)
                dataloader_train    = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, collate_fn=dataset_train.collate, drop_last=True, sampler=dataset_sampler)


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
            torch.nn.utils.clip_grad_norm_(shader.parameters(), 2.0)
            torch.nn.utils.clip_grad_norm_(neural_blendshapes.parameters(), 2.0)

            optimizer_neural_blendshapes.step()
            scheduler_neural_blendshapes.step()
            # else:

            optimizer_shader.step()
            scheduler_shader.step()


            progress_bar.set_postfix({'loss': loss.detach().cpu().item(), 'decay': decay_value.detach().cpu().item()})

            torch.cuda.empty_cache()

            # del loss, gbuffers, d_normals, pred_color_masked, cbuffers, gbuffer_mask


            # ==============================================================================================
            # V I S U A L I Z A T I O N S
            # ==============================================================================================
            if (args.visualization_frequency > 0) and (iteration == 1 or iteration % args.visualization_frequency == 0):
            
                with torch.no_grad():
                    # debug_rgb_pred, debug_gbuffer, debug_cbuffers = run(args, mesh, debug_views, ict_facekit, neural_blendshapes, shader, renderer, device, channels_gbuffer, lgt)
                    
                    ## ============== visualize ==============================
                    # visualize_training(debug_rgb_pred, debug_cbuffers, debug_gbuffer, debug_views, images_save_path, iteration, ict_facekit=ict_facekit)

                    return_dict_ = neural_blendshapes(debug_views['img'], debug_views)
                    if not os.path.exists(str(images_save_path / "grid" / f'gt_seg_0.png')):
                        gt_segs = debug_views['skin_mask'] # Shape of B, H, W, 6
                        # for each gt_seg, save gt_seg[0]
                        import cv2
                        for ith in range(gt_segs.shape[0]):
                            gt_seg = gt_segs[ith, ..., 0]
                            gt_seg = gt_seg * 255
                            gt_seg = gt_seg.cpu().numpy().astype(np.uint8)
                            cv2.imwrite(str(images_save_path / "grid" / f'gt_seg_{ith}.png'), gt_seg)

                    # bshapes = return_dict_['features'][:, :53].detach().cpu().numpy()
                    # bshapes = np.round(bshapes, 2)
                    # jawopen = bshapes[:, ict_facekit.expression_names.tolist().index('jawOpen')]
                    # eyeblink_l = bshapes[:, ict_facekit.expression_names.tolist().index('eyeBlink_L')]
                    # eyeblink_r = bshapes[:, ict_facekit.expression_names.tolist().index('eyeBlink_R')]
                    # print(f"JawOpen: {jawopen}, EyeBlink_L: {eyeblink_l}, EyeBlink_R: {eyeblink_r}")

                    debug_gbuffer = renderer.render_batch(debug_views['camera'], return_dict_['expression_mesh_posed'].contiguous(), mesh.fetch_all_normals(return_dict_['expression_mesh_posed'], mesh),
                                            channels=channels_gbuffer, with_antialiasing=True, 
                                            canonical_v=mesh.vertices, canonical_idx=mesh.indices, canonical_uv=ict_facekit.uv_neutral_mesh)
                    debug_rgb_pred, debug_cbuffers, _ = shader.shade(debug_gbuffer, debug_views, mesh, args.finetune_color, lgt)
                    visualize_training(debug_rgb_pred, debug_cbuffers, debug_gbuffer, debug_views, images_save_path, iteration, ict_facekit=ict_facekit, save_name='expression')

                    # debug_gbuffer = renderer.render_batch(debug_views['camera'], return_dict_['ict_mesh_posed'].contiguous(), mesh.fetch_all_normals(return_dict_['ict_mesh_posed'], mesh), 
                    #                         channels=channels_gbuffer, with_antialiasing=True, 
                    #                         canonical_v=mesh.vertices, canonical_idx=mesh.indices, canonical_uv=ict_facekit.uv_neutral_mesh) 
                    # debug_rgb_pred, debug_cbuffers, _ = shader.shade(debug_gbuffer, debug_views, mesh, args.finetune_color, lgt)
                    # visualize_training(debug_rgb_pred, debug_cbuffers, debug_gbuffer, debug_views, images_save_path, iteration, ict_facekit=ict_facekit, save_name='ict')

                    debug_gbuffer = renderer.render_batch(debug_views['camera'], return_dict_['ict_mesh_w_temp_posed'].contiguous(), mesh.fetch_all_normals(return_dict_['ict_mesh_w_temp_posed'], mesh), 
                                            channels=channels_gbuffer, with_antialiasing=True, 
                                            canonical_v=mesh.vertices, canonical_idx=mesh.indices, canonical_uv=ict_facekit.uv_neutral_mesh) 
                    debug_rgb_pred, debug_cbuffers, _ = shader.shade(debug_gbuffer, debug_views, mesh, args.finetune_color, lgt)
                    visualize_training(debug_rgb_pred, debug_cbuffers, debug_gbuffer, debug_views, images_save_path, iteration, ict_facekit=ict_facekit, save_name='ict_w_temp')

                    for n in range(debug_views['img'].shape[0]):                            
                        if n != 0:
                            break
                        # write_mesh(meshes_save_path / f"mesh_{iteration:06d}_ict.obj", mesh.with_vertices(return_dict_['ict_mesh'][n]).detach().to('cpu'))    
                        write_mesh(meshes_save_path / f"mesh_{iteration:06d}_ict_w_temp.obj", mesh.with_vertices(return_dict_['ict_mesh_w_temp'][n]).detach().to('cpu'))                    
                        write_mesh(meshes_save_path / f"mesh_{iteration:06d}_exp.obj", mesh.with_vertices(return_dict_['expression_mesh'][n]).detach().to('cpu'))                    
                        # save the posed meshes as well
                        # write_mesh(meshes_save_path / f"mesh_{iteration:06d}_ict_posed.obj", mesh.with_vertices(return_dict_['ict_mesh_posed'][n]).detach().to('cpu'))
                        write_mesh(meshes_save_path / f"mesh_{iteration:06d}_ict_w_temp_posed.obj", mesh.with_vertices(return_dict_['ict_mesh_w_temp_posed'][n]).detach().to('cpu'))
                        write_mesh(meshes_save_path / f"mesh_{iteration:06d}_exp_posed.obj", mesh.with_vertices(return_dict_['expression_mesh_posed'][n]).detach().to('cpu'))

                    write_mesh(meshes_save_path / f"mesh_{iteration:06d}_temp.obj", mesh.with_vertices(return_dict_['template_mesh']).detach().to('cpu'))                                
                    del debug_gbuffer, debug_cbuffers, debug_rgb_pred, return_dict_


            if iteration == 1 or iteration % (args.visualization_frequency * 10) == 0:
                print(images_save_path / "grid" / f'grid_{iteration}.png')
                if 'debug' not in run_name and not args.skip_wandb:
                    wandb.log({"Grid": [wandb.Image(str(images_save_path / "grid" / f'grid_{iteration}.png'))]}, step=iteration)

            ## ============== save intermediate ==============================
            if (args.save_frequency > 0) and (iteration == 1 or iteration % args.save_frequency == 0):
                with torch.no_grad():
                    shader.save(shaders_save_path / f'shader.pt')
                    neural_blendshapes.save(shaders_save_path / f'neural_blendshapes.pt')

            # deformer_or_shader = not deformer_or_shader

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
        dataset_val      = DatasetLoader(args, train_dir=args.eval_dir, sample_ratio=1, pre_load=False)
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
    dataset_train    = DatasetLoader(args, train_dir=args.train_dir, sample_ratio=args.sample_idx_ratio, pre_load=False)
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
