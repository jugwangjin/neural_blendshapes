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
    neural_blendshapes.set_template(ict_canonical_mesh.vertices,
                                    ict_facekit.uv_neutral_mesh)

    neural_blendshapes = neural_blendshapes.to(device)


    lmk_adaptive = None
    facs_adaptive = None
    
    neural_blendshapes_params = list(neural_blendshapes.parameters())
    neural_blendshapes_encoder_params = list(neural_blendshapes.encoder.parameters())
    neural_blendshapes_expression_params = list(neural_blendshapes.expression_deformer.parameters())
    neural_blendshapes_others_params = list(set(neural_blendshapes_params) - set(neural_blendshapes_encoder_params) - set(neural_blendshapes_expression_params))
    # adam optimizer, args.lr_encoder for the encoder, args.lr_deformer for the rest
    # optimizer_encoder = torch.optim.Adam(neural_blendshapes_encoder_params, lr=args.lr_encoder, betas=(0.1, 0.5))
    # optimizer_neural_blendshapes = torch.optim.Adam([{'params': neural_blendshapes_others_params, 'lr': args.lr_deformer},
    #                                                 {'params': neural_blendshapes_expression_params, 'lr': args.lr_jacobian}],
    #                                                 betas=(0.1, 0.5))
    optimizer_neural_blendshapes = torch.optim.Adam([{'params': neural_blendshapes_encoder_params, 'lr': args.lr_encoder},
                                                    {'params': neural_blendshapes_others_params, 'lr': args.lr_deformer},
                                                    {'params': neural_blendshapes_expression_params, 'lr': args.lr_jacobian}],
                                                    betas=(0.1, 0.5))
                                                     
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
        "semantic_stat": args.weight_semantic_stat,
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
    
    # face_alignment = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False, 
                                                            # device='cuda' if torch.cuda.is_available() else 'cpu')
    deformer_or_shader = True
    
    import wandb
    if 'debug' not in run_name:
        wandb_name = args.wandb_name if args.wandb_name is not None else run_name
        wandb.init(project="neural_blendshape", name=wandb_name, config=args)
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


        for iter_, views_subset in tqdm(enumerate(dataloader_train)):
            # views_subset = debug_views
            # if deformer_or_shader:
            iteration += 1
            progress_bar.set_description(desc=f'Epoch {epoch}, Iter {iteration}')

            pretrain = iteration < args.iterations // 3
            

            # use_jaw = iteration < args.iterations // 5
            # use_jaw = iteration < args.iterations // 3
            use_jaw = True
            pretrain_lmk = iteration < args.iterations // 6

            if iteration == args.iterations // 2:
                loss_weights['landmark'] /= 10
                loss_weights['closure'] /= 10

            # ==============================================================================================
            # encode input images
            # ==============================================================================================
            # first, permute the input images to be in the correct format
            input_image = views_subset["img"].permute(0, 3, 1, 2).to(device)

            return_dict = neural_blendshapes(input_image, views_subset)
            features = return_dict['features']
            mesh = ict_canonical_mesh.with_vertices(return_dict['template_mesh'])
            # mesh = ict_canonical_mesh.with_vertices(ict_canonical_mesh.vertices)
            deformed_vertices = return_dict["full_deformed_mesh"]

            d_normals = mesh.fetch_all_normals(deformed_vertices, mesh)
            gbuffers = renderer.render_batch(views_subset['camera'], deformed_vertices.contiguous(), d_normals, 
                                    channels=channels_gbuffer, with_antialiasing=True, 
                                    canonical_v=mesh.vertices, canonical_idx=mesh.indices, canonical_uv=ict_facekit.uv_neutral_mesh, vertex_labels = ict_facekit.vertex_labels) 
            
            cloned_ict = return_dict["full_ict_deformed_mesh"].clone()
            ict_gbuffers = renderer.render_batch(views_subset['camera'], cloned_ict.contiguous(), mesh.fetch_all_normals(cloned_ict, mesh), 
                                    channels=['canonical_position'], with_antialiasing=True, 
                                    canonical_v=mesh.vertices, canonical_idx=mesh.indices, canonical_uv=ict_facekit.uv_neutral_mesh, vertex_labels = ict_facekit.vertex_labels) 

            pred_color_masked, cbuffers, gbuffer_mask = shader.shade(gbuffers, views_subset, mesh, args.finetune_color, lgt)

            losses['shading'], pred_color, tonemapped_colors = shading_loss_batch(pred_color_masked, views_subset, views_subset['img'].size(0))
            losses['perceptual_loss'] = VGGloss(tonemapped_colors[0], tonemapped_colors[1], iteration)
            losses['mask'] = mask_loss(views_subset["mask"], gbuffer_mask)
            losses['segmentation'], losses['semantic_stat'] = segmentation_loss(views_subset, gbuffers, ict_facekit.parts_indices, ict_canonical_mesh.vertices)
            ict_seg, semantic_stat = segmentation_loss(views_subset, ict_gbuffers, ict_facekit.parts_indices, ict_canonical_mesh.vertices)
            losses['segmentation'] += ict_seg
            losses['semantic_stat'] += semantic_stat
            
            losses['landmark'], losses['closure'] = landmark_loss(ict_facekit, gbuffers, views_subset, use_jaw, device)
            ict_lmk, ict_closure = landmark_loss(ict_facekit, ict_gbuffers, views_subset, use_jaw, device)
            losses['landmark'] += ict_lmk
            losses['closure'] += ict_closure

            losses['laplacian_regularization'] = laplacian_loss_two_meshes(ict_canonical_mesh, return_dict['template_mesh'], ict_canonical_mesh.vertices)
            # losses['laplacian_regularization'] = laplacian_loss_two_meshes(ict_canonical_mesh, return_dict['full_deformed_mesh'], return_dict['full_ict_deformed_mesh'].detach())
            losses['laplacian_regularization'] += (1e-5 / args.weight_laplacian_regularization) * (return_dict['additional_jacobian']).pow(2).mean() # close to ict 

            losses['normal_regularization'] = normal_reg_loss(ict_canonical_mesh, return_dict['template_mesh'], ict_canonical_mesh.vertices) # template
            # losses['normal_regularization'] = normal_reg_loss(ict_canonical_mesh, return_dict['full_deformed_mesh'], return_dict['full_ict_deformed_mesh'].detach()) # template

            losses['feature_regularization'] = feature_regularization_loss(features, views_subset['mp_blendshape'][..., ict_facekit.mediapipe_to_ict], 
                                                                           neural_blendshapes, facs_weight=0) 

            with torch.no_grad():
                shading_decay = torch.exp(-(losses['mask']).mean()).detach()
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
            
            for idx in views_subset['idx']:            
                importance[idx] = (importance[idx] * 0.95 + 0.05 * 10*(loss - losses['shading'] * loss_weights['shading']).item()).clamp(min=1e-2)

            acc_losses.append(losses)
            acc_total_loss += loss

            if len(acc_losses) > 9:
                losses_to_log = {}
                for k in acc_losses[0].keys():
                    val = torch.stack([l[k] for l in acc_losses]).mean() * loss_weights[k]
                    # if val > 0:
                    losses_to_log[k] = val
                losses_to_log["total_loss"] = acc_total_loss / len(acc_losses)
                if 'debug' not in run_name:
                    wandb.log({k: v.item() for k, v in losses_to_log.items()}, step=iteration)

                acc_losses = []
                acc_total_loss = 0

            if iteration % 100 == 1:
                print(return_dict['features'][:, 53:])
                print(neural_blendshapes.encoder.softplus(neural_blendshapes.encoder.blendshapes_multiplier))
                print("=="*50)
                for k, v in losses.items():
                    # if k in losses_to_print:
                    v = v.mean()
                # if v > 0:
                    print(f"{k}: {v.item() * loss_weights[k]}")
                print("=="*50)

                            
                dataset_sampler = torch.utils.data.WeightedRandomSampler(importance, dataset_train.len_img, replacement=True)
                dataloader_train    = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, collate_fn=dataset_train.collate, drop_last=True, sampler=dataset_sampler)


            ict_loss = (ict_lmk * loss_weights['landmark'] + ict_closure * loss_weights['closure']).mean()


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
            torch.nn.utils.clip_grad_norm_(shader.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(neural_blendshapes.parameters(), 5.0)

            ### increase the gradients of positional encoding following tinycudnn
            # if not pretrain:
            # if deformer_or_shader:
            optimizer_neural_blendshapes.step()
            scheduler_neural_blendshapes.step()
            # else:

            optimizer_shader.step()
            scheduler_shader.step()


            progress_bar.set_postfix({'loss': loss.detach().cpu().item(), 'decay': 5*(loss - losses['shading'] * loss_weights['shading']).detach().cpu().item()})

            torch.cuda.empty_cache()

            del loss, gbuffers, d_normals, deformed_vertices, pred_color_masked, cbuffers, gbuffer_mask, features


            # ==============================================================================================
            # V I S U A L I Z A T I O N S
            # ==============================================================================================
            if (args.visualization_frequency > 0) and (iteration == 1 or iteration % args.visualization_frequency == 0):
            
                with torch.no_grad():
                    debug_rgb_pred, debug_gbuffer, debug_cbuffers = run(args, mesh, debug_views, ict_facekit, neural_blendshapes, shader, renderer, device, channels_gbuffer, lgt)
# 
                    

                    ## ============== visualize ==============================
                    visualize_training(debug_rgb_pred, debug_cbuffers, debug_gbuffer, debug_views, images_save_path, iteration, ict_facekit=ict_facekit)

                    return_dict_ = neural_blendshapes(debug_views['img'], debug_views)
                    only_expression_mesh = return_dict_['full_ict_deformed_mesh']
            
                    debug_gbuffer = renderer.render_batch(debug_views['camera'], only_expression_mesh.contiguous(), mesh.fetch_all_normals(only_expression_mesh, mesh), 
                                            channels=channels_gbuffer, with_antialiasing=True, 
                                            canonical_v=mesh.vertices, canonical_idx=mesh.indices, canonical_uv=ict_facekit.uv_neutral_mesh, vertex_labels = ict_facekit.vertex_labels) 

                    debug_rgb_pred, debug_cbuffers, _ = shader.shade(debug_gbuffer, debug_views, mesh, args.finetune_color, lgt)
                    visualize_training(debug_rgb_pred, debug_cbuffers, debug_gbuffer, debug_views, images_save_path, iteration, ict_facekit=ict_facekit, save_name='ict')
                    for n in range(debug_views['img'].shape[0]):

                        write_mesh(meshes_save_path / f"mesh_{iteration:06d}_{n}.obj", mesh.with_vertices(return_dict_['full_deformed_mesh'][n]).detach().to('cpu'))                                
                        if n != 0:
                            break
                    for nn in range(views_subset['img'].shape[0]):
                        if nn != 0:
                            break
                        only_expression_mesh = mesh.with_vertices(return_dict['only_expression_mesh'][nn]).detach().to('cpu')
                        write_mesh(meshes_save_path / f"expression_{nn:02d}.obj", only_expression_mesh)                    

                    del debug_gbuffer, debug_cbuffers, debug_rgb_pred, only_expression_mesh, return_dict_

            if iteration == 1 or iteration % (args.visualization_frequency * 10) == 0:
                print(images_save_path / "grid" / f'grid_{iteration}.png')
                if 'debug' not in run_name:
                    wandb.log({"Grid": [wandb.Image(str(images_save_path / "grid" / f'grid_{iteration}.png'))]}, step=iteration)

            ## ============== save intermediate ==============================
            if (args.save_frequency > 0) and (iteration == 1 or iteration % args.save_frequency == 0):
                with torch.no_grad():
                    write_mesh(meshes_save_path / f"mesh_{iteration:06d}.obj", mesh.with_vertices(return_dict['template_mesh']).detach().to('cpu'))                                
                    shader.save(shaders_save_path / f'shader.pt')
                    neural_blendshapes.save(shaders_save_path / f'neural_blendshapes.pt')

            # deformer_or_shader = not deformer_or_shader

    end = time.time()
    total_time = ((end - start) % 3600)
    print("TIME TAKEN (mins):", int(total_time // 60))

    if 'debug' not in run_name:
        wandb.finish()

    # ==============================================================================================
    # s a v e
    # ==============================================================================================
    with open(experiment_dir / "args.txt", "w") as text_file:
        print(f"{args}", file=text_file)
    # write_mesh(meshes_save_path / f"mesh_latest.obj", mesh.with_vertices(return_dict['template_mesh']).detach().to('cpu'))
    shader.save(shaders_save_path / f'shader_latest.pt')
    neural_blendshapes.save(shaders_save_path / f'neural_blendshapes_latest.pt')

    # ==============================================================================================
    # FINAL: qualitative and quantitative results
    # ==============================================================================================

    ## ============== free memory before evaluation ==============================
    del dataset_train, dataloader_train, debug_views, views_subset

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
