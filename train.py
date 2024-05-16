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

    neural_blendshapes = get_neural_blendshapes(model_path=model_path, train=args.train_deformer, vertex_parts=ict_facekit.vertex_parts, device=device) 
    print(ict_canonical_mesh.vertices.shape, ict_canonical_mesh.vertices.device)
    neural_blendshapes.set_template(ict_canonical_mesh.vertices,
                                    ict_facekit.uv_neutral_mesh)

    neural_blendshapes = neural_blendshapes.to(device)

    lmk_adaptive = None
    facs_adaptive = None
    
    neural_blendshapes_params = list(neural_blendshapes.parameters())
    neural_blendshapes_encoder_params = list(neural_blendshapes.encoder.parameters())
    neural_blendshapes_others_params = list(set(neural_blendshapes_params) - set(neural_blendshapes_encoder_params))
    # adam optimizer, args.lr_encoder for the encoder, args.lr_deformer for the rest
    optimizer_neural_blendshapes = torch.optim.Adam([{'params': neural_blendshapes_encoder_params, 'lr': args.lr_encoder},
                                                    {'params': neural_blendshapes_others_params, 'lr': args.lr_deformer}])
                                                     
    scheduler_milestones = [int(args.iterations / 2), int(args.iterations * 2 / 3)]
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
    
    # ==============================================================================================
    # Loss Functions
    # ==============================================================================================
    # Initialize the loss weights and losses
    loss_weights = {
        "mask": args.weight_mask,
        "laplacian_regularization": args.weight_laplacian_regularization,
        "shading": args.weight_shading,
        "perceptual_loss": args.weight_perceptual_loss,
        "landmark": args.weight_landmark,
        "closure": args.weight_closure,
        "ict": args.weight_ict,
        "ict_landmark": args.weight_ict_landmark,
        "ict_landmark_closure": args.weight_ict_closure,
        "random_ict": args.weight_ict,
        "feature_regularization": args.weight_feature_regularization,
        # "deformation_map_regularization": 1e-5,
        "cbuffers_regularization": args.weight_cbuffers_regularization,
        # "synthetic": args.weight_synthetic,
    }
    losses = {k: torch.tensor(0.0, device=device) for k in loss_weights}
    print(loss_weights)
    if loss_weights["perceptual_loss"] > 0.0:
        VGGloss = VGGPerceptualLoss().to(device)

    print("=="*50)
    shader.train()
    
    neural_blendshapes.train()
    print("Batch Size:", args.batch_size)
    print("=="*50)

    # ==============================================================================================
    # T R A I N I N G
    # ==============================================================================================


    epochs = ((args.iterations // 5) // len(dataloader_train)) + 1
    iteration = 0
    
    progress_bar = tqdm(range(epochs))
    start = time.time()

    acc_losses = []
    acc_total_loss = 0
    
    # face_alignment = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False, 
                                                            # device='cuda' if torch.cuda.is_available() else 'cpu')

    print('initializing encoder')    

    import wandb
    if 'debug' not in run_name:
        wandb_name = args.wandb_name if args.wandb_name is not None else run_name
        wandb.init(project="neural_blendshape", name=wandb_name, config=args)
    for epoch in progress_bar:
        for iter_, views_subset in enumerate(dataloader_train):
            
            input_image = views_subset["img"].permute(0, 3, 1, 2).to(device)
            landmark = views_subset['mp_landmark'].reshape(-1, 1434) # 478*3
            blendshape = views_subset['mp_blendshape'].reshape(-1, 52)
            transform_matrix = views_subset['mp_transform_matrix'].reshape(-1, 16)

            features = torch.zeros(input_image.size(0), 128, device=device)
            features = torch.cat([features, landmark, blendshape, transform_matrix], dim=-1)
            features = neural_blendshapes.encoder.tail(features)

            features[..., :53] = torch.nn.functional.tanh(features[..., :53]) / 2. + 0.5
            
            features[..., 58] = 0
            gt = views_subset['mp_blendshape'][..., ict_facekit.mediapipe_to_ict]

            loss = torch.nn.functional.l1_loss(features[..., :53], gt[..., :53])

            return_dict = neural_blendshapes(image_input=False, features=features)
            losses['ict'], losses['random_ict'], losses['ict_landmark'], losses['ict_landmark_closure'] = ict_loss(ict_facekit, return_dict, views_subset, neural_blendshapes, renderer, lmk_adaptive)

            losses['ict'] *= loss_weights['ict']
            losses['random_ict'] *= loss_weights['random_ict']
            losses['ict_landmark'] *= loss_weights['ict_landmark']
            losses['ict_landmark_closure'] *= loss_weights['ict_landmark_closure']

            loss = loss + (losses['ict'] + losses['random_ict'] + losses['ict_landmark_closure'] + losses['ict_landmark']).mean()

            optimizer_neural_blendshapes.zero_grad()
            loss.backward() 
            optimizer_neural_blendshapes.step()

            progress_bar.set_postfix({'loss': loss.mean().detach().cpu().item(), 'ict': losses['ict'].mean().detach().cpu().item(), \
                                      'r_ict': losses['random_ict'].mean().detach().cpu().item(), 'l': \
                                        losses['ict_landmark'].mean().detach().cpu().item(), 'l_closure': losses['ict_landmark_closure'].mean().detach().cpu().item()})

    # del features, landmark, blendshape, transform_matrix
    # del input_image
    # del views_subset
    # del loss
    # del return_dict
    # del losses

    losses = {k: torch.tensor(0.0, device=device) for k in loss_weights}

    epochs = (args.iterations // len(dataloader_train)) + 1
    iteration = 0
    
    progress_bar = tqdm(range(epochs))
    start = time.time()

    acc_losses = []
    acc_total_loss = 0
    
    # face_alignment = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False, 
                                                            # device='cuda' if torch.cuda.is_available() else 'cpu')
    
    import wandb
    if 'debug' not in run_name:
        wandb_name = args.wandb_name if args.wandb_name is not None else run_name
        wandb.init(project="neural_blendshape", name=wandb_name, config=args)
    for epoch in progress_bar:
        for iter_, views_subset in enumerate(dataloader_train):
            # views_subset = debug_views
            iteration += 1
            progress_bar.set_description(desc=f'Epoch {epoch}, Iter {iteration}')

            pretrain = iteration < args.iterations // 8

            # ==============================================================================================
            # encode input images
            # ==============================================================================================
            # first, permute the input images to be in the correct format
            input_image = views_subset["img"].permute(0, 3, 1, 2).to(device)

            return_dict = neural_blendshapes(input_image, views_subset)
            features = return_dict['features']
            mesh = ict_canonical_mesh.with_vertices(return_dict["full_template_deformation"] + ict_canonical_mesh.vertices)
            deformed_vertices = return_dict["full_deformed_mesh"]

            d_normals = mesh.fetch_all_normals(deformed_vertices, mesh)
            gbuffers = renderer.render_batch(views_subset['camera'], deformed_vertices.contiguous(), d_normals, 
                                    channels=channels_gbuffer, with_antialiasing=True, 
                                    canonical_v=mesh.vertices, canonical_idx=mesh.indices, canonical_uv=ict_facekit.uv_neutral_mesh) 
            pred_color_masked, cbuffers, gbuffer_mask = shader.shade(gbuffers, views_subset, mesh, args.finetune_color, lgt)

            losses['shading'], pred_color, tonemapped_colors = shading_loss_batch(pred_color_masked, views_subset, views_subset['img'].size(0))
            losses['perceptual_loss'] = VGGloss(tonemapped_colors[0], tonemapped_colors[1], iteration)
            losses['mask'] = mask_loss(views_subset["mask"], gbuffer_mask)
            losses['landmark'], losses['closure'] = landmark_loss(ict_facekit, gbuffers, views_subset, features, neural_blendshapes, lmk_adaptive, device)
            losses['laplacian_regularization'] = laplacian_loss(mesh, ict_canonical_mesh.vertices, neural_blendshapes.face_index)

            # losses['deformation_map_regularization'] = torch.zeros(deformed_vertices.shape[0], device=device)
            # for map in return_dict['deformation_maps']:
            #     deformation_map = return_dict['deformation_maps'][map]
            #     losses['deformation_map_regularization'] += torch.mean(torch.pow(deformation_map[:, :, :-1, :] - deformation_map[:, :, 1:, :], 2), dim=[1,2,3]) + \
            #                                                 torch.mean(torch.pow(deformation_map[:, :, :, :-1] - deformation_map[:, :, :, 1:], 2), dim=[1,2,3])
                
            losses['cbuffers_regularization'] = cbuffers_regularization(cbuffers)
            losses['ict'], losses['random_ict'], losses['ict_landmark'], losses['ict_landmark_closure'] = ict_loss(ict_facekit, return_dict, views_subset, neural_blendshapes, renderer, lmk_adaptive, fullhead_template=pretrain)
            
            losses['feature_regularization'] = feature_regularization_loss(features, views_subset['mp_blendshape'][..., ict_facekit.mediapipe_to_ict], 
                                                                           views_subset["landmark"], neural_blendshapes.scale, iteration, facs_adaptive, facs_weight=0)
            # losses['feature_regularization'] += torch.mean(torch.pow(10 * return_dict['full_template_deformation'][neural_blendshapes.head_index:], 2))
            
            with torch.no_grad():
                shading_decay = torch.exp(-(losses['landmark'] + losses['closure'] + losses['mask'])).detach()
            # photometric losses decay by overall geometric loss
            # if the geometric loss is 0, the photometric losses are not decayed
            # exponentially decay the photometric losses by the overall geometric loss
            losses['shading'] = losses['shading'] * shading_decay
            losses['perceptual_loss'] = losses['perceptual_loss'] * shading_decay
            torch.cuda.empty_cache()

            # synthetic loss
            # losses['synthetic'] = synthetic_loss(views_subset, neural_blendshapes, renderer, shader, dataset_train.mediapipe, ict_facekit, ict_canonical_mesh, 1, device) 

            torch.cuda.empty_cache()


            loss = torch.tensor(0., device=device) 
            
            for k, v in losses.items():
                loss += v.mean() * loss_weights[k]

            
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
                facs = features[0, :53]
                euler_angle = features[0, 53:56]
                translation = features[0, 56:59]
                # scale = features[0, -1:]

                # print(facs)
                print(euler_angle)
                print(translation)
                print(neural_blendshapes.transform_origin.data)
                print(neural_blendshapes.scale.data)

                # import pytorch3d.transforms as pt3d

                # rotation_matrix = pt3d.euler_angles_to_matrix(features[..., 53:56], convention = 'XYZ')
                # print(rotation_matrix)
                # if not pretrain:
                #     print_indices = [0,8,16,21,30]

                #     detected_landmarks = views_subset['landmark'].clone().detach()
                #     detected_landmarks[..., :-1] = detected_landmarks[..., :-1] * 2 - 1
                #     # detected_landmarks[..., :-1] = detected_landmarks[..., :-1] * 2 - 1
                #     # print(detected_landmarks)
                #     # exit()
                #     detected_landmarks[..., 2] = detected_landmarks[..., 2] * -1

                #     gt = detected_landmarks[:, print_indices]
                #     # Get the indices of landmarks used by the handle-based deformer
                #     landmark_indices = ict_facekit.landmark_indices
                    
                #     # Extract the deformed landmarks in clip space
                #     landmarks_on_clip_space = gbuffers['deformed_verts_clip_space'][:, landmark_indices].clone()
                    
                #     # Convert the deformed landmarks to normalized coordinates
                #     landmarks_on_clip_space = landmarks_on_clip_space[..., :3] / torch.clamp(landmarks_on_clip_space[..., 3:], min=1e-8)

                #     ours = landmarks_on_clip_space[:, print_indices]

                #     print(torch.cat([gt[0], ours[0]], dim=-1))


    # print(torch.cat([detected_landmarks[..., :3], ict_landmarks_clip_space], dim=-1))

            if iteration % 100 == 1:
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
            optimizer_shader.zero_grad()
        
            neural_blendshapes.zero_grad()

            loss.backward()
            
            torch.cuda.synchronize()
            torch.nn.utils.clip_grad_norm_(shader.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(neural_blendshapes.parameters(), 5.0)

            ### increase the gradients of positional encoding following tinycudnn
            # if not pretrain:
        
            if args.grad_scale and args.fourier_features == "hashgrid":
                shader.fourier_feature_transform.params.grad /= 8.0

            optimizer_shader.step()
            # clip gradients

            optimizer_neural_blendshapes.step()

            scheduler_shader.step()
            scheduler_neural_blendshapes.step()

            progress_bar.set_postfix({'loss': loss.detach().cpu().item()})

            torch.cuda.empty_cache()

            # ==============================================================================================
            # V I S U A L I Z A T I O N S
            # ==============================================================================================
            if (args.visualization_frequency > 0) and (iteration == 1 or iteration % args.visualization_frequency == 0):
            
                with torch.no_grad():
                    debug_rgb_pred, debug_gbuffer, debug_cbuffers = run(args, mesh, debug_views, ict_facekit, neural_blendshapes, shader, renderer, device, channels_gbuffer, lgt)



                    ## ============== visualize ==============================
                    visualize_training(debug_rgb_pred, debug_cbuffers, debug_gbuffer, debug_views, images_save_path, iteration)

                    return_dict_ = neural_blendshapes(debug_views["img"].to(device), debug_views)
                    
                    jaw_index = ict_facekit.expression_names.tolist().index('jawOpen')
                    eyeblink_L_index = ict_facekit.expression_names.tolist().index('eyeBlink_L')
                    eyeblink_R_index = ict_facekit.expression_names.tolist().index('eyeBlink_R')
                    facs = return_dict_['features'][:, :53]
                    print('jawopen', facs[:, jaw_index])
                    print('eyeblink_L', facs[:, eyeblink_L_index])
                    print('eyeblink_R', facs[:, eyeblink_R_index])

                    print('gt, jaw_index', debug_views['mp_blendshape'][:, ict_facekit.mediapipe_to_ict][:, jaw_index])
                    print('gt, eyeblink_L_index', debug_views['mp_blendshape'][:, ict_facekit.mediapipe_to_ict][:, eyeblink_L_index])
                    print('gt, eyeblink_R_index', debug_views['mp_blendshape'][:, ict_facekit.mediapipe_to_ict][:, eyeblink_R_index])

                    # for i, name in enumerate(ict_facekit.expression_names):
                    #     print(name, debug_views['facs'][:, sict_facekit.mediapipe_to_ict][:, i])
                    # exit()
                    ict = ict_facekit(expression_weights = return_dict_['features'][:, :53], to_canonical = True)
                    deformed_verts = neural_blendshapes.apply_deformation(ict, return_dict_['features'])

                    # mesh_ = ict_canonical_mesh.with_vertices(deformed_verts)


                    d_normals = mesh.fetch_all_normals(deformed_verts, mesh)


                    debug_gbuffer = renderer.render_batch(debug_views['camera'], deformed_verts.contiguous(), d_normals, 
                                            channels=channels_gbuffer, with_antialiasing=True, 
                                            canonical_v=mesh.vertices, canonical_idx=mesh.indices, canonical_uv=ict_facekit.uv_neutral_mesh) 

                    debug_rgb_pred, debug_cbuffers, gbuffer_mask = shader.shade(debug_gbuffer, debug_views, mesh, args.finetune_color, lgt)


                    visualize_training(debug_rgb_pred, debug_cbuffers, debug_gbuffer, debug_views, images_save_path, iteration, save_name='ict')



                    del debug_gbuffer, debug_cbuffers
            if iteration == 1 or iteration % (args.visualization_frequency * 10) == 0:
                print(images_save_path / "grid" / f'grid_{iteration}.png')
                if 'debug' not in run_name:
                    wandb.log({"Grid": [wandb.Image(str(images_save_path / "grid" / f'grid_{iteration}.png'))]}, step=iteration)

            ## ============== save intermediate ==============================
            if (args.save_frequency > 0) and (iteration == 1 or iteration % args.save_frequency == 0):
                with torch.no_grad():
                    write_mesh(meshes_save_path / f"mesh_{iteration:06d}.obj", mesh.detach().to('cpu'))                                
                    shader.save(shaders_save_path / f'shader.pt')
                    neural_blendshapes.save(shaders_save_path / f'neural_blendshapes.pt')

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
    write_mesh(meshes_save_path / f"mesh_latest.obj", mesh.detach().to('cpu'))
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
    assert dataset_train.len_img == len(dataset_train.importance)
    dataset_sampler = torch.utils.data.WeightedRandomSampler(dataset_train.importance, dataset_train.len_img, replacement=True)
    dataloader_train    = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, collate_fn=dataset_train.collate, drop_last=True, sampler=dataset_sampler)
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
