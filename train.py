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

def load_ict_facekit(args, device):
    ict_facekit = ICTFaceKitTorch(npy_dir = './assets/ict_facekit_torch.npy', canonical = Path(args.input_dir) / 'ict_identity.npy')
    ict_facekit = ict_facekit.to(device)
    # ict_facekit.eval()

    ict_canonical_mesh = Mesh(ict_facekit.canonical[0].cpu().data, ict_facekit.faces.cpu().data, device=device)
    print(ict_canonical_mesh.vmapping.shape if ict_canonical_mesh.vmapping is not None else None)
    ict_canonical_mesh.compute_connectivity()
    print(ict_canonical_mesh.vmapping.shape)
    ict_facekit.update_vmapping(ict_canonical_mesh.vmapping.cpu().data.numpy())

    return ict_facekit, ict_canonical_mesh

def main(args, device, dataset_train, dataloader_train, debug_views):
    ## ============== Dir ==============================
    run_name = args.run_name if args.run_name is not None else args.input_dir.parent.name
    images_save_path, images_eval_save_path, meshes_save_path, shaders_save_path, experiment_dir = make_dirs(args, run_name, args.finetune_color)
    copy_sources(args, run_name)

    ## ============== load ict facekit ==============================
    ict_facekit, ict_canonical_mesh = load_ict_facekit(args, device)    
    write_mesh(Path(meshes_save_path / "init_ict_canonical.obj"), ict_canonical_mesh.to('cpu'))

    # ict_identity_path = Path(experiment_dir / "stage_1" / "network_weights" / f"ict_identity_latest.pt")
    # assert os.path.exists(ict_identity_path)
    # ict_identity = torch.load(str(ict_identity_path))
    # ict_facekit.identity.data = ict_identity

    ## ============== renderer ==============================
    aabb = AABB(ict_canonical_mesh.vertices.cpu().numpy())
    ict_mesh_aabb = [torch.min(ict_canonical_mesh.vertices, dim=0).values, torch.max(ict_canonical_mesh.vertices, dim=0).values]

    renderer = Renderer(device=device)
    renderer.set_near_far(dataset_train, torch.from_numpy(aabb.corners).to(device), epsilon=0.5)
    channels_gbuffer = ['mask', 'position', 'normal', "canonical_position"]
    print("Rasterizing:", channels_gbuffer)
    
    renderer_visualization = Renderer(device=device)
    renderer_visualization.set_near_far(dataset_train, torch.from_numpy(aabb.corners).to(device), epsilon=0.5)

    # model_path = Path(experiment_dir / "stage_1" / "network_weights" / f"encoder_latest.pt")
    # assert os.path.exists(model_path)
    # encoder.load_state_dict(torch.load(str(model_path)))

    # ==============================================================================================
    # deformation 
    # ==============================================================================================

    model_path = None
    print("=="*50)
    print("Training Deformer")

    neural_blendshapes = get_neural_blendshapes(model_path=model_path, train=args.train_deformer, device=device) 
    print(ict_canonical_mesh.vertices.shape, ict_canonical_mesh.vertices.device)
    neural_blendshapes.set_template(ict_canonical_mesh.vertices, ict_facekit.coords_min, ict_facekit.coords_max)

    neural_blendshapes = neural_blendshapes.to(device)

    

    optimizer_neural_blendshapes = torch.optim.Adam(list(neural_blendshapes.parameters()), lr=args.lr_deformer)
    # for name, param in neural_blendshapes.named_parameters():
    #     print(name)
    # print(neural_blendshapes.coords_min, neural_blendshapes.coords_max)
    # exit()
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

    # ==============================================================================================
    # Loss Functions
    # ==============================================================================================
    # Initialize the loss weights and losses
    loss_weights = {
        "mask": args.weight_mask,
        "normal_regularization": args.weight_normal_regularization,
        "laplacian_regularization": args.weight_laplacian_regularization,
        "shading": args.weight_shading,
        "perceptual_loss": args.weight_perceptual_loss,
        "albedo_regularization": args.weight_albedo_regularization,
        "roughness_regularization": args.weight_roughness_regularization,
        "white_light_regularization": args.weight_white_lgt_regularization,
        "fresnel_coeff": args.weight_fresnel_coeff,
        "normal": args.weight_normal,
        "normal_laplacian": args.weight_normal_laplacian,
        "landmark": args.weight_landmark,
        "closure": args.weight_closure,
        "ict": args.weight_ict,
        "ict_landmark": args.weight_ict_landmark,
        "random_ict": args.weight_random_ict,
        # "ict_identity": args.weight_ict_identity,
        "feature_regularization": args.weight_feature_regularization,
        # "head_direction": args.weight_head_direction,
        # "direction_estimation": args.weight_direction_estimation,
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
    epochs = (args.iterations // len(dataloader_train)) + 1
    iteration = 0
    
    progress_bar = tqdm(range(epochs))
    start = time.time()

    acc_losses = []
    acc_total_loss = 0
    import wandb
    if 'debug' not in run_name:
        wandb.init(project="neural_blendshape", name=run_name, config=args)
    for epoch in progress_bar:
        for iter_, views_subset in enumerate(dataloader_train):
            # views_subset = debug_views
            iteration += 1
            progress_bar.set_description(desc=f'Epoch {epoch}, Iter {iteration}')

            # ==============================================================================================
            # update/displace vertices
            # ==============================================================================================

            # ==============================================================================================
            # encode input images
            # ==============================================================================================
            # first, permute the input images to be in the correct format
            input_image = views_subset["img"].permute(0, 3, 1, 2).to(device)

            return_dict = neural_blendshapes(input_image)

            features = return_dict['features']


            mesh = ict_canonical_mesh.with_vertices(return_dict["template_deformation"] + ict_canonical_mesh.vertices)
            
            # ==============================================================================================
            # deformation of canonical mesh
            # ==============================================================================================
            deformed_vertices = return_dict['deformed_mesh']

            d_normals = mesh.fetch_all_normals(deformed_vertices, mesh)

            # ==============================================================================================
            # R A S T E R I Z A T I O N
            # ==============================================================================================
            gbuffers = renderer.render_batch(views_subset['camera'], deformed_vertices.contiguous(), d_normals, 
                                    channels=channels_gbuffer, with_antialiasing=True, 
                                    canonical_v=mesh.vertices, canonical_idx=mesh.indices) 
            
            # ============================================= =================================================
            # loss function 
            # ==============================================================================================
            ## ======= regularization autoencoder========
            # losses['normal_regularization'] = 0
            # losses['laplacian_regularization'] = 0
            # num_meshes = deformed_vertices.shape[0]
            # for nth_mesh in range(num_meshes):
            #     mesh_ = ict_canonical_mesh.with_vertices(deformed_vertices[nth_mesh])
            losses['normal_regularization'] = normal_consistency_loss(mesh)
            losses['laplacian_regularization'] = laplacian_loss(mesh)
            # losses['normal_regularization'] /= num_meshes
            # losses['laplacian_regularization'] /= num_meshes

            ## ============== color + regularization for color ==============================
            pred_color_masked, cbuffers, gbuffer_mask = shader.shade(gbuffers, views_subset, mesh, args.finetune_color, lgt)

            losses['shading'], pred_color, tonemapped_colors = shading_loss_batch(pred_color_masked, views_subset, views_subset['img'].size(0))
            losses['perceptual_loss'] = VGGloss(tonemapped_colors[0], tonemapped_colors[1], iteration)
            
            losses['mask'] = mask_loss(views_subset["mask"], gbuffer_mask)

            ## ======= regularization color ========
            losses['albedo_regularization'] = albedo_regularization(_adaptive, shader, mesh, device, None, iteration)
            losses['white_light_regularization'] = white_light(cbuffers)
            losses['roughness_regularization'] = roughness_regularization(cbuffers["roughness"], views_subset["skin_mask"], views_subset["mask"], r_mean=args.r_mean)
            losses["fresnel_coeff"] = spec_intensity_regularization(cbuffers["ko"], views_subset["skin_mask"], views_subset["mask"])
            
            # landmark losses
            losses['landmark'], losses['closure'] = landmark_loss(ict_facekit, gbuffers, views_subset, features, device)
            # losses['landmark'], losses['closure'], losses['head_direction'], losses['direction_estimation'] = landmark_loss(ict_facekit, gbuffers, views_subset, features, device)
             
            # losses['closure'] = closure_loss(ict_facekit, gbuffers, views_subset, device)

            # normal loss
            # losses['normal_laplacian'] = normal_loss(gbuffers, views_subset, gbuffer_mask, device)
            losses['normal'], losses['normal_laplacian'] = normal_loss(gbuffers, views_subset, gbuffer_mask, device)

            # ict loss
            losses['ict'], losses['random_ict'], losses['ict_landmark'] = ict_loss(ict_facekit, return_dict, views_subset, neural_blendshapes, renderer, gbuffers)
            
            # feature regularization
            losses['feature_regularization'] = feature_regularization_loss(features)

            loss = torch.tensor(0., device=device) 
            
            for k, v in losses.items():
                loss += v * loss_weights[k]
            
            acc_losses.append(losses)
            acc_total_loss += loss

            if len(acc_losses) > 9:
                losses_to_log = {}
                for k in acc_losses[0].keys():
                    losses_to_log[k] = torch.stack([l[k] for l in acc_losses]).mean() * loss_weights[k]
                losses_to_log["total_loss"] = acc_total_loss / len(acc_losses)
                if 'debug' not in run_name:
                    wandb.log({k: v.item() for k, v in losses_to_log.items()}, step=iteration)

                acc_losses = []
                acc_total_loss = 0

            if iteration % 100 == 0:
                DIRECTION_PAIRS = torch.tensor([[36, 64],[45, 48]]).int()
                landmark_indices = ict_facekit.landmark_indices
                landmarks_on_clip_space = gbuffers['deformed_verts_clip_space'][:, landmark_indices].clone()
                landmarks_on_clip_space = landmarks_on_clip_space[..., :3] / torch.clamp(landmarks_on_clip_space[..., 3:], min=1e-8) # shape of B, N, 3
    
                detected_landmarks = views_subset['landmark'].detach().data  # shape of B N 3        
                detected_landmarks[..., :-1] = detected_landmarks[..., :-1] * 2 - 1
                detected_landmarks[..., 2] = detected_landmarks[..., 2] * -1

                detected_normal = detected_landmarks[:, DIRECTION_PAIRS[:, 0], :3] - detected_landmarks[:, DIRECTION_PAIRS[:, 1], :3]
                detected_normal = torch.cross(detected_normal[:, 0], detected_normal[:, 1], dim=1)
                detected_normal = detected_normal / (torch.norm(detected_normal, dim=1, keepdim=True) + 1e-8)

                deformed_normal = landmarks_on_clip_space[:, DIRECTION_PAIRS[:, 0], :3] - landmarks_on_clip_space[:, DIRECTION_PAIRS[:, 1], :3]
                deformed_normal = torch.cross(deformed_normal[:, 0], deformed_normal[:, 1], dim=1)
                deformed_normal = deformed_normal / (torch.norm(deformed_normal, dim=1, keepdim=True) + 1e-8)

                print(torch.cat([detected_normal, deformed_normal, detected_normal - deformed_normal], dim=1))

                print("=="*50)
                for k, v in losses.items():
                    # if k in losses_to_print:
                    if v > 0:
                        print(f"{k}: {v.item()}")
                print("=="*50)

            # ==============================================================================================
            # Optimizer step
            # ==============================================================================================
            optimizer_shader.zero_grad()
        
            neural_blendshapes.zero_grad()

            loss.backward()
            torch.cuda.synchronize()

            ### increase the gradients of positional encoding following tinycudnn
            if args.grad_scale and args.fourier_features == "hashgrid":
                shader.fourier_feature_transform.params.grad /= 8.0

            # clip gradients
            torch.nn.utils.clip_grad_norm_(shader.parameters(), 2.0)
            torch.nn.utils.clip_grad_norm_(neural_blendshapes.parameters(), 2.0)


            optimizer_shader.step()
        
            optimizer_neural_blendshapes.step()

            progress_bar.set_postfix({'loss': loss.detach().cpu().item()})

            # ==============================================================================================
            # warning: check if light mlp diverged
            # ==============================================================================================
            '''
            We do not use an activation function for the output layer of light MLP because we are learning in sRGB space where the values 
            are not restricted between 0 and 1. As a result, the light MLP diverges sometimes and predicts only zero values. 
            Hence, we have included the try and catch block to automatically restart the training during this case. 
            '''
            if iteration == 100:
                convert_uint = lambda x: torch.from_numpy(np.clip(np.rint(dataset_util.rgb_to_srgb(x).detach().cpu().numpy() * 255.0), 0, 255).astype(np.uint8)).to(device)
                try:
                    diffuse_shading = convert_uint(cbuffers["shading"])
                    specular_shading = convert_uint(cbuffers["specu"])
                    if torch.count_nonzero(diffuse_shading) == 0 or torch.count_nonzero(specular_shading) == 0:
                        raise ValueError("All values predicted from light MLP are zero")
                except ValueError as e:
                    print(f"Error: {e}")
                    raise  # Raise the exception to exit the current execution of main()
            
            # ==============================================================================================
            # V I S U A L I Z A T I O N S
            # ==============================================================================================
            if (args.visualization_frequency > 0) and (iteration == 1 or iteration % args.visualization_frequency == 0):
            
                with torch.no_grad():
                    debug_rgb_pred, debug_gbuffer, debug_cbuffers = run(args, mesh, debug_views, ict_facekit, neural_blendshapes, shader, renderer, device, channels_gbuffer, lgt)
                    ## ============== visualize ==============================
                    visualize_training(debug_rgb_pred, debug_cbuffers, debug_gbuffer, debug_views, images_save_path, iteration)
                    del debug_gbuffer, debug_cbuffers
            if iteration == 1 or iteration % (args.visualization_frequency * 10) == 0:
                print(images_save_path / "grid" / f'grid_{iteration}.png')
                if 'debug' not in run_name:
                    wandb.log({"Grid": [wandb.Image(str(images_save_path / "grid" / f'grid_{iteration}.png'))]}, step=iteration)

            ## ============== save intermediate ==============================
            if (args.save_frequency > 0) and (iteration == 1 or iteration % args.save_frequency == 0):
                with torch.no_grad():
                    write_mesh(meshes_save_path / f"mesh_{iteration:06d}.obj", mesh.detach().to('cpu'))                                
                    shader.save(shaders_save_path / f'shader_{iteration:06d}.pt')
                    neural_blendshapes.save(shaders_save_path / f'neural_blendshapes_{iteration:06d}.pt')

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
    dataloader_train    = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, collate_fn=dataset_train.collate, shuffle=True, drop_last=True)
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
            print("--"*50)
            print("Warning: Re-initializing main() because the training of light MLP diverged and all the values are zero. If the training does not restart, please end it and restart. ")
            print("--"*50)
            time.sleep(5)

        except Exception as e:
            print(e)
            print('Error: Unexpected error occurred. Aborting the training.')
            raise e

    ### ============== defaults: fine tune color ==============================
    # set_defaults_finetune(args)

    # while True:
    #     try:
    #         main(args, device, dataset_train, dataloader_train, debug_views)
    #         break  # Exit the loop if main() runs successfully
    #     except Exception as e:
    #         print(e)
    #         #time.sleep(10)
    #         print("--"*50)
    #         print("Warning: Re-initializing main() because the training of light MLP diverged and all the values are zero. If the training does not restart, please end it and restart. ")
    #         print("--"*50)
    #         raise e