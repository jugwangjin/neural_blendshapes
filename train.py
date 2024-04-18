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
    NeuralShader, get_mlp_deformer, ResnetEncoder
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
    ict_facekit.train()

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

    feature_dim = args.feature_dim
    head_deformer_layers = args.head_deformer_layers
    head_deformer_hidden_dim = args.head_deformer_hidden_dim
    head_deformer_multires = args.head_deformer_multires
    eye_deformer_layers = args.eye_deformer_layers
    eye_deformer_hidden_dim = args.eye_deformer_hidden_dim
    eye_deformer_multires = args.eye_deformer_multires

    # ==============================================================================================
    # encoder
    # ==============================================================================================
    encoder = ResnetEncoder(outsize=feature_dim)
    encoder.to(device)

    # model_path = Path(experiment_dir / "stage_1" / "network_weights" / f"encoder_latest.pt")
    # assert os.path.exists(model_path)
    # encoder.load_state_dict(torch.load(str(model_path)))

    # ==============================================================================================
    # deformation 
    # ==============================================================================================

    model_path = None
    print("=="*50)
    print("Training Deformer")

    deformer_net = get_mlp_deformer(input_feature_dim=feature_dim, head_deformer_layers=head_deformer_layers, 
                                        head_deformer_hidden_dim=head_deformer_hidden_dim, head_deformer_multires=head_deformer_multires,
                                                eye_deformer_layers=eye_deformer_layers, eye_deformer_hidden_dim=eye_deformer_hidden_dim,
                                                  eye_deformer_multires=eye_deformer_multires, 
                                                  model_path=model_path, train=args.train_deformer, device=device) 

    # set deformer_net template to make efficient
    head_template = ict_canonical_mesh.vertices[ict_facekit.head_indices].to(device)
    eye_template = ict_canonical_mesh.vertices[ict_facekit.eyeball_indices].to(device)
    print(ict_canonical_mesh.vertices.shape, head_template.shape, eye_template.shape, ict_facekit.canonical.shape)

    deformer_net.set_template((head_template, eye_template))

    optimizer_deformer = torch.optim.Adam(list(deformer_net.parameters()), lr=args.lr_deformer)
    optimizer_encoder = torch.optim.Adam(list(encoder.parameters()), lr=args.lr_encoder)
    optimizer_ict_identity = torch.optim.Adam([ict_facekit.identity], lr=args.lr_deformer * 1e-1)

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
        # "normal_regularization": args.weight_normal_regularization,
        # "laplacian_regularization": args.weight_laplacian_regularization,
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
        "random_ict": args.weight_random_ict,
        "ict_identity": args.weight_ict_identity,
        "feature_regularization": args.weight_feature_regularization,
    }

    losses = {k: torch.tensor(0.0, device=device) for k in loss_weights}
    print(loss_weights)
    if loss_weights["perceptual_loss"] > 0.0:
        VGGloss = VGGPerceptualLoss().to(device)

    print("=="*50)
    shader.train()
    
    deformer_net.train()
    encoder.train()
    ict_facekit.train()
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
    wandb.init(project="neural_blendshape", name=run_name, config=args)
    for epoch in progress_bar:
        for iter_, views_subset in enumerate(dataloader_train):
            iteration += 1
            progress_bar.set_description(desc=f'Epoch {epoch}, Iter {iteration}')

            # ==============================================================================================
            # update/displace vertices
            # ==============================================================================================
            mesh = ict_canonical_mesh.with_vertices(ict_canonical_mesh.vertices)

            # ==============================================================================================
            # encode input images
            # ==============================================================================================
            # first, permute the input images to be in the correct format
            encoder_input = views_subset["img"].permute(0, 3, 1, 2).to(device)
            features = encoder(encoder_input.to(device))
            
            # ==============================================================================================
            # deformation of canonical mesh
            # ==============================================================================================
            deformed = deformer_net(features, vertices=None)
            
            deformed_vertices = mesh.vertices[None].repeat(args.batch_size, 1, 1)
            deformed_vertices[:, ict_facekit.head_indices] = deformed["deformed_head"]
            deformed_vertices[:, ict_facekit.eyeball_indices] = deformed["deformed_eyes"]

            d_normals = mesh.fetch_all_normals(deformed_vertices, mesh)

            # ==============================================================================================
            # R A S T E R I Z A T I O N
            # ==============================================================================================
            gbuffers = renderer.render_batch(views_subset['camera'], deformed_vertices.contiguous(), d_normals, 
                                    channels=channels_gbuffer, with_antialiasing=True, 
                                    canonical_v=mesh.vertices, canonical_idx=mesh.indices) 
            
            # ==============================================================================================
            # loss function 
            # ==============================================================================================
            ## ======= regularization autoencoder========
            # losses['normal_regularization'] = 0
            # losses['laplacian_regularization'] = 0
            # num_meshes = deformed_vertices.shape[0]
            # for nth_mesh in range(num_meshes):
            #     mesh_ = ict_canonical_mesh.with_vertices(deformed_vertices[nth_mesh])
            #     losses['normal_regularization'] += normal_consistency_loss(mesh_)
            #     losses['laplacian_regularization'] += laplacian_loss(mesh_)
            # losses['normal_regularization'] /= num_meshes
            # losses['laplacian_regularization'] /= num_meshes

            ## ============== color + regularization for color ==============================
            pred_color_masked, cbuffers, gbuffer_mask = shader.shade(gbuffers, views_subset, mesh, args.finetune_color, lgt)

            losses['shading'], pred_color, tonemapped_colors = shading_loss_batch(pred_color_masked, views_subset, args.batch_size)
            losses['perceptual_loss'] = VGGloss(tonemapped_colors[0], tonemapped_colors[1], iteration)
            
            losses['mask'] = mask_loss(views_subset["mask"], gbuffer_mask)

            ## ======= regularization color ========
            losses['albedo_regularization'] = albedo_regularization(_adaptive, shader, mesh, device, None, iteration)
            losses['white_light_regularization'] = white_light(cbuffers)
            losses['roughness_regularization'] = roughness_regularization(cbuffers["roughness"], views_subset["skin_mask"], views_subset["mask"], r_mean=args.r_mean)
            losses["fresnel_coeff"] = spec_intensity_regularization(cbuffers["ko"], views_subset["skin_mask"], views_subset["mask"])
            
            # landmark losses
            losses['landmark'], losses['closure'] = landmark_loss(ict_facekit, gbuffers, views_subset, device)
            # losses['closure'] = closure_loss(ict_facekit, gbuffers, views_subset, device)

            # normal loss
            losses['normal'], losses['normal_laplacian'] = normal_loss(gbuffers, views_subset, gbuffer_mask, device)

            # ict loss
            losses['ict'], losses['random_ict'] = ict_loss(ict_facekit, deformed_vertices, features, deformer_net)
            losses['ict_identity'] = ict_identity_regularization(ict_facekit)

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
                    losses_to_log[k] = torch.stack([l[k] for l in acc_losses]).mean()
                losses_to_log["total_loss"] = acc_total_loss / len(acc_losses)
                wandb.log({k: v.item() for k, v in losses_to_log.items()}, step=iteration)

                acc_losses = []
                acc_total_loss = 0

            if iteration % 100 == 0:
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
        
            optimizer_deformer.zero_grad()
            optimizer_encoder.zero_grad()
            optimizer_ict_identity.zero_grad()

            loss.backward()
            torch.cuda.synchronize()

            ### increase the gradients of positional encoding following tinycudnn
            if args.grad_scale and args.fourier_features == "hashgrid":
                shader.fourier_feature_transform.params.grad /= 8.0

            # clip gradients
            torch.nn.utils.clip_grad_norm_(shader.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(deformer_net.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_([ict_facekit.identity], 5.0)


            optimizer_shader.step()
        
            optimizer_deformer.step()
            optimizer_encoder.step()
            optimizer_ict_identity.step()

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
                    debug_rgb_pred, debug_gbuffer, debug_cbuffers = run(args, mesh, debug_views, ict_facekit, deformer_net, encoder, shader, renderer, device, channels_gbuffer, lgt)
                    ## ============== visualize ==============================
                    visualize_training(debug_rgb_pred, debug_cbuffers, debug_gbuffer, debug_views, images_save_path, iteration)
                    del debug_gbuffer, debug_cbuffers
            if iteration % (args.visualization_frequency * 10) == 0:
                wandb.log({"Grid": [wandb.Image(images_save_path / "grid" / f'grid_{iteration}.png')]}, step=iteration)

            ## ============== save intermediate ==============================
            if (args.save_frequency > 0) and (iteration == 1 or iteration % args.save_frequency == 0):
                with torch.no_grad():
                    # write_mesh(meshes_save_path / f"mesh_{iteration:06d}.obj", mesh.detach().to('cpu'))                                
                    shader.save(shaders_save_path / f'shader_{iteration:06d}.pt')
                    deformer_net.save(shaders_save_path / f'deformer_{iteration:06d}.pt')
                    encoder.save(shaders_save_path / f'encoder_{iteration:06d}.pt')
                    torch.save(ict_facekit.identity, shaders_save_path / f'ict_identity_{iteration:06d}.pt')

    end = time.time()
    total_time = ((end - start) % 3600)
    print("TIME TAKEN (mins):", int(total_time // 60))

    wandb.finish()

    # ==============================================================================================
    # s a v e
    # ==============================================================================================
    with open(experiment_dir / "args.txt", "w") as text_file:
        print(f"{args}", file=text_file)
    # write_mesh(meshes_save_path / f"mesh_latest.obj", mesh.detach().to('cpu'))
    shader.save(shaders_save_path / f'shader_latest.pt')
    deformer_net.save(shaders_save_path / f'deformer_latest.pt')
    encoder.save(shaders_save_path / f'encoder_latest.pt')
    torch.save(ict_facekit.identity, shaders_save_path / f'ict_identity_latest.pt')

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

    quantitative_eval(args, mesh, dataloader_validate, ict_facekit, deformer_net, encoder, shader, renderer, device, channels_gbuffer, experiment_dir
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