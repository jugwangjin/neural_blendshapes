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
from pathlib import Path
import torch
# Select the device
device = torch.device('cpu')
devices = 0
if torch.cuda.is_available() and devices >= 0:
    device = torch.device(f'cuda:{devices}')

import open3d as o3d

import sys

import time

from flare.metrics import metrics

# ==============================================================================================
# evaluation
# ==============================================================================================    
def run(args, mesh, views, ict_facekit, neural_blendshapes, shader, renderer, device, channels_gbuffer, lgt):

    neural_blendshapes.eval()
    shader.eval()

    return_dict = neural_blendshapes(views["img"].to(device), views)
    # print(return_dict['features'][:, 53:])

    deformed_vertices = return_dict['expression_mesh_posed']
    
    d_normals = mesh.fetch_all_normals(deformed_vertices, mesh)
    ## ============== Rasterize ==============================
    mesh_key_name = 'expression_mesh'
    gbuffers = renderer.render_batch(views['flame_camera'], return_dict[mesh_key_name+'_posed'].contiguous(), mesh.fetch_all_normals(return_dict[mesh_key_name+'_posed'], mesh),
                            channels=channels_gbuffer + ['segmentation'], with_antialiasing=True, 
                            canonical_v=mesh.vertices, canonical_idx=mesh.indices, canonical_uv=ict_facekit.uv_neutral_mesh,
                            mesh=mesh
                            )

    
    ## ============== predict color ==============================
    rgb_pred, cbuffers, gbuffer_mask = shader.shade(gbuffers, views, mesh, args.finetune_color, lgt)

    return rgb_pred, gbuffers, cbuffers


# ==============================================================================================
# relight: run
# ==============================================================================================  
def run_relight(args, mesh, views, ict_facekit, neural_blendshapes, shader, renderer, device, channels_gbuffer, lgt_list, images_save_path):
    return_dict = neural_blendshapes(views["img"].to(device), views)

    deformed_vertices = return_dict['expression_mesh_posed']

    d_normals = mesh.fetch_all_normals(deformed_vertices, mesh)
    ## ============== Rasterize ==============================
    gbuffers = renderer.render_batch(views["camera"], deformed_vertices.contiguous(), d_normals,
                        channels=channels_gbuffer, with_antialiasing=True, 
                        canonical_v=mesh.vertices, canonical_idx=mesh.indices, canonical_uv = ict_facekit.uv_neutral_mesh)
    
    ## ============== predict color ==============================
    relit_imgs, cbuffers, gbuffer_mask = shader.relight(gbuffers, views, mesh, args.finetune_color, lgt_list)
    save_relit_intrinsic_materials(relit_imgs, views, gbuffer_mask, cbuffers, images_save_path)

# ==============================================================================================
# evaluation: numbers
# ==============================================================================================  
def quantitative_eval(args, mesh, dataloader_validate, ict_facekit, neural_blendshapes, shader, renderer, device, channels_gbuffer,
                        experiment_dir, images_eval_save_path, lgt=None, save_each=False):
    import tqdm
    for it, views in tqdm.tqdm(enumerate(dataloader_validate)):
        with torch.no_grad():
            
            neural_blendshapes.eval()
            shader.eval()

            rgb_pred, gbuffer, cbuffer = run(args, mesh, views, ict_facekit, neural_blendshapes, shader, renderer, device, 
                    channels_gbuffer, lgt=lgt)

        rgb_pred = rgb_pred * gbuffer["mask"]
        if save_each:
            save_individual_img(rgb_pred, views, gbuffer["normal"], gbuffer["mask"], cbuffer, images_eval_save_path)

    ## ============== metrics ==============================
    gt_dir = Path(args.input_dir)
    if gt_dir is not None:
        eval_list = metrics.run(images_eval_save_path, gt_dir, args.eval_dir)

    with open(str(experiment_dir / "final_eval.txt"), 'a') as f:
        f.writelines("\n"+"w/o cloth result:"+"\n")
        f.writelines("\n"+"MAE | LPIPS | SSIM | PSNR"+"\n")
        if gt_dir is not None:
            eval_list = [str(e) for e in eval_list]
            f.writelines(" ".join(eval_list))
            
# ==============================================================================================
# evaluation: numbers
# ==============================================================================================  
def measure_fps(args, mesh, dataloader_validate, ict_facekit, neural_blendshapes, shader, renderer, device, channels_gbuffer,
                        experiment_dir, images_eval_save_path, lgt=None, save_each=False):
    import tqdm

    running_times = []
    encoder_times = []
    blendshapes_times = []
    deform_times = []
    rendering_times = []


    for it, views in tqdm.tqdm(enumerate(dataloader_validate)):
        with torch.no_grad():
            
            neural_blendshapes.eval()
            shader.eval()

            start = time.time()

            return_dict = neural_blendshapes.forward_measure_time(views["img"].to(device), views)
            # print(return_dict['features'][:, 53:])

            deformed_vertices = return_dict['expression_mesh_posed']

            encoder_time = return_dict['encoder_time']

            blendshapes_time = return_dict['blendshapes_time'] 
            deform_time = return_dict['deform_time'] 
            
            rendering_start = time.time()
            d_normals = mesh.fetch_all_normals(deformed_vertices, mesh)
            ## ============== Rasterize ==============================
            mesh_key_name = 'expression_mesh'

            gbuffers = renderer.render_batch(views['flame_camera'], return_dict[mesh_key_name+'_posed'].contiguous(), d_normals,
                            channels=channels_gbuffer, with_antialiasing=True, 
                            canonical_v=mesh.vertices, canonical_idx=mesh.indices, canonical_uv=ict_facekit.uv_neutral_mesh,
                            mesh=mesh
                            )

    
            ## ============== predict color ==============================
            rgb_pred, cbuffers, gbuffer_mask = shader.shade(gbuffers, views, mesh, args.finetune_color, lgt)

            rendering_end = time.time()

            end = time.time()

            running_times.append(end-start)

            rendering_times.append(rendering_end-rendering_start)
            encoder_times.append(encoder_time)
            blendshapes_times.append(blendshapes_time)
            deform_times.append(deform_time)

            print(
                f"Runtime: {end-start}, Encoder time: {encoder_time}, Blendshapes time: {blendshapes_time}, Deform time: {deform_time}, Rendering time: {rendering_end-rendering_start}"
            )


    mean_runtime = sum(running_times) / len(running_times)
    mean_encoder_time = sum(encoder_times) / len(encoder_times)
    mean_blendshapes_time = sum(blendshapes_times) / len(blendshapes_times)
    mean_deform_time = sum(deform_times) / len(deform_times)
    mean_rendering_time = sum(rendering_times) / len(rendering_times)

    print(f"Mean runtime: {mean_runtime}, fps: {1/mean_runtime}")
    print(f"Mean encoder time: {mean_encoder_time}")
    print(f"Mean blendshapes time: {mean_blendshapes_time}")
    print(f"Mean deform time: {mean_deform_time}")
    print(f"Mean rendering time: {mean_rendering_time}")


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()


    # Select the device
    device = torch.device('cpu')
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    print(f"Using device {device}")


    original_dir = os.getcwd()
    # Add the path to the 'flare' directory
    flare_path = os.path.join(args.output_dir, args.run_name, 'sources')
    
    sys.path.insert(0, flare_path)


    from flame.FLAME import FLAME
    from flare.core import (
        Mesh, Renderer
    )
    from flare.modules import (
        NeuralShader, get_neural_blendshapes
    )
    from flare.utils import (
        AABB, read_mesh,
        save_individual_img, make_dirs, save_relit_intrinsic_materials
    )
    import nvdiffrec.render.light as light
    from flare.dataset import DatasetLoader
    from flare.dataset import dataset_util
    from flare.utils.ict_model import ICTFaceKitTorch


    from flare.dataset import DatasetLoader

    from flare.utils.ict_model import ICTFaceKitTorch
    import nvdiffrec.render.light as light
    from flare.core import (
        Mesh, Renderer
    )
    from flare.modules import (
        NeuralShader, get_neural_blendshapes
    )
    from flare.utils import (
        AABB, 
        save_manipulation_image
    )
    ## ============== load ict facekit ==============================
    dataset_train    = DatasetLoader(args, train_dir=args.train_dir, sample_ratio=args.sample_idx_ratio, pre_load=False, train=True)


    ict_facekit = ICTFaceKitTorch(npy_dir = './assets/ict_facekit_torch.npy', canonical = Path(args.input_dir) / 'ict_identity.npy')
    ict_facekit = ict_facekit.to(device)

    ict_canonical_mesh = Mesh(ict_facekit.canonical[0].cpu().data, ict_facekit.faces.cpu().data, ict_facekit=ict_facekit, device=device)
    ict_canonical_mesh.compute_connectivity()

    mesh = ict_canonical_mesh

    ## ============== renderer ==============================
    aabb = AABB(ict_canonical_mesh.vertices.cpu().numpy())
    ict_mesh_aabb = [torch.min(ict_canonical_mesh.vertices, dim=0).values, torch.max(ict_canonical_mesh.vertices, dim=0).values]

    renderer = Renderer(device=device)
    renderer.set_near_far(dataset_train, torch.from_numpy(aabb.corners).to(device), epsilon=0.5)
    channels_gbuffer = ['mask', 'position', 'normal', "canonical_position"]
    print("Rasterizing:", channels_gbuffer)
    
    renderer_visualization = Renderer(device=device)
    renderer_visualization.set_near_far(dataset_train, torch.from_numpy(aabb.corners).to(device), epsilon=0.5)

    shader = NeuralShader.load(os.path.join(args.output_dir, args.run_name, 'stage_1', 'network_weights', 'shader.pt'), device=device)
    # ==============================================================================================
    # deformation 
    # ==============================================================================================

 # neural blendshapes
    model_path = os.path.join(args.output_dir, args.run_name, 'stage_1', 'network_weights', 'neural_blendshapes.pt')
    print("=="*50)
    print("Training Deformer")
    face_normals = ict_canonical_mesh.get_vertices_face_normals(ict_facekit.neutral_mesh_canonical[0])[0]
    neural_blendshapes = get_neural_blendshapes(model_path=model_path, train=args.train_deformer, ict_facekit=ict_facekit, aabb = ict_mesh_aabb, face_normals=face_normals,device=device) 
    


    # Create directories
    run_name = args.run_name if args.run_name is not None else args.input_dir.parent.name
    images_save_path, images_eval_save_path, meshes_save_path, shaders_save_path, experiment_dir = make_dirs(args, run_name, args.finetune_color)
    


    lgt = light.create_env_rnd()    


    dataset_val      = DatasetLoader(args, train_dir=args.eval_dir, sample_ratio=100, pre_load=False)

    dataloader_validate = torch.utils.data.DataLoader(dataset_val, batch_size=1, collate_fn=dataset_val.collate)

    measure_fps(args, mesh, dataloader_validate, ict_facekit, neural_blendshapes, shader, renderer, device, channels_gbuffer, experiment_dir
                    , images_eval_save_path / "qualitative_results", lgt=lgt, save_each=True)



    dataset_val      = DatasetLoader(args, train_dir=args.eval_dir, sample_ratio=1, pre_load=False)

    dataloader_validate = torch.utils.data.DataLoader(dataset_val, batch_size=4, collate_fn=dataset_val.collate)

    quantitative_eval(args, mesh, dataloader_validate, ict_facekit, neural_blendshapes, shader, renderer, device, channels_gbuffer, experiment_dir
                    , images_eval_save_path / "qualitative_results", lgt=lgt, save_each=True)


