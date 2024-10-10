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

from arguments import config_parser
import os
from pathlib import Path
import torch
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
from flare.metrics import metrics

# Select the device
device = torch.device('cpu')
devices = 0
if torch.cuda.is_available() and devices >= 0:
    device = torch.device(f'cuda:{devices}')

from flare.utils.ict_model import ICTFaceKitTorch
import open3d as o3d

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from arguments import config_parser

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
    visualize_training, save_shading,
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
from flare.dataset import dataset_util


import imageio 
# ==============================================================================================
# evaluation
# ==============================================================================================    
def run(args, mesh, views, ict_facekit, neural_blendshapes, shader, renderer, device, channels_gbuffer, lgt):
    return_dict = neural_blendshapes(views["img"].to(device), views)
    # print(return_dict['features'][:, 53:])

    deformed_vertices = return_dict['expression_mesh_posed']
    
    d_normals = mesh.fetch_all_normals(deformed_vertices, mesh)
    ## ============== Rasterize ==============================
    gbuffer = renderer.render_batch(views["flame_camera"], deformed_vertices.contiguous(), d_normals,
                        channels=channels_gbuffer, with_antialiasing=True, 
                        canonical_v=mesh.vertices, canonical_idx=mesh.indices, canonical_uv = ict_facekit.uv_neutral_mesh)
    
    ## ============== predict color ==============================
    rgb_pred, cbuffers, gbuffer_mask = shader.shade(gbuffer, views, mesh, args.finetune_color, lgt)

    return rgb_pred, gbuffer, cbuffers


# ==============================================================================================
# evaluation: numbers
# ==============================================================================================  
@torch.no_grad()
def run_transfer(args, mesh, dataloader_validate, ict_facekit, neural_blendshapes, shader, renderer, device, channels_gbuffer,
                        experiment_dir, images_eval_save_path, lgt=None, save_each=False):
    import tqdm

    transfer_save_dir = Path(images_save_path / args.transfer_out_name)
    transfer_save_dir.mkdir(parents=True, exist_ok=True)

    Path(transfer_save_dir / "rgb").mkdir(parents=True, exist_ok=True)
    Path(transfer_save_dir / "normal").mkdir(parents=True, exist_ok=True)
    
    for it, views in tqdm.tqdm(enumerate(dataloader_validate)):



        return_dict = neural_blendshapes(views["img"].to(device), views)
        # print(return_dict['features'][:, 53:])

        blendshape = views['mp_blendshape'][..., ict_facekit.mediapipe_to_ict].reshape(-1, 53).detach()

        transform_matrix = views['mp_transform_matrix'].reshape(-1, 4, 4).detach()
        scale = torch.norm(transform_matrix[:, :3, :3], dim=-1).mean(dim=-1, keepdim=True)
        translation = transform_matrix[:, :3, 3]
        rotation_matrix = transform_matrix[:, :3, :3]
        rotation_matrix = transform_matrix[:, :3, :3] / scale[:, None]

        rotation_matrix = rotation_matrix.permute(0, 2, 1)
        rotation = p3dt.matrix_to_euler_angles(rotation_matrix, convention='XYZ')
            
        translation[:, -1] += 28
        translation *= 0

        translation[:, -1] = -1.5
        translation[:, -2] = -0.1

        features = torch.cat([blendshape, rotation, translation, scale, torch.zeros_like(translation)], dim=-1)


        ict_mesh = ict_facekit(expression_weights = features[..., :53], identity_weights = neural_blendshapes.encoder.identity_weights[None].repeat(1, 1))

        features[:, 53:] = return_dict['features'][:, 53:]
        ict_mesh_posed = neural_blendshapes.apply_deformation(ict_mesh, features, torch.ones_like(return_dict['pose_weight']) * 0.9526)


        deformed_vertices = ict_mesh_posed
        d_normals = mesh.fetch_all_normals(deformed_vertices, mesh)
        ## ============== Rasterize ==============================
        gbuffer = renderer.render_batch(views["camera"], deformed_vertices.contiguous(), d_normals,
                            channels=channels_gbuffer, with_antialiasing=True, 
                            canonical_v=mesh.vertices, canonical_idx=mesh.indices, canonical_uv = ict_facekit.uv_neutral_mesh)
        
        ## ============== predict color ==============================
        rgb_pred, cbuffers, gbuffer_mask = shader.shade(gbuffer, views, mesh, args.finetune_color, lgt)

        rgb_pred = rgb_pred * gbuffer["mask"]
        

        normals = gbuffer["normal"]
        gbuffer_mask = gbuffer["mask"]

        convert_uint = lambda x: np.clip(np.rint(dataset_util.rgb_to_srgb(x).numpy() * 255.0), 0, 255).astype(np.uint8) 
        convert_uint_255 = lambda x: (x * 255).to(torch.uint8)

    
        mask = gbuffer_mask[0].cpu()
        id = int(views["frame_name"][0])
        

        # rgb prediction
        imageio.imsave(transfer_save_dir / "rgb" / f'{id:05d}_raw.png', convert_uint(torch.cat([rgb_pred[0].cpu()], -1))) 

        ##normal
        normal = (normals[0] + 1.) / 2.
        normal = torch.cat([normal.cpu(), mask], -1)
        imageio.imsave(transfer_save_dir / "normal" / f'{id:05d}_raw.png', convert_uint_255(normal))

        output_mesh = str(id) +'_raw_shading.png'
        save_shading(rgb_pred, cbuffers, gbuffer, views, (transfer_save_dir / "rgb"), id, ict_facekit=ict_facekit, save_name=output_mesh)






        deformed_vertices = return_dict['ict_mesh_posed']
        d_normals = mesh.fetch_all_normals(deformed_vertices, mesh)
        ## ============== Rasterize ==============================
        gbuffer = renderer.render_batch(views["camera"], deformed_vertices.contiguous(), d_normals,
                            channels=channels_gbuffer, with_antialiasing=True, 
                            canonical_v=mesh.vertices, canonical_idx=mesh.indices, canonical_uv = ict_facekit.uv_neutral_mesh)
        
        ## ============== predict color ==============================
        rgb_pred, cbuffers, gbuffer_mask = shader.shade(gbuffer, views, mesh, args.finetune_color, lgt)

        rgb_pred = rgb_pred * gbuffer["mask"]
        

        normals = gbuffer["normal"]
        gbuffer_mask = gbuffer["mask"]

        convert_uint = lambda x: np.clip(np.rint(dataset_util.rgb_to_srgb(x).numpy() * 255.0), 0, 255).astype(np.uint8) 
        convert_uint_255 = lambda x: (x * 255).to(torch.uint8)

    
        mask = gbuffer_mask[0].cpu()
        id = int(views["frame_name"][0])
        

        # rgb prediction
        imageio.imsave(transfer_save_dir / "rgb" / f'{id:05d}_fc_rgb.png', convert_uint(torch.cat([rgb_pred[0].cpu()], -1))) 

        ##normal
        normal = (normals[0] + 1.) / 2.
        normal = torch.cat([normal.cpu(), mask], -1)
        imageio.imsave(transfer_save_dir / "normal" / f'{id:05d}_fc_n.png', convert_uint_255(normal))




        
        output_mesh = str(id) +'_shading_fc.png'
        save_shading(rgb_pred, cbuffers, gbuffer, views, (transfer_save_dir / "rgb"), id, ict_facekit=ict_facekit, save_name=output_mesh)
            





        convert_uint = lambda x: torch.from_numpy(np.clip(np.rint(dataset_util.rgb_to_srgb(x).detach().cpu().numpy() * 255.0), 0, 255).astype(np.uint8)).to(device)
        org_image = views['img'][0]

        org_image = convert_uint(org_image)
        imageio.imwrite(os.path.join((transfer_save_dir / "rgb"), str(id) +'_org.png'), org_image.cpu().numpy())





        deformed_vertices = return_dict['expression_mesh_posed']
        d_normals = mesh.fetch_all_normals(deformed_vertices, mesh)
        ## ============== Rasterize ==============================
        gbuffer = renderer.render_batch(views["camera"], deformed_vertices.contiguous(), d_normals,
                            channels=channels_gbuffer, with_antialiasing=True, 
                            canonical_v=mesh.vertices, canonical_idx=mesh.indices, canonical_uv = ict_facekit.uv_neutral_mesh)
        
        ## ============== predict color ==============================
        rgb_pred, cbuffers, gbuffer_mask = shader.shade(gbuffer, views, mesh, args.finetune_color, lgt)

        rgb_pred = rgb_pred * gbuffer["mask"]
        
        normals = gbuffer["normal"]
        gbuffer_mask = gbuffer["mask"]

        convert_uint = lambda x: np.clip(np.rint(dataset_util.rgb_to_srgb(x).numpy() * 255.0), 0, 255).astype(np.uint8) 
        convert_uint_255 = lambda x: (x * 255).to(torch.uint8)

    
        mask = gbuffer_mask[0].cpu()
        id = int(views["frame_name"][0])
        

        # rgb prediction
        imageio.imsave(transfer_save_dir / "rgb" / f'{id:05d}.png', convert_uint(torch.cat([rgb_pred[0].cpu()], -1))) 

        ##normal
        normal = (normals[0] + 1.) / 2.
        normal = torch.cat([normal.cpu(), mask], -1)
        imageio.imsave(transfer_save_dir / "normal" / f'{id:05d}.png', convert_uint_255(normal))



        output_mesh = str(id) +'.png'
        save_shading(rgb_pred, cbuffers, gbuffer, views, (transfer_save_dir / "rgb"), id, ict_facekit=ict_facekit, save_name=output_mesh)
            
        convert_uint = lambda x: torch.from_numpy(np.clip(np.rint(dataset_util.rgb_to_srgb(x).detach().cpu().numpy() * 255.0), 0, 255).astype(np.uint8)).to(device)
        org_image = views['img'][0]

        org_image = convert_uint(org_image)
        imageio.imwrite(os.path.join((transfer_save_dir / "rgb"), str(id) +'_org.png'), org_image.cpu().numpy())




            
if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    args.batch_size = 1
    # Select the device
    device = torch.device('cpu')
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    print(f"Using device {device}")

    ## ============== Dir ==============================
    run_name = args.run_name if args.run_name is not None else args.input_dir.parent.name
    images_save_path, images_eval_save_path, meshes_save_path, shaders_save_path, experiment_dir = make_dirs(args, run_name, args.finetune_color)

    dataset_validate    = DatasetLoader(args, train_dir=args.eval_dir, sample_ratio=25, pre_load=False, train=False,)
    dataloader_validate    = torch.utils.data.DataLoader(dataset_validate, batch_size=args.batch_size, collate_fn=dataset_validate.collate, drop_last=False, shuffle=False)

    ## =================== Load FLAME ==============================
    flame_path = args.working_dir / 'flame/FLAME2020/generic_model.pkl'
    flame_shape = dataset_validate.shape_params
    FLAMEServer = FLAME(flame_path, n_shape=100, n_exp=50, shape_params=flame_shape).to(device)

    ## ============== canonical with mouth open (jaw pose 0.4) ==============================
    FLAMEServer.canonical_exp = (dataset_validate.get_mean_expression()).to(device)
    FLAMEServer.canonical_pose = FLAMEServer.canonical_pose.to(device)
    FLAMEServer.canonical_verts, FLAMEServer.canonical_pose_feature, FLAMEServer.canonical_transformations = \
        FLAMEServer(expression_params=FLAMEServer.canonical_exp, full_pose=FLAMEServer.canonical_pose)
    if args.ghostbone:
        FLAMEServer.canonical_transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).float().to(device), FLAMEServer.canonical_transformations], 1)
    FLAMEServer.canonical_verts = FLAMEServer.canonical_verts.to(device)


    ## ============== load ict facekit ==============================
    ict_facekit = ICTFaceKitTorch(npy_dir = './assets/ict_facekit_torch.npy', canonical = Path(args.input_dir) / 'ict_identity.npy')
    ict_facekit = ict_facekit.to(device)

    ict_canonical_mesh = Mesh(ict_facekit.canonical[0].cpu().data, ict_facekit.faces.cpu().data, ict_facekit=ict_facekit, device=device)
    ict_canonical_mesh.compute_connectivity()

    ## ============== renderer ==============================
    aabb = AABB(ict_canonical_mesh.vertices.cpu().numpy())
    ict_mesh_aabb = [torch.min(ict_canonical_mesh.vertices, dim=0).values, torch.max(ict_canonical_mesh.vertices, dim=0).values]

    renderer = Renderer(device=device)
    renderer.set_near_far(dataset_validate, torch.from_numpy(aabb.corners).to(device), epsilon=0.5)
    channels_gbuffer = ['mask', 'position', 'normal', "canonical_position"]
    print("Rasterizing:", channels_gbuffer)
    
    renderer_visualization = Renderer(device=device)
    renderer_visualization.set_near_far(dataset_validate, torch.from_numpy(aabb.corners).to(device), epsilon=0.5)

    # ==============================================================================================
    # deformation 
    # ==============================================================================================

    model_path = Path(experiment_dir / "stage_1" / "network_weights" / f"neural_blendshapes.pt")
    neural_blendshapes = get_neural_blendshapes(model_path=model_path, train=args.train_deformer, vertex_parts=ict_facekit.vertex_parts, ict_facekit=ict_facekit, exp_dir = experiment_dir, lambda_=args.lambda_, aabb = ict_mesh_aabb, device=device) 
    print(ict_canonical_mesh.vertices.shape, ict_canonical_mesh.vertices.device)
    neural_blendshapes = neural_blendshapes.to(device)

    # target_model_path = Path(args.target_model_dir)
    # target_neural_blendshapes = get_neural_blendshapes(model_path=target_model_path, train=args.train_deformer, vertex_parts=ict_facekit.vertex_parts, ict_facekit=ict_facekit, exp_dir = experiment_dir, lambda_=args.lambda_, aabb = ict_mesh_aabb, device=device)
    # target_neural_blendshapes = target_neural_blendshapes.to(device)

    # # copy paramters of target_neural_blendshapes.encoder to neural_blendshapes.encoder
    # neural_blendshapes.encoder.load_state_dict(target_neural_blendshapes.encoder.state_dict())


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

    load_shader = Path(experiment_dir / "stage_1" / "network_weights" / f"shader.pt")

    assert os.path.exists(load_shader)

    shader = NeuralShader.load(load_shader, device=device)


    print("=="*50)
    shader.eval()
    neural_blendshapes.eval()

    batch_size = args.batch_size
    print("Batch Size:", batch_size)
    
    mesh = ict_canonical_mesh.with_vertices(ict_canonical_mesh.vertices)

    with torch.no_grad():
        run_transfer(args, mesh, dataloader_validate, ict_facekit, neural_blendshapes, shader, renderer, device, channels_gbuffer, experiment_dir, images_save_path, lgt=lgt, save_each=True)
