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
    visualize_training,
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
# evaluation: numbers
# ==============================================================================================  
@torch.no_grad()
def run_video(args, mesh, dataloader_validate, ict_facekit, neural_blendshapes, shader, renderer, device, channels_gbuffer,
                        experiment_dir, images_eval_save_path, lgt=None, save_each=False):
    import tqdm

    transfer_save_dir = Path(images_save_path / 'interpolation')
    transfer_save_dir.mkdir(parents=True, exist_ok=True)

    Path(transfer_save_dir / "rgb").mkdir(parents=True, exist_ok=True)
    Path(transfer_save_dir / "normal").mkdir(parents=True, exist_ok=True)
    
    exps_facs = {}
    exps_facs['happiness'] = ['cheekSquint_L', 'cheekSquint_R', 'mouthSmile_L', 'mouthSmile_R']
    exps_facs['sadness'] = ['browInnerUp_L', 'browInnerUp_R', 'browDown_L', 'browDown_R', 'mouthFrown_L', 'mouthFrown_R']
    exps_facs['surprise'] = ['browInnerUp_L', 'browInnerUp_R', 'browOuterUp_L', 'browOuterUp_R', 'eyeWide_L', 'eyeWide_R', 'jawOpen']
    exps_facs['disgust'] = ['noseSneer_L', 'noseSneer_R', 'mouthFrown_L', 'mouthFrown_R', 'mouthLowerDown_L', 'mouthLowerDown_R']

    facs_codes = {}
    for exp in exps_facs:
        facs_codes[exp] = torch.zeros(1, 53)
        for fac in exps_facs[exp]:
            facs_codes[exp][0, ict_facekit.expression_names.tolist().index(fac)] = 0.7

    pose = torch.zeros(1, 10)
    pose[0, 6] = 1 # scale

    
    for view in dataloader_validate:
        fixed_view = view
        break

    # zero - one expression - zero - to another expression .. make sequence
    facs_sequence = []
    num_interp = 15 # 0.5 sec
    for exp in exps_facs:
        for i in range(num_interp):
            facs_sequence.append(facs_codes[exp] * (i/(num_interp-1)))
        for i in range(num_interp):
            facs_sequence.append(facs_codes[exp] * (1 - i/(num_interp-1)))

    for i, facs in enumerate(facs_sequence):
        return_dict = neural_blendshapes(features = torch.cat([facs[None].to(neural_blendshapes.device), pose.to(neural_blendshapes.device)], dim=-1))

        gbuffer = renderer.render_batch(fixed_view['flame_camera'], return_dict['expression_mesh_posed'].contiguous(), mesh.fetch_all_normals(return_dict['ict_mesh_w_temp_posed'], mesh), 
                                channels=channels_gbuffer+['segmentation'], with_antialiasing=True, 
                                canonical_v=mesh.vertices, canonical_idx=mesh.indices, canonical_uv=ict_facekit.uv_neutral_mesh) 
        rgb_pred, cbuffers, _ = shader.shade(gbuffer, fixed_view, mesh, args.finetune_color, lgt)
        
        rgb_pred = rgb_pred * gbuffer["mask"]
        if save_each:
            normals = gbuffer["normal"]
            gbuffer_mask = gbuffer["mask"]

            convert_uint = lambda x: np.clip(np.rint(dataset_util.rgb_to_srgb(x).numpy() * 255.0), 0, 255).astype(np.uint8) 
            convert_uint_255 = lambda x: (x * 255).to(torch.uint8)

            
            for i in range(len(gbuffer_mask)):
                mask = gbuffer_mask[i].cpu()
                id = int(i)

                # rgb prediction
                imageio.imsave(transfer_save_dir / "rgb" / f'{id:05d}.png', convert_uint(torch.cat([rgb_pred[i].cpu(), mask], -1))) 

                ##normal
                normal = (normals[i] + 1.) / 2.
                normal = torch.cat([normal.cpu(), mask], -1)
                imageio.imsave(transfer_save_dir / "normal" / f'{id:05d}.png', convert_uint_255(normal))
    




            
if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()

    # Select the device
    device = torch.device('cpu')
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    print(f"Using device {device}")

    ## ============== Dir ==============================
    run_name = args.run_name if args.run_name is not None else args.input_dir.parent.name
    images_save_path, images_eval_save_path, meshes_save_path, shaders_save_path, experiment_dir = make_dirs(args, run_name, args.finetune_color)

    dataset_validate    = DatasetLoader(args, train_dir=args.train_dir, sample_ratio=args.sample_idx_ratio, pre_load=False, train=False)
    dataloader_validate    = torch.utils.data.DataLoader(dataset_validate, batch_size=args.batch_size, collate_fn=dataset_validate.collate, drop_last=False, )

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

    target_model_path = Path(args.target_model_dir)
    target_neural_blendshapes = get_neural_blendshapes(model_path=target_model_path, train=args.train_deformer, vertex_parts=ict_facekit.vertex_parts, ict_facekit=ict_facekit, exp_dir = experiment_dir, lambda_=args.lambda_, aabb = ict_mesh_aabb, device=device)
    target_neural_blendshapes = target_neural_blendshapes.to(device)

    # copy paramters of target_neural_blendshapes.encoder to neural_blendshapes.encoder
    neural_blendshapes.encoder.load_state_dict(target_neural_blendshapes.encoder.state_dict())


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
        run_video(args, mesh, dataloader_validate, ict_facekit, neural_blendshapes, shader, renderer, device, channels_gbuffer, experiment_dir, images_save_path, lgt=lgt, save_each=True)
