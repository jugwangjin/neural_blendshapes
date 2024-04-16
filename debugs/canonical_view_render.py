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
    NeuralShader, get_deformer_network, Displacement
)
from flare.utils import (
    AABB, read_mesh, write_mesh,
    visualize_training,
    make_dirs, set_defaults_finetune
)
import nvdiffrec.render.light as light
from test import run, quantitative_eval

import time

def main(args, device, dataset_val, debug_views, FLAMEServer):
    ## ============== Dir ==============================
    run_name = args.run_name if args.run_name is not None else args.input_dir.parent.name
    
    args.working_dir = Path('.')
    args.output_dir = Path('debug')
    run_name = Path('debug')
    images_save_path, images_eval_save_path, meshes_save_path, shaders_save_path, experiment_dir = make_dirs(args, run_name, args.finetune_color)

    verts = FLAMEServer.canonical_verts.squeeze(0)
    faces = FLAMEServer.faces_tensor

    flame_canonical_mesh: Mesh = None
    flame_canonical_mesh = Mesh(verts, faces, device=device)
    flame_canonical_mesh.compute_connectivity()
    write_mesh(Path(meshes_save_path / "init_mesh.obj"), flame_canonical_mesh.to('cpu'))

    ## ============== renderer ==============================
    aabb = AABB(flame_canonical_mesh.vertices.cpu().numpy())
    flame_mesh_aabb = [torch.min(flame_canonical_mesh.vertices, dim=0).values, torch.max(flame_canonical_mesh.vertices, dim=0).values]

    renderer = Renderer(device=device)
    renderer.set_near_far(dataset_val, torch.from_numpy(aabb.corners).to(device), epsilon=0.5)
    channels_gbuffer = ['mask', 'position', 'normal', "canonical_position"]
    print("Rasterizing:", channels_gbuffer)
    
    renderer_visualization = Renderer(device=device)
    renderer_visualization.set_near_far(dataset_val, torch.from_numpy(aabb.corners).to(device), epsilon=0.5)

    deformer_net = get_deformer_network(FLAMEServer, model_path=None, train=args.train_deformer, d_in=3, dims=args.deform_dims, 
                                           weight_norm=True, multires=0, num_exp=50, aabb=flame_mesh_aabb, ghostbone=args.ghostbone, device=device)

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
                          aabb=flame_mesh_aabb,
                          device=device)
    params = list(shader.parameters()) 

    if args.weight_albedo_regularization > 0:
        from robust_loss_pytorch.adaptive import AdaptiveLossFunction
        _adaptive = AdaptiveLossFunction(num_dims=4, float_dtype=np.float32, device=device)
        params += list(_adaptive.parameters()) ## need to train it
        
    mesh = flame_canonical_mesh.with_vertices(flame_canonical_mesh.vertices)
    
    for cam in debug_views['camera']:
        print(cam.R, cam.t, cam.K)

    debug_rgb_pred, debug_gbuffer, debug_cbuffers = debug_canonical(args, mesh, debug_views, FLAMEServer, deformer_net, shader, renderer, device, channels_gbuffer, lgt)
    ## ============== visualize ==============================
    visualize_training(debug_rgb_pred, debug_cbuffers, debug_gbuffer, debug_views, images_save_path, 0)
    del debug_gbuffer, debug_cbuffers

def debug_canonical(args, mesh, views, FLAMEServer, deformer_net, shader, renderer, device, channels_gbuffer, lgt):
    deformed_vertices = mesh.vertices.repeat(views["img"].shape[0], 1, 1)

    d_normals = mesh.fetch_all_normals(deformed_vertices, mesh)
    ## ============== Rasterize ==============================
    gbuffers = renderer.render_batch(views["camera"], deformed_vertices.contiguous(), d_normals,
                        channels=channels_gbuffer, with_antialiasing=True, 
                        canonical_v=mesh.vertices, canonical_idx=mesh.indices)
    
    ## ============== predict color ==============================
    rgb_pred, cbuffers, gbuffer_mask = shader.shade(gbuffers, views, mesh, args.finetune_color, lgt)

    return rgb_pred, gbuffers, cbuffers

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
    dataset_val      = DatasetLoader(args, train_dir=args.eval_dir, sample_ratio=24, pre_load=False)
    view_indices = np.array(args.visualization_views).astype(int)
    d_l = [dataset_val.__getitem__(idx) for idx in view_indices[2:]]
    debug_views = dataset_val.collate(d_l)

    # ==============================================================================================
    # Create trainables: FLAME + Renderer  + Downsample
    # ==============================================================================================
    ### ============== load FLAME mesh ==============================
    flame_path = args.working_dir / 'flame/FLAME2020/generic_model.pkl'
    flame_shape = dataset_val.shape_params
    FLAMEServer = FLAME(flame_path, n_shape=100, n_exp=50, shape_params=flame_shape).to(device)

    ## ============== canonical with mouth open (jaw pose 0.4) ==============================
    FLAMEServer.canonical_exp = (dataset_val.get_mean_expression()).to(device)
    FLAMEServer.canonical_pose = FLAMEServer.canonical_pose.to(device)
    FLAMEServer.canonical_verts, FLAMEServer.canonical_pose_feature, FLAMEServer.canonical_transformations = \
        FLAMEServer(expression_params=FLAMEServer.canonical_exp, full_pose=FLAMEServer.canonical_pose)
    if args.ghostbone:
        FLAMEServer.canonical_transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).float().to(device), FLAMEServer.canonical_transformations], 1)
    FLAMEServer.canonical_verts = FLAMEServer.canonical_verts.to(device)
    
    # ==============================================================================================
    # main run
    # ==============================================================================================
    with torch.no_grad():
        main(args, device, dataset_val, debug_views, FLAMEServer)
