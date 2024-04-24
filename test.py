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
from flame.FLAME import FLAME
from flare.core import (
    Mesh, Renderer
)
from flare.modules import (
    NeuralShader, get_mlp_deformer, ResnetEncoder
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

def load_ict_facekit(args, device):
    ict_facekit = ICTFaceKitTorch(npy_dir = './assets/ict_facekit_torch.npy', canonical = Path(args.input_dir) / 'ict_identity.npy')
    ict_facekit = ict_facekit.to(device)
    ict_facekit.eval()

    ict_canonical_mesh = Mesh(ict_facekit.canonical[0].cpu().data, ict_facekit.faces.cpu().data, device=device)
    
    ict_canonical_mesh.compute_connectivity()
    ict_facekit.update_vmapping(ict_canonical_mesh.vmapping.cpu().data.numpy())

    return ict_facekit, ict_canonical_mesh
# ==============================================================================================
# evaluation
# ==============================================================================================    
def run(args, mesh, views, ict_facekit, deformer_net, encoder, shader, renderer, device, channels_gbuffer, lgt):

    encoder_input = views["img"].permute(0, 3, 1, 2).to(device)
    features = encoder(encoder_input.to(device))
    # print(features[0, :53].cpu().data.numpy())

    deformed = deformer_net(features, vertices=None)

    deformed_vertices = mesh.vertices[None].repeat(views["img"].shape[0], 1, 1)
    deformed_vertices[:, ict_facekit.head_indices] = deformed["deformed_head"]
    deformed_vertices[:, ict_facekit.eyeball_indices] = deformed["deformed_eyes"]
    
    d_normals = mesh.fetch_all_normals(deformed_vertices, mesh)
    ## ============== Rasterize ==============================
    gbuffers = renderer.render_batch(views["camera"], deformed_vertices.contiguous(), d_normals,
                        channels=channels_gbuffer, with_antialiasing=True, 
                        canonical_v=mesh.vertices, canonical_idx=mesh.indices)
    
    ## ============== predict color ==============================
    rgb_pred, cbuffers, gbuffer_mask = shader.shade(gbuffers, views, mesh, args.finetune_color, lgt)

    return rgb_pred, gbuffers, cbuffers

# ==============================================================================================
# relight: run
# ==============================================================================================  
def run_relight(args, mesh, views, ict_facekit, deformer_net, encoder, shader, renderer, device, channels_gbuffer, lgt_list, images_save_path):

    encoder_input = views["img"].permute(0, 3, 1, 2).to(device)
    features = encoder(encoder_input.to(device))

    deformed = deformer_net(features, vertices=None)

    deformed_vertices = mesh.vertices[None].repeat(views["img"].shape[0], 1, 1)
    deformed_vertices[:, ict_facekit.head_indices] = deformed["deformed_head"]
    deformed_vertices[:, ict_facekit.eyeball_indices] = deformed["deformed_eyes"]
    d_normals = mesh.fetch_all_normals(deformed_vertices, mesh)
    ## ============== Rasterize ==============================
    gbuffers = renderer.render_batch(views["camera"], deformed_vertices.contiguous(), d_normals,
                        channels=channels_gbuffer, with_antialiasing=True, 
                        canonical_v=mesh.vertices, canonical_idx=mesh.indices)
    
    ## ============== predict color ==============================
    relit_imgs, cbuffers, gbuffer_mask = shader.relight(gbuffers, views, mesh, args.finetune_color, lgt_list)
    save_relit_intrinsic_materials(relit_imgs, views, gbuffer_mask, cbuffers, images_save_path)

# ==============================================================================================
# evaluation: numbers
# ==============================================================================================  
def quantitative_eval(args, mesh, dataloader_validate, ict_facekit, deformer_net, encoder, shader, renderer, device, channels_gbuffer,
                        experiment_dir, images_eval_save_path, lgt=None, save_each=False):

    for it, views in enumerate(dataloader_validate):
        with torch.no_grad():
            rgb_pred, gbuffer, cbuffer = run(args, mesh, views, ict_facekit, deformer_net, encoder, shader, renderer, device, 
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
            
if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()

    # Select the device
    device = torch.device('cpu')
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    print(f"Using device {device}")

    # Create directories
    run_name = args.run_name if args.run_name is not None else args.input_dir.parent.name
    images_save_path, images_eval_save_path, meshes_save_path, shaders_save_path, experiment_dir = make_dirs(args, run_name, args.finetune_color)
    flame_path = args.working_dir / 'flame/FLAME2020/generic_model.pkl'

    # ==============================================================================================
    # Create evalables: FLAME + Renderer + Views + Downsample
    # ==============================================================================================

    ### Read the views
    print("loading test views...")
    dataset_val      = DatasetLoader(args, train_dir=args.eval_dir, sample_ratio=args.sample_idx_ratio, pre_load=True)
    dataloader_validate = torch.utils.data.DataLoader(dataset_val, batch_size=4, collate_fn=dataset_val.collate, shuffle=False)

    ict_facekit, ict_canonical_mesh = load_ict_facekit(args, device)    

    ## ============== renderer ==============================
    aabb = AABB(ict_canonical_mesh.vertices.cpu().numpy())
    ict_mesh_aabb = [torch.min(ict_canonical_mesh.vertices, dim=0).values, torch.max(ict_canonical_mesh.vertices, dim=0).values]


    renderer = Renderer(device=device)
    renderer.set_near_far(dataset_val, torch.from_numpy(aabb.corners).to(device), epsilon=0.5)

    channels_gbuffer = ['mask', 'position', 'normal', "canonical_position"]
    print("Rasterizing:", channels_gbuffer)


    feature_dim = args.feature_dim
    head_deformer_layers = args.head_deformer_layers
    head_deformer_hidden_dim = args.head_deformer_hidden_dim
    head_deformer_multires = args.head_deformer_multires
    eye_deformer_layers = args.eye_deformer_layers
    eye_deformer_hidden_dim = args.eye_deformer_hidden_dim
    eye_deformer_multires = args.eye_deformer_multires

    model_path = Path(experiment_dir / "stage_2" / "network_weights" / f"encoder_latest.pt")
    encoder = ResnetEncoder(outsize=feature_dim)
    encoder.load_state_dict(torch.load(str(model_path)))

    model_path = Path(experiment_dir / "stage_2" / "network_weights" / f"deformer_latest.pt")
    deformer_net = get_mlp_deformer(input_feature_dim=feature_dim, head_deformer_layers=head_deformer_layers, 
                                        head_deformer_hidden_dim=head_deformer_hidden_dim, head_deformer_multires=head_deformer_multires,
                                                eye_deformer_layers=eye_deformer_layers, eye_deformer_hidden_dim=eye_deformer_hidden_dim,
                                                  eye_deformer_multires=eye_deformer_multires, 
                                                  model_path=model_path, train=args.train_deformer, device=device) 

    head_template = ict_canonical_mesh.vertices[ict_facekit.head_indices].to(device)
    eye_template = ict_canonical_mesh.vertices[ict_facekit.eyeball_indices].to(device)

    deformer_net.set_template((head_template, eye_template))

    load_shader = Path(experiment_dir / "stage_2" / "network_weights" / f"shader_latest.pt")
    assert os.path.exists(load_shader)

    shader = NeuralShader.load(load_shader, device=device)

    lgt = light.create_env_rnd()    

    print("=="*50)
    shader.eval()
    deformer_net.eval()

    batch_size = views["img"].shape[0]
    print("Batch Size:", batch_size)
    
    mesh = ict_canonical_mesh.with_vertices(ict_canonical_mesh.vertices)
    # ==============================================================================================
    # evaluation: intrinsic materials and relighting
    # ==============================================================================================  
    lgt_list = light.load_target_cubemaps(args.working_dir)
    for i in range(len(lgt_list)):
        Path(images_eval_save_path / "qualitative_results" / f"env_map_{i}" ).mkdir(parents=True, exist_ok=True)

    for it, views in enumerate(dataloader_validate):
        with torch.no_grad():
            run_relight(args, mesh, views, ict_facekit, deformer_net, encoder, shader, renderer, device, channels_gbuffer, lgt_list, images_eval_save_path / "qualitative_results")
            
    # # ==============================================================================================
    # # evaluation: qualitative and quantitative - animation
    # # ==============================================================================================  
    quantitative_eval(args, mesh, dataloader_validate, ict_facekit, deformer_net, encoder, shader, renderer, device, channels_gbuffer, experiment_dir
                    , images_eval_save_path  / "qualitative_results", lgt=lgt, save_each=True)