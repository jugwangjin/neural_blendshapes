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
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["GLOG_minloglevel"] ="2"

from arguments import config_parser

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

set_seed(20241224)

from pathlib import Path




import cv2

def sample_uv_grid(size):
    """Sample a regular grid of points in UV space."""
    u = np.linspace(0, 1, size)
    v = np.linspace(0, 1, size)
    uu, vv = np.meshgrid(u, v)
    return np.stack([uu, vv], axis=-1).reshape(-1, 2)

def barycentric_coords(p, a, b, c):
    """Compute barycentric coordinates of point p with respect to triangle (a, b, c)."""
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return u, v, w


def barycentric_coords_batch(p, a, b, c):
    """Compute barycentric coordinates of point p with respect to triangle (a, b, c)."""
    # a, b, c shape of (3)
    # p shape of (N, 3)
    a = a[None] # shape of (1, 3)
    b = b[None] # shape of (1, 3)
    c = c[None] # shape of (1, 3)
    v0 = b - a # shape of (1, 3)
    v1 = c - a # shape of (1, 3)
    v2 = p - a[None] # shape of (N, 3)
    d00 = torch.sum(v0 * v0, dim=-1) # shape of (1)
    d01 = torch.sum(v0 * v1, dim=-1) # shape of (1)
    d11 = torch.sum(v1 * v1, dim=-1) # shape of (1)
    d20 = torch.sum(v2 * v0, dim=-1) # shape of (N)
    d21 = torch.sum(v2 * v1, dim=-1) # shape of (N)
    denom = d00 * d11 - d01 * d01 # shape of (1)
    v = (d11 * d20 - d01 * d21) / denom # shape of (N)
    w = (d00 * d21 - d01 * d20) / denom # shape of (N)
    u = 1.0 - v - w # shape of (N)
    return u, v, w



def point_in_triangle(p, a, b, c):
    """Check if point p is inside triangle (a, b, c)."""
    u, v, w = barycentric_coords(p, a, b, c)
    return (u >= 0) & (v >= 0) & (w >= 0)


def point_in_triangle_batch(p, a, b, c):
    """Check if point p is inside triangle (a, b, c)."""
    u, v, w = barycentric_coords_batch(p, a, b, c) # shape of (N)
    return (u >= 0) & (v >= 0) & (w >= 0)


def create_position_map(mesh, uv_coords, uv_idx, canonical_v, output_path, size=1024):

    # Sample points from UV space
    uv_points = sample_uv_grid(size) # shape of (size * size, 2)
    uv_points = torch.tensor(uv_points, dtype=torch.float32).to(mesh.device)

    # Create a blank UV map
    uv_map = np.zeros((size, size, 3), dtype=np.float32)

    filled = torch.zeros(size, size, dtype=torch.bool).to(mesh.device)

    print(uv_idx.shape, uv_coords.shape, canonical_v.shape, mesh.indices.shape)

    import tqdm
    # Iterate over each face

    filtered_indices = mesh.indices
    # if any of mesh indices greater or equal to 11248
    # filter samely with uv_idx

    filtering_indices = mesh.indices.ge(9409)
    filtering_indices = filtering_indices.any(dim=-1)
    filtered_indices = mesh.indices[~filtering_indices]
    filtered_uv_idx = uv_idx[~filtering_indices]

    print(filtered_indices.shape, filtered_uv_idx.shape)

    pbar = tqdm.tqdm(range(filtered_uv_idx.shape[0]))
    for i in pbar:
        
        face_uv = uv_coords[filtered_uv_idx[i]]
        face_v = canonical_v[filtered_indices[i]]
        
        # print(uv_points.shape, face_uv.shape, face_v.shape)

        u, v, w = barycentric_coords_batch(uv_points, *face_uv)
        u = u.squeeze(); v = v.squeeze(); w = w.squeeze()
        
        # filter out valid points with uvw
        valid_points = u.ge(0) & v.ge(0) & w.ge(0) 
        # print(valid_points.shape)


        valid_uv_points = uv_points[valid_points]

        valid_positions = u[valid_points, None] * face_v[0:1] + v[valid_points, None] * face_v[1:2] + w[valid_points, None] * face_v[2:3]

        x = (valid_uv_points[:, 0] * (size - 1)).long().cpu().numpy()
        y = size - (valid_uv_points[:, 1] * (size - 1)).long().cpu().numpy() - 1

        uv_map[y, x] = valid_positions.cpu().numpy()

        pbar.set_description(f"Processing face {i}, valid points {valid_points.sum().item()}")

    # Save the UV map
    uv_map_ = uv_map.copy()
    uv_map_ = (uv_map_ - uv_map_.min()) / (uv_map_.max() - uv_map_.min())
    uv_map_ = (uv_map_ * 255).astype(np.uint8)
    cv2.imwrite(output_path, uv_map_)

    return uv_map


@torch.no_grad()
def main(args, device):

    original_dir = os.getcwd()
    # Add the path to the 'flare' directory
    flare_path = os.path.join(args.model_dir, args.model_name, 'sources')
    print(flare_path)
    sys.path.insert(0, flare_path)

    from flare.dataset import DatasetLoader

    from flare.modules import (
        get_neural_blendshapes, neuralshader
    )  
    from flare.utils import (
        AABB, 
        save_manipulation_image
    )
    import nvdiffrec.render.light as light
    from flare.core import (
        Mesh, Renderer
    )


    from flare.utils.ict_model import ICTFaceKitTorch
    
    '''
    dataset
    '''
    print("loading views...")

    dataset_train    = DatasetLoader(args, train_dir=args.train_dir, sample_ratio=args.sample_idx_ratio, pre_load=False, train=True)

    '''
    models
    '''

    # ict facekit
    ict_facekit = ICTFaceKitTorch(npy_dir = './assets/ict_facekit_torch.npy', canonical = Path(args.input_dir) / 'ict_identity.npy')
    ict_facekit = ict_facekit.to(device)

    ict_canonical_mesh = Mesh(ict_facekit.canonical[0].cpu().data, ict_facekit.faces.cpu().data, ict_facekit=ict_facekit, device=device)
    ict_canonical_mesh.compute_connectivity()
            
    bshapes_names = ict_facekit.expression_names.tolist()

    # renderer
    aabb = AABB(ict_canonical_mesh.vertices.cpu().numpy())
    ict_mesh_aabb = [torch.min(ict_canonical_mesh.vertices, dim=0).values, torch.max(ict_canonical_mesh.vertices, dim=0).values]

    renderer = Renderer(device=device)
    renderer.set_near_far(dataset_train, torch.from_numpy(aabb.corners).to(device), epsilon=0.5)

    channels_gbuffer = ['mask', 'position', 'normal', "canonical_position"]
    print("Rasterizing:", channels_gbuffer)
    
    renderer_visualization = Renderer(device=device)
    renderer_visualization.set_near_far(dataset_train, torch.from_numpy(aabb.corners).to(device), epsilon=0.5)

    # neural blendshapes
    model_path = os.path.join(args.output_dir, args.run_name, 'stage_1', 'network_weights', 'neural_blendshapes.pt')
    print("=="*50)
    print("Training Deformer")
    face_normals = ict_canonical_mesh.get_vertices_face_normals(ict_facekit.neutral_mesh_canonical[0])[0]
    neural_blendshapes = get_neural_blendshapes(model_path=model_path, train=args.train_deformer, ict_facekit=ict_facekit, aabb = ict_mesh_aabb, face_normals=face_normals,device=device) 
    
    neural_blendshapes = neural_blendshapes.to(device)

    # shader
    lgt = light.create_env_rnd()   
    shader = neuralshader.NeuralShader.load(os.path.join(args.output_dir, args.run_name, 'stage_1', 'network_weights', 'shader.pt'), device=device)
    shader.eval()
    output_dir = './debug/texture_map/1024'
    os.makedirs(output_dir, exist_ok=True)

    position_map = create_position_map(ict_canonical_mesh, ict_facekit.uvs, ict_facekit.uv_faces, ict_facekit.canonical[0], os.path.join(output_dir, f'{args.model_name}.png'))
    # position map shape of (size, size, 3) 
    # with neuralshader.forward, extract kd, kr, ko, all value range [0, 1]
    # kd: diffuse (albedo)
    # kr: roughness 
    # ko: specular
    kd, kr, ko = shader.custom_forward(position_map)

    # save as images
    # 1. Tensor 데이터를 NumPy로 변환
    kd_np = kd.squeeze(0).cpu().numpy()  # [H, W, C] 형식으로 변환
    kr_np = kr.squeeze(0).cpu().numpy()  # [H, W, C] 형식으로 변환
    ko_np = ko.squeeze(0).cpu().numpy()  # [H, W, C] 형식으로 변환

    # 2. Grayscale 데이터를 3채널로 복제
    if kr_np.shape[-1] == 1:  # kr이 1채널이면
        kr_np = np.repeat(kr_np, 3, axis=-1)  # 3채널로 복제

    if ko_np.shape[-1] == 1:  # ko가 1채널이면
        ko_np = np.repeat(ko_np, 3, axis=-1)  # 3채널로 복제

    kd_np = kd_np[:, :, ::-1]  # RGB → BGR
    kr_np = kr_np[:, :, ::-1]  # RGB → BGR
    ko_np = ko_np[:, :, ::-1]  # RGB → BGR

    # 3. 0~255로 스케일링 및 uint8 변환
    kd_np = (kd_np * 255).astype(np.uint8)
    kr_np = (kr_np * 255).astype(np.uint8)
    ko_np = (ko_np * 255).astype(np.uint8)
    print(kd_np.shape)
    print(kr_np.shape)
    print(ko_np.shape)
    cv2.imwrite(os.path.join(output_dir, f"{args.model_name}_kd.png"), kd_np)
    cv2.imwrite(os.path.join(output_dir, f"{args.model_name}_kr.png"), kr_np)
    cv2.imwrite(os.path.join(output_dir, f"{args.model_name}_ko.png"), ko_np)

if __name__ == '__main__':
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='/Bean/log/gwangjin/2024/nbshapes_comparisons/ours_enc_v6', help='Path to the trained model')
    parser.add_argument('--model_name', type=str, default='marcel', help='Name of the run in model_dir')
    args = parser.parse_args()

    config_file = os.path.join(args.model_dir, args.model_name, 'sources', 'config.txt')
    # Override the config file argument

    parser = config_parser()
    args2 = parser.parse_args(['--config', config_file])
    args2.run_name = args.model_name
    args2.output_dir = args.model_dir
    args2.model_dir = args.model_dir
    args2.model_name = args.model_name
    
    # Select the device
    device = torch.device('cpu')
    if torch.cuda.is_available() and args2.device >= 0:
        device = torch.device(f'cuda:{args2.device}')
    print(f"Using device {device}")

    main(args2, device)
