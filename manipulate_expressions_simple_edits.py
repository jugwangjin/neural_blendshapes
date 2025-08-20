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
from tkinter import N
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





@torch.no_grad()
def main(args, device):

    original_dir = os.getcwd()
    # Add the path to the 'flare' directory
    flare_path = os.path.join(args.model_dir, args.model_name, 'sources')
    
    sys.path.insert(0, flare_path)

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

    
    '''
    dataset
    '''
    print("loading views...")

    dataset_train    = DatasetLoader(args, train_dir=args.train_dir, sample_ratio=args.sample_idx_ratio, pre_load=False, train=True)
    args.eval_dir = [args.video_name]
    dataset_val      = DatasetLoader(args, train_dir=args.eval_dir, sample_ratio=1, pre_load=False)

    '''
    dir
    '''
    output_dir = os.path.join(args.output_dir_name)
    os.makedirs(output_dir, exist_ok=True)

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
    # neural blendshapes
    try:
        model_path = os.path.join(args.output_dir, args.run_name, 'stage_1', 'network_weights', 'neural_blendshapes_latest.pt')
    except:
        model_path = os.path.join(args.output_dir, args.run_name, 'stage_1', 'network_weights', 'neural_blendshapes.pt')
    print("=="*50)
    print("Training Deformer")
    face_normals = ict_canonical_mesh.get_vertices_face_normals(ict_facekit.neutral_mesh_canonical[0])[0]
    neural_blendshapes = get_neural_blendshapes(model_path=model_path, train=args.train_deformer, ict_facekit=ict_facekit, aabb = ict_mesh_aabb, face_normals=face_normals, device=device) 
    
    neural_blendshapes = neural_blendshapes.to(device)

    # shader
    lgt = light.create_env_rnd()    
    try:
        shader = NeuralShader.load(os.path.join(args.output_dir, args.run_name, 'stage_1', 'network_weights', 'shader_latest.pt'), device=device)
    except:
        shader = NeuralShader.load(os.path.join(args.output_dir, args.run_name, 'stage_1', 'network_weights', 'shader.pt'), device=device)

    neural_blendshapes.eval()
    shader.eval()

    '''
    manipulation_examples
    '''
    # names
    # for detail, refer to https://imotions.com/blog/learning/research-fundamentals/facial-action-coding-system/
    # 0: browDown_L
    # 1: browDown_R
    # 2: browInnerUp_L
    # 3: browInnerUp_R
    # 4: browOuterUp_L
    # 5: browOuterUp_R
    # 6: cheekPuff_L
    # 7: cheekPuff_R
    # 8: cheekSquint_L
    # 9: cheekSquint_R
    # 10: eyeBlink_L
    # 11: eyeBlink_R
    # 12: eyeLookDown_L
    # 13: eyeLookDown_R
    # 14: eyeLookIn_L
    # 15: eyeLookIn_R
    # 16: eyeLookOut_L
    # 17: eyeLookOut_R
    # 18: eyeLookUp_L
    # 19: eyeLookUp_R
    # 20: eyeSquint_L
    # 21: eyeSquint_R
    # 22: eyeWide_L
    # 23: eyeWide_R
    # 24: jawForward
    # 25: jawLeft
    # 26: jawOpen
    # 27: jawRight
    # 28: mouthClose
    # 29: mouthDimple_L
    # 30: mouthDimple_R
    # 31: mouthFrown_L
    # 32: mouthFrown_R
    # 33: mouthFunnel
    # 34: mouthLeft
    # 35: mouthLowerDown_L
    # 36: mouthLowerDown_R
    # 37: mouthPress_L
    # 38: mouthPress_R
    # 39: mouthPucker
    # 40: mouthRight
    # 41: mouthRollLower
    # 42: mouthRollUpper
    # 43: mouthShrugLower
    # 44: mouthShrugUpper
    # 45: mouthSmile_L
    # 46: mouthSmile_R
    # 47: mouthStretch_L
    # 48: mouthStretch_R
    # 49: mouthUpperUp_L
    # 50: mouthUpperUp_R
    # 51: noseSneer_L
    # 52: noseSneer_R

        # euler_angle = features[..., 53:56]
        # translation = features[..., 56:59]
        # scale = features[..., 59:60]
        # transform_origin = self.encoder.transform_origin
        # global_translation = features[..., 60:63]

    # CREATE: winking
    # CREATE: happiness
    # CREATE: surprise

    # Complete expression definitions
    expressions ={}
    expressions['wink_left'] = ['eyeBlink_L', 'mouthSmile_L']
    expressions['wink_right'] = ['eyeBlink_R', 'mouthSmile_R']
    expressions['happiness'] = ['mouthSmile_L', 'mouthSmile_R', 'cheekSquint_L', 'cheekSquint_R']
    expressions['surprise'] = ['browInnerUp_L', 'browInnerUp_R', 'browOuterUp_L', 'browOuterUp_R', 'jawOpen']
    
    # sadness
    expressions['sadness'] = ['browDown_L', 'browDown_R', 'mouthFrown_L', 'mouthFrown_R']
    
    
    

    for index in args.index:
        
        views_subset = dataset_val.collate([dataset_val.__getitem__(index)])

        import imageio
        from flare.utils import dataset_util



        # also save the gt 
        gt_iamge = views_subset['img']
        convert_uint = lambda x: np.clip(np.rint(dataset_util.rgb_to_srgb(x).detach().numpy() * 255.0), 0, 255).astype(np.uint8) 
        for i in range(len(gt_iamge)):
            gt_img_with_mask = torch.cat([gt_iamge[i].cpu().data, views_subset['mask'][i].cpu().data], -1)
            imageio.imsave(os.path.join(output_dir, f'{args.run_name}_{args.video_name}_{index}_gt_{i}.png'), convert_uint(gt_img_with_mask))


        # Create neutral face features
        features = neural_blendshapes.encoder(views_subset)
        facs = features[:, :53]
        translation = features[:, 53:56]
        rotation = features[:, 56:59]
        global_translation = features[:, 60:63]

        # Set all expressions to zero for neutral face
        facs = torch.zeros_like(facs)
        
        # Set pose to forward facing
        translation = translation * 0
        rotation = rotation * 0
        global_translation = global_translation



        # Render expressions with different activation values
        activation_values = [0.2, 0.4, 0.6, 0.8, 1.0]
        
        for expression_name, expression_components in expressions.items():
            # 1. Wink expressions (left and right)
            for activation in activation_values:
                # Left wink
                return_dict = neural_blendshapes(views_subset['img'], views_subset)
                features = return_dict['features']
                facs = features[:, :53]
                
                # Set wink expressions to activation value
                facs[:, [ict_facekit.expression_names.tolist().index(name) for name in expression_components]] = activation
                facs = facs.clamp(0, 1)
                features[:, :53] = facs
                features[:, 53:56] = rotation
                features[:, 56:59] = translation
                features[:, 60:63] = global_translation

                return_dict = neural_blendshapes(features=features)
                deformed_vertices = return_dict['expression_mesh_posed']
                d_normals = ict_canonical_mesh.fetch_all_normals(deformed_vertices, ict_canonical_mesh)
                gbuffers = renderer.render_batch(views_subset['flame_camera'], deformed_vertices.contiguous(), d_normals,
                                            channels=channels_gbuffer, with_antialiasing=True, 
                                            canonical_v=ict_canonical_mesh.vertices, canonical_idx=ict_canonical_mesh.indices, canonical_uv=ict_facekit.uv_neutral_mesh,
                                            mesh=ict_canonical_mesh
                                            )
                pred_color_masked, cbuffers, gbuffer_mask = shader.shade(gbuffers, views_subset, ict_canonical_mesh, args.finetune_color, lgt)

                file_name = os.path.join(output_dir, f'{args.run_name}_{args.video_name}_{index}_{expression_name}_{activation:.1f}.png')
                pred_color_masked_imageio = convert_uint(torch.cat([pred_color_masked[i].cpu().data, gbuffer_mask[i].cpu().data], -1))
                imageio.imsave(file_name, pred_color_masked_imageio)

        # ADD absolute neutral face
        return_dict = neural_blendshapes(views_subset['img'], views_subset)
        features = return_dict['features']
        facs = features[:, :53]
        facs = torch.zeros_like(facs)
        features[:, :53] = facs
        features[:, 53:56] = rotation
        features[:, 56:59] = translation
        features[:, 60:63] = global_translation

        return_dict = neural_blendshapes(features=features)
        deformed_vertices = return_dict['expression_mesh_posed']
        d_normals = ict_canonical_mesh.fetch_all_normals(deformed_vertices, ict_canonical_mesh)
        gbuffers = renderer.render_batch(views_subset['flame_camera'], deformed_vertices.contiguous(), d_normals,
                                        channels=channels_gbuffer, with_antialiasing=True, 
                                        canonical_v=ict_canonical_mesh.vertices, canonical_idx=ict_canonical_mesh.indices, canonical_uv=ict_facekit.uv_neutral_mesh,
                                        mesh=ict_canonical_mesh
                                        )
        pred_color_masked, cbuffers, gbuffer_mask = shader.shade(gbuffers, views_subset, ict_canonical_mesh, args.finetune_color, lgt)

        file_name = os.path.join(output_dir, f'{args.run_name}_{args.video_name}_{index}_neutral.png')
        pred_color_masked_imageio = convert_uint(torch.cat([pred_color_masked[i].cpu().data, gbuffer_mask[i].cpu().data], -1))
        imageio.imsave(file_name, pred_color_masked_imageio)


# def save_manipulation_image(rgb_pred, views, normals, gbuffer_mask, file_name, normal=True):
#     convert_uint = lambda x: np.clip(np.rint(dataset_util.rgb_to_srgb(x).detach().numpy() * 255.0), 0, 255).astype(np.uint8) 
#     convert_uint_255 = lambda x: (x * 255).to(torch.uint8)
    
#     for i in range(len(gbuffer_mask)):
#         mask = gbuffer_mask[i].cpu()

#         # rgb prediction
#         imageio.imsave(file_name.replace(f'.png', f'_rgb_{i}.png'), convert_uint(torch.cat([rgb_pred[i].cpu().data, mask], -1))) 
#         if normal:
#             ##normal
#             normal = (normals[i] + 1.) / 2.
#             normal = torch.cat([normal.cpu(), mask], -1)
#             imageio.imsave(file_name.replace(f'.png', f'_normal_{i}.png'), convert_uint_255(normal))    

                # save_manipulation_image(pred_color_masked, views_subset, gbuffers["normal"], gbuffer_mask, file_name)
        
        break




def parse_index(input_list):
    print(input_list)
    result = []
    for part in input_list:
        print(part)
        if '-' in part:
            start, end = map(int, part.split('-'))
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    return result

if __name__ == '__main__':
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='/Bean/log/gwangjin/2025/nbshapes_iccv_unit_wise/', help='Path to the trained model')
    parser.add_argument('--model_name', type=str, default='wojtek_1', help='Name of the run in model_dir')
    parser.add_argument('--video_name', type=str, default='train', help='Name of the video in dataset')
    parser.add_argument('--index', nargs='+', type=str, default=['1424-1426'], help='List of indices (e.g., 1 3-6 8-11)')
    parser.add_argument('--output_dir_name', type=str, default='figures/sample_edits', help='Path to save the results. The results will be saved in model_dir / run_name / output_dir_name')
    args = parser.parse_args()

    # Load the config file
    args_index = '0'
    config_file = os.path.join(args.model_dir, args.model_name, 'sources', 'config.txt')
    # Override the config file argument

    parser = config_parser()
    parser.add_argument('--index', nargs='+', type=str, default='0', help='List of indices (e.g., 1 3-6 8-11)')
    args2 = parser.parse_args(['--config', config_file])
    args2.index = parse_index(args.index)
    args2.video_name = args.video_name
    args2.run_name = args.model_name
    args2.output_dir = args.model_dir
    args2.output_dir_name = args.output_dir_name
    args2.model_dir = args.model_dir
    args2.model_name = args.model_name
    
    print(args2.input_dir, args2.video_name)
    print(args2.output_dir, args2.run_name)
    print(args2.index)

    # Select the device
    device = torch.device('cpu')
    if torch.cuda.is_available() and args2.device >= 0:
        device = torch.device(f'cuda:{args2.device}')
    print(f"Using device {device}")

    main(args2, device)
