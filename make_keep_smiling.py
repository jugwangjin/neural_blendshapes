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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

import os
os.environ["GLOG_minloglevel"] ="2"
import numpy as np
from pathlib import Path

import torch
from tqdm import tqdm
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


import imageio

@torch.no_grad()
def main(args, device, dataset_train2, dataset_train):


    original_dir = os.getcwd()
    # Add the path to the 'flare' directory
    flare_path = os.path.join(args.model_dir, args.model_name, 'sources')
    
    sys.path.insert(0, flare_path)

    from flame.FLAME import FLAME
    from flare.dataset import DatasetLoader
    from flare.dataset import dataset_util

    from flare.core import (
        Mesh, Renderer
    )
    from flare.modules import (
        NeuralShader, get_neural_blendshapes
    )
    from flare.utils import (
        AABB, read_mesh, write_mesh,
        visualize_training, visualize_training_no_lm,
        make_dirs, set_defaults_finetune, copy_sources
    )
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

    from flare.utils import (
        AABB, read_mesh, write_mesh,
        visualize_training, visualize_training_no_lm,
        make_dirs, set_defaults_finetune, copy_sources, save_shading
    )
    '''
    dir
    '''
    output_dir = os.path.join(args.output_dir, args.run_name, args.output_dir_name)

    '''
    models
    '''

    # ict facekit
    ict_facekit = ICTFaceKitTorch(npy_dir = './assets/ict_facekit_torch.npy', canonical = Path(args.input_dir) / 'ict_identity.npy')
    ict_facekit = ict_facekit.to(device)

    ict_canonical_mesh = Mesh(ict_facekit.canonical[0].cpu().data, ict_facekit.faces.cpu().data, ict_facekit=ict_facekit, device=device)
    ict_canonical_mesh.compute_connectivity()
            
    # renderer
    aabb = AABB(ict_canonical_mesh.vertices.cpu().numpy())
    ict_mesh_aabb = [torch.min(ict_canonical_mesh.vertices, dim=0).values, torch.max(ict_canonical_mesh.vertices, dim=0).values]

    renderer = Renderer(device=device)
    renderer.set_near_far(dataset_train2, torch.from_numpy(aabb.corners).to(device), epsilon=0.5)

    channels_gbuffer = ['mask', 'position', 'normal', "canonical_position"]
    print("Rasterizing:", channels_gbuffer)
    
    renderer_visualization = Renderer(device=device)
    renderer_visualization.set_near_far(dataset_train2, torch.from_numpy(aabb.corners).to(device), epsilon=0.5)

    # neural blendshapes
    try:
        model_path = os.path.join(args.output_dir, args.run_name, 'stage_1', 'network_weights', 'neural_blendshapes_latest.pt')
    except:
        model_path = os.path.join(args.output_dir, args.run_name, 'stage_1', 'network_weights', 'neural_blendshapes.pt')


    print("=="*50)

    print("Training Deformer")
    face_normals = ict_canonical_mesh.get_vertices_face_normals(ict_facekit.neutral_mesh_canonical[0])[0]
    neural_blendshapes = get_neural_blendshapes(model_path=model_path, train=args.train_deformer, ict_facekit=ict_facekit, aabb = ict_mesh_aabb, face_normals=face_normals,device=device) 
    
    neural_blendshapes = neural_blendshapes.to(device)

    # shader
    lgt = light.create_env_rnd()    
    try:
        shader = NeuralShader.load(os.path.join(args.output_dir, args.run_name, 'stage_1', 'network_weights', 'shader_latest.pt'), device=device)
    except:
        shader = NeuralShader.load(os.path.join(args.output_dir, args.run_name, 'stage_1', 'network_weights', 'shader.pt'), device=device)

    output_dir = args.output_dir_name
    os.makedirs(output_dir, exist_ok=True)

    dataloader_train    = torch.utils.data.DataLoader(dataset_train, batch_size=1, collate_fn=dataset_train.collate, drop_last=True, shuffle=True)



    all_brows  = ['browDown_L', 'browDown_R', 'browInnerUp_L', 'browInnerUp_R', 'browOuterUp_L', 'browOuterUp_R']

    eyeball_center = ['eyeLookUp_L', 'eyeLookUp_R', 'eyeLookDown_L', 'eyeLookDown_R', 'eyeLookIn_L', 'eyeLookIn_R', 'eyeLookOut_L', 'eyeLookOut_R']
    eyeblink = ['eyeBlink_L', 'eyeBlink_R']

    smiles = ['mouthSmile_L', 'mouthSmile_R']
    cheeks = ['cheekSquint_L', 'cheekSquint_R']
    eyebrow_raisers = ['browInnerUp_L', 'browInnerUp_R', 'browOuterUp_L', 'browOuterUp_R']
    eyebrow_downers = ['browDown_L', 'browDown_R']  

    jawopen = ['jawOpen']

    happiness = ['cheekSquint_L', 'cheekSquint_R', 'mouthSmile_L', 'mouthSmile_R']
    sadness = ['browInnerUp_L', 'browInnerUp_R', 'browDown_L', 'browDown_R', 'mouthFrown_L', 'mouthFrown_R']
    disgust = ['noseSneer_L', 'noseSneer_R', 'browDown_L', 'browDown_R', 'mouthFrown_L', 'mouthFrown_R']


    iteration = 0
    os.makedirs(output_dir, exist_ok=True)
    progress_bar = tqdm(enumerate(dataloader_train))
    for iter_, views_subset in progress_bar:
        return_dict = neural_blendshapes(views_subset['img'], views_subset)
        deformed_vertices = return_dict['expression_mesh_posed']
        d_normals = ict_canonical_mesh.fetch_all_normals(deformed_vertices, ict_canonical_mesh)

        gbuffers = renderer.render_batch(views_subset['flame_camera'], deformed_vertices.contiguous(), d_normals,
                                    channels=channels_gbuffer, with_antialiasing=True, 
                                    canonical_v=ict_canonical_mesh.vertices, canonical_idx=ict_canonical_mesh.indices, canonical_uv=ict_facekit.uv_neutral_mesh,
                                    mesh=ict_canonical_mesh
                                    )
        pred_color_masked, cbuffers, gbuffer_mask = shader.shade(gbuffers, views_subset, ict_canonical_mesh, args.finetune_color, lgt)

        file_name = os.path.join(output_dir, f'{args.run_name}_{args.video_name}_{views_subset["idx"][0]}.png')
        save_manipulation_image(pred_color_masked, views_subset, gbuffers["normal"], gbuffer_mask, file_name)

    #     # also save the gt 
        gt_iamge = views_subset['img']
        convert_uint = lambda x: np.clip(np.rint(dataset_util.rgb_to_srgb(x).detach().numpy() * 255.0), 0, 255).astype(np.uint8) 
        for i in range(len(gt_iamge)):
            gt_img_with_mask = torch.cat([gt_iamge[i].cpu().data, views_subset['mask'][i].cpu().data], -1)
            imageio.imsave(file_name.replace(f'.png', f'_gt_{i}.png'), convert_uint(gt_img_with_mask))



        features = return_dict['features']


        # # overriding facs to look front
        features[:, [ict_facekit.expression_names.tolist().index(name) for name in eyeball_center]] = features[:, [ict_facekit.expression_names.tolist().index(name) for name in eyeball_center]] * 0.1
        # features[:, [ict_facekit.expression_names.tolist().index(name) for name in eyeblink]] = 0


        facs = features[:, :53]
        rotation = features[:, 53:56]
        translation = features[:, 56:59]
        global_translation = features[:, 60:63]

        # overriding pose to forward facing
        translation = translation * 0.1
        rotation = rotation * 0.1
        global_translation = global_translation

        features[:, 53:56] = rotation
        features[:, 56:59] = translation
        features[:, 60:63] = global_translation





        # happiness_amount =         
        # sadness_amount = torch.min(facs[:, [ict_facekit.expression_names.tolist().index(name) for name in sadness]], dim=1).values 
        # facs[:, [ict_facekit.expression_names.tolist().index(name) for name in sadness]] -= sadness_amount
        # facs[:, [ict_facekit.expression_names.tolist().index(name) for name in happiness]] += sadness_amount

        return_dict = neural_blendshapes(features=features)
        deformed_vertices = return_dict['expression_mesh_posed']
        d_normals = ict_canonical_mesh.fetch_all_normals(deformed_vertices, ict_canonical_mesh)

        gbuffers = renderer.render_batch(views_subset['flame_camera'], deformed_vertices.contiguous(), d_normals,
                                    channels=channels_gbuffer, with_antialiasing=True, 
                                    canonical_v=ict_canonical_mesh.vertices, canonical_idx=ict_canonical_mesh.indices, canonical_uv=ict_facekit.uv_neutral_mesh,
                                    mesh=ict_canonical_mesh
                                    )
        pred_color_masked, cbuffers, gbuffer_mask = shader.shade(gbuffers, views_subset, ict_canonical_mesh, args.finetune_color, lgt)

        file_name = os.path.join(output_dir, f'{args.run_name}_{args.video_name}_{views_subset["idx"][0]}_fixed_gaze.png')
        save_manipulation_image(pred_color_masked, views_subset, gbuffers["normal"], gbuffer_mask, file_name)

        if iter_ > 400:
            break

    #     deformed_vertices = neural_blendshapes.ict_facekit(expression_weights = return_dict['features'][0][None, :53], identity_weights = neural_blendshapes.encoder.identity_weights[None])

    #     deformed_vertices = neural_blendshapes.apply_deformation(deformed_vertices, return_dict['features'], return_dict['pose_weight'])

    #     d_normals = ict_canonical_mesh.fetch_all_normals(deformed_vertices, ict_canonical_mesh)

    #     gbuffers = renderer.render_batch(views_subset['flame_camera'], deformed_vertices.contiguous(), d_normals,
    #                                 channels=channels_gbuffer, with_antialiasing=True, 
    #                                 canonical_v=ict_canonical_mesh.vertices, canonical_idx=ict_canonical_mesh.indices, canonical_uv=ict_facekit.uv_neutral_mesh,
    #                                 mesh=ict_canonical_mesh
    #                                 )
    #     pred_color_masked, cbuffers, gbuffer_mask = shader.shade(gbuffers, views_subset, ict_canonical_mesh, args.finetune_color, lgt)

    #     file_name = os.path.join(output_dir, f'{args.run_name}_{args.video_name}_{views_subset["idx"][0]}_personalized_no_bshapes.png')
    #     save_manipulation_image(pred_color_masked, views_subset, gbuffers["normal"], gbuffer_mask, file_name)

    #     # determine output image name. 
    #     shading_name = f'{args.run_name}_{args.video_name}_{views_subset["idx"][0]}_personalized_no_bshapes_shading.png'
    #     save_shading(pred_color_masked, cbuffers, gbuffers, views_subset, Path(output_dir), 0, ict_facekit=ict_facekit, save_name=shading_name, transparency=True)


    #     features = return_dict['features'].clone().detach()
    #     features[:, :53] = views_subset['mp_blendshape'][:, neural_blendshapes.encoder.ict_facekit.mediapipe_to_ict]
        

    #     deformed_vertices = neural_blendshapes.ict_facekit(expression_weights = features[0][None, :53], identity_weights = neural_blendshapes.encoder.identity_weights[None])

    #     deformed_vertices = neural_blendshapes.apply_deformation(deformed_vertices, return_dict['features'], return_dict['pose_weight'])

    #     d_normals = ict_canonical_mesh.fetch_all_normals(deformed_vertices, ict_canonical_mesh)

    #     gbuffers = renderer.render_batch(views_subset['flame_camera'], deformed_vertices.contiguous(), d_normals,
    #                                 channels=channels_gbuffer, with_antialiasing=True, 
    #                                 canonical_v=ict_canonical_mesh.vertices, canonical_idx=ict_canonical_mesh.indices, canonical_uv=ict_facekit.uv_neutral_mesh,
    #                                 mesh=ict_canonical_mesh
    #                                 )
    #     pred_color_masked, cbuffers, gbuffer_mask = shader.shade(gbuffers, views_subset, ict_canonical_mesh, args.finetune_color, lgt)

    #     file_name = os.path.join(output_dir, f'{args.run_name}_{args.video_name}_{views_subset["idx"][0]}_no_bshapes.png')
    #     save_manipulation_image(pred_color_masked, views_subset, gbuffers["normal"], gbuffer_mask, file_name)

    #     # determine output image name. 
    #     shading_name = f'{args.run_name}_{args.video_name}_{views_subset["idx"][0]}_no_bshapes_shading.png'
    #     save_shading(pred_color_masked, cbuffers, gbuffers, views_subset, Path(output_dir), 0, ict_facekit=ict_facekit, save_name=shading_name, transparency=True)


        # mp_bshapes = views_subset['mp_blendshape'].clone().detach()
        # mp_bshapes = mp_bshapes[:, neural_blendshapes.encoder.ict_facekit.mediapipe_to_ict]

        # features = return_dict['features']
        # features[:, :53] = mp_bshapes

        # return_dict = neural_blendshapes(features = features)
        # deformed_vertices = return_dict['expression_mesh_posed']
        # d_normals = ict_canonical_mesh.fetch_all_normals(deformed_vertices, ict_canonical_mesh)

        # gbuffers = renderer.render_batch(views_subset['flame_camera'], deformed_vertices.contiguous(), d_normals,
        #                             channels=channels_gbuffer, with_antialiasing=True, 
        #                             canonical_v=ict_canonical_mesh.vertices, canonical_idx=ict_canonical_mesh.indices, canonical_uv=ict_facekit.uv_neutral_mesh,
        #                             mesh=ict_canonical_mesh
        #                             )
        # pred_color_masked, cbuffers, gbuffer_mask = shader.shade(gbuffers, views_subset, ict_canonical_mesh, args.finetune_color, lgt)

        # file_name = os.path.join(output_dir, f'{args.run_name}_{args.video_name}_{views_subset["idx"][0]}_no_personalization.png')
        # save_manipulation_image(pred_color_masked, views_subset, gbuffers["normal"], gbuffer_mask, file_name)

        # # determine output image name. 
        # shading_name = f'{args.run_name}_{args.video_name}_{views_subset["idx"][0]}_shading_no_personalization.png'
        # save_shading(pred_color_masked, cbuffers, gbuffers, views_subset, Path(output_dir), 0, ict_facekit=ict_facekit, save_name=shading_name, transparency=True)

        # return_dict = neural_blendshapes(views_subset['img'], views_subset)
        # deformed_vertices = return_dict['expression_mesh_posed']
        # d_normals = ict_canonical_mesh.fetch_all_normals(deformed_vertices, ict_canonical_mesh)

        # gbuffers = renderer.render_batch(views_subset['flame_camera'], deformed_vertices.contiguous(), d_normals,
        #                             channels=channels_gbuffer, with_antialiasing=True, 
        #                             canonical_v=ict_canonical_mesh.vertices, canonical_idx=ict_canonical_mesh.indices, canonical_uv=ict_facekit.uv_neutral_mesh,
        #                             mesh=ict_canonical_mesh
        #                             )
        # pred_color_masked, cbuffers, gbuffer_mask = shader.shade(gbuffers, views_subset, ict_canonical_mesh, args.finetune_color, lgt)

        # file_name = os.path.join(output_dir, f'{args.run_name}_{args.video_name}_{views_subset["idx"][0]}.png')
        # save_manipulation_image(pred_color_masked, views_subset, gbuffers["normal"], gbuffer_mask, file_name)

        # # determine output image name. 
        # shading_name = f'{args.run_name}_{args.video_name}_{views_subset["idx"][0]}_shading.png'
        # save_shading(pred_color_masked, cbuffers, gbuffers, views_subset, Path(output_dir), 0, ict_facekit=ict_facekit, save_name=shading_name, transparency=True)


        # # also save the gt 
        # gt_iamge = views_subset['img']
        # convert_uint = lambda x: np.clip(np.rint(dataset_util.rgb_to_srgb(x).detach().numpy() * 255.0), 0, 255).astype(np.uint8) 
        # for i in range(len(gt_iamge)):
        #     gt_img_with_mask = torch.cat([gt_iamge[i].cpu().data, views_subset['mask'][i].cpu().data], -1)
        #     imageio.imsave(file_name.replace(f'.png', f'_gt_{i}.png'), convert_uint(gt_img_with_mask))

        #     # imageio.imsave(file_name.replace(f'.png', f'_gt_{i}.png'), convert_uint(gt_iamge[i].cpu().data))

    
        # mp_bshapes = views_subset['mp_blendshape'].clone().detach()
        # mp_bshapes = mp_bshapes[:, neural_blendshapes.encoder.ict_facekit.mediapipe_to_ict]

        # features = return_dict['features']
        # features[:, :53] = mp_bshapes

        # return_dict = neural_blendshapes(features = features)
        # deformed_vertices = return_dict['expression_mesh_posed']
        # d_normals = ict_canonical_mesh.fetch_all_normals(deformed_vertices, ict_canonical_mesh)

        # gbuffers = renderer.render_batch(views_subset['flame_camera'], deformed_vertices.contiguous(), d_normals,
        #                             channels=channels_gbuffer, with_antialiasing=True, 
        #                             canonical_v=ict_canonical_mesh.vertices, canonical_idx=ict_canonical_mesh.indices, canonical_uv=ict_facekit.uv_neutral_mesh,
        #                             mesh=ict_canonical_mesh
        #                             )
        # pred_color_masked, cbuffers, gbuffer_mask = shader.shade(gbuffers, views_subset, ict_canonical_mesh, args.finetune_color, lgt)

        # file_name = os.path.join(output_dir, f'{args.run_name}_{args.video_name}_{views_subset["idx"][0]}_no_personalization.png')
        # save_manipulation_image(pred_color_masked, views_subset, gbuffers["normal"], gbuffer_mask, file_name)

        # # determine output image name. 
        # shading_name = f'{args.run_name}_{args.video_name}_{views_subset["idx"][0]}_shading_no_personalization.png'
        # save_shading(pred_color_masked, cbuffers, gbuffers, views_subset, Path(output_dir), 0, ict_facekit=ict_facekit, save_name=shading_name, transparency=True)


        # instead of calling save_shading, write code here, as rgba image is needed





if __name__ == '__main__':
    parser = config_parser()
    parser.add_argument('--model_dir', type=str, default='/Bean/log/gwangjin/2024/nbshapes_comparisons/ours_enc_v13', help='Path to the trained model')
    parser.add_argument('--model_name', type=str, default='marcel', help='Name of the run in model_dir')
    parser.add_argument('--video_name', type=str, default='MVI_1802', help='Name of the video in dataset')
    parser.add_argument('--output_dir_name', type=str, default='figures/tracker_effects', help='Path to save the results. The results will be saved in model_dir / run_name / output_dir_name')
    args = parser.parse_args()


    config_file = os.path.join(args.model_dir, args.model_name, 'sources', 'config.txt')
    # Override the config file argument
    print(config_file)
    parser = config_parser()
    parser.add_argument('--index', nargs='+', type=str, default='0', help='List of indices (e.g., 1 3-6 8-11)')
    args2 = parser.parse_args(['--config', config_file])
    args2.video_name = args.video_name
    args2.eval_dir = [args.video_name]
    args2.run_name = args.model_name
    args2.output_dir = args.model_dir
    args2.output_dir_name = args.output_dir_name
    args2.model_dir = args.model_dir
    args2.model_name = args.model_name

    # Select the device
    device = torch.device('cpu')
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    print(f"Using device {device}")
    args = args2
    # ==============================================================================================
    # load data
    # ==============================================================================================
    print("loading train views...")


    original_dir = os.getcwd()
    # Add the path to the 'flare' directory
    flare_path = os.path.join(args.model_dir, args.model_name, 'sources')
    
    sys.path.insert(0, flare_path)

    from flare.dataset import DatasetLoader

    dataset_train    = DatasetLoader(args2, train_dir=args.train_dir, sample_ratio=1, pre_load=False, train=True)
    dataset_val      = DatasetLoader(args2, train_dir=args.eval_dir, sample_ratio=1, pre_load=False)
    

    # print(debug_views['flame_pose'].shape)
    # exit()
    # print(dataset_val[0]['flame_camera'].t, dataset_val[0]['flame_camera'].R)
    # print(dataset_train[0]['flame_camera'].t, dataset_train[0]['flame_camera'].R)
    # print(dataset_train[-1]['flame_camera'].t, dataset_train[-1]['flame_camera'].R)
    
    # exit()
    # # for i in range(100), save dataset_train[i]['img'] on debug/trainset
    # os.makedirs("debug/trainset", exist_ok=True)
    # for i in range(100):
    #     json_dict = dataset_train.all_img_path[i]
        
    #     print(json_dict["dir"] / Path(json_dict["file_path"] + ".png"))
    #     # shape of image

    # exit()
        

    
    # lbs = dataset_train.get_bshapes_lower_bounds()
    # print(lbs)
    
    # exit()

    # ==============================================================================================
    # main run
    # ==============================================================================================
    import time
    while True:
        try:
            main(args, device, dataset_train, dataset_val)
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

    