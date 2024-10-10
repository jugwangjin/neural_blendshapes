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

set_seed(20202464)

import os
os.environ["GLOG_minloglevel"] ="2"
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
from PIL import Image


def track_image(image_path, mediapipe):

        mp_image = Image.open(image_path)
        # if flip:
            
            # mp_image = mp_image.transpose(Image.FLIP_LEFT_RIGHT)
        
        # print(mp_image, mp_image.size, np.asarray(mp_image).shape, np.asarray(mp_image).dtype)  
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(mp_image))

        face_landmarker_result = mediapipe.detect(mp_image)
        mp_landmark, mp_blendshape, mp_transform_matrix = dataset_util.parse_mediapipe_output(face_landmarker_result)

        # ignore frames where no face is detected, just re-route to the next frame
        if mp_landmark is None:
            
            return None, None

        return mp_blendshape, mp_transform_matrix


@torch.no_grad
def main(args, device, dataset_train, dataloader_train, debug_views):


    ## ============== Dir ==============================
    # run_name = args.run_name if args.run_name is not None else args.input_dir.parent.name
    # images_save_path, images_eval_save_path, meshes_save_path, shaders_save_path, experiment_dir = make_dirs(args, run_name, args.finetune_color)
    # copy_sources(args, run_name)


    ## ============== load ict facekit ==============================
    ict_facekit = ICTFaceKitTorch(npy_dir = './assets/ict_facekit_torch.npy', canonical = Path(args.input_dir) / 'ict_identity.npy')
    ict_facekit = ict_facekit.to(device)

    ict_canonical_mesh = Mesh(ict_facekit.canonical[0].cpu().data, ict_facekit.faces.cpu().data, ict_facekit=ict_facekit, device=device)
    ict_canonical_mesh.compute_connectivity()


    ## ============== renderer ==============================
    aabb = AABB(ict_canonical_mesh.vertices.cpu().numpy())
    ict_mesh_aabb = [torch.min(ict_canonical_mesh.vertices, dim=0).values, torch.max(ict_canonical_mesh.vertices, dim=0).values]

    renderer = Renderer(device=device)
    renderer.set_near_far(dataset_train, torch.from_numpy(aabb.corners).to(device), epsilon=0.5)
    channels_gbuffer = ['mask', 'position', 'normal', "canonical_position"]
    print("Rasterizing:", channels_gbuffer)
    

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
    
    # make output directory
    output_dir = Path(args.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='./assets/face_landmarker.task'),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        )

    mediapipe = FaceLandmarker.create_from_options(options)

    # read each frame from args.input_video
    # save with cache directory + tmp.png
    # run mediapipe on each frame
    
    import cv2
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        cap = cv2.VideoCapture(args.input_video)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(frame_count)

        prev_mp_blendshape = None
        prev_mp_transform_matrix = None

        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            # if i > 50:
            #     break
            # if i % 50 != 0:
            #     continue

        
            # if i > 1500:
            #     break


            cv2.imwrite(f'{tmpdir}/tmp.png', frame)
            mp_blendshape, mp_transform_matrix = track_image(f'{tmpdir}/tmp.png', mediapipe)
            if mp_blendshape is None:
                mp_blendshape = prev_mp_blendshape
                mp_transform_matrix = prev_mp_transform_matrix
            
            prev_mp_blendshape = mp_blendshape
            prev_mp_transform_matrix = mp_transform_matrix

            mp_blendshape = mp_blendshape.to(device)
            mp_transform_matrix = mp_transform_matrix.to(device)


            # apply pose 
            
            mp_transform_matrix = mp_transform_matrix[None]
            scale = torch.norm(mp_transform_matrix[:, :3, :3], dim=-1).mean(dim=-1, keepdim=True)
            translation = mp_transform_matrix[:, :3, 3]
            rotation_matrix = mp_transform_matrix[:, :3, :3]
            rotation_matrix = mp_transform_matrix[:, :3, :3] / scale[:, None]

            rotation_matrix = rotation_matrix.permute(0, 2, 1)
            rotation = p3dt.matrix_to_euler_angles(rotation_matrix, convention='XYZ')
            
            translation[:, -1] += 30
            translation *= 0.02
            translation *= 0.0

            print(translation)

            ict_mesh = ict_facekit(expression_weights=mp_blendshape[ict_facekit.mediapipe_to_ict][None])

            ict_mesh = torch.einsum('bvd, bdj -> bvj', ict_mesh, rotation_matrix)  + translation

            mesh = ict_canonical_mesh.with_vertices(ict_mesh[0])

            # save the mesh
            # write_mesh(f'{str(output_dir)}/{i:05d}.obj', mesh.to('cpu'))
            # continue
            
            d_normals = mesh.fetch_all_normals(ict_mesh, mesh)

            gbuffer = renderer.render_batch(debug_views['camera'], deformed_vertices=ict_mesh, deformed_normals=d_normals,
                                            channels=channels_gbuffer, with_antialiasing=True,
                                            canonical_v=mesh.vertices, canonical_idx=mesh.indices, canonical_uv=ict_facekit.uv_neutral_mesh)
            rgb_pred, cbuffers, gbuffer_mask = shader.shade(gbuffer, debug_views, mesh, args.finetune_color, lgt)


            # determine output image name. 
            output_mesh = f'{i:05d}.png'
            save_shading(rgb_pred, cbuffers, gbuffer, debug_views, Path(tmpdir), 0, ict_facekit=ict_facekit, save_name=output_mesh)


        # Release the video capture object
        cap.release()
        
        # Create a video writer to save the side-by-side video
        output_video_path = os.path.join(args.output_directory, f'{args.output_video_name}.mp4')

        # Read the original video again to create the side-by-side video
        cap = cv2.VideoCapture(args.input_video)


        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30  # You can adjust the FPS as needed
        print(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width * 2, frame_height))

        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                # Read the rendered image
                rendered_image_path = os.path.join(tmpdir, f'{i:05d}.png')
                rendered_image = cv2.imread(rendered_image_path)

                # if it fails to read the rendered image, raise an exception
                if rendered_image is None:
                    raise Exception(f"Error reading rendered image {i}")


            except:
                print(f"Error reading rendered image {i}")
                break

            try:
                print(rendered_image.shape)
                # Resize the rendered image to match the original frame size
                rendered_image = cv2.resize(rendered_image, (frame_width, frame_height))

            except:
                print(f"Error resizing rendered image {i}")
                break

            try:
                # Concatenate the original frame and the rendered image side by side
                combined_frame = np.hstack((frame, rendered_image))
            
            except:
                print(f"Error concatenating frames {i}")
                break
            try:

                # Write the combined frame to the output video
                out.write(combined_frame)
                

            except:
                print(f"Error processing frame {i}")
                break

        # Release the video writer and capture objects
        out.release()
        cap.release()

        # Create a video writer to save the overlapped video
        overlapped_video_path = os.path.join(args.output_directory, f'{args.output_video_name}_overlap.mp4')

        # Read the original video again to create the overlapped video
        cap = cv2.VideoCapture(args.input_video)

        out = cv2.VideoWriter(overlapped_video_path, fourcc, fps, (frame_width, frame_height))

        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            try:
                # Read the rendered image
                rendered_image_path = os.path.join(tmpdir, f'{i:05d}.png')
                rendered_image = cv2.imread(rendered_image_path)

                # if it fails to read the rendered image, raise an exception
                if rendered_image is None:
                    raise Exception(f"Error reading rendered image {i}")

            except:
                print(f"Error reading rendered image {i}")
                break

            try:
                # Resize the rendered image to match the original frame size
                rendered_image = cv2.resize(rendered_image, (frame_width, frame_height))

            except:
                print(f"Error resizing rendered image {i}")
                break

            try:
                # Blend the original frame and the rendered image
                alpha = 0.5  # You can adjust the alpha value to change the blending ratio
                overlapped_frame = cv2.addWeighted(frame, alpha, rendered_image, 1 - alpha, 0)

            except:
                print(f"Error blending frames {i}")
                break

            try:
                # Write the overlapped frame to the output video
                out.write(overlapped_frame)

            except:
                print(f"Error processing frame {i}")
                break

        # Release the video writer and capture objects
        out.release()
        cap.release()


if __name__ == '__main__':
    parser = config_parser()
    # add arguments for parser
    parser.add_argument('--input_video', type=str)
    parser.add_argument('--output_directory', type=str, required=True)
    parser.add_argument('--output_video_name', type=str, required=True)
    args = parser.parse_args()


    device = torch.device('cpu')
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    print(f"Using device {device}")

    # needed arguments. 
    # train_dir
    # eval_dir

    # ==============================================================================================
    # load data
    # ==============================================================================================
    print("loading train views...")
    dataset_train    = DatasetLoader(args, train_dir=args.train_dir, sample_ratio=args.sample_idx_ratio, pre_load=False, train=True)
    dataset_val      = DatasetLoader(args, train_dir=args.eval_dir, sample_ratio=24, pre_load=False)
    # assert dataset_train.len_img == len(dataset_train.importance)
    # dataset_sampler = torch.utils.data.WeightedRandomSampler(dataset_train.importance, dataset_train.len_img, replacement=True)
    # dataloader_train    = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, collate_fn=dataset_train.collate, drop_last=True, sampler=dataset_sampler)
    dataloader_train = None
    view_indices = np.array([0]).astype(int)
    d_l = [dataset_val.__getitem__(idx) for idx in view_indices]
    debug_views = dataset_val.collate(d_l)


    del dataset_val

    main(args, device, dataset_train, dataloader_train, debug_views)