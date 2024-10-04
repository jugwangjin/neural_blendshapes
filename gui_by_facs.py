import open3d as o3d
import numpy as np
import open3d.visualization.rendering as rendering
import torch
from arguments import config_parser
import os

from flare.core import Mesh

from flare.utils.ict_model import ICTFaceKitTorch

from pathlib import Path



from flare.utils.ict_model import ICTFaceKitTorch

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

def load_model(args):
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

    return neural_blendshapes, ict_facekit
    # =============================================================================================


def load_ict_facekit(args, device):
    ict_facekit = ICTFaceKitTorch(npy_dir = './assets/ict_facekit_torch.npy', canonical = Path(args.input_dir) / 'ict_identity.npy', only_face=False)
    ict_facekit = ict_facekit.to(device)
    ict_facekit.eval()

    ict_canonical_mesh = Mesh(ict_facekit.canonical[0].cpu().data, ict_facekit.faces.cpu().data, device=device)
    
    ict_canonical_mesh.compute_connectivity()
    # ict_facekit.update_vmapping(ict_canonical_mesh.vmapping)

    return ict_facekit, ict_canonical_mesh


if __name__ == "__main__":

    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering
    import open3d as o3d
    import numpy as np



    use_point_light = True

    
    torch.no_grad()
    device='cuda'
    parser = config_parser()
    args = parser.parse_args()

    model, ict_facekit = load_model(args)

    # empty triangle mesh o3d
    ict_canonical_vertices = ict_facekit.canonical[0].cpu().data.numpy()
    ict_faces = ict_facekit.faces.cpu().data.numpy()

    ict_mesh_o3d2 = o3d.geometry.TriangleMesh()
    ict_mesh_o3d2.vertices = o3d.utility.Vector3dVector(ict_canonical_vertices)
    ict_mesh_o3d2.triangles = o3d.utility.Vector3iVector(ict_faces)

    handle_values = torch.from_numpy(np.array([0 for i in range(52)])).to(device).float()
    handle_values2 = torch.from_numpy(np.array([0 for i in range(10)])).to(device).float()


    gui.Application.instance.initialize()
    window = gui.Application.instance.create_window(f"{use_point_light=}", 900, 1000)
    scene_widget2 = gui.SceneWidget()
    scene_widget2.scene = rendering.Open3DScene(window.renderer)
    bbox = o3d.geometry.AxisAlignedBoundingBox([-2, -2, -2], [2, 2, 2])
    scene_widget2.setup_camera(60, bbox, [0, 0, 0])

    material = rendering.MaterialRecord()
    # material.base_color = [.5,1,.5,1]
    material.shader = "defaultLit"


    scene_widget2.scene.add_geometry("box2", ict_mesh_o3d2, material)

    scene_widget2.scene.scene.enable_sun_light(False)

    def set_light_dir(light_dir):
        scene_widget2.scene.scene.remove_light('light')
        if use_point_light:
            scene_widget2.scene.scene.add_point_light('light',[1,1,1],-3*light_dir,1e8,1e2,True)
        else:
            scene_widget2.scene.scene.add_directional_light('light',[1,1,1],light_dir,1e6,True)

    scene_widget2.set_on_sun_direction_changed(set_light_dir)
    set_light_dir(np.array([-1,-1,0]))

    rect = window.content_rect
    scene_widget2.frame = gui.Rect(rect.x + 300, rect.y, (rect.width-300), rect.height)


    em = window.theme.font_size

    spacing = int(np.round(0.25 * em))
    vspacing = int(np.round(0.5 * em))

    margins = gui.Margins(vspacing)

    # First panel
    panel = gui.CollapsableVert("Handle_activations", 0,
                                            gui.Margins(em, 0, 0, 0))



    import pickle
    with open('./assets/mediapipe_name_to_indices.pkl', 'rb') as f:
        MEDIAPIPE_BLENDSHAPES = pickle.load(f)

    ## Items in fixed props
    fixed_prop_grid = gui.VGrid(2, spacing,
                                            gui.Margins(em, 0, em, 0))
    
    def get_key_by_value(val, my_dict):
        """
        This function takes a value and a dictionary as input and returns the key associated with the value in the dictionary.

        Args:
            val: The value to look up in the dictionary.
            my_dict: The dictionary to search in.

        Returns:
            The key associated with the value, or None if the value is not found.
        """
        for key, value in my_dict.items():
            if value == val:
                return key
        return 'not found'

    pose_names = ['rot_x', 'rot_y', 'rot_z', 'trans_x', 'trans_y', 'trans_z', 'scale', 'global_trans_x', 'global_trans_y', 'global_trans_z']

    labels = [gui.Label(model.ict_facekit.expression_names.tolist()[i-1]) for i in range(1, 52+1)]
    labels2 = [gui.Label(pose_names[i-1]) for i in range(1, 10+1)]
    sliders = [gui.Slider(gui.Slider.DOUBLE) for i in range(1, 52+1)]
    sliders2 = [gui.Slider(gui.Slider.DOUBLE) for i in range(1, 10+1)]
    def create_fun(i):
        def fun(value):
            handle_values[i] = value
            scene_widget2.scene.remove_geometry('box2')
            # print(mesh)
            # print(np.asarray(mesh.vertices))
            # print(np.asarray(mesh.vertices).shape)

            try:
                facs_torch = handle_values.unsqueeze(0)
                pose_torch = handle_values2.unsqueeze(0)

                return_dict = model(features=torch.cat([facs_torch, pose_torch], dim=-1))
                vertices = return_dict['expression_mesh_posed']


                ict_mesh_o3d2.vertices = o3d.utility.Vector3dVector(vertices[0].cpu().data.numpy())
                ict_mesh_o3d2.compute_vertex_normals()
                scene_widget2.scene.add_geometry("box2", ict_mesh_o3d2, material)

            except Exception as e:
                print(e)
                pass
            # print("REQR")
        return fun


    def create_fun2(i):
        def fun(value):
            handle_values2[i] = value
            scene_widget2.scene.remove_geometry('box2')
            # print(mesh)
            # print(np.asarray(mesh.vertices))
            # print(np.asarray(mesh.vertices).shape)

            try:
                facs_torch = handle_values.unsqueeze(0)
                pose_torch = handle_values2.unsqueeze(0)

                return_dict = model(features=torch.cat([facs_torch, pose_torch], dim=-1))
                vertices = return_dict['expression_mesh_posed']

                ict_mesh_o3d2.vertices = o3d.utility.Vector3dVector(vertices[0].cpu().data.numpy())
                ict_mesh_o3d2.compute_vertex_normals()
                scene_widget2.scene.add_geometry("box2", ict_mesh_o3d2, material)

            except Exception as e:
                print(e)
                pass
        return fun



    for i, slider in enumerate(sliders):
        slider.set_limits(0, 1)
        slider.set_on_value_changed(create_fun(i))

    for i, slider in enumerate(sliders2):
        slider.set_limits(-1, 1)
        slider.set_on_value_changed(create_fun2(i))

    for label, slider in zip(labels, sliders):
        fixed_prop_grid.add_child(label)
        fixed_prop_grid.add_child(slider)
    for label, slider in zip(labels2, sliders2):
        fixed_prop_grid.add_child(label)
        fixed_prop_grid.add_child(slider)

    panel.add_child(fixed_prop_grid)

    rect = window.content_rect
    panel.frame = gui.Rect(rect.x, rect.y, 300, rect.height)
    window.add_child(panel)
    window.add_child(scene_widget2)
    
    app = gui.Application.instance
    app.initialize()
    mono = app.add_font(gui.FontDescription(gui.FontDescription.MONOSPACE))
    app.run()

