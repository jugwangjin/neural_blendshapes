

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
import pickle
import torch
import random
import numpy as np
import torch.nn.functional as F
import pytorch3d.transforms as pt3d

from arguments import config_parser
from pathlib import Path
from flare.core import Mesh
from flare.utils.ict_model import ICTFaceKitTorch
from flare.core.camera import Camera
from flare.modules import NeuralShader, get_neural_blendshapes
from flare.core.renderer import Renderer
import time
import nvdiffrec.render.light as light
from flare.dataset.dataset_real import DatasetLoader  # Import the DatasetLoader

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
from flare.utils import (
    AABB, read_mesh, write_mesh,
    visualize_training, visualize_training_no_lm,
    make_dirs, set_defaults_finetune, copy_sources
)
set_seed(20202464)
os.environ["GLOG_minloglevel"] = "2"

def load_model(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ict_facekit = ICTFaceKitTorch(npy_dir = './assets/ict_facekit_torch.npy', canonical = Path(args.input_dir) / 'ict_identity.npy')
    ict_facekit = ict_facekit.to(device)

    ict_canonical_mesh = Mesh(ict_facekit.canonical[0].cpu().data, ict_facekit.faces.cpu().data, ict_facekit=ict_facekit, device=device)
    ict_canonical_mesh.compute_connectivity()

    aabb = AABB(ict_canonical_mesh.vertices.cpu().numpy())
    ict_mesh_aabb = [torch.min(ict_canonical_mesh.vertices, dim=0).values, torch.max(ict_canonical_mesh.vertices, dim=0).values]

    # Load Neural Blendshapes
    neural_blendshapes = get_neural_blendshapes(
        model_path=args.model_path,
        vertex_parts=ict_facekit.vertex_parts, 
        ict_facekit=ict_facekit, 
        exp_dir = None, 
        lambda_=args.lambda_, 
        aabb = ict_mesh_aabb, 
        device=device
    )
    neural_blendshapes.eval()


    lgt = light.create_env_rnd()    
    disentangle_network_params = {
        "material_mlp_ch": args.material_mlp_ch,
        "light_mlp_ch":args.light_mlp_ch,
        "material_mlp_dims":args.material_mlp_dims,
        "light_mlp_dims":args.light_mlp_dims,
        "brdf_mlp_dims": args.brdf_mlp_dims,

    }


    load_shader = Path(args.shader_path)

    try:
        shader = NeuralShader(fourier_features='hashgrid',
            activation=args.activation,
            last_activation=torch.nn.Sigmoid(), 
            disentangle_network_params=disentangle_network_params,
            bsdf=args.bsdf,
            aabb=ict_mesh_aabb,
            device=device)
        shader = NeuralShader.load(load_shader, device=device)
        shader.eval()

    except:
        shader = NeuralShader(fourier_features='positional',
                            activation=args.activation,
                            last_activation=torch.nn.Sigmoid(), 
                            disentangle_network_params=disentangle_network_params,
                            bsdf=args.bsdf,
                            aabb=ict_mesh_aabb,
                            device=device)
        shader = NeuralShader.load(load_shader, device=device)
        shader.eval()

    # Load Renderer
    renderer = Renderer(device=device)
    lgt = light.create_env_rnd()

    return neural_blendshapes, ict_facekit, shader, lgt, renderer

def get_flame_camera(args):
    # Build a temporary dataset to get 'flame_camera'
    dataset = DatasetLoader(
        args=args,
        train_dir=args.train_dir,
        sample_ratio=100,
        pre_load=False,
        train=False,
        flip=False
    )
    # Get batch size 1 views
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=dataset.collate, drop_last=True)
    views_sample = next(iter(dataloader))
    return views_sample

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = config_parser()

    args = parser.parse_args()


    model, ict_facekit, shader, lgt, renderer = load_model(args)
    model.eval()
    shader.eval()

    # get fixed returns for neural_blendshapes
    precomputed_blendshapes = model.precompute_networks()
    # Create the mesh once

    mesh = Mesh(ict_facekit.canonical[0].cpu().data, ict_facekit.faces.cpu().data, ict_facekit=ict_facekit, device=device)
    mesh.compute_connectivity()
    # Obtain 'flame_camera' from the temporary dataset
    views_sample = get_flame_camera(args)



    handle_values = torch.zeros(53, device=device)
    handle_values2 = torch.zeros(7, device=device)

    # Initialize GUI
    app = gui.Application.instance
    app.initialize()

    window_width = 512+500  # 512 for image, 300 for sliders
    window_height = 600  # Adjust as needed
    window = gui.Application.instance.create_window("GUI", window_width, window_height)

    print('created window')

    em = window.theme.font_size
    spacing = int(np.round(0.25 * em))
    vspacing = int(np.round(0.5 * em))
    margins = gui.Margins(vspacing)

    # Create an ImageWidget to display rendered images
    image_widget = gui.ImageWidget()
    image_widget.frame = gui.Rect(500, 0, 512, 512)
    window.add_child(image_widget)

    # Create sliders panel
    panel = gui.CollapsableVert("Handle Activations", 0, gui.Margins(em, em, em, em))
    panel.frame = gui.Rect(0, 0, 500, window_height)
    window.add_child(panel)

    with open('./assets/mediapipe_name_to_indices.pkl', 'rb') as f:
        MEDIAPIPE_BLENDSHAPES = pickle.load(f)

    pose_names = ['rot_x', 'rot_y', 'rot_z', 'trans_x', 'trans_y', 'trans_z', 'scale']



    labels = [gui.Label(ict_facekit.expression_names[i]) for i in range(53)]
    labels2 = [gui.Label(pose_names[i]) for i in range(7)]
    sliders = [gui.Slider(gui.Slider.DOUBLE) for _ in range(53)]
    sliders2 = [gui.Slider(gui.Slider.DOUBLE) for _ in range(7)]


    def update_image():
        features = torch.cat([handle_values, handle_values2], dim=0).unsqueeze(0).to(device)  # Shape: [1, 62]

        with torch.no_grad():
            # Measure time for model forward pass
            start_time = time.time()
            return_dict = model.deform_with_precomputed(features, precomputed_blendshapes)
            deformed_vertices = return_dict['expression_mesh_posed']  # Adjust the key if necessary
            model_time = time.time() - start_time

            # Measure time for render_batch
            start_time = time.time()
            gbuffers = renderer.render_batch(
                views_sample['camera'],
                deformed_vertices.contiguous(),
                mesh.fetch_all_normals(deformed_vertices, mesh),
                channels=['mask', 'position', 'normal', "canonical_position"],
                with_antialiasing=True,
                canonical_v=mesh.vertices,
                canonical_idx=mesh.indices,
                canonical_uv=ict_facekit.uv_neutral_mesh,
                mesh=mesh
            )
            render_time = time.time() - start_time

            # Measure time for shader
            start_time = time.time()
            rgb_pred, cbuffers, _ = shader.shade(gbuffers, views_sample, mesh, False, lgt)
            shader_time = time.time() - start_time

            # Convert the rendered image to a numpy array
            rendered_image = rgb_pred.squeeze(0).cpu().numpy()  # Shape: [H, W, 3]
            rendered_image = (np.clip(rendered_image, 0, 1) * 255).astype(np.uint8)

            # Ensure the numpy array is contiguous
            rendered_image = np.ascontiguousarray(rendered_image)

            # Create an Open3D Image
            o3d_image = o3d.geometry.Image(rendered_image)

            # Update the image in the image_widget
            image_widget.update_image(o3d_image)
            total_time = model_time + render_time + shader_time
            fps = 1.0 / total_time if total_time > 0 else float('inf')
            print(f"Model forward pass time: {model_time:.4f} seconds | Render batch time: {render_time:.4f} seconds | Shader time: {shader_time:.4f} seconds | fps: {fps:.2f}")


    def create_fun(i):
        def fun(value):
            handle_values[i] = value
            update_image()
        return fun

    def create_fun2(i):
        def fun(value):
            handle_values2[i] = value
            update_image()
        return fun

    fixed_prop_grid = gui.VGrid(2, spacing, gui.Margins(em, em, em, em))

    for i, slider in enumerate(sliders):
        slider.set_limits(0, 1)
        slider.double_value = 0.0
        slider.set_on_value_changed(create_fun(i))

    for i, slider in enumerate(sliders2):
        slider.set_limits(-1, 1)
        slider.double_value = 0.0
        slider.set_on_value_changed(create_fun2(i))

    sliders2[-1].set_limits(0.5, 1.5)
    sliders2[-1].double_value = 1.0


    for label, slider in zip(labels, sliders):
        fixed_prop_grid.add_child(label)
        fixed_prop_grid.add_child(slider)
    for label, slider in zip(labels2, sliders2):
        fixed_prop_grid.add_child(label)
        fixed_prop_grid.add_child(slider)

    panel.add_child(fixed_prop_grid)

    # Initial image update
    update_image()

    # gui.Application.instance.run()

    app.run()