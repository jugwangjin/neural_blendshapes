import os
import pickle
import torch
import random
import numpy as np
import torch.nn.functional as F
import pytorch3d.transforms as pt3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import open3d as o3d

from arguments import config_parser
from pathlib import Path
from flare.core import Mesh
from flare.utils.ict_model import ICTFaceKitTorch
from flare.dataset.camera import Camera
from flare.modules import NeuralShader, get_neural_blendshapes
from flare.core.renderer import Renderer
from flare.utils import light
from flare.dataset.dataset_real import DatasetLoader  # Import the DatasetLoader

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(20202464)
os.environ["GLOG_minloglevel"] = "2"

def load_model(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load ICT FaceKit
    ict_facekit = ICTFaceKitTorch(
        npy_dir='./assets/ict_facekit_torch.npy',
        canonical=Path(args.input_dir) / 'ict_identity.npy',
        only_face=False
    ).to(device)
    ict_facekit.eval()

    # Load Neural Blendshapes
    neural_blendshapes = get_neural_blendshapes(
        model_path=args.model_path,
        train=False,
        ict_facekit=ict_facekit,
        device=device
    )
    neural_blendshapes.eval()

    # Load Shader
    shader = NeuralShader().to(device)
    shader.load_state_dict(torch.load(args.shader_path, map_location=device))
    shader.eval()

    # Load Renderer
    renderer = Renderer(device=device)
    lgt = light.create_env_rnd()

    return neural_blendshapes, ict_facekit, shader, lgt, renderer

def get_flame_camera(args):
    # Build a temporary dataset to get 'flame_camera'
    dataset = DatasetLoader(
        args=args,
        train_dir=[Path(args.input_dir)],
        sample_ratio=1.0,
        pre_load=False,
        train=False,
        flip=False
    )
    # Get the first sample from the dataset
    data_sample = dataset[0]
    # Obtain 'flame_camera' from the sample
    flame_camera = data_sample['flame_camera']
    # Remove the dataset object
    del dataset
    return flame_camera

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = config_parser()
    args = parser.parse_args()

    model, ict_facekit, shader, lgt, renderer = load_model(args)
    model.eval()
    shader.eval()

    # Obtain 'flame_camera' from the temporary dataset
    flame_camera = get_flame_camera(args)

    handle_values = torch.zeros(53, device=device)
    handle_values2 = torch.zeros(9, device=device)

    # Create the mesh once
    mesh = Mesh(
        ict_facekit.canonical[0],
        ict_facekit.faces,
        ict_facekit=ict_facekit,
        device=device
    )

    # Initialize GUI
    gui.Application.instance.initialize()
    window_width = 512 + 300  # 512 for image, 300 for sliders
    window_height = 600  # Adjust as needed
    window = gui.Application.instance.create_window("GUI Image Viewer", window_width, window_height)

    em = window.theme.font_size
    spacing = int(np.round(0.25 * em))
    vspacing = int(np.round(0.5 * em))
    margins = gui.Margins(vspacing)

    # Create an ImageWidget to display rendered images
    image_widget = gui.ImageWidget()
    image_widget.frame = gui.Rect(300, 0, 512, 512)
    window.add_child(image_widget)

    # Create sliders panel
    panel = gui.CollapsableVert("Handle Activations", 0, gui.Margins(em, em, em, em))
    panel.frame = gui.Rect(0, 0, 300, window_height)
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
            # Obtain deformed vertices from the model
            return_dict = model(features=features)
            deformed_vertices = return_dict['expression_mesh_posed']  # Adjust the key if necessary

            # Use the 'flame_camera' obtained from the dataset
            views = {'flame_camera': flame_camera}

            # Render GBuffers
            gbuffers = renderer.render_batch(
                flame_camera,
                deformed_vertices.contiguous(),
                mesh.fetch_all_normals(deformed_vertices, mesh),
                channels=['mask', 'position', 'normal', "canonical_position"],
                with_antialiasing=True,
                canonical_v=mesh.vertices,
                canonical_idx=mesh.indices,
                canonical_uv=ict_facekit.uv_neutral_mesh,
                mesh=mesh
            )

            # Shade to get the final image
            rgb_pred, cbuffers, _ = shader.shade(gbuffers, views, mesh, False, lgt)

            # Convert the rendered image to a numpy array
            rendered_image = rgb_pred.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Shape: [H, W, 3]
            rendered_image = (np.clip(rendered_image, 0, 1) * 255).astype(np.uint8)

            # Create an Open3D Image
            o3d_image = o3d.geometry.Image(rendered_image)

            # Update the image in the image_widget
            image_widget.update_image(o3d_image)

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

    for label, slider in zip(labels, sliders):
        fixed_prop_grid.add_child(label)
        fixed_prop_grid.add_child(slider)
    for label, slider in zip(labels2, sliders2):
        fixed_prop_grid.add_child(label)
        fixed_prop_grid.add_child(slider)

    panel.add_child(fixed_prop_grid)

    # Initial image update
    update_image()

    gui.Application.instance.run()
