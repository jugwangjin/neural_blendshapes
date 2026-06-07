

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
import time
import cv2

from arguments import config_parser
from pathlib import Path
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
import sys
set_seed(20202464)
os.environ["GLOG_minloglevel"] = "2"

def load_model(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ict_facekit = ICTFaceKitTorch(npy_dir = './assets/ict_facekit_torch.npy', canonical = Path(args.input_dir) / 'ict_identity.npy')
    ict_facekit = ict_facekit.to(device)

    ict_canonical_mesh = Mesh(
        ict_facekit.expression_reference_verts().cpu().data,
        ict_facekit.faces.cpu().data,
        ict_facekit=ict_facekit,
        device=device,
    )
    ict_canonical_mesh.compute_connectivity()

    aabb = AABB(ict_canonical_mesh.vertices.cpu().numpy())
    ict_mesh_aabb = [torch.min(ict_canonical_mesh.vertices, dim=0).values, torch.max(ict_canonical_mesh.vertices, dim=0).values]

    # neural blendshapes
    try:
        model_path = os.path.join(args.output_dir, args.run_name, 'stage_1', 'network_weights', 'neural_blendshapes_latest.pt')
    except:
        model_path = os.path.join(args.output_dir, args.run_name, 'stage_1', 'network_weights', 'neural_blendshapes.pt')

    print("Training Deformer")
    face_normals = ict_canonical_mesh.get_vertices_face_normals(ict_facekit.expression_reference_verts())[0]
    neural_blendshapes = get_neural_blendshapes(model_path=model_path, train=args.train_deformer, ict_facekit=ict_facekit, aabb = ict_mesh_aabb, face_normals=face_normals,device=device) 
    
    neural_blendshapes = neural_blendshapes.to(device)


    lgt = light.create_env_rnd()    
    disentangle_network_params = {
        "material_mlp_ch": args.material_mlp_ch,
        "light_mlp_ch":args.light_mlp_ch,
        "material_mlp_dims":args.material_mlp_dims,
        "light_mlp_dims":args.light_mlp_dims,
        "brdf_mlp_dims": args.brdf_mlp_dims,

    }


    # shader
    lgt = light.create_env_rnd()    
    try:
        shader = NeuralShader.load(os.path.join(args.output_dir, args.run_name, 'stage_1', 'network_weights', 'shader_latest.pt'), device=device)
    except:
        shader = NeuralShader.load(os.path.join(args.output_dir, args.run_name, 'stage_1', 'network_weights', 'shader.pt'), device=device)


    # Load Renderer
    renderer = Renderer(device=device)
    lgt = light.create_env_rnd()

    return neural_blendshapes, ict_facekit, shader, lgt, renderer

def get_flame_camera(args):
    # Build a temporary dataset to get 'flame_camera'
    dataset = DatasetLoader(
        args=args,
        train_dir=args.eval_dir,
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

    original_dir = os.getcwd()
    # Add the path to the 'flare' directory
    flare_path = os.path.join(args.output_dir, args.run_name, 'sources')
    
    sys.path.insert(0, flare_path)


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
    from flare.utils.ict_model import ICTFaceKitTorch


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




    # Select the device
    device = torch.device('cpu')
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    print(f"Using device {device}")


    model, ict_facekit, shader, lgt, renderer = load_model(args)
    model.eval()
    shader.eval()

    # get fixed returns for neural_blendshapes
    precomputed_blendshapes = model.precompute_networks()
    # Create the mesh once

    mesh = Mesh(
        ict_facekit.expression_reference_verts().cpu().data,
        ict_facekit.faces.cpu().data,
        ict_facekit=ict_facekit,
        device=device,
    )
    mesh.compute_connectivity()
    # Obtain 'flame_camera' from the temporary dataset
    views_sample = get_flame_camera(args)


    handle_values = torch.zeros(53, device=device)
    handle_values2 = torch.zeros(10, device=device)


    # Initialize GUI
    app = gui.Application.instance
    app.initialize()

    window_width = 768+500  # 512 for image, 300 for sliders
    window_height = 768  # Adjust as needed
    window = gui.Application.instance.create_window("GUI", window_width, window_height)

    print('created window')

    em = window.theme.font_size
    spacing = int(np.round(0.25 * em))
    vspacing = int(np.round(0.5 * em))
    margins = gui.Margins(vspacing)

    # Create an ImageWidget to display rendered images
    image_widget = gui.ImageWidget()
    image_widget.frame = gui.Rect(500, 0, 768, 768)
    window.add_child(image_widget)

    # Create sliders panel
    panel = gui.CollapsableVert("Handle Activations", 0, gui.Margins(em, em, em, em))
    panel.frame = gui.Rect(0, 0, 500, window_height)
    window.add_child(panel)

    with open('./assets/mediapipe_name_to_indices.pkl', 'rb') as f:
        MEDIAPIPE_BLENDSHAPES = pickle.load(f)

    pose_names = ['rot_x', 'rot_y', 'rot_z', 'trans_x', 'trans_y', 'trans_z', 'scale', 'global_trans_x', 'global_trans_y', 'global_trans_z']


    preset_facs = {}
    preset_facs['happiness'] = ['cheekSquint_L', 'cheekSquint_R', 'mouthSmile_L', 'mouthSmile_R']
    preset_facs['sadness'] = ['browInnerUp_L', 'browInnerUp_R', 'browDown_L', 'browDown_R', 'mouthFrown_L', 'mouthFrown_R']
    preset_facs['disgust'] = ['noseSneer_L', 'noseSneer_R', 'browDown_L', 'browDown_R', 'mouthFrown_L', 'mouthFrown_R']
    preset_facs['surprise'] = ['browOuterUp_L', 'browOuterUp_R', 'browInnerUp_L', 'browInnerUp_R', 'jawOpen']
    preset_facs['left_wink'] = ['eyeBlink_L', 'mouthSmile_L']
    preset_facs['right_wink'] = ['eyeBlink_R', 'mouthSmile_R']
    preset_facs['smile2'] = ['browInnerUp_L',  'mouthSmile_R']

    handle_values_expression = torch.zeros(len(preset_facs.keys()), device=device)
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

    preset_names = list(preset_facs.keys())

    labels_expression = [gui.Label(exp) for exp in preset_facs.keys()]
    sliders_expression = [gui.Slider(gui.Slider.DOUBLE) for _ in range(len(preset_facs.keys()))]

    labels = [gui.Label(ict_facekit.expression_names[i]) for i in range(53)]
    labels2 = [gui.Label(pose_names[i]) for i in range(10)]
    sliders = [gui.Slider(gui.Slider.DOUBLE) for _ in range(53)]
    sliders2 = [gui.Slider(gui.Slider.DOUBLE) for _ in range(10)]


    def update_image():
        features = torch.cat([handle_values, handle_values2], dim=0).unsqueeze(0).to(device)  # Shape: [1, 62]
        print(features)
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
            rendered_image = dataset_util.rgb_to_srgb(rgb_pred.squeeze(0)).cpu().numpy()  # Shape: [H, W, 3]
            rendered_image = (np.clip(rendered_image, 0, 1) * 255).astype(np.uint8)

            # Ensure the numpy array is contiguous
            rendered_image = np.ascontiguousarray(rendered_image)

            rendered_image = cv2.resize(rendered_image, (768, 768))

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
    
    def create_fun3(i):
        def fun(value):
            print("call")
            handle_values_expression[i] = value
            # get related expression names
            # expression name
            print("VALUE", value)

            expression_name = preset_names[i]
            print("EXPRESSION NAME", expression_name)
            related_facs = preset_facs[expression_name]
            print("RELATED FACS", related_facs)
            related_facs_indices = [ict_facekit.expression_names.tolist().index(fac) for fac in related_facs]
            print("RELATED FACS INDICES", related_facs_indices)

            for n in range(len(sliders_expression)):
                if n != i:
                    sliders_expression[n].double_value = 0.0
            print("INIT expression sliders to 0")

            for n in range(handle_values.shape[0]):
                if n in related_facs_indices:
                    sliders[n].double_value = value
                    handle_values[n] = value
                else:
                    sliders[n].double_value = 0.0
                    handle_values[n] = 0.0
            print("UPDATE expression sliders")
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

    for i, slider in enumerate(sliders_expression):
        slider.set_limits(0, 1)
        slider.double_value = 0.0
        slider.set_on_value_changed(create_fun3(i))

    sliders2[-4].set_limits(0, 2)
    sliders2[-4].double_value = 1

    for label, slider in zip(labels_expression, sliders_expression):
        fixed_prop_grid.add_child(label)
        fixed_prop_grid.add_child(slider)
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