import os
import sys
import random
import numpy as np
import torch
import cv2
from tqdm import tqdm
from arguments import config_parser
from pathlib import Path
from flare.dataset import dataset_util

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(20241224)

@torch.no_grad()
def main(args, device):
    original_dir = os.getcwd()
    flare_path = os.path.join(args.model_dir, args.model_name, 'sources')
    sys.path.insert(0, flare_path)

    from flare.dataset import DatasetLoader
    from flare.utils.ict_model import ICTFaceKitTorch
    import nvdiffrec.render.light as light
    from flare.core import Mesh, Renderer
    from flare.modules import NeuralShader, get_neural_blendshapes
    from flare.utils import AABB, save_manipulation_image

    # Load dataset
    print("Loading views...")
    dataset_train = DatasetLoader(args, train_dir=args.train_dir, sample_ratio=args.sample_idx_ratio, pre_load=False, train=True)
    args.eval_dir = [args.video_name]
    dataset_val = DatasetLoader(args, train_dir=args.eval_dir, sample_ratio=1, pre_load=False)

    # Setup output directory
    output_dir = os.path.join(args.output_dir, args.run_name, args.output_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    # Load models
    ict_facekit = ICTFaceKitTorch(npy_dir='./assets/ict_facekit_torch.npy', canonical=Path(args.input_dir) / 'ict_identity.npy')
    ict_facekit = ict_facekit.to(device)

    ict_canonical_mesh = Mesh(ict_facekit.canonical[0].cpu().data, ict_facekit.faces.cpu().data, ict_facekit=ict_facekit, device=device)
    ict_canonical_mesh.compute_connectivity()
            
    bshapes_names = ict_facekit.expression_names.tolist()

    # Setup renderer
    aabb = AABB(ict_canonical_mesh.vertices.cpu().numpy())
    ict_mesh_aabb = [torch.min(ict_canonical_mesh.vertices, dim=0).values, torch.max(ict_canonical_mesh.vertices, dim=0).values]

    renderer = Renderer(device=device)
    renderer.set_near_far(dataset_train, torch.from_numpy(aabb.corners).to(device), epsilon=0.5)

    channels_gbuffer = ['mask', 'position', 'normal', "canonical_position"]
    print("Rasterizing:", channels_gbuffer)
    
    # Load neural blendshapes
    model_path = os.path.join(args.output_dir, args.run_name, 'stage_1', 'network_weights', 'neural_blendshapes_latest.pt')
    print("=="*50)
    print("Loading Deformer")
    face_normals = ict_canonical_mesh.get_vertices_face_normals(ict_facekit.neutral_mesh_canonical[0])[0]
    neural_blendshapes = get_neural_blendshapes(model_path=model_path, train=False, ict_facekit=ict_facekit, aabb=ict_mesh_aabb, face_normals=face_normals, device=device) 
    neural_blendshapes = neural_blendshapes.to(device)

    # Load shader
    lgt = light.create_env_rnd()    
    shader = NeuralShader.load(os.path.join(args.output_dir, args.run_name, 'stage_1', 'network_weights', 'shader_latest.pt'), device=device)

    # Define expression groups
    # Convert lists to tuples to make them hashable
    smiles = tuple(['mouthSmile_L', 'mouthSmile_R'])
    cheeks = tuple(['cheekPuff_L', 'cheekPuff_R', 'cheekSquint_L', 'cheekSquint_R'])
    eyeball_center = tuple(['eyeLookUp_L', 'eyeLookUp_R', 'eyeLookDown_L', 'eyeLookDown_R', 'eyeLookIn_L', 'eyeLookIn_R', 'eyeLookOut_L', 'eyeLookOut_R'])
    eyeblink = tuple(['eyeBlink_L', 'eyeBlink_R'])
    eyebrow_raisers = tuple(['browInnerUp_L', 'browInnerUp_R', 'browOuterUp_L', 'browOuterUp_R'])
    happiness = tuple(['cheekSquint_L', 'cheekSquint_R', 'mouthSmile_L', 'mouthSmile_R'])
    sadness = tuple(['browInnerUp_L', 'browInnerUp_R', 'browDown_L', 'browDown_R', 'mouthFrown_L', 'mouthFrown_R'])

    
    # Define target expressions for interpolation
    # Define target expressions for interpolation
    target_expressions = [
        ("Neutral", []),
        ("Smile", [("smiles", 0.5), ("cheeks", 0.5)]),
        ("Eyebrow Raise", [("eyebrow_raisers", 0.5)]),
        ("Cheek Puff", [("cheeks", 0.5)]),
        ("Sad", [("sadness", 0.5)]),
        ("Happy", [("happiness", 0.5)])
    ]


    
    # Get a reference frame
    reference_idx = 0
    reference_view = dataset_val.collate([dataset_val.__getitem__(reference_idx)])
    
    # Extract base features from reference frame
    base_features = neural_blendshapes.encoder(reference_view)
    base_facs = base_features[:, :53].clone() * 0  # Neutral expression
    base_translation = base_features[:, 53:56].clone() * 0  # Forward facing
    base_rotation = base_features[:, 56:59].clone() * 0  # Forward facing
    base_global_translation = base_features[:, 60:63].clone()
    
    # Video settings
    fps = 30
    transition_time = 1.0  # seconds
    hold_time = 1.5  # seconds
    frames_per_transition = int(fps * transition_time)
    frames_per_hold = int(fps * hold_time)
    
    # Create video writer
    video_path = os.path.join(output_dir, f'{args.run_name}_{args.video_name}_expressions.mp4')
    temp_dir = os.path.join(output_dir, 'temp_frames')
    os.makedirs(temp_dir, exist_ok=True)
    
    frame_count = 0
    print("Generating expression animation frames...")
    
    # For each target expression
    for i in range(len(target_expressions)):
        current_expr_name, current_expr_dict = target_expressions[i]
        print(f"Processing expression: {current_expr_name}")
        
        # Create target FACS
        target_facs = base_facs.clone()
        
        # Apply expression
        # Apply expression
        for expr_type, intensity in current_expr_dict:
            if expr_type == "smiles":
                for name in smiles:
                    if name in bshapes_names:
                        idx = bshapes_names.index(name)
                        target_facs[:, idx] = intensity
            elif expr_type == "cheeks":
                for name in cheeks:
                    if name in bshapes_names:
                        idx = bshapes_names.index(name)
                        target_facs[:, idx] = intensity
            elif expr_type == "eyeball_center":
                for name in eyeball_center:
                    if name in bshapes_names:
                        idx = bshapes_names.index(name)
                        target_facs[:, idx] = intensity
            elif expr_type == "eyeblink":
                for name in eyeblink:
                    if name in bshapes_names:
                        idx = bshapes_names.index(name)
                        target_facs[:, idx] = intensity
            elif expr_type == "eyebrow_raisers":
                for name in eyebrow_raisers:
                    if name in bshapes_names:
                        idx = bshapes_names.index(name)
                        target_facs[:, idx] = intensity
            elif expr_type == "happiness":
                for name in happiness:
                    if name in bshapes_names:
                        idx = bshapes_names.index(name)
                        target_facs[:, idx] = intensity
            elif expr_type == "sadness":
                for name in sadness:
                    if name in bshapes_names:
                        idx = bshapes_names.index(name)
                        target_facs[:, idx] = intensity



    # Add other expression types similarly


        
        # Transition from neutral to expression
        for t in tqdm(range(frames_per_transition)):
            alpha = t / frames_per_transition
            current_facs = base_facs * (1 - alpha) + target_facs * alpha
            
            # Create features
            features = base_features.clone()
            features[:, :53] = current_facs
            features[:, 53:56] = base_rotation
            features[:, 56:59] = base_translation
            features[:, 60:63] = base_global_translation
            
            # Render frame
            render_and_save_frame(features, neural_blendshapes, ict_canonical_mesh, renderer, 
                                 shader, reference_view, lgt, temp_dir, frame_count, args)
            frame_count += 1
        
        # Hold expression
        for t in tqdm(range(frames_per_hold)):
            # Create features
            features = base_features.clone()
            features[:, :53] = target_facs
            features[:, 53:56] = base_rotation
            features[:, 56:59] = base_translation
            features[:, 60:63] = base_global_translation
            
            # Render frame
            render_and_save_frame(features, neural_blendshapes, ict_canonical_mesh, renderer, 
                                 shader, reference_view, lgt, temp_dir, frame_count, args)
            frame_count += 1
        
        # Transition back to neutral
        for t in tqdm(range(frames_per_transition)):
            alpha = t / frames_per_transition
            current_facs = target_facs * (1 - alpha) + base_facs * alpha
            
            # Create features
            features = base_features.clone()
            features[:, :53] = current_facs
            features[:, 53:56] = base_rotation
            features[:, 56:59] = base_translation
            features[:, 60:63] = base_global_translation
            
            # Render frame
            render_and_save_frame(features, neural_blendshapes, ict_canonical_mesh, renderer, 
                                 shader, reference_view, lgt, temp_dir, frame_count, args)
            frame_count += 1
    
    # Create video from frames
    print("Creating video from frames...")
    create_video_from_frames(temp_dir, video_path, fps)
    
    # Clean up temp frames
    if os.path.exists(video_path):
        import shutil
        shutil.rmtree(temp_dir)
        print(f"Video saved to: {video_path}")

def render_and_save_frame(features, neural_blendshapes, ict_canonical_mesh, renderer, 
                         shader, views_subset, lgt, output_dir, frame_idx, args):
    convert_uint = lambda x: np.clip(np.rint(dataset_util.rgb_to_srgb(x).detach().numpy() * 255.0), 0, 255).astype(np.uint8) 
    convert_uint_255 = lambda x: (x * 255).to(torch.uint8)
    """Render a frame and save it to disk"""
    return_dict = neural_blendshapes(features=features)
    deformed_vertices = return_dict['expression_mesh_posed']
    d_normals = ict_canonical_mesh.fetch_all_normals(deformed_vertices, ict_canonical_mesh)

    channels_gbuffer = ['mask', 'position', 'normal', "canonical_position"]
    gbuffers = renderer.render_batch(views_subset['flame_camera'], deformed_vertices.contiguous(), d_normals,
                                channels=channels_gbuffer, with_antialiasing=True, 
                                canonical_v=ict_canonical_mesh.vertices, canonical_idx=ict_canonical_mesh.indices, 
                                canonical_uv=neural_blendshapes.ict_facekit.uv_neutral_mesh,
                                mesh=ict_canonical_mesh)
    
    pred_color_masked, cbuffers, gbuffer_mask = shader.shade(gbuffers, views_subset, ict_canonical_mesh, args.finetune_color, lgt)
    mask = gbuffer_mask[0].cpu()
    
    # Convert to image and save - fixed version
    # img = pred_color_masked[0].permute(1, 2, 0).cpu().numpy()
    # img = (img * 255).astype(np.uint8)
    img = convert_uint(torch.cat([pred_color_masked[0].cpu(), mask], dim=-1).cpu())
    cv2.imwrite(os.path.join(output_dir, f'frame_{frame_idx:04d}.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def create_video_from_frames(frames_dir, output_path, fps):
    """Create a video from a directory of frames"""
    frames = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.png')])
    if not frames:
        print("No frames found!")
        return
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(frames[0])
    height, width, _ = first_frame.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Add frames to video
    for frame_path in frames:
        video.write(cv2.imread(frame_path))
    
    video.release()

if __name__ == '__main__':
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='/Bean/log/gwangjin/2024/nbshapes_comparisons/ours_enc_v13', help='Path to the trained model')
    parser.add_argument('--model_name', type=str, default='marcel', help='Name of the run in model_dir')
    parser.add_argument('--video_name', type=str, default='MVI_1802', help='Name of the video in dataset')
    parser.add_argument('--output_dir_name', type=str, default='expression_animation', help='Path to save the results')
    args = parser.parse_args()

    config_file = os.path.join(args.model_dir, args.model_name, 'sources', 'config.txt')
    
    parser = config_parser()
    args2 = parser.parse_args(['--config', config_file])
    args2.video_name = args.video_name
    args2.run_name = args.model_name
    args2.output_dir = args.model_dir
    args2.output_dir_name = args.output_dir_name
    args2.model_dir = args.model_dir
    args2.model_name = args.model_name
    args2.train_deformer = False
    
    print(f"Input dir: {args2.input_dir}, Video: {args2.video_name}")
    print(f"Output dir: {args2.output_dir}, Run name: {args2.run_name}")

    # Select the device
    device = torch.device('cpu')
    if torch.cuda.is_available() and args2.device >= 0:
        device = torch.device(f'cuda:{args2.device}')
    print(f"Using device {device}")

    main(args2, device)
