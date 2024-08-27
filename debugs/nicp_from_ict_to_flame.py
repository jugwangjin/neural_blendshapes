# what I need
# import trimesh
# use trimesh.registration.nricp_amberg or trimesh.registeration.nricp_sumner

# import flame, ICT-FAcekit

import os
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"
# os.environ['PYOPENGL_PLATFORM'] = 'egl'

from flare.dataset import *
from flame.FLAME import FLAME
from arguments import config_parser
import torch
# from flame.FLAME import FLAME
# from flare.dataset import *

# from flare.dataset import dataset_util
import numpy as np
import pickle
import os
import chumpy as ch

# from ICT_FaceKit.Scripts import face_model_io
from flare.utils.ict_model import ICTFaceKitTorch

from pathlib import Path

from pytorch3d import ops
import pytorch3d.loss

import open3d as o3d

# Offscreen Renderer
import trimesh
# import pyrender

import os



ICT_LMK_IDX = [36, 39, 42, 45, 30, 48, 54]
FLAME_LMK_IDX = [19, 22, 25, 28, 13, 31, 37]

def load_binary_pickle( filepath ):
    with open(filepath, 'rb') as f:
        data = pickle.load(f, encoding="latin1")
    return data

def mesh_points_by_barycentric_coordinates( mesh_verts, mesh_faces, lmk_face_idx, lmk_b_coords ):
    """ function: evaluation 3d points given mesh and landmark embedding
    """
    dif1 = ch.vstack([(mesh_verts[mesh_faces[lmk_face_idx], 0] * lmk_b_coords).sum(axis=1),
                    (mesh_verts[mesh_faces[lmk_face_idx], 1] * lmk_b_coords).sum(axis=1),
                    (mesh_verts[mesh_faces[lmk_face_idx], 2] * lmk_b_coords).sum(axis=1)]).T
    return dif1

def mesh_points_by_barycentric_coordinates_batch(mesh_verts, mesh_faces, lmk_face_idx, lmk_b_coords):
    """ function: evaluation 3d points given mesh and landmark embedding
    """
    lmk_face_idx = torch.tensor(lmk_face_idx).to(mesh_verts.device)
    lmk_b_coords = torch.tensor(lmk_b_coords).to(mesh_verts.device)

    # print(mesh_verts.shape, mesh_faces.shape, lmk_face_idx.shape, lmk_b_coords.shape)
    # print(mesh_verts[:, mesh_faces[lmk_face_idx]].shape)
    dif1 = torch.stack([(mesh_verts[:, mesh_faces[lmk_face_idx]][..., 0] * lmk_b_coords[None]).sum(dim=-1),
                        (mesh_verts[:, mesh_faces[lmk_face_idx]][..., 1] * lmk_b_coords[None]).sum(dim=-1),
                        (mesh_verts[:, mesh_faces[lmk_face_idx]][..., 2] * lmk_b_coords[None]).sum(dim=-1)], dim=2)
    # print(dif1.shape)
    # exit()
    return dif1

def load_embedding( file_path ):
    """ funciton: load landmark embedding, in terms of face indices and barycentric coordinates for corresponding landmarks
    note: the included example is corresponding to CMU IntraFace 49-point landmark format.
    """
    lmk_indexes_dict = load_binary_pickle( file_path )
    lmk_face_idx = lmk_indexes_dict[ 'lmk_face_idx' ].astype( np.uint32 )
    lmk_b_coords = lmk_indexes_dict[ 'lmk_b_coords' ]
    # print shapes
    print( lmk_face_idx.shape, lmk_b_coords.shape )
    return lmk_face_idx, lmk_b_coords

def main(args):
    device = torch.device('cpu')
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    print(f"Using device {device}")
    
    # dataset_train    = DatasetLoader_FACS(args, train_dir=args.train_dir, sample_ratio=1, pre_load=False)
    dataset_train    = DatasetLoader(args, train_dir=args.train_dir, sample_ratio=1, pre_load=False, train=True)
    # dataloader_train    = torch.utils.data.DataLoader(dataset_train, batch_size=16, collate_fn=dataset_train.collate, shuffle=True, drop_last=True)

    flame_path = './flame/FLAME2020/generic_model.pkl'
    flame_shape = dataset_train.shape_params
    FLAMEServer = FLAME(flame_path, n_shape=100, n_exp=50, shape_params=flame_shape, use_processed_faces=False).to(device)

    print(FLAMEServer.faces_tensor.shape, FLAMEServer.v_template.shape)
    ## ============== canonical with mouth open (jaw pose 0.4) ==============================
    FLAMEServer.canonical_exp = (dataset_train.get_mean_expression()).to(device)
    FLAMEServer.canonical_pose = FLAMEServer.canonical_pose.to(device)
    FLAMEServer.canonical_verts, FLAMEServer.canonical_pose_feature, FLAMEServer.canonical_transformations = \
        FLAMEServer(expression_params=FLAMEServer.canonical_exp, full_pose=FLAMEServer.canonical_pose)
    if args.ghostbone:
        FLAMEServer.canonical_transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).float().to(device), FLAMEServer.canonical_transformations], 1)
    FLAMEServer.canonical_verts = FLAMEServer.canonical_verts.to(device)

    load_embedding_path = './assets/flame_static_embedding.pkl'
    lmk_face_idx, lmk_b_coords = load_embedding(load_embedding_path)

    print(lmk_face_idx.shape)


    ICTmodel = ICTFaceKitTorch(npy_dir = './assets/ict_facekit_torch.npy', canonical = Path(args.input_dir) / 'ict_identity.npy').to(device)
    ICTmodel.eval()
    print(ICTmodel.neutral_mesh.shape, ICTmodel.expression_shape_modes.shape, ICTmodel.identity_shape_modes.shape)
    print(ICTmodel.num_expression, ICTmodel.num_identity)

    ICTmodel = ICTmodel.to(device)
    ICTmodel.eval()

    iterations = 3000

    mesh_save_dir = './debug/optimized_ict_and_flame_meshes/'
    os.makedirs(mesh_save_dir, exist_ok=True)
    # ================================================
    # optimize ICT-FaceKit identity code
    # ================================================

    flame_neutral_verts, _, _ = FLAMEServer(expression_params=torch.zeros(1, FLAMEServer.n_exp, device=device), full_pose=torch.zeros_like(FLAMEServer.canonical_pose))
    flame_neutral_lmks = mesh_points_by_barycentric_coordinates(flame_neutral_verts[0].cpu().data.numpy(), FLAMEServer.faces_tensor.cpu().data.numpy(), lmk_face_idx, lmk_b_coords)
    flame_neutral_lmks = torch.from_numpy(np.array(flame_neutral_lmks).astype(np.float32)).to(device)

    idt_code = torch.nn.Parameter(torch.zeros(ICTmodel.num_identity).to(device))

    zero_exp = torch.zeros(ICTmodel.num_expression).to(device)

    import tqdm
    pbar = tqdm.tqdm(range(iterations))


    # extract both landmarks 
    # 

# trimesh.registration.nricp_sumner(source_mesh, target_geometry, source_landmarks=None, target_positions=None, steps=None, distance_threshold=0.1, return_records=False, use_faces=True, use_vertex_normals=True, neighbors_count=8, face_pairs_type='vertex')
# Non Rigid Iterative Closest Points
# Non Rigid Iterative Closest Points

# Implementation of the correspondence computation part of “Sumner and Popovic 2004: Deformation Transfer for Triangle Meshes” Allows to register non-rigidly a mesh on another geometry.

# Comparison between nricp_amberg and nricp_sumner: * nricp_amberg fits to the target mesh in less steps * nricp_amberg can generate sharp edges (only vertices and their

# neighbors are considered)

# nricp_sumner tend to preserve more the original shape

# nricp_sumner parameters are easier to tune

# nricp_sumner solves for triangle positions whereas nricp_amberg solves for
# vertex transforms

# nricp_sumner is less optimized when wn > 0

# Parameters:
# source_mesh (Trimesh) – Source mesh containing both vertices and faces.

# target_geometry (Trimesh or PointCloud or (n, 3) float) – Target geometry. It can contain no faces or be a PointCloud.

# source_landmarks ((n,) int or ((n,) int, (n, 3) float)) – n landmarks on the the source mesh. Represented as vertex indices (n,) int. It can also be represented as a tuple of triangle indices and barycentric coordinates ((n,) int, (n, 3) float,).

# target_positions ((n, 3) float) – Target positions assigned to source landmarks

# steps (Core parameters of the algorithm) – Iterable of iterables (wc, wi, ws, wl, wn). wc is the correspondence term (strength of fitting), wi is the identity term (recommended value : 0.001), ws is smoothness term, wl weights the landmark importance and wn the normal importance.

# distance_threshold (float) – Distance threshold to account for a vertex match or not.

# return_records (bool) – If True, also returns all the intermediate results. It can help debugging and tune the parameters to match a specific case.

# use_faces (bool) – If True and if target geometry has faces, use proximity.closest_point to find matching points. Else use scipy’s cKDTree object.

# use_vertex_normals (bool) – If True and if target geometry has faces, interpolate the normals of the target geometry matching points. Else use face normals or estimated normals if target geometry has no faces.

# neighbors_count (int) – number of neighbors used for normal estimation. Only used if target geometry has no faces or if use_faces is False.

# face_pairs_type (str 'vertex' or 'edge') – Method to determine face pairs used in the smoothness cost. ‘vertex’ yields smoother results.

# Returns:
# result – The vertices positions of source_mesh such that it is registered non-rigidly onto the target geometry. If return_records is True, it returns the list of the vertex positions at each iteration.

# Return type:
# (n, 3) float or List[(n, 3) float]

    

if __name__ == '__main__':
    
    parser = config_parser()
    args = parser.parse_args()

    main(args)
