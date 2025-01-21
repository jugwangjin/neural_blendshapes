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
    print(lmk_face_idx)

    ICTmodel = ICTFaceKitTorch(npy_dir = './assets/ict_facekit_torch.npy', canonical = Path(args.input_dir) / 'ict_identity.npy').to(device)
    ICTmodel.eval()
    print(ICTmodel.neutral_mesh.shape, ICTmodel.expression_shape_modes.shape, ICTmodel.identity_shape_modes.shape)
    print(ICTmodel.num_expression, ICTmodel.num_identity)

    ICTmodel = ICTmodel.to(device)
    ICTmodel.eval()

    iterations = 3000


    # target json
    json_file = '/Bean/log/gwangjin/2024/nbshapes_comparisons/flare/marcel/MVI_1802/flame_params.json'
    import json
    with open(json_file, 'r') as f:
        target = json.load(f)
    print(target.keys())
    # print(target['frames'])
    # target['frames'] is a list of dict. 
    # the keys are: 'file_path', 'expression'.

    # I would like to build another dict. 
    # the key would be value of file_path (target['frames'][i]['file_path'])
    # the value would be the expression code (target['frames'][i]['expression'])
    target_dict = {}
    for frame in target['frames']:
        target_dict[frame['file_path']] = frame['expression']
    # print(target_dict.keys())
    formatted_expression = [f"{x:.4f}" for x in target_dict['./image/33']]
    print(formatted_expression)
    # exit()
    import open3d as o3d
    flame_happy_verts, _, _ = FLAMEServer(expression_params=torch.tensor(target_dict['./image/33'])[None].to(device), full_pose=torch.zeros_like(FLAMEServer.canonical_pose))
    flame_happy_o3d_mesh = o3d.geometry.TriangleMesh()
    flame_happy_o3d_mesh.vertices = o3d.utility.Vector3dVector(flame_happy_verts.cpu().data.numpy()[0])
    flame_happy_o3d_mesh.triangles = o3d.utility.Vector3iVector(FLAMEServer.faces_tensor.cpu().data.numpy())
    o3d.io.write_triangle_mesh('debug/flame_happy_mesh_explicit.obj', flame_happy_o3d_mesh)

    # also, save first 3 axis of expression coefficient of flame. 

    # os.makedirs('figures/axis', exist_ok=True)
    os.makedirs('debug/flame_pose', exist_ok=True)  

    expression_params = torch.zeros(1, FLAMEServer.n_exp).to(device)
    pose_params = torch.zeros(1, FLAMEServer.canonical_pose.shape[1]).to(device)

    for i in range(pose_params.shape[1]):
        pose_params = pose_params * 0
        pose_params[0, i] = 1
        flame_happy_verts, _, _ = FLAMEServer(expression_params=expression_params, full_pose=pose_params)
        flame_happy_o3d_mesh = o3d.geometry.TriangleMesh()
        flame_happy_o3d_mesh.vertices = o3d.utility.Vector3dVector(flame_happy_verts.cpu().data.numpy()[0])
        flame_happy_o3d_mesh.triangles = o3d.utility.Vector3iVector(FLAMEServer.faces_tensor.cpu().data.numpy())
        o3d.io.write_triangle_mesh(f'debug/flame_pose/flame_pose_{i}.obj', flame_happy_o3d_mesh)

    exit()

    # # expression_params[0, i] = 5.0
    # flame_happy_verts, _, _ = FLAMEServer(expression_params=expression_params, full_pose=torch.zeros_like(FLAMEServer.canonical_pose))
    # flame_happy_o3d_mesh = o3d.geometry.TriangleMesh()
    # flame_happy_o3d_mesh.vertices = o3d.utility.Vector3dVector(flame_happy_verts.cpu().data.numpy()[0])
    # flame_happy_o3d_mesh.triangles = o3d.utility.Vector3iVector(FLAMEServer.faces_tensor.cpu().data.numpy())
    # o3d.io.write_triangle_mesh(f'figures/axis/flame_neutral.obj', flame_happy_o3d_mesh)
    for i in range(8):
        expression_params = torch.zeros(1, FLAMEServer.n_exp).to(device)
        expression_params[0, i] = 5.0
        flame_happy_verts, _, _ = FLAMEServer(expression_params=expression_params, full_pose=torch.zeros_like(FLAMEServer.canonical_pose))
        flame_happy_o3d_mesh = o3d.geometry.TriangleMesh()
        flame_happy_o3d_mesh.vertices = o3d.utility.Vector3dVector(flame_happy_verts.cpu().data.numpy()[0])
        flame_happy_o3d_mesh.triangles = o3d.utility.Vector3iVector(FLAMEServer.faces_tensor.cpu().data.numpy())
        o3d.io.write_triangle_mesh(f'figures/axis/flame_coeff_axis_{i}.obj', flame_happy_o3d_mesh)

    # expression_weights = torch.zeros(1, 53).to(device)
    # # expression_weights[0, i] = 1
    # ict_happy_facekit_mesh = ICTmodel(expression_weights=expression_weights)
    # o3d_happy_ict = o3d.geometry.TriangleMesh()
    # o3d_happy_ict.vertices = o3d.utility.Vector3dVector(ict_happy_facekit_mesh.cpu().data.numpy()[0])
    # o3d_happy_ict.triangles = o3d.utility.Vector3iVector(ICTmodel.faces.cpu().data.numpy())
    # o3d.io.write_triangle_mesh(f'figures/axis/ict_facekit_neutral.obj', o3d_happy_ict)
    # for i in range(53):
    #     expression_weights = torch.zeros(1, 53).to(device)
    #     expression_weights[0, i] = 1
    #     ict_happy_facekit_mesh = ICTmodel(expression_weights=expression_weights)
    #     o3d_happy_ict = o3d.geometry.TriangleMesh()
    #     o3d_happy_ict.vertices = o3d.utility.Vector3dVector(ict_happy_facekit_mesh.cpu().data.numpy()[0])
    #     o3d_happy_ict.triangles = o3d.utility.Vector3iVector(ICTmodel.faces.cpu().data.numpy())
    #     o3d.io.write_triangle_mesh(f'figures/axis/ict_facekit_{i}_{ICTmodel.expression_names.tolist()[i]}.obj', o3d_happy_ict)
 

    exit()

    # ================================================
    # optimize ICT-FaceKit identity code
    # ================================================

    flame_neutral_verts, _, _ = FLAMEServer(expression_params=torch.zeros(1, FLAMEServer.n_exp, device=device), full_pose=torch.zeros_like(FLAMEServer.canonical_pose))
    flame_neutral_lmks = mesh_points_by_barycentric_coordinates(flame_neutral_verts[0].cpu().data.numpy(), FLAMEServer.faces_tensor.cpu().data.numpy(), lmk_face_idx, lmk_b_coords)
    flame_neutral_lmks = torch.from_numpy(np.array(flame_neutral_lmks).astype(np.float32)).to(device)

    print(flame_neutral_lmks)

    idt_code = torch.nn.Parameter(torch.zeros(ICTmodel.num_identity).to(device))

    zero_exp = torch.zeros(ICTmodel.num_expression).to(device)

    import tqdm
    # pbar = tqdm.tqdm(range(iterations))

    # load assets/flame_canonical.obj
    flame_optimized_mesh = trimesh.load_mesh('./assets/flame_canonical.obj')

    print(flame_optimized_mesh)

    # fill grey color on every vertices on flame_optimized_mesh 
    flame_optimized_mesh.visual.vertex_colors = np.array([[0.5, 0.5, 0.5, 1.0] for _ in range(len(flame_optimized_mesh.vertices))])
    # fill red on the vertices in landmark faces
    print(lmk_face_idx)
    print(flame_optimized_mesh.faces[lmk_face_idx])
    flame_optimized_mesh.visual.vertex_colors[flame_optimized_mesh.faces[lmk_face_idx].reshape(-1)] = [0.99, 0.0, 0.0, 1.0]  
    
    # save the mesh to debug 
    # flame_optimized_mesh.export('debug/flame_optimized_landmarks.obj')

    # make pointcloud that is placed on flame_neutral_lmks, color them green, save it on open3d
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(flame_neutral_lmks.cpu().data.numpy())
    pcd.colors = o3d.utility.Vector3dVector(np.array([[0.0, 0.99, 0.0] for _ in range(len(flame_neutral_lmks))]))
    # o3d.io.write_point_cloud('debug/flame_neutral_landmarks.ply', pcd)


    # back to ict.
    # do same thing with ictmodel
    ict_landmark_indices = ICTmodel.landmark_indices
    
    # load assets/ict_canonical.obj
    ict_optimized_mesh = trimesh.load_mesh('./assets/ict_canonical.obj')

    print(ict_optimized_mesh)

    # fill grey color on every vertices on ict_optimized_mesh
    ict_optimized_mesh.visual.vertex_colors = np.array([[0.5, 0.5, 0.5, 1.0] for _ in range(len(ict_optimized_mesh.vertices))])
    # fill red on the vertices in landmark vertices
    print(ict_landmark_indices)

    ict_optimized_mesh.visual.vertex_colors[ict_landmark_indices] = [0.99, 0.0, 0.0, 1.0]
    # ict_optimized_mesh.export('debug/ict_optimized_landmarks.obj')

    # now for non-rigid registration
    # from flame, to ict.
    source_mesh = flame_optimized_mesh
    target_geometry = ict_optimized_mesh
    source_landmarks = lmk_face_idx, lmk_b_coords
    target_positions = ict_optimized_mesh.vertices[ict_landmark_indices][17:]
    assert len(lmk_face_idx) == len(lmk_b_coords)
    # exit()

    registered_flame = trimesh.registration.nricp_sumner(source_mesh, target_geometry, source_landmarks, target_positions, steps=None, distance_threshold=0.1, 
    return_records=False, use_faces=True, use_vertex_normals=True, neighbors_count=8, face_pairs_type='vertex')
    registered_flame_sumner = registered_flame
    # export it 
    registered_flame_mesh = trimesh.Trimesh(vertices=registered_flame, faces=source_mesh.faces)
    # color the landmarks, as before 
    registered_flame_mesh.visual.vertex_colors = np.array([[0.5, 0.5, 0.5, 1.0] for _ in range(len(registered_flame_mesh.vertices))])
    registered_flame_mesh.visual.vertex_colors[registered_flame_mesh.faces[lmk_face_idx].reshape(-1)] = [0.99, 0.0, 0.0, 1.0]
    

    # registered_flame_mesh.export('debug/registered_flame_to_ict.obj')


    # chamfer distance from registered flame to ict
    registered_flame_pcd = o3d.geometry.PointCloud()
    registered_flame_pcd.points = o3d.utility.Vector3dVector(registered_flame)
    registered_flame_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    ict_pcd = o3d.geometry.PointCloud()
    ict_pcd.points = o3d.utility.Vector3dVector(ict_optimized_mesh.vertices)
    ict_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    chamfer_dist_sumner = o3d.pipelines.registration.evaluate_registration(registered_flame_pcd, ict_pcd, 0.1)
    print(chamfer_dist_sumner)



    # trimesh.registration.nricp_amberg(source_mesh, target_geometry, source_landmarks=None, target_positions=None, steps=None, distance_threshold=0.1, return_records=False, use_faces=True, use_vertex_normals=True, neighbors_count=8, face_pairs_type='vertex')

    # do same thing with trimesh.registration.nricp_amberg
    registered_flame = trimesh.registration.nricp_amberg(source_mesh, target_geometry, source_landmarks, target_positions, steps=None, eps=0.00001, gamma=0.1, distance_threshold=0.1, return_records=False, use_faces=True, use_vertex_normals=True, neighbors_count=8)
    registered_flame_amberg = registered_flame
    # export it
    registered_flame_mesh = trimesh.Trimesh(vertices=registered_flame, faces=source_mesh.faces)
    # color the landmarks, as before
    registered_flame_mesh.visual.vertex_colors = np.array([[0.5, 0.5, 0.5, 1.0] for _ in range(len(registered_flame_mesh.vertices))])
    registered_flame_mesh.visual.vertex_colors[registered_flame_mesh.faces[lmk_face_idx].reshape(-1)] = [0.99, 0.0, 0.0, 1.0]
# 
    # registered_flame_mesh.export('debug/registered_flame_to_ict_amberg.obj')

    # chamfer distance from registered flame to ict
    registered_flame_pcd = o3d.geometry.PointCloud()
    registered_flame_pcd.points = o3d.utility.Vector3dVector(registered_flame)
    registered_flame_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    ict_pcd = o3d.geometry.PointCloud()
    ict_pcd.points = o3d.utility.Vector3dVector(ict_optimized_mesh.vertices)
    ict_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    chamfer_dist_amberg = o3d.pipelines.registration.evaluate_registration(registered_flame_pcd, ict_pcd, 0.1)
    print(chamfer_dist_amberg)

    # choose lower inlier_rmse as final registered mesh. 
    registered_flame = registered_flame_amberg if chamfer_dist_amberg.inlier_rmse < chamfer_dist_sumner.inlier_rmse else registered_flame_sumner
    print(registered_flame, 'amberg' if chamfer_dist_amberg.inlier_rmse < chamfer_dist_sumner.inlier_rmse else 'sumner')
# 


    happiness = ['cheekSquint_L', 'cheekSquint_R', 'mouthSmile_L', 'mouthSmile_R']

    # 1. get happy ict facekit face 
    ict_facekit_happy_code = torch.zeros(1, 53)
    for h in happiness:
        ict_facekit_happy_code[0, ICTmodel.expression_names.tolist().index(h)] = 0.7

    ict_facekit_happy_code = ict_facekit_happy_code.to(device)

    happy_ict_facekit_mesh = ICTmodel(expression_weights=ict_facekit_happy_code) # torch.cuda.tensor shape of (1, V, 3)

    # save it as obj using o3d
    o3d_happy_ict = o3d.geometry.TriangleMesh()
    o3d_happy_ict.vertices = o3d.utility.Vector3dVector(happy_ict_facekit_mesh.cpu().data.numpy()[0])
    o3d_happy_ict.triangles = o3d.utility.Vector3iVector(ICTmodel.faces.cpu().data.numpy())
    o3d.io.write_triangle_mesh('debug/happy_ict_facekit.obj', o3d_happy_ict)



    # need to separate eyeball indices. 

    mesh = o3d.io.read_triangle_mesh("assets/canonical_eye_smpl.obj")
    # mesh = o3d.io.read_triangle_mesh("debug_view_gt_mesh_0.obj")
    print(mesh)

    

    # print(mesh.cluster_connected_triangles())  
    clustered = mesh.cluster_connected_triangles()

    cluster_idx_for_traingle, triangles_per_cluster, surface_area_per_cluster = clustered

    cluster_idx_for_vertices = np.zeros(np.asarray(mesh.vertices).shape[0])


    print(triangles_per_cluster)
    # print the index of maximum element in triangles_per_cluster
    print(np.argmax(triangles_per_cluster))



    for num, cluster in enumerate(cluster_idx_for_traingle):
        vertices_for_triangle = mesh.triangles[num]
        for v in vertices_for_triangle:
            cluster_idx_for_vertices[v] = cluster
    # vertices for eyes: not np.argmax(triangles_per_cluster) 
    # vertices for face: np.argmax(triangles_per_cluster)
    # print(cluster_idx_for_vertices)

    face_cluster = np.argmax(triangles_per_cluster)        
    
    vertices_for_eyes = np.where(cluster_idx_for_vertices != face_cluster)[0]
    vertices_for_face = np.where(cluster_idx_for_vertices == face_cluster)[0]

    print(vertices_for_eyes, vertices_for_face)
    print(np.unique(cluster_idx_for_vertices))
    print(np.unique(cluster_idx_for_traingle))
    

    tight_face_index = 6705
    face_index = 9409     
    head_index = 14062
    socket_index = 11248
    head_index=11248

    # from ict_model mesh, we use first {socket_index} vertices. 
    usable_vertices_ict = ict_optimized_mesh.vertices[:socket_index]
    usable_vertices_flame = registered_flame

    # distance matrix using torch
    usable_vertices_ict = torch.tensor(usable_vertices_ict).to(device) # shape of (socket_index, 3)
    usable_vertices_flame_ = torch.tensor(usable_vertices_flame).to(device) # shape of (V, 3)

    usable_vertices_flame_[vertices_for_eyes] += 1e3

    # distance matrix
    difference = usable_vertices_ict[:, None] - usable_vertices_flame_[None]
    distance_matrix = torch.norm(difference, dim=-1)

    # get (V) length of indices, as closest vertex indices
    closest_vertex_indices = torch.argmin(distance_matrix, dim=1)
    flame_to_ict_closest_vertex_indices = torch.argmin(distance_matrix, dim=0)
    print(closest_vertex_indices)
    print(difference.shape, distance_matrix.shape, closest_vertex_indices.shape)


    flame_closest_vertices = [[] for _ in range(usable_vertices_flame_.shape[0])]
    for i in range(closest_vertex_indices.shape[0]):
        flame_closest_vertices[closest_vertex_indices[i]].append(i) 
    
    closest_ict = []
    closest_flame = []
    num_non_empty = 0
    # now make list of pair for non-empty flame_closest_vertices
    for idx, li in enumerate(flame_closest_vertices):
        if len(li) == 0:
            continue
        num_non_empty += 1
        # print(f"flame vertex {idx} is closest to ict vertices {li}")
        # now find THE closest on in li 
        min_dist = 1e9
        min_idx = -1
        for i in li:
            dist = distance_matrix[i, idx]
            # dist = torch.norm(usable_vertices_ict[i] - usable_vertices_flame[idx])
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        closest_ict.append(min_idx)
        closest_flame.append(idx)
        # print(f"flame vertex {idx} is closest to ict vertex {min_idx}")


    usable_vertices_ict = usable_vertices_ict.cpu().data.numpy()
    usable_vertices_flame = usable_vertices_flame
    closest_vertex_indices = closest_vertex_indices.cpu().data.numpy()


    
    # 2. get happy flame face
    # we have flame to ict mapping. 
    # optimize flame expression code

    happy_ict_to_flame = happy_ict_facekit_mesh[:, closest_ict]

    flame_expression = torch.nn.Parameter(torch.zeros(1, FLAMEServer.n_exp).to(device))
    flame_pose = torch.zeros_like(FLAMEServer.canonical_pose).to(device)
    optimizer = torch.optim.SGD([flame_expression], lr=1e2)
    
    with torch.no_grad():
        flame_verts, _, _ = FLAMEServer(expression_params=flame_expression, full_pose=flame_pose)
        flame_happy_o3d_mesh = o3d.geometry.TriangleMesh()
        flame_happy_o3d_mesh.vertices = o3d.utility.Vector3dVector(flame_verts.cpu().data.numpy()[0])
        flame_happy_o3d_mesh.triangles = o3d.utility.Vector3iVector(FLAMEServer.faces_tensor.cpu().data.numpy())
        o3d.io.write_triangle_mesh('debug/flame_before_optim.obj', flame_happy_o3d_mesh)
        
        # we have same number of vertices between happy_ict_to_flame and flame_verts. 
        # draw lineset on open3d to visualize. 
        lineset_verts = np.concatenate([happy_ict_to_flame.cpu().data.numpy()[0], flame_verts[:, closest_flame].cpu().data.numpy()[0]], axis=0)
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(lineset_verts)
        lineset.lines = o3d.utility.Vector2iVector(np.array([[i, i + len(happy_ict_to_flame.cpu().data.numpy()[0])] for i in range(len(happy_ict_to_flame.cpu().data.numpy()[0]) - 1)]))
        lineset.colors = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 1.0] for _ in range(len(lineset.lines))]))

        o3d.io.write_line_set('debug/lineset_flame_happy_set.ply', lineset)

                                               
    # o3d.visualization.draw_geometries([lineset])

    # save pointcloud of happy_ict_to_flame
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(happy_ict_to_flame.cpu().data.numpy()[0])
    pcd.colors = o3d.utility.Vector3dVector(np.array([[0.0, 0.99, 0.0] for _ in range(len(happy_ict_to_flame.cpu().data.numpy()[0]))]))
    o3d.io.write_point_cloud('debug/happy_ict_to_flame.ply', pcd)

    iterations = 10000
    import tqdm
    pbar = tqdm.tqdm(range(iterations))
    for i in pbar:
        if iterations == 4000:
            # reduce lr by 10
            for g in optimizer.param_groups:
                g['lr'] = 1e1
        if iterations == 8000:
            # reduce lr by 10
            for g in optimizer.param_groups:
                g['lr'] = 1e-0

        flame_verts, _, _ = FLAMEServer(expression_params=flame_expression, full_pose=flame_pose)

        loss = ((flame_verts[:, closest_flame] - happy_ict_to_flame)*10).pow(2).mean()

        reg_loss = flame_expression.pow(2).mean() * 1e-3

        pbar.set_description(f"Loss: {loss.item():.4f}, Reg Loss: {reg_loss.item():.4f}")

        if i % 100 == 0:
            print(f"Loss: {loss.item():.4f}, Reg Loss: {reg_loss.item():.4f}")

        loss += reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    

    # save the flame expression code
    np.save('debug/flame_happy_expression.npy', flame_expression.cpu().data.numpy())
    print(flame_expression)
    print(flame_pose)
    # save mesh
    flame_happy_mesh = FLAMEServer(expression_params=flame_expression, full_pose=FLAMEServer.canonical_pose)[0]
    flame_happy_o3d_mesh = o3d.geometry.TriangleMesh()
    flame_happy_o3d_mesh.vertices = o3d.utility.Vector3dVector(flame_happy_mesh.cpu().data.numpy()[0])
    flame_happy_o3d_mesh.triangles = o3d.utility.Vector3iVector(FLAMEServer.faces_tensor.cpu().data.numpy())
    o3d.io.write_triangle_mesh('debug/flame_happy_mesh.obj', flame_happy_o3d_mesh)

    # exit()

if __name__ == '__main__':
    
    parser = config_parser()
    args = parser.parse_args()

    main(args)
