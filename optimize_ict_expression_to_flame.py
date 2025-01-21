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

    mesh_save_dir = './debug/optimized_ict_and_flame_meshes_with_shapes_FLAME_exaggerated_less_lr/'
    os.makedirs(mesh_save_dir, exist_ok=True)
    # ================================================
    # optimize ICT-FaceKit identity code
    # ================================================

    flame_neutral_verts, _, _ = FLAMEServer(expression_params=torch.zeros(1, FLAMEServer.n_exp, device=device), full_pose=FLAMEServer.canonical_pose)
    flame_neutral_lmks = mesh_points_by_barycentric_coordinates(flame_neutral_verts[0].cpu().data.numpy(), FLAMEServer.faces_tensor.cpu().data.numpy(), lmk_face_idx, lmk_b_coords)
    flame_neutral_lmks = torch.from_numpy(np.array(flame_neutral_lmks).astype(np.float32)).to(device)

    idt_code = torch.nn.Parameter(torch.zeros(ICTmodel.num_identity).to(device))
    jaw_open = torch.nn.Parameter(torch.zeros(1).to(device)+0.75)

    zero_exp = torch.zeros(ICTmodel.num_expression).to(device)

    import tqdm
    pbar = tqdm.tqdm(range(iterations))


    ict_mesh = ICTmodel(expression_weights=zero_exp[None], identity_weights=idt_code[None], to_canonical=False,)
    ict_lmk = ict_mesh[0, ICTmodel.landmark_indices][17:]

    ict_lmk_similarity_transform = ops.corresponding_points_alignment(ict_lmk[None], flame_neutral_lmks[None], estimate_scale=True)
    
    s = ict_lmk_similarity_transform.s.data.detach()
    R = ict_lmk_similarity_transform.R.data.detach()
    T = ict_lmk_similarity_transform.T.data.detach()

    idt_optimizer = torch.optim.AdamW([
        {'params': idt_code, 'lr': 1e-2},
        {'params': jaw_open, 'lr': 1e-3},
        # {'params': [s, R, T], 'lr': 1e-4}
    ], weight_decay=1e-4)

    for i in pbar:
        exp = zero_exp.detach()
        exp[ICTmodel.expression_names.tolist().index('jawOpen')] = jaw_open
        ict_mesh = ICTmodel(expression_weights=exp[None], identity_weights=idt_code[None], to_canonical=False,)
        ict_lmk = ict_mesh[0, ICTmodel.landmark_indices][17:]

        ict_lmk_similarity_transform = ops.corresponding_points_alignment(ict_lmk[None], flame_neutral_lmks[None], estimate_scale=True)
        
        rotated_ict_lmk = torch.einsum('bni, bji -> bnj', ict_lmk[None], ict_lmk_similarity_transform.R)
        aligned_ict_lmk = (ict_lmk_similarity_transform.s * rotated_ict_lmk + ict_lmk_similarity_transform.T)[0]

        rotated_ict_mesh = torch.einsum('bni, bji -> bnj', ict_mesh, ict_lmk_similarity_transform.R)
        aligned_ict_mesh = (ict_lmk_similarity_transform.s * rotated_ict_mesh + ict_lmk_similarity_transform.T)

        loss, loss_normals = pytorch3d.loss.chamfer_distance(aligned_ict_mesh, flame_neutral_verts, single_directional=True)
        loss = torch.mean(loss)
        loss += torch.mean(torch.abs(aligned_ict_lmk - flame_neutral_lmks)) * 50

        # add loss for l2 norm on idt_code
        loss += torch.mean(idt_code**2) 
        
        # s_loss = torch.mean(torch.abs(s - ict_lmk_similarity_transform.s))
        # R_loss = torch.mean(torch.abs(R - ict_lmk_similarity_transform.R))
        # T_loss = torch.mean(torch.abs(T - ict_lmk_similarity_transform.T))

        # loss += s_loss + R_loss + T_loss


        pbar.set_description(f'loss: {loss.item()}')

        idt_optimizer.zero_grad()
        loss.backward()
        idt_optimizer.step()
    
    exp = zero_exp.detach()
    exp[ICTmodel.expression_names.tolist().index('jawOpen')] = jaw_open
    # save meshes at this point
    ict_mesh = ICTmodel(expression_weights=exp[None], identity_weights=idt_code[None], to_canonical=False,)
    ict_lmk = ict_mesh[0, ICTmodel.landmark_indices][17:]

    ict_lmk_similarity_transform = ops.corresponding_points_alignment(ict_lmk[None], flame_neutral_lmks[None], estimate_scale=True)

    s = ict_lmk_similarity_transform.s.data.detach()
    R = ict_lmk_similarity_transform.R.data.detach()
    T = ict_lmk_similarity_transform.T.data.detach()

    rotated_ict_lmk = torch.einsum('bni, bji -> bnj', ict_lmk[None], R)
    aligned_ict_lmk = (s * rotated_ict_lmk + T)[0]

    rotated_ict_mesh = torch.einsum('bni, bji -> bnj', ict_mesh, R)
    aligned_ict_mesh = (s * rotated_ict_mesh + T)

    # save mesh as obj using open3d
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(aligned_ict_mesh[0].cpu().data.numpy())
    mesh_o3d.triangles = o3d.utility.Vector3iVector(ICTmodel.faces.cpu().data.numpy())
    o3d.io.write_triangle_mesh(os.path.join(mesh_save_dir, 'ict_canonical.obj'), mesh_o3d)

    # save mesh of flame
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(flame_neutral_verts[0].cpu().data.numpy())
    mesh_o3d.triangles = o3d.utility.Vector3iVector(FLAMEServer.faces_tensor.cpu().data.numpy())
    o3d.io.write_triangle_mesh(os.path.join(mesh_save_dir, 'flame_canonical.obj'), mesh_o3d)


    # ================================================
    # optimize FLAME expression coeff and pose to match ICT-FaceKit expressions
    # ================================================
    
    with torch.no_grad():
        ICTmodel_identity = idt_code.data[None].repeat(ICTmodel.num_expression, 1)
        ICTmodel_expressions = torch.eye(ICTmodel.num_expression).to(device) # N N 
        ict_mesh = ICTmodel(expression_weights=ICTmodel_expressions, identity_weights=ICTmodel_identity, to_canonical=False,)
        ict_lmk = ict_mesh[:, ICTmodel.landmark_indices][:, 17:]
        
        rotated_ict_lmk = torch.einsum('bni, bji -> bnj', ict_lmk, R)
        aligned_ict_lmk = (s[:, None, None] * rotated_ict_lmk + T[:, None])

        rotated_ict_mesh = torch.einsum('bni, bji -> bnj', ict_mesh, R)
        aligned_ict_mesh = (s[:, None, None] * rotated_ict_mesh + T[:, None])


    expressions = torch.nn.Parameter(torch.zeros(ICTmodel.num_expression, FLAMEServer.n_exp).to(device))
    poses = torch.nn.Parameter(torch.zeros(ICTmodel.num_expression, 3).to(device))
    shapes = torch.nn.Parameter(torch.zeros(ICTmodel.num_expression, FLAMEServer.n_shape).to(device))
    

    import pytorch3d.transforms as p3dt
    def euler_to_axis_angle(euler):
        b, _ = euler.shape
        mat = p3dt.euler_angles_to_matrix(euler.reshape(b, 5, 3), convention='XYZ')
        axis_angle = p3dt.matrix_to_axis_angle(mat)
        return axis_angle.reshape(b, 15)
        return p3dt.matrix_to_axis_angle(p3dt.euler_angles_to_matrix(euler, convention='XYZ'))
        # gt_pose = p3dt.matrix_to_euler_angles(p3dt.axis_angle_to_matrix(views_subset['flame_pose'].reshape(b, 5, 3)), convention='XYZ').reshape(b, 15)
    
    flame_optimizer = torch.optim.Adam([
            {'params': expressions, 'lr': 1e-3},
            {'params': poses, 'lr': 1e-3},
            {'params': shapes, 'lr': 1e-4}
        ])

    lmk_face_idx = np.asarray(lmk_face_idx).astype(np.int32)
    lmk_b_coords = np.asarray(lmk_b_coords).astype(np.float32)
    
    pbar = tqdm.tqdm(range(iterations))
    for i in pbar:
        pose_batch = torch.zeros(ICTmodel.num_expression, 15).to(device)
        pose_batch[:, 6:9] = poses
        flame_verts, _, _ = FLAMEServer.forward_with_shapes(shape_params=shapes, expression_params=expressions, full_pose=euler_to_axis_angle(pose_batch))
        
        flame_lmks = mesh_points_by_barycentric_coordinates_batch(flame_verts, FLAMEServer.faces_tensor, lmk_face_idx, lmk_b_coords)
        
        # ict_lmk_similarity_transform = ops.corresponding_points_alignment(ict_lmk, flame_lmks, estimate_scale=True)
        
        

        loss, loss_normals = pytorch3d.loss.chamfer_distance(aligned_ict_mesh, flame_verts, single_directional=True)
        loss = torch.mean(loss)

        loss += torch.mean(torch.abs(aligned_ict_lmk - flame_lmks)) * 10

        # add loss for l2 norm on idt_code
        loss += torch.mean(expressions**2) * 1e-3 + torch.mean(poses**2) * 1e-3 + torch.mean(shapes**2) * 1e-3
 
        pbar.set_description(f'loss: {loss.item()}')

        flame_optimizer.zero_grad()
        loss.backward()
        flame_optimizer.step()
        

    # fetch all optimized variables into a single dict
    optimized_ict_identity = idt_code.data
    optimized_flame_shapes = shapes.data
    optimized_flame_expressions = expressions.data
    optimized_flame_poses = torch.zeros(poses.shape[0], 15).to(device)
    optimized_flame_poses[:, 6:9] = poses.data
    optimized_flame_poses = euler_to_axis_angle(optimized_flame_poses.data)

    optimized_flame_poses_input = torch.zeros(poses.shape[0], 15).to(device)
    optimized_flame_poses_input[:, 6:9] = poses.data
    optimized_flame_poses_input = euler_to_axis_angle(optimized_flame_poses_input)

    # print(optimized_ict_identity, optimized_flame_expressions, optimized_flame_poses)
    print(optimized_ict_identity.mean(dim=0), optimized_ict_identity.std(dim=0))
    print(optimized_flame_shapes.mean(dim=0), optimized_flame_shapes.std(dim=0))
    print(optimized_flame_expressions.mean(dim=0), optimized_flame_expressions.std(dim=0))
    print(optimized_flame_poses.mean(dim=0), optimized_flame_poses.std(dim=0))
    print(optimized_ict_identity.shape, optimized_flame_expressions.shape, optimized_flame_poses.shape)
    

    flame_verts, _, _ = FLAMEServer(expression_params=optimized_flame_expressions, full_pose=optimized_flame_poses_input)
    flame_lmks = mesh_points_by_barycentric_coordinates_batch(flame_verts, FLAMEServer.faces_tensor, lmk_face_idx, lmk_b_coords)
    
    # ict_lmk_similarity_transform = ops.corresponding_points_alignment(ict_lmk, flame_lmks, estimate_scale=True)
 
    optimized_sRT = ict_lmk_similarity_transform

    # save the optimized identity code and jaw weight as a pickle file
    optimized_identity = idt_code.data.cpu().numpy()
    optimized = {}
    optimized['ict_identity'] = optimized_ict_identity
    optimized['flame_shapes'] = optimized_flame_shapes
    optimized['flame_expression'] = optimized_flame_expressions
    optimized['flame_pose'] = optimized_flame_poses
    optimized['flame_s'] = s.data
    optimized['flame_T'] = T.data
    optimized['flame_R'] = R.data
    
    with open(os.path.join(mesh_save_dir, 'optimized.pkl'), 'wb') as f:
        pickle.dump(optimized, f)

    # print flame expression and flame pose for each ict expression
    # print name as well
    for i in range(optimized_flame_expressions.size(0)):
        print(f'ict expression {i}: {ICTmodel.expression_names.tolist()[i]}')
        print(f'flame shapes for ict expression {i}: {np.round(optimized_flame_shapes[i].cpu().data.numpy(), 3)}')
        print(f'flame expression for ict expression {i}: {np.round(optimized_flame_expressions[i].cpu().data.numpy(), 3)}')
        print(f'flame pose for ict expression {i}: {np.round(optimized_flame_poses[i].cpu().data.numpy(), 3)}')

    # analysis on sTR - batch-directional statistics
    print(s.shape, T.shape, R.shape)
    print(s)
    print(T)
    print(R)
    

    # Render each optimized mesh
    for i in range(optimized_flame_expressions.size(0)):
        ict_expression = torch.zeros(1, ICTmodel.num_expression).to(device)
        ict_expression[:, i] = 1.5
        ict_identity = optimized_ict_identity[None].to(device)

        ict_mesh = ICTmodel(expression_weights=ict_expression, identity_weights=ict_identity, to_canonical=False,)

        flame_verts, _, _ = FLAMEServer.forward_with_shapes(shape_params=optimized_flame_shapes[i][None], expression_params=optimized_flame_expressions[i][None], full_pose=optimized_flame_poses[i][None])
        
        rotated_ict_mesh = torch.einsum('bni, bji -> bnj', ict_mesh, R)
        aligned_ict_mesh = (s * rotated_ict_mesh + T)

        # print the chamfer distance 
        loss, loss_normals = pytorch3d.loss.chamfer_distance(aligned_ict_mesh, flame_verts, single_directional=True)
        print(f'chamfer distance for expression {i}: {loss.item()}')

        # Create a mesh node
        ict_mesh = trimesh.Trimesh(vertices=aligned_ict_mesh[0].cpu().data.numpy(), faces=ICTmodel.faces.cpu().data.numpy())
        flame_mesh = trimesh.Trimesh(vertices=flame_verts[0].cpu().data.numpy(), faces=FLAMEServer.faces_tensor.cpu().data.numpy())

        # save mesh
        ict_mesh.export(f'{mesh_save_dir}/ict_exp_{i}_{ICTmodel.expression_names[i]}_ict.obj')
        flame_mesh.export(f'{mesh_save_dir}/ict_exp_{i}_{ICTmodel.expression_names[i]}_flame.obj')


if __name__ == '__main__':
    
    parser = config_parser()
    args = parser.parse_args()

    main(args)
