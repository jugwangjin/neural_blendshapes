import os
import hashlib

import igl
import torch
import open3d as o3d
import numpy as np

from mesh import MeshWrapper
from scipy import sparse
from scipy.spatial import KDTree
from pytorch3d.transforms import rotation_6d_to_matrix, quaternion_to_matrix, matrix_to_rotation_6d
from utils import DIR_CACHE, log

def hash_template_node(V:np.array, F:np.array, indices_node:np.array):
    """
    Compute hash identifying the template mesh + node configuration
    """
    sha = hashlib.sha1(V.tobytes()+F.tobytes()+indices_node.tobytes())
    return sha.hexdigest()

def compute_weight_heat(V:np.array, F:np.array, indices_node:np.array):
    """
    Compute weight map for sparse nodes scattered on surface
    Using heat diffusion method
    """
    hash = hash_template_node(V, F, indices_node)
    path_geodesic_cache = os.path.join(DIR_CACHE, f'geodesic_{hash}.npy')
    path_weight_cache = os.path.join(DIR_CACHE, f'graphweight_{hash}.npy')
    os.makedirs(os.path.dirname(path_geodesic_cache), exist_ok=True)
    if os.path.exists(path_weight_cache):
        log.debug("loading graph weight cache")
        v_weight = np.load(path_weight_cache)
    else:
        if os.path.exists(path_geodesic_cache):
            log.debug("loading geodesic cache")
            v_m_dist = np.load(path_geodesic_cache)
        else:
            v_m_dist = np.zeros((len(V), len(indices_node)), dtype=np.float32)
            log.debug("computing geodesic cache")
            for i, idx_node in enumerate(indices_node):
                indices_source = np.array([idx_node], dtype=np.int32)
                distances = np.array(igl.heat_geodesic(V, F.astype(np.int32), 0.01, indices_source), dtype=np.float32)
                v_m_dist[:, i] = distances
            np.save(path_geodesic_cache, v_m_dist)
            log.debug("Now geodesic cache is baked")
    
        log.debug("Computing deformation graph weights")
        area = igl.doublearea(V, F) / 2.0
        v_area = np.zeros((len(V)), dtype=np.float32)
        v_area[F[:,0]] += area
        v_area[F[:,1]] += area
        v_area[F[:,2]] += area

        # Heat equiblem
        v_min_dist = np.min(v_m_dist, axis=1)
        one = np.ones(len(V), dtype=np.float32)
        v_min_dist_2 = np.divide(one, v_min_dist + 1e-12)
        v_min_dist_2 = np.divide(v_min_dist_2, v_min_dist + 1e-12)
        v_min_dist_2 *= v_area

        L = igl.cotmatrix(V, F)
        A = sparse.diags(v_min_dist_2) - sparse.csr_matrix(L)

        v_weight = np.zeros((len(V), len(indices_node)), dtype=np.float32)
        for i, idx_node in enumerate(indices_node):
            one = np.ones(len(V), dtype=np.float32)
            tmp_v_min_dist_2 = np.divide(one, v_m_dist[:, i] + 1e-8)
            tmp_v_min_dist_2 = np.divide(tmp_v_min_dist_2, v_m_dist[:, i] + 1e-8)
    
            b = np.where(v_m_dist[:,i] < v_min_dist * 1.0001, v_min_dist_2, 0)
            v_weight[:, i] = sparse.linalg.spsolve(A, b)
        
        for i, m in enumerate(indices_node):
            v_weight[m,:] = 0
            v_weight[m,i] = 1
 
        v_weight_l = v_weight.sum(axis=1)
        v_weight = v_weight / v_weight_l[:, np.newaxis]

        np.save(path_weight_cache, v_weight)
        log.debug("Now deformation graph weight cache is baked")
    
    return v_weight

class DeformationGraph(torch.nn.Module):
    def __init__(self, omesh:MeshWrapper, device:str ='cuda:0', desired_tri_graph:int =300):
        super().__init__()
        self.device = device
        self.mesh_nodes = omesh.o3dmesh_ori.simplify_quadric_decimation(desired_tri_graph)
        self.mesh_nodes.compute_adjacency_list()
        _V_nodes = np.array(self.mesh_nodes.vertices)

        # Convert vertices in simplified mesh to closest vertices in original mesh
        kdtree = KDTree(omesh.o3dmesh_ori.vertices)
        _, idx_nn_to_nodes = kdtree.query(_V_nodes)
        _V_nodes = omesh.V_ori[idx_nn_to_nodes]

        _W_nodes = compute_weight_heat(omesh.V_ori, omesh.F_ori, idx_nn_to_nodes)

        # create torch tensors for working with the graph
        self.W_nodes = torch.from_numpy(_W_nodes).float().to(device) # (#V x #C)
        self.V_nodes_init = torch.from_numpy(_V_nodes).float().to(device)
        self.idx_nn_to_nodes = torch.from_numpy(idx_nn_to_nodes).long().to(device)
        self.num_nodes = self.W_nodes.shape[1]

        # small matrix. just iterate with python for
        self.A = torch.zeros([self.num_nodes, self.num_nodes], dtype=torch.float32)
        for i in range(self.num_nodes):
            num_neighbor = len(self.mesh_nodes.adjacency_list[i])
            for j in self.mesh_nodes.adjacency_list[i]:
                self.A[i, j] = 1 / num_neighbor
        self.A = self.A.to(device)

    def get_init_params(self, size_batch):
        """
        Create optimizable torch tensors for deformation.
        """
        V_nodes = torch.tile(
            0.0*self.V_nodes_init, (size_batch, 1, 1)
            ).detach().requires_grad_(True)
        rot6d_nodes = torch.tile(
            torch.tensor([1, 0, 0, 0, 1, 0],
                         dtype=torch.float32, device=self.device),
                         (size_batch, self.num_nodes, 1)
                         ).detach().requires_grad_(True)

        return V_nodes, rot6d_nodes

    def regularization_loss(self, X, V_nodes, rot6d_nodes):
        X = X.detach() # regularization for nodes only
        R = rotation_6d_to_matrix(rot6d_nodes) # (B, C, 3, 3)
        center = X[...,self.idx_nn_to_nodes,:]
        disp = center.unsqueeze(2) - center.unsqueeze(1) # (B, C, C, 3)
        Rdisp = (torch.einsum('bvcd,bckd->bvck', disp, R) 
                 + center.unsqueeze(1) + V_nodes.unsqueeze(1) 
                 - center.unsqueeze(2) - V_nodes.unsqueeze(2))
        norm = Rdisp.norm(dim=-1)**2 # (B, C, C)
        mean = (norm*self.A).sum(-1) # (B, C)
        return mean.mean()
    
    def forward(self, X, V_nodes, rot6d_nodes):
        """
        X               (B, #V, 3) base shape
        V_nodes         (B, #C, 3) deformation graph node position
        rot6d_nodes     (B, #C, 6) deformation graph node rotation

        Returns)
        Deformed tensor (B, #V, 3)
        """

        # Goal: X'_i = \sum_j^C w_{ij}*R_j(x_i - c_j) + c_j

        R = rotation_6d_to_matrix(rot6d_nodes) # (B, C, 3, 3)
        center = X[...,self.idx_nn_to_nodes,:]
        disp = X.unsqueeze(2) - center.unsqueeze(1)
        Rdisp = torch.einsum('bvcd,bckd->bvck', disp, R) + center.unsqueeze(1) + V_nodes.unsqueeze(1) # (B, V, C, 3)
        lerp_Rdisp = (self.W_nodes[None,...,None] * Rdisp).sum(dim=2)

        return lerp_Rdisp