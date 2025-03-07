# Code: https://github.com/fraunhoferhhi/neural-deferred-shading/tree/main
# Modified/Adapted by: Shrisha Bharadwaj

import torch

from flare.core import Mesh

def laplacian_loss(mesh: Mesh):
    """ Compute the Laplacian term as the mean squared Euclidean norm of the differential coordinates.

    Args:
        mesh (Mesh): Mesh used to build the differential coordinates.
    """

    L = mesh.laplacian
    V = mesh.vertices
    
    loss = L.mm(V)
    loss = loss.norm(dim=1)**2
    
    return loss.mean()


def laplacian_loss(mesh: Mesh, canonical_vertices, face_index, head_index):
    """ Compute the Laplacian term as the mean squared Euclidean norm of the differential coordinates.

    Args:
        mesh (Mesh): Mesh used to build the differential coordinates.
    """

    L = mesh.laplacian
    V = mesh.vertices

    mesh_laplacian = L.mm(V)
    canonical_laplacian = L.mm(canonical_vertices)

    # print(mesh_laplacian.shape)
    # exit()

    loss = torch.pow(mesh_laplacian - canonical_laplacian, 2) 


    # add Euclidean norm of the vertex coordinates
    # loss += torch.pow((V - canonical_vertices).norm(dim=-1), 2)
    return loss.mean()
    loss = L.mm(V)
    loss = loss.norm(dim=1)**2
    
    return loss.mean()



def laplacian_loss_given_lap(mesh: Mesh, L, vertices, head_index=11248):
    """ Compute the Laplacian term as the mean squared Euclidean norm of the differential coordinates.

    Args:
        mesh (Mesh): Mesh used to build the differential coordinates.
    """

    # L = mesh.laplacian

    if len(vertices.shape) == 2:
        vertices = vertices.unsqueeze(0)

    vertices = vertices[:, :head_index]

    for b in range(vertices.shape[0]):
        loss = L.mm(vertices[b])
        loss = loss.norm(dim=1)**2
        loss = loss.mean()
        if b == 0:
            total_loss = loss
        else:
            total_loss += loss

    return total_loss / vertices.shape[0]
    
    loss = L.mm(V)
    loss = loss.norm(dim=1)**2
    
    return loss.mean()


def laplacian_loss_two_meshes(mesh, vertices1, vertices2, L, head_index=14062):
    """ Compute the Laplacian term as the mean squared Euclidean norm of the differential coordinates.

    Args:
        mesh (Mesh): Mesh used to build the differential coordinates.
    """

    L = mesh.laplacian
    
    V1 = vertices1
    V2 = vertices2

    if len(V1.shape) == 2:
        V1 = V1.unsqueeze(0)
        V2 = V2.unsqueeze(0)

    V1 = V1 * 10
    V2 = V2 * 10

    for b in range(V1.shape[0]):
        v1_lap = L.mm(V1[b])
        v2_lap = L.mm(V2[b])
        loss = torch.pow(v1_lap - v2_lap, 2)
        loss = loss.mean()
        if b == 0:
            total_loss = loss
        else:
            total_loss += loss
    return total_loss / V1.shape[0]


def laplacian_loss_single_mesh(mesh, vertices1):
    """ Compute the Laplacian term as the mean squared Euclidean norm of the differential coordinates.

    Args:
        mesh (Mesh): Mesh used to build the differential coordinates.
    """

    L = mesh.laplacian
    V1 = vertices1

    if len(V1.shape) == 2:
        V1 = V1.unsqueeze(0)

    total_loss = []
    for b in range(V1.shape[0]):
        v1_lap = L.mm(V1[b])
        loss = v1_lap.norm(dim=-1)**2
        total_loss.append(loss)
    
    return torch.stack(total_loss).mean()


def normal_reg_loss(mesh, mesh1, mesh2, head_index=14062):
    loss = 1 - torch.cosine_similarity(mesh1.vertex_normals, mesh2.vertex_normals, dim=-1)
    # loss = 1 - torch.cosine_similarity(mesh1.vertex_normals[:head_index], mesh2.vertex_normals[:head_index], dim=-1)
    loss = loss**2
    loss = loss.mean()
    
    return loss


def FLAME_loss_function(FLAMEServer,flame_expression, flame_pose, deformed_vertices, flame_pair_indices, ict_pair_indices, target_range=None):
    flame_gt, _, _ = FLAMEServer(flame_expression, flame_pose)
    ict_gt = flame_gt[:, flame_pair_indices]
    # flame_loss = (deformed_vertices[:, ict_pair_indices] - ict_gt)
    deformed_vertices = deformed_vertices[:, ict_pair_indices]
    if target_range is not None:
        deformed_vertices = deformed_vertices[:, target_range]
        ict_gt = ict_gt[:, target_range]

    # flame_loss = torch.nn.functional.l1_loss(deformed_vertices, ict_gt)
    flame_loss = (deformed_vertices * 5 - ict_gt * 5).pow(2).mean()

    # flame_loss = flame_loss.pow(2).mean()
    return flame_loss



def normal_consistency_loss(mesh: Mesh):
    """ Compute the normal consistency term as the cosine similarity between neighboring face normals.

    Args:
        mesh (Mesh): Mesh with face normals.
    """

    loss = 1 - torch.cosine_similarity(mesh.face_normals[mesh.connected_faces[:, 0]], mesh.face_normals[mesh.connected_faces[:, 1]], dim=1)
    return (loss**2).mean()

def normal_cosine_loss(views, gbuffers):
    """ Compute the normal consistency term as the cosine similarity between deformed vertices

    Args:
        mesh (Mesh): Mesh with face normals.
    """
    loss = 0
    for view, gbuffer in zip(views, gbuffers):
        deformed_normals = gbuffer["normal"] * view.skin_mask
        gt_normals = view.normals * view.skin_mask
        loss += 1 - torch.cosine_similarity(deformed_normals.view(-1, 3), gt_normals.view(-1, 3), dim=1)
    return (loss**2).mean()

