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

def laplacian_loss_two_meshes(mesh, vertrices1, vertices2):
    """ Compute the Laplacian term as the mean squared Euclidean norm of the differential coordinates.

    Args:
        mesh (Mesh): Mesh used to build the differential coordinates.
    """

    L = mesh.laplacian
    V1 = vertrices1
    V2 = vertices2

    if len(V1.shape) == 2:
        V1 = V1.unsqueeze(0)
        V2 = V2.unsqueeze(0)

    for b in range(V1.shape[0]):
        v1_lap = L.mm(V1[b])
        v2_lap = L.mm(V2[b])
        loss = torch.pow(v1_lap - v2_lap, 2)
        if b == 0:
            total_loss = loss
        else:
            total_loss += loss
    return total_loss.mean()

def normal_reg_loss(mesh, vertices1, vertices2):
    if len(vertices1.shape) == 2:
        vertices1 = vertices1.unsqueeze(0)
        vertices2 = vertices2.unsqueeze(0)

    for b in range(vertices1.shape[0]):
        mesh1 = mesh.with_vertices(vertices1[b])
        mesh2 = mesh.with_vertices(vertices2[b])

        loss = 1 - torch.cosine_similarity(mesh1.face_normals, mesh2.face_normals, dim=-1)
        if b == 0:
            total_loss = loss
        else:
            total_loss += loss

    return total_loss.mean()

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

