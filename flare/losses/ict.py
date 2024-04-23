
import pytorch3d.transforms as p3dt
import pytorch3d.ops as p3do
import torch

DIRECTION_PAIRS = torch.tensor([[36, 64],[45, 48]]).int()

def align_to_canonical(source, target, landmark_indices):
    to_canonical = p3do.corresponding_points_alignment(source[:, landmark_indices], target[:, landmark_indices], estimate_scale=True)

    aligned = torch.einsum('bni, bji -> bnj', source, to_canonical.R)
    aligned = (to_canonical.s[:, None, None] * aligned + to_canonical.T[:, None])

    return aligned

def ict_loss(ict_facekit, deformed_vertices, features, deformer_net):
    ict = ict_facekit(expression_weights = features[:, :53], to_canonical = True)

    aligned_ict = align_to_canonical(ict, deformed_vertices, ict_facekit.landmark_indices)

    aligned_ict = aligned_ict[:, (ict_facekit.face_indices + ict_facekit.eyeball_indices)]

    deformed_vertices = deformed_vertices[:, (ict_facekit.face_indices + ict_facekit.eyeball_indices)]

    ict_loss = torch.mean(torch.pow(aligned_ict * 10 - deformed_vertices * 10, 2))
    # ict_loss = torch.mean(torch.abs(aligned_ict * 10 - deformed_vertices * 10))

    # sample random feature and apply same loss
    with torch.no_grad():
        random_features = features.clone().detach()
        random_features[:, :53] = torch.nn.functional.sigmoid(torch.randn(features.shape[0], 53))

    random_ict = ict_facekit(expression_weights = random_features[:, :53], to_canonical = True)

    random_deformed = deformer_net(random_features, vertices=None)    
    random_deformed_vertices = torch.zeros_like(ict_facekit.canonical).repeat(random_features.size(0), 1, 1)
    random_deformed_vertices[:, ict_facekit.head_indices] = random_deformed["deformed_head"]
    random_deformed_vertices[:, ict_facekit.eyeball_indices] = random_deformed["deformed_eyes"]

    aligned_random_ict = align_to_canonical(random_ict, random_deformed_vertices, ict_facekit.landmark_indices)

    aligned_random_ict = aligned_random_ict[:, (ict_facekit.face_indices + ict_facekit.eyeball_indices)]

    random_deformed_vertices = random_deformed_vertices[:, (ict_facekit.face_indices + ict_facekit.eyeball_indices)]

    random_ict_loss = torch.mean(torch.pow(aligned_random_ict * 10 - random_deformed_vertices * 10, 2))
    # random_ict_loss = torch.mean(torch.abs(aligned_random_ict * 10 - random_deformed_vertices * 10))

    return ict_loss, random_ict_loss

def ict_identity_regularization(ict_facekit):
    identity_loss = torch.mean(ict_facekit.identity ** 2)

    return identity_loss
