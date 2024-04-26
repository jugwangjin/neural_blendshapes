
import pytorch3d.transforms as p3dt
import pytorch3d.ops as p3do
import torch
from core import Renderer

DIRECTION_PAIRS = torch.tensor([[36, 64],[45, 48]]).int()

def align_to_canonical(source, target, landmark_indices):
    to_canonical = p3do.corresponding_points_alignment(source[:, landmark_indices], target[:, landmark_indices], estimate_scale=True)

    aligned = torch.einsum('bni, bji -> bnj', source, to_canonical.R)
    aligned = (to_canonical.s[:, None, None] * aligned + to_canonical.T[:, None])

    return aligned

def ict_loss(ict_facekit, return_dict, features, views_subset, renderer, gbuffers):
    features = return_dict['features']
    ict = ict_facekit(expression_weights = features[:, :53], to_canonical = True)

    deformed_vertices = return_dict['expression_deformation'] + ict_facekit.canonical

    aligned_ict = align_to_canonical(ict, deformed_vertices, ict_facekit.landmark_indices)

    ict_loss = torch.mean(torch.pow(aligned_ict * 10 - deformed_vertices * 10, 2))

    ict_landmarks_clip_space = renderer.get_vertices_clip_space(gbuffers, aligned_ict[:, ict_facekit.landmark_indices])
    ict_landmarks_clip_space = ict_landmarks_clip_space[..., :3] / torch.clamp(ict_landmarks_clip_space[..., 3:], min=1e-8)

    with torch.no_grad():
        # Normalize the detected landmarks to the range [-1, 1]
        detected_landmarks = views_subset['landmark'].detach().data
        detected_landmarks[..., :-1] = detected_landmarks[..., :-1] * 2 - 1
        detected_landmarks[..., 2] = detected_landmarks[..., 2] * -1
    
    ict_landmark_loss = torch.mean(torch.pow(detected_landmarks[..., :2] - ict_landmarks_clip_space[..., :2], 2) * detected_landmarks[..., -1:])

    return ict_loss, ict_landmark_loss

def ict_identity_regularization(ict_facekit):
    identity_loss = torch.mean(ict_facekit.identity ** 2)

    return identity_loss
