
import pytorch3d.transforms as p3dt
import pytorch3d.ops as p3do
import torch

from .geometry import laplacian_loss
from .landmark import closure_loss_block

import pytorch3d.transforms as pt3d
DIRECTION_PAIRS = torch.tensor([[36, 64],[45, 48]]).int()

EYELID_PAIRS = torch.tensor([[37, 41], [38, 40], [43, 47], [44, 46]]).int()
LIP_PAIRS = torch.tensor([[61, 67], [62, 66], [63, 65]]).int()

HALF_PI = torch.pi / 2

def align_to_canonical(source, target, landmark_indices):
    to_canonical = p3do.corresponding_points_alignment(source[:, landmark_indices], target[:, landmark_indices], estimate_scale=True)

    aligned = torch.einsum('bni, bji -> bnj', source, to_canonical.R)
    aligned = (to_canonical.s[:, None, None] * aligned + to_canonical.T[:, None])

    return aligned, to_canonical

def closure_loss_fun(gt_closure, deformed_closure, confidence):
    gt_closure_distance = torch.norm(gt_closure, dim=-1)
    deformed_closure_distance = torch.norm(deformed_closure, dim=-1)
    return torch.mean(torch.pow(gt_closure_distance - deformed_closure_distance, 2) * confidence)


def ict_loss(ict_facekit, return_dict, views_subset, neural_blendshapes, renderer, lmk_adaptive, fullhead_template=False):
    features = return_dict['features']

    # ICT loss
    ict = ict_facekit(expression_weights = features[:, :53], to_canonical = True)

    deformed_vertices = return_dict['full_expression_deformation'] + ict_facekit.canonical

    ict_ = ict.detach().clone()
    ict_loss = torch.pow(ict_ * 10 - deformed_vertices * 10, 2).reshape(ict_.shape[0], -1).mean(dim=-1)

    # Random ICT Loss
    with torch.no_grad():
        random_facs = torch.zeros_like(features)
        for b in range(features.shape[0]):
            weights = torch.tensor([1/i for i in range(1, 53)])
            random_integer = torch.multinomial(weights, 1).item() + 1
            random_indices = torch.randint(0, 53, (random_integer,))
            # for 10 and 11, randomly append it to random_indices
            if torch.rand(1) > 0.5:
                random_indices = torch.cat([random_indices, torch.tensor([10])])
            if torch.rand(1) > 0.5:
                random_indices = torch.cat([random_indices, torch.tensor([11])])
            random_indices = random_indices.unique()

            # sample 0 to 1 for each indices
            random_facs[b, random_indices] = torch.rand_like(random_facs[b, random_indices])
        random_pose = torch.rand_like(random_facs) * torch.pi - HALF_PI
        random_features = torch.cat([random_facs[..., :53], random_pose[..., 53:]], dim=-1)
        
    random_ict = ict_facekit(expression_weights = random_features[..., :53], to_canonical = True)
    
    random_return_dict = neural_blendshapes(image_input=False, features=random_features)
    random_deformed_vertices = random_return_dict['full_expression_deformation'] + ict_facekit.canonical

    random_ict_loss = torch.pow(random_ict * 10 - random_deformed_vertices * 10, 2).reshape(random_ict.shape[0], -1).mean(dim=-1)

    # ICT landmark loss
    ict_landmarks = ict[:, ict_facekit.landmark_indices]

    ict_landmarks = neural_blendshapes.apply_deformation(ict_landmarks, features)
    ict_landmarks_clip_space = renderer.get_vertices_clip_space_from_view(views_subset['camera'], ict_landmarks)
    ict_landmarks_clip_space = ict_landmarks_clip_space[..., :3] / torch.clamp(ict_landmarks_clip_space[..., 3:], min=1e-8)

    detected_landmarks = views_subset['landmark'].clone().detach()
    detected_landmarks[..., :-1] = detected_landmarks[..., :-1] * 2 - 1
    detected_landmarks[..., 2] = detected_landmarks[..., 2] * -1

    # ict jaw line landmark positions does not match with common 68 landmarks
    ict_landmark_loss = (torch.pow(detected_landmarks[:, 17:, :2] - ict_landmarks_clip_space[:, 17:, :2], 2) * detected_landmarks[:, 17:, -1:]).reshape(detected_landmarks.shape[0], -1).mean(dim=-1)

    eye_closure_loss = closure_loss_block(detected_landmarks, ict_landmarks_clip_space, EYELID_PAIRS) * 32
    lip_closure_loss = closure_loss_block(detected_landmarks, ict_landmarks_clip_space, LIP_PAIRS)

    closure_loss = eye_closure_loss + lip_closure_loss
    
    ict_landmark_loss = ict_landmark_loss
    ict_landmark_closure_loss = closure_loss

    return ict_loss, random_ict_loss, ict_landmark_loss, ict_landmark_closure_loss


def ict_identity_regularization(ict_facekit):
    identity_loss = torch.mean(ict_facekit.identity ** 2)

    return identity_loss