import torch 

EYELID_PAIRS = torch.tensor([[37, 41], [38, 40], [43, 47], [44, 46]]).int()
LIP_PAIRS = torch.tensor([[61, 67], [62, 66], [63, 65]]).int()

DIRECTION_PAIRS = torch.tensor([[36, 64],[45, 48]]).int()

import pytorch3d.transforms as pt3d

def closure_loss_block(gt, estimated, pairs):
    gt_closure = gt[:, pairs[:, 0], :2] - gt[:, pairs[:, 1], :2]
    estimated_closure = estimated[:, pairs[:, 0], :2] - estimated[:, pairs[:, 1], :2]
    confidence = torch.minimum(gt[:, pairs[:, 0], -1:], gt[:, pairs[:, 1], -1:])
    
    closure_loss = ((estimated_closure - gt_closure).pow(2) * confidence).reshape(gt_closure.shape[0], -1).mean(dim=-1)
    return closure_loss

def landmark_loss(ict_facekit, gbuffers, views_subset, features, nueral_blendshapes, lmk_adaptive, device):
    """
    Calculates the landmark loss by comparing the detected landmarks with the deformed landmarks.

    Args:
        handle_based_deformer (object): The handle-based deformer object.
        gbuffers (dict): A dictionary containing the deformed vertices in clip space.
        views_subset (dict): A dictionary containing the detected landmarks.
        device (torch.device): The device on which the calculations are performed.

    Returns:
        torch.Tensor: The calculated landmark loss.
    """
    # Get the indices of landmarks used by the handle-based deformer
    landmark_indices = ict_facekit.landmark_indices
    landmarks_on_clip_space = gbuffers['deformed_verts_clip_space'][:, landmark_indices]
    landmarks_on_clip_space = landmarks_on_clip_space[..., :3] / torch.clamp(landmarks_on_clip_space[..., 3:], min=1e-8)

    detected_landmarks = views_subset['landmark'].clone().detach()
    detected_landmarks[..., :-1] = detected_landmarks[..., :-1] * 2 - 1
    detected_landmarks[..., 2] = detected_landmarks[..., 2] * -1

    landmark_loss = ((detected_landmarks[:, 17:, :2] - landmarks_on_clip_space[:, 17:, :2]).pow(2) * detected_landmarks[:, 17:, -1:]).reshape(detected_landmarks.shape[0], -1).mean(dim=-1)

    eye_closure_loss = closure_loss_block(detected_landmarks, landmarks_on_clip_space, EYELID_PAIRS) * 8
    lip_closure_loss = closure_loss_block(detected_landmarks, landmarks_on_clip_space, LIP_PAIRS)

    closure_loss = eye_closure_loss + lip_closure_loss

    return landmark_loss, closure_loss
