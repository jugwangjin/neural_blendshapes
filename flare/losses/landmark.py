import torch 

EYELID_PAIRS = torch.tensor([[37, 41], [38, 40], [43, 47], [44, 46]]).int()
LIP_PAIRS = torch.tensor([[61, 67], [62, 66], [63, 65]]).int()

DIRECTION_PAIRS = torch.tensor([[36, 64],[45, 48]]).int()

import pytorch3d.transforms as pt3d

# closure_blocks = [list(range(68))] # left eye, right eye, mouth
closure_blocks = [list(range(60, 68)), ] # left eye, right eye, mouth
# closure_blocks = [list(range(36, 42)), list(range(42, 48)), list(range(60, 68))] # left eye, right eye, mouth

def closure_loss_block(gt, estimated, pairs):
    gt_closure = gt[:, pairs[:, 0], :2] - gt[:, pairs[:, 1], :2]
    estimated_closure = estimated[:, pairs[:, 0], :2] - estimated[:, pairs[:, 1], :2]
    confidence = torch.minimum(gt[:, pairs[:, 0], -1:], gt[:, pairs[:, 1], -1:])
    
    closure_loss = ((estimated_closure - gt_closure).pow(2) * confidence).mean()
    return closure_loss

def landmark_loss_function(ict_facekit, gbuffers, views_subset, use_jaw, device):
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
    landmarks_on_clip_space_ = gbuffers['deformed_verts_clip_space'][:, landmark_indices]
    landmarks_on_clip_space = torch.zeros_like(landmarks_on_clip_space_)
    landmarks_on_clip_space[..., :2] = landmarks_on_clip_space_[..., :2] / torch.clamp(landmarks_on_clip_space_[..., -1:], min=1e-8) 
    landmarks_on_clip_space[..., 2:] = landmarks_on_clip_space_[..., 2:]
    
    # Align detected_landmarks to landmarks_on_clip_space
    with torch.no_grad():
        detected_landmarks = views_subset['landmark'].clone().detach()
        detected_landmarks[..., :2] = detected_landmarks[..., :2] * 2 - 1
        detected_landmarks[..., 2] = detected_landmarks[..., 2] * -2
        mean_z_detected_landmarks = detected_landmarks[..., 2].mean(dim=-1, keepdim=True)[0]
        detected_landmarks[..., 2] -= mean_z_detected_landmarks

    mean_z_landmarks_on_clip_space = landmarks_on_clip_space[..., 2].mean(dim=-1, keepdim=True)[0]
    landmarks_on_clip_space[..., 2] -= mean_z_landmarks_on_clip_space

    # print statistics of both landmarks
    # print(detected_landmarks[..., :2].mean(dim=1), detected_landmarks[..., :2].std(dim=1), detected_landmarks[..., :2].amin(dim=1), detected_landmarks[..., :2].amax(dim=1))
    # print(landmarks_on_clip_space[..., :2].mean(dim=1), landmarks_on_clip_space[..., :2].std(dim=1), landmarks_on_clip_space[..., :2].amin(dim=1), landmarks_on_clip_space[..., :2].amax(dim=1))

    starting_index = 0


    # landmark_loss = ((detected_landmarks[:, starting_index:, :2] - landmarks_on_clip_space[:, starting_index:, :2]).pow(2) * detected_landmarks[:, starting_index:, -1:])
    # if use_jaw:
    #     landmark_loss[:, :17] *= 0.25
    # else:   
    #     landmark_loss[:, :17] *= 0
    # landmark_loss[:, 17:27] *= 0.5
    # landmark_loss = landmark_loss.mean()

    closure_loss = 0
    for block in closure_blocks:
        gt_closure = detected_landmarks[:, None, block, :2] - detected_landmarks[:, block, None, :2]
        estimated_closure = landmarks_on_clip_space[:, None, block, :2] - landmarks_on_clip_space[:, block, None, :2]
        confidence = torch.minimum(detected_landmarks[:, None, block, -1:], detected_landmarks[:, block, None, -1:])

        closure_loss += ((estimated_closure - gt_closure).pow(2) * confidence).mean()

        # closure_loss += torch.nn.functional.huber_loss(estimated_closure, gt_closure)
        # closure_loss += closure_loss_.mean()
        
    if use_jaw:
        detected_landmarks[:, :17, -1] *= 0.1
    else:
        detected_landmarks[:, :17, -1] *= 0

    cross_confidence = (detected_landmarks[:, :, None, -1] * detected_landmarks[:, None, :, -1]) # shape of (bs, 68, 68)

    detected_landmarks = detected_landmarks[:, :]
    landmarks_on_clip_space = landmarks_on_clip_space[:, :]
    
    landmark_loss = ((detected_landmarks[:, :, :-1] - landmarks_on_clip_space[:, :, :-1]) * detected_landmarks[:, :, -1:]).pow(2).mean()
    # landmark_loss = torch.nn.functional.huber_loss(detected_landmarks, landmarks_on_clip_space)

    detected_cross_landmarks = torch.norm(detected_landmarks[:, :, None, :2] - detected_landmarks[:, None, :, :2], dim=-1) # shape of (bs, 68, 68)
    clip_space_cross_landmarks = torch.norm(landmarks_on_clip_space[:, :, None, :2] - landmarks_on_clip_space[:, None, :, :2], dim=-1) # shape of (bs, 68, 68)

    cross_landmark_loss = ((detected_cross_landmarks - clip_space_cross_landmarks).pow(2) * cross_confidence).mean()

    landmark_loss = landmark_loss + cross_landmark_loss

    return landmark_loss, closure_loss
