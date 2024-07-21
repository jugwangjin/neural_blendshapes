import torch 

EYELID_PAIRS = torch.tensor([[37, 41], [38, 40], [43, 47], [44, 46]]).int()
LIP_PAIRS = torch.tensor([[61, 67], [62, 66], [63, 65]]).int()

DIRECTION_PAIRS = torch.tensor([[36, 64],[45, 48]]).int()

import pytorch3d.transforms as pt3d

# closure_blocks = [list(range(68))] # left eye, right eye, mouth
closure_blocks = [list(range(48, 68))] # left eye, right eye, mouth
# closure_blocks = [list(range(36, 42)), list(range(42, 48)), list(range(48, 68))] # left eye, right eye, mouth

def closure_loss_block(gt, estimated, pairs):
    gt_closure = gt[:, pairs[:, 0], :3] - gt[:, pairs[:, 1], :3]
    estimated_closure = estimated[:, pairs[:, 0], :3] - estimated[:, pairs[:, 1], :3]
    confidence = torch.minimum(gt[:, pairs[:, 0], -1:], gt[:, pairs[:, 1], -1:])
    
    closure_loss = ((estimated_closure - gt_closure).pow(2) * confidence).mean()
    return closure_loss

def landmark_loss(ict_facekit, gbuffers, views_subset, use_jaw, device):
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

    detected_landmarks = views_subset['landmark'].clone().detach()
    detected_landmarks[..., :2] = detected_landmarks[..., :2] * 2 - 1
    detected_landmarks[..., 2] = detected_landmarks[..., 2] * -2
    # multiply by 0.25 for first 17 landmarks at the last axis of detected_landmarks
    # reduce the weight of the jaw landmarks
    detected_landmarks[:, :17, -1] *= 0.25

    # both landmark on clip space and detected landmarks, subtract by the minimum value on z axis
    # to make the z axis value positive
    landmarks_on_clip_space[..., 2] -= landmarks_on_clip_space[..., 2].min(dim=-1, keepdim=True)[0]
    detected_landmarks[..., 2] -= detected_landmarks[..., 2].min(dim=-1, keepdim=True)[0]

    
    starting_index = 0 if use_jaw else 17


    landmark_loss = ((detected_landmarks[:, starting_index:, :3] - landmarks_on_clip_space[:, starting_index:, :3]).pow(2) * detected_landmarks[:, starting_index:, -1:])
    landmark_loss[..., -1] *= 0.25
    landmark_loss = landmark_loss.mean()

    # landmark_loss = ((detected_landmarks[:, starting_index:, :3] - landmarks_on_clip_space[:, starting_index:, :3]).pow(2) * detected_landmarks[:, starting_index:, -1:]).mean()

    closure_loss = 0
    for block in closure_blocks:
        gt_closure = detected_landmarks[:, None, block, :3] - detected_landmarks[:, block, None, :3]
        estimated_closure = landmarks_on_clip_space[:, None, block, :3] - landmarks_on_clip_space[:, block, None, :3]
        confidence = torch.minimum(detected_landmarks[:, None, block, -1:], detected_landmarks[:, block, None, -1:])

        closure_loss_ = ((estimated_closure - gt_closure).pow(2) * confidence)
        closure_loss_[..., -1] *= 0.25
        closure_loss += closure_loss_.mean()
        # closure_loss += ((estimated_closure - gt_closure).pow(2) * confidence).mean()
        

    # gt_closure = detected_landmarks[:, None, starting_index:, :3] - detected_landmarks[:, starting_index:, None, :3] 
    # gt_closure_confidence = detected_landmarks[:, None, starting_index:, -1:] * detected_landmarks[:, starting_index:, None, -1:]

    # estimated_closure = landmarks_on_clip_space[:, None, starting_index:, :3] - landmarks_on_clip_space[:, starting_index:, None, :3]

    # closure_loss = ((estimated_closure - gt_closure).pow(2) * gt_closure_confidence).mean()

    # eye_closure_loss = closure_loss_block(detected_landmarks, landmarks_on_clip_space, EYELID_PAIRS) * 2
    # lip_closure_loss = closure_loss_block(detected_landmarks, landmarks_on_clip_space, LIP_PAIRS)

    # closure_loss = eye_closure_loss + lip_closure_loss

    return landmark_loss, closure_loss
