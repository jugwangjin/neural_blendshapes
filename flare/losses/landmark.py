import torch 

EYELID_PAIRS = torch.tensor([[37, 41], [38, 40], [43, 47], [44, 46]]).int()
LIP_PAIRS = torch.tensor([[61, 67], [62, 66], [63, 65]]).int()

DIRECTION_PAIRS = torch.tensor([[36, 64],[45, 48]]).int()

import pytorch3d.transforms as pt3d

def closure_loss_block(gt, estimated, pairs):
    gt_closure = gt[:, pairs[:, 0], :2] - gt[:, pairs[:, 1], :2]
    estimated_closure = estimated[:, pairs[:, 0], :2] - estimated[:, pairs[:, 1], :2]
    confidence = torch.minimum(gt[:, pairs[:, 0], -1:], gt[:, pairs[:, 1], -1:])
    
    closure_loss = (torch.abs(estimated_closure - gt_closure) * confidence).reshape(gt_closure.shape[0], -1).mean(dim=-1)
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

    landmark_loss = (torch.abs(detected_landmarks[:, 17:, :2] - landmarks_on_clip_space[:, 17:, :2]) * detected_landmarks[:, 17:, -1:]).reshape(detected_landmarks.shape[0], -1).mean(dim=-1)

    eye_closure_loss = closure_loss_block(detected_landmarks, landmarks_on_clip_space, EYELID_PAIRS) * 8
    lip_closure_loss = closure_loss_block(detected_landmarks, landmarks_on_clip_space, LIP_PAIRS)

    closure_loss = eye_closure_loss + lip_closure_loss

    return landmark_loss, closure_loss


def direction_loss(ict_facekit, gbuffers, views_subset, device):
    # return torch.tensor(0)
    with torch.no_grad():
        detected_landmarks = views_subset['landmark'].detach().data  # shape of B N 4
        detected_landmarks[:, :, :-1] = detected_landmarks[:, :, :-1] 
        detected_landmarks[:, :, 2] = detected_landmarks[:, :, 2] * -1

        detected_normal = detected_landmarks[:, DIRECTION_PAIRS[:, 0], :3] - detected_landmarks[:, DIRECTION_PAIRS[:, 1], :3]
        detected_normal = torch.cross(detected_normal[:, 0], detected_normal[:, 1], dim=1)
        detected_normal = detected_normal / torch.norm(detected_normal, dim=1, keepdim=True)

    landmark_indices = ict_facekit.landmark_indices
    landmarks_on_clip_space = gbuffers['deformed_verts_clip_space'][:, landmark_indices]
    landmarks_on_clip_space = landmarks_on_clip_space[:, :, :3] / torch.clamp(landmarks_on_clip_space[:, :, 3:], min=1e-8) # shape of B, N, 3

    # print(torch.cat([detected_landmarks[:, DIRECTION_PAIRS[:, 0]], landmarks_on_clip_space[:, DIRECTION_PAIRS[:, 0]]], dim=-1), )
    # print(torch.cat([detected_landmarks[:, DIRECTION_PAIRS[:, 1]], landmarks_on_clip_space[:, DIRECTION_PAIRS[:, 1]]], dim=-1))

    deformed_normal = landmarks_on_clip_space[:, DIRECTION_PAIRS[:, 0], :3] - landmarks_on_clip_space[:, DIRECTION_PAIRS[:, 1], :3]
    deformed_normal = torch.cross(deformed_normal[:, 0], deformed_normal[:, 1], dim=1)
    deformed_normal = deformed_normal / torch.norm(deformed_normal, dim=1, keepdim=True)
    
    direction_loss = torch.mean(torch.abs(detected_normal - deformed_normal))

    # print(torch.cat([detected_normal, deformed_normal, detected_normal - deformed_normal], dim=1))
    # print_landmarks = [18, 26, 30, 66, 36, 39, 43, 45]
    # for i in range(5):
    
    #     print((detected_landmarks[i, print_landmarks, 2] + 3))
    #     print(landmarks_on_clip_space[i, print_landmarks, 2])
    # exit()
    return direction_loss



# def closure_loss(ict_facekit, gbuffers, views_subset, device):
#     """
#     Calculates the eye closure loss by comparing the detected eye closure with the deformer eye closure.

#     Args:
#         handle_based_deformer (object): The handle-based deformer object.
#         gbuffers (dict): A dictionary containing the deformed vertices in clip space.
#         views_subset (dict): A dictionary containing the detected landmarks.
#         device (torch.device): The device on which the calculations are performed.

#     Returns:
#         torch.Tensor: The calculated eye closure loss.
#     """
#     # Get the indices of landmarks used by the handle-based deformer
#     landmark_indices = ict_facekit.landmark_indices
    
#     # Extract the deformed landmarks in clip space
#     landmarks_on_clip_space = gbuffers['deformed_verts_clip_space'][:, landmark_indices]
    
#     # Normalize the detected landmarks to the range [-1, 1]
#     detected_landmarks = views_subset['landmark'] * 2 - 1  # shape of B N 3
    
#     # Set the z-coordinate of detected landmarks to 0 where the corresponding deformed landmark is behind the camera
#     detected_landmarks[:, :, 2] = torch.where(landmarks_on_clip_space[:, :, 3] < 0, torch.tensor(0.0).to(device), detected_landmarks[:, :, 2])
    
#     landmarks_on_clip_space = landmarks_on_clip_space[:, :, :2] / torch.clamp(landmarks_on_clip_space[:, :, 3:], min=1e-8) # shape of B, N, 3

#     return (eye_closure_loss + lip_closure_loss) 
