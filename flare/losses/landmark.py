import torch 

EYELID_PAIRS = torch.tensor([[37, 41], [38, 40], [43, 47], [44, 46]]).int()
LIP_PAIRS = torch.tensor([[61, 67], [62, 66], [63, 65]]).int()

def landmark_loss(ict_facekit, gbuffers, views_subset, device):
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
    
    # Extract the deformed landmarks in clip space
    landmarks_on_clip_space = gbuffers['deformed_verts_clip_space'][:, landmark_indices]
    # landmarks_on_clip_space = gbuffers['deformed_verts_clip_space'][:, landmark_indices[17:]]
    
    # Normalize the detected landmarks to the range [-1, 1]
    detected_landmarks = views_subset['landmark'] * 2 - 1  # shape of B N 3
    # detected_landmarks = views_subset['landmark'][:, 17:] * 2 - 1  # shape of B N 3
    
    # Set the z-coordinate of detected landmarks to 0 where the corresponding deformed landmark is behind the camera
    detected_landmarks[:, :, 2] = torch.where(landmarks_on_clip_space[:, :, 3] < 0, torch.tensor(0.0).to(device), detected_landmarks[:, :, 2])
    
    # Convert the deformed landmarks to normalized coordinates
    landmarks_on_clip_space = landmarks_on_clip_space[:, :, :2] / torch.clamp(landmarks_on_clip_space[:, :, 3:], min=1e-8) # shape of B, N, 3
    
    # Calculate the loss by comparing the detected landmarks with the deformed landmarks
    landmark_loss = torch.mean(torch.abs(detected_landmarks[:, :, :2] - landmarks_on_clip_space) * detected_landmarks[:, :, 2:])
    
    # print(detected_landmarks[0], landmarks_on_clip_space[0], landmark_loss)


    #print(detected_landmarks.shape, landmarks_on_clip_space.shape)

    # Calculate the eye closure loss by comparing the detected eye closure with the deformer eye closure
    detected_eye_closure = detected_landmarks[:, EYELID_PAIRS[:, 0], :2] - detected_landmarks[:, EYELID_PAIRS[:, 1], :2]
    deformer_eye_closure = landmarks_on_clip_space[:, EYELID_PAIRS[:, 0], :2] - landmarks_on_clip_space[:, EYELID_PAIRS[:, 1], :2]
    closure_confidence = detected_landmarks[:, EYELID_PAIRS[:, 0], 2:] * detected_landmarks[:, EYELID_PAIRS[:, 1], 2:]
    #print(detected_eye_closure.shape, deformer_eye_closure.shape, closure_confidence.shape)
    eye_closure_loss = torch.mean(torch.abs(detected_eye_closure - deformer_eye_closure) * closure_confidence)
    
    # calculate lip closure loss 
    detected_lip_closure = detected_landmarks[:, LIP_PAIRS[:, 0], :2] - detected_landmarks[:, LIP_PAIRS[:, 1], :2]
    deformer_lip_closure = landmarks_on_clip_space[:, LIP_PAIRS[:, 0], :2] - landmarks_on_clip_space[:, LIP_PAIRS[:, 1], :2]
    #print(detected_eye_closure.shape, deformer_eye_closure.shape)
    closure_confidence = detected_landmarks[:, LIP_PAIRS[:, 0], 2:] * detected_landmarks[:, LIP_PAIRS[:, 1], 2:]

    lip_closure_loss = torch.mean(torch.abs(detected_lip_closure - deformer_lip_closure) * closure_confidence)

    closure_loss = eye_closure_loss + lip_closure_loss

    return landmark_loss, closure_loss


def closure_loss(ict_facekit, gbuffers, views_subset, device):
    """
    Calculates the eye closure loss by comparing the detected eye closure with the deformer eye closure.

    Args:
        handle_based_deformer (object): The handle-based deformer object.
        gbuffers (dict): A dictionary containing the deformed vertices in clip space.
        views_subset (dict): A dictionary containing the detected landmarks.
        device (torch.device): The device on which the calculations are performed.

    Returns:
        torch.Tensor: The calculated eye closure loss.
    """
    # Get the indices of landmarks used by the handle-based deformer
    landmark_indices = ict_facekit.landmark_indices
    
    # Extract the deformed landmarks in clip space
    landmarks_on_clip_space = gbuffers['deformed_verts_clip_space'][:, landmark_indices]
    
    # Normalize the detected landmarks to the range [-1, 1]
    detected_landmarks = views_subset['landmark'] * 2 - 1  # shape of B N 3
    
    # Set the z-coordinate of detected landmarks to 0 where the corresponding deformed landmark is behind the camera
    detected_landmarks[:, :, 2] = torch.where(landmarks_on_clip_space[:, :, 3] < 0, torch.tensor(0.0).to(device), detected_landmarks[:, :, 2])
    
    landmarks_on_clip_space = landmarks_on_clip_space[:, :, :2] / torch.clamp(landmarks_on_clip_space[:, :, 3:], min=1e-8) # shape of B, N, 3

    return (eye_closure_loss + lip_closure_loss) 
