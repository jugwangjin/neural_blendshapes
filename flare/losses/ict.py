
import pytorch3d.transforms as p3dt
import pytorch3d.ops as p3do
import torch


DIRECTION_PAIRS = torch.tensor([[36, 64],[45, 48]]).int()

def align_to_canonical(source, target, landmark_indices):
    to_canonical = p3do.corresponding_points_alignment(source[:, landmark_indices], target[:, landmark_indices], estimate_scale=True)

    aligned = torch.einsum('bni, bji -> bnj', source, to_canonical.R)
    aligned = (to_canonical.s[:, None, None] * aligned + to_canonical.T[:, None])

    return aligned, to_canonical

def ict_loss(ict_facekit, return_dict, views_subset, neural_blendshapes, renderer, gbuffers):
    features = return_dict['features']
    ict = ict_facekit(expression_weights = features[:, :53], to_canonical = True)

    frontal_indices = ict_facekit.face_indices + ict_facekit.eyeball_indices

    deformed_vertices = return_dict['expression_deformation'] + ict_facekit.canonical

    deformed_vertices_w_template = deformed_vertices + return_dict['template_deformation']

    # aligned_ict, to_canonical = align_to_canonical(ict, deformed_vertices, ict_facekit.landmark_indices)

    ict_loss = torch.mean(torch.pow(ict * 10 - deformed_vertices * 10, 2)) + \
                torch.mean(torch.pow(ict[:, frontal_indices] * 10 - deformed_vertices_w_template[:, deformed_vertices_w_template] * 10, 2)) 

    # ict_landmarks_clip_space = renderer.get_vertices_clip_space(gbuffers, aligned_ict[:, ict_facekit.landmark_indices])
    # ict_landmarks_clip_space = ict_landmarks_clip_space[..., :3] / torch.clamp(ict_landmarks_clip_space[..., 3:], min=1e-8)

    # with torch.no_grad():
    #     # Normalize the detected landmarks to the range [-1, 1]
    #     detected_landmarks = views_subset['landmark'].clone().detach()
    #     detected_landmarks[..., :-1] = detected_landmarks[..., :-1] * 2 - 1
    #     detected_landmarks[..., 2] = detected_landmarks[..., 2] * -1

    #     min_detected_landmarks_z = detected_landmarks[:, :, 2].min(dim=1, keepdim=True)[0]

    #     min_landmarks_on_clip_space_z = ict_landmarks_clip_space[:, :, 2].min(dim=1, keepdim=True)[0]

    #     detected_landmarks[:, :, 2] = (detected_landmarks[:, :, 2] - min_detected_landmarks_z) + min_landmarks_on_clip_space_z
    
    # ict_landmark_loss = torch.mean(torch.pow(detected_landmarks[..., :3] - ict_landmarks_clip_space[..., :3], 2) * detected_landmarks[..., -1:])


    with torch.no_grad():
        random_features = torch.randn_like(features)
        random_features[..., :53] = torch.nn.functional.sigmoid(random_features[..., :53] * 2)
        random_features[..., 53:] = features[..., 53:].clone().detach()

    random_ict = ict_facekit(expression_weights = random_features[:, :53], to_canonical = True)
    
    random_return_dict = neural_blendshapes(image_input=False, features=random_features)
    random_deformed_vertices = random_return_dict['expression_deformation'] + ict_facekit.canonical

    # random_aligned_ict = torch.einsum('bni, bji -> bnj', random_ict, to_canonical.R)
    # random_aligned_ict = (to_canonical.s[:, None, None] * random_aligned_ict + to_canonical.T[:, None])

    # random_aligned_ict = align_to_canonical(random_ict, random_deformed_vertices, ict_facekit.landmark_indices)

    random_ict_loss = torch.mean(torch.pow(random_ict * 10 - random_deformed_vertices * 10, 2))
    


    return ict_loss, random_ict_loss
    # return ict_loss, random_ict_loss, ict_landmark_loss

def ict_identity_regularization(ict_facekit):
    identity_loss = torch.mean(ict_facekit.identity ** 2)

    return identity_loss