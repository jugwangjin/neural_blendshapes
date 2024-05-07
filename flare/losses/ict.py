
import pytorch3d.transforms as p3dt
import pytorch3d.ops as p3do
import torch

from .geometry import laplacian_loss

import pytorch3d.transforms as pt3d
DIRECTION_PAIRS = torch.tensor([[36, 64],[45, 48]]).int()

EYELID_PAIRS = torch.tensor([[37, 41], [38, 40], [43, 47], [44, 46]]).int()
LIP_PAIRS = torch.tensor([[61, 67], [62, 66], [63, 65]]).int()

def align_to_canonical(source, target, landmark_indices):
    to_canonical = p3do.corresponding_points_alignment(source[:, landmark_indices], target[:, landmark_indices], estimate_scale=True)

    aligned = torch.einsum('bni, bji -> bnj', source, to_canonical.R)
    aligned = (to_canonical.s[:, None, None] * aligned + to_canonical.T[:, None])

    return aligned, to_canonical


def ict_loss(ict_facekit, return_dict, views_subset, neural_blendshapes, renderer, gbuffers, lmk_adaptive, fullhead_template=False):
    features = return_dict['features']
    ict = ict_facekit(expression_weights = features[:, :53], to_canonical = True).clone().detach()

    frontal_indices = ict_facekit.face_indices + ict_facekit.eyeball_indices


    deformed_vertices = return_dict['full_expression_deformation'] + ict_facekit.canonical

    deformed_vertices_w_template = deformed_vertices + return_dict['full_template_deformation']


    # canonical_landmarks = ict
    # euler_angle = features[:, 53:56]
    # translation = features[:, 56:59]
    # # scale = features[:, -1:]

    # local_coordinate_vertices = canonical_landmarks + return_dict['full_template_deformation'][None] - neural_blendshapes.transform_origin[None, None]

    # local_coordinate_vertices = local_coordinate_vertices

    # rotation_matrix = pt3d.euler_angles_to_matrix(euler_angle, convention = 'XYZ')

    # transformed_ict = (torch.bmm(local_coordinate_vertices, rotation_matrix.transpose(1, 2)) + translation[:, None, :] + neural_blendshapes.transform_origin[None, None]).clone().detach  z()

    if fullhead_template:
        ict_loss = torch.mean(torch.pow(ict * 10 - deformed_vertices * 10, 2)) + \
                torch.mean(torch.pow(ict * 10 - deformed_vertices_w_template * 10, 2))
        
    else:
        ict_loss = torch.mean(torch.pow(ict * 10 - deformed_vertices * 10, 2)) + \
                torch.mean(torch.pow(ict[:, frontal_indices] * 10 - deformed_vertices_w_template[:, frontal_indices] * 10, 2))
                # torch.mean(torch.pow(transformed_ict[:, frontal_indices] - return_dict['full_deformed_mesh'][:, frontal_indices], 2)) 





    random_features = torch.randn_like(features)
    random_features[..., :53] = torch.nn.functional.sigmoid(random_features[..., :53] * 2)

    random_ict = ict_facekit(expression_weights = random_features[..., :53], to_canonical = True).clone().detach()
    
    random_return_dict = neural_blendshapes(image_input=False, lmks=None, features=random_features)
    random_deformed_vertices = random_return_dict['full_expression_deformation'] + ict_facekit.canonical

    # random_deformed_vertices_w_template = random_deformed_vertices + random_return_dict['full_template_deformation']

    random_ict_loss = torch.mean(torch.pow(random_ict * 10 - random_deformed_vertices * 10, 2)) 
                    #   0.1 * torch.mean(torch.pow(random_ict[:, frontal_indices] * 10 - random_deformed_vertices_w_template[:, frontal_indices] * 10, 2))


    canonical_landmarks = ict[:, ict_facekit.landmark_indices].clone().detach()

    transformed_ict = neural_blendshapes.apply_deformation(canonical_landmarks, features)

    # euler_angle = features[:, 53:56]
    # translation = features[:, 56:59]
    # scale = features[:, -1:]


    # local_coordinate_vertices = canonical_landmarks - neural_blendshapes.transform_origin[None, None]

    # local_coordinate_vertices = local_coordinate_vertices 

    # rotation_matrix = pt3d.eulzr_angles_to_matrix(euler_angle, convention = 'XYZ')

    # transformed_ict = torch.einsum('bvd, bdj -> bvj', local_coordinate_vertices, rotation_matrix) + translation[:, None, :] + neural_blendshapes.transform_origin[None, None]

    ict_landmarks_clip_space = renderer.get_vertices_clip_space(gbuffers, transformed_ict)
    ict_landmarks_clip_space = ict_landmarks_clip_space[..., :3] / torch.clamp(ict_landmarks_clip_space[..., 3:], min=1e-8)

    detected_landmarks = views_subset['landmark'].clone().detach()
    detected_landmarks[..., :-1] = detected_landmarks[..., :-1] * 2 - 1
    # print(detected_landmarks)
    # exit()
    detected_landmarks[..., 2] = detected_landmarks[..., 2] * -1

    ict_landmark_loss = torch.mean(torch.pow(detected_landmarks[..., :2] - ict_landmarks_clip_space[..., :2], 2) * detected_landmarks[..., -1:])
    # ict_landmark_loss = torch.mean(lmk_adaptive.lossfun(((detected_landmarks[..., :2] - ict_landmarks_clip_space[..., :2]) * detected_landmarks[..., -1:]).view(-1, 2)**2))


    detected_eye_closure = detected_landmarks[:, EYELID_PAIRS[:, 0], :2] - detected_landmarks[:, EYELID_PAIRS[:, 1], :2]
    deformer_eye_closure = ict_landmarks_clip_space[:, EYELID_PAIRS[:, 0], :2] - ict_landmarks_clip_space[:, EYELID_PAIRS[:, 1], :2]
    closure_confidence = torch.minimum(detected_landmarks[:, EYELID_PAIRS[:, 0], -1:], detected_landmarks[:, EYELID_PAIRS[:, 1], -1:])
    
    # eye_closure_loss = torch.mean(lmk_adaptive.lossfun(((detected_eye_closure - deformer_eye_closure) * closure_confidence).view(-1, 2)**2))
    eye_closure_loss = torch.mean(torch.pow(detected_eye_closure - deformer_eye_closure, 2) * closure_confidence)
    
    detected_lip_closure = detected_landmarks[:, LIP_PAIRS[:, 0], :2] - detected_landmarks[:, LIP_PAIRS[:, 1], :2]
    deformer_lip_closure = ict_landmarks_clip_space[:, LIP_PAIRS[:, 0], :2] - ict_landmarks_clip_space[:, LIP_PAIRS[:, 1], :2]
    closure_confidence = torch.minimum(detected_landmarks[:, LIP_PAIRS[:, 0], -1:], detected_landmarks[:, LIP_PAIRS[:, 1], -1:])

    # lip_closure_loss = torch.mean(lmk_adaptive.lossfun(((detected_lip_closure - deformer_lip_closure) * closure_confidence).view(-1, 2)**2))
    lip_closure_loss = torch.mean(torch.pow(detected_lip_closure - deformer_lip_closure, 2) * closure_confidence)

    closure_loss = eye_closure_loss + lip_closure_loss
    
    ict_landmark_loss = ict_landmark_loss
    ict_landmark_closure_loss = closure_loss


    






    # return ict_loss, random_ict_loss
    return ict_loss, random_ict_loss, ict_landmark_loss, ict_landmark_closure_loss

def ict_identity_regularization(ict_facekit):
    identity_loss = torch.mean(ict_facekit.identity ** 2)

    return identity_loss