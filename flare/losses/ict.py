
import pytorch3d.transforms as p3dt
import pytorch3d.ops as p3do
import torch

import pytorch3d.transforms as pt3d
DIRECTION_PAIRS = torch.tensor([[36, 64],[45, 48]]).int()

EYELID_PAIRS = torch.tensor([[37, 41], [38, 40], [43, 47], [44, 46]]).int()
LIP_PAIRS = torch.tensor([[61, 67], [62, 66], [63, 65]]).int()

def align_to_canonical(source, target, landmark_indices):
    to_canonical = p3do.corresponding_points_alignment(source[:, landmark_indices], target[:, landmark_indices], estimate_scale=True)

    aligned = torch.einsum('bni, bji -> bnj', source, to_canonical.R)
    aligned = (to_canonical.s[:, None, None] * aligned + to_canonical.T[:, None])

    return aligned, to_canonical

def ict_loss(ict_facekit, return_dict, views_subset, neural_blendshapes, renderer, gbuffers):
    template_deformation = return_dict['full_template_deformation']
    frontal_indices = ict_facekit.face_indices + ict_facekit.eyeball_indices

    ict_loss = torch.mean(torch.pow(template_deformation[frontal_indices], 2))

    # return ict_loss, random_ict_loss
    return ict_loss

def ict_identity_regularization(ict_facekit):
    identity_loss = torch.mean(ict_facekit.identity ** 2)

    return identity_loss