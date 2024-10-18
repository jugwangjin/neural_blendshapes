import torch
import pytorch3d.transforms as p3dt
def feature_regularization_loss(feature, gt_facs, neural_blendshapes, bshape_modulation, views, facs_weight=0, mult=1):
    facs = feature[..., :53]
    rotation = feature[..., 53:56]
    translation = feature[..., 56:59]
    scale = feature[..., 59]

    after_translation = feature[..., 60:63]
    
    bsize = facs.shape[0]

    # latent_regularization = torch.pow(scale - 1, 2).mean() 
    facs_reg = (facs - gt_facs).pow(2).mean() * mult
 
    # z_reg = (torch.pow(translation[:, -1], 2).mean(3 + torch.pow(after_translation[:, -1], 2).mean() )
    l1_reg = (facs.clamp(1e-3)).pow(0.75).mean() * 1e-2
    # pseudo_l0_reg = (facs).clamp(min=1e-3).pow(0.5).mean() * 1e1

    # if facs is less than 0 or greater than 1, it will be penalized 
    range_reg = (facs.clamp(max=0).pow(2).mean() + (facs-1).clamp(min=0).pow(2).mean()) * 1e2

    transform_matrix = views['mp_transform_matrix'].reshape(-1, 4, 4).detach()
    scale = torch.norm(transform_matrix[:, :3, :3], dim=-1).mean(dim=-1, keepdim=True)
    rotation_matrix = transform_matrix[:, :3, :3]
    rotation_matrix = transform_matrix[:, :3, :3] / scale[:, None]

    rotation_matrix = rotation_matrix.permute(0, 2, 1)
    mp_rotation = p3dt.matrix_to_euler_angles(rotation_matrix, convention='XYZ')

    rotation_reg = torch.pow(rotation - mp_rotation, 2).mean() * mult

    transform_origin = neural_blendshapes.encoder.transform_origin
    
    transform_origin_reg = torch.pow(transform_origin[0], 2).mean() + torch.pow(translation, 2).mean()

    # loss =  latent_regularization   
    loss =  l1_reg + facs_reg + rotation_reg + transform_origin_reg + range_reg
    
    return loss



 