import torch
import pytorch3d.transforms as p3dt
def feature_regularization_loss(feature, gt_facs, neural_blendshapes, bshape_modulation, views, mode, facs_weight=0, mult=1, rot_mult=1):
    facs = feature[..., :53]
    rotation = feature[..., 53:56]
    translation = feature[..., 56:59]
    scale = feature[..., 59]

    after_translation = feature[..., 60:63]
    
    bsize = facs.shape[0]

    eyeball_indices = neural_blendshapes.ict_facekit.left_eyeball_blendshape_indices + neural_blendshapes.ict_facekit.right_eyeball_blendshape_indices

    mode = mode.to(facs.device)

    facs_reg_weights = (gt_facs - mode[None]).clamp(0, 1)  # min 0, max 1.  for 0 -> high weight, for 1 -> low weight. exponential decay
    facs_reg_weights = torch.exp(-facs_reg_weights)
    target_facs = (gt_facs - mode[None]).clamp(min=0, max=1)
    
    facs_reg = ((facs - target_facs).pow(2) * facs_reg_weights)
    facs_reg[:, eyeball_indices] *= 1e2
    facs_reg = facs_reg.mean() * 1e-1 * mult + feature[:, -53:].pow(2).mean()
 
    l1_reg = (facs).abs().mean() * 1e-3
    
    range_reg = (facs.clamp(max=0).pow(2).mean() + (facs-1).clamp(min=0).pow(2).mean()) * 1e2

    transform_matrix = views['mp_transform_matrix'].reshape(-1, 4, 4).detach()
    scale = torch.norm(transform_matrix[:, :3, :3], dim=-1).mean(dim=-1, keepdim=True)
    rotation_matrix = transform_matrix[:, :3, :3]
    rotation_matrix = transform_matrix[:, :3, :3] / scale[:, None]

    rotation_matrix = rotation_matrix.permute(0, 2, 1)
    mp_rotation = p3dt.matrix_to_euler_angles(rotation_matrix, convention='XYZ')

    rotation_reg = (torch.pow(rotation - mp_rotation, 2).mean()) * 1e-3 * rot_mult


    transform_origin = neural_blendshapes.encoder.transform_origin
    
    transform_origin_reg = (torch.pow(transform_origin[0], 2).mean() + torch.pow(after_translation, 2).mean() ) * 1e-1

    # loss =  latent_regularization   
    loss =  l1_reg + facs_reg + rotation_reg + transform_origin_reg + range_reg
    # print(f"l1_reg: {l1_reg.item():.4f}, facs_reg: {facs_reg.item():.4f}, rotation_reg: {rotation_reg.item():.4f}, transform_origin_reg: {transform_origin_reg.item():.4f}, range_reg: {range_reg.item():.4f}")
    
    return loss



 