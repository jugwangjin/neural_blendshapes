import torch
import pytorch3d.transforms as p3dt
def feature_regularization_loss(feature, gt_facs, neural_blendshapes, bshape_modulation, views, mode, facs_weight=0, mult=1, rot_mult=1, random_features_batch_size=8):
    facs = feature[..., :53]
    rotation = feature[..., 53:56]
    translation = feature[..., 56:59]
    scale = feature[..., 59]

    after_translation = feature[..., 60:63]
    
    bsize = facs.shape[0]

    eyeball_indices = neural_blendshapes.ict_facekit.left_eyeball_blendshape_indices + neural_blendshapes.ict_facekit.right_eyeball_blendshape_indices

    mode = mode.to(facs.device)

    # facs_reg_weights = (gt_facs).clamp(0, 1)  # min 0, max 1.  for 0 -> high weight, for 1 -> low weight. exponential decay
    
    # facs_reg_weights = torch.exp(- 5 * facs_reg_weights)
    target_facs = (gt_facs).clamp(min=0, max=1)
    
    
    facs_reg = ((facs - target_facs)) 
    # facs_reg = ((facs - target_facs).pow(2) * facs_reg_weights)
    facs_reg[:, eyeball_indices] *= 1e1
    facs_reg = facs_reg.pow(2).mean() * mult + feature[..., -53:].pow(2).mean() 

    random_features = torch.randn(random_features_batch_size, neural_blendshapes.encoder.encoder.feature_size, device=facs.device) * 0.02
    random_facs = torch.rand(random_features_batch_size, 53, device=facs.device)

    bshapes_additional_features = neural_blendshapes.encoder.encoder.blendshapes_prefix(random_facs)
    bshapes_out = neural_blendshapes.encoder.encoder.bshapes_tail(torch.cat([random_features, bshapes_additional_features], dim=-1))

    facs_reg += bshapes_out.pow(2).mean() 
    

    transform_matrix = views['mp_transform_matrix'].reshape(-1, 4, 4).detach()
    scale = torch.norm(transform_matrix[:, :3, :3], dim=-1).mean(dim=-1, keepdim=True)
    rotation_matrix = transform_matrix[:, :3, :3]
    rotation_matrix = transform_matrix[:, :3, :3] / scale[:, None]

    rotation_matrix = rotation_matrix.permute(0, 2, 1)
    mp_rotation = p3dt.matrix_to_euler_angles(rotation_matrix, convention='XYZ')

    rotation_reg = (torch.pow(rotation - mp_rotation, 2).mean()) * rot_mult

    loss =  facs_reg + rotation_reg
    
    return loss



 