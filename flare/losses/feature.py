import torch
import pytorch3d.transforms as p3dt
def feature_regularization_loss(feature, gt_facs, neural_blendshapes, bshape_modulation, views, mode, random_flame_pose=None, random_landmark=None, facs_weight=0, mult=1, rot_mult=1, random_features_batch_size=64):
    facs = feature[..., :53]
    rotation = feature[..., 53:56]
    translation = feature[..., 56:59]
    global_translation = feature[..., 60:63]
    scale = feature[..., 59:60]

    eyeball_indices = neural_blendshapes.ict_facekit.left_eyeball_blendshape_indices + neural_blendshapes.ict_facekit.right_eyeball_blendshape_indices

    mode = mode.to(facs.device)

    # facs_reg_weights = (gt_facs).clamp(0, 1)  # min 0, max 1.  for 0 -> high weight, for 1 -> low weight. exponential decay
    
    # # facs_reg_weights = torch.exp(- 5 * facs_reg_weights)
    target_facs = (gt_facs).clamp(min=0, max=1)
    
    
    facs_reg = ((facs - target_facs)) 
    # facs_reg = ((facs - target_facs).pow(2) * facs_reg_weights
    facs_reg[:, eyeball_indices] *= 5e1
    facs_reg = facs_reg.pow(2).mean() * mult 

    # 새로운 인코더 구조에 맞게 수정
    # random_features = torch.randn(random_features_batch_size, neural_blendshapes.encoder.encoder.feature_size, device=facs.device) * 0.2
    # random_facs = torch.rand(random_features_batch_size, 53, device=facs.device) ** 3
    
    # random_mp_translation = torch.randn(random_features_batch_size, 3, device=facs.device) * 0.1
    # random_translation = torch.randn(random_features_batch_size, 3, device=facs.device) * 0.05

    # random_rotation = torch.randn(random_features_batch_size, 3, device=facs.device) * 0.2

    # fixed_indices = [0, 16, 36, 45, 27, 33]
    # random_landmark = random_landmark[:, fixed_indices, :3].reshape(-1, 18)

    # 새로운 인코더 구조 사용
    # layers_features = neural_blendshapes.encoder.encoder.layers_prefix(torch.cat([random_features, random_landmark, random_facs, random_rotation], dim=-1))
    # bshapes_out = neural_blendshapes.encoder.encoder.bshapes_tail(torch.cat([layers_features, random_landmark, random_facs, random_rotation], dim=-1))[:, :-10]

    facs_reg += feature[:, 63:].pow(2).mean() 
    # facs_reg += bshapes_out.pow(2).mean() + feature[:, 63:].pow(2).mean() * 1e-2

    translation_reg = (torch.pow(translation, 2).mean()) * 1e-4 + (torch.pow(global_translation, 2).mean()) * 1e-4 + (torch.pow(scale, 2).mean()) * 1e-4


    # loss =  facs_reg 
    loss =  facs_reg + translation_reg
    
    # 
    return loss



 