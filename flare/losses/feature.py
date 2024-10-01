import torch
def feature_regularization_loss(feature, gt_facs, neural_blendshapes, facs_weight=0):
    facs = feature[..., :53]
    rotation = feature[..., 53:56]
    translation = feature[..., 56:59]
    scale = feature[..., 59]

    after_translation = feature[..., 60:63]
    
    bsize = facs.shape[0]

    # latent_regularization = torch.pow(scale - 1, 2).mean() 

    facs_reg = (facs - gt_facs).pow(2).mean() * 1e-3
 
    # z_reg = (torch.pow(translation[:, -1], 2).mean(3 + torch.pow(after_translation[:, -1], 2).mean() )
    l1_reg = (facs.clamp(1e-3)).pow(0.5).mean() * 1e-3
    # pseudo_l0_reg = (facs).clamp(min=1e-3).pow(0.5).mean() * 1e1

    # if facs is less than 0 or greater than 1, it will be penalized 
    range_reg = (facs.clamp(max=0).pow(2).mean() + (facs-1).clamp(min=0).pow(2).mean()) * 1e1

    mult_reg = neural_blendshapes.encoder.softplus(neural_blendshapes.encoder.bshapes_multiplier).pow(2).mean() 

    # loss =  latent_regularization   
    loss = facs_reg + range_reg + l1_reg + mult_reg
    
    return loss



 