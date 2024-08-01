import torch
def feature_regularization_loss(feature, gt_facs, neural_blendshapes, facs_weight=0):
    facs = feature[..., :53]
    rotation = feature[..., 53:56]
    translation = feature[..., 56:59]
    scale = feature[..., 59:]
    
    bsize = facs.shape[0]

    latent_regularization = torch.pow(scale - 1, 2).mean() 

    facs_reg = (facs - gt_facs).pow(2).mean()
 
    z_reg = torch.pow(translation[:, -1], 2).mean()

    l1_reg = (facs).clamp(min=0).mean() 
    # pseudo_l0_reg = (facs).clamp(min=1e-3).pow(0.5).mean() * 1e1

    # if facs is less than 0 or greater than 1, it will be penalized 
    range_reg = (facs.clamp(max=0).pow(2).mean() + (facs-1).clamp(min=0).pow(2).mean()) * 1e4

    # loss =  latent_regularization  
    loss =  latent_regularization  + facs_reg + z_reg + l1_reg + range_reg
    
    if facs_weight > 0:
        return loss + (torch.pow(facs - gt_facs, 2)).mean() * facs_weight
    return loss



 