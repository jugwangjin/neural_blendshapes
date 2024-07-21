import torch
def feature_regularization_loss(feature, gt_facs, neural_blendshapes, facs_weight=0):
    facs = feature[..., :53]
    rotation = feature[..., 53:56]
    translation = feature[..., 56:59]
    scale = feature[..., 59:]
    
    bsize = facs.shape[0]

    latent_regularization = torch.pow(scale - 1, 2).mean() 

    facs_reg = (facs - gt_facs).pow(2).mean() 
 
    # loss =  latent_regularization  
    loss =  latent_regularization  + facs_reg
    
    if facs_weight > 0:
        return loss + (torch.pow(facs - gt_facs, 2)).mean() * facs_weight
    return loss



