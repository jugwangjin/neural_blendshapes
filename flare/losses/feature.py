import torch
def feature_regularization_loss(feature, gt_facs,  gt_lmks, model_scale, iterations, facs_adaptive, facs_weight=0):
    facs = feature[..., :53]
    rotation = feature[..., 53:56]
    translation = feature[..., 56:59]
    scale = model_scale
    
    bsize = facs.shape[0]
    
    facs_regularization = 0

    latent_regularization = (torch.pow(rotation, 2) * 1e-1 +  torch.pow(translation, 2) + torch.pow(scale - 1, 2)).reshape(bsize, -1).mean(dim=-1) 

    loss = facs_regularization + latent_regularization 
    
    if facs_weight > 0:
        return loss + (torch.pow(facs - gt_facs, 2)).reshape(bsize, -1).mean(dim=-1) * facs_weight
    return loss
