import torch
def feature_regularization_loss(feature, gt_facs, neural_blendshapes, facs_weight=0):
    facs = feature[..., :53]
    rotation = feature[..., 53:56]
    translation = feature[..., 56:59]
    scale = feature[..., 59:]
    
    bsize = facs.shape[0]
    
    
    

    latent_regularization = (torch.pow(rotation, 2) * 1e-1 +  torch.pow(translation, 2) + torch.pow(scale - 1, 2)).mean() 

    sparsity_regularization = torch.pow(facs.clamp(1e-4), 0.5).mean() * 1e-2

    loss =  latent_regularization + sparsity_regularization
    
    if facs_weight > 0:
        return loss + (torch.pow(facs - gt_facs, 2)).mean() * facs_weight
    return loss
