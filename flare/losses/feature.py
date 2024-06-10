import torch
def feature_regularization_loss(feature, gt_facs, neural_blendshapes, facs_weight=0):
    facs = feature[..., :53]
    rotation = feature[..., 53:56]
    translation = feature[..., 56:59]
    scale = feature[..., 59:]
    
    bsize = facs.shape[0]
    
    

    latent_regularization = (torch.pow(rotation, 2) * 1e-1 +  torch.pow(translation, 2) + torch.pow(scale - 1, 2)).mean() 
                            #  + (neural_blendshapes.encoder.softplus(neural_blendshapes.encoder.blendshapes_bias) - 1).pow(2).mean()

    loss =  latent_regularization 
    
    if facs_weight > 0:
        return loss + (torch.pow(facs - gt_facs, 2)).mean() * facs_weight
    return loss
