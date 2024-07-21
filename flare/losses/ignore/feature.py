import torch
def feature_regularization_loss(feature, gt_facs, neural_blendshapes, facs_weight=0):
    facs = feature[..., :53]
    rotation = feature[..., 53:56]
    translation = feature[..., 56:59]
    scale = feature[..., 59:]
    
    bsize = facs.shape[0]

<<<<<<< HEAD
    latent_regularization = (torch.pow(rotation, 2) * 1e-2 +  torch.pow(translation, 2) + torch.pow(scale - 1, 2)).mean() 

    loss =  latent_regularization 
=======
    latent_regularization = torch.pow(scale - 1, 2).mean() 

    facs_reg = (facs - gt_facs).pow(2).mean() * 1e-1
 
    loss =  latent_regularization  + facs_reg
>>>>>>> 1b71a7be5d1dd173b29e13c9613c78426d1ae066
    
    if facs_weight > 0:
        return loss + (torch.pow(facs - gt_facs, 2)).mean() * facs_weight
    return loss



