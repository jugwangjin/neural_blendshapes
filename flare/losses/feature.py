import torch
def feature_regularization_loss(feature, gt_facs, neural_blendshapes, facs_weight=0):
    facs = feature[..., :53]
    rotation = feature[..., 53:56]
    translation = feature[..., 56:59]
    scale = feature[..., 59:]
    
    bsize = facs.shape[0]

    latent_regularization = (torch.pow(rotation, 2) * 1e-2 +  torch.pow(translation, 2) + torch.pow(scale - 1, 2)).mean() 

    # sparsity_regularization = torch.pow(facs.clamp(1e-2), 0.5).mean() * 1e-3
    # add large regularization for features larger than 1 
    # sparsity_regularization += torch.pow((facs-1).clamp(0), 2).mean() * 1e5

    # sparsity_regularization = torch.pow(facs - gt_facs, 2).mean() * 1e-2

    # panelty if facs is smaller than gt_facs
    sparsity_regularization = torch.pow((gt_facs - facs), 2).mean() 

    # panelty if facs is larger than 1
    sparsity_regularization += torch.pow((facs-1).clamp(0), 2).mean() * 1e5

    # identity_regularization = torch.pow(neural_blendshapes.encoder.identity_code, 2).mean() 

    loss =  latent_regularization + sparsity_regularization 
    
    if facs_weight > 0:
        return loss + (torch.pow(facs - gt_facs, 2)).mean() * facs_weight
    return loss



