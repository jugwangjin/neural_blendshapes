import torch
def feature_regularization_loss(feature, gt_facs,  gt_lmks, model_scale, iterations, facs_adaptive, facs_weight=0):
    facs = feature[..., :53]
    rotation = feature[..., 53:56]
    translation = feature[..., 56:59]
    scale = model_scale
    
    bsize = facs.size(0)
    if bsize > 1:
        gt_facs_sign = -(gt_facs[None] - gt_facs[:, None]) # shape of B, B, 53
        facs_diff = (facs[None] - facs[:, None]) # shape of B, B, 53
    
        facs_loss = ((facs_diff * gt_facs_sign) + 1).reshape(bsize, -1).mean(dim=-1)
    else:
        facs_loss = torch.tensor(0)
    
    facs_regularization = facs_loss * 1e3 + (torch.pow(facs + 1e-2, 0.75)).reshape(bsize, -1).mean(dim=-1)

    latent_regularization = (torch.pow(rotation, 2) * 1e-1 +  torch.pow(translation, 2) + torch.pow(scale - 1, 2)).reshape(bsize, -1).mean(dim=-1) 

    loss = facs_regularization + latent_regularization 
    
    if facs_weight > 0:
        return loss + (torch.abs(facs - gt_facs)).reshape(bsize, -1).mean(dim=-1) * facs_weight
    return loss
