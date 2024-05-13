import torch
def feature_regularization_loss(feature, gt_facs,  gt_lmks, model_scale, iterations, facs_adaptive, facs_weight=0):
    facs = feature[..., :53]
    rotation = feature[..., 53:56]
    translation = feature[..., 56:59]
    scale = model_scale
    
    # facs: shape of B, 53
    # gt facs: shape of B, 53
    # I would like to extract only signs of gt facs 
    bsize = facs.size(0)
    if bsize > 1:
        gt_facs_sign = -10 * (gt_facs[None] - gt_facs[:, None]) # shape of B, B, 53
        # gt_facs_sign = -torch.sign(gt_facs_sign) # shape of B, B, 53

        facs_diff = facs[None] - facs[:, None] # shape of B, B, 53
    
        facs_loss = torch.mean(facs_diff * gt_facs_sign)
    else:
        facs_loss = torch.tensor(0)
    

    # facs regularization is to be pseudo L0 Norm
    # facs_regularization = torch.mean(torch.pow(facs+1e-1, 0.75)) * 1e-4
    facs_regularization = facs_loss * 5e3 + torch.mean(torch.pow(facs+1e-1, 0.75)) * 1e-2

    # latent regularization: rotation, translation to be zero, scale to be 1
    latent_regularization = torch.mean(torch.pow(rotation, 2)) * 1e-1 +  torch.mean(torch.pow(translation, 2)) + torch.mean(torch.pow(scale - 1, 2))

    # lmk_regularization = torch.mean(torch.pow(estim_lmks - gt_lmks[..., :3].reshape(-1, 68*3), 2)) * 1e2

    loss = facs_regularization + latent_regularization 
    
    if facs_weight > 0:
        return loss + torch.mean(torch.abs(facs - gt_facs)) * facs_weight
        # return loss + torch.mean(facs_adaptive.lossfun((facs - gt_facs)**2)) * facs_weight
    # else:
    return loss
