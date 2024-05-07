import torch
def feature_regularization_loss(feature, gt_facs, estim_lmks, gt_lmks, iterations, facs_adaptive, facs_weight=0):
    facs = feature[..., :53]
    rotation = feature[..., 53:56]
    translation = feature[..., 56:59]
    scale = feature[..., -1:]
    

    # facs regularization is to be pseudo L0 Norm
    facs_regularization = torch.mean(torch.pow(facs+1e-6, 0.75))

    # latent regularization: rotation, translation to be zero, scale to be 1
    latent_regularization = torch.mean(torch.pow(rotation, 2)) * 1e-1 +  1e2 * torch.mean(torch.pow(translation, 2)) 

    

    loss = facs_regularization * 1e-3 + latent_regularization + torch.mean(torch.pow(estim_lmks - gt_lmks[..., :3].reshape(-1, 68*3), 2))
    
    if facs_weight > 0:
        return loss + torch.mean(torch.pow(facs - gt_facs, 2)) * facs_weight
        # return loss + torch.mean(facs_adaptive.lossfun((facs - gt_facs)**2)) * facs_weight
    else:
        return loss