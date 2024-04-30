import torch
def feature_regularization_loss(feature, gt_facs, iterations):
    facs = feature[..., :53]
    rotation = feature[..., 53:56]
    translation = feature[..., 56:59]
    scale = feature[..., -1:]
    

    # facs regularization is to be pseudo L0 Norm
    facs_regularization = torch.mean(torch.pow(facs+1e-6, 0.75))

    # latent regularization: rotation, translation to be zero, scale to be 1
    latent_regularization = torch.mean(torch.pow(rotation, 2)) * 1e-2 + torch.mean(torch.pow(translation, 2)) + torch.mean(torch.pow(scale - 1, 2))


    loss = facs_regularization * 1e-3 + latent_regularization
    if iterations < 1000:
        return loss + torch.mean(torch.pow(facs - gt_facs, 2)) * 1e3
    else:
        return loss