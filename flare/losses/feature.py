import torch
def feature_regularization_loss(feature):
    facs = feature[..., :53]
    rotation = feature[..., 53:56]
    translation = feature[..., 56:59]
    scale = feature[..., -1:]
    

    # facs regularization is to be pseudo L0 Norm
    facs_regularization = torch.mean(torch.pow(facs+1e-6, 0.5))

    # latent regularization: rotation, translation to be zero, scale to be 1
    latent_regularization = torch.mean(torch.pow(rotation, 2)) + torch.mean(torch.pow(translation, 2)) + torch.mean(torch.pow(scale - 1, 2))


    loss = facs_regularization + latent_regularization
    return loss