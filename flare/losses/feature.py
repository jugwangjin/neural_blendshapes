import torch
def feature_regularization_loss(feature):
    facs = feature[:, :53]
    latent = feature[:, 53:-3]

    # facs regularization is to be pseudo L0 Norm
    facs_regularization = torch.mean(torch.pow(facs+1e-6, 0.5))

    latent_regularization = torch.mean(latent ** 2)

    loss = facs_regularization * 1e-1 + latent_regularization
    return loss