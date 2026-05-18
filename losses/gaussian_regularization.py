import torch


def loss_scale_prior(log_scale, max_log_scale=0.0):
    return torch.nn.functional.relu(log_scale - max_log_scale).mean()


def loss_scale(scale, max_scale=0.05):
    return torch.nn.functional.relu(scale - max_scale).mean()


def loss_opacity(opacity):
    return (opacity - 0.5).pow(2).mean()
