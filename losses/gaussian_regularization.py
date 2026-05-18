import torch


def loss_scale_prior(log_scale, max_log_scale=0.0):
    return torch.nn.functional.relu(log_scale - max_log_scale).mean()
