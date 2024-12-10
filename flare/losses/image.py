# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import torch



import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

#----------------------------------------------------------------------------
# HDR image losses
#----------------------------------------------------------------------------

def _tonemap_srgb(f):
    return torch.where(f > 0.0031308, torch.pow(torch.clamp(f, min=0.0031308), 1.0/2.4)*1.055 - 0.055, 12.92*f)


def image_loss_fn(img, target):

    img    = _tonemap_srgb(torch.log(torch.clamp(img, min=0, max=65535) + 1))
    target = _tonemap_srgb(torch.log(torch.clamp(target, min=0, max=65535) + 1))

    out = (img - target).abs().mean()
    # out = torch.nn.functional.l1_loss(img, target)
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of image_loss contains inf or NaN"
    return out, img, target


def mask_loss_function(masks, gbuffers, epsilon=1e-8, loss_function = torch.nn.MSELoss()):
    """ Compute the mask term as the mean difference between the original masks and the rendered masks.
    
    Args:
        views (List[View]): Views with masks
        gbuffers (List[Dict[str, torch.Tensor]]): G-buffers for each view with the 'mask' channel
        loss_function (Callable): Function for comparing the masks or generally a set of pixels
    """
    
    intersection = (masks * gbuffers).sum(dim=(1, 2, 3))
    dice_union = masks.sum(dim=(1, 2, 3)) + gbuffers.sum(dim=(1, 2, 3)) + epsilon

    # Compute Dice-based loss (aligned with L_latent)
    iou_loss = 1 - (2 * intersection / dice_union).mean() 
        
    mse_loss = (masks - gbuffers).pow(2).mean()

    return mse_loss + iou_loss


def segmentation_loss_function(masks, gbuffers, epsilon=1e-8):
    intersection = (masks * gbuffers).sum(dim=(1, 2, 3))
    dice_union = masks.sum(dim=(1, 2, 3)) + gbuffers.sum(dim=(1, 2, 3)) + epsilon

    # Compute Dice-based loss (aligned with L_latent)
    iou_loss = 1 - (2 * intersection / dice_union).mean() 
        
    mse_loss = (masks - gbuffers).pow(2).mean()

    return mse_loss + iou_loss

def shading_loss_batch(pred_color_masked, views, batch_size):
    """ Compute the image loss term as the mean difference between the original images and the rendered images from a shader.
    """

    color_loss, tonemap_pred, tonemap_target = image_loss_fn(pred_color_masked[..., :3] * views["mask"], views["img"] * views["mask"])

    
    input = tonemap_pred.permute(0, 3, 1, 2)
    target = tonemap_target.permute(0, 3, 1, 2)

    ssim_loss = (1.0 - ssim(input, target))

    color_loss = 0.95 * color_loss + 0.05 * ssim_loss

    return color_loss, pred_color_masked[..., :3], [tonemap_pred, tonemap_target]





