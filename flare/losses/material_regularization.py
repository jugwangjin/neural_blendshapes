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

import torch.nn.functional as F

high_res = (512, 512)
def upsample(buffer, high_res):
    if buffer.shape[1] == high_res[0] and buffer.shape[2] == high_res[1]:
        return buffer
    # Convert from (B, H, W, C) -> (B, C, H, W)
    buffer = buffer.permute(0, 3, 1, 2)
    
    # Perform bilinear upsampling
    upsampled = F.interpolate(buffer, size=high_res, mode='bilinear', align_corners=False)
    
    # Convert back from (B, C, H, W) -> (B, H, W, C)
    return upsampled.permute(0, 2, 3, 1)



def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)


def albedo_regularization(_adaptive, shader, mesh, device, displacements, iteration=0, mult=None):
    position = mesh.vertices
    
    pe_input = shader.apply_pe(position=position)
    kd = shader.material_mlp(pe_input)[..., :4]

    # add jitter for loss function
    jitter_pos = position + torch.normal(mean=0, std=0.01, size=position.shape, device=device)
    jitter_pe_input = shader.apply_pe(position=jitter_pos)

    kd_jitter = shader.material_mlp(jitter_pe_input)[..., :4]
    loss_fn = torch.nn.MSELoss(reduction='none')
    kd_grad = loss_fn(kd_jitter, kd)
    loss = torch.mean(_adaptive.lossfun(kd_grad.view(-1, 4)))
    if mult is None:
        return loss * min(1.0, iteration / 6000)
    else:
        return loss * mult


def white_light(cbuffers):
    loss = 0.0

    shading = cbuffers["shading"]
    white = (shading[..., 0:1] + shading[..., 1:2] + shading[..., 2:3]) / 3.0
    masked_pts = (shading - white)
    loss += torch.mean(torch.abs(masked_pts))

    return loss

def roughness_regularization(roughness, semantic, mask, r_mean):
    skin_mask = (torch.sum(semantic[..., 2:3], axis=-1)).unsqueeze(-1)
    skin_mask = skin_mask * mask 

    loss = 0.0
    mask = (skin_mask > 0.0).int().bool()
    roughness_skin = roughness[mask]
    # Ablation tablutaed in Section 5.
    mean_rough = r_mean # 0.5 default
    std_rough = 0.100
    z_score = (roughness_skin-mean_rough) / std_rough

    loss = torch.mean(torch.max(torch.zeros_like(z_score), (torch.abs(z_score) - 2)))
    return loss

def spec_intensity_regularization(rho, semantic, mask):
    skin_mask = (torch.sum(semantic[..., 2:3], axis=-1)).unsqueeze(-1)
    skin_mask = skin_mask * mask 

    loss = 0.0
    mask = (skin_mask > 0.0).int().bool()

    rho = upsample(rho, high_res)
    rho_skin = rho[mask]
    # pre-computed
    mean_rough = 0.3753
    std_rough = 0.1655
    z_score = (rho_skin-mean_rough) / std_rough

    loss = torch.mean(torch.max(torch.zeros_like(z_score), (torch.abs(z_score) - 2)))
    return loss