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

def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)


def white_light_regularization(lights):
    # gray light regularization.
    self_occ_color = lights[..., :3]
    env_light_color = lights[..., 3:6]

    # self_occ_color to be gray and ones
    self_occ_loss = (self_occ_color - self_occ_color.mean(dim=-1, keepdim=True)).pow(2).mean() + (1 - self_occ_color).pow(2).mean()

    # env light to be gray and zero
    env_light_loss = (env_light_color - env_light_color.mean(dim=-1, keepdim=True)).pow(2).mean() + env_light_color.pow(2).mean()

    return self_occ_loss + env_light_loss

    lights_gray = lights.mean(dim=-1, keepdim=True)
    return (lights - lights_gray).pow(2).mean() + (1 - lights).pow(2).mean()

    lights = torch.cat([self_occ_color, env_light_color], dim=-1)
    return lights.pow(2).mean()
    diffuse_shading = lights[..., 6:]

    return (diffuse_shading - diffuse_shading.mean(dim=-1, keepdim=True)).abs().mean()

def material_regularization_function(values, semantic, mask, specular=False, roughness=False):
    assert specular or roughness, "At least one of specular or roughness should be True"
    skin_mask = ((semantic[..., 2] * mask[..., 0]) > 0.0).int().bool()
    skin_mask = skin_mask.view(-1)
    values_skin = values[skin_mask]
    mean = 0.3753 if specular else 0.5
    std = 0.1655 if specular else 0.1

    z_score = (values_skin - mean) / std

    loss = torch.mean(torch.max(torch.zeros_like(z_score), (torch.abs(z_score) - 2)))
    return loss


def cbuffers_regularization(cbuffers):
    material = cbuffers["material"]
    light = cbuffers["light"]

    # diffuse = material[..., :3]
    bsize = material.shape[0]
    
    # roughness to be zero
    loss = (roughness**2).mean()

    # light to be white
    loss += ((light[..., :3] - 1.0) ** 2).mean()

    return loss


def albedo_regularization(_adaptive, shader, mesh, device, displacements, iteration=0):
    position = mesh.vertices
    
    pe_input = shader.apply_pe(position=position, normalize=True)
    kd = shader.diffuse_mlp(pe_input)[..., :3]

    # add jitter for loss function
    jitter_pos = position + torch.normal(mean=0, std=0.01, size=position.shape, device=device)
    jitter_pe_input = shader.apply_pe(position=jitter_pos)

    kd_jitter = shader.diffuse_mlp(jitter_pe_input)[..., :3]
    loss_fn = torch.nn.MSELoss(reduction='none')
    kd_grad = loss_fn(kd_jitter, kd)
    loss = torch.mean(kd_grad.view(-1, 3))
    return loss


def white_light(cbuffers):
    loss = 0.0

    shading = cbuffers["shading"][..., :3]
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
    rho_skin = rho[mask]
    # pre-computed
    mean_rough = 0.3753
    std_rough = 0.1655
    z_score = (rho_skin-mean_rough) / std_rough

    loss = torch.mean(torch.max(torch.zeros_like(z_score), (torch.abs(z_score) - 2)))
    return loss