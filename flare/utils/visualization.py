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

from pathlib import Path
import torch.nn.functional as F
import torch
import torchvision.utils as torch_utils
import numpy as np
from PIL import Image
import imageio 
from flare.dataset import dataset_util

# ==============================================================================================
# util functions
# ==============================================================================================
def add_directionlight(normals, device, light_pos=None):
    '''
        normals: [bz, nv, 3]
        lights: [bz, nlight, 6]
    returns:
        shading: [bz, nv, 3]
    '''
    if light_pos is not None:
        light_positions = light_pos[None, :, :].expand(1, -1, -1).float().to(device)
    else:
        light_positions = torch.Tensor([
                    [-1,1,1],
                    [1,1,1],
                    [-1,-1,1],
                    [1,-1,1],
                    [0,0,1]
                    ])[None,:,:].expand(1, -1, -1).float().to(device)
    light_intensities = (torch.ones_like(light_positions).float()*1.2).to(device)
    lights = torch.cat((light_positions, light_intensities), 2).to(device)
    light_direction = lights[:,:,:3]; light_intensities = lights[:,:,3:]
    directions_to_lights = F.normalize(light_direction[:,:,None,:].expand(-1,-1,normals.shape[1],-1), dim=3).to(device)
    normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
    shading = normals_dot_lights[:,:,:,None]*light_intensities[:,:,None,:]
    return shading.mean(1)

def add_directionlight_(normals, device, light_pos):
    '''
        normals: [bz, nv, 3]
        lights: [bz, nlight, 6]
    returns:
        shading: [bz, nv, 3]
    '''

    light_positions = light_pos[None, :, :].expand(1, -1, -1).float().to(device)
    light_intensities = (torch.ones_like(light_positions).float()*1.2).to(device)
    directions_to_lights = F.normalize(light_positions[:,:,None,:].repeat(1, 1, normals.shape[1], 1), dim=3).to(device)
    normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
    shading = normals_dot_lights[:,:,:,None]*light_intensities[:,:,None,:]
    return shading.mean(1)

def list_torchgrid(save_list, grid_path="out/", save_name="test.png", nrow=5, save=True, scale_factor=255, normalize=False, scale_each=False):
    def lin2img(tensor):
        batch_size, H, W, channels = tensor.shape
        return tensor.permute(0, 3, 1, 2)

    if not torch.is_tensor(save_list):
        concatenate_imgs = torch.cat(save_list, dim=0)
    else:
        concatenate_imgs = save_list
    concatenate_imgs = lin2img(concatenate_imgs)

    grid = torch_utils.make_grid(concatenate_imgs, scale_each=scale_each, normalize=normalize, nrow=nrow)
    grid = grid.permute(1, 2, 0)
    tensor = (grid * scale_factor).to(torch.uint8)

    if save:
        img = Image.fromarray(tensor.detach().cpu().numpy())
        img.save(grid_path / save_name)
    else:
        return tensor

def _tonemap_srgb(f):
    return torch.where(f > 0.0031308, torch.pow(torch.clamp(f, min=0.0031308), 1.0/2.4)*1.055 - 0.055, 12.92*f)

def reinhard(f: torch.Tensor) -> torch.Tensor:
    return f/(1+f)

def unwrap_cubemap(f, device):
    convert2row = lambda x: list_torchgrid(x, nrow=len(x), save=False, scale_factor=255)
    
    zero_tensor = torch.zeros((1, 512, 512, 3)).to(device)
    row_1 = [zero_tensor, f[2].unsqueeze(0), zero_tensor, zero_tensor]
    row_2 = [f[1].unsqueeze(0), f[5].unsqueeze(0),f[0].unsqueeze(0), f[4].unsqueeze(0)]
    row_3 = [zero_tensor,f[3].unsqueeze(0), zero_tensor, zero_tensor]
    return [convert2row(row_1).unsqueeze(0), convert2row(row_2).unsqueeze(0), convert2row(row_3).unsqueeze(0)]

def add_buffer(cbuffers, gbuffer_mask, color_list, convert_uint):
    H, W = 512, 512
    grid_path = "test"
    device = cbuffers["specu"].device
    if 'roughness' in cbuffers:
        k_r = cbuffers['roughness'].reshape(-1, H, W, 1) 
        color_list += [list_torchgrid(k_r, grid_path, save_name=None, nrow=1, save=False, scale_factor=255).unsqueeze(0)]
    if 'specular_intensity' in cbuffers:
        ko = cbuffers['specular_intensity'].reshape(-1, H, W, 1) 
        color_list += [list_torchgrid(ko, grid_path, save_name=None, nrow=1, save=False, scale_factor=255).unsqueeze(0)]

def diffuse_specular(cbuffers, gbuffer_mask, color_list, convert_uint):
    H, W = 512, 512


def save_shading(shaded_image, cbuffers, debug_gbuffer, debug_view, images_save_path, iteration, save_name=None, ict_facekit=None, transparency=False):
    device = shaded_image.device
    convert_uint = lambda x: torch.from_numpy(np.clip(np.rint(dataset_util.rgb_to_srgb(x).detach().cpu().numpy() * 255.0), 0, 255).astype(np.uint8)).to(device)

    gbuffer_mask = debug_gbuffer["mask"]
    Bz, H, W, _ = debug_gbuffer["mask"].shape
    # Save a normal map in camera space
    R = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], device=device, dtype=torch.float32)
    normal_image = (0.5*(debug_gbuffer["normal"] @ debug_view["camera"][0].R.T @ R.T + 1)) * gbuffer_mask 

    shading = add_directionlight(debug_gbuffer["normal"].reshape([1, -1, 3]), device)
    shading = shading.reshape(debug_gbuffer["normal"].shape)
    shading = shading * gbuffer_mask 

    shading = shading.permute(0, 3, 1, 2)
    shading = shading * 255

    if len(shading.shape) == 3:
        shading = shading.unsqueeze(0)

    shading = shading[0].detach().cpu().numpy().astype(np.uint8).transpose(1, 2, 0)

    if transparency:
        alpha_channel = (gbuffer_mask[0, :, :, 0] * 255).detach().cpu().numpy().astype(np.uint8)
        shading = np.dstack((shading, alpha_channel))

    img = Image.fromarray(shading)
    img.save(images_save_path / save_name)
    return






# ==================== visualizations =================================
def visualize_training(shaded_image, cbuffers, debug_gbuffer, debug_view, images_save_path, iteration, save_name=None, ict_facekit=None):
    device = shaded_image.device
    convert_uint = lambda x: torch.from_numpy(np.clip(np.rint(dataset_util.rgb_to_srgb(x).detach().cpu().numpy() * 255.0), 0, 255).astype(np.uint8)).to(device)

    gbuffer_mask = debug_gbuffer["mask"]
    Bz, H, W, _ = debug_gbuffer["mask"].shape
    grid_path = (images_save_path / "grid")
    grid_path.mkdir(parents=True, exist_ok=True)

    grid_path_each = (images_save_path / "grid_each")
    grid_path_each.mkdir(parents=True, exist_ok=True)

    # Save a normal map in camera space
    R = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], device=device, dtype=torch.float32)
    normal_image = (0.5*(debug_gbuffer["normal"] @ debug_view["camera"][0].R.T @ R.T + 1)) * gbuffer_mask 

    shading = add_directionlight(debug_gbuffer["normal"].reshape([1, -1, 3]), device)
    shading = shading.reshape(debug_gbuffer["normal"].shape)
    shading = shading * gbuffer_mask 


    # both landmarks_on_clip_space and detected_landmarks are in clip space -> x, y in range of [-1, 1]

    landmark_indices = ict_facekit.landmark_indices
    landmarks_on_clip_space = debug_gbuffer['deformed_verts_clip_space'].clone()[:, landmark_indices]
    landmarks_on_clip_space = landmarks_on_clip_space[..., :3] / torch.clamp(landmarks_on_clip_space[..., 3:4], min=1e-8) 
    

    detected_landmarks = debug_view['landmark'].clone().detach()
    detected_landmarks[..., :2] = detected_landmarks[..., :2] * 2 - 1
    # detected_landmarks[..., 2] = detected_landmarks[..., 2] * -1
    
    org_images = debug_view['img'].clone().detach()

    # draw landmarks on shaded image, debug_view["img"], normal_image, shading
    # landmarks on clip space for red 3x3 dot
    # detected landmarks for green 3x3 dot
    for i in range(Bz):
        for j in range(landmarks_on_clip_space.shape[1]):
            x, y = int((landmarks_on_clip_space[i, j, 0] + 1) * W / 2), int((landmarks_on_clip_space[i, j, 1] + 1) * H / 2)
            shaded_image[i, y-1:y+2, x-1:x+2] = torch.tensor([1, 0, 0], device=device)
            # org_images[i, y-1:y+2, x-1:x+2] = torch.tensor([1, 0, 0], device=device)
            normal_image[i, y-1:y+2, x-1:x+2] = torch.tensor([1, 0, 0], device=device)
            shading[i, y-1:y+2, x-1:x+2] = torch.tensor([1, 0, 0], device=device)

            x_, y_ = int((detected_landmarks[i, j, 0] + 1) * W / 2), int((detected_landmarks[i, j, 1] + 1) * H / 2)
            confidences = detected_landmarks[i, j, -1]
            shaded_image[i, y_-1:y_+2, x_-1:x_+2] = torch.tensor([0, 1, 0], device=device) * confidences
            org_images[i, y_-1:y_+2, x_-1:x_+2] = torch.tensor([0, 1, 0], device=device) * confidences
            normal_image[i, y_-1:y_+2, x_-1:x_+2] = torch.tensor([0, 1, 0], device=device) * confidences
            shading[i, y_-1:y_+2, x_-1:x_+2] = torch.tensor([0, 1, 0], device=device) * confidences

    save_individual_img(shaded_image, debug_view, normal_image, gbuffer_mask, cbuffers, grid_path_each)
    color_list = []
    color_list += [list_torchgrid(convert_uint(org_images), grid_path, save_name=None, nrow=1, save=False, scale_factor=1).unsqueeze(0)]
    color_list += [list_torchgrid(convert_uint(shaded_image), grid_path, save_name=None, nrow=1, save=False, scale_factor=1).unsqueeze(0)]

    # if 'shading' in cbuffers:
    #     color_list += [list_torchgrid(convert_uint(cbuffers["albedo"]), grid_path, save_name=None, nrow=1, save=False, scale_factor=1).unsqueeze(0)]
    #     color_list += [list_torchgrid(convert_uint(cbuffers["shading"] * gbuffer_mask), grid_path, save_name=None, nrow=1, save=False, scale_factor=1).unsqueeze(0)]
    #     add_buffer(cbuffers, gbuffer_mask, color_list, convert_uint) ## visualize roughness and specular intensity
    color_list += [list_torchgrid(normal_image, grid_path, save_name=None, nrow=1, save=False, scale_factor=255).unsqueeze(0)]
    color_list += [list_torchgrid(shading.to(device), grid_path, save_name=None, nrow=1, save=False, scale_factor=255).unsqueeze(0)]

    if 'canonical_position' in debug_gbuffer:
        canonical_position = debug_gbuffer['canonical_position']
        canonical_position = (canonical_position + 1) / 2
        canonical_position = canonical_position * gbuffer_mask
        color_list += [list_torchgrid(convert_uint(canonical_position), grid_path, save_name=None, nrow=1, save=False, scale_factor=255).unsqueeze(0)]

    if save_name is None:
        save_name = f'grid_{iteration}.png'
    else:
        save_name = f'grid_{iteration}_{save_name}.png'
    list_torchgrid(color_list, grid_path, save_name, nrow=len(color_list), scale_factor=1)
    del color_list

    return shaded_image




# ==================== visualizations =================================
def visualize_training_no_lm(shaded_image, cbuffers, debug_gbuffer, debug_view, images_save_path, iteration, save_name=None, ict_facekit=None):
    device = shaded_image.device
    convert_uint = lambda x: torch.from_numpy(np.clip(np.rint(dataset_util.rgb_to_srgb(x).detach().cpu().numpy() * 255.0), 0, 255).astype(np.uint8)).to(device)

    gbuffer_mask = debug_gbuffer["mask"]
    Bz, H, W, _ = debug_gbuffer["mask"].shape
    grid_path = (images_save_path / "grid")
    grid_path.mkdir(parents=True, exist_ok=True)

    grid_path_each = (images_save_path / "grid_each")
    grid_path_each.mkdir(parents=True, exist_ok=True)

    # Save a normal map in camera space
    R = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], device=device, dtype=torch.float32)
    normal_image = (0.5*(debug_gbuffer["normal"] @ debug_view["camera"][0].R.T @ R.T + 1)) * gbuffer_mask 

    shading = add_directionlight(debug_gbuffer["normal"].reshape([1, -1, 3]), device)
    shading = shading.reshape(debug_gbuffer["normal"].shape)
    shading = shading * gbuffer_mask 



    save_individual_img(shaded_image, debug_view, normal_image, gbuffer_mask, cbuffers, grid_path_each)
    color_list = []
    color_list += [list_torchgrid(convert_uint(debug_view['img'].clone().detach()), grid_path, save_name=None, nrow=1, save=False, scale_factor=1).unsqueeze(0)]
    color_list += [list_torchgrid(convert_uint(shaded_image), grid_path, save_name=None, nrow=1, save=False, scale_factor=1).unsqueeze(0)]

    # if 'shading' in cbuffers:
    #     color_list += [list_torchgrid(convert_uint(cbuffers["albedo"]), grid_path, save_name=None, nrow=1, save=False, scale_factor=1).unsqueeze(0)]
    #     color_list += [list_torchgrid(convert_uint(cbuffers["shading"] * gbuffer_mask), grid_path, save_name=None, nrow=1, save=False, scale_factor=1).unsqueeze(0)]
    #     add_buffer(cbuffers, gbuffer_mask, color_list, convert_uint) ## visualize roughness and specular intensity
    color_list += [list_torchgrid(normal_image, grid_path, save_name=None, nrow=1, save=False, scale_factor=255).unsqueeze(0)]
    color_list += [list_torchgrid(shading.to(device), grid_path, save_name=None, nrow=1, save=False, scale_factor=255).unsqueeze(0)]

    # if there is canonical_position in debug_gbuffer, visualize it
    # assume that the canonical position is in [-1, 1]

    if save_name is None:
        save_name = f'grid_{iteration}.png'
    else:
        save_name = f'grid_{iteration}_{save_name}.png'
    list_torchgrid(color_list, grid_path, save_name, nrow=len(color_list), scale_factor=1)
    del color_list

    return shaded_image


# ==============================================================================================
# SAVE IMAGE FOR QUALITATIVE EVALUATION
# ==============================================================================================  
def save_relit_intrinsic_materials(relit_imgs, views, gbuffer_mask, buffers, images_save_path):
    convert_uint = lambda x: np.clip(np.rint(dataset_util.rgb_to_srgb(x).numpy() * 255.0), 0, 255).astype(np.uint8) 
    convert_uint_wo_mask = lambda x: torch.from_numpy(np.clip(np.rint(dataset_util.rgb_to_srgb(x).detach().cpu().numpy() * 255.0), 0, 255).astype(np.uint8))

    Path(images_save_path / "albedo").mkdir(parents=True, exist_ok=True)
    Path(images_save_path / "roughness").mkdir(parents=True, exist_ok=True)
    Path(images_save_path / "specular_intensity").mkdir(parents=True, exist_ok=True)

    for i in range(len(gbuffer_mask)):
        mask = gbuffer_mask[i].cpu()
        id = int(views["frame_name"][i])

        ### intrinsic materials
        imageio.imsave(images_save_path / "albedo" / f'{id:0d}.png', convert_uint(torch.cat([buffers['albedo'][i].cpu(), mask], -1)))
        imageio.imsave(images_save_path / "roughness" / f'{id:0d}.png', convert_uint(torch.cat([buffers['roughness'][i].cpu().repeat(1, 1, 3), mask], -1)))
        imageio.imsave(images_save_path / "specular_intensity" / f'{id:0d}.png', convert_uint(torch.cat([buffers['specular_intensity'][i].cpu().repeat(1, 1, 3), mask], -1)))

    for i in range(len(relit_imgs)):
        for j in range(len(gbuffer_mask)):
            mask = gbuffer_mask[j].cpu()
            id = int(views["frame_name"][j])
            imageio.imsave(images_save_path / f"env_map_{i}" / f'{id:05d}.png', convert_uint_wo_mask(relit_imgs[i][j])) 


def save_individual_img(rgb_pred, views, normals, gbuffer_mask, buffers, images_save_path):
    convert_uint = lambda x: np.clip(np.rint(dataset_util.rgb_to_srgb(x).numpy() * 255.0), 0, 255).astype(np.uint8) 
    convert_uint_255 = lambda x: (x * 255).to(torch.uint8)

    Path(images_save_path / "rgb").mkdir(parents=True, exist_ok=True)
    Path(images_save_path / "normal").mkdir(parents=True, exist_ok=True)
    

    for i in range(len(gbuffer_mask)):
        mask = gbuffer_mask[i].cpu()
        id = int(views["frame_name"][i])

        # rgb prediction
        imageio.imsave(images_save_path / "rgb" / f'{id:05d}.png', convert_uint(torch.cat([rgb_pred[i].cpu(), mask], -1))) 


        ##normal
        normal = (normals[i] + 1.) / 2.
        normal = torch.cat([normal.cpu(), mask], -1)
        imageio.imsave(images_save_path / "normal" / f'{id:05d}.png', convert_uint_255(normal))

        

def save_manipulation_image(rgb_pred, views, normals, gbuffer_mask, file_name, normal=True):
    convert_uint = lambda x: np.clip(np.rint(dataset_util.rgb_to_srgb(x).detach().numpy() * 255.0), 0, 255).astype(np.uint8) 
    convert_uint_255 = lambda x: (x * 255).to(torch.uint8)
    
    for i in range(len(gbuffer_mask)):
        mask = gbuffer_mask[i].cpu()

        # rgb prediction
        imageio.imsave(file_name.replace(f'.png', f'_rgb_{i}.png'), convert_uint(torch.cat([rgb_pred[i].cpu().data, mask], -1))) 
        if normal:
            ##normal
            normal = (normals[i] + 1.) / 2.
            normal = torch.cat([normal.cpu(), mask], -1)
            imageio.imsave(file_name.replace(f'.png', f'_normal_{i}.png'), convert_uint_255(normal))    