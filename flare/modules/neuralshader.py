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

from flare.modules.fc import FC
from flare.modules.embedder import get_embedder
from flare.modules.embedding_roughness_np import generate_ide_fn
import numpy as np
import torch
import tinycudann as tcnn
import nvdiffrec.render.renderutils.ops as ru
import nvdiffrast.torch as dr
# from . import util
from nvdiffrec.render import util
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


def make_module(module):
    # Create a module instance if we don't already have one
    if isinstance(module, torch.nn.Module):
        return module
    else:
        return module()

class NeuralShader(torch.nn.Module):

    def __init__(self,
                 activation='relu',
                 last_activation=None,
                 fourier_features='positional',
                 disentangle_network_params=None,
                 bsdf='pbr',
                 aabb=None,
                 device='cpu'):

        super().__init__()
        self.device = device
        self.aabb = aabb
        self.bsdf = bsdf
        # ==============================================================================================
        # PE
        # ==============================================================================================
        if fourier_features == 'positional':
            print("STAGE 1: Using positional encoding (NeRF) for intrinsic materials")
            self.fourier_feature_transform, channels = get_embedder(multires=9)
            self.inp_size = channels
        elif fourier_features == 'hashgrid':
            print("STAGE 2: Using hashgrid (tinycudann) for intrinsic materials")
            # ==============================================================================================
            # used for 2nd stage training
            # ==============================================================================================
            # Setup positional encoding, see https://github.com/NVlabs/tiny-cuda-nn for details
            desired_resolution = 4096
            base_grid_resolution = 16
            num_levels = 16
            per_level_scale = np.exp(np.log(desired_resolution / base_grid_resolution) / (num_levels-1))
            enc_cfg =  {
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": base_grid_resolution,
                "per_level_scale" : per_level_scale
            }

            gradient_scaling = 64.0
            self.fourier_feature_transform = tcnn.Encoding(3, enc_cfg).to(device)
            self.fourier_feature_transform.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] / gradient_scaling if grad_i[0] is not None else None, ))
            self.inp_size = self.fourier_feature_transform.n_output_dims

        # ==============================================================================================
        # create MLP
        # ==============================================================================================
        self.material_mlp_ch = disentangle_network_params['material_mlp_ch']
        self.material_mlp_1 = FC(self.inp_size, 128, [128, 128, 128], activation, None).to(device) #sigmoid
        self.material_mlp_2 = FC(128 + 20, 3, [128], activation, last_activation).to(device) #sigmoid
        
        self.light_mlp = FC(38, 3, disentangle_network_params["light_mlp_dims"], activation=activation, last_activation=None, bias=True).to(device) 
        self.dir_enc_func = generate_ide_fn(deg_view=3, device=self.device)
        self.dir_enc_func_normals = generate_ide_fn(deg_view=3, device=self.device)
        
        print(disentangle_network_params)

        if fourier_features == "hashgrid":
            self.material_mlp_1.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] * gradient_scaling if grad_i[0] is not None else None, ))
            self.material_mlp_2.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] * gradient_scaling if grad_i[0] is not None else None, ))

        # Store the config
        self._config = {
            "activation":activation,
            "last_activation":last_activation,
            "fourier_features":fourier_features,
            "disentangle_network_params":disentangle_network_params,
            "bsdf":bsdf,
            "aabb":aabb,
        }

    def custom_forward(self, position):
        position = np.expand_dims(position, axis=0)
        position = torch.from_numpy(position).to(self.device)
        bz, h, w, ch = position.shape
        pe_input = self.apply_pe(position=position)


        # ==============================================================================================
        # Albedo ; roughness; specular intensity 
        # ==============================================================================================   
        all_tex = self.material_mlp(pe_input.view(-1, self.inp_size).to(torch.float32)) 
        kd = all_tex[..., :3].view(bz, h, w, ch) 
        kr = all_tex[..., 3:4] 
        kr = kr.view(bz, h, w, 1).to(self.device)
        ko = all_tex[..., 4:5]
        ko = ko.view(bz, h, w, 1)

        return kd, kr, ko

    def forward(self, position, gbuffer, view_direction, mesh, light, deformed_position, skin_mask=None):
        bz, h, w, ch = position.shape
        pe_input = self.apply_pe(position=position).view(-1, self.inp_size)

        view_dir = view_direction[:, None, None, :]
        normal_bend = self.get_shading_normals(deformed_position, view_dir, gbuffer, mesh).view(-1, 3)
        normal_encoded = self.dir_enc_func(normal_bend, torch.ones_like(normal_bend[:, 0:1]) * 0.5)


        # view_dirs = gbuffer["position"].view(-1, 3) - view_direction[:, None, None, :].repeat(1, h, w, 1).view(-1, 3)

        # normal_encoded = self.dir_enc_func(normal_bend, torch.ones_like(normal_bend[:, 0:1]) * 0.1)

        material_1_output = self.material_mlp_1(pe_input)
        material_2_input = torch.cat([material_1_output, normal_encoded], dim=-1)
        color = self.material_mlp_2(material_2_input).view(bz, h, w, -1)

        # material_input = torch.cat([pe_input, normal_bend], dim=-1)
        # color = self.material_mlp(material_input).view(bz, h, w, -1)

        return color

    # ==============================================================================================
    # prepare the final color output
    # ==============================================================================================    
    def shade(self, gbuffer, views, mesh, finetune_color, lgt):

        positions = gbuffer["canonical_position"]
        batch_size, H, W, ch = positions.shape

        view_direction = torch.cat([v.center.unsqueeze(0) for v in views['camera']], dim=0)
        if finetune_color:
            ### skin mask for fresnel coefficient
            skin_mask = (torch.sum(views["skin_mask"][..., 1:5], axis=-1)).unsqueeze(-1)
            skin_mask = skin_mask * views["mask"] 
            skin_mask_bool = (skin_mask > 0.0).int().bool()
        else:
            skin_mask_bool = None

        ### compute the final color, and c-buffers 
        pred_color = self.forward(positions, gbuffer, view_direction, mesh, light=lgt,
                                            deformed_position=gbuffer["position"], skin_mask=skin_mask_bool)
        pred_color = pred_color.view(positions.shape) 

        ### !! we mask directly with alpha values from the rasterizer !! ###
        pred_color_masked = torch.lerp(torch.zeros((batch_size, H, W, 4)).to(self.device), 
                                    torch.concat([pred_color, torch.ones_like(pred_color[..., 0:1]).to(self.device)], axis=3), gbuffer["mask_low_res"].float())
    
        ### we antialias the final color here (!)
        pred_color_masked = dr.antialias(pred_color_masked.contiguous(), gbuffer["rast"], gbuffer["deformed_verts_clip_space"], mesh.indices.int())
        pred_color_masked = upsample(pred_color_masked, high_res)
        buffers = {}

        return pred_color_masked[..., :3], buffers, pred_color_masked[..., -1:]

    # ==============================================================================================
    # misc functions
    # ==============================================================================================
    def get_shading_normals(self, position, view_dir, gbuffer, mesh):
        ''' flip the backward facing normals
        '''
        normal = ru.prepare_shading_normal(position, view_dir, None, 
                                           gbuffer["vertex_normals"], gbuffer["tangent_normals"], gbuffer["face_normals"], two_sided_shading=True, opengl=True, use_python=False)
        gbuffer["normal"] =  dr.antialias(normal.contiguous(), gbuffer["rast"], gbuffer["deformed_verts_clip_space"], mesh.indices.int())
        downscaled_normal = gbuffer["normal"]
        gbuffer["normal"] = upsample(gbuffer["normal"], high_res)
        return downscaled_normal
    
    def apply_pe(self, position):
        ## normalize PE input 
        position = (position.view(-1, 3) - self.aabb[0][None, ...]) / (self.aabb[1][None, ...] - self.aabb[0][None, ...])
        position = torch.clamp(position, min=0, max=1)
        pe_input = self.fourier_feature_transform(position.contiguous()).to(torch.float32)
        return pe_input

    @classmethod
    def load(cls, path, device='cpu'):
        data = torch.load(path, map_location=device)

        shader = cls(**data['config'], device=device)
        shader.load_state_dict(data['state_dict'], strict=False)

        return shader

    def save(self, path):
        data = {
            'version': 2,
            'config': self._config,
            'state_dict': self.state_dict()
        }

        torch.save(data, path)