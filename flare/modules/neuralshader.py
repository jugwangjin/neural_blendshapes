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
                 existing_encoder=None,
                 device='cpu'):

        super().__init__()
        self.device = device
        self.aabb = aabb
        self.bsdf = bsdf
        self.activation = activation
        self.last_activation = last_activation
        self.fourier_features = fourier_features
        self.disentangle_network_params = disentangle_network_params
        
        # ==============================================================================================
        # PE
        # ==============================================================================================
        if existing_encoder is not None:
            self.fourier_feature_transform = existing_encoder
            self.inp_size = self.fourier_feature_transform.n_output_dims

        else:
            if fourier_features == 'positional':
                print("STAGE 1: Using positional encoding (NeRF) for intrinsic materials")
                self.fourier_feature_transform, channels = get_embedder(multires=4)
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

                self.gradient_scaling = 1.0
                self.fourier_feature_transform = tcnn.Encoding(3, enc_cfg).to(self.device)
                # self.fourier_feature_transform.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] / gradient_scaling, ))
                self.fourier_feature_transform.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] / self.gradient_scaling if grad_i[0] is not None else None, ))
                self.inp_size = self.fourier_feature_transform.n_output_dims

        # ==============================================================================================
        # create MLP
        # ==============================================================================================
        # self.material_mlp_ch = disentangle_network_params['material_mlp_ch']
        self.diffuse_mlp_ch = 4 # diffuse 3 and roughness 1
        self.diffuse_mlp = FC(self.inp_size, 32, self.disentangle_network_params["material_mlp_dims"], self.activation, None).to(self.device) #sigmoid
        self.last_act = make_module(self.last_activation)
        
        self.env_light_mlp = FC(3 + 32, 3, self.disentangle_network_params["light_mlp_dims"], self.activation, None).to(self.device) #sigmoid

        self.self_occ_mlp = FC(3 + 32, 3, self.disentangle_network_params["light_mlp_dims"], self.activation, None).to(self.device) # reflvec / normal for input


        if fourier_features == "hashgrid":
            self.diffuse_mlp.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] * self.gradient_scaling, ))
            self.self_occ_mlp.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] * self.gradient_scaling, ))
            self.env_light_mlp.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] * self.gradient_scaling, ))

        print(disentangle_network_params)

        self.dir_enc_func = generate_ide_fn(deg_view=3, device=self.device)

        # Store the config
        self._config = {
            "activation":activation,
            "last_activation":last_activation,
            "fourier_features":fourier_features,
            "disentangle_network_params":disentangle_network_params,
            "bsdf":bsdf,
            "aabb":aabb,
        }


        # Need: coords, normal, reflvec
    def forward(self, position, gbuffer, view_direction, mesh, light, deformed_position, skin_mask=None):
        bz, h, w, ch = position.shape
        # uv_coordinates = gbuffer["uv_coordinates"]
        canonical_position = gbuffer["canonical_position"]
        deformed_position = deformed_position
        pe_input = self.apply_pe(position=canonical_position, normalize=True)

        '''
        diffuse
        '''
        diffuse_mlp_input = pe_input.view(-1, self.inp_size)
        diffuse_mlp_output = self.diffuse_mlp(diffuse_mlp_input)
        diffuse_color = self.last_act(diffuse_mlp_output[..., :3])
        roughness = self.last_act(diffuse_mlp_output[..., 3:4])

        view_dir = view_direction[:, None, None, :]

        '''
        env light
        '''
        normal_bend_temp_pose = self.get_shading_normals(deformed_position, view_dir, gbuffer, mesh, target='temp_pose')
        # normal_bend_temp_pose_enc = self.dir_enc_func(normal_bend_temp_pose.view(-1, 3), roughness.view(-1, 1))
        env_light_mlp_input = torch.cat([diffuse_mlp_output, normal_bend_temp_pose.view(-1, 3)], dim=1)

        env_light_mlp_output = self.env_light_mlp(env_light_mlp_input)
        env_light_color = self.last_act(env_light_mlp_output[..., :3])

        '''
        self-occlusion -> normals
        '''
        normal_bend_exp_no_pose = self.get_shading_normals(deformed_position, view_dir, gbuffer, mesh, target='exp_no_pose')
        # normal_bend_exp_no_pose_enc = self.dir_enc_func(normal_bend_exp_no_pose.view(-1, 3), roughness.view(-1, 1))
        self_occ_mlp_input = torch.cat([diffuse_mlp_output, normal_bend_exp_no_pose.view(-1, 3)], dim=1)

        self_occ_mlp_output = self.self_occ_mlp(self_occ_mlp_input)
        self_occ_color = self.last_act(self_occ_mlp_output)

        '''
        summary
        '''
        color = diffuse_color * self_occ_color + env_light_color  # the diffuse color -> when the light is maximum. the others decrease the value according to normals

        lights = torch.cat([self_occ_color, env_light_color], dim=-1)

        '''
        for normal visualization
        '''
        normal_bend = self.get_shading_normals(deformed_position, view_dir, gbuffer, mesh)

        return color, None, lights

    def get_mask(self, gbuffer, views, mesh, finetune_color, lgt):
        
        positions = gbuffer["canonical_position"]
        batch_size, H, W, ch = positions.shape

        ### !! we mask directly with alpha values from the rasterizer !! ###
        pred_color_masked = torch.lerp(torch.zeros((batch_size, H, W, 1)).to(self.device), 
                                    torch.ones((batch_size, H, W, 1)).to(self.device), gbuffer["mask"].float())

        ### we antialias the final color here (!)
        pred_color_masked = dr.antialias(pred_color_masked.contiguous(), gbuffer["rast"], gbuffer["deformed_verts_clip_space"], mesh.indices.int())


        return None, None, pred_color_masked[..., -1:]

    # ==============================================================================================
    # prepare the final color output
    # ==============================================================================================    
    def shade(self, gbuffer, views, mesh, finetune_color, lgt):


        positions = gbuffer["canonical_position"]
        # positions = dr.antialias(positions, gbuffer['rast'], gbuffer['deformed_vertices_clip_space'], mesh.indices) 

        batch_size, H, W, ch = positions.shape

        view_direction = torch.cat([v.center.unsqueeze(0) for v in views['flame_camera']], dim=0)
        if finetune_color:
            ### skin mask for fresnel coefficient
            skin_mask = (torch.sum(views["skin_mask"][..., :3], axis=-1)).unsqueeze(-1)
            skin_mask = skin_mask * views["mask"] 
            skin_mask_bool = (skin_mask > 0.0).int().bool()
        else:
            skin_mask_bool = None

        ### compute the final color, and c-buffers 
        pred_color, material, light = self.forward(positions, gbuffer, view_direction, mesh, light=lgt,
                                            deformed_position=gbuffer["position"], skin_mask=skin_mask_bool)
        pred_color = pred_color.view(positions.shape) 

        ### !! we mask directly with alpha values from the rasterizer !! ###
        pred_color_masked = torch.lerp(torch.zeros((batch_size, H, W, 4)).to(self.device), 
                                    torch.concat([pred_color, torch.ones_like(pred_color[..., 0:1]).to(self.device)], axis=3), gbuffer["mask_low_res"].float())
        
    
        ### we antialias the final color here (!)
        pred_color_masked = dr.antialias(pred_color_masked.contiguous(), gbuffer["rast"], gbuffer["deformed_verts_clip_space"], mesh.indices.int())
        
        pred_color_masked = upsample(pred_color_masked, high_res)

        cbuffers = {}
        cbuffers['material'] = material
        cbuffers['light'] = light

        return pred_color_masked[..., :3], cbuffers, pred_color_masked[..., -1:]

    # ==============================================================================================
    # prepare the final color output
    # ==============================================================================================    
    def relight(self, gbuffer, views, mesh, finetune_color, lgt_list):

        deformed_position=gbuffer["position"]
        position = gbuffer["canonical_position"]
        bz, h, w, ch = position.shape
        pe_input = self.apply_pe(position=position)

        view_direction = torch.cat([v.center.unsqueeze(0) for v in views['flame_camera']], dim=0)
        view_dir = view_direction[:, None, None, :]

        if finetune_color:
            ### skin mask for fresnel coefficient
            skin_mask = (torch.sum(views["skin_mask"][..., :3], axis=-1)).unsqueeze(-1)
            skin_mask = skin_mask * views["mask"] 
            skin_mask_ = (skin_mask > 0.0).int().bool()
            fresnel_constant = torch.ones((bz, h, w, 1)).to(self.device) * 0.047
            fresnel_constant[skin_mask_] = 0.028
        else:
            fresnel_constant = 0.04

        ### compute the final color, and c-buffers 
        normal_bend = self.get_shading_normals(deformed_position, view_dir, gbuffer, mesh)

        # ==============================================================================================
        # Albedo ; roughness; specular intensity 
        # ==============================================================================================   
        all_tex = self.material_mlp(pe_input.view(-1, self.inp_size).to(torch.float32)) 
        kd = all_tex[..., :3].view(bz, h, w, ch) 
        kr = all_tex[..., 3:4] 
        kr = kr.view(bz, h, w, 1).to(self.device)
        ko = all_tex[..., 4:5]
        ko = ko.view(bz, h, w, 1)

        # ==============================================================================================
        # relight: use function from nvdiffrec 
        # ============================================================================================== 
        relit_imgs = []
        for lgt in lgt_list:
            pred_color, buffers = lgt.shading_pbr(deformed_position, normal_bend, kd, kr, view_dir, ko, normal_bend, fresnel_constant)
            pred_color = pred_color.view(position.shape) 

            bg = np.fliplr(lgt.base[5].detach().cpu().numpy())
            bg = bg.copy()
            bg = torch.from_numpy(bg).to(self.device).unsqueeze(0).repeat(bz, 1, 1, 1)
            bg_ = torch.concat([bg, torch.zeros((bz, h, w, 1)).to(self.device)], dim=-1)
            
            ### !! we mask directly with background here !! ###
            pred_color_masked = torch.lerp(bg_, torch.concat([pred_color, torch.ones_like(pred_color[..., 0:1]).to(self.device)], axis=3), gbuffer["mask"].float())
            
            pred_color_masked = dr.antialias(pred_color_masked.contiguous(), gbuffer["rast"], gbuffer["deformed_verts_clip_space"], mesh.indices.int())
            relit_imgs.append(pred_color_masked[..., :3])
        
        roughness_masked = torch.lerp(torch.zeros((bz, h, w, 1)).to(self.device), kr, gbuffer["mask"].float())
        ko_masked = torch.lerp(torch.zeros((bz, h, w, 1)).to(self.device), ko, gbuffer["mask"].float())
        albedo_masked = torch.lerp(torch.zeros((bz, h, w, 4)).to(self.device), 
                                    torch.concat([kd, torch.ones((bz, h, w, 1)).to(self.device)], axis=3), gbuffer["mask"].float())
        
        ### we antialias the final color here (!)
        roughness_masked = dr.antialias(roughness_masked.contiguous(), gbuffer["rast"], gbuffer["deformed_verts_clip_space"], mesh.indices.int())
        ko_masked = dr.antialias(ko_masked.contiguous(), gbuffer["rast"], gbuffer["deformed_verts_clip_space"], mesh.indices.int())
        
        buffers["albedo"] = albedo_masked[..., :3]
        buffers["roughness"] = roughness_masked[..., :1]
        buffers["specular_intensity"] = ko_masked[..., :1]
        return relit_imgs, buffers, pred_color_masked[..., -1:]

    # ==============================================================================================
    # misc functions
    # ==============================================================================================
    def get_shading_normals(self, position, view_dir, gbuffer, mesh, target=None):
        ''' flip the backward facing normals
        '''

        if target == 'exp_no_pose':
            v_norm = gbuffer['vertex_normals_exp_no_pose']
            t_norm = gbuffer['tangent_normals_exp_no_pose']
            f_norm = gbuffer['face_normals_exp_no_pose']
            target_dict_name = 'normal_exp_no_pose'
            # clip_space_name = 'deformed_verts_clip_space_exp_no_pose'
            
        elif target == 'temp_pose':
            v_norm = gbuffer['vertex_normals_temp_pose']
            t_norm = gbuffer['tangent_normals_temp_pose']
            f_norm = gbuffer['face_normals_temp_pose']
            target_dict_name = 'normal_temp_pose'
            # clip_space_name = 'deformed_verts_clip_space_temp_pose'

        else:
            v_norm = gbuffer['vertex_normals']
            t_norm = gbuffer['tangent_normals']
            f_norm = gbuffer['face_normals']
            target_dict_name = 'normal'
        clip_space_name = 'deformed_verts_clip_space'

        '''
        two_sided_shading true -> false
        '''
        normal = ru.prepare_shading_normal(position, view_dir, None, 
                                           v_norm, t_norm, f_norm, two_sided_shading=False, opengl=True, use_python=False)
        ret_val = dr.antialias(normal.contiguous(), gbuffer["rast"], gbuffer[clip_space_name], mesh.indices.int())
        gbuffer[target_dict_name] = upsample(ret_val, high_res)
        return ret_val
    
    def apply_pe(self, position, normalize=False):
        ## normalize PE input 
        position = position.view(-1, 3)
        if normalize:
            position = (position - self.aabb[0][None, ...]) / (self.aabb[1][None, ...] - self.aabb[0][None, ...])
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


        
    def update_mlp(self, fourier_features, existing_encoder = None):
         # ==============================================================================================
        # PE
        # ==============================================================================================
        if existing_encoder is not None:
            self.fourier_feature_transform = existing_encoder
            self.inp_size = self.fourier_feature_transform.n_output_dims

        else:
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

                self.gradient_scaling = 128.0
                self.fourier_feature_transform = tcnn.Encoding(3, enc_cfg).to(self.device)
                # self.fourier_feature_transform.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] / gradient_scaling, ))
                self.fourier_feature_transform.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] / self.gradient_scaling if grad_i[0] is not None else None, ))
                self.inp_size = self.fourier_feature_transform.n_output_dims

        # ==============================================================================================
        # create MLP
        # ==============================================================================================
        # self.material_mlp_ch = disentangle_network_params['material_mlp_ch']
        self.material_mlp_ch = 5 # diffuse 3 and roughness 1
        self.material_mlp = FC(self.inp_size, self.material_mlp_ch, self.disentangle_network_params["material_mlp_dims"], self.activation, self.last_activation).to(self.device) #sigmoid
        
        # self.brdf_mlp = FC(2, 3, self.disentangle_network_params["brdf_mlp_dims"], self.activation, self.last_activation).to(self.device) # inp size for coords, 20 for normal and reflvec

        if fourier_features == "hashgrid":
            self.gradient_scaling = 64.0
            self.material_mlp.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] * self.gradient_scaling, ))

        self.light_mlp = FC(self.inp_size + 40, 6, self.disentangle_network_params["light_mlp_dims"], activation=self.activation, last_activation=self.last_activation).to(self.device) # reflvec / normal for input
