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
        self.activation = activation
        self.last_activation = last_activation
        self.fourier_features = fourier_features
        self.disentangle_network_params = disentangle_network_params
        
        self.update_mlp(fourier_features)

        print(disentangle_network_params)

        self.light_mlp = FC(3+3, 1, self.disentangle_network_params["light_mlp_dims"], activation=self.activation, last_activation=self.last_activation).to(self.device) 
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

    def update_mlp(self, fourier_features):
         # ==============================================================================================
        # PE
        # ==============================================================================================
        if fourier_features == 'positional':
            print("STAGE 1: Using positional encoding (NeRF) for intrinsic materials")
            self.fourier_feature_transform, channels = get_embedder(multires=6)
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
        self.material_mlp_ch = 6 # diffuse 3 and roughness 1
        self.material_mlp = FC(self.inp_size, self.material_mlp_ch, self.disentangle_network_params["material_mlp_dims"], self.activation, self.last_activation).to(self.device) #sigmoid
        
        if fourier_features == "hashgrid":
            self.gradient_scaling = 128.0
            self.material_mlp.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] * self.gradient_scaling, ))


    def forward(self, position, gbuffer, view_direction, mesh, light, deformed_position, skin_mask=None):
        bz, h, w, ch = position.shape
        uv_coordinates = gbuffer["uv_coordinates"]
        deformed_position = deformed_position
        pe_input = self.apply_pe(position=uv_coordinates)

        view_dir = view_direction[:, None, None, :]
        normal_bend = self.get_shading_normals(deformed_position, view_dir, gbuffer, mesh)

        kr_max = torch.ones((bz, h, w, 1))
        kr_max = kr_max.to(self.device)                    

        wo = util.safe_normalize(view_dir - deformed_position)
        reflvec = util.safe_normalize(util.reflect(wo, normal_bend))        
        # view_dir = self.dir_enc_func(normal_bend.view(-1, 3), kr_max.view(-1, 1))
        # view_dir = self.dir_enc_func(wo.view(-1, 3), kr_max.view(-1, 1))
        # view_dir = self.dir_enc_func(reflvec.view(-1, 3), kr_max.view(-1, 1))

        material = self.material_mlp(pe_input.view(-1, self.inp_size).to(torch.float32)) 

        diffuse = material[..., :3]
        specular = material[..., 3:6]
        
        light_mlp_input = torch.cat([uv_coordinates.view(-1, 3), reflvec.view(-1, 3)], dim=1)
        # light_mlp_input = torch.cat([view_dir.view(-1, 20), pe_input.view(-1, self.inp_size)], dim=1)

        light = self.light_mlp(light_mlp_input)

        specular = light * specular

        color = diffuse + specular

        return color, material, light

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
        batch_size, H, W, ch = positions.shape

        view_direction = torch.cat([v.center.unsqueeze(0) for v in views['camera']], dim=0)
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
                                    torch.concat([pred_color, torch.ones_like(pred_color[..., 0:1]).to(self.device)], axis=3), gbuffer["mask"].float())
        
    
        ### we antialias the final color here (!)
        pred_color_masked = dr.antialias(pred_color_masked.contiguous(), gbuffer["rast"], gbuffer["deformed_verts_clip_space"], mesh.indices.int())
        
        cbuffers = {}
        # cbuffers['material'] = material
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

        view_direction = torch.cat([v.center.unsqueeze(0) for v in views['camera']], dim=0)
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
    def get_shading_normals(self, position, view_dir, gbuffer, mesh):
        ''' flip the backward facing normals
        '''
        normal = ru.prepare_shading_normal(position, view_dir, None, 
                                           gbuffer["vertex_normals"], gbuffer["tangent_normals"], gbuffer["face_normals"], two_sided_shading=True, opengl=True, use_python=False)
        gbuffer["normal"] =  dr.antialias(normal.contiguous(), gbuffer["rast"], gbuffer["deformed_verts_clip_space"], mesh.indices.int())
        return gbuffer["normal"]
    
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