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

def generate_planes():
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.
    """
    return torch.tensor([[[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]],
                            [[1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]],
                            [[0, 0, 1],
                            [1, 0, 0],
                            [0, 1, 0]]], dtype=torch.float32)

class NeuralShader(torch.nn.Module):

    def __init__(self,
                 activation='relu',
                 last_activation=None,
                 fourier_features='positional',
                 disentangle_network_params=None,
                 bsdf='pbr',
                 aabb=None,
                 device='cpu',
                 canonical_v = None,
                 ):

        super().__init__()
        self.device = device
        self.aabb = aabb
        self.bsdf = bsdf
        # ==============================================================================================
        # PE
        # ==============================================================================================
        if fourier_features == 'positional':
            print("Stage 1: ")
            self.grid_size = 128
            self.grid_dim = 16
        elif fourier_features == 'hashgrid':
            self.grid_size = 512
            self.grid_dim = 64
            print("STAGE 2: Using hashgrid (tinycudann) for intrinsic materials")


        # self.view_embedder, channels = get_embedder(multires=2)
        self.view_size = 3
        # self.view_size = channels

        # define triplane
        self.planes = torch.nn.Parameter(torch.randn(3, self.grid_dim, self.grid_size, self.grid_size).to(device))

        # define color mlp
        mlp_input_dim = self.grid_dim * 3 + self.view_size
        hidden_dim = 128
        self.color_mlp = FC(mlp_input_dim, 3, [hidden_dim, hidden_dim], activation, last_activation).to(device)

        self.plane_axes = generate_planes().to(device)

        # Store the config
        self._config = {
            "activation":activation,
            "last_activation":last_activation,
            "fourier_features":fourier_features,
            "disentangle_network_params":disentangle_network_params,
            "bsdf":bsdf,
            "aabb":aabb,
        }

    def view_embedder(self, x):
        return x

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


    def project_onto_planes(self, planes, coordinates):
        """
        Does a projection of a 3D point onto a batch of 2D planes,
        returning 2D plane coordinates.

        Takes plane axes of shape n_planes, 3, 3
        # Takes coordinates of shape B, H, W, 3
        # returns projections of shape B*n_planes, H, W, 2
        """
        B, H, W, C = coordinates.shape
        n_planes, _, _ = planes.shape

        coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1, -1).reshape(B*n_planes, H, W, 3)
        inv_planes = torch.linalg.inv(planes).repeat(B, 1, 1).reshape(B*n_planes, 3, 3)
        projections = torch.einsum('nhwi, nij -> nhwj', coordinates, inv_planes)
        return projections[..., :2] 


    def sample_from_planes(self, coordinates, mode='bilinear', padding_mode='zeros', box_warp=None):
        '''
        plane_axis : self.plane_axes
        plane_features : self.planes
        coordinates : input, shape of (B, H, W, 3)

        outputs: B, H, W, grid_dim*3
        '''
        assert padding_mode == 'zeros'
        n_planes, C, H, W = self.planes.shape
        B, H, W, C = coordinates.shape

        coordinates = self.apply_aabb(coordinates).view(B, H, W, 3) * 2 - 1 # normalize to -1, 1

        projected_coordinates = self.project_onto_planes(self.plane_axes, coordinates) # shape of (B*n_planes, H, W, 2)
        output_features = torch.nn.functional.grid_sample(self.planes.repeat(B,1,1,1), projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False).view(B, 3, self.grid_dim, H, W).permute(0, 3, 4, 1, 2).view(B, H, W, self.grid_dim*3)

        
        return output_features


    def forward(self, position, gbuffer, view_direction, mesh, light, deformed_position, skin_mask=None):
        bz, h, w, ch = position.shape
        

        view_dir = view_direction[:, None, None, :]
        normal_bend = self.get_shading_normals(deformed_position, view_dir, gbuffer, mesh).view(-1, 3)
        normal_bend_encoded = self.view_embedder(normal_bend)
        # normal_encoded = self.dir_enc_func(normal_bend, torch.ones_like(normal_bend[:, 0:1]) * 0.5)
        plane_features = self.sample_from_planes(position) # shape of B, H, W, grid_dim*3

        mlp_input = torch.cat([plane_features.reshape(-1, self.grid_dim*3), normal_bend_encoded], dim=-1)

        # view_dirs = gbuffer["position"].view(-1, 3) - view_direction[:, None, None, :].repeat(1, h, w, 1).view(-1, 3)

        # normal_encoded = self.dir_enc_func(normal_bend, torch.ones_like(normal_bend[:, 0:1]) * 0.1)

        color = self.color_mlp(mlp_input).view(bz, h, w, -1)

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

    def apply_aabb(self, position):
        position = (position.view(-1, 3) - self.aabb[0][None, ...]) / (self.aabb[1][None, ...] - self.aabb[0][None, ...])
        position = torch.clamp(position, min=0, max=1)
        return position

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