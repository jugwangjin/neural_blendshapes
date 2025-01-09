import numpy as np
import torch
from torch import nn
from .encoder import ResnetEncoder
from flare.modules.embedder import *

# import tinycudann as tcnn
import pytorch3d.transforms as pt3d
import pytorch3d.ops as pt3o

from .geometry import compute_matrix, laplacian_uniform, laplacian_density

import tinycudann as tcnn
SCALE_CONSTANT = 0.25


def initialize_weights(m, gain=0.1):

    # iterate over layers, apply if it is nn.Linear

    for l in m.children():
        if isinstance(l, nn.Linear):
            # nn.init.xavier_uniform_(l.weight, gain=gain)
            try:
                l.bias.data.zero_()
            except:
                pass

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x


class ConstantTemplate(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.constant = torch.nn.Parameter(torch.zeros(shape))

    def forward(self, *args, **kwargs):
        return self.constant



class MLPTemplate(nn.Module):
    def __init__(self, inp_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(inp_dim, 512),
            # nn.LayerNorm(512),
            nn.Softplus(beta=100),
            nn.Linear(512, 512),
            # nn.LayerNorm(512),
            nn.Softplus(beta=100),
            nn.Linear(512, 512),
            # nn.LayerNorm(512),
            nn.Softplus(beta=100),
            nn.Linear(512, 3, bias=False)
        )

    def forward(self, x):
        return self.mlp(x)


class NeuralBlendshapes(nn.Module):
    def __init__(self, ict_facekit, aabb  = None, tight_face_normals=None):
        super().__init__()
        self.encoder = ResnetEncoder(53+6, ict_facekit)

        self.aabb = aabb

        self.ict_facekit = ict_facekit
        self.tight_face_index = 6705

        desired_resolution = 2048
        base_grid_resolution = 16
        num_levels = 16
        per_level_scale = np.exp(np.log(desired_resolution / base_grid_resolution) / (num_levels-1))
        enc_cfg =  {
            "otype": "HashGrid",
            "n_levels": num_levels,
            "n_features_per_level": 2,
            "log2_hashmap_size": 18,
            "base_resolution": base_grid_resolution,
            "per_level_scale" : per_level_scale
        }

        self.fourier_feature_transform = tcnn.Encoding(3, enc_cfg).to('cuda')
        self.inp_size = self.fourier_feature_transform.n_output_dims


        self.include_identity_on_encoding = True

        if self.include_identity_on_encoding:
            self.inp_size += 3

        # self.inp_size = 3

        # multires=9
        # self.embed_fn, input_ch = get_embedder(multires)


        self.expression_deformer = nn.Sequential(
            nn.Linear(3, 512),
            # nn.LayerNorm(512),
            nn.Softplus(beta=100),
            nn.Linear(512, 512),
            # nn.LayerNorm(512),
            nn.Softplus(beta=100),
            nn.Linear(512, 512),
            # nn.LayerNorm(512),
            nn.Softplus(beta=100),
            nn.Linear(512, 54*3, bias=False)
        )

        self.register_buffer('tight_face_details', torch.zeros(self.tight_face_index, 54))

        # self.tight_face_details = torch.nn.Parameter(torch.zeros(self.tight_face_index, 54))
        if tight_face_normals is None:
            self.register_buffer('tight_face_normals', torch.zeros(self.tight_face_index, 3))
        else:
            self.register_buffer('tight_face_normals', tight_face_normals * 1e-2)

        self.template_deformer = MLPTemplate(3)
        self.template_embedder = Identity()

        # initialize last layer of template deformer with low weights
        torch.nn.init.normal_(self.template_deformer.mlp[-1].weight, mean=0.0, std=1.0 / 256)

        # initialize last layer of expression deformer with low weights
        torch.nn.init.normal_(self.expression_deformer[-1].weight, mean=0.0, std=1.0 / 512)

        for l in self.template_deformer.mlp:
            if isinstance(l, nn.Linear):
                torch.nn.init.constant_(l.bias, 0.0) if l.bias is not None else None
                # n_in = l.weight.size(1)
                # torch.nn.init.normal_(l.weight, mean=0.0, std=1.0 / math.sqrt(n_in))

        # init expression deformer with low weights
        for l in self.expression_deformer:
            if isinstance(l, nn.Linear):
                torch.nn.init.constant_(l.bias, 0.0) if l.bias is not None else None
                # n_in = l.weight.size(1)
                # torch.nn.init.normal_(l.weight, mean=0.0, std=1.0 / math.sqrt(n_in))
                

        self.pose_weight = nn.Sequential(
                    nn.Linear(3, 32),
                    nn.Softplus(beta=100),
                    nn.Linear(32, 32),
                    nn.Softplus(beta=100),
                    nn.Linear(32,1),
                    nn.Sigmoid()
        )

        initialize_weights(self.pose_weight, gain=0.01)
        self.pose_weight[-2].weight.data.zero_()
        self.pose_weight[-2].bias.data[0] = 3.


    def encode_position(self, coords):
        template = self.ict_facekit.canonical[0] # shape of V, 3

        org_coords = coords 

        unsqueezed = False
        if len(coords.shape) == 2:
            coords = coords[None]
            unsqueezed = True
        
        b, v, _ = coords.shape
        coords = coords.reshape(b*v, -1)

        if self.aabb is not None:
            coords = (coords - self.aabb[0][None, ...]) / (self.aabb[1][None, ...] - self.aabb[0][None, ...])
            coords = torch.clamp(coords, min=0, max=1)
            coords = coords * 0.95 + 0.025
        else:
            aabb_min = torch.min(template, dim=0)[0][None] * 1.2
            aabb_max = torch.max(template, dim=0)[0][None] * 1.2

            coords = (coords - aabb_min) / (aabb_max - aabb_min)

        encoded_coords = self.fourier_feature_transform(coords)

        encoded_coords = encoded_coords.reshape(b, v, -1)
        if unsqueezed:
            encoded_coords = encoded_coords[0]
            
        if self.include_identity_on_encoding:
            encoded_coords = torch.cat([encoded_coords, org_coords], dim=-1)

        return encoded_coords


    def remove_teeth(self, face):
        return face
        face[..., 17039:21451, :] = 0
        return face

    @torch.no_grad()
    def precompute_networks(self):
        # precompute the fixed mlps and return. 
        template = self.ict_facekit.canonical[0]
        pose_weight = self.pose_weight(self.ict_facekit.canonical[0])

        template_mesh_u_delta = self.template_deformer(template)

        expression_mesh_delta_u = self.expression_deformer(template).reshape(template.shape[0], 54, 3)

        return {'template_mesh_u_delta': template_mesh_u_delta, 'expression_mesh_delta_u': expression_mesh_delta_u, 'pose_weight': pose_weight}
        
    @torch.no_grad()
    def deform_with_precomputed(self, features, precomputed):
        template = self.ict_facekit.canonical[0]
        pose_weight = precomputed['pose_weight']

        template_mesh_delta = precomputed['template_mesh_u_delta']
        expression_mesh_delta_u = precomputed['expression_mesh_delta_u']
        expression_mesh_delta = torch.einsum('bn, mnd -> bmd', features[..., :53], expression_mesh_delta_u[:, :53])

        tight_face_details = torch.einsum('bn, dn -> bd', features[:, :53], self.tight_face_details) # shape of B, 6705
        tight_face_details = tight_face_details[..., None] * self.tight_face_normals[None] # shape of B, 6705, 3

        expression_mesh_delta[:, :self.tight_face_index] += tight_face_details

        template_mesh_delta = template_mesh_delta + expression_mesh_delta_u[:, -1]

        ict_mesh = self.ict_facekit(expression_weights = features[..., :53], identity_weights = self.encoder.identity_weights[None].repeat(features.shape[0], 1))

        ict_mesh_w_temp = ict_mesh + template_mesh_delta[None]

        return_dict = {}
        expression_mesh = ict_mesh_w_temp + expression_mesh_delta
        expression_mesh_posed = self.apply_deformation(expression_mesh, features, pose_weight)
        return_dict['expression_mesh_posed'] = self.remove_teeth(expression_mesh_posed)

        return return_dict
        

    def forward(self, image=None, views=None, features=None, image_input=True, pretrain=False):
        if features is None:
            bshape=None
            if features is not None:
                bshape = features
            return_dict = {} 
            features = self.encoder(views)

            if bshape is not None:
                features[:, :53] = bshape
        else:
            return_dict = {} 
            features = features

        return_dict['features'] = features
        
        bsize = features.shape[0]
        template = self.ict_facekit.canonical[0]
        pose_weight = self.pose_weight(self.ict_facekit.canonical[0])

        encoded_points = template
        # encoded_points = torch.cat([self.encode_position(template)], dim=-1)
        encoded_points2 = template
        # encoded_points2 = self.encode_position(template)
        # encoded_points2 = torch.cat([self.encode_position(template, sec=True)], dim=-1)

        template_mesh_delta = self.template_deformer(encoded_points)

        template_mesh = self.ict_facekit.neutral_mesh_canonical[0] + template_mesh_delta

        # self.ict_facekit.update_eyeball_centers(template_mesh)

        template_mesh_posed = self.apply_deformation(template_mesh[None], features, pose_weight)

        ict_mesh = self.ict_facekit(expression_weights = features[..., :53], identity_weights = self.encoder.identity_weights[None].repeat(bsize, 1))

        ict_mesh_w_temp = ict_mesh + template_mesh_delta[None]

        ict_mesh_w_temp_posed = self.apply_deformation(ict_mesh_w_temp, features, pose_weight)

        return_dict['ict_mesh_posed'] = self.remove_teeth(self.apply_deformation(ict_mesh, features, pose_weight))
        return_dict['template_mesh'] = self.remove_teeth(template_mesh)
        return_dict['template_mesh_posed'] = self.remove_teeth(template_mesh_posed)
        return_dict['ict_mesh_w_temp'] = self.remove_teeth(ict_mesh_w_temp)
        return_dict['ict_mesh_w_temp_posed'] = self.remove_teeth(ict_mesh_w_temp_posed)

        return_dict['expression_mesh_delta'] = torch.zeros_like(template_mesh_delta)

        if pretrain:
            return return_dict

        expression_mesh_delta_u = self.expression_deformer(encoded_points2).reshape(template.shape[0], 54, 3)

        feat = features[:, :53].clamp(0, 1)
        feat = torch.cat([feat, torch.ones_like(feat[:, :1])], dim=1)

        expression_mesh_delta = torch.einsum('bn, mnd -> bmd', feat, expression_mesh_delta_u)
        
        # self.tight_face_details have shape of 6705, 53
        # feat has shape of B, 53
        # einsum and multiplying it with tight_face_normals will have shape of B, 6705, 3
        # add to expression_mesh_delta
        tight_face_details = torch.einsum('bn, dn -> bd', feat, self.tight_face_details) # shape of B, 6705
        tight_face_details = tight_face_details[..., None] * self.tight_face_normals[None] # shape of B, 6705, 3
        expression_mesh_delta[:, :self.tight_face_index] += tight_face_details

        expression_mesh = ict_mesh_w_temp + expression_mesh_delta

        expression_mesh_posed = self.apply_deformation(expression_mesh, features, pose_weight)
    
        return_dict['expression_mesh'] = self.remove_teeth(expression_mesh)
        return_dict['expression_mesh_posed'] = self.remove_teeth(expression_mesh_posed)

        return_dict['expression_mesh_delta'] = expression_mesh_delta

        return_dict['pose_weight'] = pose_weight

        return return_dict


    def get_expression_delta(self):

        template = self.ict_facekit.canonical[0]
        uv_coords = self.ict_facekit.uv_neutral_mesh[0]
        encoded_points = template
        # encoded_points = self.encode_position(template)
        # encoded_points = torch.cat([self.encode_position(template, sec=True)], dim=-1)

        expression_mesh_delta_u = self.expression_deformer(encoded_points).reshape(template.shape[0], 54, 3).permute(1, 0, 2)
        return expression_mesh_delta_u


    def apply_deformation(self, vertices, features, weights=None):
        euler_angle = features[..., 53:56]
        translation = features[..., 56:59]
        scale = features[..., 59:60]
        transform_origin = self.encoder.transform_origin
        global_translation = features[..., 60:63]
        # translation = self.encoder.translation
        

        # transform_origin = torch.cat([torch.zeros(1, device=vertices.device), self.encoder.transform_origin[1:]], dim=0)

        B, V, _ = vertices.shape
                # Convert Euler angles to rotation matrices
        rotation_matrix = pt3d.euler_angles_to_matrix(euler_angle, convention='XYZ')  # Shape: (B, 3, 3)

        # Apply scale
        scaled_vertices = vertices * scale[:, None]

        # Translate vertices to the rotation origin
        centered_vertices = scaled_vertices - transform_origin[None, None, :]

        # Apply rotation
        rotated_vertices = torch.einsum('bij,bvj->bvi', rotation_matrix, centered_vertices)

        # Translate back from the rotation origin
        deformed_vertices = rotated_vertices + transform_origin[None, None, :]

        # Apply global translation
        deformed_vertices = deformed_vertices + translation[:, None, :]

        # Apply pose weights if provided
        if weights is not None:
            weights = weights.view(1, V, 1)
            deformed_vertices = deformed_vertices * weights + scaled_vertices * (1 - weights)

        return deformed_vertices + global_translation[:, None, :]

    def save(self, path):
        data = {
            'state_dict': self.state_dict()
        }
        torch.save(data, path)  

    def to(self, device):
        super().to(device)
        return self

def get_neural_blendshapes(model_path=None, train=True, ict_facekit=None, aabb=None, tight_face_normals=None, device='cuda'):
    neural_blendshapes = NeuralBlendshapes(ict_facekit, aabb=aabb, tight_face_normals=tight_face_normals)
    neural_blendshapes.to(device)

    import os
    if (os.path.exists(str(model_path))):
        print("Loading model from: ", str(model_path))
        params = torch.load(str(model_path))
        neural_blendshapes.load_state_dict(params["state_dict"], strict=True)
    elif model_path is not None:
        print('Model path is provided but the model is not found. Initializing with random weights.')
        raise Exception("Model not found")

    if train:
        neural_blendshapes.train()
    else:
        neural_blendshapes.eval()

    return neural_blendshapes        