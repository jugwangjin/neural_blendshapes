import numpy as np
import torch
from torch import nn
from .encoder import ResnetEncoder
from flare.modules.embedder import *

# import tinycudann as tcnn
import pytorch3d.transforms as pt3d
import pytorch3d.ops as pt3o

import os


from .geometry import compute_matrix, laplacian_uniform, laplacian_density
from .solvers import CholeskySolver, solve
from .parameterize import to_differential, from_differential
from scipy.spatial import KDTree

import tinycudann as tcnn
SCALE_CONSTANT = 0.25

# different activation functions
class GaussianActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-x**2/(2*self.a**2))

def initialize_weights(m, gain=0.1):

    # iterate over layers, apply if it is nn.Linear

    for l in m.children():
        if isinstance(l, nn.Linear):
            # nn.init.xavier_uniform_(l.weight, gain=gain)
            try:
                l.bias.data.zero_()
            except:
                pass

class mylayernorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.layernorm = nn.LayerNorm(num_features)
    def forward(self, x):
        if len(x.shape) == 3:
            b, v, c = x.shape
            x = x.reshape(b*v, c)
            x = self.layernorm(x)
            x = x.reshape(b, v, c)
            return x
        else:
            return self.layernorm(x)


class mygroupnorm(nn.Module):
    def __init__(self, num_groups, num_channels):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.groupnorm = nn.GroupNorm(num_groups, num_channels)
    def forward(self, x):
        if len(x.shape) == 3:
            b, v, c = x.shape
            x = x.reshape(b*v, c)
            x = self.groupnorm(x)
            x = x.reshape(b, v, c)
            return x
        else:
            return self.groupnorm(x)
            
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
            nn.Linear(inp_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 3, bias=False)
        )

    def forward(self, x):
        return self.mlp(x)


class NeuralBlendshapes(nn.Module):
    def __init__(self, vertex_parts, ict_facekit, exp_dir, aabb  = None, lambda_=32):
        super().__init__()
        self.encoder = ResnetEncoder(53+6, ict_facekit)

        self.aabb = aabb

        self.ict_facekit = ict_facekit
        self.tight_face_index = 6705
        self.face_index = 9409     
        self.head_index = 14062

        # self.socket_index = 11248
        self.socket_index = 14062

        vertices = ict_facekit.neutral_mesh_canonical[0].cpu().data.numpy()
        faces = ict_facekit.faces.cpu().data.numpy()

        faces = faces
        faces = faces
        faces = faces

        self.register_buffer('L', laplacian_uniform(torch.from_numpy(vertices).to('cuda'), torch.from_numpy(faces).to('cuda')))

        self.lambda_ = lambda_
        if self.lambda_ > 0:

            alpha = 1 - 1 / float(lambda_) if lambda_ > 1 else None
            alpha = None

            self.register_buffer('M', compute_matrix(torch.from_numpy(vertices).to('cuda'), torch.from_numpy(faces).to('cuda'), lambda_, alpha=alpha, density=False))

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

        self.fourier_feature_transform = tcnn.Encoding(3, enc_cfg).to('cuda')
        self.fourier_feature_transform2 = tcnn.Encoding(3, enc_cfg).to('cuda')
        self.inp_size = self.fourier_feature_transform.n_output_dims

        self.include_identity_on_encoding = True

        if self.include_identity_on_encoding:
            self.inp_size += 3

        self.inp_size = 3

        multires=9
        self.embed_fn, input_ch = get_embedder(multires)


        self.expression_deformer = nn.Sequential(
            nn.Linear(self.inp_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 53*3, bias=False)
        )
        self.template_deformer = MLPTemplate(self.inp_size)
        self.template_embedder = Identity()

        self.pose_weight = nn.Sequential(
                    nn.Linear(3, 32),
                    nn.ReLU(),
                    nn.Linear(32, 32),
                    nn.ReLU(),
                    nn.Linear(32,1),
                    nn.Sigmoid()
        )

        # multiply 0.1 to all weights of expression_deformer and template_deformer.mlp
        for l in self.expression_deformer.children():
            if isinstance(l, nn.Linear):
                try:
                    l.weight.data *= 0.1
                    l.bias.data.zero_()
                except Exception as e:
                    print("Error in initializing weights", e)
                    pass
        
        for l in self.template_deformer.mlp.children():
            if isinstance(l, nn.Linear):
                try:
                    l.weight.data *= 0.1
                    l.bias.data.zero_()
                except Exception as e:
                    print("Error in initializing weights", e)
                    pass

        initialize_weights(self.template_deformer, gain=0.01)
        initialize_weights(self.pose_weight, gain=0.01)
        self.pose_weight[-2].weight.data.zero_()
        self.pose_weight[-2].bias.data[0] = 3.


        # self.gradient_scaling = 1.0
        # self.fourier_feature_transform.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] / self.gradient_scaling if grad_i[0] is not None else None, ))
        # self.fourier_feature_transform2.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] / self.gradient_scaling if grad_i[0] is not None else None, ))
        # self.expression_deformer.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] * self.gradient_scaling if grad_i[0] is not None else None, ))
        # self.template_deformer.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] * self.gradient_scaling if grad_i[0] is not None else None, ))


    def solve(self, x):
        if self.lambda_ == 0:
            return x
        if len(x.shape) == 2:
            return from_differential(self.M, x, 'Cholesky')
        else:
            res = torch.zeros_like(x)
            for i in range(x.shape[0]):
                res[i] = from_differential(self.M, x[i], 'Cholesky')
            return res

    def invert(self, u):
        if len(u.shape) == 2:
            return to_differential(self.M, u)
        else:
            res = torch.zeros_like(u)
            for i in range(u.shape[0]):
                res[i] = to_differential(self.M, u[i])
            return res


    def encode_position(self, coords, sec=False):
        template = self.ict_facekit.canonical[0] # shape of V, 3

        if len(coords.shape) == 2:
            coords = coords[None]

        b, v, _ = coords.shape
        coords = coords.reshape(b*v, -1)
        aabb_coords = (coords - self.aabb[0][None, ...]) / (self.aabb[1][None, ...] - self.aabb[0][None, ...])

        aabb_coords = aabb_coords.reshape(b, v, -1)

        if len(coords.shape) == 2:
            aabb_coords = aabb_coords[0]
        return aabb_coords

        org_coords = coords * 0.25

        unsqueezed = False
        if len(coords.shape) == 2:
            coords = coords[None]
            unsqueezed = True
        
        b, v, _ = coords.shape
        coords = coords.reshape(b*v, -1)

        if self.aabb is not None:
            coords = (coords - self.aabb[0][None, ...]) / (self.aabb[1][None, ...] - self.aabb[0][None, ...])
            coords = torch.clamp(coords, min=0, max=1)
        else:
            aabb_min = torch.min(template, dim=0)[0][None] * 1.2
            aabb_max = torch.max(template, dim=0)[0][None] * 1.2

            coords = (coords - aabb_min) / (aabb_max - aabb_min)

        if not sec:
            encoded_coords = self.fourier_feature_transform(coords)
        else:
            encoded_coords = self.fourier_feature_transform(coords)

        encoded_coords = encoded_coords.reshape(b, v, -1)
        if unsqueezed:
            encoded_coords = encoded_coords[0]
            
        if self.include_identity_on_encoding:
            encoded_coords = torch.cat([encoded_coords, org_coords], dim=-1)

        return encoded_coords


    def remove_teeth(self, face):
        face[..., 17039:21451, :] = 0
        return face


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

        encoded_points = torch.cat([self.encode_position(template)], dim=-1)
        encoded_points2 = torch.cat([self.encode_position(template, sec=True)], dim=-1)

        template_mesh_u_delta = self.template_deformer(encoded_points)
        
        template_mesh_delta = self.solve(template_mesh_u_delta) 

        template_mesh = self.ict_facekit.neutral_mesh_canonical[0] + template_mesh_delta

        template_mesh_posed = self.apply_deformation(template_mesh[None], features, pose_weight)

        ict_mesh = self.ict_facekit(expression_weights = features[..., :53].clamp(0,1), identity_weights = self.encoder.identity_weights[None].repeat(bsize, 1))

        ict_mesh_w_temp = ict_mesh + template_mesh_delta[None]

        ict_mesh_w_temp_posed = self.apply_deformation(ict_mesh_w_temp, features, pose_weight)

        return_dict['ict_mesh_posed'] = self.remove_teeth(self.apply_deformation(ict_mesh, features, pose_weight))
        return_dict['template_mesh'] = self.remove_teeth(template_mesh)
        return_dict['template_mesh_posed'] = self.remove_teeth(template_mesh_posed)
        return_dict['ict_mesh_w_temp'] = self.remove_teeth(ict_mesh_w_temp)
        return_dict['ict_mesh_w_temp_posed'] = self.remove_teeth(ict_mesh_w_temp_posed)

        if pretrain:
            return return_dict

        expression_input = torch.cat([encoded_points2[None].repeat(bsize, 1, 1), \
                                        ], \
                                    dim=2) # B V ? 
        
        expression_mesh_delta_u = self.expression_deformer(expression_input).reshape(bsize, template.shape[0], 53, 3)
        expression_mesh_delta_u = expression_mesh_delta_u * (features[:, None, :53, None].clamp(0, 1).clone().detach())
        expression_mesh_delta_u = expression_mesh_delta_u.sum(dim=2)

        expression_mesh_delta = self.solve(expression_mesh_delta_u)
        expression_mesh = ict_mesh_w_temp + expression_mesh_delta

        expression_mesh_posed = self.apply_deformation(expression_mesh, features, pose_weight)
        return_dict['expression_mesh'] = self.remove_teeth(expression_mesh)
        return_dict['expression_mesh_posed'] = self.remove_teeth(expression_mesh_posed)

        return_dict['pose_weight'] = pose_weight

        return return_dict


    def get_expression_delta(self, blendshapes):
        bsize = blendshapes.shape[0]

        template = self.ict_facekit.canonical[0]
        uv_coords = self.ict_facekit.uv_neutral_mesh[0]
        encoded_points = torch.cat([self.encode_position(template, sec=True)], dim=-1)

        expression_input = torch.cat([encoded_points[None].repeat(bsize, 1, 1), \
                                        ], \
                                    dim=2) # B V ? 
        expression_mesh_delta_u = self.expression_deformer(expression_input).reshape(bsize, template.shape[0], 53, 3)
        return expression_mesh_delta_u


    def apply_deformation(self, vertices, features, weights=None):
        euler_angle = features[..., 53:56]
        translation = features[..., 56:59]
        scale = features[..., 59:60]
        transform_origin = self.encoder.transform_origin
        
        if weights is None:
            weights = torch.ones_like(vertices[..., :1])

        # transform_origin = torch.cat([torch.zeros(1, device=vertices.device), self.encoder.transform_origin[1:]], dim=0)

        B, V, _ = vertices.shape
        rotation_matrix = pt3d.euler_angles_to_matrix(euler_angle[:, None].repeat(1, V, 1) * weights, convention = 'XYZ')
        local_coordinate_vertices = (vertices - transform_origin[None, None,]) * scale[:, None]
        deformed_mesh = torch.einsum('bvd, bvdj -> bvj', local_coordinate_vertices, rotation_matrix) + translation[:, None, :] + transform_origin[None, None] 
        return deformed_mesh

    def save(self, path):
        data = {
            'state_dict': self.state_dict()
        }
        torch.save(data, path)  

    def to(self, device):
        super().to(device)
        return self

def get_neural_blendshapes(model_path=None, train=True, vertex_parts=None, ict_facekit=None, exp_dir=None, lambda_=16, aabb=None, device='cuda'):
    neural_blendshapes = NeuralBlendshapes(vertex_parts, ict_facekit, exp_dir, lambda_ = lambda_, aabb=aabb)
    neural_blendshapes.to(device)

    import os
    if (os.path.exists(str(model_path))):
        print("Loading model from: ", str(model_path))
        params = torch.load(str(model_path))
        neural_blendshapes.load_state_dict(params["state_dict"], strict=False)
    elif model_path is not None:
        print('Model path is provided but the model is not found. Initializing with random weights.')
        raise Exception("Model not found")

    if train:
        neural_blendshapes.train()
    else:
        neural_blendshapes.eval()

    return neural_blendshapes        