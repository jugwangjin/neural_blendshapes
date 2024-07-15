import numpy as np
import torch
from torch import nn
from .encoder import ResnetEncoder
from flare.modules.embedder import *

# import tinycudann as tcnn
import pytorch3d.transforms as pt3d
import pytorch3d.ops as pt3o

import os


from .geometry import compute_matrix, laplacian_uniform
from .solvers import CholeskySolver, solve
from .parameterize import to_differential, from_differential

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
            nn.init.xavier_uniform_(l.weight, gain=gain)
            l.bias.data.zero_()

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
            

class NeuralBlendshapes(nn.Module):
    def __init__(self, vertex_parts, ict_facekit, exp_dir, lambda_=16):
        super().__init__()
        self.encoder = ResnetEncoder(53+6, ict_facekit)

        self.ict_facekit = ict_facekit
        self.tight_face_index = 6705
        self.face_index = 9409     
        self.head_index = 14062

        self.socket_index = 11248

        interior_vertices = ict_facekit.canonical[0, self.socket_index:]
        exterior_vertices = ict_facekit.canonical[0, :self.socket_index]

        # find closest face/head from eyeball, gums, teeth
        _, interior_displacement_index, _ = pt3o.knn_points(interior_vertices[None], exterior_vertices[None], K=1, return_nn=False)
        self.interior_displacement_index = interior_displacement_index[0, :, 0]
        
        vertices = ict_facekit.canonical[0].cpu().data.numpy()
        faces = ict_facekit.faces.cpu().data.numpy()

        faces = faces[faces[:, 0] < self.socket_index]
        faces = faces[faces[:, 1] < self.socket_index]
        faces = faces[faces[:, 2] < self.socket_index]

        self.register_buffer('L', laplacian_uniform(torch.from_numpy(vertices[:self.socket_index]).to('cuda'), torch.from_numpy(faces).to('cuda')))

        alpha = 1 - 1 / float(lambda_)

        self.register_buffer('M', compute_matrix(torch.from_numpy(vertices[:self.socket_index]).to('cuda'), torch.from_numpy(faces).to('cuda'), lambda_, alpha=None))

        code_dim = 53

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
        self.inp_size = self.fourier_feature_transform.n_output_dims

        self.gradient_scaling = 128.0
        self.fourier_feature_transform.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] / self.gradient_scaling if grad_i[0] is not None else None, ))
        self.inp_size = self.fourier_feature_transform.n_output_dims


        self.expression_deformer = nn.Sequential(nn.Linear(self.inp_size + 3 + 3 + 53, 256),
                                                  nn.Softplus(),
                                                  nn.Linear(256, 256), 
                                                  nn.Softplus(),
                                                  nn.Linear(256, 256),
                                                  nn.Softplus(),
                                                  nn.Linear(256, 256),
                                                  nn.Softplus(),
                                                  nn.Linear(256, 3))
        
        self.template_deformer = nn.Sequential(nn.Linear(self.inp_size + 3, 128),
                                                  nn.Softplus(),
                                                  nn.Linear(128, 128), 
                                                  nn.Softplus(),
                                                  nn.Linear(128, 128),
                                                  nn.Softplus(),
                                                  nn.Linear(128, 3))

        self.pose_weight = nn.Sequential(
                    nn.Linear(3, 32),
                    nn.Softplus(),
                    nn.Linear(32, 32),
                    nn.Softplus(),
                    nn.Linear(32,1),
                    nn.Sigmoid()
        )

        # last layer to all zeros, to make zero deformation as the default            
        initialize_weights(self.expression_deformer, gain=0.1)
        self.expression_deformer[-1].bias.data.zero_()

        initialize_weights(self.template_deformer, gain=0.1)
        self.template_deformer[-1].bias.data.zero_()

        # by default, weight to almost ones
        initialize_weights(self.pose_weight, gain=0.01)
        self.pose_weight[-2].weight.data.zero_()
        self.pose_weight[-2].bias.data[0] = 3.


    def solve(self, x):
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


    def deform_by_jacobian(self, mesh_u):
        template = self.ict_facekit.canonical[0]

        mesh = self.solve(mesh_u)
        
        mesh = mesh - torch.mean(mesh, dim=1, keepdim=True) + self.encoder.global_translation[None, None]

        deformation = mesh - template[None, :self.socket_index]
        deformation = torch.cat([deformation, deformation[:, self.interior_displacement_index]], dim=1)

        mesh = template[None] + deformation

        return mesh


    def encode_position(self, coords):
        template = self.ict_facekit.canonical[0] # shape of V, 3
        aabb_min = torch.min(template, dim=0)[0][None] * 1.2
        aabb_max = torch.max(template, dim=0)[0][None] * 1.2


        unsqueezed = False
        if len(coords.shape) == 2:
            coords = coords[None]
            unsqueezed = True
        
        b, v, _ = coords.shape
        coords = coords.reshape(b*v, -1)

        coords = (coords - aabb_min) / (aabb_max - aabb_min)

        encoded_coords = self.fourier_feature_transform(coords)
        encoded_coords = encoded_coords.reshape(b, v, -1)
        if unsqueezed:
            encoded_coords = encoded_coords[0]
        return encoded_coords


    def forward(self, image=None, views=None, features=None, image_input=True):
        return_dict = {} 
        features = self.encoder(views)
        return_dict['features'] = features
    
        bsize = features.shape[0]
        template = self.ict_facekit.canonical[0][:self.socket_index]

        pose_weight = self.pose_weight(self.ict_facekit.canonical[0])

        encoded_points = torch.cat([self.encode_position(template), template], dim=-1)

        template_mesh_u = self.invert(template)
        ict_mesh = self.ict_facekit(expression_weights = features[..., :53])[:, :self.socket_index]
        ict_mesh_u = self.invert(ict_mesh) - template_mesh_u[None]

        template_mesh_u_delta = self.template_deformer(encoded_points)

        ict_mesh_w_temp = self.deform_by_jacobian(template_mesh_u[None] + template_mesh_u_delta[None] + ict_mesh_u)
        ict_mesh_w_temp_posed = self.apply_deformation(ict_mesh_w_temp, features, pose_weight)

        template_mesh = self.deform_by_jacobian(template_mesh_u[None] + template_mesh_u_delta[None])[0]
        
        return_dict['template_mesh'] = template_mesh
        return_dict['ict_mesh_w_temp'] = ict_mesh_w_temp
        return_dict['ict_mesh_w_temp_posed'] = ict_mesh_w_temp_posed

        expression_input = torch.cat([encoded_points[None].repeat(bsize, 1, 1), \
                                    #   ict_mesh_encoded_points, \
                                        features[:, None, :53].repeat(1, encoded_points.shape[0], 1), \
                                        template_mesh_u[None] + \
                                        template_mesh_u_delta[None] + \
                                        ict_mesh_u
                                        ], \
                                    dim=2) # B V ? 
        expression_mesh_delta_u = self.expression_deformer(expression_input)
        expression_mesh = self.deform_by_jacobian(template_mesh_u[None] + \
                                                template_mesh_u_delta[None] + \
                                                ict_mesh_u + \
                                                expression_mesh_delta_u)
        # print(expression_mesh_delta_u.min(), expression_mesh_delta_u.max(), expression_mesh_delta_u.mean(), expression_mesh_delta_u.std())
        expression_mesh_posed = self.apply_deformation(expression_mesh, features, pose_weight)

        return_dict['expression_mesh'] = expression_mesh
        return_dict['expression_mesh_posed'] = expression_mesh_posed

        return_dict['pose_weight'] = pose_weight

        return return_dict

    def apply_deformation(self, vertices, features, weights=None):
        euler_angle = features[..., 53:56]
        translation = features[..., 56:59]
        scale = features[..., 59:]

        # print(euler_angle.shape, translation.shape, scale.shape)

        if weights is None:
            weights = torch.ones_like(vertices[..., :1])

        B, V, _ = vertices.shape
        rotation_matrix = pt3d.euler_angles_to_matrix(euler_angle[:, None].repeat(1, V, 1) * weights, convention = 'XYZ')
        local_coordinate_vertices = (vertices  - self.encoder.transform_origin[None, None]) * scale[:, None]
        deformed_mesh = torch.einsum('bvd, bvdj -> bvj', local_coordinate_vertices, rotation_matrix) + translation[:, None, :] + self.encoder.transform_origin[None, None] 
        # deformed_mesh = torch.einsum('bvd, bvdj -> bvj', local_coordinate_vertices, rotation_matrix) + translation[:, None, :] * weights + self.encoder.transform_origin[None, None] 

        return deformed_mesh

    def save(self, path):
        data = {
            'state_dict': self.state_dict()
        }
        torch.save(data, path)  

    def to(self, device):
        super().to(device)
        # self.source_mesh.to(device)
        # self.source_mesh_tight_face.to(device)
        return self

def get_neural_blendshapes(model_path=None, train=True, vertex_parts=None, ict_facekit=None, exp_dir=None, device='cuda'):
    neural_blendshapes = NeuralBlendshapes(vertex_parts, ict_facekit, exp_dir)
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