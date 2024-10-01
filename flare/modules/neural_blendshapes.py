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
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
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
            # print(alpha)
            # exit()

            self.register_buffer('M', compute_matrix(torch.from_numpy(vertices).to('cuda'), torch.from_numpy(faces).to('cuda'), lambda_, alpha=alpha, density=False))


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

        # self.gradient_scaling = 64.0
        self.fourier_feature_transform = tcnn.Encoding(3, enc_cfg).to('cuda')
        # self.fourier_feature_transform.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] / gradient_scaling, ))
        # self.fourier_feature_transform.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] / self.gradient_scaling if grad_i[0] is not None else None, ))
        self.inp_size = self.fourier_feature_transform.n_output_dims
        
        # nn.init.uniform_(self.fourier_feature_transform.params, -1e-1, 1e-1)

        self.include_identity_on_encoding = True

        if self.include_identity_on_encoding:
            self.inp_size += 3

        self.gradient_scaling = 64.
        self.fourier_feature_transform.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] / self.gradient_scaling if grad_i[0] is not None else None, ))
        # self.inp_size = self.fourier_feature_transform.n_output_dims

        # print(self.inp_size)

        self.expression_deformer = nn.Sequential(
            nn.Linear(self.inp_size + 53, 512),
            # nn.Linear(self.inp_size + 3 + 3 + 53, 512),
            # mygroupnorm(num_groups=4, num_channels=512),
            nn.ReLU(),
            nn.Linear(512, 512),
            # mygroupnorm(num_groups=4, num_channels=512),
            nn.ReLU(),
            nn.Linear(512, 512),
            # mygroupnorm(num_groups=4, num_channels=512),
            nn.ReLU(),
            nn.Linear(512, 512),
            # mygroupnorm(num_groups=4, num_channels=512),
            nn.ReLU(),
            nn.Linear(512, 3)
            # nn.Linear(512, 53*3)
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

        # last layer to all zeros, to make zero deformation as the default            
        # initialize_weights(self.expression_deformer, gain=0.01)
        self.expression_deformer[-1].weight.data.zero_()
        self.expression_deformer[-1].bias.data.zero_()

        initialize_weights(self.template_deformer, gain=0.01)
        self.template_deformer.mlp[-1].weight.data.zero_()
        self.template_deformer.mlp[-1].bias.data.zero_()

        # by default, weight to almost ones
        initialize_weights(self.pose_weight, gain=0.01)
        self.pose_weight[-2].weight.data.zero_()
        self.pose_weight[-2].bias.data[0] = 3.


    def register_gradient_hooks(self):
        for layer in self.expression_deformer:
            if isinstance(layer, nn.Linear):
                layer.weight.register_hook(lambda grad: grad * 0.5)
                if layer.bias is not None:
                    layer.bias.register_hook(lambda grad: grad * 0.5)

    def downscale_gradients(self, grad):
        return grad * 0.25


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


    def deform_by_jacobian(self, mesh_u):
        template = self.ict_facekit.neutral_mesh_canonical[0]

        mesh = self.solve(mesh_u)
        
        mesh = mesh - torch.mean(mesh, dim=1, keepdim=True) + self.encoder.global_translation[None, None]

        deformation = mesh - template[None, :self.socket_index]
        deformation = torch.cat([deformation, deformation[:, self.interior_displacement_index]], dim=1)

        mesh = template[None] + deformation

        return mesh

    def encode_position(self, coords):
        template = self.ict_facekit.canonical[0] # shape of V, 3
        org_coords = coords * 0.25


        unsqueezed = False
        if len(coords.shape) == 2:
            coords = coords[None]
            unsqueezed = True
        
        b, v, _ = coords.shape
        coords = coords.reshape(b*v, -1)

        if self.aabb is not None:
            # print(torch.amin(coords, dim=0), torch.amax(coords, dim=0))
            # print(self.aabb)
            # print(coords.amin(dim=0), coords.amax(dim=0))
            coords = (coords - self.aabb[0][None, ...]) / (self.aabb[1][None, ...] - self.aabb[0][None, ...])
            # print(coords.amin(dim=0), coords.amax(dim=0))
            coords = torch.clamp(coords, min=0, max=1)
            # print(self.aabb)
            # exit()
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


    def remove_teeth(self, verts):
        if len(verts.shape) == 2:
            verts[14062:21451] = verts[14062:21451] * 0
        else:
            verts[:, 14062:21451] = verts[:, 14062:21451] * 0
        return verts


    def forward(self, image=None, views=None, features=None, image_input=True, pretrain=False):
        bshape=None
        if features is not None:
            bshape = features
        return_dict = {} 
        features = self.encoder(views)

        if bshape is not None:
            features[:, :53] = bshape

        return_dict['features'] = features
    
        bsize = features.shape[0]
        template = self.ict_facekit.canonical[0]
        neutral_template = self.ict_facekit.neutral_mesh_canonical[0]

        uv_coords = self.ict_facekit.uv_neutral_mesh[0]


        pose_weight = self.pose_weight(self.ict_facekit.canonical[0])
# 
        # encoded_points = torch.cat([self.encode_position(uv_coords)], dim=-1)
        encoded_points = torch.cat([self.encode_position(template)], dim=-1)

        template_mesh_u_delta = self.template_deformer(encoded_points)
        # print(encoded_points.shape, template_mesh_u_delta.shape)
        # print(template.amin(), template.amax())
        # print(encoded_points.amin(), encoded_points.amax())
        template_mesh_delta = self.solve(template_mesh_u_delta) 

        template_mesh = self.ict_facekit.neutral_mesh_canonical[0] + template_mesh_delta

        ict_mesh = self.ict_facekit(expression_weights = features[..., :53], identity_weights = self.encoder.identity_weights[None].repeat(bsize, 1))

        ict_mesh_w_temp = ict_mesh + template_mesh_delta[None]

        ict_mesh_w_temp_posed = self.apply_deformation(ict_mesh_w_temp, features, pose_weight)

        return_dict['template_mesh'] = self.remove_teeth(template_mesh)
        return_dict['ict_mesh_w_temp'] = self.remove_teeth(ict_mesh_w_temp)
        return_dict['ict_mesh_w_temp_posed'] = self.remove_teeth(ict_mesh_w_temp_posed)

        if pretrain:
            return return_dict

        expression_input = torch.cat([encoded_points[None].repeat(bsize, 1, 1), \
                                        features[:, None, :53].repeat(1, template.shape[0], 1), \
                                        # self.encode_position(ict_mesh_w_temp[:, :self.socket_index]), \
                                        ], \
                                    dim=2) # B V ? 
        
        expression_mesh_delta_u = self.expression_deformer(expression_input).reshape(bsize, template.shape[0], 3)
        # expression_mesh_delta_u = self.expression_deformer(expression_input).reshape(bsize, template.shape[0], 53, 3)
        # expression_mesh_delta_u = expression_mesh_delta_u * features[:, None, :53, None]
        
        # expression_mesh_delta_u = expression_mesh_delta_u.sum(dim=2)
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
        encoded_points = torch.cat([self.encode_position(template)], dim=-1)
        # encoded_points = torch.cat([self.encode_position(uv_coords)], dim=-1)

        expression_input = torch.cat([encoded_points[None].repeat(bsize, 1, 1), \
                                        blendshapes[:, None].repeat(1, template.shape[0], 1), \
                                        ], \
                                    dim=2) # B V ? 
        expression_mesh_delta_u = self.expression_deformer(expression_input).reshape(bsize, template.shape[0], 3)
        # expression_mesh_delta_u = self.expression_deformer(expression_input).reshape(bsize, template.shape[0], 53, 3)
        # expression_mesh_delta_u = expression_mesh_delta_u * blendshapes[:, None, :53, None]
        return expression_mesh_delta_u


    def apply_deformation(self, vertices, features, weights=None):
        euler_angle = features[..., 53:56]
        translation = features[..., 56:59]
        scale = features[..., 59:60]
        transform_origin = self.encoder.transform_origin
        
        global_translation = features[..., 60:63] 

        if weights is None:
            weights = torch.ones_like(vertices[..., :1])

        transform_origin = torch.cat([torch.zeros(1, device=vertices.device), self.encoder.transform_origin[1:]], dim=0)

        B, V, _ = vertices.shape
        rotation_matrix = pt3d.euler_angles_to_matrix(euler_angle[:, None].repeat(1, V, 1) * weights, convention = 'XYZ')
        local_coordinate_vertices = (vertices - transform_origin[None, None,]) * scale[:, None]
        deformed_mesh = torch.einsum('bvd, bvdj -> bvj', local_coordinate_vertices, rotation_matrix) + weights * translation[:, None, :] + transform_origin[None, None] + global_translation[:, None]

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