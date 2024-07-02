import numpy as np
import torch
from torch import nn
from .encoder import ResnetEncoder
from flare.modules.embedder import *

# import tinycudann as tcnn
import pytorch3d.transforms as pt3d
import pytorch3d.ops as pt3o

from flare.modules.NJF_sourcemesh import SourceMesh
import os

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
    def __init__(self, vertex_parts, ict_facekit, exp_dir, lambda_=4):
        super().__init__()
        self.encoder = ResnetEncoder(53+6, ict_facekit)

        self.ict_facekit = ict_facekit
        self.tight_face_index = 6705
        self.face_index = 9409     
        self.head_index = 14062

        self.socket_index = 11248

        interior_vertices = ict_facekit.canonical[0, self.head_index:]
        exterior_vertices = ict_facekit.canonical[0, :self.head_index]

        # find closest face/head from eyeball, gums, teeth
        _, interior_displacement_index, _ = pt3o.knn_points(interior_vertices[None], exterior_vertices[None], K=1, return_nn=False)
        self.interior_displacement_index = interior_displacement_index[0, :, 0]
        
        vertices = ict_facekit.canonical[0].cpu().data.numpy()
        faces = ict_facekit.faces.cpu().data.numpy()
        
        # remove source_mesh directory
        if os.path.exists(os.path.join(str(exp_dir), 'source_mesh')):
            os.system('rm -r ' + os.path.join(str(exp_dir), 'source_mesh'))
        os.makedirs(os.path.join(str(exp_dir), 'source_mesh'), exist_ok=True)
        self.source_mesh = SourceMesh(source_ind=None, source_dir = os.path.join(str(exp_dir), 'source_mesh'), \
                          extra_source_fields=[], random_scale=1, use_wks=False, random_centering=False, cpuonly=False)
        
        # filter faces
        faces = ict_facekit.faces.cpu().data.numpy()
        faces = faces[faces[:, 0] < self.head_index]
        faces = faces[faces[:, 1] < self.head_index]
        faces = faces[faces[:, 2] < self.head_index]

        self.source_mesh.load(source_v = vertices[:self.head_index], source_f = faces)

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

        # self.fourier_feature_transform, self.inp_size = get_embedder(10, input_dims=3)

        # self.expression_deformer = nn.Sequential(nn.Linear(self.inp_size + 3 + self.inp_size + 3 + code_dim + 9, 256),
        self.expression_deformer = nn.Sequential(nn.Linear(self.inp_size + 3 + code_dim + 9, 256),
                                                #   mygroupnorm(num_groups=32, num_channels=256),
                                                  nn.ReLU(),
                                                  nn.Linear(256, 256), 
                                                #   mygroupnorm(num_groups=32, num_channels=256),
                                                  nn.ReLU(),
                                                  nn.Linear(256, 256),
                                                #   mygroupnorm(num_groups=32, num_channels=256),
                                                  nn.ReLU(),
                                                  nn.Linear(256, 256),
                                                #   mygroupnorm(num_groups=32, num_channels=256),
                                                  nn.ReLU(),
                                                  nn.Linear(256, 9))
        
        self.template_deformer = nn.Sequential(nn.Linear(self.inp_size + 3 , 128),
                                                #   mygroupnorm(num_groups=16, num_channels=128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 128), 
                                                #   mygroupnorm(num_groups=16, num_channels=128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 128),
                                                #   mygroupnorm(num_groups=16, num_channels=128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 9))

        self.pose_weight = nn.Sequential(
                    nn.Linear(3, 32),
                    nn.ReLU(),
                    nn.Linear(32, 32),
                    nn.ReLU(),
                    nn.Linear(32,1),
                    nn.Sigmoid()
        )

        # last layer to all zeros, to make zero deformation as the default            
        initialize_weights(self.expression_deformer, gain=0.01)
        self.expression_deformer[-1].weight.data.zero_()
        self.expression_deformer[-1].bias.data.zero_()

        initialize_weights(self.template_deformer, gain=0.01)
        self.template_deformer[-1].weight.data.zero_()
        self.template_deformer[-1].bias.data.zero_()

        # by default, weight to almost ones
        initialize_weights(self.pose_weight, gain=0.01)
        self.pose_weight[-2].weight.data.zero_()
        self.pose_weight[-2].bias.data[0] = 3.


    def deform_by_jacobian(self, jacobian):
        assert len(jacobian.shape) == 4 # B V 3 3 
        template = self.ict_facekit.canonical[0]
        mesh = []
        for b in range(jacobian.shape[0]):
            mesh.append(self.source_mesh.vertices_from_jacobians(jacobian[b:b+1]))
        mesh = torch.cat(mesh, dim=0)
        mesh = mesh - torch.mean(mesh, dim=1, keepdim=True) + self.encoder.global_translation[None, None]

        deformation = mesh - template[None, :self.head_index]
        deformation = torch.cat([deformation, deformation[:, self.interior_displacement_index]], dim=1)

        mesh = template[None] + deformation

        return mesh


    def encode_position(self, coords):
        template = self.ict_facekit.canonical[0] # shape of V, 3
        aabb_min = torch.min(template, dim=0)[0][None]
        aabb_max = torch.max(template, dim=0)[0][None]


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
        template = self.ict_facekit.canonical[0][:self.head_index]

        pose_weight = self.pose_weight(self.ict_facekit.canonical[0])

        points = self.source_mesh.get_centroids_and_normals()
        encoded_points = self.encode_position(points[..., :3])
        encoded_points = torch.cat([encoded_points, points[..., 3:]], dim=-1)

        template_jacobian = self.source_mesh.jacobians_from_vertices(template[None])[0]

        # pose optimization
        ict_mesh = self.ict_facekit(expression_weights = features[..., :53])
        ict_jacobian = self.source_mesh.jacobians_from_vertices(ict_mesh[:, :self.head_index]) - template_jacobian[None]
        # ict_mesh = self.deform_by_jacobian(ict_jacobian + template_jacobian[None])
        # ict_mesh_posed = self.apply_deformation(ict_mesh, features, pose_weight)

        # return_dict['ict_mesh'] = ict_mesh
        # return_dict['ict_mesh_posed'] = ict_mesh_posed

        # grad cut for features, pose_weight
        # features = features.detach()
        # pose_weight = pose_weight.detach()
        # ict_mesh = ict_mesh.detach()
        # ict_jacobian = ict_jacobian.detach()

        # template optimization
        template_deformation_jacobian = self.template_deformer(encoded_points).reshape(-1, 3, 3)

        ict_mesh_w_temp = self.deform_by_jacobian(ict_jacobian + template_jacobian[None] + template_deformation_jacobian[None])
        ict_mesh_w_temp_posed = self.apply_deformation(ict_mesh_w_temp, features, pose_weight)

        template_mesh = self.deform_by_jacobian(template_jacobian[None] + template_deformation_jacobian[None])[0]
        
        return_dict['template_mesh'] = template_mesh
        return_dict['ict_mesh_w_temp'] = ict_mesh_w_temp
        return_dict['ict_mesh_w_temp_posed'] = ict_mesh_w_temp_posed

        # grad cut for template_jacobian, template_deformation_jacobian
        template_deformation_jacobian = template_deformation_jacobian.detach()

        # expression optimization
        # ict_mesh_points = self.source_mesh.get_centroids_from_vertices(ict_mesh_w_temp)
        # ict_mesh_encoded_points = torch.cat([self.encode_position(ict_mesh_points[..., :3]), \
                                            # ict_mesh_points[..., 3:]], dim=-1)  
        
        expression_input = torch.cat([encoded_points[None].repeat(bsize, 1, 1), \
                                    #   ict_mesh_encoded_points, \
                                        features[:, None, :53].repeat(1, ict_jacobian.shape[1], 1), \
                                        ict_jacobian.reshape(bsize, -1, 9) + \
                                        template_jacobian.reshape(-1, 9)[None].repeat(bsize, 1, 1) + \
                                        template_deformation_jacobian.reshape(-1, 9)[None].repeat(bsize, 1, 1), \
                                        ], \
                                    dim=2) # B V ? 
        expression_jacobian = self.expression_deformer(expression_input).reshape(bsize, -1, 3, 3)
        expression_mesh = self.deform_by_jacobian(template_jacobian[None] + \
                                                    ict_jacobian + \
                                                    expression_jacobian + \
                                                    template_deformation_jacobian[None])
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
        deformed_mesh = torch.einsum('bvd, bvdj -> bvj', local_coordinate_vertices, rotation_matrix) + translation[:, None, :] * weights + self.encoder.transform_origin[None, None] 

        return deformed_mesh

    def save(self, path):
        data = {
            'state_dict': self.state_dict()
        }
        torch.save(data, path)  

    def to(self, device):
        super().to(device)
        self.source_mesh.to(device)
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