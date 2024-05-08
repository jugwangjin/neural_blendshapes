import numpy as np
import torch
from torch import nn
from .encoder import ResnetEncoder
from flare.modules.embedder import *

# import tinycudann as tcnn
import pytorch3d.transforms as pt3d

def initialize_weights(m, gain=0.1):
    
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.xavier_uniform_(param.data, gain=gain)
        if 'bias' in name:
            param.data.zero_()

class NeuralBlendshapes(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = ResnetEncoder(62)

        self.coords_encoder, dim = get_embedder(6, input_dims=6)


        self.expression_deformer = nn.Sequential(
                                    nn.Linear(dim+53, 256),
                                    nn.SiLU(),
                                    nn.Linear(256, 256),
                                    nn.SiLU(),
                                    nn.Linear(256, 128),
                                    nn.SiLU(),
                                    nn.Linear(128, 128),
                                    nn.SiLU(),
                                    nn.Linear(128, 128),
                                    nn.SiLU(),
                                    nn.Linear(128, 3)
                                    )
        
        self.only_coords_encoder, dim = get_embedder(3, input_dims=3)
        
        self.template_deformer = nn.Sequential(
                    nn.Linear(dim, 64),
                    nn.SiLU(),
                    nn.Linear(64,32),
                    nn.SiLU(),
                    nn.Linear(32,32),
                    nn.SiLU(),
                    nn.Linear(32,3)
        )
        
        self.pose_weight = nn.Sequential(
                    nn.Linear(dim+6, 64),
                    nn.SiLU(),
                    nn.Linear(64,32),
                    nn.SiLU(),
                    nn.Linear(32,32),
                    nn.SiLU(),
                    nn.Linear(32,1),
                    nn.Sigmoid(),
        )

        self.transform_origin = torch.nn.Parameter(torch.tensor([0., 0., 0.]))
        self.scale = torch.nn.Parameter(torch.tensor([1.]))

        initialize_weights(self.expression_deformer, gain=0.1)
        initialize_weights(self.template_deformer, gain=0.1)
        initialize_weights(self.pose_weight, gain=0.1)
    

    def set_template(self, template, uv_template, vertex_parts=None, full_shape=None, head_indices=None, eyeball_indices=None):
        self.register_buffer('template', torch.cat([template, uv_template[0]], dim=1))     
        self.register_buffer('encoded_vertices', self.coords_encoder(self.template))
        self.register_buffer('encoded_only_vertices', self.only_coords_encoder(template))



    def forward(self, image=None, lmks=None, image_input=True, features=None, vertices=None, coords_min=None, coords_max=None):
        if image_input:
            # print(image.shape)
            if image.shape[1] != 3 and image.shape[3] == 3:
                image = image.permute(0, 3, 1, 2)
            # print(image.shape)
            features, estim_lmks = self.image_encoder(image, lmks)
        else:
            estim_lmks = None
        # print(features.shape, estim_lmks.shape)
        if vertices is None:
            vertices = self.template
            encoded_vertices = self.encoded_vertices
            encoded_only_vertices = self.encoded_only_vertices
        else:
            encoded_vertices = self.coords_encoder(vertices)
            encoded_only_vertices = self.only_coords_encoder(vertices[..., :3])


        template_deformation = self.template_deformer(encoded_only_vertices)

        deformer_input = torch.cat([encoded_vertices[None].repeat(features.shape[0], 1, 1), \
                                    features[:, None, :53].repeat(1, encoded_vertices.shape[0], 1)], dim=2)

        B, V, _ = deformer_input.shape

        expression_deformation = self.expression_deformer(deformer_input)

        expression_vertices = vertices[None][..., :3] + expression_deformation + template_deformation[None]

        pose_weight = self.pose_weight(torch.cat([encoded_only_vertices[None].repeat(features.shape[0], 1, 1), \
                                                features[:, None, 53:59].repeat(1, V, 1)], dim=2)) # shape of B V 1
        
        deformed_mesh = self.apply_deformation(expression_vertices, features, pose_weight)

        # rotation_matrix = pt3d.euler_angles_to_matrix(euler_angle[:, None].repeat(1, V, 1) * pose_weight, convention = 'XYZ')

        # local_coordinate_vertices = expression_vertices - self.transform_origin[None, None]

        # local_coordinate_vertices = local_coordinate_vertices

        # deformed_mesh = torch.einsum('bvd, bvdj -> bvj', local_coordinate_vertices, rotation_matrix) + translation[:, None, :] + self.transform_origin[None, None]

        return_dict = {} 
        return_dict['features'] = features

        return_dict['estim_lmks'] = estim_lmks

        return_dict['full_template_deformation'] = template_deformation
        return_dict['full_expression_deformation'] = expression_deformation
        return_dict['full_expression_mesh'] = expression_vertices
        return_dict['pose_weight'] = pose_weight
        return_dict['full_deformed_mesh'] = deformed_mesh

        return return_dict
        

    def apply_deformation(self, vertices, features, weights=None):
        

        euler_angle = features[..., 53:56]
        translation = features[..., 56:59]
        scale = self.scale

        if weights is None:
            weights = torch.ones_like(vertices[..., :1])

        B, V, _ = vertices.shape
        # print(features.shape, euler_angle.shape)
        # rotation_matrix = pt3d.euler_angles_to_matrix(euler_angle, convention = 'XYZ')
        # print(rotation_matrix[0])
        # print(rotation_matrix[1])

        rotation_matrix = pt3d.euler_angles_to_matrix(euler_angle[:, None].repeat(1, V, 1) * weights, convention = 'XYZ')

        

        local_coordinate_vertices = (vertices  - self.transform_origin[None, None] * 0.1) * scale
        deformed_mesh = torch.einsum('bvd, bvdj -> bvj', local_coordinate_vertices, rotation_matrix) + translation[:, None, :] + self.transform_origin[None, None] * 0.1

        return deformed_mesh

    def save(self, path):
        data = {
            'state_dict': self.state_dict()
        }
        torch.save(data, path)  


def get_neural_blendshapes(model_path=None, train=True, device='cuda'):
    neural_blendshapes = NeuralBlendshapes()
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