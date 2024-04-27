import numpy as np
import torch
from torch import nn
from .encoder import ResnetEncoder

import tinycudann as tcnn
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
        self.image_encoder = ResnetEncoder(60)

        L = 8; F = 4; log2_T = 19; N_min = 16
        b = np.exp(np.log(2048/N_min)/(L-1))

        self.coord_encoder = tcnn.Encoding(
                        n_input_dims=3, 
                        encoding_config={
                            "otype": "Grid",
                            "type": "Hash",
                            "n_levels": L,
                            "n_features_per_level": F,
                            "log2_hashmap_size": log2_T,
                            "base_resolution": N_min,
                            "per_level_scale": b,
                            "interpolation": "Linear"
                            },      
                        ).to('cuda')


        self.expression_deformer = nn.Sequential(
                                    nn.Linear(32+53, 128),
                                    nn.Softplus(),
                                    nn.Linear(128, 128),
                                    nn.Softplus(),
                                    nn.Linear(128, 128),
                                    nn.Softplus(),
                                    nn.Linear(128, 3)
                                    )
        
        self.template_deformer = nn.Sequential(
                    nn.Linear(32, 32),
                    nn.Softplus(),
                    nn.Linear(32,32),
                    nn.Softplus(),
                    nn.Linear(32,3)
        )
        
        self.pose_weight = nn.Sequential(
                    nn.Linear(3+60, 64),
                    nn.Softplus(),
                    nn.Linear(64,64),
                    nn.Softplus(),
                    nn.Linear(64,1),
                    nn.Tanh(),
        )

        self.transform_origin = torch.nn.Parameter(torch.tensor([0., 0., 0.]))

        initialize_weights(self.expression_deformer)
        initialize_weights(self.template_deformer, gain=0.01)
        initialize_weights(self.pose_weight, gain=0.01)

        
    def set_template(self, template, coords_min, coords_max):
        assert len(template.shape) == 2, "template should be a tensor shape of (num_vertices, 3) but got shape of {}".format(template.shape)
        self.register_buffer('template', template)

        self.register_buffer('coords_min', coords_min)
        self.register_buffer('coords_max', coords_max)

        self.register_buffer('range', self.coords_max - self.coords_min)


    def forward(self, image=None, image_input=True, features=None, vertices=None, coords_min=None, coords_max=None):
        if image_input:
            if image.shape[1] != 3 and image.shape[3] == 3:
                image = image.permute(0, 3, 1, 2)

            features = self.image_encoder(image)

        if vertices is None:
            vertices = self.template
        #     coords_min = self.coords_min
        #     coords_max = self.coords_max
        #     range = self.range

        # else:
        #     coords_min = coords_min if coords_min is not None else self.coords_min
        #     coords_max = coords_max if coords_max is not None else self.coords_max
        #     range = coords_max - coords_min

        encoded_vertices = self.coord_encoder((vertices + 1) / 2).type_as(vertices)

        template_deformation = self.template_deformer(encoded_vertices) 

        # concat feature and encoded vertices. vertices has shape of N_VERTICES, 16 and features has shape of N_BATCH, 64
        deformer_input = torch.cat([encoded_vertices[None].repeat(features.shape[0], 1, 1), \
                                    features[:, None, :53].repeat(1, encoded_vertices.shape[0], 1)], dim=2)

        B, V, _ = deformer_input.shape

        expression_deformation = self.expression_deformer(deformer_input)

        
        expression_vertices = vertices[None] + expression_deformation + template_deformation[None]

        pose_weight = self.pose_weight(torch.cat([expression_vertices, \
                                                features[:, None].repeat(1, V, 1)], dim=2)) # shape of B V 1
        
        euler_angle = features[:, 53:56]
        translation = features[:, 56:59]
        scale = features[:, -1:]

        local_coordinate_vertices = expression_vertices - self.transform_origin[None, None]

        scale = scale[:, None].repeat(1, V, 3)
        local_coordinate_vertices = local_coordinate_vertices * scale

        rotation_matrix = pt3d.euler_angles_to_matrix(euler_angle, convention = 'XYZ')

        deformed_mesh = torch.bmm(local_coordinate_vertices, rotation_matrix.transpose(1, 2)) + translation[:, None, :] + self.transform_origin[None, None]


        
        return_dict = {} 
        return_dict['features'] = features
        return_dict['template_deformation'] = template_deformation
        return_dict['expression_deformation'] = expression_deformation
        return_dict['expression_mesh'] = expression_vertices
        return_dict['pose_weight'] = pose_weight
        return_dict['deformed_mesh'] = deformed_mesh

        return return_dict
        
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